//! Preallocated device-resident off-policy replay storage.

use crate::{DType, Engine, Error, Matrix, Result};

use super::lowering;

/// Shape, capacity, and action representation of one replay buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayConfig {
	/// Maximum number of retained transitions.
	pub capacity: usize,
	/// Per-transition observation shape, excluding the batch axis.
	pub observation_shape: Vec<usize>,
	/// Per-transition action shape; empty denotes one scalar action.
	pub action_shape: Vec<usize>,
	/// Action storage representation; currently F32 or I32.
	pub action_dtype: DType,
}

/// One batch of transitions to append to a [`ReplayBuffer`].
#[derive(Clone)]
pub struct ReplayTransition {
	observation: Matrix,
	action: Matrix,
	next_observation: Matrix,
	reward: Matrix,
	terminated: Matrix,
	truncated: Matrix,
}

impl ReplayTransition {
	/// Construct an unchecked replay transition carrier.
	///
	/// [`ReplayBuffer::append`] validates every field against its configuration.
	pub fn new(
		observation: Matrix,
		action: Matrix,
		next_observation: Matrix,
		reward: Matrix,
		terminated: Matrix,
		truncated: Matrix,
	) -> Self {
		Self {
			observation,
			action,
			next_observation,
			reward,
			terminated,
			truncated,
		}
	}

	/// Return the observations.
	pub const fn observation(&self) -> &Matrix {
		&self.observation
	}

	/// Return the actions.
	pub const fn action(&self) -> &Matrix {
		&self.action
	}

	/// Return the successor observations.
	pub const fn next_observation(&self) -> &Matrix {
		&self.next_observation
	}

	/// Return the scalar rewards.
	pub const fn reward(&self) -> &Matrix {
		&self.reward
	}

	/// Return termination flags.
	pub const fn terminated(&self) -> &Matrix {
		&self.terminated
	}

	/// Return truncation flags.
	pub const fn truncated(&self) -> &Matrix {
		&self.truncated
	}
}

/// Device-resident replay values, optionally including sampled source indices.
pub struct ReplayBatch {
	pub(crate) observation: Matrix,
	pub(crate) action: Matrix,
	pub(crate) next_observation: Matrix,
	pub(crate) reward: Matrix,
	pub(crate) terminated: Matrix,
	pub(crate) truncated: Matrix,
	pub(crate) index: Matrix,
}

impl ReplayBatch {
	/// Return observations with a leading storage or sample axis.
	pub const fn observation(&self) -> &Matrix {
		&self.observation
	}

	/// Return actions with a leading storage or sample axis.
	pub const fn action(&self) -> &Matrix {
		&self.action
	}

	/// Return successor observations with a leading storage or sample axis.
	pub const fn next_observation(&self) -> &Matrix {
		&self.next_observation
	}

	/// Return scalar rewards.
	pub const fn reward(&self) -> &Matrix {
		&self.reward
	}

	/// Return termination flags.
	pub const fn terminated(&self) -> &Matrix {
		&self.terminated
	}

	/// Return truncation flags.
	pub const fn truncated(&self) -> &Matrix {
		&self.truncated
	}

	/// Return sampled source indices; storage batches leave this matrix unused.
	pub const fn index(&self) -> &Matrix {
		&self.index
	}
}

/// Circular fixed-capacity owner for off-policy transitions.
///
/// Batched append and deterministic uniform sampling are recorded on the one
/// owning engine. Cursor and size are host control metadata and never require a
/// Matrix readback.
pub struct ReplayBuffer {
	config: ReplayConfig,
	storage: ReplayBatch,
	observation_elements: u32,
	action_elements: u32,
	size: usize,
	cursor: usize,
}

impl ReplayBuffer {
	/// Allocate empty circular replay storage on `engine`.
	///
	/// # Errors
	///
	/// Returns an error for zero capacity, non-F32/I32 actions, invalid field
	/// ranks or extents, GPU-index overflow, or allocation failure.
	pub fn new(engine: &Engine, config: ReplayConfig) -> Result<Self> {
		if config.capacity == 0 || !matches!(config.action_dtype, DType::F32 | DType::I32) {
			return Err(Error::invalid_argument(
				"ReplayBuffer requires positive capacity and F32 or I32 actions",
			));
		}
		let observation_elements = field_elements(&config.observation_shape, false)?;
		let action_elements = field_elements(&config.action_shape, true)?;
		let observation_count = config
			.capacity
			.checked_mul(observation_elements)
			.ok_or_else(|| {
				Error::resource_exhausted("replay observation storage overflows usize")
			})?;
		let action_count = config
			.capacity
			.checked_mul(action_elements)
			.ok_or_else(|| Error::resource_exhausted("replay action storage overflows usize"))?;
		for (label, count) in [
			("capacity", config.capacity),
			("observation storage", observation_count),
			("action storage", action_count),
		] {
			u32::try_from(count).map_err(|_| {
				Error::resource_exhausted(format!("replay {label} exceeds GPU u32 indexing"))
			})?;
		}
		let observation_elements = u32::try_from(observation_elements)
			.map_err(|_| Error::resource_exhausted("replay observation field exceeds u32"))?;
		let action_elements = u32::try_from(action_elements)
			.map_err(|_| Error::resource_exhausted("replay action field exceeds u32"))?;
		let mut observation_shape = vec![config.capacity];
		observation_shape.extend_from_slice(&config.observation_shape);
		let mut action_shape = vec![config.capacity];
		action_shape.extend_from_slice(&config.action_shape);
		let handle = engine.handle();
		let storage = ReplayBatch {
			observation: Matrix::allocate(
				&handle,
				observation_shape.clone(),
				observation_count,
				DType::F32,
			)?,
			action: Matrix::allocate(&handle, action_shape, action_count, config.action_dtype)?,
			next_observation: Matrix::allocate(
				&handle,
				observation_shape,
				observation_count,
				DType::F32,
			)?,
			reward: Matrix::allocate(&handle, vec![config.capacity], config.capacity, DType::F32)?,
			terminated: Matrix::allocate(
				&handle,
				vec![config.capacity],
				config.capacity,
				DType::U8,
			)?,
			truncated: Matrix::allocate(
				&handle,
				vec![config.capacity],
				config.capacity,
				DType::U8,
			)?,
			index: Matrix::allocate(&handle, vec![config.capacity], config.capacity, DType::U32)?,
		};
		Ok(Self {
			config,
			storage,
			observation_elements,
			action_elements,
			size: 0,
			cursor: 0,
		})
	}

	/// Record one batched circular append.
	///
	/// # Errors
	///
	/// Returns an error unless the batch is nonempty, no larger than capacity,
	/// all fields match the configured shape/dtype/engine contract, source and
	/// retained storage do not alias, and GPU recording succeeds.
	pub fn append(&mut self, transition: &ReplayTransition) -> Result<()> {
		let batch = lowering::replay::append(
			transition,
			&self.storage,
			&self.config,
			self.cursor,
			self.observation_elements,
			self.action_elements,
		)?;
		self.cursor = (self.cursor + batch) % self.config.capacity;
		self.size = self.config.capacity.min(self.size.saturating_add(batch));
		Ok(())
	}

	/// Record deterministic uniform sampling with replacement.
	///
	/// The returned U32 index vector identifies every sampled physical replay
	/// slot exactly.
	///
	/// # Errors
	///
	/// Returns an error while storage is empty, for a zero sample count, on GPU
	/// indexing/allocation failure, or when recording fails.
	pub fn sample(&self, batch_size: usize, seed: u64) -> Result<ReplayBatch> {
		if self.size == 0 || batch_size == 0 {
			return Err(Error::failed_precondition(
				"ReplayBuffer sample requires nonempty storage and a positive batch size",
			));
		}
		lowering::replay::sample(
			&self.storage,
			&self.config,
			self.size,
			batch_size,
			seed,
			self.observation_elements,
			self.action_elements,
		)
	}

	/// Forget all retained transitions without recording device work.
	///
	/// Existing bytes remain inaccessible until overwritten by later appends.
	pub fn reset(&mut self) {
		self.size = 0;
		self.cursor = 0;
	}

	/// Return whether capacity is completely occupied.
	pub fn is_full(&self) -> bool {
		self.size == self.config.capacity
	}

	/// Return the number of accessible transitions.
	pub const fn len(&self) -> usize {
		self.size
	}

	/// Return whether no transition is accessible.
	pub const fn is_empty(&self) -> bool {
		self.size == 0
	}

	/// Return the fixed transition capacity.
	pub const fn capacity(&self) -> usize {
		self.config.capacity
	}

	/// Return the physical slot that receives the next append lane.
	pub const fn cursor(&self) -> usize {
		self.cursor
	}

	/// Return the validated replay configuration.
	pub const fn config(&self) -> &ReplayConfig {
		&self.config
	}
}

fn field_elements(shape: &[usize], allow_scalar: bool) -> Result<usize> {
	if (!allow_scalar && shape.is_empty()) || shape.len() > 7 || shape.contains(&0) {
		return Err(Error::invalid_argument(
			"replay fields require positive rank-1..=7 observations and rank-0..=7 actions",
		));
	}
	shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::resource_exhausted("replay field size overflows usize"))
	})
}
