//! Preallocated device-resident off-policy replay storage.

use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result};

use super::validation::shader_u32;

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
			.ok_or_else(|| Error::resource_exhausted("replay observation storage overflows usize"))?;
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
			terminated: Matrix::allocate(&handle, vec![config.capacity], config.capacity, DType::U8)?,
			truncated: Matrix::allocate(&handle, vec![config.capacity], config.capacity, DType::U8)?,
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
		let batch = append_dispatch(
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
		sample_dispatch(
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

// ---------------------------------------------------------------------------
// Private dispatch helpers
// ---------------------------------------------------------------------------

fn append_dispatch(
	transition: &ReplayTransition,
	storage: &ReplayBatch,
	config: &ReplayConfig,
	cursor: usize,
	observation_elements: u32,
	action_elements: u32,
) -> Result<usize> {
	const OPERATION: &str = "oa::ml::replay::append_batch";
	if transition.reward().dtype() != DType::F32 || transition.reward().shape().len() != 1 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires F32 reward [batch]"
		)));
	}
	let batch = transition.reward().shape()[0];
	if batch == 0 || batch > config.capacity {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} batch must be in 1..=capacity"
		)));
	}
	let observation_shape = prefixed_shape(batch, &config.observation_shape);
	let action_shape = prefixed_shape(batch, &config.action_shape);
	let vector_shape = [batch];
	let fields = [
		(
			transition.observation(),
			DType::F32,
			observation_shape.as_slice(),
		),
		(
			transition.action(),
			config.action_dtype,
			action_shape.as_slice(),
		),
		(
			transition.next_observation(),
			DType::F32,
			observation_shape.as_slice(),
		),
		(transition.reward(), DType::F32, vector_shape.as_slice()),
		(transition.terminated(), DType::U8, vector_shape.as_slice()),
		(transition.truncated(), DType::U8, vector_shape.as_slice()),
	];
	for (matrix, dtype, shape) in &fields {
		if matrix.dtype() != *dtype || matrix.shape() != *shape {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition does not match the configured replay schema"
			)));
		}
		if !transition
			.observation()
			.engine_handle()
			.same_as(matrix.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must belong to one engine"
			)));
		}
	}
	let destinations = storage_fields(storage);
	for destination in destinations {
		if !transition
			.observation()
			.engine_handle()
			.same_as(destination.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition and replay buffer must belong to one engine"
			)));
		}
		if fields
			.iter()
			.any(|(source, _, _)| source.storage().same_as(destination.storage()))
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must not alias replay storage"
			)));
		}
	}
	let cursor = shader_u32(cursor, "cursor", OPERATION)?;
	let capacity = shader_u32(config.capacity, "capacity", OPERATION)?;
	let batch_u32 = shader_u32(batch, "batch", OPERATION)?;
	let observation_count = batch_u32.checked_mul(observation_elements).ok_or_else(|| {
		Error::resource_exhausted(format!("{OPERATION} observation count exceeds u32"))
	})?;
	let action_count = batch_u32
		.checked_mul(action_elements)
		.ok_or_else(|| Error::resource_exhausted(format!("{OPERATION} action count exceeds u32")))?;
	let inputs = [
		transition.observation(),
		transition.action(),
		transition.next_observation(),
		transition.reward(),
		transition.terminated(),
		transition.truncated(),
		storage.observation(),
		storage.action(),
		storage.next_observation(),
		storage.reward(),
		storage.terminated(),
		storage.truncated(),
	];
	let outputs = storage_fields(storage);
	let buffers = [
		BufferBinding::read(transition.observation().storage()),
		BufferBinding::read(transition.action().storage()),
		BufferBinding::read(transition.next_observation().storage()),
		BufferBinding::read(transition.reward().storage()),
		BufferBinding::read(transition.terminated().storage()),
		BufferBinding::read(transition.truncated().storage()),
		BufferBinding::write(storage.observation().storage()),
		BufferBinding::write(storage.action().storage()),
		BufferBinding::write(storage.next_observation().storage()),
		BufferBinding::write(storage.reward().storage()),
		BufferBinding::write(storage.terminated().storage()),
		BufferBinding::write(storage.truncated().storage()),
	];
	let push_constants = [
		PushConstant::U32(cursor),
		PushConstant::U32(capacity),
		PushConstant::U32(batch_u32),
		PushConstant::U32(observation_elements),
		PushConstant::U32(action_elements),
	];
	let attributes = unsigned_attributes(&[
		("cursor", cursor),
		("capacity", capacity),
		("batch", batch_u32),
		("observation_elements", observation_elements),
		("action_elements", action_elements),
	]);
	let kernel = KernelId::MlReplayAppendBatchF32;
	{
		let inputs: &[&Matrix] = &inputs;
		let outputs: &[&Matrix] = &outputs;
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(observation_count.max(action_count).max(batch_u32)),
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}?;
	Ok(batch)
}

#[allow(
	clippy::too_many_arguments,
	reason = "replay sampling keeps capacity, sample geometry, seed, and field widths explicit"
)]
fn sample_dispatch(
	storage: &ReplayBatch,
	config: &ReplayConfig,
	size: usize,
	batch: usize,
	seed: u64,
	observation_elements: u32,
	action_elements: u32,
) -> Result<ReplayBatch> {
	const OPERATION: &str = "oa::ml::replay::sample";
	let size = shader_u32(size, "size", OPERATION)?;
	let batch_u32 = shader_u32(batch, "batch", OPERATION)?;
	let observation_count = batch
		.checked_mul(observation_elements as usize)
		.ok_or_else(|| Error::resource_exhausted("replay sample observation size overflows usize"))?;
	let action_count = batch
		.checked_mul(action_elements as usize)
		.ok_or_else(|| Error::resource_exhausted("replay sample action size overflows usize"))?;
	let observation_count_u32 = shader_u32(observation_count, "observation count", OPERATION)?;
	let action_count_u32 = shader_u32(action_count, "action count", OPERATION)?;
	let mut observation_shape = vec![batch];
	observation_shape.extend_from_slice(&config.observation_shape);
	let mut action_shape = vec![batch];
	action_shape.extend_from_slice(&config.action_shape);
	let engine = storage.observation().engine_handle();
	let result = ReplayBatch {
		observation: Matrix::allocate(
			engine,
			observation_shape.clone(),
			observation_count,
			DType::F32,
		)?,
		action: Matrix::allocate(engine, action_shape, action_count, config.action_dtype)?,
		next_observation: Matrix::allocate(engine, observation_shape, observation_count, DType::F32)?,
		reward: Matrix::allocate(engine, vec![batch], batch, DType::F32)?,
		terminated: Matrix::allocate(engine, vec![batch], batch, DType::U8)?,
		truncated: Matrix::allocate(engine, vec![batch], batch, DType::U8)?,
		index: Matrix::allocate(engine, vec![batch], batch, DType::U32)?,
	};
	let inputs = storage_fields(storage);
	let outputs = [
		result.observation(),
		result.action(),
		result.next_observation(),
		result.reward(),
		result.terminated(),
		result.truncated(),
		result.index(),
	];
	let buffers = [
		BufferBinding::read(storage.observation().storage()),
		BufferBinding::read(storage.action().storage()),
		BufferBinding::read(storage.next_observation().storage()),
		BufferBinding::read(storage.reward().storage()),
		BufferBinding::read(storage.terminated().storage()),
		BufferBinding::read(storage.truncated().storage()),
		BufferBinding::write(result.observation().storage()),
		BufferBinding::write(result.action().storage()),
		BufferBinding::write(result.next_observation().storage()),
		BufferBinding::write(result.reward().storage()),
		BufferBinding::write(result.terminated().storage()),
		BufferBinding::write(result.truncated().storage()),
		BufferBinding::write(result.index().storage()),
	];
	let push_constants = [
		PushConstant::U32(size),
		PushConstant::U32(batch_u32),
		PushConstant::U32(observation_elements),
		PushConstant::U32(action_elements),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "size".into(),
			value: u64::from(size),
		},
		OpAttribute::UnsignedInteger {
			name: "batch".into(),
			value: u64::from(batch_u32),
		},
		OpAttribute::UnsignedInteger {
			name: "observation_elements".into(),
			value: u64::from(observation_elements),
		},
		OpAttribute::UnsignedInteger {
			name: "action_elements".into(),
			value: u64::from(action_elements),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
	];
	let kernel = KernelId::MlReplaySampleF32;
	{
		let inputs: &[&Matrix] = &inputs;
		let outputs: &[&Matrix] = &outputs;
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel
				.linear_workgroups(observation_count_u32.max(action_count_u32).max(batch_u32)),
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}?;
	Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn storage_fields(batch: &ReplayBatch) -> [&Matrix; 6] {
	[
		batch.observation(),
		batch.action(),
		batch.next_observation(),
		batch.reward(),
		batch.terminated(),
		batch.truncated(),
	]
}

fn prefixed_shape(first: usize, suffix: &[usize]) -> Vec<usize> {
	let mut shape = Vec::with_capacity(suffix.len() + 1);
	shape.push(first);
	shape.extend_from_slice(suffix);
	shape
}

fn unsigned_attributes(values: &[(&'static str, u32)]) -> Vec<OpAttribute> {
	values
		.iter()
		.map(|(name, value)| OpAttribute::UnsignedInteger {
			name: (*name).into(),
			value: u64::from(*value),
		})
		.collect()
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
