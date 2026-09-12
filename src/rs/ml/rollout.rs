//! Fixed-capacity device-resident on-policy rollout storage.

use crate::{DType, Engine, Error, Matrix, Result};

use super::{advantage, lowering};

/// Shape and capacity of one categorical on-policy rollout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RolloutConfig {
	/// Number of time steps retained per collection cycle.
	pub time: usize,
	/// Number of vector environments represented by every step.
	pub environments: usize,
	/// Per-environment observation shape, excluding the environment axis.
	pub observation_shape: Vec<usize>,
}

/// One vector-environment transition whose values remain device-resident.
#[derive(Clone)]
pub struct RolloutTransition {
	observation: Matrix,
	action: Matrix,
	reward: Matrix,
	value: Matrix,
	next_value: Matrix,
	log_probability: Matrix,
	terminated: Matrix,
	truncated: Matrix,
}

impl RolloutTransition {
	/// Construct one categorical vector-environment transition.
	///
	/// Shape, dtype, and engine compatibility are checked by
	/// [`RolloutBuffer::append`] against that buffer's configuration.
	#[allow(
		clippy::too_many_arguments,
		reason = "the transition carrier keeps all eight donor rollout fields explicit"
	)]
	pub fn new(
		observation: Matrix,
		action: Matrix,
		reward: Matrix,
		value: Matrix,
		next_value: Matrix,
		log_probability: Matrix,
		terminated: Matrix,
		truncated: Matrix,
	) -> Self {
		Self {
			observation,
			action,
			reward,
			value,
			next_value,
			log_probability,
			terminated,
			truncated,
		}
	}

	/// Return the batched observation.
	pub const fn observation(&self) -> &Matrix {
		&self.observation
	}

	/// Return the categorical action indices.
	pub const fn action(&self) -> &Matrix {
		&self.action
	}

	/// Return the rewards.
	pub const fn reward(&self) -> &Matrix {
		&self.reward
	}

	/// Return the estimated values.
	pub const fn value(&self) -> &Matrix {
		&self.value
	}

	/// Return the estimated values after each transition.
	pub const fn next_value(&self) -> &Matrix {
		&self.next_value
	}

	/// Return the action log probabilities under the collecting policy.
	pub const fn log_probability(&self) -> &Matrix {
		&self.log_probability
	}

	/// Return the environment termination mask.
	pub const fn terminated(&self) -> &Matrix {
		&self.terminated
	}

	/// Return the environment truncation mask.
	pub const fn truncated(&self) -> &Matrix {
		&self.truncated
	}
}

/// Time-major matrices retained by one [`RolloutBuffer`].
pub struct RolloutBatch {
	observation: Matrix,
	action: Matrix,
	reward: Matrix,
	value: Matrix,
	next_value: Matrix,
	old_log_probability: Matrix,
	terminated: Matrix,
	truncated: Matrix,
	valid: Matrix,
	advantage: Matrix,
	returns: Matrix,
}

impl RolloutBatch {
	/// Return observations shaped `[time, environments, ...observation_shape]`.
	pub const fn observation(&self) -> &Matrix {
		&self.observation
	}

	/// Return categorical actions shaped `[time, environments]`.
	pub const fn action(&self) -> &Matrix {
		&self.action
	}

	/// Return rewards shaped `[time, environments]`.
	pub const fn reward(&self) -> &Matrix {
		&self.reward
	}

	/// Return values shaped `[time, environments]`.
	pub const fn value(&self) -> &Matrix {
		&self.value
	}

	/// Return next values shaped `[time, environments]`.
	pub const fn next_value(&self) -> &Matrix {
		&self.next_value
	}

	/// Return collection-policy log probabilities shaped `[time, environments]`.
	pub const fn old_log_probability(&self) -> &Matrix {
		&self.old_log_probability
	}

	/// Return termination flags shaped `[time, environments]`.
	pub const fn terminated(&self) -> &Matrix {
		&self.terminated
	}

	/// Return truncation flags shaped `[time, environments]`.
	pub const fn truncated(&self) -> &Matrix {
		&self.truncated
	}

	/// Return the slot-validity mask shaped `[time, environments]`.
	pub const fn valid(&self) -> &Matrix {
		&self.valid
	}

	/// Return generalized advantages shaped `[time, environments]`.
	pub const fn advantage(&self) -> &Matrix {
		&self.advantage
	}

	/// Return value targets shaped `[time, environments]`.
	pub const fn returns(&self) -> &Matrix {
		&self.returns
	}
}

/// Stateful fixed-capacity owner for one categorical on-policy rollout.
///
/// Appends, validity reset, and final GAE are recorded on the owning engine;
/// host observation remains the explicit submit-and-wait boundary. Storage is
/// allocated once and overwritten in place across completed collection cycles.
pub struct RolloutBuffer {
	config: RolloutConfig,
	batch: RolloutBatch,
	observation_elements: u32,
	size: usize,
	finalized: bool,
}

impl RolloutBuffer {
	/// Allocate one empty fixed-capacity rollout on `engine`.
	///
	/// # Errors
	///
	/// Returns an error unless time and environment counts are nonzero, the
	/// observation rank is in `1..=6`, every observation extent is positive,
	/// all GPU-indexed counts fit `u32`, and every matrix allocation succeeds.
	pub fn new(engine: &Engine, config: RolloutConfig) -> Result<Self> {
		if config.time == 0
			|| config.environments == 0
			|| !(1..=6).contains(&config.observation_shape.len())
			|| config.observation_shape.contains(&0)
		{
			return Err(Error::invalid_argument(
				"RolloutBuffer requires nonzero time/environments and a positive observation rank in 1..=6",
			));
		}
		let observation_elements = config
			.observation_shape
			.iter()
			.try_fold(1_usize, |count, extent| count.checked_mul(*extent))
			.ok_or_else(|| Error::resource_exhausted("rollout observation size overflows usize"))?;
		let rollout_elements = config
			.time
			.checked_mul(config.environments)
			.ok_or_else(|| Error::resource_exhausted("rollout size overflows usize"))?;
		let observation_count = rollout_elements
			.checked_mul(observation_elements)
			.ok_or_else(|| {
				Error::resource_exhausted("rollout observation storage overflows usize")
			})?;
		let observation_elements = u32::try_from(observation_elements).map_err(|_| {
			Error::resource_exhausted("rollout observation size exceeds GPU u32 indexing")
		})?;
		u32::try_from(rollout_elements)
			.map_err(|_| Error::resource_exhausted("rollout size exceeds GPU u32 indexing"))?;
		u32::try_from(observation_count).map_err(|_| {
			Error::resource_exhausted("rollout observation storage exceeds GPU u32 indexing")
		})?;

		let scalar_shape = vec![config.time, config.environments];
		let mut observation_shape = scalar_shape.clone();
		observation_shape.extend_from_slice(&config.observation_shape);
		let handle = engine.handle();
		let allocate =
			|shape: Vec<usize>, count: usize, dtype| Matrix::allocate(&handle, shape, count, dtype);
		let batch = RolloutBatch {
			observation: allocate(observation_shape, observation_count, DType::F32)?,
			action: allocate(scalar_shape.clone(), rollout_elements, DType::I32)?,
			reward: allocate(scalar_shape.clone(), rollout_elements, DType::F32)?,
			value: allocate(scalar_shape.clone(), rollout_elements, DType::F32)?,
			next_value: allocate(scalar_shape.clone(), rollout_elements, DType::F32)?,
			old_log_probability: allocate(scalar_shape.clone(), rollout_elements, DType::F32)?,
			terminated: allocate(scalar_shape.clone(), rollout_elements, DType::U8)?,
			truncated: allocate(scalar_shape.clone(), rollout_elements, DType::U8)?,
			valid: allocate(scalar_shape.clone(), rollout_elements, DType::U8)?,
			advantage: allocate(scalar_shape.clone(), rollout_elements, DType::F32)?,
			returns: allocate(scalar_shape, rollout_elements, DType::F32)?,
		};
		Ok(Self {
			config,
			batch,
			observation_elements,
			size: 0,
			finalized: false,
		})
	}

	/// Record one fused device-side append into the current time step.
	///
	/// # Errors
	///
	/// Returns an error after finalization, at capacity, when any transition
	/// field violates the configured shape/dtype/engine contract or aliases
	/// retained rollout storage, or when GPU recording fails.
	pub fn append(&mut self, transition: &RolloutTransition) -> Result<()> {
		if self.finalized {
			return Err(Error::failed_precondition(
				"RolloutBuffer append requires reset after finalize",
			));
		}
		if self.size == self.config.time {
			return Err(Error::resource_exhausted(
				"RolloutBuffer append exceeds rollout capacity",
			));
		}
		lowering::rollout::append(
			transition,
			&self.batch,
			self.size,
			self.config.environments,
			&self.config.observation_shape,
			self.observation_elements,
		)?;
		self.size += 1;
		Ok(())
	}

	/// Record generalized advantage estimation into the retained batch.
	///
	/// Calling this again after successful finalization is an idempotent no-op.
	///
	/// # Errors
	///
	/// Returns an error until the rollout is full or when GAE validation or GPU
	/// recording fails.
	pub fn finalize(&mut self, config: advantage::GaeConfig) -> Result<()> {
		if self.finalized {
			return Ok(());
		}
		if self.size != self.config.time {
			return Err(Error::failed_precondition(
				"RolloutBuffer finalize requires a complete rollout",
			));
		}
		advantage::gae_into(
			&self.batch.reward,
			&self.batch.value,
			&self.batch.next_value,
			&self.batch.terminated,
			&self.batch.truncated,
			&mut self.batch.advantage,
			&mut self.batch.returns,
			config,
		)?;
		self.finalized = true;
		Ok(())
	}

	/// Begin a new collection cycle and clear the validity mask on the device.
	///
	/// Previously allocated matrices are retained and overwritten by later
	/// appends. Pending work remains ordered by the engine's execution session.
	///
	/// # Errors
	///
	/// Returns an error when GPU recording fails.
	pub fn reset(&mut self) -> Result<()> {
		lowering::rollout::reset(&self.batch.valid)?;
		self.size = 0;
		self.finalized = false;
		Ok(())
	}

	/// Return whether all configured time steps have been appended.
	pub fn is_full(&self) -> bool {
		self.size == self.config.time
	}

	/// Return whether final GAE has been recorded for this cycle.
	pub const fn is_finalized(&self) -> bool {
		self.finalized
	}

	/// Return the number of appended time steps.
	pub const fn len(&self) -> usize {
		self.size
	}

	/// Return whether no time steps have been appended.
	pub const fn is_empty(&self) -> bool {
		self.size == 0
	}

	/// Return the fixed time-step capacity.
	pub const fn capacity(&self) -> usize {
		self.config.time
	}

	/// Return the validated rollout configuration.
	pub const fn config(&self) -> &RolloutConfig {
		&self.config
	}

	/// Return the retained time-major batch matrices.
	pub const fn batch(&self) -> &RolloutBatch {
		&self.batch
	}

	pub(crate) fn abort_unsubmitted(&mut self) {
		self.size = 0;
		self.finalized = false;
	}
}
