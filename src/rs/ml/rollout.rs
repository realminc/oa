//! Fixed-capacity device-resident on-policy rollout storage.

use crate::runtime::ComputeDispatch;
use crate::runtime::SemanticDispatch;
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result};

use super::{advantage, validation::shader_u32};

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
			.ok_or_else(|| Error::resource_exhausted("rollout observation storage overflows usize"))?;
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
		append_dispatch(
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
		reset_dispatch(&self.batch.valid)?;
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

// ---------------------------------------------------------------------------
// Private dispatch helpers
// ---------------------------------------------------------------------------

fn append_dispatch(
	transition: &RolloutTransition,
	batch: &RolloutBatch,
	step: usize,
	environments: usize,
	observation_shape: &[usize],
	observation_elements: u32,
) -> Result<()> {
	const OPERATION: &str = "oa::ml::rollout::append";
	let mut expected_observation_shape = Vec::with_capacity(observation_shape.len() + 1);
	expected_observation_shape.push(environments);
	expected_observation_shape.extend_from_slice(observation_shape);
	let vector_shape = [environments];
	let fields = [
		(
			transition.observation(),
			DType::F32,
			expected_observation_shape.as_slice(),
		),
		(transition.action(), DType::I32, vector_shape.as_slice()),
		(transition.reward(), DType::F32, vector_shape.as_slice()),
		(transition.value(), DType::F32, vector_shape.as_slice()),
		(transition.next_value(), DType::F32, vector_shape.as_slice()),
		(
			transition.log_probability(),
			DType::F32,
			vector_shape.as_slice(),
		),
		(transition.terminated(), DType::U8, vector_shape.as_slice()),
		(transition.truncated(), DType::U8, vector_shape.as_slice()),
	];
	for (matrix, dtype, shape) in &fields {
		if matrix.dtype() != *dtype || matrix.shape() != *shape {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition does not match the configured categorical shape/dtype contract"
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
	let destinations = batch_destinations(batch);
	for destination in destinations {
		if !transition
			.observation()
			.engine_handle()
			.same_as(destination.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition and rollout buffer must belong to one engine"
			)));
		}
		if fields
			.iter()
			.any(|(source, _, _)| source.storage().same_as(destination.storage()))
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must not alias retained rollout storage"
			)));
		}
	}

	let step = shader_u32(step, "step", OPERATION)?;
	let environments = shader_u32(environments, "environment count", OPERATION)?;
	let observation_count = environments
		.checked_mul(observation_elements)
		.ok_or_else(|| {
			Error::resource_exhausted(format!("{OPERATION} observation count exceeds u32"))
		})?;
	let work_items = observation_count.max(environments);
	let inputs = [
		transition.observation(),
		transition.action(),
		transition.reward(),
		transition.value(),
		transition.next_value(),
		transition.log_probability(),
		transition.terminated(),
		transition.truncated(),
		batch.observation(),
		batch.action(),
		batch.reward(),
		batch.value(),
		batch.next_value(),
		batch.old_log_probability(),
		batch.terminated(),
		batch.truncated(),
		batch.valid(),
	];
	let outputs = batch_destinations(batch);
	let buffers = [
		BufferBinding::read(transition.observation().storage()),
		BufferBinding::read(transition.action().storage()),
		BufferBinding::read(transition.reward().storage()),
		BufferBinding::read(transition.value().storage()),
		BufferBinding::read(transition.next_value().storage()),
		BufferBinding::read(transition.log_probability().storage()),
		BufferBinding::read(transition.terminated().storage()),
		BufferBinding::read(transition.truncated().storage()),
		BufferBinding::write(batch.observation().storage()),
		BufferBinding::write(batch.action().storage()),
		BufferBinding::write(batch.reward().storage()),
		BufferBinding::write(batch.value().storage()),
		BufferBinding::write(batch.next_value().storage()),
		BufferBinding::write(batch.old_log_probability().storage()),
		BufferBinding::write(batch.terminated().storage()),
		BufferBinding::write(batch.truncated().storage()),
		BufferBinding::write(batch.valid().storage()),
	];
	let push_constants = [
		PushConstant::U32(step),
		PushConstant::U32(environments),
		PushConstant::U32(observation_elements),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "step".into(),
			value: u64::from(step),
		},
		OpAttribute::UnsignedInteger {
			name: "environments".into(),
			value: u64::from(environments),
		},
		OpAttribute::UnsignedInteger {
			name: "observation_elements".into(),
			value: u64::from(observation_elements),
		},
	];
	let kernel = KernelId::MlRolloutAppendF32;
	{
		let inputs: &[&Matrix] = &inputs;
		let outputs: &[&Matrix] = &outputs;
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(work_items),
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
	}
}

fn reset_dispatch(valid: &Matrix) -> Result<()> {
	const OPERATION: &str = "oa::ml::rollout::reset";
	if valid.dtype() != DType::U8 || valid.shape().len() != 2 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a rank-two U8 validity mask"
		)));
	}
	let count = shader_u32(valid.num_elements(), "element count", OPERATION)?;
	let buffers = [BufferBinding::write(valid.storage())];
	let push_constants = [PushConstant::U32(count)];
	let kernel = KernelId::MlRolloutResetU8;
	{
		let inputs: &[&Matrix] = &[valid];
		let outputs: &[&Matrix] = &[valid];
		let attributes: &[OpAttribute] = &[];
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(count),
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
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn batch_destinations(batch: &RolloutBatch) -> [&Matrix; 9] {
	[
		batch.observation(),
		batch.action(),
		batch.reward(),
		batch.value(),
		batch.next_value(),
		batch.old_log_probability(),
		batch.terminated(),
		batch.truncated(),
		batch.valid(),
	]
}
