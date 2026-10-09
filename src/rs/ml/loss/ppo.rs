//! PPO (clipped policy + full combined) losses.

use crate::ml::autograd;
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, OperationContract, Result, matrix};

use super::common::shader_u32;

/// Coefficients for the PPO objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpoLossConfig {
	/// Symmetric policy-ratio clipping distance.
	pub clip_epsilon: f32,
	/// Multiplier for critic mean-squared error.
	pub value_coefficient: f32,
	/// Multiplier subtracted for policy entropy.
	pub entropy_coefficient: f32,
}

impl Default for PpoLossConfig {
	fn default() -> Self {
		Self {
			clip_epsilon: 0.2,
			value_coefficient: 0.5,
			entropy_coefficient: 0.01,
		}
	}
}

/// PPO loss components retained for metrics and reverse mode.
pub struct PpoLossResult {
	/// Mean negative clipped surrogate objective.
	pub policy_loss: Matrix,
	/// Mean-squared critic error.
	pub value_loss: Matrix,
	/// Mean policy entropy.
	pub entropy: Matrix,
	/// `policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy`.
	pub total_loss: Matrix,
}

/// Compute the mean negative clipped PPO surrogate objective.
///
/// # Errors
///
/// Returns an error unless all inputs are matching nonempty same-engine FP32
/// matrices and `clip_epsilon` is finite in `(0, 1)`.
pub fn ppo_clipped_policy(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::PPO_CLIPPED_POLICY;
	let count = validate_ppo_inputs(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
		contract.name(),
	)?;
	let per_sample = Matrix::allocate(
		new_log_probability.engine_handle(),
		new_log_probability.shape().to_vec(),
		new_log_probability.num_elements(),
		DType::F32,
	)?;
	let sum = Matrix::allocate(
		new_log_probability.engine_handle(),
		Vec::new(),
		1,
		DType::F32,
	)?;
	let output = Matrix::allocate(
		new_log_probability.engine_handle(),
		Vec::new(),
		1,
		DType::F32,
	)?;
	let policy_buffers = [
		BufferBinding::read(new_log_probability.storage()),
		BufferBinding::read(old_log_probability.storage()),
		BufferBinding::read(advantage.storage()),
		BufferBinding::write(per_sample.storage()),
	];
	let policy_push = [PushConstant::U32(count), PushConstant::F32(clip_epsilon)];
	let sum_buffers = [
		BufferBinding::read(per_sample.storage()),
		BufferBinding::write(sum.storage()),
	];
	let sum_push = [PushConstant::U32(count)];
	let mean_buffers = [
		BufferBinding::read(sum.storage()),
		BufferBinding::write(output.storage()),
	];
	let mean_push = [PushConstant::U32(1), PushConstant::F32(1.0 / count as f32)];
	let kernel = KernelId::MlPpoClippedPolicyF32;
	let attributes = [OpAttribute::Float {
		name: "clip_epsilon".into(),
		value: f64::from(clip_epsilon),
	}];
	let dispatches = [
		ComputeDispatch {
			kernel,
			buffers: &policy_buffers,
			push_constants: &policy_push,
			workgroups: kernel.linear_workgroups(count),
		},
		ComputeDispatch {
			kernel: KernelId::MatrixSumF32,
			buffers: &sum_buffers,
			push_constants: &sum_push,
			workgroups: [1, 1, 1],
		},
		ComputeDispatch {
			kernel: KernelId::MatrixScaleF32,
			buffers: &mean_buffers,
			push_constants: &mean_push,
			workgroups: [1, 1, 1],
		},
	];
	new_log_probability.engine_handle().record_split_semantic(
		&dispatches,
		SemanticDispatch {
			contract,
			inputs: &[new_log_probability, old_log_probability, advantage],
			outputs: &[&output],
			attributes: &attributes,
		},
	)?;
	autograd::record_ppo_clipped_policy(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
		&output,
	)?;
	Ok(output)
}

/// Compute the explicit clipped-policy adjoint for new log probability.
///
/// # Errors
///
/// Returns the same validation and runtime errors as [`ppo_clipped_policy`].
pub fn ppo_clipped_policy_backward(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
) -> Result<Matrix> {
	let contract = crate::core::operation::ml::PPO_CLIPPED_POLICY_BACKWARD;
	let count = validate_ppo_inputs(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
		contract.name(),
	)?;
	let gradient = Matrix::allocate(
		new_log_probability.engine_handle(),
		new_log_probability.shape().to_vec(),
		new_log_probability.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(new_log_probability.storage()),
		BufferBinding::read(old_log_probability.storage()),
		BufferBinding::read(advantage.storage()),
		BufferBinding::write(gradient.storage()),
	];
	let push_constants = [PushConstant::U32(count), PushConstant::F32(clip_epsilon)];
	let attributes = [OpAttribute::Float {
		name: "clip_epsilon".into(),
		value: f64::from(clip_epsilon),
	}];
	let kernel = KernelId::MlPpoClippedPolicyBackwardF32;
	{
		let inputs: &[&Matrix] = &[new_log_probability, old_log_probability, advantage];
		let outputs: &[&Matrix] = &[&gradient];
		let attributes: &[OpAttribute] = &attributes;
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
	}?;
	Ok(gradient)
}

/// Compose PPO policy, value, entropy, and total losses.
///
/// All four outputs are FP32 scalars under one semantic PPO operation. Old log
/// probability, advantage, and target return remain detached by their child
/// operation contracts.
///
/// # Errors
///
/// Returns an error unless all rollout fields are matching nonempty
/// same-engine FP32 matrices and every coefficient is valid.
pub fn ppo(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	value: &Matrix,
	target_return: &Matrix,
	entropy: &Matrix,
	config: PpoLossConfig,
) -> Result<PpoLossResult> {
	const CONTRACT: OperationContract = crate::core::operation::ml::PPO;
	let inputs = [
		new_log_probability,
		old_log_probability,
		advantage,
		value,
		target_return,
		entropy,
	];
	for input in inputs {
		if input.shape() != new_log_probability.shape()
			|| input.dtype() != DType::F32
			|| !new_log_probability
				.engine_handle()
				.same_as(input.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{} requires matching nonempty same-engine F32 rollout fields",
				CONTRACT.name()
			)));
		}
	}
	if new_log_probability.num_elements() == 0
		|| !config.clip_epsilon.is_finite()
		|| !(0.0..1.0).contains(&config.clip_epsilon)
		|| !config.value_coefficient.is_finite()
		|| config.value_coefficient < 0.0
		|| !config.entropy_coefficient.is_finite()
		|| config.entropy_coefficient < 0.0
	{
		return Err(Error::invalid_argument(format!(
			"{} requires valid clipping and non-negative finite coefficients",
			CONTRACT.name()
		)));
	}
	let lowering = new_log_probability
		.engine_handle()
		.begin_semantic_lowering()?;
	let policy_loss = ppo_clipped_policy(
		new_log_probability,
		old_log_probability,
		advantage,
		config.clip_epsilon,
	)?;
	let value_loss = super::mse::mse(value, target_return)?;
	let entropy = matrix::reshape_semantic_output(
		&matrix::scale(
			&matrix::sum(entropy, -1)?,
			1.0 / new_log_probability.num_elements() as f32,
		)?,
		Vec::new(),
	)?;
	let total_loss = matrix::sub(
		&matrix::add(
			&policy_loss,
			&matrix::scale(&value_loss, config.value_coefficient)?,
		)?,
		&matrix::scale(&entropy, config.entropy_coefficient)?,
	)?;
	let result = PpoLossResult {
		policy_loss,
		value_loss,
		entropy,
		total_loss,
	};
	let attributes = [
		OpAttribute::Float {
			name: "clip_epsilon".into(),
			value: f64::from(config.clip_epsilon),
		},
		OpAttribute::Float {
			name: "value_coefficient".into(),
			value: f64::from(config.value_coefficient),
		},
		OpAttribute::Float {
			name: "entropy_coefficient".into(),
			value: f64::from(config.entropy_coefficient),
		},
	];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &inputs,
		outputs: &[
			&result.policy_loss,
			&result.value_loss,
			&result.entropy,
			&result.total_loss,
		],
		attributes: &attributes,
	})?;
	Ok(result)
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

fn validate_ppo_inputs(
	new_log_probability: &Matrix,
	old_log_probability: &Matrix,
	advantage: &Matrix,
	clip_epsilon: f32,
	operation: &'static str,
) -> Result<u32> {
	for input in [old_log_probability, advantage] {
		if input.shape() != new_log_probability.shape() || input.dtype() != DType::F32 {
			return Err(Error::invalid_argument(format!(
				"{operation} requires matching F32 inputs"
			)));
		}
		if !new_log_probability
			.engine_handle()
			.same_as(input.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{operation} inputs must belong to the same engine"
			)));
		}
	}
	if new_log_probability.dtype() != DType::F32 || new_log_probability.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonempty F32 inputs"
		)));
	}
	if !clip_epsilon.is_finite() || !(0.0..1.0).contains(&clip_epsilon) {
		return Err(Error::invalid_argument(format!(
			"{operation} requires clip epsilon in (0, 1)"
		)));
	}
	shader_u32(
		new_log_probability.num_elements(),
		"element count",
		operation,
	)
}
