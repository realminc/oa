//! SAC (Soft Actor-Critic) critic and actor losses.

use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, OperationContract, Result, matrix};

use super::mse::mse;

/// Discount and entropy temperature used by SAC losses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SacLossConfig {
	/// Next-state value discount in the inclusive range `[0, 1]`.
	pub discount: f32,
	/// Non-negative entropy temperature.
	pub entropy_coefficient: f32,
}

impl Default for SacLossConfig {
	fn default() -> Self {
		Self {
			discount: 0.99,
			entropy_coefficient: 0.2,
		}
	}
}

/// Detached target and twin-critic SAC loss components.
pub struct SacCriticLossResult {
	/// Detached entropy-regularized Bellman target, shaped `[batch]`.
	pub target_q: Matrix,
	/// Mean-squared loss for the first critic.
	pub q1_loss: Matrix,
	/// Mean-squared loss for the second critic.
	pub q2_loss: Matrix,
	/// Sum of both critic losses.
	pub total_loss: Matrix,
}

/// Compose the donor twin-critic SAC objective.
///
/// The target is `reward + discount * (min(next_q1, next_q2) - alpha *
/// next_log_probability)` outside terminal states. Truncation retains that
/// bootstrap. All next-state target fields are detached from reverse mode.
///
/// # Errors
///
/// Returns an error unless all numerical fields are matching nonempty
/// same-engine F32 vectors, both boundary masks are matching U8 vectors,
/// discount is finite in `[0, 1]`, entropy coefficient is non-negative and
/// finite, the batch fits the shader ABI, and lowering succeeds.
#[allow(
	clippy::too_many_arguments,
	reason = "the SAC critic contract keeps every transition and target field explicit"
)]
pub fn sac_critic(
	q1: &Matrix,
	q2: &Matrix,
	reward: &Matrix,
	next_q1: &Matrix,
	next_q2: &Matrix,
	next_log_probability: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	config: SacLossConfig,
) -> Result<SacCriticLossResult> {
	const CONTRACT: OperationContract = crate::core::operation::ml::SAC_CRITIC;
	let [batch] = q1.shape() else {
		return Err(Error::invalid_argument(format!(
			"{} requires vector Q values",
			CONTRACT.name()
		)));
	};
	let numerical = [q1, q2, reward, next_q1, next_q2, next_log_probability];
	let valid = *batch > 0
		&& numerical.into_iter().all(|input| {
			input.shape() == q1.shape()
				&& input.dtype() == DType::F32
				&& q1.engine_handle().same_as(input.engine_handle())
		}) && [terminated, truncated].into_iter().all(|input| {
		input.shape() == q1.shape()
			&& input.dtype() == DType::U8
			&& q1.engine_handle().same_as(input.engine_handle())
	}) && config.discount.is_finite()
		&& (0.0..=1.0).contains(&config.discount)
		&& config.entropy_coefficient.is_finite()
		&& config.entropy_coefficient >= 0.0;
	if !valid {
		return Err(Error::invalid_argument(format!(
			"{} requires matching same-engine F32 vectors, U8 boundaries, discount in [0, 1], and a non-negative entropy coefficient",
			CONTRACT.name()
		)));
	}
	let batch_u32 = u32::try_from(*batch)
		.map_err(|_| Error::invalid_argument(format!("{} batch exceeds u32", CONTRACT.name())))?;
	let lowering = q1.engine_handle().begin_semantic_lowering()?;
	let target_q = sac_target_impl(
		&lowering,
		reward,
		next_q1,
		next_q2,
		next_log_probability,
		terminated,
		truncated,
		batch_u32,
		config.discount,
		config.entropy_coefficient,
	)?;
	let q1_loss = mse(q1, &target_q)?;
	let q2_loss = mse(q2, &target_q)?;
	let total_loss = matrix::add(&q1_loss, &q2_loss)?;
	let result = SacCriticLossResult {
		target_q,
		q1_loss,
		q2_loss,
		total_loss,
	};
	let attributes = [
		OpAttribute::Float {
			name: "discount".into(),
			value: f64::from(config.discount),
		},
		OpAttribute::Float {
			name: "entropy_coefficient".into(),
			value: f64::from(config.entropy_coefficient),
		},
	];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[
			q1,
			q2,
			reward,
			next_q1,
			next_q2,
			next_log_probability,
			terminated,
			truncated,
		],
		outputs: &[
			&result.target_q,
			&result.q1_loss,
			&result.q2_loss,
			&result.total_loss,
		],
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Compute the donor entropy-regularized SAC actor objective.
///
/// The scalar result is `mean(alpha * log_probability - min(q1, q2))` and is
/// differentiable through all three inputs.
///
/// # Errors
///
/// Returns an error unless inputs are matching nonempty same-engine F32 vectors,
/// entropy coefficient is finite and non-negative, and lowering succeeds.
pub fn sac_actor(
	q1: &Matrix,
	q2: &Matrix,
	log_probability: &Matrix,
	entropy_coefficient: f32,
) -> Result<Matrix> {
	const CONTRACT: OperationContract = crate::core::operation::ml::SAC_ACTOR;
	let [batch] = q1.shape() else {
		return Err(Error::invalid_argument(format!(
			"{} requires vector Q values",
			CONTRACT.name()
		)));
	};
	let valid = *batch > 0
		&& [q1, q2, log_probability].into_iter().all(|input| {
			input.shape() == q1.shape()
				&& input.dtype() == DType::F32
				&& q1.engine_handle().same_as(input.engine_handle())
		}) && entropy_coefficient.is_finite()
		&& entropy_coefficient >= 0.0;
	if !valid {
		return Err(Error::invalid_argument(format!(
			"{} requires matching nonempty same-engine F32 vectors and a non-negative entropy coefficient",
			CONTRACT.name()
		)));
	}
	let lowering = q1.engine_handle().begin_semantic_lowering()?;
	let difference = matrix::sub(q1, q2)?;
	let minimum = matrix::scale(
		&matrix::sub(&matrix::add(q1, q2)?, &matrix::abs(&difference)?)?,
		0.5,
	)?;
	let per_sample = matrix::sub(
		&matrix::scale(log_probability, entropy_coefficient)?,
		&minimum,
	)?;
	let loss = matrix::reshape_semantic_output(
		&matrix::scale(&matrix::sum(&per_sample, -1)?, 1.0 / *batch as f32)?,
		Vec::new(),
	)?;
	let attributes = [OpAttribute::Float {
		name: "entropy_coefficient".into(),
		value: f64::from(entropy_coefficient),
	}];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[q1, q2, log_probability],
		outputs: &[&loss],
		attributes: &attributes,
	})?;
	Ok(loss)
}

// ---------------------------------------------------------------------------
// Private implementation (formerly in lowering/loss.rs)
// ---------------------------------------------------------------------------

#[allow(
	clippy::too_many_arguments,
	reason = "the private SAC target lowering retains all six donor fields and its scalar ABI"
)]
fn sac_target_impl(
	lowering: &crate::runtime::SemanticLoweringScope,
	reward: &Matrix,
	next_q1: &Matrix,
	next_q2: &Matrix,
	next_log_probability: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	batch: u32,
	discount: f32,
	entropy_coefficient: f32,
) -> Result<Matrix> {
	let output = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(reward.storage()),
		BufferBinding::read(next_q1.storage()),
		BufferBinding::read(next_q2.storage()),
		BufferBinding::read(next_log_probability.storage()),
		BufferBinding::read(terminated.storage()),
		BufferBinding::read(truncated.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::F32(discount),
		PushConstant::F32(entropy_coefficient),
	];
	let kernel = KernelId::MlSacTargetF32;
	lowering.record_physical(ComputeDispatch {
		kernel,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: kernel.linear_workgroups(batch),
	})?;
	Ok(output)
}
