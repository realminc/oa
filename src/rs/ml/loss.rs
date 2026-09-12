use crate::{DType, Error, Matrix, OpAttribute, Result, matrix};

use crate::runtime::SemanticDispatch;

use super::{autograd, lowering::loss as dispatch};

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

/// Discount applied to the non-terminal next-state value in DQN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DqnLossConfig {
	/// Next-state value discount in the inclusive range `[0, 1]`.
	pub discount: f32,
}

impl Default for DqnLossConfig {
	fn default() -> Self {
		Self { discount: 0.99 }
	}
}

/// Selected action values, detached targets, and scalar DQN loss.
pub struct DqnLossResult {
	/// Q values selected by the batch's action indices, shaped `[batch]`.
	pub selected_q: Matrix,
	/// Detached Bellman targets, shaped `[batch]`.
	pub target_q: Matrix,
	/// Mean unit-beta Smooth L1 loss between selected and target Q values.
	pub loss: Matrix,
}

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
	let output = dispatch::ppo_clipped_policy(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
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
	dispatch::ppo_clipped_policy_backward(
		new_log_probability,
		old_log_probability,
		advantage,
		clip_epsilon,
	)
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
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::PPO;
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
	let value_loss = mse(value, target_return)?;
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

/// Compose the donor DQN selected-action and detached Bellman-target loss.
///
/// Termination suppresses next-state bootstrap. Truncation deliberately keeps
/// bootstrap because it ends only the current trace, not the underlying task.
/// Reverse mode therefore connects `loss` and `selected_q` only to `q`.
///
/// # Errors
///
/// Returns an error unless Q and next-Q are same-engine nonempty F32 `[B,A]`
/// matrices with `A > 1`, action is I32 `[B]`, reward is F32 `[B]`, boundary
/// masks are U8 `[B]`, all fields share one engine, extents fit the shader ABI,
/// and discount is finite in `[0, 1]`.
#[allow(
	clippy::too_many_arguments,
	reason = "the DQN contract keeps every transition field explicit"
)]
pub fn dqn(
	q: &Matrix,
	action: &Matrix,
	reward: &Matrix,
	next_q: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	config: DqnLossConfig,
) -> Result<DqnLossResult> {
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::DQN;
	let [batch, actions] = q.shape() else {
		return Err(Error::invalid_argument(format!(
			"{} requires rank-two Q values",
			CONTRACT.name()
		)));
	};
	let vector_shape = [*batch];
	let valid = *batch > 0
		&& *actions > 1
		&& q.dtype() == DType::F32
		&& next_q.shape() == q.shape()
		&& next_q.dtype() == DType::F32
		&& action.shape() == vector_shape
		&& action.dtype() == DType::I32
		&& reward.shape() == vector_shape
		&& reward.dtype() == DType::F32
		&& terminated.shape() == vector_shape
		&& terminated.dtype() == DType::U8
		&& truncated.shape() == vector_shape
		&& truncated.dtype() == DType::U8
		&& [action, reward, next_q, terminated, truncated]
			.into_iter()
			.all(|input| q.engine_handle().same_as(input.engine_handle()))
		&& config.discount.is_finite()
		&& (0.0..=1.0).contains(&config.discount);
	if !valid {
		return Err(Error::invalid_argument(format!(
			"{} requires same-engine F32 Q/next-Q [B,A], I32 action [B], F32 reward [B], U8 boundaries [B], A > 1, and discount in [0, 1]",
			CONTRACT.name()
		)));
	}
	let batch_u32 = u32::try_from(*batch)
		.map_err(|_| Error::invalid_argument(format!("{} batch exceeds u32", CONTRACT.name())))?;
	let actions_u32 = u32::try_from(*actions).map_err(|_| {
		Error::invalid_argument(format!("{} action count exceeds u32", CONTRACT.name()))
	})?;
	u32::try_from(q.num_elements()).map_err(|_| {
		Error::invalid_argument(format!("{} Q element count exceeds u32", CONTRACT.name()))
	})?;

	let lowering = q.engine_handle().begin_semantic_lowering()?;
	let target_q = dispatch::dqn_target(
		&lowering,
		reward,
		next_q,
		terminated,
		truncated,
		batch_u32,
		actions_u32,
		config.discount,
	)?;
	let action_column = matrix::reshape(action, vec![*batch, 1])?;
	let selected_q = matrix::reshape_semantic_output(
		&matrix::gather_last_dim(q, &action_column)?,
		vec![*batch],
	)?;
	let loss = smooth_l1(&selected_q, &target_q)?;
	let result = DqnLossResult {
		selected_q,
		target_q,
		loss,
	};
	let attributes = [OpAttribute::Float {
		name: "discount".into(),
		value: f64::from(config.discount),
	}];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[q, action, reward, next_q, terminated, truncated],
		outputs: &[&result.selected_q, &result.target_q, &result.loss],
		attributes: &attributes,
	})?;
	Ok(result)
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
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::SAC_CRITIC;
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
	let target_q = dispatch::sac_target(
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
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::SAC_ACTOR;
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

/// Compute mean unit-beta Smooth L1 loss for matching FP32 matrices.
///
/// The target is treated as detached; reverse mode differentiates only the
/// prediction.
///
/// # Errors
///
/// Returns an error for empty, mismatched, non-FP32, or cross-engine inputs, or
/// when allocation or runtime recording fails.
pub fn smooth_l1(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let output = dispatch::smooth_l1(prediction, target)?;
	autograd::record_smooth_l1(prediction, target, &output)?;
	Ok(output)
}

/// Compute mean squared error for matching FP32 matrices.
///
/// The target is treated as detached; reverse mode differentiates only the
/// prediction.
///
/// # Errors
///
/// Returns an error for empty, mismatched, non-FP32, or cross-engine inputs, or
/// when allocation or runtime recording fails.
pub fn mse(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let output = dispatch::mse(prediction, target)?;
	autograd::record_mse(prediction, target, &output)?;
	Ok(output)
}

/// Compute mean absolute error for matching FP32 matrices.
///
/// The target is treated as detached; reverse mode differentiates only the
/// prediction. Equal prediction and target values use the zero subgradient.
///
/// # Errors
///
/// Returns an error for empty, mismatched, non-FP32, or cross-engine inputs, or
/// when allocation or runtime recording fails.
pub fn l1(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let output = dispatch::l1(prediction, target)?;
	autograd::record_l1(prediction, target, &output)?;
	Ok(output)
}

/// Compute mean binary cross entropy over matching FP32 probability matrices.
///
/// Predictions use the donor's `[1e-7, 1 - 1e-7]` numerical clamp. Targets are
/// treated as detached; reverse mode differentiates only the prediction.
///
/// # Errors
///
/// Returns an error for empty, mismatched, non-FP32, or cross-engine inputs, or
/// when allocation or runtime recording fails.
pub fn bce(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	let output = dispatch::bce(prediction, target)?;
	autograd::record_bce(prediction, target, &output)?;
	Ok(output)
}

/// Compute mean cross-entropy over rank-two FP32 logits and U32 or I32 class targets.
///
/// `logits` has shape `[N, C]`, `targets` has shape `[N]`, and the returned
/// matrix is an FP32 scalar. I32 targets must be non-negative; any negative or
/// out-of-range target produces NaN without an out-of-bounds access.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, ownership, allocation, or
/// runtime recording are invalid.
pub fn cross_entropy(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	let output = dispatch::cross_entropy(logits, targets)?;
	autograd::record_cross_entropy(logits, targets, &output)?;
	Ok(output)
}

/// Compute mean cross-entropy over the rows selected by a floating FP32 mask.
///
/// `logits` has shape `[N, C]`; `targets` and `mask` have shape `[N]`.
/// A zero mask value excludes its row and produces an exact-zero logits
/// adjoint. `valid_count` is the caller-provided normalization denominator and
/// must be in `1..=N`; it is not read back or recomputed from device storage.
/// Reverse mode differentiates only `logits`.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, ownership, `valid_count`,
/// allocation, or runtime recording are invalid.
pub fn masked_cross_entropy(
	logits: &Matrix,
	targets: &Matrix,
	mask: &Matrix,
	valid_count: usize,
) -> Result<Matrix> {
	let output = dispatch::masked_cross_entropy(logits, targets, mask, valid_count)?;
	autograd::record_masked_cross_entropy(logits, targets, mask, valid_count, &output)?;
	Ok(output)
}
