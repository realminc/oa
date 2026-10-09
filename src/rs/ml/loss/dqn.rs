//! DQN (Deep Q-Network) target and combined loss.

use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, OperationContract, Result, matrix};

use super::smooth_l1::smooth_l1;

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
	const CONTRACT: OperationContract = crate::core::operation::ml::DQN;
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
	let target_q = dqn_target_impl(
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
	let selected_q =
		matrix::reshape_semantic_output(&matrix::gather_last_dim(q, &action_column)?, vec![*batch])?;
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

// ---------------------------------------------------------------------------
// Private implementation (formerly in lowering/loss.rs)
// ---------------------------------------------------------------------------

#[allow(
	clippy::too_many_arguments,
	reason = "the private DQN target lowering retains all four donor fields and its shape ABI"
)]
fn dqn_target_impl(
	lowering: &crate::runtime::SemanticLoweringScope,
	reward: &Matrix,
	next_q: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	batch: u32,
	actions: u32,
	discount: f32,
) -> Result<Matrix> {
	let output = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	let buffers = [
		BufferBinding::read(reward.storage()),
		BufferBinding::read(next_q.storage()),
		BufferBinding::read(terminated.storage()),
		BufferBinding::read(truncated.storage()),
		BufferBinding::write(output.storage()),
	];
	let push_constants = [
		PushConstant::U32(batch),
		PushConstant::U32(actions),
		PushConstant::F32(discount),
	];
	let kernel = KernelId::MlDqnTargetF32;
	lowering.record_physical(ComputeDispatch {
		kernel,
		buffers: &buffers,
		push_constants: &push_constants,
		workgroups: kernel.linear_workgroups(batch),
	})?;
	Ok(output)
}
