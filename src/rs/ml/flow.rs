use crate::{Matrix, Result};

use super::{autograd, lowering::flow as dispatch};

/// Linear flow state and its constant path velocity.
pub struct FlowMatchBatch {
	/// Interpolated state `x(t)`.
	pub state: Matrix,
	/// Constant linear-path velocity `noise - clean`.
	pub velocity: Matrix,
}

/// Construct `x(t) = clean + t * (noise - clean)` and `v = noise - clean`.
///
/// A rank-one time vector matching the batch dimension expands across every
/// non-batch axis. Scalars and ordinarily broadcastable time shapes are also
/// accepted.
///
/// # Errors
///
/// Returns an error unless every input is same-engine F32, clean and noise are
/// matching and nonempty, time broadcasts to them, and recording succeeds.
pub fn linear_match(clean: &Matrix, noise: &Matrix, time: &Matrix) -> Result<FlowMatchBatch> {
	let output = dispatch::linear_match(clean, noise, time)?;
	autograd::record_flow_linear_match(clean, noise, time, &output.state, &output.velocity)?;
	Ok(FlowMatchBatch {
		state: output.state,
		velocity: output.velocity,
	})
}

/// Advance one explicit Euler step: `state + velocity * delta_time`.
///
/// # Errors
///
/// Returns an error unless state and velocity are matching nonempty
/// same-engine F32 matrices, `delta_time` is finite, and recording succeeds.
pub fn euler_step(state: &Matrix, velocity: &Matrix, delta_time: f32) -> Result<Matrix> {
	let output = dispatch::euler_step(state, velocity, delta_time)?;
	autograd::record_flow_euler_step(state, velocity, delta_time, &output)?;
	Ok(output)
}

/// Compute mean squared error over broadcast-selected valid elements.
///
/// The mask contributes its values to the denominator. An all-zero mask
/// returns exact zero by clamping the denominator to one. Target and mask are
/// detached; reverse mode differentiates only prediction.
///
/// # Errors
///
/// Returns an error unless prediction and target are matching nonempty
/// same-engine F32 matrices, mask is same-engine F32 and broadcastable to
/// prediction, and recording succeeds.
pub fn masked_mse(prediction: &Matrix, target: &Matrix, mask: &Matrix) -> Result<Matrix> {
	let output = dispatch::masked_mse(prediction, target, mask)?;
	autograd::record_flow_masked_mse(prediction, target, mask, &output.denominator, &output.loss)?;
	Ok(output.loss)
}
