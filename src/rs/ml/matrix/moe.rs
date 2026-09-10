//! Sparse mixture-of-experts Matrix operations.

use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::moe as algebra};

/// Normalize probabilities over the selected experts for each token.
///
/// `probabilities` is F32 `[T, E]`; `expert_indices` is I32 `[T, K]` with
/// `1 <= K <= E`. The routing decision is detached from reverse mode, while
/// gradients propagate to the selected probability entries.
///
/// # Errors
///
/// Returns an error for incompatible shapes, dtypes, engine ownership, shader
/// ABI overflow, allocation failure, or runtime recording failure.
pub fn moe_route_weights(probabilities: &Matrix, expert_indices: &Matrix) -> Result<Matrix> {
	let output = algebra::route_weights(probabilities, expert_indices)?;
	autograd::record_moe_route_weights(probabilities, expert_indices, &output)?;
	Ok(output)
}

pub(in crate::ml) fn moe_route_weights_backward(
	output_gradient: &Matrix,
	probabilities: &Matrix,
	expert_indices: &Matrix,
	route_weights: &Matrix,
) -> Result<Matrix> {
	algebra::route_weights_backward(
		output_gradient,
		probabilities,
		expert_indices,
		route_weights,
	)
}
