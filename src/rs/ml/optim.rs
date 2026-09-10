//! Stateless optimizer-adjacent transformations.

use crate::{Matrix, Result};

/// Clip the combined L2 norm of nonempty FP32 gradients in place.
///
/// The operation records one GPU reduction followed by one GPU scaling pass.
/// It does not read gradients back to the CPU or wait for completion. Empty
/// matrices are ignored; at most sixteen nonempty gradients are admitted.
///
/// # Errors
///
/// Returns an error when `max_norm` is negative or non-finite, more than
/// sixteen nonempty gradients are supplied, gradients do not share one engine,
/// a gradient is not FP32, storage is duplicated, a size exceeds the shader
/// ABI, or allocation and runtime recording fail.
pub fn clip_grad_norm(gradients: &[Matrix], max_norm: f32) -> Result<()> {
	let gradients = gradients.iter().collect::<Vec<_>>();
	super::lowering::optim::clip_grad_norm(&gradients, max_norm)
}
