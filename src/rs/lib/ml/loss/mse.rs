//! Mean squared error loss.

use crate::ml::autograd;
use crate::runtime::KernelId;
use crate::{Matrix, Result};

use super::common::{mean_pointwise_loss, pointwise_loss_backward};

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
	let output = mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlMseF32,
		crate::core::operation::ml::MSE,
	)?;
	autograd::record_mse(prediction, target, &output)?;
	Ok(output)
}

pub(in crate::ml) fn mse_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlMseBackwardF32,
		crate::core::operation::ml::MSE_BACKWARD,
	)
}
