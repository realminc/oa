//! Binary cross-entropy loss.

use crate::ml::autograd;
use crate::runtime::KernelId;
use crate::{Matrix, Result};

use super::common::{mean_pointwise_loss, pointwise_loss_backward};

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
	let output = mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlBceF32,
		crate::core::operation::ml::BCE,
	)?;
	autograd::record_bce(prediction, target, &output)?;
	Ok(output)
}

pub(in crate::ml) fn bce_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlBceBackwardF32,
		crate::core::operation::ml::BCE_BACKWARD,
	)
}
