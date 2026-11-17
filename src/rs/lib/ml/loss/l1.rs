//! L1 (mean absolute error) loss.

use crate::ml::autograd;
use crate::runtime::KernelId;
use crate::{Matrix, Result};

use super::common::{mean_pointwise_loss, pointwise_loss_backward};

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
	let output = mean_pointwise_loss(
		prediction,
		target,
		KernelId::MlL1F32,
		crate::core::operation::ml::L1,
	)?;
	autograd::record_l1(prediction, target, &output)?;
	Ok(output)
}

pub(in crate::ml) fn l1_backward(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	pointwise_loss_backward(
		prediction,
		target,
		KernelId::MlL1BackwardF32,
		crate::core::operation::ml::L1_BACKWARD,
	)
}
