use crate::{Matrix, Result};

use super::{autograd, lowering::loss as dispatch};

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
