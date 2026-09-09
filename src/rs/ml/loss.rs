use crate::{Matrix, Result};

use super::{autograd, kernels};

/// Compute mean cross-entropy over rank-two FP32 logits and U32 class targets.
///
/// `logits` has shape `[N, C]`, `targets` has shape `[N]`, and the returned
/// matrix is an FP32 scalar. An out-of-range target produces NaN without an
/// out-of-bounds access.
///
/// # Errors
///
/// Returns an error when ranks, shapes, dtypes, ownership, allocation, or
/// runtime recording are invalid.
pub fn cross_entropy(logits: &Matrix, targets: &Matrix) -> Result<Matrix> {
	let output = kernels::cross_entropy(logits, targets)?;
	autograd::record_cross_entropy(logits, targets, &output)?;
	Ok(output)
}
