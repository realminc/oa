//! Explicit host observations for machine-learning metrics.

use crate::{DType, Error, Matrix, Result, matrix};

/// Read one scalar FP32 loss value.
///
/// This is an explicit host-observation boundary: pending work is submitted and
/// the producing event is awaited before the value is returned.
///
/// # Errors
///
/// Returns an error unless `loss` contains exactly one F32 value, or submission,
/// completion, or readback fails.
pub fn scalar_loss(loss: &Matrix) -> Result<f32> {
	if loss.dtype() != DType::F32 || loss.num_elements() != 1 {
		return Err(Error::invalid_argument(format!(
			"ml scalar_loss requires exactly one F32 value; found shape {:?} {}",
			loss.shape(),
			loss.dtype().token()
		)));
	}
	Ok(loss.read_f32()?[0])
}

/// Compute last-axis categorical accuracy as a fraction in `[0, 1]`.
///
/// Argmax and counting execute on the owning device. Only the resulting U32
/// scalar crosses the host boundary; this function then waits for that scalar.
/// Equal maximum logits select the first class, matching the OA C++ contract.
///
/// # Errors
///
/// Returns an error when [`matrix::categorical_accuracy_count`] rejects the
/// logits or labels, or submission, completion, or scalar readback fails.
pub fn accuracy(logits: &Matrix, labels: &Matrix) -> Result<f32> {
	let count = matrix::categorical_accuracy_count(logits, labels)?;
	let correct = count.read::<u32>()?[0];
	Ok(correct as f32 / labels.num_elements() as f32)
}
