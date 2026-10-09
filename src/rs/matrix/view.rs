//! Zero-copy Matrix views and their autograd lineage.

use crate::{Matrix, Result, matrix::autograd};

/// Return a zero-copy dense view with a different shape.
///
/// The result owns a new semantic value identity but shares storage, pending
/// execution state, dtype, and element order with `input`. When a gradient tape
/// is active, its adjoint reshapes the incoming gradient back to the input shape.
///
/// # Errors
///
/// Returns an error when the requested shape overflows or changes the number of
/// elements.
pub fn reshape(input: &Matrix, shape: impl Into<Vec<usize>>) -> Result<Matrix> {
	let output = input.reshape_view(shape.into())?;
	autograd::record_reshape(input, &output)?;
	Ok(output)
}

pub(crate) fn reshape_semantic_output(input: &Matrix, shape: Vec<usize>) -> Result<Matrix> {
	let output = input.reshape_semantic_output(shape)?;
	autograd::record_reshape(input, &output)?;
	Ok(output)
}

/// Restore the source shape without copying device storage.
pub(crate) fn reshape_backward(output_gradient: &Matrix, input_shape: &[usize]) -> Result<Matrix> {
	output_gradient.reshape(input_shape.to_vec())
}
