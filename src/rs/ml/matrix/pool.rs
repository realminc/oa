//! ML-owned Matrix pooling operations.

use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::pool as dispatch};

/// Pooled values and their exact U32 input argmax indices.
#[must_use]
pub struct MaxPool2dResult {
	/// Maximum value selected from each pooling window.
	pub output: Matrix,
	/// Flat input index selected for each output element.
	pub indices: Matrix,
}

/// Apply two-dimensional average pooling to an NCHW Matrix.
///
/// Padding positions do not contribute to either the sum or divisor. The
/// output spatial extent is `floor((input + 2 * padding - kernel_size) /
/// stride) + 1`, matching the OA C++ operation.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty rank-four F32 Matrix,
/// `kernel_size` and `stride` are nonzero, the kernel fits the padded input,
/// every shader extent is representable, or runtime recording fails.
pub fn avg_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let output = dispatch::avg_pool_2d(input, kernel_size, stride, padding)?;
	autograd::record_avg_pool_2d(input, &output, kernel_size, stride, padding)?;
	Ok(output)
}

pub(in crate::ml) fn avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	dispatch::avg_pool_2d_backward(input, output_gradient, kernel_size, stride, padding)
}

/// Apply two-dimensional max pooling to an NCHW Matrix.
///
/// Equal maxima select the first valid input in row-major window order. The
/// returned U32 argmax indices are explicit because they are part of the donor
/// operation contract and make the deterministic adjoint reusable.
///
/// # Errors
///
/// Returns an error under the same geometry and runtime conditions as
/// [`avg_pool_2d`].
pub fn max_pool_2d(
	input: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<MaxPool2dResult> {
	let result = dispatch::max_pool_2d(input, kernel_size, stride, padding)?;
	autograd::record_max_pool_2d(
		input,
		&result.output,
		result.indices.clone(),
		kernel_size,
		stride,
		padding,
	)?;
	Ok(MaxPool2dResult {
		output: result.output,
		indices: result.indices,
	})
}

pub(in crate::ml) fn max_pool_2d_backward(
	input: &Matrix,
	indices: &Matrix,
	output_gradient: &Matrix,
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	dispatch::max_pool_2d_backward(
		input,
		indices,
		output_gradient,
		kernel_size,
		stride,
		padding,
	)
}

/// Adaptively average-pool an NCHW Matrix to an exact spatial extent.
///
/// Each output bin independently uses floor/ceiling boundaries along height
/// and width. This supports rectangular and non-divisible geometry without
/// approximating the operation as a square fixed-window pool.
///
/// # Errors
///
/// Returns an error unless the input is nonempty rank-four F32, both output
/// extents are nonzero, all geometry fits the shader ABI, or recording fails.
pub fn adaptive_avg_pool_2d(
	input: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	let output = dispatch::adaptive_avg_pool_2d(input, output_height, output_width)?;
	autograd::record_adaptive_avg_pool_2d(input, &output, output_height, output_width)?;
	Ok(output)
}

pub(in crate::ml) fn adaptive_avg_pool_2d_backward(
	input: &Matrix,
	output_gradient: &Matrix,
	output_height: usize,
	output_width: usize,
) -> Result<Matrix> {
	dispatch::adaptive_avg_pool_2d_backward(input, output_gradient, output_height, output_width)
}
