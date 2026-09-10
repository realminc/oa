//! ML-owned Matrix convolution operations.

use crate::{Matrix, Result};

use crate::ml::{Parameter, autograd, lowering::conv as dispatch};

/// Apply one-dimensional convolution to an NCL Matrix through im2col and GEMM.
///
/// `weight` uses `[output_channels, input_channels, kernel]` layout and `bias`
/// uses `[output_channels]`. Dilation applies within each kernel window.
///
/// # Errors
///
/// Returns an error unless all inputs are nonempty same-engine F32 matrices,
/// shapes agree, stride and dilation are nonzero, the effective kernel fits
/// the padded input, every extent fits the shader ABI, or recording fails.
pub fn conv_1d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let output = dispatch::conv_1d(input, weight, bias, stride, padding, dilation)?;
	autograd::record_conv_1d(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		stride,
		padding,
		dilation,
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "module forwarding keeps parameter identity separate from Matrix values"
)]
pub(in crate::ml) fn conv_1d_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: (Parameter, Matrix, u64),
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let (bias_parameter, bias_value, bias_version) = bias;
	let output = dispatch::conv_1d(input, &weight_value, &bias_value, stride, padding, dilation)?;
	autograd::record_conv_1d(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		Some((bias_parameter, bias_version)),
		bias_value,
		stride,
		padding,
		dilation,
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_1d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::conv_1d_backward(input, weight, output_gradient, stride, padding, dilation)
}

/// Apply bias-free one-dimensional transposed convolution to an NCL Matrix.
///
/// `weight` uses `[input_channels, output_channels, kernel]` layout. The output
/// length is `(input_length - 1) * stride - 2 * padding + effective_kernel`.
///
/// # Errors
///
/// Returns an error unless both inputs are nonempty same-engine F32 matrices,
/// channel geometry agrees, stride and dilation are nonzero, the output extent
/// is positive and fits the shader ABI, or runtime recording fails.
pub fn conv_transpose_1d(
	input: &Matrix,
	weight: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let output = dispatch::conv_transpose_1d(input, weight, stride, padding, dilation)?;
	autograd::record_conv_transpose_1d(
		input,
		&output,
		None,
		weight.clone(),
		stride,
		padding,
		dilation,
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_transpose_1d_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let output = dispatch::conv_transpose_1d(input, &weight_value, stride, padding, dilation)?;
	autograd::record_conv_transpose_1d(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		stride,
		padding,
		dilation,
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_transpose_1d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Result<(Matrix, Matrix)> {
	dispatch::conv_transpose_1d_backward(input, weight, output_gradient, stride, padding, dilation)
}

/// Apply two-dimensional transposed convolution to an NCHW Matrix.
///
/// `weight` uses `[input_channels, output_channels, kernel, kernel]` layout and
/// `bias` uses `[output_channels]`. Each output spatial extent is
/// `(input_extent - 1) * stride - 2 * padding + kernel`.
///
/// # Errors
///
/// Returns an error unless all inputs are nonempty same-engine F32 matrices,
/// channel and square-kernel geometry agree, stride is nonzero, output extents
/// are positive and fit the shader ABI, or runtime recording fails.
pub fn conv_transpose_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let output = dispatch::conv_transpose_2d(input, weight, bias, stride, padding)?;
	autograd::record_conv_transpose_2d(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		stride,
		padding,
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "module forwarding keeps parameter identity separate from Matrix values"
)]
pub(in crate::ml) fn conv_transpose_2d_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: (Parameter, Matrix, u64),
	stride: usize,
	padding: usize,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let (bias_parameter, bias_value, bias_version) = bias;
	let output = dispatch::conv_transpose_2d(input, &weight_value, &bias_value, stride, padding)?;
	autograd::record_conv_transpose_2d(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		Some((bias_parameter, bias_version)),
		bias_value,
		stride,
		padding,
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_transpose_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::conv_transpose_2d_backward(input, weight, output_gradient, stride, padding)
}

/// Apply grouped two-dimensional convolution to an NCHW Matrix.
///
/// `weight` uses `[output_channels, input_channels / groups, kernel, kernel]`
/// layout and `bias` uses `[output_channels]`. The output spatial extent is
/// `floor((input + 2 * padding - kernel) / stride) + 1`.
///
/// # Errors
///
/// Returns an error unless all inputs are same-engine F32 matrices, shapes and
/// group divisibility agree, the square kernel fits the padded input, every
/// extent fits the shader ABI, or runtime recording fails.
pub fn conv_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<Matrix> {
	let output = dispatch::conv_2d(input, weight, bias, stride, padding, groups)?;
	autograd::record_conv_2d(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		stride,
		padding,
		groups,
	)?;
	Ok(output)
}

#[allow(
	clippy::too_many_arguments,
	reason = "module forwarding keeps parameter identity separate from Matrix values"
)]
pub(in crate::ml) fn conv_2d_parameterized(
	input: &Matrix,
	weight: (Parameter, Matrix, u64),
	bias: (Parameter, Matrix, u64),
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<Matrix> {
	let (weight_parameter, weight_value, weight_version) = weight;
	let (bias_parameter, bias_value, bias_version) = bias;
	let output = dispatch::conv_2d(input, &weight_value, &bias_value, stride, padding, groups)?;
	autograd::record_conv_2d(
		input,
		&output,
		Some((weight_parameter, weight_version)),
		weight_value,
		Some((bias_parameter, bias_version)),
		bias_value,
		stride,
		padding,
		groups,
	)?;
	Ok(output)
}

pub(in crate::ml) fn conv_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	stride: usize,
	padding: usize,
	groups: usize,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::conv_2d_backward(input, weight, output_gradient, stride, padding, groups)
}
