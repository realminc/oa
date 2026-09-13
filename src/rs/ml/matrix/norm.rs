use crate::{Matrix, Result};

use crate::ml::{autograd, lowering::matrix as dispatch};

/// Batch-normalized values and the per-channel statistics saved for backward.
#[must_use]
pub struct BatchNorm2dResult {
	/// Normalized and affine-transformed NCHW values.
	pub output: Matrix,
	/// Population mean for each channel.
	pub mean: Matrix,
	/// Population variance for each channel.
	pub variance: Matrix,
}

/// Complete adjoints of broadcast-affine gated RMS normalization.
pub struct RmsNormGatedBackward {
	/// Gradient of the normalized input.
	pub input: Matrix,
	/// Gradient of the broadcast affine weight.
	pub weight: Matrix,
	/// Gradient of the optional affine bias.
	pub bias: Option<Matrix>,
	/// Gradient of the SiLU gate input.
	pub gate: Matrix,
}

/// Complete adjoints of channel-axis normalization over BCT storage.
pub struct ChannelNormBackward {
	/// Gradient of the BCT input.
	pub input: Matrix,
	/// Gradient of the channel affine weight.
	pub weight: Matrix,
	/// Gradient of the channel affine bias.
	pub bias: Matrix,
}

pub(in crate::ml) fn channel_norm_forward(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
	relu: bool,
) -> Result<Matrix> {
	let output = dispatch::channel_norm(input, weight, bias, epsilon, relu)?;
	autograd::record_channel_norm(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		relu,
	)?;
	Ok(output)
}

/// Normalize the channel axis of a nonempty FP32 `[B, C, T]` Matrix.
///
/// # Errors
///
/// Returns an error unless the affine vectors match `C`, all values belong to
/// one Engine, epsilon is finite and positive, `C <= 1024`, or recording fails.
pub fn channel_norm(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	channel_norm_forward(input, weight, bias, epsilon, false)
}

/// Normalize the channel axis of `[B, C, T]` and fuse the following ReLU.
///
/// # Errors
///
/// Returns an error under the same conditions as [`channel_norm`], or when
/// runtime recording fails.
pub fn channel_norm_relu(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	channel_norm_forward(input, weight, bias, epsilon, true)
}

/// Compute complete explicit channel-normalization adjoints.
///
/// # Errors
///
/// Returns an error when the forward contract or output gradient is invalid,
/// or runtime recording fails.
pub fn channel_norm_backward(
	input: &Matrix,
	weight: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<ChannelNormBackward> {
	let result = dispatch::channel_norm_backward(input, weight, None, output_gradient, epsilon)?;
	Ok(ChannelNormBackward {
		input: result.input,
		weight: result.weight,
		bias: result.bias,
	})
}

/// Compute complete adjoints of fused channel normalization and ReLU.
///
/// # Errors
///
/// Returns an error when saved output or gradient does not match the forward
/// contract, or runtime recording fails.
pub fn channel_norm_relu_backward(
	input: &Matrix,
	weight: &Matrix,
	forward_output: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<ChannelNormBackward> {
	let result = dispatch::channel_norm_backward(
		input,
		weight,
		Some(forward_output),
		output_gradient,
		epsilon,
	)?;
	Ok(ChannelNormBackward {
		input: result.input,
		weight: result.weight,
		bias: result.bias,
	})
}

/// Normalize an NCHW Matrix using statistics computed from the current batch.
///
/// # Errors
///
/// Returns an error unless `input` is nonempty rank-four F32, `weight` and
/// `bias` are same-engine F32 channel vectors, `epsilon` is finite and
/// positive, or runtime recording fails.
pub fn batch_norm_2d(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<BatchNorm2dResult> {
	let result = batch_norm_2d_forward(input, weight, bias, epsilon)?;
	autograd::record_batch_norm_2d(
		input,
		&result.output,
		result.mean.clone(),
		result.variance.clone(),
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		true,
	)?;
	Ok(BatchNorm2dResult {
		output: result.output,
		mean: result.mean,
		variance: result.variance,
	})
}

pub(in crate::ml) fn batch_norm_2d_forward(
	input: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<dispatch::BatchNorm2dOutput> {
	dispatch::batch_norm_2d(input, weight, bias, epsilon)
}

/// Normalize an NCHW Matrix using explicit per-channel statistics.
///
/// This is the stateless inference form used by [`crate::ml::nn::BatchNorm2d`]
/// in evaluation mode.
///
/// # Errors
///
/// Returns an error unless every value is same-engine F32, the channel-vector
/// shapes match the input, `epsilon` is finite and positive, or runtime
/// recording fails.
pub fn batch_norm_2d_with_stats(
	input: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let output = batch_norm_2d_with_stats_forward(input, mean, variance, weight, bias, epsilon)?;
	autograd::record_batch_norm_2d(
		input,
		&output,
		mean.clone(),
		variance.clone(),
		None,
		weight.clone(),
		None,
		bias.clone(),
		epsilon,
		false,
	)?;
	Ok(output)
}

pub(in crate::ml) fn batch_norm_2d_with_stats_forward(
	input: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	weight: &Matrix,
	bias: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	dispatch::batch_norm_2d_with_stats(input, mean, variance, weight, bias, epsilon)
}

pub(in crate::ml) fn batch_norm_2d_running_update(
	running_mean: &Matrix,
	running_variance: &Matrix,
	batch_mean: &Matrix,
	batch_variance: &Matrix,
	momentum: f32,
) -> Result<(Matrix, Matrix)> {
	dispatch::batch_norm_2d_running_update(
		running_mean,
		running_variance,
		batch_mean,
		batch_variance,
		momentum,
	)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the adjoint consumes the exact saved BatchNorm state explicitly"
)]
pub(in crate::ml) fn batch_norm_2d_backward(
	input: &Matrix,
	weight: &Matrix,
	mean: &Matrix,
	variance: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
	training: bool,
) -> Result<(Matrix, Matrix, Matrix)> {
	dispatch::batch_norm_2d_backward(
		input,
		weight,
		mean,
		variance,
		output_gradient,
		epsilon,
		training,
	)
}

/// Apply weighted root-mean-square normalization over the final dimension.
///
/// # Errors
///
/// Returns an error unless `input` is a nonempty F32 Matrix, `weight` is a
/// same-engine F32 vector matching its final dimension, `epsilon` is finite and
/// positive, or runtime recording fails.
pub fn rms_norm(input: &Matrix, weight: &Matrix, epsilon: f32) -> Result<Matrix> {
	let output = dispatch::rms_norm(input, weight, epsilon)?;
	autograd::record_rms_norm(input, &output, None, weight.clone(), epsilon)?;
	Ok(output)
}

/// Apply per-row RMS normalization, a cyclic broadcast affine, and a SiLU gate.
///
/// Normalization is over the final input dimension. The affine value may have
/// one or more leading groups; those groups repeat over the flattened input
/// rows. An optional bias must exactly match `weight`.
///
/// # Errors
///
/// Returns an error for incompatible shape, dtype, Engine ownership or
/// epsilon, or when runtime recording fails.
pub fn rms_norm_gated(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	epsilon: f32,
) -> Result<Matrix> {
	let output = dispatch::rms_norm_gated(input, weight, bias, gate, epsilon)?;
	autograd::record_rms_norm_gated(
		input,
		&output,
		None,
		weight.clone(),
		None,
		bias.cloned(),
		gate,
		epsilon,
	)?;
	Ok(output)
}

/// Compute all gated RMS normalization adjoints explicitly.
///
/// # Errors
///
/// Returns an error for an incompatible output gradient or forward contract,
/// or when runtime recording fails.
pub fn rms_norm_gated_backward(
	input: &Matrix,
	weight: &Matrix,
	bias: Option<&Matrix>,
	gate: &Matrix,
	output_gradient: &Matrix,
	epsilon: f32,
) -> Result<RmsNormGatedBackward> {
	let result =
		dispatch::rms_norm_gated_backward(input, weight, bias, gate, output_gradient, epsilon)?;
	Ok(RmsNormGatedBackward {
		input: result.input,
		weight: result.weight,
		bias: result.bias,
		gate: result.gate,
	})
}
