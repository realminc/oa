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
