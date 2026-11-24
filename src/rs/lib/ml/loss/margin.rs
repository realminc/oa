//! Margin-family losses composed over Matrix operations; reverse mode flows
//! through the composed operation recipes, so no dedicated adjoint kernels are
//! required. Every function returns the scalar batch mean.

use crate::matrix;
use crate::{Error, Matrix, Result};

use super::common::validate_pointwise_loss_inputs;

/// Compute the margin ranking loss `mean(max(0, -y * (x1 - x2) + margin))`.
///
/// `target` is the ±1 direction mask matching the input shapes.
///
/// # Errors
///
/// Returns an error for mismatched, empty, or non-FP32 inputs, or when runtime
/// recording fails.
pub fn margin_ranking(
	input1: &Matrix,
	input2: &Matrix,
	target: &Matrix,
	margin: f32,
) -> Result<Matrix> {
	validate_pointwise_loss_inputs(input1, input2, "margin_ranking")?;
	validate_pointwise_loss_inputs(input1, target, "margin_ranking")?;
	let hinge = matrix::clamp_min(
		&matrix::add_scalar(
			&matrix::mul(&matrix::neg(&matrix::sub(input1, input2)?)?, target)?,
			margin,
		)?,
		0.0,
	)?;
	matrix::mean(&hinge, -1)?.reshape([])
}

/// Compute the soft margin loss `mean(log(1 + exp(-y * x)))`.
///
/// Uses the stable softplus form `max(z, 0) + log(1 + exp(-|z|))` with
/// `z = -y * x`, so large margins saturate without overflow.
///
/// # Errors
///
/// Returns an error for mismatched, empty, or non-FP32 inputs, or when runtime
/// recording fails.
pub fn soft_margin(prediction: &Matrix, target: &Matrix) -> Result<Matrix> {
	validate_pointwise_loss_inputs(prediction, target, "soft_margin")?;
	let logits = matrix::neg(&matrix::mul(prediction, target)?)?;
	let stable = matrix::add(
		&matrix::clamp_min(&logits, 0.0)?,
		&matrix::log(&matrix::add_scalar(
			&matrix::exp(&matrix::neg(&matrix::abs(&logits)?)?)?,
			1.0,
		)?)?,
	)?;
	matrix::mean(&stable, -1)?.reshape([])
}

/// Compute the multiclass hinge loss
/// `mean_i((1/C) * sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j]))`.
///
/// `input` is an `[N, C]` score matrix and `target` an `I32 [N, 1]` class-index
/// matrix. The `j == y_i` hinge term equals `margin`, so it is subtracted
/// exactly after the full row sum.
///
/// # Errors
///
/// Returns an error for non-rank-two, empty, or non-FP32 inputs, mismatched
/// targets, or when runtime recording fails.
pub fn multi_margin(input: &Matrix, target: &Matrix, margin: f32) -> Result<Matrix> {
	let operation = "multi_margin";
	if !margin.is_finite() || margin < 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} margin must be a nonnegative finite scalar"
		)));
	}
	let correct = matrix::gather_last_dim(input, target)?;
	let width = input.shape()[1];
	let hinge = matrix::clamp_min(
		&matrix::add_scalar(&matrix::sub(input, &correct)?, margin)?,
		0.0,
	)?;
	let row_loss = matrix::scale(
		&matrix::sub_scalar(&matrix::sum(&hinge, 1)?, margin)?,
		1.0 / width as f32,
	)?;
	matrix::mean(&row_loss, -1)?.reshape([])
}

/// Compute the triplet margin loss `mean(max(0, d(a, p) - d(a, n) + margin))`
/// with L2 distances over the last axis.
///
/// All three inputs share one `[N, D]` shape; `swap` (the donor's distance
/// exchange) is not part of this contract.
///
/// # Errors
///
/// Returns an error for mismatched, empty, non-rank-two, or non-FP32 inputs, or
/// when runtime recording fails.
pub fn triplet_margin(
	anchor: &Matrix,
	positive: &Matrix,
	negative: &Matrix,
	margin: f32,
) -> Result<Matrix> {
	let operation = "triplet_margin";
	if !margin.is_finite() {
		return Err(Error::invalid_argument(format!(
			"{operation} margin must be a finite scalar"
		)));
	}
	validate_pointwise_loss_inputs(anchor, positive, operation)?;
	validate_pointwise_loss_inputs(anchor, negative, operation)?;
	if anchor.shape().len() != 2 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two [N, D] inputs"
		)));
	}
	let distance = |left: &Matrix, right: &Matrix| -> Result<Matrix> {
		let diff = matrix::sub(left, right)?;
		matrix::sqrt(&matrix::sum(&matrix::mul(&diff, &diff)?, 1)?)
	};
	let positive_distance = distance(anchor, positive)?;
	let negative_distance = distance(anchor, negative)?;
	let hinge = matrix::clamp_min(
		&matrix::add_scalar(
			&matrix::sub(&positive_distance, &negative_distance)?,
			margin,
		)?,
		0.0,
	)?;
	matrix::mean(&hinge, -1)?.reshape([])
}

/// Compute the cosine embedding loss over `[N, D]` pairs.
///
/// Rows with `target == 1` contribute `1 - cos(x1, x2)`; rows with
/// `target == -1` contribute `max(0, cos(x1, x2) - margin)`. `target` is an
/// `[N, 1]` ±1 mask.
///
/// # Errors
///
/// Returns an error for mismatched, empty, non-rank-two, or non-FP32 inputs, or
/// when runtime recording fails.
pub fn cosine_embedding(
	input1: &Matrix,
	input2: &Matrix,
	target: &Matrix,
	margin: f32,
) -> Result<Matrix> {
	let operation = "cosine_embedding";
	if !margin.is_finite() {
		return Err(Error::invalid_argument(format!(
			"{operation} margin must be a finite scalar"
		)));
	}
	validate_pointwise_loss_inputs(input1, input2, operation)?;
	validate_pointwise_loss_inputs(target, target, operation)?;
	if input1.shape().len() != 2
		|| target.shape() != [input1.shape()[0], 1]
		|| !target.engine_handle().same_as(input1.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two [N, D] inputs and an [N, 1] target"
		)));
	}
	let norm = |input: &Matrix| -> Result<Matrix> {
		matrix::sqrt(&matrix::sum(&matrix::mul(input, input)?, 1)?)
	};
	let numerator = matrix::sum(&matrix::mul(input1, input2)?, 1)?;
	let cosine = matrix::div(&numerator, &matrix::mul(&norm(input1)?, &norm(input2)?)?)?;
	let positive_mask = matrix::scale(&matrix::add_scalar(target, 1.0)?, 0.5)?;
	let similarity_loss = matrix::sub_scalar(&matrix::neg(&cosine)?, -1.0)?;
	let dissimilarity_loss = matrix::clamp_min(&matrix::sub_scalar(&cosine, margin)?, 0.0)?;
	let per_example = matrix::add(
		&matrix::mul(&positive_mask, &similarity_loss)?,
		&matrix::mul(
			&matrix::sub_scalar(&matrix::neg(&positive_mask)?, -1.0)?,
			&dissimilarity_loss,
		)?,
	)?;
	matrix::mean(&per_example, -1)?.reshape([])
}
