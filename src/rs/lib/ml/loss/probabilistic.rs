//! Probabilistic losses composed over Matrix operations; reverse mode flows
//! through the composed operation recipes. Every function returns the scalar
//! batch mean.

use crate::matrix;
use crate::{Error, Matrix, Result};

use super::common::validate_pointwise_loss_inputs;

/// Compute the KL divergence loss `mean(target * (log target - input))`.
///
/// `input` holds log-probabilities and `target` holds probabilities, matching
/// `torch.nn.functional.kl_div` with log-target input. The `target == 0`
/// contribution is exact zero; targets below `1e-12` clamp before the log so
/// the product stays finite.
///
/// # Errors
///
/// Returns an error for mismatched, empty, or non-FP32 inputs, or when runtime
/// recording fails.
pub fn kl_div(input: &Matrix, target: &Matrix) -> Result<Matrix> {
	validate_pointwise_loss_inputs(input, target, "kl_div")?;
	let log_target = matrix::log(&matrix::clamp_min(target, 1e-12)?)?;
	let divergence = matrix::sub(
		&matrix::mul(target, &log_target)?,
		&matrix::mul(target, input)?,
	)?;
	matrix::mean(&divergence, -1)?.reshape([])
}

/// Compute the negative log likelihood loss `-mean(x[i, target_i])`.
///
/// `input` is an `[N, C]` score matrix (usually log-probabilities) and
/// `target` an `I32 [N, 1]` class-index matrix.
///
/// # Errors
///
/// Returns an error for non-rank-two, empty, or non-FP32 inputs, mismatched or
/// out-of-range targets, or when runtime recording fails.
pub fn nll(input: &Matrix, target: &Matrix) -> Result<Matrix> {
	let selected = matrix::gather_last_dim(input, target)?;
	matrix::neg(&matrix::mean(&selected, -1)?)?.reshape([])
}

/// Compute the Gaussian negative log likelihood
/// `mean(0.5 * (log var + (x - t)^2 / var) + full * 0.5 * log(2 pi))`.
///
/// `variance` is clamped below at `eps`, matching the donor's stability rule.
/// `full` adds the constant `0.5 * log(2 pi)` term.
///
/// # Errors
///
/// Returns an error for mismatched, empty, or non-FP32 inputs, nonpositive
/// `eps`, or when runtime recording fails.
pub fn gaussian_nll(
	prediction: &Matrix,
	target: &Matrix,
	variance: &Matrix,
	eps: f32,
	full: bool,
) -> Result<Matrix> {
	let operation = "gaussian_nll";
	validate_pointwise_loss_inputs(prediction, target, operation)?;
	validate_pointwise_loss_inputs(prediction, variance, operation)?;
	if !eps.is_finite() || eps <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{operation} eps must be a positive finite scalar"
		)));
	}
	let clamped = matrix::clamp_min(variance, eps)?;
	let residual = matrix::sub(prediction, target)?;
	let term = matrix::scale(
		&matrix::add(
			&matrix::log(&clamped)?,
			&matrix::div(&matrix::mul(&residual, &residual)?, &clamped)?,
		)?,
		0.5,
	)?;
	let term = if full {
		matrix::add_scalar(&term, 0.5 * std::f32::consts::TAU.ln())?
	} else {
		term
	};
	matrix::mean(&term, -1)?.reshape([])
}

/// Compute the Poisson negative log likelihood
/// `mean(exp(input) - target * input)` for log-rate `input`.
///
/// Matches the donor's `log_input` contract: `input` is the log-rate, so the
/// rate is `exp(input)`. The Stirling correction (`full`) is not part of this
/// contract.
///
/// # Errors
///
/// Returns an error for mismatched, empty, or non-FP32 inputs, or when runtime
/// recording fails.
pub fn poisson_nll(input: &Matrix, target: &Matrix) -> Result<Matrix> {
	validate_pointwise_loss_inputs(input, target, "poisson_nll")?;
	let term = matrix::sub(&matrix::exp(input)?, &matrix::mul(target, input)?)?;
	matrix::mean(&term, -1)?.reshape([])
}

/// Compute the multi-label soft margin loss (binary cross-entropy with
/// logits, averaged over classes then batch).
///
/// `input` is an `[N, C]` logit matrix and `target` an `[N, C]` 0/1 mask.
/// `weight` optionally scales each class term by an `[N, C]`-broadcastable
/// weight matrix. Uses the stable softplus form so large logits do not
/// overflow.
///
/// # Errors
///
/// Returns an error for mismatched, empty, non-rank-two, or non-FP32 inputs, or
/// when runtime recording fails.
pub fn multi_label_soft_margin(
	input: &Matrix,
	target: &Matrix,
	weight: Option<&Matrix>,
) -> Result<Matrix> {
	let operation = "multi_label_soft_margin";
	validate_pointwise_loss_inputs(input, target, operation)?;
	if input.shape().len() != 2 {
		return Err(Error::invalid_argument(format!(
			"{operation} requires rank-two [N, C] inputs"
		)));
	}
	let mut term = matrix::add(
		&matrix::clamp_min(input, 0.0)?,
		&matrix::add(
			&matrix::neg(&matrix::mul(input, target)?)?,
			&matrix::log(&matrix::add_scalar(
				&matrix::exp(&matrix::neg(&matrix::abs(input)?)?)?,
				1.0,
			)?)?,
		)?,
	)?;
	if let Some(weight) = weight {
		if !weight.engine_handle().same_as(input.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{operation} weight must belong to the same engine"
			)));
		}
		term = matrix::mul(&term, weight)?;
	}
	matrix::mean(&matrix::mean(&term, 1)?, -1)?.reshape([])
}
