use crate::{DType, Error, Matrix, OpAttribute, Result, matrix};

use crate::runtime::SemanticDispatch;

use super::lowering;

/// Discount parameters for generalized advantage estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaeConfig {
	/// Reward discount applied to value bootstrap and the reverse trace.
	pub gamma: f32,
	/// Trace discount applied to the next advantage.
	pub lambda: f32,
}

impl Default for GaeConfig {
	fn default() -> Self {
		Self {
			gamma: 0.99,
			lambda: 0.95,
		}
	}
}

/// Generalized advantages and their matching value targets.
pub struct GaeResult {
	/// Advantage estimates with the same `[time, environments]` shape as reward.
	pub advantage: Matrix,
	/// Returns computed as `advantage + value`.
	pub returns: Matrix,
}

/// Standardize a nonempty FP32 advantage tensor over all elements.
///
/// The result is `(advantage - mean) / sqrt(variance + epsilon)`. The operation
/// remains device-resident and differentiable, and records one semantic
/// identity over its composite Matrix lowering.
///
/// # Errors
///
/// Returns an error when `advantage` is empty or not FP32, `epsilon` is not
/// positive and finite, a size exceeds the admitted shader ABI, allocation
/// fails, or composite lowering fails.
pub fn normalize(advantage: &Matrix, epsilon: f32) -> Result<Matrix> {
	const CONTRACT: crate::OperationContract = crate::core::operation::ml::NORMALIZE;
	if advantage.dtype() != DType::F32 || advantage.num_elements() == 0 {
		return Err(Error::invalid_argument(format!(
			"{} requires a nonempty F32 matrix",
			CONTRACT.name()
		)));
	}
	if !epsilon.is_finite() || epsilon <= 0.0 {
		return Err(Error::invalid_argument(format!(
			"{} requires a positive finite epsilon",
			CONTRACT.name()
		)));
	}
	let lowering = advantage.engine_handle().begin_semantic_lowering()?;
	let inverse_count = 1.0 / advantage.num_elements() as f32;
	let mean = matrix::scale(&matrix::sum(advantage, -1)?, inverse_count)?;
	let centered = matrix::sub(advantage, &mean)?;
	let variance = matrix::scale(
		&matrix::sum(&matrix::mul(&centered, &centered)?, -1)?,
		inverse_count,
	)?;
	let denominator = matrix::sqrt(&matrix::add_scalar(&variance, epsilon)?)?;
	let result = matrix::div(&centered, &denominator)?;
	let attributes = [OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(epsilon),
	}];
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &[advantage],
		outputs: &[&result],
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Compute generalized advantage estimates over a fixed rollout.
///
/// Termination disables value bootstrap. Termination and truncation both stop
/// the reverse trace, while truncation deliberately retains one-step bootstrap.
///
/// # Errors
///
/// Returns an error unless all inputs share one engine and one nonempty
/// `[time, environments]` shape, values are F32, boundary masks are U8, both
/// discounts are finite in `[0, 1]`, and GPU recording succeeds.
pub fn gae(
	reward: &Matrix,
	value: &Matrix,
	next_value: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	config: GaeConfig,
) -> Result<GaeResult> {
	lowering::advantage::gae(reward, value, next_value, terminated, truncated, config)
}

/// Record generalized advantage estimation into caller-owned output storage.
///
/// This is the allocation-free form used by fixed-capacity rollout storage. It
/// has the same semantic identity and numerical contract as [`gae`].
///
/// # Errors
///
/// Returns the errors from [`gae`], and rejects output matrices that do not
/// match the rollout's engine, F32 dtype, and shape or alias any input/each other.
#[allow(
	clippy::too_many_arguments,
	reason = "the allocation-free API keeps every rollout field and both outputs explicit"
)]
pub fn gae_into(
	reward: &Matrix,
	value: &Matrix,
	next_value: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	advantage: &mut Matrix,
	returns: &mut Matrix,
	config: GaeConfig,
) -> Result<()> {
	lowering::advantage::gae_into(
		reward, value, next_value, terminated, truncated, advantage, returns, config,
	)
}
