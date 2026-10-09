//! Action-policy transformations.

use crate::{DType, Error, Matrix, OpAttribute, OperationContract, Result, matrix};

use crate::runtime::SemanticDispatch;

/// One categorical action and its differentiable policy statistics.
pub struct PolicyResult {
	/// I32 action index for every environment.
	pub action: Matrix,
	/// Log probability of each selected action.
	pub log_probability: Matrix,
	/// Categorical entropy for every environment.
	pub entropy: Matrix,
	/// Critic value passed through without a storage copy.
	pub value: Matrix,
}

/// One bounded continuous action and its differentiable policy statistics.
pub struct ContinuousPolicyResult {
	/// Action after tanh squashing and affine mapping to the requested range.
	pub action: Matrix,
	/// Unsquashed normal sample, retained for exact policy reevaluation.
	pub raw_action: Matrix,
	/// Summed corrected log probability for every environment.
	pub log_probability: Matrix,
	/// Summed base-normal entropy for every environment.
	pub entropy: Matrix,
	/// Critic value passed through without a storage copy.
	pub value: Matrix,
}

#[derive(Clone, Copy)]
struct ContinuousRange {
	minimum: f32,
	maximum: f32,
	epsilon: f32,
}

/// Differentiably evaluate stored categorical actions.
///
/// Gradients flow through `logits`, `log_probability`, and `entropy`; action
/// indices remain discrete. `action` and `value` are zero-copy semantic output
/// aliases, matching OA C++.
///
/// # Errors
///
/// Returns an error when logits are not FP32 `[E,A]` with `E > 0` and `A > 1`,
/// actions are not I32 `[E]`, values are not FP32 `[E]`, inputs do not share
/// one engine, or composite lowering fails.
pub fn evaluate_categorical(
	logits: &Matrix,
	action: &Matrix,
	value: &Matrix,
) -> Result<PolicyResult> {
	validate_categorical(
		crate::core::operation::ml::EVALUATE_CATEGORICAL.name(),
		logits,
		Some(action),
		value,
	)?;
	let lowering = logits.engine_handle().begin_semantic_lowering()?;
	let result = evaluate_categorical_lowering(logits, action, value)?;
	let inputs = [logits, action, value];
	let outputs = [
		&result.action,
		&result.log_probability,
		&result.entropy,
		&result.value,
	];
	lowering.commit(SemanticDispatch {
		contract: crate::core::operation::ml::EVALUATE_CATEGORICAL,
		inputs: &inputs,
		outputs: &outputs,
		attributes: &[],
	})?;
	Ok(result)
}

/// Sample and evaluate one categorical action per environment.
///
/// A nonzero seed is deterministic. Seed zero follows the process-local OA RNG
/// convention. The action is sampled from the same logits used for returned
/// log probabilities and entropy.
///
/// # Errors
///
/// Returns an error when logits are not FP32 `[E,A]` with `E > 0` and `A > 1`,
/// values are not FP32 `[E]` on the same engine, or composite lowering fails.
pub fn sample_categorical(logits: &Matrix, value: &Matrix, seed: u64) -> Result<PolicyResult> {
	validate_categorical(
		crate::core::operation::ml::SAMPLE_CATEGORICAL.name(),
		logits,
		None,
		value,
	)?;
	let lowering = logits.engine_handle().begin_semantic_lowering()?;
	let action = matrix::sample_logits(logits, 1.0, 0, 1.0, seed)?;
	let result = evaluate_categorical(logits, &action, value)?;
	let inputs = [logits, value];
	let outputs = [
		&result.action,
		&result.log_probability,
		&result.entropy,
		&result.value,
	];
	let attributes = [OpAttribute::UnsignedInteger {
		name: "seed".into(),
		value: seed,
	}];
	lowering.commit(SemanticDispatch {
		contract: crate::core::operation::ml::SAMPLE_CATEGORICAL,
		inputs: &inputs,
		outputs: &outputs,
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Differentiably evaluate a stored pre-tanh normal action.
///
/// Log standard deviation is clamped to `[-20, 2]`. Actions are mapped from
/// `[-1, 1]` into `[minimum, maximum]`; log probability includes the donor tanh
/// Jacobian correction, while entropy is the unsquashed diagonal-normal value.
/// `raw_action` and `value` are zero-copy semantic output aliases.
///
/// # Errors
///
/// Returns an error when mean, log-standard-deviation, and raw-action are not
/// matching nonempty FP32 `[E,A]` matrices, value is not FP32 `[E]`, inputs do
/// not share one engine, bounds/epsilon are invalid, a size exceeds the shader
/// ABI, or composite lowering fails.
pub fn evaluate_tanh_normal(
	mean: &Matrix,
	log_stddev: &Matrix,
	raw_action: &Matrix,
	value: &Matrix,
	minimum: f32,
	maximum: f32,
	epsilon: f32,
) -> Result<ContinuousPolicyResult> {
	const CONTRACT: OperationContract = crate::core::operation::ml::EVALUATE_TANH_NORMAL;
	let range = ContinuousRange {
		minimum,
		maximum,
		epsilon,
	};
	validate_continuous(
		CONTRACT.name(),
		mean,
		log_stddev,
		Some(raw_action),
		value,
		range,
	)?;
	let lowering = mean.engine_handle().begin_semantic_lowering()?;
	let result = evaluate_tanh_normal_lowering(mean, log_stddev, raw_action, value, range)?;
	let inputs = [mean, log_stddev, raw_action, value];
	let outputs = [
		&result.action,
		&result.raw_action,
		&result.log_probability,
		&result.entropy,
		&result.value,
	];
	let attributes = continuous_attributes(range, None);
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &inputs,
		outputs: &outputs,
		attributes: &attributes,
	})?;
	Ok(result)
}

/// Sample a reparameterized tanh-normal policy.
///
/// The returned raw action is `mean + exp(clamped_log_stddev) * noise`, so
/// gradients flow through both distribution parameters while Philox noise stays
/// detached. A nonzero seed is deterministic; seed zero follows OA's
/// process-local RNG convention.
///
/// # Errors
///
/// Returns an error when mean/log-standard-deviation are not matching nonempty
/// FP32 `[E,A]` matrices, value is not FP32 `[E]` on the same engine,
/// bounds/epsilon are invalid, a size exceeds the shader ABI, or composite
/// lowering fails.
pub fn sample_tanh_normal(
	mean: &Matrix,
	log_stddev: &Matrix,
	value: &Matrix,
	minimum: f32,
	maximum: f32,
	seed: u64,
	epsilon: f32,
) -> Result<ContinuousPolicyResult> {
	const CONTRACT: OperationContract = crate::core::operation::ml::SAMPLE_TANH_NORMAL;
	let range = ContinuousRange {
		minimum,
		maximum,
		epsilon,
	};
	validate_continuous(CONTRACT.name(), mean, log_stddev, None, value, range)?;
	let lowering = mean.engine_handle().begin_semantic_lowering()?;
	let clamped_log_stddev = matrix::clamp_max(&matrix::clamp_min(log_stddev, -20.0)?, 2.0)?;
	let noise = matrix::philox_normal(mean, 0.0, 1.0, seed)?;
	let raw_action = matrix::add(
		mean,
		&matrix::mul(&matrix::exp(&clamped_log_stddev)?, &noise)?,
	)?;
	let result = evaluate_tanh_normal(
		mean,
		&clamped_log_stddev,
		&raw_action,
		value,
		minimum,
		maximum,
		epsilon,
	)?;
	let inputs = [mean, log_stddev, value];
	let outputs = [
		&result.action,
		&result.raw_action,
		&result.log_probability,
		&result.entropy,
		&result.value,
	];
	let attributes = continuous_attributes(range, Some(seed));
	lowering.commit(SemanticDispatch {
		contract: CONTRACT,
		inputs: &inputs,
		outputs: &outputs,
		attributes: &attributes,
	})?;
	Ok(result)
}

fn evaluate_categorical_lowering(
	logits: &Matrix,
	action: &Matrix,
	value: &Matrix,
) -> Result<PolicyResult> {
	let environments = logits.shape()[0];
	let log_probability_all = matrix::log_softmax(logits, 1)?;
	let action_column = matrix::reshape(action, [environments, 1])?;
	let selected = matrix::gather_last_dim(&log_probability_all, &action_column)?;
	let probability = matrix::exp(&log_probability_all)?;
	let entropy_column = matrix::neg(&matrix::sum(
		&matrix::mul(&probability, &log_probability_all)?,
		1,
	)?)?;
	Ok(PolicyResult {
		action: action.clone(),
		log_probability: matrix::reshape_semantic_output(&selected, vec![environments])?,
		entropy: matrix::reshape_semantic_output(&entropy_column, vec![environments])?,
		value: value.clone(),
	})
}

fn evaluate_tanh_normal_lowering(
	mean: &Matrix,
	log_stddev: &Matrix,
	raw_action: &Matrix,
	value: &Matrix,
	range: ContinuousRange,
) -> Result<ContinuousPolicyResult> {
	const LOG_TWO_PI: f32 = 1.837_877_1;
	let environments = mean.shape()[0];
	let scale = 0.5 * (range.maximum - range.minimum);
	let bias = 0.5 * (range.maximum + range.minimum);
	let log_stddev = matrix::clamp_max(&matrix::clamp_min(log_stddev, -20.0)?, 2.0)?;
	let stddev = matrix::exp(&log_stddev)?;
	let normalized = matrix::div(&matrix::sub(raw_action, mean)?, &stddev)?;
	let base_log_probability = matrix::scale(
		&matrix::add_scalar(
			&matrix::add(
				&matrix::mul(&normalized, &normalized)?,
				&matrix::scale(&log_stddev, 2.0)?,
			)?,
			LOG_TWO_PI,
		)?,
		-0.5,
	)?;
	let squashed = super::matrix::tanh(raw_action)?;
	let action = matrix::add_scalar(&matrix::scale(&squashed, scale)?, bias)?;
	let jacobian = matrix::log(&matrix::add_scalar(
		&matrix::scale(&matrix::mul(&squashed, &squashed)?, -1.0)?,
		1.0 + range.epsilon,
	)?)?;
	let corrected = matrix::sub(
		&base_log_probability,
		&matrix::add_scalar(&jacobian, scale.ln())?,
	)?;
	let entropy_per_dimension = matrix::add_scalar(&log_stddev, 0.5 * (1.0 + LOG_TWO_PI))?;
	let log_probability_column = matrix::sum(&corrected, 1)?;
	let entropy_column = matrix::sum(&entropy_per_dimension, 1)?;
	Ok(ContinuousPolicyResult {
		action,
		raw_action: raw_action.clone(),
		log_probability: matrix::reshape_semantic_output(&log_probability_column, vec![environments])?,
		entropy: matrix::reshape_semantic_output(&entropy_column, vec![environments])?,
		value: value.clone(),
	})
}

fn continuous_attributes(range: ContinuousRange, seed: Option<u64>) -> Vec<OpAttribute> {
	let mut attributes = vec![
		OpAttribute::Float {
			name: "minimum".into(),
			value: f64::from(range.minimum),
		},
		OpAttribute::Float {
			name: "maximum".into(),
			value: f64::from(range.maximum),
		},
	];
	if let Some(seed) = seed {
		attributes.push(OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		});
	}
	attributes.push(OpAttribute::Float {
		name: "epsilon".into(),
		value: f64::from(range.epsilon),
	});
	attributes
}

fn validate_categorical(
	operation: &'static str,
	logits: &Matrix,
	action: Option<&Matrix>,
	value: &Matrix,
) -> Result<()> {
	let [environments, actions] = logits.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} expects FP32 logits [E,A]"
		)));
	};
	if logits.dtype() != DType::F32
		|| *environments == 0
		|| *actions <= 1
		|| value.dtype() != DType::F32
		|| value.shape() != [*environments]
		|| !logits.engine_handle().same_as(value.engine_handle())
	{
		return Err(Error::invalid_argument(format!(
			"{operation} expects FP32 logits [E,A] and values [E] on one engine"
		)));
	}
	u32::try_from(*environments)
		.map_err(|_| Error::invalid_argument(format!("{operation} environment count exceeds u32")))?;
	if let Some(action) = action
		&& (action.dtype() != DType::I32
			|| action.shape() != [*environments]
			|| !logits.engine_handle().same_as(action.engine_handle()))
	{
		return Err(Error::invalid_argument(format!(
			"{operation} expects I32 actions [E] on the logits engine"
		)));
	}
	Ok(())
}

fn validate_continuous(
	operation: &'static str,
	mean: &Matrix,
	log_stddev: &Matrix,
	raw_action: Option<&Matrix>,
	value: &Matrix,
	range: ContinuousRange,
) -> Result<()> {
	let [environments, actions] = mean.shape() else {
		return Err(Error::invalid_argument(format!(
			"{operation} expects FP32 mean [E,A]"
		)));
	};
	let same_engine = mean.engine_handle().same_as(log_stddev.engine_handle())
		&& mean.engine_handle().same_as(value.engine_handle())
		&& raw_action.is_none_or(|raw| mean.engine_handle().same_as(raw.engine_handle()));
	if mean.dtype() != DType::F32
		|| *environments == 0
		|| *actions == 0
		|| log_stddev.dtype() != DType::F32
		|| log_stddev.shape() != mean.shape()
		|| value.dtype() != DType::F32
		|| value.shape() != [*environments]
		|| raw_action.is_some_and(|raw| raw.dtype() != DType::F32 || raw.shape() != mean.shape())
		|| !same_engine
		|| !range.minimum.is_finite()
		|| !range.maximum.is_finite()
		|| range.minimum >= range.maximum
		|| !range.epsilon.is_finite()
		|| !(0.0..1.0).contains(&range.epsilon)
		|| range.epsilon == 0.0
	{
		return Err(Error::invalid_argument(format!(
			"{operation} expects matching FP32 [E,A] inputs, FP32 values [E], finite ordered bounds, and epsilon in (0,1) on one engine"
		)));
	}
	u32::try_from(*environments)
		.map_err(|_| Error::invalid_argument(format!("{operation} environment count exceeds u32")))?;
	u32::try_from(mean.element_count())
		.map_err(|_| Error::invalid_argument(format!("{operation} element count exceeds u32")))?;
	Ok(())
}
