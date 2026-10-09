use crate::runtime::ComputeDispatch;
use crate::runtime::{BufferBinding, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Error, Matrix, OpAttribute, Result, matrix};

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
	validate_inputs(reward, value, next_value, terminated, truncated, config)?;
	let mut advantage = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	let mut returns = Matrix::allocate(
		reward.engine_handle(),
		reward.shape().to_vec(),
		reward.num_elements(),
		DType::F32,
	)?;
	gae_into(
		reward,
		value,
		next_value,
		terminated,
		truncated,
		&mut advantage,
		&mut returns,
		config,
	)?;
	Ok(GaeResult { advantage, returns })
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
	const OPERATION: &str = "oa::ml::advantage::gae";
	validate_inputs(reward, value, next_value, terminated, truncated, config)?;
	for output in [&*advantage, &*returns] {
		if !reward.engine_handle().same_as(output.engine_handle())
			|| output.dtype() != DType::F32
			|| output.shape() != reward.shape()
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} outputs must be same-engine F32 matrices matching the rollout shape"
			)));
		}
	}
	let inputs = [reward, value, next_value, terminated, truncated];
	if advantage.storage().same_as(returns.storage())
		|| inputs.iter().any(|input| {
			input.storage().same_as(advantage.storage()) || input.storage().same_as(returns.storage())
		}) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} outputs must not alias inputs or each other"
		)));
	}
	let time = u32::try_from(reward.shape()[0])
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} time exceeds u32")))?;
	let environments = u32::try_from(reward.shape()[1])
		.map_err(|_| Error::invalid_argument(format!("{OPERATION} environment count exceeds u32")))?;
	let buffers = [
		BufferBinding::read(reward.storage()),
		BufferBinding::read(value.storage()),
		BufferBinding::read(next_value.storage()),
		BufferBinding::read(terminated.storage()),
		BufferBinding::read(truncated.storage()),
		BufferBinding::write(advantage.storage()),
		BufferBinding::write(returns.storage()),
	];
	let push_constants = [
		PushConstant::U32(time),
		PushConstant::U32(environments),
		PushConstant::F32(config.gamma),
		PushConstant::F32(config.lambda),
	];
	let attributes = [
		OpAttribute::Float {
			name: "gamma".to_owned(),
			value: f64::from(config.gamma),
		},
		OpAttribute::Float {
			name: "lambda".to_owned(),
			value: f64::from(config.lambda),
		},
	];
	let kernel = KernelId::MlGaeF32;
	{
		let inputs: &[&Matrix] = &inputs;
		let outputs: &[&Matrix] = &[advantage, returns];
		let attributes: &[OpAttribute] = &attributes;
		let dispatch = ComputeDispatch {
			kernel,
			buffers: &buffers,
			push_constants: &push_constants,
			workgroups: kernel.linear_workgroups(environments),
		};
		let contract = dispatch.kernel.semantic_contract().ok_or_else(|| {
			Error::internal(format!(
				"lowering-only kernel {} cannot own one semantic ML operation",
				dispatch.kernel.report_name()
			))
		})?;
		let engine = inputs
			.first()
			.ok_or_else(|| Error::internal("semantic ML dispatch requires an input"))?
			.engine_handle();
		engine.record_semantic(
			dispatch,
			SemanticDispatch {
				contract,
				inputs,
				outputs,
				attributes,
			},
		)
	}
}

fn validate_inputs(
	reward: &Matrix,
	value: &Matrix,
	next_value: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	config: GaeConfig,
) -> Result<()> {
	const OPERATION: &str = "oa::ml::advantage::gae";
	for input in [reward, value, next_value, terminated, truncated] {
		if !reward.engine_handle().same_as(input.engine_handle()) {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} inputs must belong to the same engine"
			)));
		}
		if input.shape() != reward.shape() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} inputs must have one matching shape"
			)));
		}
	}
	if reward.shape().len() != 2 || reward.shape().contains(&0) {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a nonempty [time, environments] rollout"
		)));
	}
	if reward.dtype() != DType::F32
		|| value.dtype() != DType::F32
		|| next_value.dtype() != DType::F32
		|| terminated.dtype() != DType::U8
		|| truncated.dtype() != DType::U8
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires F32 reward/value inputs and U8 boundary masks"
		)));
	}
	if !config.gamma.is_finite()
		|| !(0.0..=1.0).contains(&config.gamma)
		|| !config.lambda.is_finite()
		|| !(0.0..=1.0).contains(&config.lambda)
	{
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires finite gamma and lambda in [0, 1]"
		)));
	}
	Ok(())
}
