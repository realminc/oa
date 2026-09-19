use crate::ml::advantage::{GaeConfig, GaeResult};
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::record_semantic;

pub(in crate::ml) fn gae(
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

#[allow(clippy::too_many_arguments)]
pub(in crate::ml) fn gae_into(
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
	record_semantic(
		&inputs,
		&[advantage, returns],
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(environments),
	)
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
