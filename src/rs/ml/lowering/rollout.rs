//! Private lowering for fixed-capacity rollout storage mutations.

use crate::ml::rollout::{RolloutBatch, RolloutTransition};
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32};

pub(in crate::ml) fn append(
	transition: &RolloutTransition,
	batch: &RolloutBatch,
	step: usize,
	environments: usize,
	observation_shape: &[usize],
	observation_elements: u32,
) -> Result<()> {
	const OPERATION: &str = "oa::ml::rollout::append";
	let mut expected_observation_shape = Vec::with_capacity(observation_shape.len() + 1);
	expected_observation_shape.push(environments);
	expected_observation_shape.extend_from_slice(observation_shape);
	let vector_shape = [environments];
	let fields = [
		(
			transition.observation(),
			DType::F32,
			expected_observation_shape.as_slice(),
		),
		(transition.action(), DType::I32, vector_shape.as_slice()),
		(transition.reward(), DType::F32, vector_shape.as_slice()),
		(transition.value(), DType::F32, vector_shape.as_slice()),
		(transition.next_value(), DType::F32, vector_shape.as_slice()),
		(
			transition.log_probability(),
			DType::F32,
			vector_shape.as_slice(),
		),
		(transition.terminated(), DType::U8, vector_shape.as_slice()),
		(transition.truncated(), DType::U8, vector_shape.as_slice()),
	];
	for (matrix, dtype, shape) in &fields {
		if matrix.dtype() != *dtype || matrix.shape() != *shape {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition does not match the configured categorical shape/dtype contract"
			)));
		}
		if !transition
			.observation()
			.engine_handle()
			.same_as(matrix.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must belong to one engine"
			)));
		}
	}
	let destinations = batch_destinations(batch);
	for destination in destinations {
		if !transition
			.observation()
			.engine_handle()
			.same_as(destination.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition and rollout buffer must belong to one engine"
			)));
		}
		if fields
			.iter()
			.any(|(source, _, _)| source.storage().same_as(destination.storage()))
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must not alias retained rollout storage"
			)));
		}
	}

	let step = shader_u32(step, "step", OPERATION)?;
	let environments = shader_u32(environments, "environment count", OPERATION)?;
	let observation_count = environments
		.checked_mul(observation_elements)
		.ok_or_else(|| {
			Error::resource_exhausted(format!("{OPERATION} observation count exceeds u32"))
		})?;
	let work_items = observation_count.max(environments);
	let inputs = [
		transition.observation(),
		transition.action(),
		transition.reward(),
		transition.value(),
		transition.next_value(),
		transition.log_probability(),
		transition.terminated(),
		transition.truncated(),
		batch.observation(),
		batch.action(),
		batch.reward(),
		batch.value(),
		batch.next_value(),
		batch.old_log_probability(),
		batch.terminated(),
		batch.truncated(),
		batch.valid(),
	];
	let outputs = batch_destinations(batch);
	let buffers = [
		BufferBinding::read(transition.observation().storage()),
		BufferBinding::read(transition.action().storage()),
		BufferBinding::read(transition.reward().storage()),
		BufferBinding::read(transition.value().storage()),
		BufferBinding::read(transition.next_value().storage()),
		BufferBinding::read(transition.log_probability().storage()),
		BufferBinding::read(transition.terminated().storage()),
		BufferBinding::read(transition.truncated().storage()),
		BufferBinding::write(batch.observation().storage()),
		BufferBinding::write(batch.action().storage()),
		BufferBinding::write(batch.reward().storage()),
		BufferBinding::write(batch.value().storage()),
		BufferBinding::write(batch.next_value().storage()),
		BufferBinding::write(batch.old_log_probability().storage()),
		BufferBinding::write(batch.terminated().storage()),
		BufferBinding::write(batch.truncated().storage()),
		BufferBinding::write(batch.valid().storage()),
	];
	let push_constants = [
		PushConstant::U32(step),
		PushConstant::U32(environments),
		PushConstant::U32(observation_elements),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "step".into(),
			value: u64::from(step),
		},
		OpAttribute::UnsignedInteger {
			name: "environments".into(),
			value: u64::from(environments),
		},
		OpAttribute::UnsignedInteger {
			name: "observation_elements".into(),
			value: u64::from(observation_elements),
		},
	];
	let kernel = KernelId::MlRolloutAppendF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(work_items),
	)
}

pub(in crate::ml) fn reset(valid: &Matrix) -> Result<()> {
	const OPERATION: &str = "oa::ml::rollout::reset";
	if valid.dtype() != DType::U8 || valid.shape().len() != 2 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires a rank-two U8 validity mask"
		)));
	}
	let count = shader_u32(valid.num_elements(), "element count", OPERATION)?;
	let buffers = [BufferBinding::write(valid.storage())];
	let push_constants = [PushConstant::U32(count)];
	let kernel = KernelId::MlRolloutResetU8;
	record_semantic(
		&[valid],
		&[valid],
		&[],
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(count),
	)
}

fn batch_destinations(batch: &RolloutBatch) -> [&Matrix; 9] {
	[
		batch.observation(),
		batch.action(),
		batch.reward(),
		batch.value(),
		batch.next_value(),
		batch.old_log_probability(),
		batch.terminated(),
		batch.truncated(),
		batch.valid(),
	]
}
