//! Private lowering for circular replay append and deterministic sampling.

use crate::ml::replay::{ReplayBatch, ReplayConfig, ReplayTransition};
use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{DType, Error, Matrix, OpAttribute, Result};

use super::common::{record_semantic, shader_u32};

pub(in crate::ml) fn append(
	transition: &ReplayTransition,
	storage: &ReplayBatch,
	config: &ReplayConfig,
	cursor: usize,
	observation_elements: u32,
	action_elements: u32,
) -> Result<usize> {
	const OPERATION: &str = "oa::ml::replay::append_batch";
	if transition.reward().dtype() != DType::F32 || transition.reward().shape().len() != 1 {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} requires F32 reward [batch]"
		)));
	}
	let batch = transition.reward().shape()[0];
	if batch == 0 || batch > config.capacity {
		return Err(Error::invalid_argument(format!(
			"{OPERATION} batch must be in 1..=capacity"
		)));
	}
	let observation_shape = prefixed_shape(batch, &config.observation_shape);
	let action_shape = prefixed_shape(batch, &config.action_shape);
	let vector_shape = [batch];
	let fields = [
		(
			transition.observation(),
			DType::F32,
			observation_shape.as_slice(),
		),
		(
			transition.action(),
			config.action_dtype,
			action_shape.as_slice(),
		),
		(
			transition.next_observation(),
			DType::F32,
			observation_shape.as_slice(),
		),
		(transition.reward(), DType::F32, vector_shape.as_slice()),
		(transition.terminated(), DType::U8, vector_shape.as_slice()),
		(transition.truncated(), DType::U8, vector_shape.as_slice()),
	];
	for (matrix, dtype, shape) in &fields {
		if matrix.dtype() != *dtype || matrix.shape() != *shape {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition does not match the configured replay schema"
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
	let destinations = storage_fields(storage);
	for destination in destinations {
		if !transition
			.observation()
			.engine_handle()
			.same_as(destination.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition and replay buffer must belong to one engine"
			)));
		}
		if fields
			.iter()
			.any(|(source, _, _)| source.storage().same_as(destination.storage()))
		{
			return Err(Error::invalid_argument(format!(
				"{OPERATION} transition fields must not alias replay storage"
			)));
		}
	}
	let cursor = shader_u32(cursor, "cursor", OPERATION)?;
	let capacity = shader_u32(config.capacity, "capacity", OPERATION)?;
	let batch_u32 = shader_u32(batch, "batch", OPERATION)?;
	let observation_count = batch_u32.checked_mul(observation_elements).ok_or_else(|| {
		Error::resource_exhausted(format!("{OPERATION} observation count exceeds u32"))
	})?;
	let action_count = batch_u32.checked_mul(action_elements).ok_or_else(|| {
		Error::resource_exhausted(format!("{OPERATION} action count exceeds u32"))
	})?;
	let inputs = [
		transition.observation(),
		transition.action(),
		transition.next_observation(),
		transition.reward(),
		transition.terminated(),
		transition.truncated(),
		storage.observation(),
		storage.action(),
		storage.next_observation(),
		storage.reward(),
		storage.terminated(),
		storage.truncated(),
	];
	let outputs = storage_fields(storage);
	let buffers = [
		BufferBinding::read(transition.observation().storage()),
		BufferBinding::read(transition.action().storage()),
		BufferBinding::read(transition.next_observation().storage()),
		BufferBinding::read(transition.reward().storage()),
		BufferBinding::read(transition.terminated().storage()),
		BufferBinding::read(transition.truncated().storage()),
		BufferBinding::write(storage.observation().storage()),
		BufferBinding::write(storage.action().storage()),
		BufferBinding::write(storage.next_observation().storage()),
		BufferBinding::write(storage.reward().storage()),
		BufferBinding::write(storage.terminated().storage()),
		BufferBinding::write(storage.truncated().storage()),
	];
	let push_constants = [
		PushConstant::U32(cursor),
		PushConstant::U32(capacity),
		PushConstant::U32(batch_u32),
		PushConstant::U32(observation_elements),
		PushConstant::U32(action_elements),
	];
	let attributes = unsigned_attributes(&[
		("cursor", cursor),
		("capacity", capacity),
		("batch", batch_u32),
		("observation_elements", observation_elements),
		("action_elements", action_elements),
	]);
	let kernel = KernelId::MlReplayAppendBatchF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(observation_count.max(action_count).max(batch_u32)),
	)?;
	Ok(batch)
}

#[allow(
	clippy::too_many_arguments,
	reason = "replay sampling keeps capacity, sample geometry, seed, and field widths explicit"
)]
pub(in crate::ml) fn sample(
	storage: &ReplayBatch,
	config: &ReplayConfig,
	size: usize,
	batch: usize,
	seed: u64,
	observation_elements: u32,
	action_elements: u32,
) -> Result<ReplayBatch> {
	const OPERATION: &str = "oa::ml::replay::sample";
	let size = shader_u32(size, "size", OPERATION)?;
	let batch_u32 = shader_u32(batch, "batch", OPERATION)?;
	let observation_count = batch
		.checked_mul(observation_elements as usize)
		.ok_or_else(|| {
			Error::resource_exhausted("replay sample observation size overflows usize")
		})?;
	let action_count = batch
		.checked_mul(action_elements as usize)
		.ok_or_else(|| Error::resource_exhausted("replay sample action size overflows usize"))?;
	let observation_count_u32 = shader_u32(observation_count, "observation count", OPERATION)?;
	let action_count_u32 = shader_u32(action_count, "action count", OPERATION)?;
	let mut observation_shape = vec![batch];
	observation_shape.extend_from_slice(&config.observation_shape);
	let mut action_shape = vec![batch];
	action_shape.extend_from_slice(&config.action_shape);
	let engine = storage.observation().engine_handle();
	let result = ReplayBatch {
		observation: Matrix::allocate(
			engine,
			observation_shape.clone(),
			observation_count,
			DType::F32,
		)?,
		action: Matrix::allocate(engine, action_shape, action_count, config.action_dtype)?,
		next_observation: Matrix::allocate(
			engine,
			observation_shape,
			observation_count,
			DType::F32,
		)?,
		reward: Matrix::allocate(engine, vec![batch], batch, DType::F32)?,
		terminated: Matrix::allocate(engine, vec![batch], batch, DType::U8)?,
		truncated: Matrix::allocate(engine, vec![batch], batch, DType::U8)?,
		index: Matrix::allocate(engine, vec![batch], batch, DType::U32)?,
	};
	let inputs = storage_fields(storage);
	let outputs = [
		result.observation(),
		result.action(),
		result.next_observation(),
		result.reward(),
		result.terminated(),
		result.truncated(),
		result.index(),
	];
	let buffers = [
		BufferBinding::read(storage.observation().storage()),
		BufferBinding::read(storage.action().storage()),
		BufferBinding::read(storage.next_observation().storage()),
		BufferBinding::read(storage.reward().storage()),
		BufferBinding::read(storage.terminated().storage()),
		BufferBinding::read(storage.truncated().storage()),
		BufferBinding::write(result.observation().storage()),
		BufferBinding::write(result.action().storage()),
		BufferBinding::write(result.next_observation().storage()),
		BufferBinding::write(result.reward().storage()),
		BufferBinding::write(result.terminated().storage()),
		BufferBinding::write(result.truncated().storage()),
		BufferBinding::write(result.index().storage()),
	];
	let push_constants = [
		PushConstant::U32(size),
		PushConstant::U32(batch_u32),
		PushConstant::U32(observation_elements),
		PushConstant::U32(action_elements),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "size".into(),
			value: u64::from(size),
		},
		OpAttribute::UnsignedInteger {
			name: "batch".into(),
			value: u64::from(batch_u32),
		},
		OpAttribute::UnsignedInteger {
			name: "observation_elements".into(),
			value: u64::from(observation_elements),
		},
		OpAttribute::UnsignedInteger {
			name: "action_elements".into(),
			value: u64::from(action_elements),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
	];
	let kernel = KernelId::MlReplaySampleF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(observation_count_u32.max(action_count_u32).max(batch_u32)),
	)?;
	Ok(result)
}

fn storage_fields(batch: &ReplayBatch) -> [&Matrix; 6] {
	[
		batch.observation(),
		batch.action(),
		batch.next_observation(),
		batch.reward(),
		batch.terminated(),
		batch.truncated(),
	]
}

fn prefixed_shape(first: usize, suffix: &[usize]) -> Vec<usize> {
	let mut shape = Vec::with_capacity(suffix.len() + 1);
	shape.push(first);
	shape.extend_from_slice(suffix);
	shape
}

fn unsigned_attributes(values: &[(&'static str, u32)]) -> Vec<OpAttribute> {
	values
		.iter()
		.map(|(name, value)| OpAttribute::UnsignedInteger {
			name: (*name).into(),
			value: u64::from(*value),
		})
		.collect()
}
