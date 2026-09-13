use std::path::Path;

use crate::{DType, Engine, Error, Matrix, Result};

use super::{
	CheckpointOptimizer, Module,
	model_file::{
		ModelFile, Optimizer as PersistedOptimizer, Progress, ScalarType, Tensor, TensorEncoding,
	},
	optimizer::{OptimizerCheckpoint, OptimizerRestore},
};

mod manager;

pub use manager::{CheckpointManager, CheckpointManagerConfig};

pub(super) struct CheckpointProgress {
	pub(super) step: u64,
	pub(super) metric: f64,
	pub(super) metric_name: String,
	pub(super) lower_is_better: bool,
}

/// Save a module and its complete optimizer state in the native OA `.oam` format.
///
/// Registration-derived parameter paths become the Weights index. Persistent
/// module buffers become the State index; non-persistent buffers are excluded.
/// This is an explicit host-observation boundary. The completed file is
/// durably written to a sibling temporary path and atomically renamed.
///
/// # Errors
///
/// Returns an error for ambiguous module ownership, optimizer/model mismatch,
/// unsupported Matrix metadata, readback failure, arithmetic overflow, or a
/// filesystem failure.
pub fn save_checkpoint(
	path: impl AsRef<Path>,
	model: &dyn Module,
	optimizer: &dyn CheckpointOptimizer,
) -> Result<()> {
	save_checkpoint_with_progress(path.as_ref(), model, optimizer, None)
}

pub(super) fn save_checkpoint_with_progress(
	path: &Path,
	model: &dyn Module,
	optimizer: &dyn CheckpointOptimizer,
	progress: Option<CheckpointProgress>,
) -> Result<()> {
	let named = model.all_named_parameters()?;
	let buffers = model.all_named_buffers()?;
	let state_u32 = model.all_named_state_u32()?;
	let optimizer_state = optimizer.checkpoint_state()?;
	validate_optimizer_snapshot(&optimizer_state)?;
	if named.len() != optimizer_state.parameters.len() {
		return Err(Error::invalid_argument(
			"checkpoint optimizer does not cover the complete model parameter tree",
		));
	}

	let mut file = ModelFile::new();
	for entry in named {
		let parameter = entry.parameter();
		if !optimizer_state
			.parameters
			.iter()
			.any(|candidate| candidate.same_as(&parameter))
		{
			return Err(Error::invalid_argument(
				"checkpoint optimizer parameter does not match the model tree",
			));
		}
		file.weights
			.push(matrix_tensor(entry.path(), &parameter.data())?);
	}
	for buffer in buffers.into_iter().filter(|buffer| buffer.persistent()) {
		file.state
			.push(matrix_tensor(buffer.path(), &buffer.data())?);
	}
	for state in state_u32 {
		file.state.push(Tensor::dense(
			state.path(),
			DType::U32,
			Vec::new(),
			state.value().to_le_bytes().to_vec(),
		)?);
	}

	let mut first_moment = Vec::new();
	let mut second_moment = Vec::new();
	for first in &optimizer_state.first_state {
		first_moment.extend(first.read_f32()?);
	}
	for second in &optimizer_state.second_state {
		second_moment.extend(second.read_f32()?);
	}
	let step = i64::try_from(optimizer_state.step)
		.map_err(|_| Error::resource_exhausted("optimizer step exceeds .oam progress range"))?;
	file.optimizer = Some(PersistedOptimizer {
		kind: optimizer_state.kind.to_owned(),
		learning_rate: optimizer_state.learning_rate,
		beta1: optimizer_state.beta1,
		beta2: optimizer_state.beta2,
		epsilon: optimizer_state.epsilon,
		weight_decay: optimizer_state.weight_decay,
		step,
		first_moment,
		second_moment,
	});
	file.progress = match progress {
		Some(progress) => {
			if progress.step != optimizer_state.step || !progress.metric.is_finite() {
				return Err(Error::invalid_argument(
					"checkpoint progress does not match completed optimizer state",
				));
			}
			Progress {
				step,
				learning_rate: optimizer_state.learning_rate,
				best_metric: progress.metric as f32,
				lower_is_better: progress.lower_is_better,
				metric_name: progress.metric_name,
				..Progress::default()
			}
		}
		None => Progress {
			step,
			learning_rate: optimizer_state.learning_rate,
			..Progress::default()
		},
	};
	file.save(path)
}

/// Load a native OA `.oam` checkpoint into existing matching owners.
///
/// The complete file, section hashes, tensor metadata, module paths, shapes,
/// dtypes, and optimizer payload are validated before any live handle changes.
/// `engine` remains the sole owner of newly uploaded storage.
///
/// # Errors
///
/// Returns [`crate::ErrorKind::CheckpointCorrupt`] for malformed, truncated, or
/// integrity-invalid `.oam` data. Returns another error for a valid file that
/// does not match the destination model/optimizer, or for allocation/runtime
/// and filesystem failures.
pub fn load_checkpoint(
	engine: &Engine,
	path: impl AsRef<Path>,
	model: &dyn Module,
	optimizer: &mut dyn CheckpointOptimizer,
) -> Result<()> {
	load_checkpoint_impl(engine, path.as_ref(), model, optimizer, None)
}

pub(super) fn load_checkpoint_at_step(
	engine: &Engine,
	path: &Path,
	model: &dyn Module,
	optimizer: &mut dyn CheckpointOptimizer,
	expected_step: u64,
) -> Result<()> {
	load_checkpoint_impl(engine, path, model, optimizer, Some(expected_step))
}

fn load_checkpoint_impl(
	engine: &Engine,
	path: &Path,
	model: &dyn Module,
	optimizer: &mut dyn CheckpointOptimizer,
	expected_step: Option<u64>,
) -> Result<()> {
	let file = ModelFile::load(path)?;
	if let Some(expected_step) = expected_step {
		let progress_step = u64::try_from(file.progress.step)
			.map_err(|_| Error::invalid_argument(".oam progress step is negative"))?;
		if progress_step != expected_step {
			return Err(Error::invalid_argument(
				"checkpoint filename step does not match .oam progress",
			));
		}
	}
	let named = model.all_named_parameters()?;
	let buffers = model
		.all_named_buffers()?
		.into_iter()
		.filter(|buffer| buffer.persistent())
		.collect::<Vec<_>>();
	let state_u32 = model.all_named_state_u32()?;
	let optimizer_state = optimizer.checkpoint_state()?;
	validate_optimizer_snapshot(&optimizer_state)?;
	let expected_state_count = buffers
		.len()
		.checked_add(state_u32.len())
		.ok_or_else(|| Error::resource_exhausted("checkpoint state count overflows usize"))?;
	if file.weights.len() != named.len()
		|| file.state.len() != expected_state_count
		|| named.len() != optimizer_state.parameters.len()
	{
		return Err(Error::invalid_argument(
			".oam tensor count does not match the destination module",
		));
	}

	let engine_handle = engine.handle();
	if named.iter().any(|entry| {
		!entry
			.parameter()
			.data()
			.engine_handle()
			.same_as(&engine_handle)
	}) || buffers
		.iter()
		.any(|entry| !entry.data().engine_handle().same_as(&engine_handle))
	{
		return Err(Error::invalid_argument(
			"checkpoint engine does not own the destination module",
		));
	}

	let mut loaded_parameters = Vec::with_capacity(named.len());
	for (expected, tensor) in named.iter().zip(&file.weights) {
		let parameter = expected.parameter();
		let data = parameter.data();
		validate_tensor_contract(tensor, expected.path(), &data)?;
		loaded_parameters.push((parameter, matrix_from_tensor(engine, tensor)?));
	}
	let mut loaded_buffers = Vec::with_capacity(buffers.len());
	for (expected, tensor) in buffers.iter().zip(file.state.iter().take(buffers.len())) {
		let data = expected.data();
		validate_tensor_contract(tensor, expected.path(), &data)?;
		loaded_buffers.push((expected, matrix_from_tensor(engine, tensor)?));
	}
	let mut loaded_state_u32 = Vec::with_capacity(state_u32.len());
	for (expected, tensor) in state_u32.iter().zip(file.state.iter().skip(buffers.len())) {
		validate_state_u32_contract(tensor, expected.path())?;
		let value =
			u32::from_le_bytes(
				tensor.data.as_slice().try_into().map_err(|_| {
					Error::invalid_argument(".oam u32 state has invalid byte count")
				})?,
			);
		loaded_state_u32.push((expected, value));
	}

	let persisted = file
		.optimizer
		.as_ref()
		.ok_or_else(|| Error::invalid_argument(".oam has no optimizer state"))?;
	if persisted.kind != optimizer_state.kind {
		return Err(Error::invalid_argument(format!(
			".oam optimizer {} cannot restore {}",
			persisted.kind, optimizer_state.kind
		)));
	}
	let step = u64::try_from(persisted.step)
		.map_err(|_| Error::invalid_argument(".oam optimizer step is negative"))?;
	let scalar_count = named.iter().try_fold(0_usize, |total, entry| {
		total
			.checked_add(entry.parameter().data().num_elements())
			.ok_or_else(|| Error::resource_exhausted("checkpoint parameter count overflow"))
	})?;
	let expected_first = if optimizer_state.first_state.is_empty() {
		0
	} else {
		scalar_count
	};
	let expected_second = if optimizer_state.second_state.is_empty() {
		0
	} else {
		scalar_count
	};
	if persisted.first_moment.len() != expected_first
		|| persisted.second_moment.len() != expected_second
	{
		return Err(Error::invalid_argument(
			".oam optimizer state does not cover the destination parameters",
		));
	}
	let first_state = matrices_from_flat(engine, &named, &persisted.first_moment)?;
	let second_state = matrices_from_flat(engine, &named, &persisted.second_moment)?;
	for (parameter, (loaded, _)) in optimizer_state.parameters.iter().zip(&loaded_parameters) {
		if !parameter.same_as(loaded) {
			return Err(Error::invalid_argument(
				"checkpoint optimizer order differs from module traversal",
			));
		}
	}
	for (parameter, _) in &loaded_parameters {
		parameter.validate_can_update()?;
	}
	let restore = OptimizerRestore {
		kind: persisted.kind.clone(),
		step,
		learning_rate: persisted.learning_rate,
		beta1: persisted.beta1,
		beta2: persisted.beta2,
		epsilon: persisted.epsilon,
		weight_decay: persisted.weight_decay,
		first_state,
		second_state,
	};
	optimizer.validate_checkpoint_state(&restore)?;

	// All fallible wire, ownership, shape, and allocation work has completed.
	for (parameter, data) in loaded_parameters {
		parameter.replace_data(data)?;
		parameter.clear_gradient();
	}
	for (buffer, data) in loaded_buffers {
		buffer.replace_data(data)?;
	}
	for (state, value) in loaded_state_u32 {
		state.replace_value(value);
	}
	optimizer.restore_checkpoint_state(restore)
}

fn validate_optimizer_snapshot(state: &OptimizerCheckpoint) -> Result<()> {
	let parameter_count = state.parameters.len();
	if parameter_count == 0
		|| (!state.first_state.is_empty() && state.first_state.len() != parameter_count)
		|| (!state.second_state.is_empty() && state.second_state.len() != parameter_count)
	{
		return Err(Error::invalid_argument(
			"optimizer persistence state does not match its parameter set",
		));
	}
	Ok(())
}

fn matrices_from_flat(
	engine: &Engine,
	parameters: &[super::NamedParameter],
	values: &[f32],
) -> Result<Vec<Matrix>> {
	if values.is_empty() {
		return Ok(Vec::new());
	}
	let mut matrices = Vec::with_capacity(parameters.len());
	let mut offset = 0_usize;
	for entry in parameters {
		let data = entry.parameter().data();
		let end = offset
			.checked_add(data.num_elements())
			.ok_or_else(|| Error::resource_exhausted("optimizer state offset overflows usize"))?;
		matrices.push(Matrix::from_f32(
			engine,
			data.shape().to_vec(),
			&values[offset..end],
		)?);
		offset = end;
	}
	Ok(matrices)
}

fn matrix_tensor(name: &str, matrix: &Matrix) -> Result<Tensor> {
	let shape = matrix
		.shape()
		.iter()
		.map(|extent| {
			u64::try_from(*extent)
				.map_err(|_| Error::resource_exhausted("checkpoint extent exceeds u64"))
		})
		.collect::<Result<Vec<_>>>()?;
	let data = match matrix.dtype() {
		DType::U8 => matrix.read::<u8>()?,
		DType::F32 => matrix
			.read::<f32>()?
			.into_iter()
			.flat_map(f32::to_le_bytes)
			.collect(),
		DType::I32 => matrix
			.read::<i32>()?
			.into_iter()
			.flat_map(i32::to_le_bytes)
			.collect(),
		DType::U32 => matrix
			.read::<u32>()?
			.into_iter()
			.flat_map(u32::to_le_bytes)
			.collect(),
	};
	Tensor::dense(name, matrix.dtype(), shape, data)
}

fn validate_tensor_contract(tensor: &Tensor, path: &str, matrix: &Matrix) -> Result<()> {
	let dtype = tensor_dtype(tensor)?;
	let shape = tensor_shape(tensor)?;
	if tensor.name != path || dtype != matrix.dtype() || shape != matrix.shape() {
		return Err(Error::invalid_argument(format!(
			".oam tensor {} does not match destination {path}",
			tensor.name
		)));
	}
	if tensor.encoding != TensorEncoding::Dense || tensor.block_size != 0 {
		return Err(Error::invalid_argument(format!(
			"checkpoint restore requires dense tensor {path}"
		)));
	}
	Ok(())
}

fn validate_state_u32_contract(tensor: &Tensor, path: &str) -> Result<()> {
	if tensor.name != path
		|| tensor.dtype != ScalarType::U32
		|| !tensor.shape.is_empty()
		|| tensor.encoding != TensorEncoding::Dense
		|| tensor.block_size != 0
		|| tensor.data.len() != size_of::<u32>()
	{
		return Err(Error::invalid_argument(format!(
			".oam scalar state {} does not match destination {path}",
			tensor.name
		)));
	}
	Ok(())
}

fn matrix_from_tensor(engine: &Engine, tensor: &Tensor) -> Result<Matrix> {
	let shape = tensor_shape(tensor)?;
	match tensor_dtype(tensor)? {
		DType::U8 => Matrix::from_slice(engine, shape, &tensor.data),
		DType::F32 => Matrix::from_slice(engine, shape, &decode_f32(&tensor.data)?),
		DType::I32 => Matrix::from_slice(engine, shape, &decode_i32(&tensor.data)?),
		DType::U32 => Matrix::from_slice(engine, shape, &decode_u32(&tensor.data)?),
	}
}

fn tensor_dtype(tensor: &Tensor) -> Result<DType> {
	match tensor.dtype {
		ScalarType::U8 => Ok(DType::U8),
		ScalarType::F32 => Ok(DType::F32),
		ScalarType::I32 => Ok(DType::I32),
		ScalarType::U32 => Ok(DType::U32),
		_ => Err(Error::invalid_argument(format!(
			"checkpoint restore does not support persisted dtype for {}",
			tensor.name
		))),
	}
}

fn tensor_shape(tensor: &Tensor) -> Result<Vec<usize>> {
	tensor
		.shape
		.iter()
		.map(|extent| {
			usize::try_from(*extent)
				.map_err(|_| Error::invalid_argument(".oam extent exceeds usize"))
		})
		.collect()
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
	decode_words(bytes, f32::from_le_bytes)
}

fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>> {
	decode_words(bytes, i32::from_le_bytes)
}

fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>> {
	decode_words(bytes, u32::from_le_bytes)
}

fn decode_words<T>(bytes: &[u8], decode: impl Fn([u8; 4]) -> T) -> Result<Vec<T>> {
	let (chunks, remainder) = bytes.as_chunks::<4>();
	let values = chunks.iter().copied().map(decode).collect::<Vec<_>>();
	if !remainder.is_empty() {
		return Err(Error::checkpoint_corrupt(
			"dense tensor byte count is not word aligned",
		));
	}
	Ok(values)
}
