//! Donor satellite NLP named-work workload: the frozen canonical byte
//! Transformer batch step and split-batch gradient over retained session
//! objects.
//!
//! Ported from the donor-private application
//! `sdk/cpp/apps/net/satellite/nlpStep.cpp`. The wire arguments and result
//! encodings are byte-compatible with the donor: every workload object is a
//! schema-bounded session object, the result object is a `U8` blob carrying the
//! encoded step or gradient record, and every failure maps to the donor status
//! categories through the session's [`Error`] boundary.
//!
//! The frozen recipe is the canonical OA NLP comparison workload: corpus,
//! context length 16, batch size 64, 300 steps, seed `20260714`, and the
//! donor Byte Transformer parameter layout of 25760 `F32` values.

use crate::ml::{AdamW, GradientTape, Module, loss};
use crate::net::satellite::SatelliteProtocol;
use crate::net::satellite::session::{
	SatelliteClientSession, SatelliteNamedRequest, SatelliteNamedResult,
};
use crate::runtime::Event;
use crate::sdk::ml::nlp::{
	BATCH_SIZE, ByteTransformer, CONTEXT_LENGTH, CORPUS, RNG_SEED, TRAINING_STEPS,
};
use crate::{DType, Engine, Error, Matrix, Result};

/// Named-work operation that executes the frozen full-batch optimizer step.
pub(crate) const NLP_STEP_OPERATION: &str = "nlp-byte-transformer-step-v1";
/// Named-work operation that evaluates the frozen microbatch gradient.
pub(crate) const NLP_GRADIENT_OPERATION: &str = "nlp-byte-transformer-gradient-v1";
/// Flat `F32` parameter count of the canonical donor Byte Transformer.
pub(crate) const NLP_STEP_PARAMETER_COUNT: usize = 25_760;
/// Logit values sampled into the step result for corruption detection.
pub(crate) const NLP_STEP_LOGIT_SAMPLE_COUNT: usize = 64;

const ARGUMENT_MAGIC: u32 = 0x3150_4c4e; // "NLP1", little endian.
const ARGUMENT_VERSION: u32 = 1;
const GRADIENT_ARGUMENT_MAGIC: u32 = 0x3147_4c4e; // "NLG1", little endian.
const GRADIENT_ARGUMENT_VERSION: u32 = 1;
const RESULT_MAGIC: u32 = 0x3152_534e; // "NSR1", little endian.
const RESULT_VERSION: u32 = 1;
const GRADIENT_RESULT_MAGIC: u32 = 0x3152_474e; // "NGR1", little endian.
const GRADIENT_RESULT_VERSION: u32 = 1;
const PREDICTED_POSITIONS: u32 = (BATCH_SIZE * CONTEXT_LENGTH) as u32;
const VOCAB_SIZE: u32 = 256;
/// Donor `NlpSuiteRecipe::learningRate()` for the Transformer architecture.
const NLP_LEARNING_RATE: f32 = 0.01;

/// Frozen recipe batch: deterministic canonical sample indices and the
/// corpus windows they select for one training step.
pub(crate) struct SatelliteNlpBatch {
	pub step_index: u64,
	pub sample_indices: Vec<u32>,
	pub inputs: Vec<u32>,
	pub targets: Vec<u32>,
}

/// Decoded `nlp-byte-transformer-step-v1` result record.
pub(crate) struct SatelliteNlpStepResult {
	pub seed: u64,
	pub step_index: u64,
	pub predicted_positions: u32,
	pub vocab_size: u32,
	pub loss: f32,
	pub logit_min: f32,
	pub logit_max: f32,
	pub logit_mean: f32,
	pub logit_l2: f32,
	pub initial_parameter_hash: [u8; 32],
	pub input_hash: [u8; 32],
	pub target_hash: [u8; 32],
	pub parameter_layout_hash: [u8; 32],
	pub updated_parameter_hash: [u8; 32],
	pub logit_sample: Vec<f32>,
	pub gradients: Vec<f32>,
	pub updated_parameters: Vec<f32>,
}

/// Decoded `nlp-byte-transformer-gradient-v1` result record.
pub(crate) struct SatelliteNlpGradientResult {
	pub seed: u64,
	pub step_index: u64,
	pub batch_offset: u32,
	pub batch_count: u32,
	pub predicted_positions: u32,
	pub loss: f32,
	pub initial_parameter_hash: [u8; 32],
	pub input_hash: [u8; 32],
	pub target_hash: [u8; 32],
	pub parameter_layout_hash: [u8; 32],
	pub gradients: Vec<f32>,
}

struct StepArguments {
	seed: u64,
	step_index: u64,
	batch_offset: u32,
	sample_indices: Vec<u32>,
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
	out.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(out: &mut Vec<u8>, value: u64) {
	out.extend_from_slice(&value.to_le_bytes());
}

fn append_f32(out: &mut Vec<u8>, value: f32) {
	append_u32(out, value.to_bits());
}

fn append_hash(out: &mut Vec<u8>, hash: &[u8; 32]) {
	out.extend_from_slice(hash);
}

fn read_u32(data: &[u8]) -> u32 {
	u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn read_u64(data: &[u8]) -> u64 {
	u64::from_le_bytes([
		data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
	])
}

fn read_f32(data: &[u8]) -> f32 {
	f32::from_bits(read_u32(data))
}

fn decode_step_arguments(bytes: &[u8]) -> Result<StepArguments> {
	const HEADER_BYTES: usize = 28;
	if bytes.len() < HEADER_BYTES {
		return Err(Error::data_loss(
			"satellite NLP step: argument header is truncated",
		));
	}
	if read_u32(&bytes[0..]) != ARGUMENT_MAGIC || read_u32(&bytes[4..]) != ARGUMENT_VERSION {
		return Err(Error::failed_precondition(
			"satellite NLP step: argument version is unsupported",
		));
	}
	let seed = read_u64(&bytes[8..]);
	let step_index = read_u64(&bytes[16..]);
	let count = read_u32(&bytes[24..]);
	if count != BATCH_SIZE as u32 || bytes.len() != HEADER_BYTES + count as usize * 4 {
		return Err(Error::data_loss(
			"satellite NLP step: sample-index payload is invalid",
		));
	}
	let mut sample_indices = Vec::with_capacity(count as usize);
	for index in 0..count as usize {
		sample_indices.push(read_u32(&bytes[HEADER_BYTES + index * 4..]));
	}
	Ok(StepArguments {
		seed,
		step_index,
		batch_offset: 0,
		sample_indices,
	})
}

fn decode_gradient_arguments(bytes: &[u8]) -> Result<StepArguments> {
	const HEADER_BYTES: usize = 32;
	if bytes.len() < HEADER_BYTES {
		return Err(Error::data_loss(
			"satellite NLP gradient: argument header is truncated",
		));
	}
	if read_u32(&bytes[0..]) != GRADIENT_ARGUMENT_MAGIC
		|| read_u32(&bytes[4..]) != GRADIENT_ARGUMENT_VERSION
	{
		return Err(Error::failed_precondition(
			"satellite NLP gradient: argument version is unsupported",
		));
	}
	let seed = read_u64(&bytes[8..]);
	let step_index = read_u64(&bytes[16..]);
	let batch_offset = read_u32(&bytes[24..]);
	let count = read_u32(&bytes[28..]);
	let global_batch = BATCH_SIZE as u32;
	if count == 0
		|| batch_offset >= global_batch
		|| count > global_batch - batch_offset
		|| bytes.len() != HEADER_BYTES + count as usize * 4
	{
		return Err(Error::data_loss(
			"satellite NLP gradient: microbatch payload is invalid",
		));
	}
	let mut sample_indices = Vec::with_capacity(count as usize);
	for index in 0..count as usize {
		sample_indices.push(read_u32(&bytes[HEADER_BYTES + index * 4..]));
	}
	Ok(StepArguments {
		seed,
		step_index,
		batch_offset,
		sample_indices,
	})
}

fn validate_sample_indices(arguments: &StepArguments) -> Result<()> {
	if arguments.seed != RNG_SEED || arguments.step_index >= TRAINING_STEPS as u64 {
		return Err(Error::failed_precondition(
			"satellite NLP step: seed or step index is outside the frozen recipe",
		));
	}
	let expected = satellite_build_nlp_batch(arguments.step_index)?;
	for (batch, &sample) in arguments.sample_indices.iter().enumerate() {
		let global_batch = arguments.batch_offset as usize + batch;
		if sample != expected.sample_indices[global_batch] {
			return Err(Error::failed_precondition(
				"satellite NLP step: sample indices do not match the frozen sampler",
			));
		}
	}
	Ok(())
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
	if !bytes.len().is_multiple_of(size_of::<f32>()) {
		return Err(Error::data_loss(
			"satellite NLP step: FP32 payload has a partial element",
		));
	}
	Ok(
		bytes
			.as_chunks::<4>()
			.0
			.iter()
			.map(|c| read_f32(c))
			.collect(),
	)
}

fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>> {
	if !bytes.len().is_multiple_of(size_of::<u32>()) {
		return Err(Error::data_loss(
			"satellite NLP step: UInt32 payload has a partial element",
		));
	}
	Ok(
		bytes
			.as_chunks::<4>()
			.0
			.iter()
			.map(|c| read_u32(c))
			.collect(),
	)
}

/// Donor `ScalarType` numbering for byte-identical parameter layout hashing.
fn wire_dtype_byte(dtype: DType) -> u8 {
	match dtype {
		DType::F32 => 0,
		DType::I32 => 4,
		DType::U8 => 5,
		DType::U32 => 10,
	}
}

pub(crate) fn parameter_layout_hash(named: &[crate::ml::NamedParameter]) -> [u8; 32] {
	let mut layout = Vec::new();
	for entry in named {
		let path = entry.path();
		let data = entry.parameter().data();
		append_u32(&mut layout, path.len() as u32);
		layout.extend_from_slice(path.as_bytes());
		layout.push(wire_dtype_byte(data.dtype()));
		layout.push(data.shape().len() as u8);
		for &dim in data.shape() {
			append_u64(&mut layout, dim as u64);
		}
	}
	SatelliteProtocol::stable_digest(&layout, &[])
}

pub(crate) fn load_parameters(
	engine: &Engine,
	named: &[crate::ml::NamedParameter],
	flat: &[f32],
) -> Result<()> {
	let mut offset = 0_usize;
	for entry in named {
		let parameter = entry.parameter();
		let data = parameter.data();
		if data.dtype() != DType::F32 {
			return Err(Error::dtype_mismatch(
				"satellite NLP step: canonical model parameter is not FP32",
			));
		}
		let count = data.num_elements();
		if offset > flat.len() || count > flat.len() - offset {
			return Err(Error::shape_mismatch(
				"satellite NLP step: flat parameter bundle is truncated",
			));
		}
		let loaded =
			Matrix::new_from_slice(engine, data.shape().to_vec(), &flat[offset..offset + count])
				.map_err(|_| Error::internal("satellite NLP step: parameter upload failed"))?;
		parameter.set_requires_grad(true);
		parameter.replace_data(loaded)?;
		offset += count;
	}
	if offset != flat.len() {
		return Err(Error::shape_mismatch(
			"satellite NLP step: flat parameter bundle has trailing values",
		));
	}
	Ok(())
}

fn copy_f32(matrix: &Matrix, out: &mut Vec<f32>) -> Result<()> {
	if matrix.dtype() != DType::F32 {
		return Err(Error::dtype_mismatch(
			"satellite NLP step: FP32 readback contract changed",
		));
	}
	out.extend(matrix.read_f32()?);
	Ok(())
}

fn encode_result(result: &SatelliteNlpStepResult) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(
		256
			+ (result.logit_sample.len() + result.gradients.len() + result.updated_parameters.len())
				* size_of::<f32>(),
	);
	append_u32(&mut bytes, RESULT_MAGIC);
	append_u32(&mut bytes, RESULT_VERSION);
	append_u64(&mut bytes, result.seed);
	append_u64(&mut bytes, result.step_index);
	append_u32(&mut bytes, result.predicted_positions);
	append_u32(&mut bytes, result.vocab_size);
	append_u32(&mut bytes, result.gradients.len() as u32);
	append_u32(&mut bytes, result.logit_sample.len() as u32);
	append_f32(&mut bytes, result.loss);
	append_f32(&mut bytes, result.logit_min);
	append_f32(&mut bytes, result.logit_max);
	append_f32(&mut bytes, result.logit_mean);
	append_f32(&mut bytes, result.logit_l2);
	append_hash(&mut bytes, &result.initial_parameter_hash);
	append_hash(&mut bytes, &result.input_hash);
	append_hash(&mut bytes, &result.target_hash);
	append_hash(&mut bytes, &result.parameter_layout_hash);
	append_hash(&mut bytes, &result.updated_parameter_hash);
	for &value in &result.logit_sample {
		append_f32(&mut bytes, value);
	}
	for &value in &result.gradients {
		append_f32(&mut bytes, value);
	}
	for &value in &result.updated_parameters {
		append_f32(&mut bytes, value);
	}
	bytes
}

fn encode_gradient_result(result: &SatelliteNlpGradientResult) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(192 + result.gradients.len() * size_of::<f32>());
	append_u32(&mut bytes, GRADIENT_RESULT_MAGIC);
	append_u32(&mut bytes, GRADIENT_RESULT_VERSION);
	append_u64(&mut bytes, result.seed);
	append_u64(&mut bytes, result.step_index);
	append_u32(&mut bytes, result.batch_offset);
	append_u32(&mut bytes, result.batch_count);
	append_u32(&mut bytes, result.predicted_positions);
	append_u32(&mut bytes, result.gradients.len() as u32);
	append_f32(&mut bytes, result.loss);
	append_hash(&mut bytes, &result.initial_parameter_hash);
	append_hash(&mut bytes, &result.input_hash);
	append_hash(&mut bytes, &result.target_hash);
	append_hash(&mut bytes, &result.parameter_layout_hash);
	for &value in &result.gradients {
		append_f32(&mut bytes, value);
	}
	bytes
}

fn build_profile(
	engine: &Engine,
	diagnostics: crate::runtime::ExecutionPlanDiagnostics,
	workload: &str,
	batch_offset: u32,
	batch_count: u32,
) -> Vec<u8> {
	let info = engine.device_info();
	format!(
		"workload={workload};batch_offset={batch_offset};batch_count={batch_count};\
device={};driver={};vulkan={};precision=FP32;nodes={};barriers={};graphs=1;\
submissions={};semantic_operations={};input_bindings={};\
dnn_partitions={};dnn_recognized_partitions={};dnn_applied_partitions={};\
dnn_fallback_partitions={};dnn_unexpected_fallbacks={};fallbacks={};\
kernel_selection_coverage=gemm-v1;kernel_selections={};kernel_fallbacks={}",
		info.name(),
		info.driver_name(),
		info.api_version(),
		diagnostics.node_count(),
		diagnostics.barrier_count(),
		diagnostics.submission_count(),
		diagnostics.semantic_operation_count(),
		diagnostics.input_binding_count(),
		diagnostics.dnn_partition_count(),
		diagnostics.dnn_recognized_partition_count(),
		diagnostics.dnn_applied_partition_count(),
		diagnostics.dnn_fallback_partition_count(),
		diagnostics.dnn_unexpected_fallback_count(),
		diagnostics.fallback_count(),
		diagnostics.node_count(),
		diagnostics.dnn_unexpected_fallback_count(),
	)
	.into_bytes()
}

/// LE cursor over an encoded result record; every short read maps to
/// `DataLoss`, matching the donor decode contract.
struct Cursor<'a> {
	bytes: &'a [u8],
	offset: usize,
}

impl<'a> Cursor<'a> {
	fn new(bytes: &'a [u8]) -> Self {
		Self { bytes, offset: 0 }
	}

	fn truncated() -> Error {
		Error::data_loss("satellite NLP step: result payload is truncated")
	}

	fn remaining(&self) -> usize {
		self.bytes.len() - self.offset
	}

	fn u32(&mut self) -> Result<u32> {
		if self.remaining() < 4 {
			return Err(Self::truncated());
		}
		let value = read_u32(&self.bytes[self.offset..]);
		self.offset += 4;
		Ok(value)
	}

	fn u64(&mut self) -> Result<u64> {
		if self.remaining() < 8 {
			return Err(Self::truncated());
		}
		let value = read_u64(&self.bytes[self.offset..]);
		self.offset += 8;
		Ok(value)
	}

	fn f32(&mut self) -> Result<f32> {
		if self.remaining() < 4 {
			return Err(Self::truncated());
		}
		let value = read_f32(&self.bytes[self.offset..]);
		self.offset += 4;
		Ok(value)
	}

	fn hash(&mut self, out: &mut [u8; 32]) -> Result<()> {
		if self.remaining() < out.len() {
			return Err(Self::truncated());
		}
		out.copy_from_slice(&self.bytes[self.offset..self.offset + 32]);
		self.offset += 32;
		Ok(())
	}

	fn f32_vector(&mut self, count: u32, out: &mut Vec<f32>) -> Result<()> {
		if count as usize > self.remaining() / size_of::<f32>() {
			return Err(Self::truncated());
		}
		out.reserve(count as usize);
		for _ in 0..count {
			out.push(read_f32(&self.bytes[self.offset..]));
			self.offset += 4;
		}
		Ok(())
	}
}

/// GPU handles produced inside the capture; read back after the completion
/// event completes, matching the donor recording-scope contract.
struct CapturedBatch {
	parameter_layout_hash: [u8; 32],
	loss: Matrix,
	logits: Option<Matrix>,
	gradient_matrices: Vec<Matrix>,
	updated_matrices: Vec<Matrix>,
}

struct BatchExecution {
	parameter_layout_hash: [u8; 32],
	loss: f32,
	logit_min: f32,
	logit_max: f32,
	logit_mean: f32,
	logit_l2: f32,
	logit_sample: Vec<f32>,
	gradients: Vec<f32>,
	updated_parameters: Vec<f32>,
	updated_parameter_hash: [u8; 32],
	completion: Event,
	diagnostics: crate::runtime::ExecutionPlanDiagnostics,
}

fn execute_batch(
	engine: &Engine,
	request: &SatelliteNamedRequest<'_>,
	arguments: &StepArguments,
	apply_update: bool,
	collect_logits: bool,
) -> Result<BatchExecution> {
	validate_sample_indices(arguments)?;
	let batch_count = u32::try_from(arguments.sample_indices.len())
		.map_err(|_| Error::resource_exhausted("satellite NLP workload: batch count overflows u32"))?;
	let predicted_positions = batch_count
		.checked_mul(CONTEXT_LENGTH as u32)
		.ok_or_else(|| {
			Error::resource_exhausted("satellite NLP workload: position count overflows u32")
		})?;
	let parameter_object = &request.inputs[0];
	let input_object = &request.inputs[1];
	let target_object = &request.inputs[2];
	if parameter_object.dtype != DType::F32
		|| parameter_object.shape != [NLP_STEP_PARAMETER_COUNT as i64]
		|| input_object.dtype != DType::U32
		|| target_object.dtype != DType::U32
		|| input_object.shape != [i64::from(batch_count), CONTEXT_LENGTH as i64]
		|| target_object.shape != [i64::from(batch_count), CONTEXT_LENGTH as i64]
		|| request.expected_version != parameter_object.version
		|| request.expected_hash != parameter_object.hash
	{
		return Err(Error::failed_precondition(
			"satellite NLP workload: object shape, dtype, version, or hash is invalid",
		));
	}
	let parameter_values = decode_f32(parameter_object.data)?;
	let input_values = decode_u32(input_object.data)?;
	let target_values = decode_u32(target_object.data)?;
	if parameter_values.len() != NLP_STEP_PARAMETER_COUNT
		|| input_values.len() != predicted_positions as usize
		|| target_values.len() != predicted_positions as usize
	{
		return Err(Error::shape_mismatch(
			"satellite NLP workload: decoded object sizes are invalid",
		));
	}
	let corpus = CORPUS.as_bytes();
	for (batch, &sample) in arguments.sample_indices.iter().enumerate() {
		let start = sample as usize;
		for position in 0..CONTEXT_LENGTH {
			let flat = batch * CONTEXT_LENGTH + position;
			let input_match = corpus
				.get(start + position)
				.is_some_and(|&corpus| input_values[flat] == u32::from(corpus));
			let target_match = corpus
				.get(start + position + 1)
				.is_some_and(|&corpus| target_values[flat] == u32::from(corpus));
			if !input_match || !target_match {
				return Err(Error::failed_precondition(
					"satellite NLP workload: input or target differs from the frozen batch",
				));
			}
		}
	}

	let (plan, captured) = engine.capture(|| -> Result<CapturedBatch> {
		let model = ByteTransformer::new(engine)?;
		let named = model.all_named_parameters()?;
		if model.num_parameters()? != NLP_STEP_PARAMETER_COUNT {
			return Err(Error::failed_precondition(
				"satellite NLP workload: model layout changed without a version bump",
			));
		}
		let layout_hash = parameter_layout_hash(&named);
		load_parameters(engine, &named, &parameter_values)?;
		let parameters = model.all_parameters()?;
		let mut optimizer = apply_update
			.then(|| AdamW::new(parameters.clone(), NLP_LEARNING_RATE))
			.transpose()?;
		if let Some(optimizer) = &optimizer {
			optimizer.zero_grad();
		}
		let inputs = Matrix::new_from_slice(
			engine,
			[batch_count as usize, CONTEXT_LENGTH],
			&input_values,
		)?;
		let targets = Matrix::new_from_slice(engine, [predicted_positions as usize], &target_values)?;
		let tape = GradientTape::new();
		let logits = model.forward(&inputs)?;
		let batch_loss = loss::cross_entropy(&logits, &targets)?;
		tape.backward(&batch_loss)?;
		if let Some(optimizer) = &mut optimizer {
			optimizer.step()?;
		}
		let mut gradient_matrices = Vec::with_capacity(parameters.len());
		let mut updated_matrices = Vec::with_capacity(parameters.len());
		for parameter in &parameters {
			let gradient = parameter
				.gradient()
				.ok_or_else(|| Error::data_loss("satellite NLP workload: a model gradient is missing"))?;
			gradient_matrices.push(gradient);
			if apply_update {
				updated_matrices.push(parameter.data());
			}
		}
		Ok(CapturedBatch {
			parameter_layout_hash: layout_hash,
			loss: batch_loss,
			logits: collect_logits.then_some(logits),
			gradient_matrices,
			updated_matrices,
		})
	})?;

	let event = engine.submit(&plan)?;
	event.wait()?;
	if !engine.handle().owns_event(&event) || !event.is_complete().unwrap_or(false) {
		return Err(Error::internal(
			"satellite NLP workload: exact completion event provenance failed",
		));
	}

	let mut result = BatchExecution {
		parameter_layout_hash: captured.parameter_layout_hash,
		loss: 0.0,
		logit_min: f32::INFINITY,
		logit_max: f32::NEG_INFINITY,
		logit_mean: 0.0,
		logit_l2: 0.0,
		logit_sample: Vec::new(),
		gradients: Vec::with_capacity(NLP_STEP_PARAMETER_COUNT),
		updated_parameters: Vec::with_capacity(NLP_STEP_PARAMETER_COUNT),
		updated_parameter_hash: [0; 32],
		completion: event,
		diagnostics: plan.diagnostics(),
	};
	let loss_values = captured.loss.read_f32()?;
	let Some(&loss_value) = loss_values.first() else {
		return Err(Error::data_loss(
			"satellite NLP workload: loss readback is empty",
		));
	};
	if !loss_value.is_finite() {
		return Err(Error::data_loss(
			"satellite NLP workload: loss is not finite",
		));
	}
	result.loss = loss_value;
	if let Some(logits) = &captured.logits {
		let all_logits = logits.read_f32()?;
		if all_logits.len() != predicted_positions as usize * VOCAB_SIZE as usize {
			return Err(Error::shape_mismatch(
				"satellite NLP workload: logits shape changed without a version bump",
			));
		}
		let mut sum = 0.0_f64;
		let mut sum_squares = 0.0_f64;
		for &value in &all_logits {
			if !value.is_finite() {
				return Err(Error::data_loss(
					"satellite NLP workload: logits contain a non-finite value",
				));
			}
			result.logit_min = result.logit_min.min(value);
			result.logit_max = result.logit_max.max(value);
			sum += f64::from(value);
			sum_squares += f64::from(value) * f64::from(value);
		}
		result.logit_mean = (sum / all_logits.len() as f64) as f32;
		result.logit_l2 = sum_squares.sqrt() as f32;
		result
			.logit_sample
			.extend_from_slice(&all_logits[..NLP_STEP_LOGIT_SAMPLE_COUNT]);
	}
	for matrix in &captured.gradient_matrices {
		copy_f32(matrix, &mut result.gradients)?;
	}
	for matrix in &captured.updated_matrices {
		copy_f32(matrix, &mut result.updated_parameters)?;
	}
	if result.gradients.len() != NLP_STEP_PARAMETER_COUNT
		|| (apply_update && result.updated_parameters.len() != NLP_STEP_PARAMETER_COUNT)
	{
		return Err(Error::data_loss(
			"satellite NLP workload: parameter result count is invalid",
		));
	}
	if !result.gradients.iter().all(|value| value.is_finite()) {
		return Err(Error::data_loss(
			"satellite NLP workload: gradients contain a non-finite value",
		));
	}
	if !result
		.updated_parameters
		.iter()
		.all(|value| value.is_finite())
	{
		return Err(Error::data_loss(
			"satellite NLP workload: updated parameters contain a non-finite value",
		));
	}
	if apply_update {
		result.updated_parameter_hash = satellite_hash_f32(&result.updated_parameters);
	}
	Ok(result)
}

/// Stable digest over the LE `F32` encoding used by named-work object hashes.
pub(crate) fn satellite_hash_f32(values: &[f32]) -> [u8; 32] {
	let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
	for &value in values {
		append_f32(&mut bytes, value);
	}
	SatelliteProtocol::stable_digest(&bytes, &[])
}

/// Stable digest over the LE `U32` encoding used by named-work object hashes.
pub(crate) fn satellite_hash_u32(values: &[u32]) -> [u8; 32] {
	let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
	for &value in values {
		append_u32(&mut bytes, value);
	}
	SatelliteProtocol::stable_digest(&bytes, &[])
}

/// Deterministic canonical batch for `step_index`, matching the donor's
/// `satelliteBuildNlpBatch`: cursor `step_index * BATCH_SIZE mod limit`, then
/// stride-7 corpus windows.
///
/// # Errors
///
/// Returns `OutOfRange` when `step_index` leaves the frozen recipe, and
/// `FailedPrecondition` when the canonical corpus is too short.
pub(crate) fn satellite_build_nlp_batch(step_index: u64) -> Result<SatelliteNlpBatch> {
	if step_index >= TRAINING_STEPS as u64 {
		return Err(Error::out_of_range(
			"satellite NLP batch: step index is outside the frozen recipe",
		));
	}
	let corpus = CORPUS.as_bytes();
	if corpus.len() <= CONTEXT_LENGTH + 1 {
		return Err(Error::failed_precondition(
			"satellite NLP batch: canonical corpus is too short",
		));
	}
	let limit = corpus.len() - CONTEXT_LENGTH - 1;
	let cursor = (step_index * BATCH_SIZE as u64) % limit as u64;
	let mut batch = SatelliteNlpBatch {
		step_index,
		sample_indices: Vec::with_capacity(BATCH_SIZE),
		inputs: Vec::with_capacity(BATCH_SIZE * CONTEXT_LENGTH),
		targets: Vec::with_capacity(BATCH_SIZE * CONTEXT_LENGTH),
	};
	for row in 0..BATCH_SIZE {
		let start = ((cursor + row as u64 * 7) % limit as u64) as usize;
		batch.sample_indices.push(start as u32);
		for position in 0..CONTEXT_LENGTH {
			batch.inputs.push(u32::from(corpus[start + position]));
			batch.targets.push(u32::from(corpus[start + position + 1]));
		}
	}
	Ok(batch)
}

/// Donor parameter-layout digest over a model's named parameter order,
/// dtype, rank, and extents.
pub(crate) fn satellite_nlp_parameter_layout_hash(model: &ByteTransformer) -> Result<[u8; 32]> {
	Ok(parameter_layout_hash(&model.all_named_parameters()?))
}

/// Start the frozen full-batch step named operation.
///
/// # Errors
///
/// Returns `InvalidArgument` when the sample-index count differs from the
/// frozen batch, `OutOfRange` when the result version would overflow, and the
/// session errors produced by [`SatelliteClientSession::start_named`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn satellite_start_nlp_step(
	client: &mut SatelliteClientSession,
	parameters: u64,
	tokens: u64,
	targets: u64,
	output: u64,
	seed: u64,
	step_index: u64,
	sample_indices: &[u32],
	expected_version: u64,
	expected_hash: &[u8; 32],
) -> Result<u64> {
	if sample_indices.len() != BATCH_SIZE {
		return Err(Error::invalid_argument(
			"satellite NLP step: sample-index count differs from the frozen batch",
		));
	}
	if expected_version == u64::MAX {
		return Err(Error::out_of_range(
			"satellite NLP step: result version would overflow",
		));
	}
	let mut arguments = Vec::with_capacity(28 + sample_indices.len() * 4);
	append_u32(&mut arguments, ARGUMENT_MAGIC);
	append_u32(&mut arguments, ARGUMENT_VERSION);
	append_u64(&mut arguments, seed);
	append_u64(&mut arguments, step_index);
	append_u32(&mut arguments, sample_indices.len() as u32);
	for &sample in sample_indices {
		append_u32(&mut arguments, sample);
	}
	client.start_named(
		NLP_STEP_OPERATION,
		&[parameters, tokens, targets],
		output,
		&arguments,
		expected_version,
		expected_hash,
	)
}

/// Start the frozen microbatch gradient named operation.
///
/// # Errors
///
/// Returns `InvalidArgument` when the microbatch leaves the frozen global
/// batch, and the session errors produced by
/// [`SatelliteClientSession::start_named`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn satellite_start_nlp_gradient(
	client: &mut SatelliteClientSession,
	parameters: u64,
	tokens: u64,
	targets: u64,
	output: u64,
	seed: u64,
	step_index: u64,
	batch_offset: u32,
	sample_indices: &[u32],
	expected_version: u64,
	expected_hash: &[u8; 32],
) -> Result<u64> {
	let global_batch = BATCH_SIZE as u32;
	if sample_indices.len() > global_batch as usize {
		return Err(Error::invalid_argument(
			"satellite NLP gradient: microbatch exceeds the frozen global batch",
		));
	}
	let count = sample_indices.len() as u32;
	if count == 0 || batch_offset >= global_batch || count > global_batch - batch_offset {
		return Err(Error::invalid_argument(
			"satellite NLP gradient: microbatch is outside the frozen global batch",
		));
	}
	let mut arguments = Vec::with_capacity(32 + sample_indices.len() * 4);
	append_u32(&mut arguments, GRADIENT_ARGUMENT_MAGIC);
	append_u32(&mut arguments, GRADIENT_ARGUMENT_VERSION);
	append_u64(&mut arguments, seed);
	append_u64(&mut arguments, step_index);
	append_u32(&mut arguments, batch_offset);
	append_u32(&mut arguments, count);
	for &sample in sample_indices {
		append_u32(&mut arguments, sample);
	}
	client.start_named(
		NLP_GRADIENT_OPERATION,
		&[parameters, tokens, targets],
		output,
		&arguments,
		expected_version,
		expected_hash,
	)
}

/// Execute the frozen `nlp-byte-transformer-step-v1` named operation.
///
/// Validates the three retained input objects against the frozen recipe,
/// replays the canonical byte Transformer inside one captured plan, submits,
/// waits, and returns the encoded step record as a `U8` result object.
///
/// # Errors
///
/// Returns `InvalidArgument` for a wrong operation or input count,
/// `FailedPrecondition` for a batch that diverges from the frozen sampler,
/// `ShapeMismatch`/`DtypeMismatch` for broken object contracts, and
/// `DataLoss` for non-finite or truncated outputs.
pub(crate) fn satellite_execute_nlp_step(
	engine: &Engine,
	request: &SatelliteNamedRequest<'_>,
) -> Result<SatelliteNamedResult> {
	if request.operation != NLP_STEP_OPERATION || request.inputs.len() != 3 {
		return Err(Error::invalid_argument(
			"satellite NLP step: request operation or input count is invalid",
		));
	}
	let arguments = decode_step_arguments(request.arguments)?;
	let execution = execute_batch(engine, request, &arguments, true, true)?;
	let result = SatelliteNlpStepResult {
		seed: arguments.seed,
		step_index: arguments.step_index,
		predicted_positions: PREDICTED_POSITIONS,
		vocab_size: VOCAB_SIZE,
		loss: execution.loss,
		logit_min: execution.logit_min,
		logit_max: execution.logit_max,
		logit_mean: execution.logit_mean,
		logit_l2: execution.logit_l2,
		initial_parameter_hash: request.inputs[0].hash,
		input_hash: request.inputs[1].hash,
		target_hash: request.inputs[2].hash,
		parameter_layout_hash: execution.parameter_layout_hash,
		updated_parameter_hash: execution.updated_parameter_hash,
		logit_sample: execution.logit_sample,
		gradients: execution.gradients,
		updated_parameters: execution.updated_parameters,
	};
	let data = encode_result(&result);
	Ok(SatelliteNamedResult {
		version: request
			.expected_version
			.checked_add(1)
			.ok_or_else(|| Error::out_of_range("satellite NLP step: result version would overflow"))?,
		dtype: DType::U8,
		shape: vec![data.len() as i64],
		data,
		profile: build_profile(
			engine,
			execution.diagnostics,
			NLP_STEP_OPERATION,
			0,
			BATCH_SIZE as u32,
		),
		completion: execution.completion,
	})
}

/// Execute the frozen `nlp-byte-transformer-gradient-v1` named operation.
///
/// Same contract as [`satellite_execute_nlp_step`] but without the optimizer
/// update or the logit sample; the result object version echoes
/// `expected_version` unchanged, matching the donor.
///
/// # Errors
///
/// Returns the same categories as [`satellite_execute_nlp_step`].
pub(crate) fn satellite_execute_nlp_gradient(
	engine: &Engine,
	request: &SatelliteNamedRequest<'_>,
) -> Result<SatelliteNamedResult> {
	if request.operation != NLP_GRADIENT_OPERATION || request.inputs.len() != 3 {
		return Err(Error::invalid_argument(
			"satellite NLP gradient: request operation or input count is invalid",
		));
	}
	let arguments = decode_gradient_arguments(request.arguments)?;
	let execution = execute_batch(engine, request, &arguments, false, false)?;
	let batch_count = u32::try_from(arguments.sample_indices.len())
		.map_err(|_| Error::resource_exhausted("satellite NLP gradient: batch count overflows u32"))?;
	let result = SatelliteNlpGradientResult {
		seed: arguments.seed,
		step_index: arguments.step_index,
		batch_offset: arguments.batch_offset,
		batch_count,
		predicted_positions: batch_count * CONTEXT_LENGTH as u32,
		loss: execution.loss,
		initial_parameter_hash: request.inputs[0].hash,
		input_hash: request.inputs[1].hash,
		target_hash: request.inputs[2].hash,
		parameter_layout_hash: execution.parameter_layout_hash,
		gradients: execution.gradients,
	};
	let data = encode_gradient_result(&result);
	Ok(SatelliteNamedResult {
		version: request.expected_version,
		dtype: DType::U8,
		shape: vec![data.len() as i64],
		data,
		profile: build_profile(
			engine,
			execution.diagnostics,
			NLP_GRADIENT_OPERATION,
			arguments.batch_offset,
			batch_count,
		),
		completion: execution.completion,
	})
}

/// Route an admitted named-work request to the step or gradient executor.
///
/// # Errors
///
/// Returns `InvalidArgument` when the operation is not admitted.
pub(crate) fn satellite_execute_nlp_work(
	engine: &Engine,
	request: &SatelliteNamedRequest<'_>,
) -> Result<SatelliteNamedResult> {
	if request.operation == NLP_STEP_OPERATION {
		return satellite_execute_nlp_step(engine, request);
	}
	if request.operation == NLP_GRADIENT_OPERATION {
		return satellite_execute_nlp_gradient(engine, request);
	}
	Err(Error::invalid_argument(
		"satellite NLP worker: operation is not admitted",
	))
}

/// Decode an `NSR1` step result record.
///
/// # Errors
///
/// Returns `DataLoss` for truncated or trailing bytes and
/// `FailedPrecondition` for a version or shape contract change.
pub(crate) fn satellite_decode_nlp_step_result(bytes: &[u8]) -> Result<SatelliteNlpStepResult> {
	let mut cursor = Cursor::new(bytes);
	let magic = cursor.u32();
	let version = cursor.u32();
	let (Ok(magic), Ok(version)) = (magic, version) else {
		return Err(Error::data_loss(
			"satellite NLP step: result header is truncated",
		));
	};
	if magic != RESULT_MAGIC || version != RESULT_VERSION {
		return Err(Error::failed_precondition(
			"satellite NLP step: result version is unsupported",
		));
	}
	let seed = cursor.u64();
	let step = cursor.u64();
	let positions = cursor.u32();
	let vocab = cursor.u32();
	let parameter_count = cursor.u32();
	let sample_count = cursor.u32();
	let loss = cursor.f32();
	let minimum = cursor.f32();
	let maximum = cursor.f32();
	let mean = cursor.f32();
	let l2 = cursor.f32();
	let (
		Ok(seed),
		Ok(step),
		Ok(positions),
		Ok(vocab),
		Ok(parameter_count),
		Ok(sample_count),
		Ok(loss),
		Ok(minimum),
		Ok(maximum),
		Ok(mean),
		Ok(l2),
	) = (
		seed,
		step,
		positions,
		vocab,
		parameter_count,
		sample_count,
		loss,
		minimum,
		maximum,
		mean,
		l2,
	)
	else {
		return Err(Error::data_loss(
			"satellite NLP step: result scalar section is truncated",
		));
	};
	if positions != PREDICTED_POSITIONS
		|| vocab != VOCAB_SIZE
		|| parameter_count as usize != NLP_STEP_PARAMETER_COUNT
		|| sample_count as usize != NLP_STEP_LOGIT_SAMPLE_COUNT
	{
		return Err(Error::failed_precondition(
			"satellite NLP step: result shape contract changed",
		));
	}
	let mut result = SatelliteNlpStepResult {
		seed,
		step_index: step,
		predicted_positions: positions,
		vocab_size: vocab,
		loss,
		logit_min: minimum,
		logit_max: maximum,
		logit_mean: mean,
		logit_l2: l2,
		initial_parameter_hash: [0; 32],
		input_hash: [0; 32],
		target_hash: [0; 32],
		parameter_layout_hash: [0; 32],
		updated_parameter_hash: [0; 32],
		logit_sample: Vec::new(),
		gradients: Vec::new(),
		updated_parameters: Vec::new(),
	};
	cursor.hash(&mut result.initial_parameter_hash)?;
	cursor.hash(&mut result.input_hash)?;
	cursor.hash(&mut result.target_hash)?;
	cursor.hash(&mut result.parameter_layout_hash)?;
	cursor.hash(&mut result.updated_parameter_hash)?;
	cursor.f32_vector(sample_count, &mut result.logit_sample)?;
	cursor.f32_vector(parameter_count, &mut result.gradients)?;
	cursor.f32_vector(parameter_count, &mut result.updated_parameters)?;
	if cursor.remaining() != 0 {
		return Err(Error::data_loss(
			"satellite NLP step: result payload has trailing bytes",
		));
	}
	Ok(result)
}

/// Decode an `NGR1` gradient result record.
///
/// # Errors
///
/// Returns `DataLoss` for truncated or trailing bytes and
/// `FailedPrecondition` for a version or shape contract change.
pub(crate) fn satellite_decode_nlp_gradient_result(
	bytes: &[u8],
) -> Result<SatelliteNlpGradientResult> {
	let mut cursor = Cursor::new(bytes);
	let magic = cursor.u32();
	let version = cursor.u32();
	let seed = cursor.u64();
	let step = cursor.u64();
	let batch_offset = cursor.u32();
	let batch_count = cursor.u32();
	let positions = cursor.u32();
	let parameter_count = cursor.u32();
	let loss = cursor.f32();
	let (
		Ok(magic),
		Ok(version),
		Ok(seed),
		Ok(step),
		Ok(batch_offset),
		Ok(batch_count),
		Ok(positions),
		Ok(parameter_count),
		Ok(loss),
	) = (
		magic,
		version,
		seed,
		step,
		batch_offset,
		batch_count,
		positions,
		parameter_count,
		loss,
	)
	else {
		return Err(Error::data_loss(
			"satellite NLP gradient: result scalar section is truncated",
		));
	};
	let global_batch = BATCH_SIZE as u32;
	if magic != GRADIENT_RESULT_MAGIC
		|| version != GRADIENT_RESULT_VERSION
		|| batch_count == 0
		|| batch_offset >= global_batch
		|| batch_count > global_batch - batch_offset
		|| positions != batch_count * CONTEXT_LENGTH as u32
		|| parameter_count as usize != NLP_STEP_PARAMETER_COUNT
	{
		return Err(Error::failed_precondition(
			"satellite NLP gradient: result shape or version contract changed",
		));
	}
	let mut result = SatelliteNlpGradientResult {
		seed,
		step_index: step,
		batch_offset,
		batch_count,
		predicted_positions: positions,
		loss,
		initial_parameter_hash: [0; 32],
		input_hash: [0; 32],
		target_hash: [0; 32],
		parameter_layout_hash: [0; 32],
		gradients: Vec::new(),
	};
	cursor.hash(&mut result.initial_parameter_hash)?;
	cursor.hash(&mut result.input_hash)?;
	cursor.hash(&mut result.target_hash)?;
	cursor.hash(&mut result.parameter_layout_hash)?;
	cursor.f32_vector(parameter_count, &mut result.gradients)?;
	if cursor.remaining() != 0 {
		return Err(Error::data_loss(
			"satellite NLP gradient: result payload has trailing bytes",
		));
	}
	Ok(result)
}

#[cfg(test)]
#[path = "../../../../test/rs/net/satellite_nlp_unit.rs"]
mod tests;
