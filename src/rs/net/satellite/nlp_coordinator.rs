//! Donor `nlpCoordinator` port: the frozen byte-Transformer training
//! coordinators over the private satellite NLP named-work surface.
//!
//! `satellite_run_split_batch_nlp` splits each canonical batch into a local
//! microbatch and a remote microbatch, reduces the weighted gradients on the
//! coordinator, and applies one authoritative AdamW update per step.
//! `satellite_run_standalone_nlp` runs the identical frozen recipe entirely
//! locally. Both finalize through the donor checkpoint/generation quality
//! contract.

use std::path::PathBuf;
use std::time::Instant;

use super::nlp_step::{
	NLP_STEP_PARAMETER_COUNT, SatelliteNlpBatch, satellite_build_nlp_batch,
	satellite_decode_nlp_gradient_result, satellite_hash_f32, satellite_hash_u32,
	satellite_nlp_parameter_layout_hash, satellite_start_nlp_gradient,
};
use super::session::SatelliteClientSession;
use crate::ml::{AdamW, GradientTape, Module, load_checkpoint, loss, save_checkpoint};
use crate::runtime::{ExecutionPlan, ExecutionPlanDiagnostics};
use crate::sdk::ml::nlp::{
	BATCH_SIZE, ByteTransformer, CONTEXT_LENGTH, CORPUS, GENERATION_LENGTH, GENERATION_PROMPT,
	RNG_SEED, TRAINING_STEPS, generate_bytes_greedy,
};
use crate::{Engine, Error, Matrix, Result};

const PARAMETER_OBJECT: u64 = 1;
const INPUT_OBJECT: u64 = 2;
const TARGET_OBJECT: u64 = 3;
const RESULT_OBJECT: u64 = 4;
const NLP_LEARNING_RATE: f32 = 0.01;

/// Donor `SatelliteNlpCoordinatorConfig` port.
pub(crate) struct SatelliteNlpCoordinatorConfig {
	pub steps: u32,
	pub local_batch_count: u32,
	pub initial_version: u64,
	pub overlap_remote: bool,
	pub checkpoint_path: PathBuf,
}

impl Default for SatelliteNlpCoordinatorConfig {
	fn default() -> Self {
		Self {
			steps: TRAINING_STEPS as u32,
			local_batch_count: 32,
			initial_version: 1,
			overlap_remote: false,
			checkpoint_path: PathBuf::new(),
		}
	}
}

/// Donor `SatelliteNlpCoordinatorReport` port.
#[derive(Default)]
pub(crate) struct SatelliteNlpCoordinatorReport {
	pub completed_steps: u32,
	pub optimizer_step: u64,
	pub initial_loss: f32,
	pub final_loss: f32,
	pub final_accuracy: f32,
	pub initial_parameter_hash: [u8; 32],
	pub final_parameter_hash: [u8; 32],
	pub parameter_layout_hash: [u8; 32],
	pub local_kernel_selections: u64,
	pub local_kernel_fallbacks: u64,
	pub remote_kernel_selections: u64,
	pub remote_kernel_fallbacks: u64,
	pub checkpoint_round_trip: bool,
	pub generation_quality_passed: bool,
	pub generated: Vec<u8>,
	pub complete_wall_ms: f64,
}

struct LocalGradient {
	loss: f32,
	values: Vec<f32>,
	diagnostics: ExecutionPlanDiagnostics,
}

/// Submit `plan`, wait for its exact completion event, and verify the event is
/// engine-owned and complete, matching the donor `submitAndWait` contract.
fn submit_plan_and_wait(engine: &Engine, plan: &ExecutionPlan) -> Result<()> {
	let event = engine.submit(plan)?;
	event.wait()?;
	if !engine.handle().owns_event(&event) || !event.is_complete().unwrap_or(false) {
		return Err(Error::internal(
			"satellite NLP coordinator: exact local completion failed",
		));
	}
	Ok(())
}

/// Submit the pending eager batch once, matching the donor ambient
/// `engine.submit()` used for model initialization and evaluation work.
fn checkpoint_and_wait(engine: &Engine) -> Result<()> {
	let event = engine.checkpoint()?;
	event.wait()?;
	if !engine.handle().owns_event(&event) || !event.is_complete().unwrap_or(false) {
		return Err(Error::internal(
			"satellite NLP coordinator: exact local completion failed",
		));
	}
	Ok(())
}

fn append_f32_values(matrix: &Matrix, out: &mut Vec<f32>) -> Result<()> {
	if matrix.dtype() != crate::DType::F32 {
		return Err(Error::dtype_mismatch(
			"satellite NLP coordinator: expected an FP32 matrix",
		));
	}
	out.extend(matrix.read_f32()?);
	Ok(())
}

/// Flatten the model's current parameters in deterministic named order.
fn copy_parameters(model: &ByteTransformer) -> Result<Vec<f32>> {
	let mut values = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for parameter in &model.all_parameters()? {
		append_f32_values(&parameter.data(), &mut values)?;
	}
	if values.len() != NLP_STEP_PARAMETER_COUNT {
		return Err(Error::failed_precondition(
			"satellite NLP coordinator: canonical parameter count changed",
		));
	}
	Ok(values)
}

/// Local microbatch gradient for the persistent coordinator model, matching
/// the donor `executeLocalGradient`: zeroed tape, captured forward/backward,
/// exact completion, and finite checked readback.
fn execute_local_gradient(
	engine: &Engine,
	model: &ByteTransformer,
	optimizer: &AdamW,
	inputs: &[u32],
	targets: &[u32],
) -> Result<LocalGradient> {
	if inputs.is_empty()
		|| inputs.len() != targets.len()
		|| !inputs.len().is_multiple_of(CONTEXT_LENGTH)
	{
		return Err(Error::invalid_argument(
			"satellite NLP coordinator: local microbatch shape is invalid",
		));
	}
	let batch_count = inputs.len() / CONTEXT_LENGTH;
	let parameters = model.all_parameters()?;
	optimizer.zero_grad();
	let (plan, (loss_matrix, gradient_matrices)) =
		engine.capture(|| {
			let input = Matrix::new_from_slice(engine, [batch_count, CONTEXT_LENGTH], inputs)?;
			let target = Matrix::new_from_slice(engine, [inputs.len()], targets)?;
			let tape = GradientTape::new();
			let logits = model.forward(&input)?;
			let step_loss = loss::cross_entropy(&logits, &target)?;
			tape.backward(&step_loss)?;
			let mut gradients = Vec::with_capacity(parameters.len());
			for parameter in &parameters {
				gradients.push(parameter.gradient().ok_or_else(|| {
					Error::data_loss("satellite NLP coordinator: local gradient is missing")
				})?);
			}
			Ok((step_loss, gradients))
		})?;
	submit_plan_and_wait(engine, &plan)?;
	let diagnostics = plan.diagnostics();
	let loss_value = loss_matrix.read_f32()?[0];
	if !loss_value.is_finite() {
		return Err(Error::data_loss(
			"satellite NLP coordinator: local loss is not finite",
		));
	}
	let mut values = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for matrix in &gradient_matrices {
		append_f32_values(matrix, &mut values)?;
	}
	if values.len() != NLP_STEP_PARAMETER_COUNT {
		return Err(Error::data_loss(
			"satellite NLP coordinator: local gradient count is invalid",
		));
	}
	if !values.iter().all(|value| value.is_finite()) {
		return Err(Error::data_loss(
			"satellite NLP coordinator: local gradient is not finite",
		));
	}
	Ok(LocalGradient {
		loss: loss_value,
		values,
		diagnostics,
	})
}

/// Authoritative optimizer update against a supplied reduced gradient,
/// matching the donor `applyGradient`: per-parameter replacement of the
/// gradient storage, one AdamW step, exact completion.
fn apply_gradient(
	engine: &Engine,
	model: &ByteTransformer,
	optimizer: &mut AdamW,
	gradient: &[f32],
) -> Result<()> {
	let parameters = model.all_parameters()?;
	let (plan, ()) = engine.capture(|| {
		let mut offset = 0_usize;
		for parameter in &parameters {
			let data = parameter.data();
			let count = data.num_elements();
			if offset > gradient.len() || count > gradient.len() - offset {
				return Err(Error::shape_mismatch(
					"satellite NLP coordinator: reduced gradient is truncated",
				));
			}
			let matrix = Matrix::new_from_slice(
				engine,
				data.shape().to_vec(),
				&gradient[offset..offset + count],
			)?;
			parameter.clear_gradient();
			parameter.accumulate_gradient(matrix)?;
			offset += count;
		}
		if offset != gradient.len() {
			return Err(Error::shape_mismatch(
				"satellite NLP coordinator: reduced gradient has trailing values",
			));
		}
		optimizer.step()
	})?;
	submit_plan_and_wait(engine, &plan)?;
	optimizer.complete_captured_step()
}

/// Full-batch standalone step, matching the donor `executeStandaloneStep`:
/// zeroed tape, captured forward/backward/AdamW update, finite checked loss.
fn execute_standalone_step(
	engine: &Engine,
	model: &ByteTransformer,
	optimizer: &mut AdamW,
	batch: &SatelliteNlpBatch,
) -> Result<(f32, ExecutionPlanDiagnostics)> {
	optimizer.zero_grad();
	let (plan, loss_matrix) = engine.capture(|| {
		let input = Matrix::new_from_slice(engine, [BATCH_SIZE, CONTEXT_LENGTH], &batch.inputs)?;
		let target = Matrix::new_from_slice(engine, [batch.targets.len()], &batch.targets)?;
		let tape = GradientTape::new();
		let logits = model.forward(&input)?;
		let step_loss = loss::cross_entropy(&logits, &target)?;
		tape.backward(&step_loss)?;
		optimizer.step()?;
		Ok(step_loss)
	})?;
	submit_plan_and_wait(engine, &plan)?;
	optimizer.complete_captured_step()?;
	let loss_value = loss_matrix.read_f32()?[0];
	if !loss_value.is_finite() {
		return Err(Error::data_loss(
			"satellite NLP coordinator: standalone loss is not finite",
		));
	}
	Ok((loss_value, plan.diagnostics()))
}

/// The donor's `kernel_selections`/`kernel_fallbacks` gate over the plan's
/// executable-node count and unexpected-fallback counter, matching the
/// execution-graph report's selection vocabulary.
fn accumulate_local_selection(
	diagnostics: ExecutionPlanDiagnostics,
	report: &mut SatelliteNlpCoordinatorReport,
) -> Result<()> {
	let selections = diagnostics.node_count() as u64;
	let fallbacks = u64::from(diagnostics.dnn_unexpected_fallback_count());
	if selections == 0
		|| fallbacks != 0
		|| report.local_kernel_selections > u64::MAX - selections
		|| report.local_kernel_fallbacks > u64::MAX - fallbacks
	{
		return Err(Error::failed_precondition(
			"satellite NLP coordinator: local GEMM selection gate failed",
		));
	}
	report.local_kernel_selections += selections;
	report.local_kernel_fallbacks += fallbacks;
	Ok(())
}

/// Donor `profileCounter`: parse `key=value` from the remote profile record,
/// accepting `;` or end-of-input as the value terminator.
fn profile_counter(profile: &[u8], key: &str) -> Result<u64> {
	let text = std::str::from_utf8(profile)
		.map_err(|_| Error::data_loss("satellite NLP coordinator: remote profile is not UTF-8"))?;
	let position = text.find(key).ok_or_else(|| {
		Error::data_loss("satellite NLP coordinator: remote profile counter is missing")
	})?;
	let digits = &text[position + key.len()..];
	let end = digits.find(';').unwrap_or(digits.len());
	digits[..end]
		.parse::<u64>()
		.map_err(|_| Error::data_loss("satellite NLP coordinator: remote profile counter is malformed"))
}

/// Argmax accuracy of the frozen batch targets, matching the donor
/// `evaluateAccuracy` host-comparison contract.
fn evaluate_accuracy(
	engine: &Engine,
	model: &ByteTransformer,
	batch: &SatelliteNlpBatch,
) -> Result<f32> {
	let inputs = Matrix::new_from_slice(engine, [BATCH_SIZE, CONTEXT_LENGTH], &batch.inputs)?;
	let logits = model.forward(&inputs)?;
	checkpoint_and_wait(engine)?;
	let host = logits.read_f32()?;
	let rows = batch.targets.len();
	if host.len() != rows * 256 {
		return Err(Error::shape_mismatch(
			"satellite NLP coordinator: evaluation logits shape is invalid",
		));
	}
	let mut correct = 0_usize;
	for (row, &target) in batch.targets.iter().enumerate() {
		let base = row * 256;
		let mut best = 0_usize;
		for column in 1..256 {
			if host[base + column] > host[base + best] {
				best = column;
			}
		}
		if best == target as usize {
			correct += 1;
		}
	}
	Ok(correct as f32 / rows as f32)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	haystack
		.windows(needle.len())
		.position(|window| window == needle)
}

/// Donor `generationQuality`: ASCII-space/lowercase continuation, a 16-byte
/// corpus prefix match, and 90% eight-gram corpus coverage.
fn generation_quality(generated: &[u8]) -> bool {
	let prompt = GENERATION_PROMPT.as_bytes();
	if generated.len() != prompt.len() + GENERATION_LENGTH {
		return false;
	}
	let continuation = &generated[prompt.len()..];
	if continuation
		.iter()
		.any(|&value| value != b' ' && !value.is_ascii_lowercase())
	{
		return false;
	}
	let corpus = CORPUS.as_bytes();
	let mut best_prefix = 0_usize;
	let mut from = 0_usize;
	while let Some(relative) = find_subslice(&corpus[from..], prompt) {
		let start = from + relative + prompt.len();
		let mut matched = 0_usize;
		while matched < continuation.len()
			&& start + matched < corpus.len()
			&& continuation[matched] == corpus[start + matched]
		{
			matched += 1;
		}
		best_prefix = best_prefix.max(matched);
		from += relative + 1;
	}
	const NGRAM: usize = 8;
	let mut supported = 0_usize;
	let mut total = 0_usize;
	for index in 0..=generated.len().saturating_sub(NGRAM) {
		total += 1;
		if find_subslice(corpus, &generated[index..index + NGRAM]).is_some() {
			supported += 1;
		}
	}
	let coverage = if total == 0 {
		0.0
	} else {
		supported as f32 / total as f32
	};
	best_prefix >= 16 && coverage >= 0.90
}

/// Donor `finalizeReport`: optimizer/hash/accuracy/generation accounting,
/// checkpoint save/reload round trip, and the canonical quality gate.
fn finalize_report(
	engine: &Engine,
	model: &ByteTransformer,
	optimizer: &mut AdamW,
	last_batch: &SatelliteNlpBatch,
	config: &SatelliteNlpCoordinatorConfig,
	report: &mut SatelliteNlpCoordinatorReport,
) -> Result<()> {
	report.optimizer_step = u64::from(optimizer.step_count());
	report.final_parameter_hash = satellite_hash_f32(&copy_parameters(model)?);
	report.final_accuracy = evaluate_accuracy(engine, model, last_batch)?;
	report.generated = generate_bytes_greedy(
		engine,
		model,
		GENERATION_PROMPT.as_bytes(),
		GENERATION_LENGTH,
	)?;
	report.generation_quality_passed = generation_quality(&report.generated);

	save_checkpoint(&config.checkpoint_path, model, optimizer)?;
	let reloaded = ByteTransformer::new(engine)?;
	let mut reloaded_optimizer = AdamW::new(reloaded.all_parameters()?, NLP_LEARNING_RATE)?;
	load_checkpoint(
		engine,
		&config.checkpoint_path,
		&reloaded,
		&mut reloaded_optimizer,
	)?;
	let reloaded_hash = satellite_hash_f32(&copy_parameters(&reloaded)?);
	if reloaded_hash != report.final_parameter_hash
		|| reloaded_optimizer.step_count() != optimizer.step_count()
	{
		return Err(Error::data_loss(
			"satellite NLP coordinator: checkpoint state changed on reload",
		));
	}
	let reloaded_generated = generate_bytes_greedy(
		engine,
		&reloaded,
		GENERATION_PROMPT.as_bytes(),
		GENERATION_LENGTH,
	)?;
	if reloaded_generated != report.generated {
		return Err(Error::data_loss(
			"satellite NLP coordinator: checkpoint generation changed on reload",
		));
	}
	report.checkpoint_round_trip = true;
	if report.completed_steps != config.steps
		|| report.optimizer_step != u64::from(config.steps)
		|| !report.initial_loss.is_finite()
		|| !report.final_loss.is_finite()
		|| !report.final_accuracy.is_finite()
	{
		return Err(Error::data_loss(
			"satellite NLP coordinator: completion accounting is invalid",
		));
	}
	if config.steps == TRAINING_STEPS as u32
		&& (report.final_loss >= report.initial_loss
			|| report.final_accuracy < 0.85
			|| !report.generation_quality_passed)
	{
		return Err(Error::failed_precondition(
			"satellite NLP coordinator: 300-step quality gate failed",
		));
	}
	Ok(())
}

fn drop_step_objects(client: &mut SatelliteClientSession) -> Result<()> {
	client.drop_object(INPUT_OBJECT)?;
	client.drop_object(TARGET_OBJECT)?;
	client.drop_object(RESULT_OBJECT)
}

/// Donor `satelliteRunSplitBatchNlp`: each canonical batch is split into a
/// local microbatch and a remote microbatch executed through the authenticated
/// satellite session; the coordinator reduces the weighted gradients and
/// applies one authoritative AdamW update per step.
///
/// # Errors
///
/// Returns `InvalidArgument` for an invalid split, step count, version, or
/// checkpoint path, `FailedPrecondition` for broken remote identity or
/// selection gates, `DataLoss` for malformed or non-finite remote results, and
/// the session errors produced by the named-work lifecycle.
pub(crate) fn satellite_run_split_batch_nlp(
	engine: &Engine,
	client: &mut SatelliteClientSession,
	config: &SatelliteNlpCoordinatorConfig,
) -> Result<SatelliteNlpCoordinatorReport> {
	if config.steps == 0
		|| config.steps > TRAINING_STEPS as u32
		|| config.local_batch_count == 0
		|| config.local_batch_count >= BATCH_SIZE as u32
		|| config.initial_version == 0
		|| config.initial_version > u64::MAX - u64::from(config.steps)
		|| config.checkpoint_path.as_os_str().is_empty()
	{
		return Err(Error::invalid_argument(
			"satellite NLP coordinator: invalid steps, split, version, or checkpoint path",
		));
	}

	let model = ByteTransformer::new(engine)?;
	if model.num_parameters()? != NLP_STEP_PARAMETER_COUNT {
		return Err(Error::failed_precondition(
			"satellite NLP coordinator: canonical parameter count changed",
		));
	}
	let mut optimizer = AdamW::new(model.all_parameters()?, NLP_LEARNING_RATE)?;
	checkpoint_and_wait(engine)?;
	let mut parameter_values = copy_parameters(&model)?;

	let mut report = SatelliteNlpCoordinatorReport {
		parameter_layout_hash: satellite_nlp_parameter_layout_hash(&model)?,
		..SatelliteNlpCoordinatorReport::default()
	};
	report.initial_parameter_hash = satellite_hash_f32(&parameter_values);
	let mut parameter_version = config.initial_version;
	let mut parameter_hash = report.initial_parameter_hash;
	let parameter_shape = [NLP_STEP_PARAMETER_COUNT as i64];
	let started = Instant::now();
	client.put_f32_versioned(
		PARAMETER_OBJECT,
		parameter_version,
		&parameter_shape,
		&parameter_values,
	)?;

	let remote_count = BATCH_SIZE as u32 - config.local_batch_count;
	let local_elements = config.local_batch_count as usize * CONTEXT_LENGTH;
	let remote_elements = remote_count as usize * CONTEXT_LENGTH;
	let local_weight = config.local_batch_count as f32 / BATCH_SIZE as f32;
	let remote_weight = remote_count as f32 / BATCH_SIZE as f32;
	let remote_shape = [i64::from(remote_count), CONTEXT_LENGTH as i64];
	let mut last_batch = satellite_build_nlp_batch(0)?;

	for step in 0..config.steps {
		let batch = satellite_build_nlp_batch(u64::from(step))?;
		last_batch = batch;
		let remote_inputs = &last_batch.inputs[local_elements..local_elements + remote_elements];
		let remote_targets = &last_batch.targets[local_elements..local_elements + remote_elements];
		let remote_samples = &last_batch.sample_indices[config.local_batch_count as usize..BATCH_SIZE];
		client.put_u32(INPUT_OBJECT, &remote_shape, remote_inputs)?;
		client.put_u32(TARGET_OBJECT, &remote_shape, remote_targets)?;
		let request = satellite_start_nlp_gradient(
			client,
			PARAMETER_OBJECT,
			INPUT_OBJECT,
			TARGET_OBJECT,
			RESULT_OBJECT,
			RNG_SEED,
			u64::from(step),
			config.local_batch_count,
			remote_samples,
			parameter_version,
			&parameter_hash,
		)?;

		let (waited_result, local_result) = if config.overlap_remote {
			std::thread::scope(|scope| {
				let wait_handle = scope.spawn(|| client.wait(request));
				let local = execute_local_gradient(
					engine,
					&model,
					&optimizer,
					&last_batch.inputs[..local_elements],
					&last_batch.targets[..local_elements],
				);
				(
					wait_handle.join().expect("remote wait thread panicked"),
					local,
				)
			})
		} else {
			let local = execute_local_gradient(
				engine,
				&model,
				&optimizer,
				&last_batch.inputs[..local_elements],
				&last_batch.targets[..local_elements],
			);
			(client.wait(request), local)
		};
		let local = local_result?;
		accumulate_local_selection(local.diagnostics, &mut report)?;
		let waited = waited_result?;
		let result_bytes = SatelliteClientSession::read_bytes_result(&waited)?;
		let remote = satellite_decode_nlp_gradient_result(&result_bytes)?;
		if remote.seed != RNG_SEED
			|| remote.step_index != u64::from(step)
			|| remote.batch_offset != config.local_batch_count
			|| remote.batch_count != remote_count
			|| remote.predicted_positions != remote_count * CONTEXT_LENGTH as u32
			|| remote.initial_parameter_hash != parameter_hash
			|| remote.input_hash != satellite_hash_u32(remote_inputs)
			|| remote.target_hash != satellite_hash_u32(remote_targets)
			|| remote.parameter_layout_hash != report.parameter_layout_hash
			|| remote.gradients.len() != local.values.len()
			|| !remote.loss.is_finite()
		{
			return Err(Error::data_loss(
				"satellite NLP coordinator: remote gradient identity is invalid",
			));
		}
		let profile = SatelliteClientSession::read_profile(&waited)?;
		let selections = profile_counter(&profile, "kernel_selections=")?;
		let fallbacks = profile_counter(&profile, "kernel_fallbacks=")?;
		if selections == 0 || report.remote_kernel_selections > u64::MAX - selections {
			return Err(Error::failed_precondition(
				"satellite NLP coordinator: remote GEMM selection gate failed",
			));
		}
		if fallbacks != 0 || report.remote_kernel_fallbacks > u64::MAX - fallbacks {
			return Err(Error::failed_precondition(
				"satellite NLP coordinator: remote GEMM selection gate failed",
			));
		}
		report.remote_kernel_selections += selections;
		report.remote_kernel_fallbacks += fallbacks;
		drop_step_objects(client)?;

		let mut reduced = vec![0.0_f32; local.values.len()];
		for (index, value) in reduced.iter_mut().enumerate() {
			*value = local.values[index] * local_weight + remote.gradients[index] * remote_weight;
			if !value.is_finite() {
				return Err(Error::data_loss(
					"satellite NLP coordinator: reduced gradient is not finite",
				));
			}
		}
		let step_loss = local.loss * local_weight + remote.loss * remote_weight;
		if step == 0 {
			report.initial_loss = step_loss;
		}
		report.final_loss = step_loss;
		apply_gradient(engine, &model, &mut optimizer, &reduced)?;
		parameter_values = copy_parameters(&model)?;
		parameter_hash = satellite_hash_f32(&parameter_values);
		client.drop_object(PARAMETER_OBJECT)?;
		parameter_version += 1;
		client.put_f32_versioned(
			PARAMETER_OBJECT,
			parameter_version,
			&parameter_shape,
			&parameter_values,
		)?;
		report.completed_steps = step + 1;
	}

	report.final_parameter_hash = parameter_hash;
	report.complete_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
	client.drop_object(PARAMETER_OBJECT)?;
	finalize_report(
		engine,
		&model,
		&mut optimizer,
		&last_batch,
		config,
		&mut report,
	)?;
	Ok(report)
}

/// Donor `satelliteRunStandaloneNlp`: the identical frozen recipe executed
/// entirely locally, with the same accounting, checkpoint, and quality gates.
///
/// # Errors
///
/// Returns `InvalidArgument` for an invalid step count or checkpoint path,
/// `FailedPrecondition` for the canonical quality gate, and `DataLoss` for
/// non-finite or inconsistent final state.
pub(crate) fn satellite_run_standalone_nlp(
	engine: &Engine,
	config: &SatelliteNlpCoordinatorConfig,
) -> Result<SatelliteNlpCoordinatorReport> {
	if config.steps == 0
		|| config.steps > TRAINING_STEPS as u32
		|| config.checkpoint_path.as_os_str().is_empty()
	{
		return Err(Error::invalid_argument(
			"satellite NLP standalone: invalid steps or checkpoint path",
		));
	}

	let model = ByteTransformer::new(engine)?;
	if model.num_parameters()? != NLP_STEP_PARAMETER_COUNT {
		return Err(Error::failed_precondition(
			"satellite NLP standalone: canonical parameter count changed",
		));
	}
	let mut optimizer = AdamW::new(model.all_parameters()?, NLP_LEARNING_RATE)?;
	checkpoint_and_wait(engine)?;
	let initial_parameters = copy_parameters(&model)?;

	let mut report = SatelliteNlpCoordinatorReport {
		parameter_layout_hash: satellite_nlp_parameter_layout_hash(&model)?,
		initial_parameter_hash: satellite_hash_f32(&initial_parameters),
		..SatelliteNlpCoordinatorReport::default()
	};
	let started = Instant::now();
	let mut last_batch = satellite_build_nlp_batch(0)?;
	for step in 0..config.steps {
		let batch = satellite_build_nlp_batch(u64::from(step))?;
		last_batch = batch;
		let (loss_value, diagnostics) =
			execute_standalone_step(engine, &model, &mut optimizer, &last_batch)?;
		accumulate_local_selection(diagnostics, &mut report)?;
		if step == 0 {
			report.initial_loss = loss_value;
		}
		report.final_loss = loss_value;
		report.completed_steps = step + 1;
	}
	report.complete_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
	finalize_report(
		engine,
		&model,
		&mut optimizer,
		&last_batch,
		config,
		&mut report,
	)?;
	Ok(report)
}

#[cfg(test)]
#[path = "../../../../test/rs/net/satellite_nlp_coordinator_unit.rs"]
mod tests;
