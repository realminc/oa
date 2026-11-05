//! Donor `testSatelliteNlp.cpp` coverage for the private satellite NLP
//! named-work surface: the frozen byte-Transformer step and split-batch
//! gradient executors, the deterministic batch builder, and the wire result
//! decoders. The server serves on the test thread because
//! [`SatelliteServerSession`] borrows a `!Send` engine; the client runs on a
//! spawned thread.

use std::cell::Cell;
use std::rc::Rc;
use std::thread;

use super::super::SatelliteProtocol;
use super::super::session::*;
use super::*;

use crate::TcpListener;
use crate::ml::{AdamW, GradientTape, Module, loss};
use crate::sdk::ml::nlp::{ByteTransformer, CONTEXT_LENGTH, CORPUS, RNG_SEED, TRAINING_STEPS};
use crate::{DType, Engine, Error, ErrorKind, Matrix, Result};

const BATCH_SIZE: usize = 64;

fn satellite_mac(secret: &[u8], data: &[u8], custom: &str, out: &mut [u8; 32]) -> Result<()> {
	out.copy_from_slice(&crate::crypto::kmac256(
		secret,
		data,
		custom.as_bytes(),
		32,
	)?);
	Ok(())
}

fn make_secret(seed: u8) -> SatelliteSecret {
	let bytes: Vec<u8> = (0..32).map(|i| seed.wrapping_add(i)).collect();
	SatelliteSecret::from_bytes(&bytes).expect("32-byte secret")
}

fn make_config(seed: u8, device_name: String) -> SatelliteSessionConfig {
	let mut config = SatelliteSessionConfig::new(make_secret(seed), satellite_mac);
	config.io_timeout_ms = 30_000;
	config.limits.max_payload_bytes = SatelliteProtocol::MAX_PAYLOAD_BYTES as u32;
	config.limits.max_resident_bytes = 16 * 1024 * 1024;
	config.limits.max_objects = 16;
	config.limits.max_inflight = 1;
	config.device_name = device_name;
	config.named_operations = vec![
		NLP_STEP_OPERATION.to_owned(),
		NLP_GRADIENT_OPERATION.to_owned(),
	];
	config
}

fn nlp_config(seed: u8, device_name: String, calls: Rc<Cell<u32>>) -> SatelliteSessionConfig {
	let mut config = make_config(seed, device_name);
	config.named_work = Some(Box::new(move |engine, request| {
		calls.set(calls.get() + 1);
		satellite_execute_nlp_work(engine, request)
	}));
	config
}

fn test_engine() -> Result<Engine> {
	Engine::builder()
		.devices(crate::DeviceSelection::Index(0))
		.build()
}

fn serve_with_client<'a, T: Send + 'static>(
	listener: &TcpListener,
	mut server: SatelliteServerSession<'a>,
	client: impl FnOnce(u16) -> T + Send + 'static,
) -> (Result<()>, SatelliteServerSession<'a>, T) {
	let port = listener.port();
	let client_thread = thread::spawn(move || client(port));
	let accepted = listener.accept().expect("accepted stream");
	let status = server.serve(accepted);
	(status, server, client_thread.join().expect("client thread"))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(values.len() * 4);
	for &value in values {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	bytes
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(values.len() * 4);
	for &value in values {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	bytes
}

fn near(actual: f32, expected: f32) -> bool {
	near_tol(actual, expected, 1e-5, 1e-5)
}

fn near_tol(actual: f32, expected: f32, abs: f32, rel: f32) -> bool {
	(actual - expected).abs() <= abs + rel * actual.abs().max(expected.abs())
}

fn expect_vector_near(actual: &[f32], expected: &[f32], abs: f32, rel: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			near_tol(actual, expected, abs, rel),
			"mismatch at element {index}: {actual} vs {expected}"
		);
	}
}

fn hash_is_zero(hash: &[u8; 32]) -> bool {
	hash.iter().all(|&value| value == 0)
}

/// Deterministic frozen recipe inputs: canonical initial parameters and the
/// step-0 batch. Model construction is eager work; `Engine::checkpoint`
/// submits it exactly once, matching the donor's ambient submit.
fn build_frozen_inputs(engine: &Engine) -> Result<(Vec<f32>, Vec<u32>, Vec<u32>)> {
	let model = ByteTransformer::new(engine)?;
	let parameters: Vec<Matrix> = model
		.all_parameters()?
		.iter()
		.map(|parameter| parameter.data())
		.collect();
	let event = engine.checkpoint()?;
	event.wait()?;
	let mut flat = Vec::new();
	for parameter in &parameters {
		flat.extend(parameter.read_f32()?);
	}
	assert_eq!(flat.len(), NLP_STEP_PARAMETER_COUNT);
	let batch = satellite_build_nlp_batch(0)?;
	Ok((flat, batch.inputs, batch.targets))
}

struct LocalStepResult {
	parameter_layout_hash: [u8; 32],
	loss: f32,
	logit_min: f32,
	logit_max: f32,
	logit_mean: f32,
	logit_l2: f32,
	logit_sample: Vec<f32>,
	gradients: Vec<f32>,
	updated_parameters: Vec<f32>,
}

/// Local full-batch optimizer-step oracle mirroring the donor's
/// `runLocalOracle`.
fn run_local_oracle(
	engine: &Engine,
	parameters: &[f32],
	inputs: &[u32],
	targets: &[u32],
) -> Result<LocalStepResult> {
	let batch_count = inputs.len() / CONTEXT_LENGTH;
	type OracleCapture = ([u8; 32], Matrix, Matrix, Vec<Matrix>, Vec<Matrix>);
	let (plan, captured) = engine.capture(|| -> Result<OracleCapture> {
		let model = ByteTransformer::new(engine)?;
		let named = model.all_named_parameters()?;
		let layout_hash = parameter_layout_hash(&named);
		load_parameters(engine, &named, parameters)?;
		let parameters_ref = model.all_parameters()?;
		let mut optimizer = AdamW::new(parameters_ref.clone(), 0.01)?;
		optimizer.zero_grad();
		let input = Matrix::new_from_slice(engine, [batch_count, CONTEXT_LENGTH], inputs)?;
		let target = Matrix::new_from_slice(engine, [inputs.len()], targets)?;
		let tape = GradientTape::new();
		let logits = model.forward(&input)?;
		let step_loss = loss::cross_entropy(&logits, &target)?;
		tape.backward(&step_loss)?;
		optimizer.step()?;
		let mut gradient_matrices = Vec::with_capacity(parameters_ref.len());
		let mut data_matrices = Vec::with_capacity(parameters_ref.len());
		for parameter in &parameters_ref {
			let gradient = parameter
				.gradient()
				.ok_or_else(|| Error::data_loss("satellite NLP oracle: a model gradient is missing"))?;
			gradient_matrices.push(gradient);
			data_matrices.push(parameter.data());
		}
		Ok((
			layout_hash,
			step_loss,
			logits,
			gradient_matrices,
			data_matrices,
		))
	})?;
	let event = engine.submit(&plan)?;
	event.wait()?;
	if !engine.handle().owns_event(&event) || !event.is_complete().unwrap_or(false) {
		return Err(Error::internal(
			"satellite NLP oracle: completion provenance failed",
		));
	}
	let (layout_hash, loss_matrix, logits, gradient_matrices, data_matrices) = captured;
	let loss_value = loss_matrix.read_f32()?[0];
	let all_logits = logits.read_f32()?;
	let mut logit_min = f32::INFINITY;
	let mut logit_max = f32::NEG_INFINITY;
	let mut sum = 0.0_f64;
	let mut sum_squares = 0.0_f64;
	for &value in &all_logits {
		assert!(value.is_finite(), "satellite NLP oracle: non-finite logit");
		logit_min = logit_min.min(value);
		logit_max = logit_max.max(value);
		sum += f64::from(value);
		sum_squares += f64::from(value) * f64::from(value);
	}
	let mut gradients = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for matrix in &gradient_matrices {
		gradients.extend(matrix.read_f32()?);
	}
	let mut updated = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for matrix in &data_matrices {
		updated.extend(matrix.read_f32()?);
	}
	Ok(LocalStepResult {
		parameter_layout_hash: layout_hash,
		loss: loss_value,
		logit_min,
		logit_max,
		logit_mean: (sum / all_logits.len() as f64) as f32,
		logit_l2: sum_squares.sqrt() as f32,
		logit_sample: all_logits[..NLP_STEP_LOGIT_SAMPLE_COUNT].to_vec(),
		gradients,
		updated_parameters: updated,
	})
}

/// Local microbatch gradient oracle mirroring the donor's `runLocalGradient`.
fn run_local_gradient(
	engine: &Engine,
	parameters: &[f32],
	inputs: &[u32],
	targets: &[u32],
) -> Result<LocalStepResult> {
	if inputs.is_empty()
		|| inputs.len() != targets.len()
		|| !inputs.len().is_multiple_of(CONTEXT_LENGTH)
	{
		return Err(Error::invalid_argument(
			"satellite NLP oracle: invalid microbatch shape",
		));
	}
	let batch_count = inputs.len() / CONTEXT_LENGTH;
	let (plan, captured) = engine.capture(|| -> Result<([u8; 32], Matrix, Vec<Matrix>)> {
		let model = ByteTransformer::new(engine)?;
		let named = model.all_named_parameters()?;
		let layout_hash = parameter_layout_hash(&named);
		load_parameters(engine, &named, parameters)?;
		let parameters_ref = model.all_parameters()?;
		let input = Matrix::new_from_slice(engine, [batch_count, CONTEXT_LENGTH], inputs)?;
		let target = Matrix::new_from_slice(engine, [inputs.len()], targets)?;
		let tape = GradientTape::new();
		let logits = model.forward(&input)?;
		let step_loss = loss::cross_entropy(&logits, &target)?;
		tape.backward(&step_loss)?;
		let mut gradient_matrices = Vec::with_capacity(parameters_ref.len());
		for parameter in &parameters_ref {
			let gradient = parameter
				.gradient()
				.ok_or_else(|| Error::data_loss("satellite NLP oracle: a model gradient is missing"))?;
			gradient_matrices.push(gradient);
		}
		Ok((layout_hash, step_loss, gradient_matrices))
	})?;
	let event = engine.submit(&plan)?;
	event.wait()?;
	let (layout_hash, loss_matrix, gradient_matrices) = captured;
	let mut gradients = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for matrix in &gradient_matrices {
		gradients.extend(matrix.read_f32()?);
	}
	Ok(LocalStepResult {
		parameter_layout_hash: layout_hash,
		loss: loss_matrix.read_f32()?[0],
		logit_min: 0.0,
		logit_max: 0.0,
		logit_mean: 0.0,
		logit_l2: 0.0,
		logit_sample: Vec::new(),
		gradients,
		updated_parameters: Vec::new(),
	})
}

/// Authoritative optimizer update against a supplied reduced gradient,
/// mirroring the donor's `applyAuthoritativeGradient`.
fn apply_authoritative_gradient(
	engine: &Engine,
	parameters: &[f32],
	gradient_values: &[f32],
) -> Result<Vec<f32>> {
	let (plan, updated) = engine.capture(|| -> Result<Vec<Matrix>> {
		let model = ByteTransformer::new(engine)?;
		let named = model.all_named_parameters()?;
		load_parameters(engine, &named, parameters)?;
		let parameters_ref = model.all_parameters()?;
		let mut offset = 0_usize;
		for parameter in &parameters_ref {
			let data = parameter.data();
			let count = data.num_elements();
			if offset > gradient_values.len() || count > gradient_values.len() - offset {
				return Err(Error::shape_mismatch(
					"satellite NLP oracle: reduced gradient is truncated",
				));
			}
			let gradient = Matrix::new_from_slice(
				engine,
				data.shape().to_vec(),
				&gradient_values[offset..offset + count],
			)?;
			parameter.accumulate_gradient(gradient)?;
			offset += count;
		}
		if offset != gradient_values.len() {
			return Err(Error::shape_mismatch(
				"satellite NLP oracle: reduced gradient has trailing values",
			));
		}
		let mut optimizer = AdamW::new(parameters_ref.clone(), 0.01)?;
		optimizer.step()?;
		Ok(
			parameters_ref
				.iter()
				.map(|parameter| parameter.data())
				.collect(),
		)
	})?;
	let event = engine.submit(&plan)?;
	event.wait()?;
	let mut values = Vec::with_capacity(NLP_STEP_PARAMETER_COUNT);
	for matrix in &updated {
		values.extend(matrix.read_f32()?);
	}
	Ok(values)
}

#[test]
fn batch_builder_matches_the_frozen_sampler() -> Result<()> {
	let first = satellite_build_nlp_batch(0)?;
	assert_eq!(first.sample_indices.len(), BATCH_SIZE);
	assert_eq!(first.inputs.len(), BATCH_SIZE * CONTEXT_LENGTH);
	assert_eq!(first.targets.len(), first.inputs.len());
	let corpus = CORPUS.as_bytes();
	let limit = corpus.len() - CONTEXT_LENGTH - 1;
	for (index, &sample) in first.sample_indices.iter().enumerate() {
		assert_eq!(sample as usize, index * 7 % limit);
	}
	assert_eq!(
		satellite_build_nlp_batch(TRAINING_STEPS as u64)
			.err()
			.unwrap()
			.kind(),
		ErrorKind::OutOfRange
	);
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_nlp_replicated_step_matches_local_oracle() -> Result<()> {
	let engine = test_engine()?;
	let (initial_parameters, inputs, targets) = build_frozen_inputs(&engine)?;
	let initial_hash = satellite_hash_f32(&initial_parameters);
	let initial_version = 7_u64;
	let limit = CORPUS.len() - CONTEXT_LENGTH - 1;
	let sample_indices: Vec<u32> = (0..BATCH_SIZE)
		.map(|index| (index * 7 % limit) as u32)
		.collect();

	// The workload boundary independently rejects a batch that does not match
	// the declared canonical sample indices, before recording GPU work.
	let mut corrupted = inputs.clone();
	corrupted[0] ^= 1;
	let mut step_arguments = Vec::new();
	append_u32(&mut step_arguments, ARGUMENT_MAGIC);
	append_u32(&mut step_arguments, ARGUMENT_VERSION);
	append_u64(&mut step_arguments, RNG_SEED);
	append_u64(&mut step_arguments, 0);
	append_u32(&mut step_arguments, BATCH_SIZE as u32);
	for &sample in &sample_indices {
		append_u32(&mut step_arguments, sample);
	}
	let parameter_bytes = encode_f32(&initial_parameters);
	let input_bytes = encode_u32(&corrupted);
	let target_bytes = encode_u32(&targets);
	let parameter_shape = [NLP_STEP_PARAMETER_COUNT as i64];
	let batch_shape = [BATCH_SIZE as i64, CONTEXT_LENGTH as i64];
	let invalid_batch = SatelliteNamedRequest {
		operation: NLP_STEP_OPERATION.to_owned(),
		inputs: vec![
			SatelliteObjectView {
				id: 1,
				version: 0,
				dtype: DType::F32,
				shape: &parameter_shape,
				data: &parameter_bytes,
				hash: initial_hash,
			},
			SatelliteObjectView {
				id: 2,
				version: 0,
				dtype: DType::U32,
				shape: &batch_shape,
				data: &input_bytes,
				hash: satellite_hash_u32(&corrupted),
			},
			SatelliteObjectView {
				id: 3,
				version: 0,
				dtype: DType::U32,
				shape: &batch_shape,
				data: &target_bytes,
				hash: satellite_hash_u32(&targets),
			},
		],
		arguments: &step_arguments,
		expected_version: 0,
		expected_hash: initial_hash,
	};
	let invalid = satellite_execute_nlp_step(&engine, &invalid_batch);
	assert_eq!(invalid.err().unwrap().kind(), ErrorKind::FailedPrecondition);

	let local = run_local_oracle(&engine, &initial_parameters, &inputs, &targets)?;

	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let calls = Rc::new(Cell::new(0_u32));
	let device_name = engine.device_info().name().to_owned();
	let server = SatelliteServerSession::new_with_engine(
		&engine,
		nlp_config(0xa0, device_name.clone(), Rc::clone(&calls)),
	);
	let expected_parameters = initial_parameters.clone();
	let expected_inputs = inputs.clone();
	let expected_targets = targets.clone();
	let (server_status, server, client_status) = serve_with_client(
		&listener,
		server,
		move |port| -> Result<SatelliteNlpStepResult> {
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, make_config(0xa0, device_name))?;
			client.put_f32_versioned(1, initial_version, &parameter_shape, &expected_parameters)?;
			client.put_u32(2, &batch_shape, &expected_inputs)?;
			client.put_u32(3, &batch_shape, &expected_targets)?;

			assert_eq!(
				satellite_start_nlp_step(
					&mut client,
					1,
					2,
					3,
					4,
					RNG_SEED,
					0,
					&sample_indices,
					u64::MAX,
					&initial_hash,
				)
				.unwrap_err()
				.kind(),
				ErrorKind::OutOfRange
			);
			let mut wrong_hash = initial_hash;
			wrong_hash[0] ^= 0x01;
			assert_eq!(
				satellite_start_nlp_step(
					&mut client,
					1,
					2,
					3,
					4,
					RNG_SEED,
					0,
					&sample_indices,
					initial_version,
					&wrong_hash,
				)
				.unwrap_err()
				.kind(),
				ErrorKind::FailedPrecondition
			);
			assert_eq!(
				client.drop_object(4).unwrap_err().kind(),
				ErrorKind::NotFound
			);

			// A pre-submission cancellation must not invoke the workload or
			// create its destination. The same destination is then reusable.
			let cancelled_request = satellite_start_nlp_step(
				&mut client,
				1,
				2,
				3,
				4,
				RNG_SEED,
				0,
				&sample_indices,
				initial_version,
				&initial_hash,
			)?;
			client.cancel(cancelled_request)?;
			assert_eq!(
				client.drop_object(4).unwrap_err().kind(),
				ErrorKind::NotFound
			);

			let request = satellite_start_nlp_step(
				&mut client,
				1,
				2,
				3,
				4,
				RNG_SEED,
				0,
				&sample_indices,
				initial_version,
				&initial_hash,
			)?;
			let waited = client.wait(request)?;
			let bytes = SatelliteClientSession::read_bytes_result(&waited)?;
			let remote = satellite_decode_nlp_step_result(&bytes)?;

			assert_eq!(
				satellite_decode_nlp_step_result(&bytes[..bytes.len() - 1])
					.err()
					.unwrap()
					.kind(),
				ErrorKind::DataLoss
			);
			let mut trailing_bytes = bytes.clone();
			trailing_bytes.push(0);
			assert_eq!(
				satellite_decode_nlp_step_result(&trailing_bytes)
					.err()
					.unwrap()
					.kind(),
				ErrorKind::DataLoss
			);
			let mut wrong_version_bytes = bytes.clone();
			wrong_version_bytes[4] ^= 1;
			assert_eq!(
				satellite_decode_nlp_step_result(&wrong_version_bytes)
					.err()
					.unwrap()
					.kind(),
				ErrorKind::FailedPrecondition
			);

			assert_eq!(remote.seed, RNG_SEED);
			assert_eq!(remote.step_index, 0);
			assert_eq!(
				remote.predicted_positions as usize,
				BATCH_SIZE * CONTEXT_LENGTH
			);
			assert_eq!(remote.vocab_size, 256);
			assert_eq!(remote.initial_parameter_hash, initial_hash);
			assert_eq!(remote.input_hash, satellite_hash_u32(&expected_inputs));
			assert_eq!(remote.target_hash, satellite_hash_u32(&expected_targets));
			assert!(!hash_is_zero(&remote.parameter_layout_hash));

			let profile = SatelliteClientSession::read_profile(&waited)?;
			let profile_text = String::from_utf8(profile).expect("profile text");
			assert!(profile_text.contains("workload=nlp-byte-transformer-step-v1"));
			assert!(profile_text.contains("batch_offset=0;batch_count=64"));

			let retained = client.get_result(request)?;
			assert_eq!(SatelliteClientSession::read_bytes_result(&retained)?, bytes);

			client.drop_object(1)?;
			client.drop_object(2)?;
			client.drop_object(3)?;
			client.drop_object(4)?;
			client.close()?;
			Ok(remote)
		},
	);
	let remote = client_status.expect("client session");
	server_status.expect("server session");
	assert_eq!(calls.get(), 1);
	assert_eq!(remote.parameter_layout_hash, local.parameter_layout_hash);
	assert_eq!(
		remote.updated_parameter_hash,
		satellite_hash_f32(&local.updated_parameters)
	);
	assert!(near(remote.loss, local.loss));
	assert!(near(remote.logit_min, local.logit_min));
	assert!(near(remote.logit_max, local.logit_max));
	assert!(near(remote.logit_mean, local.logit_mean));
	assert!(near_tol(remote.logit_l2, local.logit_l2, 1e-4, 1e-5));
	expect_vector_near(&remote.logit_sample, &local.logit_sample, 1e-5, 1e-5);
	expect_vector_near(&remote.gradients, &local.gradients, 1e-5, 1e-5);
	expect_vector_near(
		&remote.updated_parameters,
		&local.updated_parameters,
		1e-6,
		1e-6,
	);
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());
	assert_ne!(server.last_gpu_event_value(), 0);
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_nlp_split_batch_gradient_and_authoritative_update_match_full_batch() -> Result<()> {
	let engine = test_engine()?;
	let (initial_parameters, inputs, targets) = build_frozen_inputs(&engine)?;
	let initial_hash = satellite_hash_f32(&initial_parameters);
	let full = run_local_oracle(&engine, &initial_parameters, &inputs, &targets)?;

	let local_count = BATCH_SIZE / 2;
	let remote_count = BATCH_SIZE - local_count;
	let local_elements = local_count * CONTEXT_LENGTH;
	let remote_elements = remote_count * CONTEXT_LENGTH;
	let local = run_local_gradient(
		&engine,
		&initial_parameters,
		&inputs[..local_elements],
		&targets[..local_elements],
	)?;

	let limit = CORPUS.len() - CONTEXT_LENGTH - 1;
	let sample_indices: Vec<u32> = (0..BATCH_SIZE)
		.map(|index| (index * 7 % limit) as u32)
		.collect();
	let remote_inputs = &inputs[local_elements..local_elements + remote_elements];
	let remote_targets = &targets[local_elements..local_elements + remote_elements];
	let remote_samples = &sample_indices[local_count..];

	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let calls = Rc::new(Cell::new(0_u32));
	let device_name = engine.device_info().name().to_owned();
	let server = SatelliteServerSession::new_with_engine(
		&engine,
		nlp_config(0xa0, device_name.clone(), Rc::clone(&calls)),
	);
	let expected_parameters = initial_parameters.clone();
	let remote_inputs_owned = remote_inputs.to_vec();
	let remote_targets_owned = remote_targets.to_vec();
	let remote_samples_owned = remote_samples.to_vec();
	let (server_status, server, client_status) = serve_with_client(
		&listener,
		server,
		move |port| -> Result<SatelliteNlpGradientResult> {
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, make_config(0xa0, device_name))?;
			let initial_version = 11_u64;
			let parameter_shape = [NLP_STEP_PARAMETER_COUNT as i64];
			let remote_shape = [remote_count as i64, CONTEXT_LENGTH as i64];
			client.put_f32_versioned(1, initial_version, &parameter_shape, &expected_parameters)?;
			client.put_u32(2, &remote_shape, &remote_inputs_owned)?;
			client.put_u32(3, &remote_shape, &remote_targets_owned)?;

			assert!(
				satellite_start_nlp_gradient(
					&mut client,
					1,
					2,
					3,
					4,
					RNG_SEED,
					0,
					BATCH_SIZE as u32,
					&remote_samples_owned,
					initial_version,
					&initial_hash,
				)
				.is_err()
			);

			let request = satellite_start_nlp_gradient(
				&mut client,
				1,
				2,
				3,
				4,
				RNG_SEED,
				0,
				local_count as u32,
				&remote_samples_owned,
				initial_version,
				&initial_hash,
			)?;
			let waited = client.wait(request)?;
			let bytes = SatelliteClientSession::read_bytes_result(&waited)?;
			let remote = satellite_decode_nlp_gradient_result(&bytes)?;
			assert!(satellite_decode_nlp_gradient_result(&bytes[..bytes.len() - 1]).is_err());
			assert_eq!(remote.seed, RNG_SEED);
			assert_eq!(remote.step_index, 0);
			assert_eq!(remote.batch_offset, local_count as u32);
			assert_eq!(remote.batch_count, remote_count as u32);
			assert_eq!(
				remote.predicted_positions as usize,
				remote_count * CONTEXT_LENGTH
			);
			assert_eq!(remote.initial_parameter_hash, initial_hash);
			assert_eq!(remote.input_hash, satellite_hash_u32(&remote_inputs_owned));
			assert_eq!(
				remote.target_hash,
				satellite_hash_u32(&remote_targets_owned)
			);

			let profile = SatelliteClientSession::read_profile(&waited)?;
			let profile_text = String::from_utf8(profile).expect("profile text");
			assert!(profile_text.contains("workload=nlp-byte-transformer-gradient-v1"));
			assert!(profile_text.contains("batch_offset=32;batch_count=32"));

			client.drop_object(2)?;
			client.drop_object(3)?;
			client.drop_object(4)?;
			client.close()?;
			Ok(remote)
		},
	);
	let remote = client_status.expect("client session");
	server_status.expect("server session");
	assert_eq!(calls.get(), 1);
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());

	// Split-batch FP32 reduction versus the full-batch gradient.
	let local_weight = local_count as f32 / BATCH_SIZE as f32;
	let remote_weight = remote_count as f32 / BATCH_SIZE as f32;
	assert!(near_tol(
		local.loss * local_weight + remote.loss * remote_weight,
		full.loss,
		2e-5,
		2e-5
	));
	assert_eq!(remote.parameter_layout_hash, local.parameter_layout_hash);
	assert_eq!(local.gradients.len(), remote.gradients.len());
	let mut reduced = vec![0.0_f32; local.gradients.len()];
	for (index, value) in reduced.iter_mut().enumerate() {
		*value = local.gradients[index] * local_weight + remote.gradients[index] * remote_weight;
	}
	expect_vector_near(&reduced, &full.gradients, 2e-6, 2e-4);

	let authoritative = apply_authoritative_gradient(&engine, &initial_parameters, &reduced)?;
	let full_authoritative =
		apply_authoritative_gradient(&engine, &initial_parameters, &full.gradients)?;
	expect_vector_near(&full_authoritative, &full.updated_parameters, 2e-6, 2e-4);

	// A split reduction changes FP32 summation order; the first-step AdamW
	// response is exactly the normalized-gradient difference.
	let adam_eps = 1e-8_f32;
	let learning_rate = 0.01_f32;
	assert_eq!(authoritative.len(), full_authoritative.len());
	for index in 0..authoritative.len() {
		let normalized = |gradient: f32| gradient / (gradient.abs() + adam_eps);
		let expected_difference =
			-learning_rate * (normalized(reduced[index]) - normalized(full.gradients[index]));
		let actual_difference = authoritative[index] - full_authoritative[index];
		assert!(
			(actual_difference - expected_difference).abs() <= 2e-6,
			"AdamW split-reduction response at element {index}"
		);
	}

	// A stale expected-version/hash pair is rejected before the workload runs;
	// the acknowledged pair is admitted.
	let authoritative_hash = satellite_hash_f32(&authoritative);
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let calls = Rc::new(Cell::new(0_u32));
	let device_name = engine.device_info().name().to_owned();
	let server = SatelliteServerSession::new_with_engine(
		&engine,
		nlp_config(0xa0, device_name.clone(), Rc::clone(&calls)),
	);
	let stale_inputs = remote_inputs.to_vec();
	let stale_targets = remote_targets.to_vec();
	let stale_samples = remote_samples.to_vec();
	let (server_status, server, client_status) =
		serve_with_client(&listener, server, move |port| -> Result<()> {
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, make_config(0xa0, device_name))?;
			let initial_version = 11_u64;
			let parameter_shape = [NLP_STEP_PARAMETER_COUNT as i64];
			let remote_shape = [remote_count as i64, CONTEXT_LENGTH as i64];
			client.put_f32_versioned(5, initial_version + 1, &parameter_shape, &authoritative)?;
			client.put_u32(2, &remote_shape, &stale_inputs)?;
			client.put_u32(3, &remote_shape, &stale_targets)?;
			assert_eq!(
				satellite_start_nlp_gradient(
					&mut client,
					5,
					2,
					3,
					6,
					RNG_SEED,
					0,
					local_count as u32,
					&stale_samples,
					initial_version,
					&authoritative_hash,
				)
				.unwrap_err()
				.kind(),
				ErrorKind::FailedPrecondition
			);
			let acknowledged = satellite_start_nlp_gradient(
				&mut client,
				5,
				2,
				3,
				6,
				RNG_SEED,
				0,
				local_count as u32,
				&stale_samples,
				initial_version + 1,
				&authoritative_hash,
			)?;
			client.wait(acknowledged)?;
			client.drop_object(2)?;
			client.drop_object(3)?;
			client.drop_object(5)?;
			client.drop_object(6)?;
			client.close()
		});
	client_status.expect("client session");
	server_status.expect("server session");
	assert_eq!(calls.get(), 1);
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());
	Ok(())
}
