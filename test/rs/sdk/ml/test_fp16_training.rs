//! FP16-storage training-parity evidence.
//!
//! One NLP workload (the canonical byte-level RNN next-token gate) and one
//! vision workload (the ALM temporal-convolution VQ tokenizer) run identical
//! seeded training under the default F32 math mode and under
//! `OA_FORCE_PRECISION=FP16`. Each FP16 run first proves the lowered kernels
//! execute inside the workload through the execution-plan census, then the
//! loss trajectory is compared against the F32 run and every master weight
//! is asserted to remain F32.

use oa::ml::Module;

static PRECISION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Scoped `OA_FORCE_PRECISION` override; serialized so concurrent ignored
/// tests never observe a torn mode.
struct PrecisionMode {
	previous: Option<std::ffi::OsString>,
	_guard: std::sync::MutexGuard<'static, ()>,
}

impl PrecisionMode {
	fn set(value: &str) -> Self {
		let guard = PRECISION_LOCK
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		let previous = std::env::var_os("OA_FORCE_PRECISION");
		// SAFETY: every mode-sensitive test serializes on PRECISION_LOCK.
		unsafe { std::env::set_var("OA_FORCE_PRECISION", value) };
		Self {
			previous,
			_guard: guard,
		}
	}
}

impl Drop for PrecisionMode {
	fn drop(&mut self) {
		// SAFETY: same serialized scope as `set`.
		unsafe {
			match &self.previous {
				Some(value) => std::env::set_var("OA_FORCE_PRECISION", value),
				None => std::env::remove_var("OA_FORCE_PRECISION"),
			}
		}
	}
}

fn plan_kernel_names(plan: &oa::ExecutionPlan, label: &str) -> Vec<String> {
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json(label))
		.expect("execution report must be valid JSON");
	report["nodes"]
		.as_array()
		.expect("execution nodes must be an array")
		.iter()
		.map(|node| node["kernel"].as_str().unwrap_or_default().to_owned())
		.collect()
}

fn assert_f16_kernels_recorded(kernels: &[String], context: &str) {
	assert!(
		kernels.iter().any(|name| name == "matrix.cast_f32_f16.f16"),
		"{context}: FP16 mode must record F32->F16 storage casts, found {kernels:?}"
	);
	assert!(
		kernels.iter().any(|name| name.contains("_f16.")),
		"{context}: FP16 mode must record an FP16-storage kernel, found {kernels:?}"
	);
}

/// NLP workload: `STEPS` eager AdamW steps of the canonical byte RNN.
/// Returns the per-step loss trajectory and asserts every master weight
/// remains F32 after the run.
fn train_byte_rnn(_engine: &oa::Engine) -> oa::Result<(Vec<f32>, bool)> {
	const STEPS: usize = 48;
	let model = sdk::ml::nlp::ByteRnn::new()?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
	let mut sampler = sdk::ml::nlp::ByteSampler::new(sdk::ml::nlp::BATCH_SIZE)?;
	let mut losses = Vec::with_capacity(STEPS);
	for _ in 0..STEPS {
		let (input_values, target_values) = sampler.next_values()?;
		let input = oa::Matrix::new_from_slice(
			[sdk::ml::nlp::BATCH_SIZE, sdk::ml::nlp::CONTEXT_LENGTH],
			&input_values,
		)?;
		let target = oa::Matrix::new_from_slice(
			[sdk::ml::nlp::BATCH_SIZE * sdk::ml::nlp::CONTEXT_LENGTH],
			&target_values,
		)?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		optimizer.step()?;
		optimizer.zero_grad();
		losses.push(loss.read_f32()?[0]);
	}
	let master_weights_f32 = model
		.all_parameters()?
		.iter()
		.all(|parameter| parameter.data().dtype() == oa::DType::F32);
	Ok((losses, master_weights_f32))
}

/// Vision workload: `train_tokenizer` over the fixed tiny clip set.
/// Returns (initial, final) reconstruction losses and asserts every master
/// weight remains F32 after the run.
fn train_alm_tokenizer(engine: &oa::Engine) -> oa::Result<(f32, f32, bool)> {
	use sdk::ml::alm::{AlmTokenizer, AlmTokenizerConfig};
	let config = AlmTokenizerConfig {
		input_dim: 3,
		width: 4,
		code_dim: 4,
		num_codes: 2,
		downsample_stages: 1,
		depth: 1,
		commitment_beta: 0.25,
		ema_decay: 0.9,
		ema_epsilon: 1.0e-5,
		dead_threshold: 0.0,
	};
	let tokenizer = AlmTokenizer::new_with_seed(config, 11)?;
	let clips = vec![
		vec![0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8],
		vec![0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0],
	];
	let report = sdk::ml::alm::training::train_tokenizer(
		engine,
		&tokenizer,
		&clips,
		sdk::ml::alm::training::TokenizerTrainingConfig {
			epochs: 3,
			batch_size: 2,
			sequence_len: 4,
			learning_rate: 1.0e-3,
			minimum_learning_rate: 1.0e-4,
			warmup_steps: 1,
			weight_decay: 0.0,
			seed_codebook: true,
			enable_gpu_timing: false,
			show_progress: false,
			checkpoint: None,
		},
	)?;
	let master_weights_f32 = tokenizer
		.all_parameters()?
		.iter()
		.all(|parameter| parameter.data().dtype() == oa::DType::F32);
	Ok((
		report.initial_reconstruction_loss,
		report.final_reconstruction_loss,
		master_weights_f32,
	))
}

test_vk!(
	fp16_storage_byte_rnn_loss_parity_and_f32_master_weights,
	engine,
	{
		let (f32_losses, f32_master_weights) = train_byte_rnn(&engine)?;
		assert!(f32_master_weights, "F32 run must keep F32 master weights");
		let _mode = PrecisionMode::set("FP16");
		// Census: the workload's linear projections must lower to FP16 storage.
		let probe = sdk::ml::nlp::ByteRnn::new()?;
		let probe_input =
			oa::Matrix::new_from_slice([1, sdk::ml::nlp::CONTEXT_LENGTH], b"to be or not to ")?;
		let (plan, _logits) = engine.capture(|| probe.forward(&probe_input))?;
		assert_f16_kernels_recorded(
			&plan_kernel_names(&plan, "byte_rnn_forward"),
			"byte rnn forward",
		);
		let (f16_losses, f16_master_weights) = train_byte_rnn(&engine)?;
		assert!(
			f16_master_weights,
			"FP16-storage run must keep F32 master weights"
		);
		assert_eq!(f32_losses.len(), f16_losses.len());
		for (step, (reference, candidate)) in f32_losses.iter().zip(&f16_losses).enumerate() {
			let gap = (candidate - reference).abs();
			assert!(
				gap <= 0.20,
				"step {step}: FP16 loss {candidate} diverges {gap} from F32 {reference}"
			);
		}
		let f32_initial = f32_losses[0];
		let f16_initial = f16_losses[0];
		let f32_final = f32_losses[f32_losses.len() - 1];
		let f16_final = f16_losses[f16_losses.len() - 1];
		println!(
			"byte rnn fp16 parity: initial f32={f32_initial:.6} f16={f16_initial:.6} \
		 final f32={f32_final:.6} f16={f16_final:.6}"
		);
		assert!(f16_final < f16_initial, "FP16 run must reduce the loss");
		Ok(())
	}
);

test_vk!(
	fp16_storage_alm_tokenizer_loss_parity_and_f32_master_weights,
	engine,
	{
		use sdk::ml::alm::{AlmTokenizer, AlmTokenizerConfig};
		let (f32_initial, f32_final, f32_master_weights) = train_alm_tokenizer(&engine)?;
		assert!(f32_master_weights, "F32 run must keep F32 master weights");
		let _mode = PrecisionMode::set("FP16");
		// Census: the workload's codebook/quantization reductions must lower to
		// FP16 storage where the math mode is admitted.
		let probe = AlmTokenizer::new_with_seed(
			AlmTokenizerConfig {
				input_dim: 3,
				width: 4,
				code_dim: 4,
				num_codes: 2,
				downsample_stages: 1,
				depth: 1,
				commitment_beta: 0.25,
				ema_decay: 0.9,
				ema_epsilon: 1.0e-5,
				dead_threshold: 0.0,
			},
			11,
		)?;
		let probe_input = oa::Matrix::new_from_f32(
			[2, 4, 3],
			&[
				0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5, 0.4,
				0.4, 0.3, 0.2, 0.2, 0.1, 0.0,
			],
		)?;
		let latent = probe.encode(&probe_input)?;
		engine.checkpoint()?.wait()?;
		let (plan, _result) = engine.capture(|| probe.quantize(&latent))?;
		assert_f16_kernels_recorded(
			&plan_kernel_names(&plan, "alm_tokenizer_quantize"),
			"alm tokenizer quantize",
		);
		let (f16_initial, f16_final, f16_master_weights) = train_alm_tokenizer(&engine)?;
		assert!(
			f16_master_weights,
			"FP16-storage run must keep F32 master weights"
		);
		println!(
			"alm tokenizer fp16 parity: initial f32={f32_initial:.6} f16={f16_initial:.6} \
		 final f32={f32_final:.6} f16={f16_final:.6}"
		);
		assert!(
			(f16_initial - f32_initial).abs() <= 0.05,
			"initial reconstruction loss must match: f32={f32_initial} f16={f16_initial}"
		);
		assert!(
			(f16_final - f32_final).abs() <= 0.05,
			"final reconstruction loss must stay close: f32={f32_final} f16={f16_final}"
		);
		Ok(())
	}
);
