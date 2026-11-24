use oa::ml::{ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary};
use oa::{Cli, LogMetrics};

// Translation-style task: emit the reversed source sequence. Fixed source and
// target lengths keep the contract free of padding masks; BOS is slot zero.
const VOCAB_SIZE: usize = 14; // 0 = BOS/PAD, 1 = EOS, 2..=13 = symbols
const EOS: u32 = 1;
const SOURCE_LENGTH: usize = 10;
const TARGET_LENGTH: usize = SOURCE_LENGTH + 2; // BOS + symbols + EOS
const TRAINING_STEPS: usize = 300;
const BATCH_SIZE: usize = 64;
const EVAL_SEQUENCES: usize = 64;

#[derive(Default)]
struct Seq2SeqConfig {
	steps: u64,
	batch: u64,
	lr: f32,
}

fn main() -> oa::Result<()> {
	let mut cli = Cli::new(
		"tu_nlp_seq2seq_transformer",
		"OA Tutorial — Seq2Seq Transformer",
	);
	cli.add_option(
		"--steps",
		|c: &mut Seq2SeqConfig| &mut c.steps,
		"training steps",
	);
	cli.add_option(
		"--batch",
		|c: &mut Seq2SeqConfig| &mut c.batch,
		"batch size",
	);
	cli.add_option("--lr", |c: &mut Seq2SeqConfig| &mut c.lr, "learning rate");
	if !cli.parse() {
		return Ok(());
	}
	let steps = if cli.config().steps > 0 {
		cli.config().steps as usize
	} else {
		TRAINING_STEPS
	};
	let batch = if cli.config().batch > 0 {
		cli.config().batch as usize
	} else {
		BATCH_SIZE
	};
	let lr = if cli.config().lr > 0.0 {
		cli.config().lr
	} else {
		0.01
	};

	let engine = oa::Engine::new()?;
	oa::Engine::set_default(engine.clone());
	let model = oa::ml::nn::EncoderDecoder::new_with_seed(
		VOCAB_SIZE,
		VOCAB_SIZE,
		SOURCE_LENGTH,
		TARGET_LENGTH,
		32,
		64,
		2,
		2,
		1.0e-5,
		0x5_E2C_DE0,
	)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, lr)?;
	let mut rng = SeqRng::new(0xDA7A_5EED);

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Seq2Seq Transformer · encoder-decoder (autograd)  ║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!("task: translate a symbol sequence into its reversal");
	println!(
		"vocab: {VOCAB_SIZE} (BOS + EOS + {} symbols)",
		VOCAB_SIZE - 2
	);
	println!("model: EncoderDecoder — bidirectional encoder, causal decoder with cross-attention");
	println!(
		"params: {}    Optimizer: AdamW(lr={lr})\n",
		model.num_parameters()?
	);
	println!(
		"training: {steps} steps · batch={batch} · source={SOURCE_LENGTH} · target={TARGET_LENGTH}"
	);

	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			batch_size: batch as u64,
			sequence_length: (SOURCE_LENGTH + TARGET_LENGTH) as u64,
			sequence_unit: "token".into(),
			timer_name: "seq2seq_transformer_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut metrics = LogMetrics::new();
	let log_dir = oa::Path::var_rel("log/nlp/seq2seq_transformer");
	if let Err(e) = metrics.open(&log_dir) {
		eprintln!("[OA] metrics unavailable: {e}");
	}

	let mut initial_loss = 0.0_f32;
	let mut last_source = Vec::new();
	let mut last_target = Vec::new();
	let mut last_labels = Vec::new();
	for _ in 0..steps {
		assert!(training.step(
			|| {
				let (source, target, labels) = next_batch(&mut rng, batch);
				last_source.clone_from(&source);
				last_target.clone_from(&target);
				last_labels.clone_from(&labels);
				Ok((
					oa::Matrix::new_from_slice([batch, SOURCE_LENGTH], &source)?,
					oa::Matrix::new_from_slice([batch, TARGET_LENGTH], &target)?,
					oa::Matrix::new_from_slice([batch * TARGET_LENGTH], &labels)?,
				))
			},
			|(source, target_tokens, labels)| {
				let tape = oa::ml::GradientTape::new();
				let logits = model.forward(&source, &target_tokens)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &labels)?;
				tape.backward(&loss)?;
				Ok(loss)
			},
		)?);
		let snap = training.snapshot();
		if snap.step_count() == 1 {
			initial_loss = snap.last_loss().unwrap_or_default();
		}
		if let Some(loss) = snap.last_loss() {
			metrics.log_scalar("loss", snap.step_count() as i64, loss as f64);
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let training_result = training.finish()?;
	let final_training_loss = training_result.last_loss().unwrap_or_default();

	let source = oa::Matrix::new_from_slice([batch, SOURCE_LENGTH], &last_source)?;
	let target_tokens = oa::Matrix::new_from_slice([batch, TARGET_LENGTH], &last_target)?;
	let labels = oa::Matrix::new_from_slice([batch * TARGET_LENGTH], &last_labels)?;
	let logits = model.forward(&source, &target_tokens)?;
	let evaluation_loss = oa::ml::loss::cross_entropy(&logits, &labels)?.read_f32()?[0];
	let eval_token_accuracy = token_accuracy(&logits, &last_labels)?;

	// Autoregressive evaluation: greedy-decode fresh sequences through
	// repeated uncached decode passes and report exact-match rate.
	let mut exact_matches = 0usize;
	for _ in 0..EVAL_SEQUENCES {
		let (eval_source, _, eval_labels) = next_batch(&mut rng, 1);
		let source_matrix = oa::Matrix::new_from_slice([1, SOURCE_LENGTH], &eval_source)?;
		let encoder_output = model.encode(&source_matrix)?;
		let mut decoded = vec![0_u32; TARGET_LENGTH];
		for position in 0..TARGET_LENGTH - 1 {
			let prefix = oa::Matrix::new_from_slice([1, TARGET_LENGTH], &decoded)?;
			let logits = model.decode(&prefix, &encoder_output)?;
			let row = logits
				.read_f32()?
				.as_chunks::<VOCAB_SIZE>()
				.0
				.get(position)
				.expect("decode logits cover the target context")
				.to_vec();
			let next = row
				.iter()
				.enumerate()
				.max_by(|left, right| left.1.total_cmp(right.1))
				.map(|(index, _)| index as u32)
				.expect("nonempty vocabulary logits");
			decoded[position + 1] = next;
			if next == EOS {
				break;
			}
		}
		if decoded[1..] == eval_labels[..TARGET_LENGTH - 1] {
			exact_matches += 1;
		}
	}
	let exact_match = exact_matches as f32 / EVAL_SEQUENCES as f32;

	println!("\nEvaluation:");
	println!(
		"  random-loss baseline ln({VOCAB_SIZE}) = {:.4}",
		(VOCAB_SIZE as f64).ln()
	);
	println!("  loss: evaluation {evaluation_loss:.6} · last training {final_training_loss:.6}");
	println!(
		"  teacher-forced token accuracy: {:.1}%",
		eval_token_accuracy * 100.0
	);
	println!(
		"  greedy exact-match: {exact_matches}/{EVAL_SEQUENCES} sequences ({:.1}%)",
		exact_match * 100.0
	);
	if let Some(diagnostics) = program_diagnostics {
		println!(
			"program: {} nodes · {} recording · {} cache hits · {} uploads",
			diagnostics.node_count(),
			diagnostics.command_recording_count(),
			diagnostics.command_cache_hit_count(),
			diagnostics.input_upload_count(),
		);
	}

	if steps == TRAINING_STEPS && batch == BATCH_SIZE && lr == 0.01 {
		assert!(initial_loss > 2.0);
		assert!(evaluation_loss < 0.15);
		assert!(eval_token_accuracy > 0.95);
		assert!(exact_match > 0.90);
	}

	let checkpoint = oa::Path::temp().join("oars_seq2seq_transformer.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let reloaded = oa::ml::nn::EncoderDecoder::new_with_seed(
		VOCAB_SIZE,
		VOCAB_SIZE,
		SOURCE_LENGTH,
		TARGET_LENGTH,
		32,
		64,
		2,
		2,
		1.0e-5,
		0x5_E2C_DE0,
	)?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, lr)?;
	oa::ml::load_checkpoint(&checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
	let reloaded_logits = reloaded.forward(&source, &target_tokens)?;
	assert_eq!(
		token_accuracy(&reloaded_logits, &last_labels)?,
		eval_token_accuracy,
	);
	oa::Filesystem::remove_file(&checkpoint).expect("remove completed seq2seq checkpoint");
	if oa::EnvFlag::is_set("OA_JIT_TRACE") {
		println!("fusion census: {}", engine.fusion_stats());
	}
	Ok(())
}

/// One `(source, decoder_input, labels)` batch: source is a random symbol
/// sequence, decoder input is `BOS + reversed(source) + EOS`, and labels are
/// `reversed(source) + EOS + BOS` for all-position cross-entropy.
fn next_batch(rng: &mut SeqRng, batch: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
	let mut source = Vec::with_capacity(batch * SOURCE_LENGTH);
	let mut target = Vec::with_capacity(batch * TARGET_LENGTH);
	let mut labels = Vec::with_capacity(batch * TARGET_LENGTH);
	for _ in 0..batch {
		let sequence = (0..SOURCE_LENGTH)
			.map(|_| 2 + rng.next_u32() % (VOCAB_SIZE as u32 - 2))
			.collect::<Vec<_>>();
		let mut reversed = sequence.clone();
		reversed.reverse();
		source.extend_from_slice(&sequence);
		target.push(0);
		target.extend_from_slice(&reversed);
		target.push(EOS);
		labels.extend_from_slice(&reversed);
		labels.push(EOS);
		labels.push(0);
	}
	(source, target, labels)
}

/// Greedy per-token accuracy of `[rows, vocab]` logits against flat labels.
fn token_accuracy(logits: &oa::Matrix, labels: &[u32]) -> oa::Result<f32> {
	let logits = logits.read_f32()?;
	let mut correct = 0usize;
	for (row, label) in logits.as_chunks::<VOCAB_SIZE>().0.iter().zip(labels.iter()) {
		let predicted = row
			.iter()
			.enumerate()
			.max_by(|left, right| left.1.total_cmp(right.1))
			.map(|(index, _)| index as u32)
			.expect("nonempty vocabulary logits");
		if predicted == *label {
			correct += 1;
		}
	}
	Ok(correct as f32 / labels.len() as f32)
}

/// Small deterministic generator — the batch stream must be reproducible for
/// the tutorial gate, not cryptographically strong.
struct SeqRng(u64);

impl SeqRng {
	const fn new(seed: u64) -> Self {
		Self(seed | 1)
	}

	fn next_u32(&mut self) -> u32 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		(self.0 >> 32) as u32
	}
}
