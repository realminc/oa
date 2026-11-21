use oa::ml::{ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary};
use oa::plot::{Figure, FigureConfig, LineStyle, Theme};
use oa::{Cli, Color, LogMetrics};
use sdk::ml::nlp;

#[derive(Default)]
struct NlpConfig {
	steps: u64,
	batch: u64,
	lr: f32,
	plot: String,
}

fn main() -> oa::Result<()> {
	let mut cli = Cli::new("tu_nlp_char_transformer", "OA Tutorial — Char Transformer");
	cli.add_option(
		"--steps",
		|c: &mut NlpConfig| &mut c.steps,
		"training steps",
	);
	cli.add_option("--batch", |c: &mut NlpConfig| &mut c.batch, "batch size");
	cli.add_option("--lr", |c: &mut NlpConfig| &mut c.lr, "learning rate");
	cli.add_option(
		"--plot",
		|c: &mut NlpConfig| &mut c.plot,
		"render the loss curve to an image file",
	);
	if !cli.parse() {
		return Ok(());
	}
	let steps = if cli.config().steps > 0 {
		cli.config().steps as usize
	} else {
		nlp::TRAINING_STEPS
	};
	let batch = if cli.config().batch > 0 {
		cli.config().batch as usize
	} else {
		nlp::BATCH_SIZE
	};
	let lr = if cli.config().lr > 0.0 {
		cli.config().lr
	} else {
		0.01
	};

	let engine = oa::Engine::new()?;
	oa::Engine::set_default(engine.clone());
	let model = nlp::CharTransformer::new()?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, lr)?;
	let mut sampler = nlp::CharSampler::new(batch)?;

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Char Transformer · all-position LM (autograd)     ║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!(
		"tokenizer: character · vocab={} (a-z + space)",
		nlp::CHAR_VOCAB_SIZE
	);
	println!("Task: dense next-character prediction at every position\n");
	println!("Model: token+position embedding → TransformerBlock → LayerNorm → Linear");
	println!(
		"params: {}    Optimizer: AdamW(lr={lr})\n",
		model.num_parameters()?
	);
	println!(
		"training: {steps} steps · batch={batch} · sequence={} character tokens",
		nlp::CONTEXT_LENGTH
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
			sequence_length: nlp::CONTEXT_LENGTH as u64,
			sequence_unit: "token".into(),
			timer_name: "char_transformer_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut metrics = LogMetrics::new();
	let log_dir = oa::Path::var_rel("log/nlp/char_transformer");
	if let Err(e) = metrics.open(&log_dir) {
		eprintln!("[OA] metrics unavailable: {e}");
	}

	let mut initial_loss = 0.0_f32;
	let mut loss_curve = Vec::new();
	let mut last_input = Vec::new();
	let mut last_target = Vec::new();
	for _ in 0..steps {
		assert!(training.step(
			|| {
				let (input_values, target_values) = sampler.next_values()?;
				last_input.clone_from(&input_values);
				last_target.clone_from(&target_values);
				Ok((
					oa::Matrix::new_from_slice([batch, nlp::CONTEXT_LENGTH], &input_values)?,
					oa::Matrix::new_from_slice([batch * nlp::CONTEXT_LENGTH], &target_values)?,
				))
			},
			|(input, target)| {
				let tape = oa::ml::GradientTape::new();
				let logits = model.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
				tape.backward(&loss)?;
				Ok(loss)
			},
		)?);
		let snap = training.snapshot();
		if snap.step_count() == 1 {
			initial_loss = snap.last_loss().unwrap_or_default();
		}
		if let Some(loss) = snap.last_loss() {
			loss_curve.push(loss);
			metrics.log_scalar("loss", snap.step_count() as i64, loss as f64);
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let training_result = training.finish()?;
	let final_training_loss = training_result.last_loss().unwrap_or_default();

	let input = oa::Matrix::new_from_slice([batch, nlp::CONTEXT_LENGTH], &last_input)?;
	let target = oa::Matrix::new_from_slice([batch * nlp::CONTEXT_LENGTH], &last_target)?;
	let logits = model.forward(&input)?;
	let evaluation_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&logits, &target)?;
	let generated = nlp::generate_greedy(
		&engine,
		&model,
		nlp::GENERATION_PROMPT,
		nlp::GENERATION_LENGTH,
	)?;

	println!("\nEvaluation:");
	println!(
		"  Random-loss baseline ln({}) = {:.4}",
		nlp::CHAR_VOCAB_SIZE,
		(nlp::CHAR_VOCAB_SIZE as f64).ln()
	);
	println!("  loss: evaluation {evaluation_loss:.6} · last training {final_training_loss:.6}");
	println!(
		"  character-token accuracy: {:.1}% (compare within Char only)",
		accuracy * 100.0
	);
	println!(
		"\nGeneration:\n  prompt: {:?}\n  generated: {:?}",
		nlp::GENERATION_PROMPT,
		generated
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

	let plot_path = cli.config().plot.as_str();
	if !plot_path.is_empty() {
		write_loss_curve(&loss_curve, "Char Transformer", plot_path)?;
		println!("plot: {plot_path}");
	}

	if steps == nlp::TRAINING_STEPS && batch == nlp::BATCH_SIZE && lr == 0.01 {
		assert!(initial_loss > 3.0);
		assert!((evaluation_loss - nlp::CHAR_TRANSFORMER_CPP_FINAL_LOSS).abs() < 0.02);
		assert!((accuracy - nlp::CHAR_TRANSFORMER_CPP_ACCURACY).abs() < 0.01);
		assert_eq!(generated, nlp::CHAR_TRANSFORMER_REFERENCE_GENERATION);
	}

	let checkpoint = oa::Path::temp().join("oars_char_transformer.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let reloaded = nlp::CharTransformer::new()?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, lr)?;
	oa::ml::load_checkpoint(&checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&input)?, &target)?,
		accuracy
	);
	assert_eq!(
		nlp::generate_greedy(
			&engine,
			&reloaded,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH
		)?,
		generated
	);
	oa::Filesystem::remove_file(&checkpoint).expect("remove completed Transformer checkpoint");
	if oa::EnvFlag::is_set("OA_JIT_TRACE") {
		println!("fusion census: {}", engine.fusion_stats());
	}
	Ok(())
}

/// Render the recorded loss curve through `oa::plot` — the same headless GPU
/// composition `tu_viewer_video` uses for its poster asset.
fn write_loss_curve(loss_curve: &[f32], title: &str, path: &str) -> oa::Result<()> {
	let mut fig = Figure::new(FigureConfig {
		title: format!("{title} training"),
		width: 960,
		height: 540,
		theme: Theme::Dark,
		..Default::default()
	});
	fig.title(&format!("{title} — dense next-character LM"));
	let steps: Vec<f32> = (1..=loss_curve.len()).map(|s| s as f32).collect();
	let ax = fig.ax(0, 0);
	ax.title("training loss");
	ax.x_label("optimizer step");
	ax.y_label("cross entropy");
	ax.grid(true);
	ax.legend(true);
	let baseline = (nlp::CHAR_VOCAB_SIZE as f32).ln();
	let peak_loss = loss_curve.iter().copied().fold(baseline, f32::max);
	ax.limits(0.0, loss_curve.len() as f32 + 1.0, 0.0, peak_loss + 0.3);
	ax.plot_xy(
		&steps,
		loss_curve,
		LineStyle {
			color: Color::new(0.24, 0.65, 1.0, 1.0),
			label: "train".to_owned(),
			width: 1.6,
			antialias_samples: 4,
			..Default::default()
		},
	);
	ax.plot_xy(
		&[
			*steps.first().unwrap_or(&0.0),
			*steps.last().unwrap_or(&1.0),
		],
		&[baseline, baseline],
		LineStyle {
			color: Color::new(0.565, 0.565, 0.565, 1.0),
			label: format!("ln({}) baseline", nlp::CHAR_VOCAB_SIZE),
			width: 1.35,
			antialias_samples: 4,
			..Default::default()
		},
	);
	fig.save_to(path)
}
