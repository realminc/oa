use oa::Color;
use oa::ml::{
	BpeTokenizer, ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary,
};
use oa::plot::{Figure, FigureConfig, LineStyle, Theme};
use sdk::ml::nlp;

#[derive(Default)]
struct Options {
	steps: usize,
	plot: String,
}

pub struct Tutorial<'text> {
	pub title: &'text str,
	pub description: &'text str,
	pub timer_name: &'text str,
	pub checkpoint_stem: &'text str,
	pub learning_rate: f32,
	pub expected_loss: f32,
	pub expected_accuracy: f32,
	pub expected_generation: &'text [u8],
}

pub fn run<M: Module>(
	tutorial: Tutorial<'_>,
	construct: impl Fn(&oa::Engine) -> oa::Result<M>,
) -> oa::Result<()> {
	let Tutorial {
		title,
		description,
		timer_name,
		checkpoint_stem,
		learning_rate,
		expected_loss,
		expected_accuracy,
		expected_generation,
	} = tutorial;
	let mut cli = oa::Cli::new("tu_nlp_bpe", title);
	cli.add_option(
		"--steps",
		|options: &mut Options| &mut options.steps,
		"training steps (default 300)",
	);
	cli.add_option(
		"--plot",
		|options: &mut Options| &mut options.plot,
		"render the loss curve to an image file",
	);
	if !cli.parse() {
		return Ok(());
	}
	let steps = if cli.config().steps == 0 {
		nlp::TRAINING_STEPS
	} else {
		cli.config().steps
	};
	let engine = oa::Engine::new()?;
	let mut tokenizer = BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	let corpus_tokens = tokenizer.encode_text(nlp::CORPUS);
	assert_eq!(tokenizer.decode_text(&corpus_tokens)?, nlp::CORPUS);
	let bytes_per_token = nlp::CORPUS.len() as f64 / corpus_tokens.len() as f64;
	let model = construct(&engine)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, learning_rate)?;
	let mut sampler = nlp::BpeSampler::new(nlp::BATCH_SIZE, &tokenizer)?;

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  {title:<62}║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!(
		"tokenizer: byte BPE · vocab={} (256 bytes + {} merges)",
		tokenizer.vocab_size(),
		tokenizer.num_merges()
	);
	println!(
		"compression: {} bytes → {} tokens · {:.3} byte/token · {:.1}% fewer positions",
		nlp::CORPUS.len(),
		corpus_tokens.len(),
		bytes_per_token,
		100.0 * (1.0 - corpus_tokens.len() as f64 / nlp::CORPUS.len() as f64)
	);
	println!(
		"context coverage: {} BPE tokens ≈ {:.1} source bytes at corpus average",
		nlp::CONTEXT_LENGTH,
		nlp::CONTEXT_LENGTH as f64 * bytes_per_token
	);
	println!("Task: dense next-token prediction at every position\n");
	println!("Model: {description}");
	println!(
		"params: {}    Optimizer: AdamW(lr={learning_rate})\n",
		model.num_parameters()?
	);

	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			batch_size: nlp::BATCH_SIZE as u64,
			sequence_length: nlp::CONTEXT_LENGTH as u64,
			sequence_unit: "BPE token".into(),
			source_unit: "byte".into(),
			timer_name: timer_name.into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut initial_loss = 0.0_f32;
	let mut loss_curve = Vec::new();
	let mut last_input = Vec::new();
	let mut last_target = Vec::new();
	for _ in 0..steps {
		let source_bytes = u64::try_from(sampler.next_batch_bytes()?)
			.expect("usize source-byte count fits u64 on supported Rust targets");
		training.record_source_units(source_bytes);
		assert!(training.step(
			|| {
				let (input_values, target_values) = sampler.next_values()?;
				last_input.clone_from(&input_values);
				last_target.clone_from(&target_values);
				Ok((
					oa::Matrix::new_from_slice(
						&engine,
						[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
						&input_values,
					)?,
					oa::Matrix::new_from_slice(
						&engine,
						[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
						&target_values,
					)?,
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
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let result = training.finish()?;
	let final_training_loss = result.last_loss().unwrap_or_default();

	let input =
		oa::Matrix::new_from_slice(&engine, [nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH], &last_input)?;
	let target = oa::Matrix::new_from_slice(
		&engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&last_target,
	)?;
	let logits = model.forward(&input)?;
	let accuracy = nlp::accuracy(&logits, &target)?;
	let evaluation_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let generated = nlp::generate_bpe_greedy(
		&engine,
		&model,
		&tokenizer,
		nlp::GENERATION_PROMPT.as_bytes(),
		nlp::GENERATION_LENGTH,
	)?;

	println!("\nEvaluation:");
	println!(
		"  Random-loss baseline ln({}) = {:.4}",
		tokenizer.vocab_size(),
		(tokenizer.vocab_size() as f64).ln()
	);
	println!(
		"  final batch: {:.3} byte/token · {:.4} bits/byte",
		sampler.last_batch_bytes_per_token(),
		f64::from(final_training_loss) / std::f64::consts::LN_2 / sampler.last_batch_bytes_per_token()
	);
	println!("  BPE-token accuracy: {:.1}%", accuracy * 100.0);
	println!("  post-update evaluation loss: {evaluation_loss:.6}");
	println!(
		"\nGeneration:\n  prompt: {:?}\n  generated: {:?}",
		nlp::GENERATION_PROMPT,
		String::from_utf8_lossy(&generated)
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
		write_loss_curve(
			&engine,
			&loss_curve,
			title,
			tokenizer.vocab_size(),
			plot_path,
		)?;
		println!("plot: {plot_path}");
	}

	assert!(initial_loss > 5.0);
	assert!(final_training_loss < initial_loss);
	assert!(evaluation_loss.is_finite());
	// Exact canonical convergence applies to the complete 300-step recipe.
	if steps == nlp::TRAINING_STEPS {
		assert!((evaluation_loss - expected_loss).abs() < 0.001);
		assert!((accuracy - expected_accuracy).abs() < 0.001);
		assert_eq!(generated, expected_generation);
	}

	let base = oa::Path::temp().join(format!("{checkpoint_stem}-{}", std::process::id()));
	let checkpoint = base.with_extension("oam");
	let vocabulary = base.with_extension("bpe");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	tokenizer.save(&vocabulary)?;
	let mut reloaded_tokenizer = BpeTokenizer::new(256);
	reloaded_tokenizer.load(&vocabulary)?;
	let reloaded = construct(&engine)?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, learning_rate)?;
	oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&input)?, &target)?,
		accuracy
	);
	assert_eq!(
		nlp::generate_bpe_greedy(
			&engine,
			&reloaded,
			&reloaded_tokenizer,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?,
		generated
	);
	oa::Filesystem::remove_file(&checkpoint).expect("remove completed BPE checkpoint");
	oa::Filesystem::remove_file(&vocabulary).expect("remove completed BPE vocabulary");
	if oa::EnvFlag::is_set("OA_JIT_TRACE") {
		println!("fusion census: {}", engine.fusion_stats());
	}
	Ok(())
}

/// Render the recorded loss curve through `oa::plot` — the same headless GPU
/// composition `tu_viewer_video` uses for its poster asset.
fn write_loss_curve(
	engine: &oa::Engine,
	loss_curve: &[f32],
	title: &str,
	vocab_size: usize,
	path: &str,
) -> oa::Result<()> {
	let mut fig = Figure::new(FigureConfig {
		title: format!("{title} training"),
		width: 960,
		height: 540,
		theme: Theme::Dark,
		..Default::default()
	});
	fig.title(&format!("{title} — dense next-token LM"));
	let steps: Vec<f32> = (1..=loss_curve.len()).map(|s| s as f32).collect();
	let ax = fig.ax(0, 0);
	ax.title("training loss");
	ax.x_label("optimizer step");
	ax.y_label("cross entropy");
	ax.grid(true);
	ax.legend(true);
	let baseline = (vocab_size as f32).ln();
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
			label: format!("ln({vocab_size}) baseline"),
			width: 1.35,
			antialias_samples: 4,
			..Default::default()
		},
	);
	fig.save_to(engine, path)
}
