use oa::ml::{
	BpeTokenizer, ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary,
};
use oa::sdk::ml::nlp;

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
			total_steps: nlp::TRAINING_STEPS as u64,
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
	let mut last_input = Vec::new();
	let mut last_target = Vec::new();
	for _ in 0..nlp::TRAINING_STEPS {
		let source_bytes = u64::try_from(sampler.next_batch_bytes()?)
			.expect("usize source-byte count fits u64 on supported Rust targets");
		training.record_source_units(source_bytes);
		assert!(training.step(
			|| {
				let (input_values, target_values) = sampler.next_values()?;
				last_input.clone_from(&input_values);
				last_target.clone_from(&target_values);
				Ok((
					oa::Matrix::from_slice(
						&engine,
						[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
						&input_values,
					)?,
					oa::Matrix::from_slice(
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
		if training.snapshot().step_count() == 1 {
			initial_loss = training.snapshot().last_loss().unwrap_or_default();
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let result = training.finish()?;
	let final_training_loss = result.last_loss().unwrap_or_default();

	let input = oa::Matrix::from_slice(&engine, [nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH], &last_input)?;
	let target = oa::Matrix::from_slice(
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

	assert!(initial_loss > 5.0);
	assert!(final_training_loss < initial_loss);
	assert!((evaluation_loss - expected_loss).abs() < 0.001);
	assert!((accuracy - expected_accuracy).abs() < 0.001);
	assert_eq!(generated, expected_generation);

	let base = oa::Path::temp().join(checkpoint_stem);
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
	Ok(())
}
