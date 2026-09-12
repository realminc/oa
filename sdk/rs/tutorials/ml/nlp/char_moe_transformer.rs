use oa::ml::{ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary, nlp};

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = nlp::CharMoeTransformer::new(&engine)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Char MoE Transformer · all-position LM           ║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!(
		"tokenizer: character · vocab={} (a-z + space)",
		nlp::CHAR_VOCAB_SIZE
	);
	println!("Task: dense next-character prediction at every position\n");
	println!(
		"Model: token+position embedding → attention → sparse MoE(E={}, K={}, DFF={}) → LayerNorm → Linear",
		nlp::MOE_NUM_EXPERTS,
		nlp::MOE_EXPERTS_PER_TOKEN,
		nlp::MOE_EXPERT_HIDDEN_WIDTH
	);
	println!(
		"params: {}    Optimizer: AdamW(lr=0.01)\n",
		model.num_parameters()?
	);
	println!(
		"training: {} steps · batch={} · sequence={} character tokens",
		nlp::TRAINING_STEPS,
		nlp::BATCH_SIZE,
		nlp::CONTEXT_LENGTH
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
			sequence_unit: "token".into(),
			timer_name: "char_moe_transformer_training_step".into(),
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
	while training.step(
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
	)? {
		if training.snapshot().step_count() == 1 {
			initial_loss = training.snapshot().last_loss().unwrap_or_default();
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let training_result = training.finish()?;
	let final_training_loss = training_result.last_loss().unwrap_or_default();

	let input =
		oa::Matrix::from_slice(&engine, [nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH], &last_input)?;
	let target = oa::Matrix::from_slice(
		&engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&last_target,
	)?;
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
	println!("  character-token accuracy: {:.1}%", accuracy * 100.0);
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

	assert!(initial_loss > 3.0);
	assert!((evaluation_loss - nlp::CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS).abs() < 0.02);
	assert!((accuracy - nlp::CHAR_MOE_TRANSFORMER_CPP_ACCURACY).abs() < 0.01);
	assert_eq!(generated, nlp::CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION);

	let checkpoint = std::env::temp_dir().join("oars_char_moe_transformer.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let reloaded = nlp::CharMoeTransformer::new(&engine)?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
	oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
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
	std::fs::remove_file(checkpoint).expect("remove completed MoE Transformer checkpoint");
	Ok(())
}
