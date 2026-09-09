use std::time::Instant;

use oa::ml::{Module, nlp};

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = nlp::CharTransformer::new(&engine)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (initial_input, initial_target) = sampler.next_values()?;
	let input = oa::Matrix::from_slice(
		&engine,
		[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
		&initial_input,
	)?;
	let target = oa::Matrix::from_slice(
		&engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&initial_target,
	)?;
	let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let mut initial_loss = 0.0_f32;
	let mut final_loss = 0.0_f32;
	let started = Instant::now();

	for step in 0..nlp::TRAINING_STEPS {
		if step != 0 {
			let (next_input, next_target) = sampler.next_values()?;
			program.upload_input(&input, &next_input)?;
			program.upload_input(&target, &next_target)?;
		}
		final_loss = program.replay_and_wait(&engine, &mut optimizer)?;
		if step == 0 {
			initial_loss = final_loss;
		}
		if (step + 1) % 50 == 0 || step == 0 {
			println!("step {:>3}/300 loss {:.6}", step + 1, final_loss);
		}
	}
	let elapsed = started.elapsed();
	let evaluation_loss =
		oa::ml::loss::cross_entropy(&model.forward(&input)?, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&model.forward(&input)?, &target)?;
	let generated = nlp::generate_greedy(
		&engine,
		&model,
		nlp::GENERATION_PROMPT,
		nlp::GENERATION_LENGTH,
	)?;

	println!(
		"loss: initial {initial_loss:.6} final {evaluation_loss:.6} (last training {final_loss:.6})"
	);
	println!("accuracy: {:.1}%", accuracy * 100.0);
	println!(
		"wall: {:.3} ms/step",
		1000.0 * elapsed.as_secs_f64() / nlp::TRAINING_STEPS as f64
	);
	println!("prompt: {:?}", nlp::GENERATION_PROMPT);
	println!("generated: {generated:?}");
	let diagnostics = program.diagnostics();
	println!(
		"program: {} nodes · {} recording · {} cache hits · {} uploads",
		diagnostics.node_count(),
		diagnostics.command_recording_count(),
		diagnostics.command_cache_hit_count(),
		diagnostics.input_upload_count(),
	);
	assert!(initial_loss > 3.0);
	assert!((evaluation_loss - nlp::CHAR_TRANSFORMER_CPP_FINAL_LOSS).abs() < 0.02);
	assert!((accuracy - nlp::CHAR_TRANSFORMER_CPP_ACCURACY).abs() < 0.01);
	assert_eq!(generated, nlp::CHAR_TRANSFORMER_REFERENCE_GENERATION);
	let checkpoint = std::env::temp_dir().join("oars_char_transformer.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let reloaded = nlp::CharTransformer::new(&engine)?;
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
	std::fs::remove_file(checkpoint).expect("remove completed Transformer checkpoint");
	Ok(())
}
