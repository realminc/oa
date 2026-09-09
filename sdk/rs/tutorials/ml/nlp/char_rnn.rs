use std::time::Instant;

use oa::ml::{Module, nlp};

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = nlp::CharRnn::new(&engine)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let mut initial_loss = 0.0_f32;
	let mut final_loss = 0.0_f32;
	let mut final_batch = None;
	let started = Instant::now();

	for step in 0..nlp::TRAINING_STEPS {
		let (input, target) = sampler.next(&engine)?;
		optimizer.zero_grad();
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let flat_target = target.reshape([nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH])?;
		let loss = oa::ml::loss::cross_entropy(&logits, &flat_target)?;
		tape.backward(&loss)?;
		final_loss = loss.read_f32()?[0];
		if step == 0 {
			initial_loss = final_loss;
		}
		optimizer.step()?;
		if (step + 1) % 50 == 0 || step == 0 {
			println!("step {:>3}/300 loss {:.6}", step + 1, final_loss);
		}
		final_batch = Some((input, flat_target));
	}
	let elapsed = started.elapsed();
	let (input, target) = final_batch.expect("the fixed 300-step loop is nonempty");
	let accuracy = nlp::accuracy(&model.forward(&input)?, &target)?;
	let generated = nlp::generate_greedy(
		&engine,
		&model,
		nlp::GENERATION_PROMPT,
		nlp::GENERATION_LENGTH,
	)?;

	println!("loss: initial {initial_loss:.6} final {final_loss:.6}");
	println!("accuracy: {:.1}%", accuracy * 100.0);
	println!(
		"wall: {:.3} ms/step",
		1000.0 * elapsed.as_secs_f64() / nlp::TRAINING_STEPS as f64
	);
	println!("prompt: {:?}", nlp::GENERATION_PROMPT);
	println!("generated: {generated:?}");

	assert!(initial_loss > 3.0);
	assert!(final_loss < 0.25);
	assert!(accuracy > 0.9);
	assert_eq!(generated, nlp::CHAR_RNN_REFERENCE_GENERATION);
	Ok(())
}
