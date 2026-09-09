use std::time::Duration;

use oa::ml::nlp;

const WARMUP_REPLAYS: usize = 5;
const MEASURED_REPLAYS: usize = 15;

fn median_device_time(engine: &oa::Engine, plan: &oa::ExecutionPlan) -> oa::Result<Duration> {
	for _ in 0..WARMUP_REPLAYS {
		engine.submit(plan)?.wait()?;
	}
	let mut samples = Vec::with_capacity(MEASURED_REPLAYS);
	for _ in 0..MEASURED_REPLAYS {
		samples.push(engine.submit_timed(plan)?.device_duration()?);
	}
	samples.sort_unstable();
	Ok(samples[MEASURED_REPLAYS / 2])
}

fn milliseconds(duration: Duration) -> f64 {
	duration.as_secs_f64() * 1_000.0
}

fn profile_linear(
	engine: &oa::Engine,
	batch: usize,
	input_features: usize,
	output_features: usize,
) -> oa::Result<Duration> {
	let input = oa::matrix::full(engine, [batch, input_features], 0.125)?;
	let target = oa::Matrix::from_slice(engine, [batch], &vec![0_u32; batch])?;
	let linear = oa::ml::nn::Linear::with_seed(engine, input_features, output_features, 1)?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = linear.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	median_device_time(engine, &plan)
}

fn profile_layer_norm(engine: &oa::Engine, rows: usize, columns: usize) -> oa::Result<Duration> {
	let input = oa::matrix::full(engine, [rows, columns], 0.125)?;
	let target = oa::Matrix::from_slice(engine, [rows], &vec![0_u32; rows])?;
	let layer_norm = oa::ml::nn::LayerNorm::new(engine, columns, 1.0e-5)?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = layer_norm.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	median_device_time(engine, &plan)
}

fn profile_attention(
	engine: &oa::Engine,
	batch: usize,
	sequence: usize,
	width: usize,
	heads: usize,
) -> oa::Result<Duration> {
	let rows = batch * sequence;
	let query = oa::matrix::full(engine, [rows, width], 0.125)?;
	let key = oa::matrix::full(engine, [rows, width], -0.25)?;
	let value = oa::matrix::full(engine, [rows, width], 0.375)?;
	let target = oa::Matrix::from_slice(engine, [rows], &vec![0_u32; rows])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits =
			oa::ml::scaled_dot_product_attention_causal(&query, &key, &value, sequence, heads)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	median_device_time(engine, &plan)
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = nlp::CharTransformer::new(&engine)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	let target = target.reshape([nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH])?;

	let (forward_plan, logits) = engine.capture(|| model.forward(&input))?;
	let forward = median_device_time(&engine, &forward_plan)?;
	let loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];

	let (training_plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let forward_backward = median_device_time(&engine, &training_plan)?;

	println!("exploratory fixed-graph profile; loss {loss:.6}");
	println!("forward GPU median: {:.3} ms", milliseconds(forward));
	println!(
		"forward + backward GPU median: {:.3} ms",
		milliseconds(forward_backward)
	);
	println!(
		"inferred backward GPU median: {:.3} ms",
		milliseconds(forward_backward.saturating_sub(forward))
	);
	for (input_features, output_features) in [(32, 32), (32, 64), (64, 32), (32, 27)] {
		let duration = profile_linear(&engine, 1024, input_features, output_features)?;
		println!(
			"Linear 1024x{input_features} -> {output_features} forward/backward GPU: {:.3} ms",
			milliseconds(duration)
		);
	}
	println!(
		"LayerNorm 1024x32 forward/backward GPU: {:.3} ms",
		milliseconds(profile_layer_norm(&engine, 1024, 32)?)
	);
	println!(
		"attention 64x16x32x1 forward/backward GPU: {:.3} ms",
		milliseconds(profile_attention(&engine, 64, 16, 32, 1)?)
	);
	Ok(())
}
