use std::time::Duration;

use oa::ml::Module;
use oa::sdk::ml::nlp;

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

/// Interleaved A/B sampling of two plans in one process. On this
/// class of integrated device, run-to-run clock drift is comparable
/// to the per-dispatch deltas being measured, so fused and unfused
/// samples must be collected alternately rather than in separate
/// processes.
fn interleaved_medians(
	engine: &oa::Engine,
	candidate: &oa::ExecutionPlan,
	reference: &oa::ExecutionPlan,
) -> oa::Result<(Duration, Duration)> {
	for _ in 0..WARMUP_REPLAYS {
		engine.submit(candidate)?.wait()?;
		engine.submit(reference)?.wait()?;
	}
	let mut candidate_samples = Vec::with_capacity(MEASURED_REPLAYS);
	let mut reference_samples = Vec::with_capacity(MEASURED_REPLAYS);
	for _ in 0..MEASURED_REPLAYS {
		candidate_samples.push(engine.submit_timed(candidate)?.device_duration()?);
		reference_samples.push(engine.submit_timed(reference)?.device_duration()?);
	}
	candidate_samples.sort_unstable();
	reference_samples.sort_unstable();
	Ok((
		candidate_samples[MEASURED_REPLAYS / 2],
		reference_samples[MEASURED_REPLAYS / 2],
	))
}

fn milliseconds(duration: Duration) -> f64 {
	duration.as_secs_f64() * 1_000.0
}

/// Executable kernel histogram of a captured plan, most frequent first.
///
/// Uses the handle-free `oa.execution_graph` report; kernel names are
/// diagnostic routing evidence, not a public API contract.
fn kernel_census(plan: &oa::ExecutionPlan, name: &str) -> String {
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json(name)).unwrap_or_default();
	let mut counts = std::collections::BTreeMap::<String, usize>::new();
	for node in report["nodes"].as_array().into_iter().flatten() {
		let kernel = node["kernel"].as_str().unwrap_or("unknown");
		*counts.entry(kernel.to_owned()).or_default() += 1;
	}
	let mut counts = counts.into_iter().collect::<Vec<_>>();
	counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
	counts
		.iter()
		.map(|(kernel, count)| format!("{kernel}={count}"))
		.collect::<Vec<_>>()
		.join(" ")
}

fn profile_linear(
	engine: &oa::Engine,
	batch: usize,
	input_features: usize,
	output_features: usize,
) -> oa::Result<Duration> {
	let input = oa::matrix::full(engine, [batch, input_features], 0.125)?;
	let target = oa::Matrix::new_from_slice(engine, [batch], &vec![0_u32; batch])?;
	let linear = oa::ml::nn::Linear::new_with_seed(engine, input_features, output_features, 1)?;
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
	let target = oa::Matrix::new_from_slice(engine, [rows], &vec![0_u32; rows])?;
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
	let target = oa::Matrix::new_from_slice(engine, [rows], &vec![0_u32; rows])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits =
			oa::ml::matrix::scaled_dot_product_attention_causal(&query, &key, &value, sequence, heads)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	median_device_time(engine, &plan)
}

fn profile_sdpa_provider(
	engine: &oa::Engine,
	batch_heads: usize,
	sequence: usize,
	head_dim: usize,
	flash: bool,
) -> oa::Result<Duration> {
	let shape = [batch_heads, sequence, head_dim];
	let query = oa::matrix::full(engine, shape, 0.125)?;
	let key = oa::matrix::full(engine, shape, -0.25)?;
	let value = oa::matrix::full(engine, shape, 0.375)?;
	let target = oa::matrix::full(engine, shape, 0.0)?;
	let scale = 1.0 / (head_dim as f32).sqrt();
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let output = if flash {
			oa::ml::matrix::flash_attention_causal(&query, &key, &value, scale)?
		} else {
			oa::ml::matrix::scaled_dot_product_attention(&query, &key, &value, None, scale, true)?
		};
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	median_device_time(engine, &plan)
}

fn profile_sdpa_forward_provider(
	engine: &oa::Engine,
	batch_heads: usize,
	sequence: usize,
	head_dim: usize,
	flash: bool,
) -> oa::Result<Duration> {
	let shape = [batch_heads, sequence, head_dim];
	let query = oa::matrix::full(engine, shape, 0.125)?;
	let key = oa::matrix::full(engine, shape, -0.25)?;
	let value = oa::matrix::full(engine, shape, 0.375)?;
	let scale = 1.0 / (head_dim as f32).sqrt();
	let (plan, _) = engine.capture(|| {
		if flash {
			oa::ml::matrix::flash_attention_causal(&query, &key, &value, scale)
		} else {
			oa::ml::matrix::scaled_dot_product_attention(&query, &key, &value, None, scale, true)
		}
	})?;
	median_device_time(engine, &plan)
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = nlp::CharTransformer::new(&engine)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	let target = target.reshape([nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH])?;

	// backward accumulates into `Parameter::gradient` when one exists, so
	// every captured plan must see cleared gradients to record the same
	// semantics — otherwise the second capture gains a `matrix.add` per
	// parameter that the first never pays.
	let grad_reset = oa::ml::Sgd::new(model.all_parameters()?, 0.0, 0.0, 0.0)?;
	// The reference forces structural fusion off; the candidate keeps the
	// caller's route, so a suite run that disables fusion measures it.
	let caller_route = std::env::var_os("OA_DISABLE_EPILOGUE_FUSION");
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let (reference_forward, reference_logits) = engine.capture(|| model.forward(&input))?;
	engine.submit(&reference_forward)?.wait()?;
	grad_reset.zero_grad();
	let (reference_training, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	engine.submit(&reference_training)?.wait()?;
	match caller_route {
		Some(value) => unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", value) },
		None => unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") },
	}
	let (forward_plan, logits) = engine.capture(|| model.forward(&input))?;
	engine.submit(&forward_plan)?.wait()?;
	grad_reset.zero_grad();
	let (training_plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	engine.submit(&training_plan)?.wait()?;

	let loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let reference_loss = oa::ml::loss::cross_entropy(&reference_logits, &target)?.read_f32()?[0];

	let (forward, reference_forward_time) =
		interleaved_medians(&engine, &forward_plan, &reference_forward)?;
	let (forward_backward, reference_forward_backward) =
		interleaved_medians(&engine, &training_plan, &reference_training)?;
	if oa::EnvFlag::is_set("OA_JIT_TRACE") {
		println!(
			"kernel census forward: {}",
			kernel_census(&forward_plan, "forward")
		);
		println!(
			"kernel census forward+backward: {}",
			kernel_census(&training_plan, "forward+backward")
		);
	}
	if let Some(path) = std::env::var_os("OA_GRAPH_REPORT") {
		std::fs::write(&path, training_plan.debug_report_json("forward+backward"))
			.expect("graph report write must succeed");
	}

	println!("exploratory fixed-graph profile; interleaved A/B in one process");
	println!("loss fused {loss:.6} · unfused {reference_loss:.6}");
	println!(
		"forward GPU median: fused {:.3} ms · unfused {:.3} ms",
		milliseconds(forward),
		milliseconds(reference_forward_time)
	);
	println!(
		"forward + backward GPU median: fused {:.3} ms · unfused {:.3} ms",
		milliseconds(forward_backward),
		milliseconds(reference_forward_backward)
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
	let standard = profile_sdpa_provider(&engine, 64, 16, 32, false)?;
	let flash = profile_sdpa_provider(&engine, 64, 16, 32, true)?;
	let standard_forward = profile_sdpa_forward_provider(&engine, 64, 16, 32, false)?;
	let flash_forward = profile_sdpa_forward_provider(&engine, 64, 16, 32, true)?;
	println!(
		"SDPA [64,16,32] standard forward GPU: {:.3} ms",
		milliseconds(standard_forward)
	);
	println!(
		"SDPA [64,16,32] Flash forward GPU: {:.3} ms",
		milliseconds(flash_forward)
	);
	println!(
		"SDPA [64,16,32] Flash forward gain: {:.2}%",
		(1.0 - flash_forward.as_secs_f64() / standard_forward.as_secs_f64()) * 100.0
	);
	println!(
		"SDPA [64,16,32] standard forward/backward GPU: {:.3} ms",
		milliseconds(standard)
	);
	println!(
		"SDPA [64,16,32] Flash forward/backward GPU: {:.3} ms",
		milliseconds(flash)
	);
	println!(
		"SDPA [64,16,32] Flash gain: {:.2}%",
		(1.0 - flash.as_secs_f64() / standard.as_secs_f64()) * 100.0
	);
	println!("fusion census: {}", engine.fusion_stats());
	Ok(())
}
