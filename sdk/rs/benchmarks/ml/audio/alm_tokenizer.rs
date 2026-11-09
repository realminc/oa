use std::time::Duration;

use oa::ml::Module;
use oa::sdk::ml::alm::{AlmTokenizer, AlmTokenizerConfig};

const WARMUP_REPLAYS: usize = 5;
const MEASURED_REPLAYS: usize = 15;

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

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = AlmTokenizer::new_with_seed(
		&engine,
		AlmTokenizerConfig {
			input_dim: 32,
			width: 64,
			code_dim: 32,
			num_codes: 128,
			downsample_stages: 2,
			depth: 2,
			commitment_beta: 0.25,
			ema_decay: 0.99,
			ema_epsilon: 1e-5,
			dead_threshold: 2.0,
		},
		0xC0FFEE,
	)?;
	// `[batch, frames, input_dim]` motion features; the encode stage
	// transposes to NCL before the Conv1d stack.
	let input = oa::Matrix::new_from_f32(
		&engine,
		[4, 64, 32],
		&vec![0.02; 4 * 64 * 32]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [4, 64, 32], &vec![0.0; 4 * 64 * 32])?;

	// backward accumulates into `Parameter::gradient` when one exists, so
	// every captured plan must see cleared gradients to record the same
	// semantics — otherwise the second capture gains a `matrix.add` per
	// parameter that the first never pays.
	let grad_reset = oa::ml::Sgd::new(model.all_parameters()?, 0.0, 0.0, 0.0)?;
	// The reference forces structural fusion off; the candidate keeps the
	// caller's route, so a suite run that disables fusion measures it.
	let caller_route = std::env::var_os("OA_DISABLE_EPILOGUE_FUSION");
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let (reference_forward, reference_reconstruction) = engine.capture(|| model.forward(&input))?;
	engine.submit(&reference_forward)?.wait()?;
	grad_reset.zero_grad();
	let (reference_training, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let reconstruction = model.forward(&input)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	engine.submit(&reference_training)?.wait()?;
	match caller_route {
		Some(value) => unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", value) },
		None => unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") },
	}
	let (forward_plan, reconstruction) = engine.capture(|| model.forward(&input))?;
	engine.submit(&forward_plan)?.wait()?;
	grad_reset.zero_grad();
	let (training_plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let reconstruction = model.forward(&input)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	engine.submit(&training_plan)?.wait()?;

	let loss = oa::ml::loss::mse(&reconstruction, &target)?.read_f32()?[0];
	let reference_loss = oa::ml::loss::mse(&reference_reconstruction, &target)?.read_f32()?[0];

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
		std::fs::write(
			std::path::Path::new(&path).with_extension("unfused.json"),
			reference_training.debug_report_json("forward+backward"),
		)
		.expect("unfused graph report write must succeed");
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
	println!("fusion census: {}", engine.fusion_stats());
	Ok(())
}
