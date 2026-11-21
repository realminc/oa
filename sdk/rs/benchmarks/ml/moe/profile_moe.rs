//! MoE forward + backward fixed-graph profile at the Trade PPO update shape.
//!
//! Captures `nn::Moe` (width 64, hidden 128, 4 experts, top-k 1) under a
//! gradient tape and reports interleaved GPU medians for three plans:
//! `fused` (caller route), `rl_off` (MoE epilogue rules suppressed via
//! `OA_DISABLE_RL_FUSION`, other fusion intact) and `unfused` (structural
//! epilogue fusion off). `OA_DISABLE_FUSION_RULES=<csv>` additionally
//! suppresses named rules in the fused plan for cross-process bisection.
//! `--tokens` rescales the token count.
//!
//! `OA_GRAPH_REPORT=<path>` writes the fused training report and
//! `OA_JIT_TRACE=1` prints the executable kernel census.

use std::time::Duration;

use oa::ml::Module;

fn env_usize(name: &str, default: usize) -> usize {
	std::env::var(name)
		.ok()
		.and_then(|value| value.parse().ok())
		.unwrap_or(default)
}

/// i915 DVFS needs sustained work to ramp; a handful of short replays
/// measures the idle clock rather than the device. Warmup and sample
/// counts stay env-tunable so qualification runs can raise both.
fn warmup_replays() -> usize {
	env_usize("OA_PROFILE_WARMUP", 30)
}

fn measured_replays() -> usize {
	env_usize("OA_PROFILE_SAMPLES", 60)
}

/// Round-robin A/B/C sampling of three plans in one process. On this class
/// of integrated device, run-to-run clock drift is comparable to the
/// per-dispatch deltas being measured, so route medians must come from
/// interleaved samples rather than separate processes.
fn interleaved_medians(
	engine: &oa::Engine,
	plans: &[&oa::ExecutionPlan],
) -> oa::Result<Vec<Vec<Duration>>> {
	for _ in 0..warmup_replays() {
		for plan in plans {
			engine.submit(plan)?.wait()?;
		}
	}
	let samples = measured_replays();
	let mut durations = vec![Vec::with_capacity(samples); plans.len()];
	for _ in 0..samples {
		for (index, plan) in plans.iter().enumerate() {
			durations[index].push(engine.submit_timed(plan)?.device_duration()?);
		}
	}
	for samples in &mut durations {
		samples.sort_unstable();
	}
	Ok(durations)
}

fn summary(samples: &[Duration]) -> String {
	let count = samples.len();
	format!(
		"min {:.3} p25 {:.3} p50 {:.3} p75 {:.3} max {:.3}",
		milliseconds(samples[0]),
		milliseconds(samples[count / 4]),
		milliseconds(samples[count / 2]),
		milliseconds(samples[count * 3 / 4]),
		milliseconds(samples[count - 1])
	)
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
	oa::Engine::set_default(engine.clone());
	let tokens = env_usize("OA_MOE_PROFILE_TOKENS", 4096);
	let model = oa::ml::nn::Moe::new_with_seed(64, 128, 4, 1, 1e-5, 0x0a11_ce55)?;
	let input = oa::Matrix::new_from_f32(
		[tokens, 64],
		&(0..tokens * 64)
			.map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.05)
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32([tokens, 64], &vec![0.0; tokens * 64])?;

	// backward accumulates into `Parameter::gradient` when one exists, so
	// every captured plan must see cleared gradients to record the same
	// semantics — otherwise the second capture gains a `matrix.add` per
	// parameter that the first never pays.
	let grad_reset = oa::ml::Sgd::new(model.all_parameters()?, 0.0, 0.0, 0.0)?;
	let training = |input: &oa::Matrix| -> oa::Result<oa::Matrix> {
		let tape = oa::ml::GradientTape::new();
		let output = model.forward(input)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	};

	// `OA_DISABLE_EPILOGUE_FUSION` and `OA_DISABLE_RL_FUSION` are live env
	// reads, so each capture sees the route its toggle selected.
	let caller_epilogue = std::env::var_os("OA_DISABLE_EPILOGUE_FUSION");
	let caller_rl = std::env::var_os("OA_DISABLE_RL_FUSION");
	unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", "1") };
	let (unfused_plan, unfused_output) = engine.capture(|| model.forward(&input))?;
	engine.submit(&unfused_plan)?.wait()?;
	grad_reset.zero_grad();
	let (unfused_training, _) = engine.capture(|| training(&input))?;
	engine.submit(&unfused_training)?.wait()?;
	unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") };
	unsafe { std::env::set_var("OA_DISABLE_RL_FUSION", "1") };
	let (rl_forward, _) = engine.capture(|| model.forward(&input))?;
	engine.submit(&rl_forward)?.wait()?;
	grad_reset.zero_grad();
	let (rl_training, _) = engine.capture(|| training(&input))?;
	engine.submit(&rl_training)?.wait()?;
	match caller_rl {
		Some(value) => unsafe { std::env::set_var("OA_DISABLE_RL_FUSION", value) },
		None => unsafe { std::env::remove_var("OA_DISABLE_RL_FUSION") },
	}
	match caller_epilogue {
		Some(value) => unsafe { std::env::set_var("OA_DISABLE_EPILOGUE_FUSION", value) },
		None => unsafe { std::env::remove_var("OA_DISABLE_EPILOGUE_FUSION") },
	}
	let (forward_plan, output) = engine.capture(|| model.forward(&input))?;
	engine.submit(&forward_plan)?.wait()?;
	grad_reset.zero_grad();
	let (training_plan, _) = engine.capture(|| training(&input))?;
	engine.submit(&training_plan)?.wait()?;

	let loss = oa::ml::loss::mse(&output, &target)?.read_f32()?[0];
	let reference_loss = oa::ml::loss::mse(&unfused_output, &target)?.read_f32()?[0];

	let forward_times = interleaved_medians(&engine, &[&forward_plan, &rl_forward, &unfused_plan])?;
	let training_times =
		interleaved_medians(&engine, &[&training_plan, &rl_training, &unfused_training])?;

	println!("exploratory fixed-graph profile; interleaved A/B/C in one process");
	println!("tokens {tokens} loss fused {loss:.6} · unfused {reference_loss:.6}");
	println!("forward fused    : {}", summary(&forward_times[0]));
	println!("forward rl_off   : {}", summary(&forward_times[1]));
	println!("forward unfused  : {}", summary(&forward_times[2]));
	println!("fwd+bwd fused    : {}", summary(&training_times[0]));
	println!("fwd+bwd rl_off   : {}", summary(&training_times[1]));
	println!("fwd+bwd unfused  : {}", summary(&training_times[2]));
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
	println!("fusion census: {}", engine.fusion_stats());
	Ok(())
}
