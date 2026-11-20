//! Exploratory tile-route isolation: captures the commands-CNN
//! training plan under four route combinations (incumbent, pool tile,
//! conv tile, both) in one process by toggling `OA_TILE_*` between
//! captures, then interleaves timed replays of each candidate against
//! the incumbent plan. Numbers are exploratory evidence, not the
//! qualified runner protocol.

use std::time::Duration;

use oa::ml::Module;

const WARMUP_REPLAYS: usize = 5;
const MEASURED_REPLAYS: usize = 15;

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

fn capture_training(
	engine: &oa::Engine,
	model: &impl Module,
	input: &oa::Matrix,
	target: &oa::Matrix,
	conv: bool,
	pool: bool,
) -> oa::Result<oa::ExecutionPlan> {
	if conv {
		unsafe { std::env::set_var("OA_TILE_CONV", "1") };
	} else {
		unsafe { std::env::remove_var("OA_TILE_CONV") };
	}
	if pool {
		unsafe { std::env::set_var("OA_TILE_POOL", "1") };
	} else {
		unsafe { std::env::remove_var("OA_TILE_POOL") };
	}
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	engine.submit(&plan)?.wait()?;
	Ok(plan)
}

fn capture_forward(
	engine: &oa::Engine,
	model: &impl Module,
	input: &oa::Matrix,
	conv: bool,
	pool: bool,
) -> oa::Result<oa::ExecutionPlan> {
	if conv {
		unsafe { std::env::set_var("OA_TILE_CONV", "1") };
	} else {
		unsafe { std::env::remove_var("OA_TILE_CONV") };
	}
	if pool {
		unsafe { std::env::set_var("OA_TILE_POOL", "1") };
	} else {
		unsafe { std::env::remove_var("OA_TILE_POOL") };
	}
	let (plan, _) = engine.capture(|| model.forward(input))?;
	engine.submit(&plan)?.wait()?;
	Ok(plan)
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = sdk::ml::audio::commands_cnn(&engine, 40, 63, 8)?;
	let input = oa::Matrix::new_from_f32(
		&engine,
		[8, 1, 40, 63],
		&vec![0.05; 8 * 40 * 63]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_slice(&engine, [8], &[0_u32, 1, 2, 3, 4, 5, 6, 7])?;
	let grad_reset = oa::ml::Sgd::new(model.all_parameters()?, 0.0, 0.0, 0.0)?;

	let incumbent_forward = capture_forward(&engine, &model, &input, false, false)?;
	let conv_forward = capture_forward(&engine, &model, &input, true, false)?;
	let pool_forward = capture_forward(&engine, &model, &input, false, true)?;
	for (label, plan) in [("conv-fwd", &conv_forward), ("pool-fwd", &pool_forward)] {
		let (candidate, reference) = interleaved_medians(&engine, plan, &incumbent_forward)?;
		println!(
			"{label} tile forward GPU median: {:.3} ms · incumbent {:.3} ms",
			milliseconds(candidate),
			milliseconds(reference)
		);
	}

	let incumbent = capture_training(&engine, &model, &input, &target, false, false)?;
	grad_reset.zero_grad();
	let pool = capture_training(&engine, &model, &input, &target, false, true)?;
	grad_reset.zero_grad();
	let conv = capture_training(&engine, &model, &input, &target, true, false)?;
	grad_reset.zero_grad();
	let both = capture_training(&engine, &model, &input, &target, true, true)?;
	unsafe { std::env::remove_var("OA_TILE_CONV") };
	unsafe { std::env::remove_var("OA_TILE_POOL") };

	for (label, plan) in [("pool", &pool), ("conv", &conv), ("both", &both)] {
		let (candidate, reference) = interleaved_medians(&engine, plan, &incumbent)?;
		println!(
			"{label} tile forward+backward GPU median: {:.3} ms · incumbent {:.3} ms",
			milliseconds(candidate),
			milliseconds(reference)
		);
	}
	println!("fusion census: {}", engine.fusion_stats());
	Ok(())
}
