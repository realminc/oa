//! Exploratory attention tile-route isolation: captures SDPA plans under
//! incumbent and `OA_TILE_ATTENTION` routes in one process by toggling
//! the flag between captures, then interleaves timed replays of each
//! tile candidate against its incumbent plan. `OA_JIT_FUSION` stays set
//! throughout so the fused `ml.small_causal_sdpa.f32` region forms on
//! both sides of the causal comparison. Numbers are exploratory
//! evidence, not the qualified runner protocol.

use std::time::Duration;

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

enum Route {
	Standard,
	Flash,
	Causal,
}

fn capture(
	engine: &oa::Engine,
	route: &Route,
	tile: bool,
	batch_heads: usize,
	sequence: usize,
	head_dim: usize,
	training: bool,
) -> oa::Result<oa::ExecutionPlan> {
	if tile {
		unsafe { std::env::set_var("OA_TILE_ATTENTION", "1") };
	} else {
		unsafe { std::env::remove_var("OA_TILE_ATTENTION") };
	}
	let shape = [batch_heads, sequence, head_dim];
	let rows = batch_heads * sequence;
	let scale = 1.0 / (head_dim as f32).sqrt();
	let query = oa::matrix::full(shape, 0.125)?;
	let key = oa::matrix::full(shape, -0.25)?;
	let value = oa::matrix::full(shape, 0.375)?;
	let packed_query = query.reshape([rows, head_dim])?;
	let packed_key = key.reshape([rows, head_dim])?;
	let packed_value = value.reshape([rows, head_dim])?;
	let target = oa::matrix::full(shape, 0.0)?;
	let (plan, _) = engine.capture(|| {
		let forward = |query: &oa::Matrix, key: &oa::Matrix, value: &oa::Matrix| match route {
			Route::Standard => {
				oa::ml::matrix::scaled_dot_product_attention(query, key, value, None, scale, true)
			}
			Route::Flash => oa::ml::matrix::flash_attention_causal(query, key, value, scale),
			Route::Causal => oa::ml::matrix::scaled_dot_product_attention_causal(
				&packed_query,
				&packed_key,
				&packed_value,
				sequence,
				1,
			)?
			.reshape(shape),
		};
		if training {
			let tape = oa::ml::GradientTape::new();
			let output = forward(&query, &key, &value)?;
			let loss = oa::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		} else {
			forward(&query, &key, &value)
		}
	})?;
	engine.submit(&plan)?.wait()?;
	Ok(plan)
}

fn compare(
	engine: &oa::Engine,
	route: &Route,
	name: &str,
	batch_heads: usize,
	sequence: usize,
	head_dim: usize,
	training: bool,
) -> oa::Result<()> {
	let incumbent = capture(
		engine,
		route,
		false,
		batch_heads,
		sequence,
		head_dim,
		training,
	)?;
	let tile = capture(
		engine,
		route,
		true,
		batch_heads,
		sequence,
		head_dim,
		training,
	)?;
	let (candidate, reference) = interleaved_medians(engine, &tile, &incumbent)?;
	let phase = if training { "fwd+bwd" } else { "fwd" };
	println!(
		"{name} [{batch_heads},{sequence},{head_dim}] {phase} tile GPU median: {:.3} ms · incumbent {:.3} ms",
		milliseconds(candidate),
		milliseconds(reference)
	);
	Ok(())
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	oa::Engine::set_default(engine.clone());
	// The fused seven-node region only forms while the opt-in fusion
	// route is enabled; hold it for the whole suite so the causal
	// comparisons measure kernel substitution, not region formation.
	unsafe { std::env::set_var("OA_JIT_FUSION", "1") };
	for (batch_heads, sequence, head_dim) in [
		(64_usize, 16_usize, 32_usize),
		(32, 32, 16),
		(64, 64, 32),
		(32, 128, 16),
	] {
		for (route, name) in [
			(Route::Standard, "standard"),
			(Route::Flash, "flash"),
			(Route::Causal, "causal-fused"),
		] {
			for training in [false, true] {
				compare(
					&engine,
					&route,
					name,
					batch_heads,
					sequence,
					head_dim,
					training,
				)?;
			}
		}
	}
	println!("fusion census: {}", engine.fusion_stats());
	Ok(())
}
