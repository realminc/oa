//! Optional CC-40 measurements on the same captured plans as the media oracles.

use std::time::Instant;

use oa::{Engine, Error, ErrorKind, Event, ExecutionPlan, Result};
use serde_json::{Value, json};

pub(crate) fn record(
	engine: &Engine,
	plan: &ExecutionPlan,
	workload: Value,
	observe: impl Fn(Event) -> Result<()>,
) -> Result<()> {
	const WARMUP: usize = 3;
	const SAMPLES: usize = 11;
	const SOAK_REPLAYS: usize = 256;

	let mode = match std::env::var("OA_CC40_BENCH") {
		Ok(mode) => mode,
		Err(std::env::VarError::NotPresent) => return Ok(()),
		Err(_) => return Err(Error::invalid_argument("OA_CC40_BENCH must be Unicode")),
	};
	let replays = match mode.as_str() {
		"timing" => SAMPLES,
		"soak" => SOAK_REPLAYS,
		_ => {
			return Err(Error::invalid_argument(
				"OA_CC40_BENCH must be timing or soak",
			));
		}
	};

	for _ in 0..WARMUP {
		observe(engine.submit(plan)?)?;
	}
	let mut wall_ms = Vec::with_capacity(replays);
	let mut device_ms = Vec::new();
	let device_timing_unavailable = if mode == "timing" {
		match engine
			.submit_timed(plan)
			.and_then(|event| event.device_duration())
		{
			Ok(_) => None,
			Err(error) if error.kind() == ErrorKind::MissingCapability => Some(error.to_string()),
			Err(error) => return Err(error),
		}
	} else {
		None
	};
	for _ in 0..replays {
		let start = Instant::now();
		let event = engine.submit(plan)?;
		let wall = observe(event);
		let elapsed = start.elapsed();
		wall?;
		wall_ms.push(elapsed.as_secs_f64() * 1_000.0);
		if mode == "timing" && device_timing_unavailable.is_none() {
			let duration = engine.submit_timed(plan)?.device_duration()?;
			device_ms.push(duration.as_secs_f64() * 1_000.0);
		}
	}
	let diagnostics = plan.diagnostics();
	assert_eq!(diagnostics.fallback_count(), 0);
	assert_eq!(diagnostics.input_upload_count(), 0);
	assert_eq!(diagnostics.input_rebinding_count(), 0);
	let captured_bytes = plan
		.captured_resources()
		.iter()
		.map(|resource| resource.byte_len() as u64)
		.sum::<u64>();
	eprintln!(
		"OA_CC40 {}",
		json!({
			"mode": mode,
			"workload": workload,
			"warmup_replays": WARMUP,
			"replays": replays,
			"wall_ms": wall_ms,
			"device_ms": device_ms,
			"device_timing_unavailable": device_timing_unavailable,
			"nodes": diagnostics.node_count(),
			"queue_runs": diagnostics.queue_run_count(),
			"cross_queue_edges": diagnostics.cross_queue_edge_count(),
			"captured_resource_bytes": captured_bytes,
			"peak_live_buffer_bytes": diagnostics.peak_memory_bytes(),
			"fallbacks": diagnostics.fallback_count(),
		})
	);
	Ok(())
}
