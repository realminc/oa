use super::*;
use crate::{Engine, matrix};

#[test]
#[ignore = "requires a timestamp-capable Vulkan compute device"]
fn timed_chunk_span_retains_first_query_and_final_completion() -> Result<()> {
	let engine = Engine::new()?;
	let zero = matrix::full(&engine, [16, 16], 0.0)?;
	let (plan, _output) = engine.capture(|| matrix::mat_mul_nt(&zero, &zero))?;
	let first = engine.submit_timed(&plan)?;
	let mut last = engine.submit_timed(&plan)?;
	let final_duration = last.device_duration()?;
	let final_epoch = last.epoch();
	last.set_timing_start(&first);
	drop(first);
	assert_eq!(last.epoch(), final_epoch);
	assert!(last.is_complete()?);
	let duration = last.device_duration()?;
	assert!(
		duration > final_duration,
		"span must include the earlier dispatch"
	);
	assert_eq!(last.try_device_duration()?, Some(duration));
	drop(engine);
	assert_eq!(last.device_duration()?, duration);
	Ok(())
}
