use super::*;
use crate::{Runtime, matrix};

#[test]
#[ignore = "requires a timestamp-capable Vulkan compute device"]
fn timed_chunk_span_retains_first_query_and_final_completion() -> Result<()> {
	let runtime = Runtime::new()?;
	let zero = matrix::full([16, 16], 0.0)?;
	let (plan, _output) = runtime.capture(|| matrix::matmul(&zero, &zero))?;
	let first = runtime.submit_timed(&plan)?;
	let mut last = runtime.submit_timed(&plan)?;
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
	drop(runtime);
	assert_eq!(last.device_duration()?, duration);
	Ok(())
}
