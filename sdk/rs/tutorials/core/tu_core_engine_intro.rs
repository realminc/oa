// OA Tutorial — Engine construction and explicit submission.
//
// Shows the three-step eager authoring pattern:
//   1. construct Engine
//   2. record domain operations — no submit/wait required
//   3. read results back with explicit host observation
//
// The Engine owns the Vulkan instance, logical device, allocator, queues, and
// execution scheduling.  Domain operations like `matrix::add` record into its
// private eager session and return a Matrix value without requiring a public
// graph or submit ceremony.  Host observation (`read_f32`) flushes and waits.
//
// Explicit `engine.checkpoint()` demonstrates the available advanced boundary
// for orchestration, capture, and profiling without host observation.
//
// Mirrors sdk/cpp/tutorials/core/tuCoreEngine.cpp.
use oa::{Engine, matrix};

fn main() -> oa::Result<()> {
	let engine = Engine::new()?;

	println!("OA engine");

	// Domain operations record into the engine's private eager session.  No
	// public graph or context object is required.  The matrix values are
	// returned immediately as semantic handles.
	let a = matrix::ones(&engine, [64, 128])?;
	let b = matrix::full(&engine, [64, 128], 2.0_f32)?;
	let output = matrix::add(&a, &b)?;

	// `engine.checkpoint()` submits all pending eager work and returns an
	// Event without blocking the host.  Calling it is optional — host
	// observation (read_f32 below) also flushes and waits.
	let checkpoint = engine.checkpoint()?;
	checkpoint.wait()?;

	// `read_f32` is the explicit device-to-host boundary.  It flushes any
	// remaining eager work, waits for the result, and copies to a Vec.
	let values = output.read_f32()?;
	let first = values.first().copied().unwrap_or(f32::NAN);

	println!("  add(Ones, full(2))[0] = {first:.1}");
	println!("  shape: {:?}", output.shape());
	println!("  elements: {}", values.len());

	assert!(
		(first - 3.0_f32).abs() <= 1e-5,
		"unexpected value: expected 3.0 but got {first}"
	);
	assert_eq!(output.shape(), [64, 128]);
	assert_eq!(values.len(), 64 * 128);
	Ok(())
}
