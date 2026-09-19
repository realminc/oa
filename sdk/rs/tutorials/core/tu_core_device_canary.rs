// OA Tutorial — device admission canary.
//
// Constructs an Engine, runs a minimal captured matrix operation, and prints
// a structured summary of the device and execution evidence.  Exit code 0
// confirms the device is admitted; any engine or dispatch failure exits with
// a non-zero code through the `oa::Result` propagation.
//
// The JSON execution-plan report is suitable for capture into an evidence
// bundle or a scheduler admission record.  Set OA_VK_VALIDATION=1 to enable
// Vulkan validation layers.
//
// Mirrors the intent of sdk/cpp/tutorials/core/tuCoreDeviceCanary.cpp.
use oa::{Engine, matrix};

fn main() -> oa::Result<()> {
	let engine = Engine::builder().build().map_err(|e| {
		eprintln!("device canary: Engine construction failed: {e}");
		e
	})?;

	// Capture a minimal plan: ones + full(2) → add.  The plan diagnostics
	// and debug report are the device-admission evidence.
	let (plan, result) = engine.capture(|| {
		let a = matrix::ones(&engine, [64, 128])?;
		let b = matrix::full(&engine, [64, 128], 2.0_f32)?;
		matrix::add(&a, &b)
	})?;

	let event = engine.submit(&plan)?;
	event.wait()?;

	let values = result.read_f32()?;
	let first = values.first().copied().unwrap_or(f32::NAN);

	let diagnostics = plan.diagnostics();
	let report_json = plan.debug_report_json("device_canary");

	println!("OA device canary");
	println!("  nodes:    {}", diagnostics.node_count());
	println!("  barriers: {}", diagnostics.barrier_count());
	println!("  schema:   {}", diagnostics.schema_owned_node_count());
	println!("  add(Ones, full(2))[0] = {first:.1}");
	println!();
	println!("{report_json}");

	assert!(
		(first - 3.0_f32).abs() <= 1e-5,
		"device canary arithmetic failed: expected 3.0 but got {first}"
	);
	assert_eq!(diagnostics.node_count(), 1, "expected one compute node");
	Ok(())
}
