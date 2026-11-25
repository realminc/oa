//! `oa tune` — offline measured-route tuning evidence runs.
//!
//! Runs a GEMM shape suite under the tile-search flags so each candidate
//! signature accrues measured samples; `Runtime` drop commits the route
//! observations and block artifacts into the device-scoped `.vkcache`,
//! which later runs replay without re-measurement.

use std::error::Error;

use oa::{DeviceSelection, Matrix, Runtime};

/// `MxNxK` tile-search suite used when `--shapes` is not given: small,
/// square, wide-N and deep-K representatives of the admitted geometry set.
const DEFAULT_SHAPES: &str = "64x64x32,128x128x128,256x256x256,512x512x512,64x512x256";

/// Run the measured GEMM tile search over `shapes` for `samples`
/// dispatches each on device `device`.
///
/// `shapes` is a comma-separated `MxNxK` list; empty selects
/// [`DEFAULT_SHAPES`]. The opt-in tile flags are enabled for this process
/// before runtime creation so the measured candidates are offered.
pub fn run(device: usize, samples: u32, shapes: &str) -> Result<(), Box<dyn Error>> {
	// SAFETY: the process-wide opt-in route flags are set once before any
	// runtime or worker thread exists in this single-command scope.
	for (name, value) in [
		("OA_TILE_GEMM", "1"),
		("OA_TILE_GEMM_SEARCH", "1"),
		("OA_JIT_FUSION", "1"),
		("OA_JIT_ROUTE", "specialized"),
	] {
		unsafe { std::env::set_var(name, value) };
	}
	let runtime = Runtime::builder()
		.devices(DeviceSelection::Index(device))
		.build()?;
	let list = if shapes.is_empty() {
		DEFAULT_SHAPES
	} else {
		shapes
	};
	let mut geometries = Vec::new();
	for entry in list.split(',') {
		let parts: Vec<usize> = entry
			.trim()
			.split('x')
			.map(str::parse)
			.collect::<Result<_, _>>()?;
		let [m, n, k] = parts.as_slice() else {
			return Err(format!("invalid MxNxK geometry `{entry}`").into());
		};
		geometries.push((*m, *n, *k));
	}
	for (m, n, k) in &geometries {
		let left = Matrix::new_from_f32_on(&runtime, [*m, *k], &fill(m * k, 3))?;
		let right = Matrix::new_from_f32_on(&runtime, [*n, *k], &fill(n * k, 11))?;
		for _ in 0..samples {
			let _ = oa::matrix::matmul(&left, &right)?;
		}
		runtime.checkpoint()?.wait()?;
		println!("tuned {m}x{n}x{k}: {samples} dispatches");
	}
	println!(
		"tuned {} geometries on device {device}; route observations persist to the device-scoped .vkcache",
		geometries.len()
	);
	Ok(())
}

/// Deterministic fill — mirrors the test oracle's mixed-sign sequence so
/// tuned dispatches exercise representative operand magnitudes.
fn fill(count: usize, seed: u64) -> Vec<f32> {
	let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15);
	(0..count)
		.map(|_| {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			((state % 2011) as f32 - 1000.0) / 383.0
		})
		.collect()
}
