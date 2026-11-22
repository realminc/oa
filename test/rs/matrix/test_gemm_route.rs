//! F32 GEMM route and kernel correctness device tests.
//!
//! The wide-tile `GemmTiledWide` variant (128x128x16 workgroup tiles, 8x8
//! register tiles, double-buffered shared memory) is structurally preferred
//! for canonical problems with M and N of at least 128. These tests prove the
//! lowered kernel — not a fallback — recorded through the execution-plan
//! census and that its output matches an independent CPU oracle across exact
//! tile boundaries, odd tails, and boundary-crossing sizes.

fn values(length: usize, salt: usize) -> Vec<f32> {
	(0..length)
		.map(|index| (((index * 17 + salt) % 29) as f32 - 14.0) / 7.0)
		.collect()
}

fn cpu_mat_mul_nt(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
	let mut output = vec![0.0; m * n];
	for row in 0..m {
		for column in 0..n {
			for inner in 0..k {
				output[row * n + column] += left[row * k + inner] * right[column * k + inner];
			}
		}
	}
	output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"element {index}: expected {expected}, found {actual}, error {error} exceeds {tolerance}"
		);
	}
}

fn kernel_names(plan: &oa::ExecutionPlan, operation: &str) -> Vec<String> {
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json(operation))
		.expect("execution report must be valid JSON");
	report["nodes"]
		.as_array()
		.expect("execution nodes must be an array")
		.iter()
		.map(|node| node["kernel"].as_str().unwrap_or_default().to_owned())
		.collect()
}

test_vk!(
	tiled_wide_route_records_wide_kernel_and_matches_oracle,
	engine,
	{
		// Every case carries at least 64 output tiles of the 128x64 grid so the
		// structural preference admits the wide variant; the mix still crosses
		// M, N, and K tails plus the single-K-tile boundary.
		for (m, n, k) in [
			(1024_usize, 1024_usize, 16_usize),
			(1024, 1024, 1024),
			(1025, 1023, 33),
			(1030, 770, 193),
			(1024, 1024, 17),
			(512, 1024, 64),
			(256, 4096, 96),
			(2048, 2048, 512),
		] {
			let left_values = values(m * k, 3);
			let right_values = values(n * k, 11);
			let expected = cpu_mat_mul_nt(&left_values, &right_values, m, n, k);
			let left = oa::Matrix::new_from_f32([m, k], &left_values)?;
			let right = oa::Matrix::new_from_f32([n, k], &right_values)?;
			let (plan, output) = engine.capture(|| oa::matrix::mat_mul_nt(&left, &right))?;
			let kernels = kernel_names(&plan, "mat_mul_nt");
			assert_eq!(
				kernels,
				vec!["matrix.mat_mul_nt_tiled_wide.f32"],
				"m={m} n={n} must route to the wide tile"
			);
			engine.submit(&plan)?.wait()?;
			assert_eq!(output.shape(), [m, n]);
			assert_close(&output.read_f32()?, &expected, 1.0e-3);
		}
		Ok(())
	}
);

test_vk!(sub_tile_extents_keep_established_route, engine, {
	// Below the wide-tile occupancy preference an established route must be
	// selected instead; the wide variant must not silently take problems it
	// was not admitted for.
	for (m, n, k) in [
		(64_usize, 64_usize, 32_usize),
		(127, 128, 64),
		(128, 63, 64),
		(1, 128, 256),
		(512, 512, 64),
		(512, 960, 128),
		(1024, 64, 32),
		(1024, 32, 64),
	] {
		let left_values = values(m * k, 3);
		let right_values = values(n * k, 11);
		let expected = cpu_mat_mul_nt(&left_values, &right_values, m, n, k);
		let left = oa::Matrix::new_from_f32([m, k], &left_values)?;
		let right = oa::Matrix::new_from_f32([n, k], &right_values)?;
		let (plan, output) = engine.capture(|| oa::matrix::mat_mul_nt(&left, &right))?;
		let kernels = kernel_names(&plan, "mat_mul_nt");
		assert!(
			!kernels.contains(&"matrix.mat_mul_nt_tiled_wide.f32".to_owned()),
			"m={m} n={n} must not route to the wide tile, found {kernels:?}"
		);
		println!("m={m} n={n} k={k} kernels={kernels:?}");
		engine.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 1.0e-3);
	}
	Ok(())
});
