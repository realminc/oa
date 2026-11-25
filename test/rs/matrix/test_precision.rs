//! FP16-storage math-mode device tests.
//!
//! `OA_FORCE_PRECISION=FP16` admits the FP16-storage/FP32-accumulate lowering
//! only when the device has enabled `storageBuffer16BitAccess` and
//! `shaderFloat16`; every test here additionally proves the lowered kernels —
//! not an F32 fallback — actually recorded through the execution-plan census.

static PRECISION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Scoped `OA_FORCE_PRECISION` override; serialized so concurrent ignored
/// tests never observe a torn mode. `pub(crate)` because the mode is
/// process-wide — route census tests outside this module must hold the same
/// lock while they assert F32 kernel identity.
pub(crate) struct PrecisionMode {
	previous: Option<std::ffi::OsString>,
	_guard: std::sync::MutexGuard<'static, ()>,
}

impl PrecisionMode {
	/// Lock the mode and remove the variable — the unforced default route.
	pub(crate) fn remove() -> Self {
		let guard = PRECISION_LOCK
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		let previous = std::env::var_os("OA_FORCE_PRECISION");
		// SAFETY: every mode-sensitive test serializes on PRECISION_LOCK.
		unsafe { std::env::remove_var("OA_FORCE_PRECISION") };
		Self {
			previous,
			_guard: guard,
		}
	}

	fn set(value: &str) -> Self {
		let guard = PRECISION_LOCK
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		let previous = std::env::var_os("OA_FORCE_PRECISION");
		// SAFETY: every mode-sensitive test serializes on PRECISION_LOCK.
		unsafe { std::env::set_var("OA_FORCE_PRECISION", value) };
		Self {
			previous,
			_guard: guard,
		}
	}
}

impl Drop for PrecisionMode {
	fn drop(&mut self) {
		// SAFETY: same serialized scope as `set`.
		unsafe {
			match &self.previous {
				Some(value) => std::env::set_var("OA_FORCE_PRECISION", value),
				None => std::env::remove_var("OA_FORCE_PRECISION"),
			}
		}
	}
}

fn values(length: usize, salt: usize) -> Vec<f32> {
	(0..length)
		.map(|index| (((index * 17 + salt) % 29) as f32 - 14.0) / 7.0)
		.collect()
}

/// Narrow each element to binary16 storage and widen back — the exact values
/// the FP16-storage kernels observe.
fn f16_rounded(values: &[f32]) -> Vec<f32> {
	values
		.iter()
		.map(|value| oa::F16::from_f32(*value).to_f32())
		.collect()
}

fn cpu_matmul_nt(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
	assert_eq!(actual.len(), expected.len());
	let mut worst = 0.0_f32;
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		worst = worst.max(error);
		assert!(
			error <= tolerance,
			"{context}: element {index}: expected {expected}, found {actual}, error {error} exceeds {tolerance}"
		);
	}
	println!("{context}: worst absolute error {worst} (tolerance {tolerance})");
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

test_vk!(fp16_matrix_storage_round_trips_exact_bits, runtime, {
	let values = [
		0.0_f32,
		-0.0,
		1.0,
		-2.5,
		65504.0,
		6.1035e-5,
		f32::INFINITY,
		-3.5,
	];
	let storage: Vec<oa::F16> = values
		.iter()
		.map(|value| oa::F16::from_f32(*value))
		.collect();
	let matrix = oa::Matrix::new_from_slice([8], &storage)?;
	assert_eq!(matrix.dtype(), oa::DType::F16);
	let readback = matrix.read::<oa::F16>()?;
	for (index, (written, read)) in storage.iter().zip(&readback).enumerate() {
		assert_eq!(
			written.to_bits(),
			read.to_bits(),
			"element {index} storage bits drifted"
		);
	}
	Ok(())
});

test_vk!(fp16_storage_gemm_matches_narrowed_f32_oracle, runtime, {
	let _mode = PrecisionMode::set("FP16");
	for (m, n, k) in [
		(2_usize, 2_usize, 3_usize),
		(9, 3, 5),
		(64, 64, 16),
		(70, 33, 17),
		(128, 96, 64),
		(5, 130, 259),
	] {
		let left_values = values(m * k, 3);
		let right_values = values(n * k, 11);
		// The kernel sees exactly these narrowed operands; the F32 CPU oracle
		// over narrowed inputs isolates storage rounding from the accumulation
		// contract.
		let expected = cpu_matmul_nt(
			&f16_rounded(&left_values),
			&f16_rounded(&right_values),
			m,
			n,
			k,
		);
		let left = oa::Matrix::new_from_f32([m, k], &left_values)?;
		let right = oa::Matrix::new_from_f32([n, k], &right_values)?;
		let (plan, output) = runtime.capture(|| oa::matrix::matmul(&left, &right))?;
		let kernels = kernel_names(&plan, "matmul");
		assert_eq!(
			kernels,
			vec![
				"matrix.cast_f32_f16.f16",
				"matrix.cast_f32_f16.f16",
				"matrix.matmul_nt_f16.f16",
			],
			"FP16 mode must record the cast+cast+f16 lowering for {m}x{n}x{k}"
		);
		runtime.submit(&plan)?.wait()?;
		assert_eq!(output.dtype(), oa::DType::F32);
		// FP32 accumulation over binary16 operands: summation-order drift only.
		// k <= 259 keeps the worst-case ordering gap well under 1e-2 on the
		// [-2, 2] value domain.
		assert_close(
			&output.read_f32()?,
			&expected,
			1.0e-2,
			&format!("f16 gemm {m}x{n}x{k}"),
		);
	}
	Ok(())
});

test_vk!(
	fp16_storage_wide_gemm_matches_narrowed_f32_oracle,
	runtime,
	{
		let _mode = PrecisionMode::set("FP16");
		// At least 64 tiles of the 128x64 grid admits the wide F16-storage
		// variant; the shape mix still crosses M/N tails and an odd K that
		// forces the scalar (unaligned) load path.
		for (m, n, k) in [
			(1030_usize, 770_usize, 193_usize),
			(1024, 1024, 512),
			(512, 1024, 64),
			(1025, 1023, 16),
		] {
			let left_values = values(m * k, 3);
			let right_values = values(n * k, 11);
			let expected = cpu_matmul_nt(
				&f16_rounded(&left_values),
				&f16_rounded(&right_values),
				m,
				n,
				k,
			);
			let left = oa::Matrix::new_from_f32([m, k], &left_values)?;
			let right = oa::Matrix::new_from_f32([n, k], &right_values)?;
			let (plan, output) = runtime.capture(|| oa::matrix::matmul(&left, &right))?;
			let kernels = kernel_names(&plan, "matmul");
			assert!(
				kernels.contains(&"matrix.matmul_nt_tiled_wide_f16.f16".to_owned()),
				"FP16 mode must record the tiled_wide_f16 kernel for {m}x{n}x{k}, 				found {kernels:?}"
			);
			runtime.submit(&plan)?.wait()?;
			assert_eq!(output.dtype(), oa::DType::F32);
			assert_close(
				&output.read_f32()?,
				&expected,
				1.0e-2,
				&format!("f16 tiled wide gemm {m}x{n}x{k}"),
			);
		}
		Ok(())
	}
);

test_vk!(fp16_storage_small_gemm_keeps_f16_route, runtime, {
	let _mode = PrecisionMode::set("FP16");
	// Below the 64-tile occupancy preference the established f16 variant
	// stays selected; the wide kernel must not take small problems.
	for (m, n, k) in [
		(64_usize, 64_usize, 16_usize),
		(512, 512, 64),
		(1024, 32, 64),
	] {
		let left_values = values(m * k, 3);
		let right_values = values(n * k, 11);
		let expected = cpu_matmul_nt(
			&f16_rounded(&left_values),
			&f16_rounded(&right_values),
			m,
			n,
			k,
		);
		let left = oa::Matrix::new_from_f32([m, k], &left_values)?;
		let right = oa::Matrix::new_from_f32([n, k], &right_values)?;
		let (plan, output) = runtime.capture(|| oa::matrix::matmul(&left, &right))?;
		let kernels = kernel_names(&plan, "matmul");
		assert_eq!(
			kernels,
			vec![
				"matrix.cast_f32_f16.f16",
				"matrix.cast_f32_f16.f16",
				"matrix.matmul_nt_f16.f16",
			],
			"m={m} n={n} must keep the f16 route, found {kernels:?}"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(
			&output.read_f32()?,
			&expected,
			1.0e-2,
			&format!("f16 small gemm {m}x{n}x{k}"),
		);
	}
	Ok(())
});

test_vk!(fp16_storage_sum_matches_narrowed_f32_oracle, runtime, {
	let _mode = PrecisionMode::set("FP16");
	for length in [1_usize, 7, 256, 4097, 131_072] {
		let input_values = values(length, 5);
		let expected: f32 = f16_rounded(&input_values).iter().sum();
		let input = oa::Matrix::new_from_f32([length], &input_values)?;
		let (plan, output) = runtime.capture(|| oa::matrix::sum(&input, -1))?;
		let kernels = kernel_names(&plan, "sum");
		assert_eq!(
			kernels,
			vec!["matrix.cast_f32_f16.f16", "matrix.sum_f16.f16"],
			"FP16 mode must record the cast+sum_f16 lowering for {length} elements"
		);
		runtime.submit(&plan)?.wait()?;
		let actual = output.read_f32()?;
		assert_eq!(actual.len(), 1);
		// The CPU oracle sums left-to-right while the kernel folds 256 lanes
		// then tree-reduces; 131k elements of [-2, 2] stay under 1e-2.
		assert_close(&actual, &[expected], 1.0e-2, &format!("f16 sum {length}"));
	}
	Ok(())
});

test_vk!(f32_default_route_records_f32_kernels, runtime, {
	// Regression anchor: without the FP16 request the established F32 routes
	// stay — the precision mode never changes unforced semantics.
	let _mode = PrecisionMode::remove();
	let left_values = values(8 * 5, 3);
	let right_values = values(7 * 5, 11);
	let left = oa::Matrix::new_from_f32([8, 5], &left_values)?;
	let right = oa::Matrix::new_from_f32([7, 5], &right_values)?;
	let (plan, _output) = runtime.capture(|| oa::matrix::matmul(&left, &right))?;
	let kernels = kernel_names(&plan, "matmul");
	assert!(
		kernels.iter().all(|name| name.ends_with(".f32")),
		"unforced matmul must record only f32 kernels, found {kernels:?}"
	);
	Ok(())
});

test_vk!(fp16_storage_linear_matches_narrowed_f32_oracle, runtime, {
	let _mode = PrecisionMode::set("FP16");
	for (batch, input_features, output_features) in [
		(2_usize, 3_usize, 4_usize),
		(5, 7, 9),
		(64, 64, 64),
		(33, 130, 17),
	] {
		let input_values = values(batch * input_features, 3);
		let weight_values = values(output_features * input_features, 11);
		let bias_values = values(output_features, 23);
		// The ML linear route narrows input and weight to binary16 storage,
		// accumulates the product in F32, and adds the untouched F32 bias.
		let narrowed_input = f16_rounded(&input_values);
		let narrowed_weight = f16_rounded(&weight_values);
		let mut expected = cpu_matmul_nt(
			&narrowed_input,
			&narrowed_weight,
			batch,
			output_features,
			input_features,
		);
		for (index, value) in expected.iter_mut().enumerate() {
			*value += bias_values[index % output_features];
		}
		let input = oa::Matrix::new_from_f32([batch, input_features], &input_values)?;
		let weight = oa::Matrix::new_from_f32([output_features, input_features], &weight_values)?;
		let bias = oa::Matrix::new_from_f32([output_features], &bias_values)?;
		let layer = oa::ml::nn::Linear::new_from_matrices(weight, bias)?;
		let (plan, output) = runtime.capture(|| layer.forward(&input))?;
		let kernels = kernel_names(&plan, "linear");
		assert_eq!(
			kernels,
			vec![
				"matrix.cast_f32_f16.f16",
				"matrix.cast_f32_f16.f16",
				"matrix.matmul_nt_f16.f16",
				"matrix.bias_add_broadcast.f32",
			],
			"FP16 mode must record the cast+cast+f16+bias lowering for {batch}x{input_features}x{output_features}"
		);
		runtime.submit(&plan)?.wait()?;
		assert_eq!(output.dtype(), oa::DType::F32);
		assert_close(
			&output.read_f32()?,
			&expected,
			1.0e-2,
			&format!("f16 linear {batch}x{input_features}x{output_features}"),
		);
	}
	Ok(())
});
