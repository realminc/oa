//! Native Q4/Q8 quantized-weight contracts.
//!
//! Donor: `oacpp/test/cpp/ml/quant/testFnMatrixQuant.cpp`.

use oa::ml::matrix::{QuantMatrix, Quantization, dequantize, matmul, quantize};

const BLOCK_SIZE: usize = 32;
const Q4_BYTES_PER_BLOCK: usize = 16;
const Q8_BYTES_PER_BLOCK: usize = 32;

fn make_input(_runtime: &oa::Runtime, values: &[f32]) -> oa::Result<oa::Matrix> {
	oa::Matrix::new_from_f32([values.len()], values)
}

fn quantize_nibble(value: f32, scale: f32) -> u8 {
	let rounded = (value / scale).round_ties_even() as i32;
	(rounded.clamp(-7, 7) + 7) as u8
}

fn reference_q4_payload(values: &[f32], scales: &[f32]) -> Vec<u8> {
	let mut payload = vec![0x77_u8; scales.len() * Q4_BYTES_PER_BLOCK];
	for (i, value) in values.iter().enumerate() {
		let nibble = quantize_nibble(*value, scales[i / BLOCK_SIZE]);
		if i % 2 == 0 {
			payload[i / 2] = (payload[i / 2] & 0xF0) | nibble;
		} else {
			payload[i / 2] = (payload[i / 2] & 0x0F) | (nibble << 4);
		}
	}
	payload
}

fn quantize_q8_value(value: f32, scale: f32) -> i8 {
	let rounded = (value / scale).round_ties_even() as i32;
	rounded.clamp(-127, 127) as i8
}

fn reference_q8_payload(values: &[f32], scales: &[f32]) -> Vec<u8> {
	let mut payload = vec![0_u8; scales.len() * Q8_BYTES_PER_BLOCK];
	for (i, value) in values.iter().enumerate() {
		payload[i] = quantize_q8_value(*value, scales[i / BLOCK_SIZE]) as u8;
	}
	payload
}

fn dequantize_q4_oracle(payload: &[u8], scales: &[f32], count: usize) -> Vec<f32> {
	(0..count)
		.map(|i| {
			let packed = payload[i / 2];
			let nibble = if i % 2 == 0 {
				packed & 0x0F
			} else {
				packed >> 4
			};
			let scale = scales[i / BLOCK_SIZE];
			if scale.is_finite() && scale > 0.0 {
				(i32::from(nibble) - 7) as f32 * scale
			} else {
				0.0
			}
		})
		.collect()
}

fn dequantize_q8_oracle(payload: &[u8], scales: &[f32], count: usize) -> Vec<f32> {
	(0..count)
		.map(|i| {
			let scale = scales[i / BLOCK_SIZE];
			if scale.is_finite() && scale > 0.0 {
				(payload[i] as i8) as f32 * scale
			} else {
				0.0
			}
		})
		.collect()
}

fn reference_matmul_nt(
	input: &[f32],
	weight: &[f32],
	rows: usize,
	output_features: usize,
	input_features: usize,
) -> Vec<f32> {
	let mut output = vec![0.0_f32; rows * output_features];
	for row in 0..rows {
		for column in 0..output_features {
			let mut sum = 0.0_f32;
			for k in 0..input_features {
				sum += input[row * input_features + k] * weight[column * input_features + k];
			}
			output[row * output_features + column] = sum;
		}
	}
	output
}

fn expect_near(actual: &[f32], expected: &[f32], relative_tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (i, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
		let tolerance = relative_tolerance * 1.0_f32.max(expected.abs());
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {i}: expected {expected}, got {actual} (tolerance {tolerance})"
		);
	}
}

fn expect_round_trip(
	runtime: &oa::Runtime,
	values: &[f32],
	quantization: Quantization,
) -> oa::Result<()> {
	let input = make_input(runtime, values)?;
	let quantized = quantize(&input, quantization)?;
	let scales = quantized.scale().read_f32()?;
	let output = dequantize(&quantized)?.read_f32()?;
	assert_eq!(output.len(), values.len());
	for (i, (actual, expected)) in output.iter().zip(values.iter()).enumerate() {
		let tolerance = scales[i / BLOCK_SIZE] * 0.5001 + 1.0e-6;
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {i}: expected {expected}, got {actual} (tolerance {tolerance})"
		);
	}
	Ok(())
}

test_vk!(
	q4_compute_scale_owns_every_block_and_partial_tail,
	runtime,
	{
		let mut values = vec![0.0_f32; 73];
		values[0] = -7.0;
		values[33] = 14.0;
		values[72] = -21.0;
		let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q4)?;
		expect_near(&quantized.scale().read_f32()?, &[1.0, 2.0, 3.0], 1.0e-6);
		Ok(())
	}
);

test_vk!(q4_zero_blocks_use_finite_unit_scale, runtime, {
	let quantized = quantize(&make_input(&runtime, &vec![0.0; 65])?, Quantization::Q4)?;
	assert_eq!(quantized.scale().read_f32()?, vec![1.0, 1.0, 1.0]);
	Ok(())
});

test_vk!(
	q4_packed_payload_matches_independent_aligned_oracle,
	runtime,
	{
		let values: Vec<f32> = (0..50).map(|i| ((i * 11) % 29 - 14) as f32).collect();
		let input = make_input(&runtime, &values)?;
		let quantized = quantize(&input, Quantization::Q4)?;
		let scales = quantized.scale().read_f32()?;
		let payload = quantized.payload().read::<u8>()?;
		assert_eq!(payload.len(), 32);
		assert_eq!(payload, reference_q4_payload(&values, &scales));
		for (i, byte) in payload.iter().enumerate().skip(25) {
			assert_eq!(*byte, 0x77, "padding byte {i}");
		}
		Ok(())
	}
);

test_vk!(
	q4_odd_sizes_and_partial_blocks_round_trip_within_half_step,
	runtime,
	{
		for count in [1_usize, 7, 31, 32, 33, 50, 257] {
			let values: Vec<f32> = (0..count).map(|i| (i as f32 * 0.17).sin() * 13.0).collect();
			expect_round_trip(&runtime, &values, Quantization::Q4)?;
		}
		Ok(())
	}
);

test_vk!(q4_sign_and_extreme_values_are_preserved, runtime, {
	let values: Vec<f32> = (0..32)
		.map(|i| if i % 2 == 0 { -7.0 } else { 7.0 })
		.collect();
	let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q4)?;
	assert_eq!(dequantize(&quantized)?.read_f32()?, values);
	Ok(())
});

test_vk!(
	q4_nonfinite_values_encode_as_zero_and_do_not_poison_scale,
	runtime,
	{
		let mut values = vec![0.0_f32; 64];
		values[0] = f32::NAN;
		values[1] = f32::INFINITY;
		values[2] = f32::NEG_INFINITY;
		values[3] = -7.0;
		values[32] = 7.0e-9;
		let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q4)?;
		let scales = quantized.scale().read_f32()?;
		assert_eq!(scales.len(), 2);
		expect_near(&scales, &[1.0, 1.0e-9], 1.0e-6);
		let output = dequantize(&quantized)?.read_f32()?;
		assert_eq!(output[0], 0.0);
		assert_eq!(output[1], 0.0);
		assert_eq!(output[2], 0.0);
		assert_eq!(output[3], -7.0);
		assert!((output[32] - 7.0e-9).abs() <= 1.0e-14);
		assert!(output.iter().all(|value| value.is_finite()));
		Ok(())
	}
);

test_vk!(
	q4_invalid_scale_values_produce_deterministic_zero_blocks,
	runtime,
	{
		let payload = oa::Matrix::new_from_slice([48], &[0x77_u8; 48])?;
		let scale = oa::Matrix::new_from_f32([3], &[f32::NAN, 0.0, -1.0])?;
		let weight = QuantMatrix::from_planes(payload, scale, vec![96], Quantization::Q4)?;
		assert_eq!(dequantize(&weight)?.read_f32()?, vec![0.0; 96]);
		Ok(())
	}
);

test_vk!(q4_repeated_quantization_is_byte_deterministic, runtime, {
	let values: Vec<f32> = (0..129).map(|i| (i as f32 * 0.031).cos() * 20.0).collect();
	let input = make_input(&runtime, &values)?;
	let first = quantize(&input, Quantization::Q4)?;
	let second = quantize(&input, Quantization::Q4)?;
	assert_eq!(
		first.payload().read::<u8>()?,
		second.payload().read::<u8>()?
	);
	Ok(())
});

test_vk!(
	q8_compute_scale_owns_every_block_and_partial_tail,
	runtime,
	{
		let mut values = vec![0.0_f32; 73];
		values[0] = -127.0;
		values[33] = 254.0;
		values[72] = -381.0;
		let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q8)?;
		expect_near(&quantized.scale().read_f32()?, &[1.0, 2.0, 3.0], 1.0e-6);
		Ok(())
	}
);

test_vk!(q8_payload_matches_independent_aligned_oracle, runtime, {
	let values: Vec<f32> = (0..50).map(|i| ((i * 37) % 251 - 125) as f32).collect();
	let input = make_input(&runtime, &values)?;
	let quantized = quantize(&input, Quantization::Q8)?;
	let scales = quantized.scale().read_f32()?;
	let payload = quantized.payload().read::<u8>()?;
	assert_eq!(payload.len(), 64);
	assert_eq!(payload, reference_q8_payload(&values, &scales));
	for (i, byte) in payload.iter().enumerate().skip(50) {
		assert_eq!(*byte, 0, "padding byte {i}");
	}
	Ok(())
});

test_vk!(
	q8_odd_sizes_and_partial_blocks_round_trip_within_half_step,
	runtime,
	{
		for count in [1_usize, 7, 31, 32, 33, 50, 257] {
			let values: Vec<f32> = (0..count)
				.map(|i| (i as f32 * 0.17).sin() * 130.0)
				.collect();
			expect_round_trip(&runtime, &values, Quantization::Q8)?;
		}
		Ok(())
	}
);

test_vk!(q8_sign_and_extreme_values_are_preserved, runtime, {
	let values: Vec<f32> = (0..32)
		.map(|i| if i % 2 == 0 { -127.0 } else { 127.0 })
		.collect();
	let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q8)?;
	assert_eq!(dequantize(&quantized)?.read_f32()?, values);
	Ok(())
});

test_vk!(
	q8_nonfinite_values_encode_as_zero_and_do_not_poison_scale,
	runtime,
	{
		let mut values = vec![0.0_f32; 64];
		values[0] = f32::NAN;
		values[1] = f32::INFINITY;
		values[2] = f32::NEG_INFINITY;
		values[3] = -127.0;
		values[32] = 127.0e-9;
		let quantized = quantize(&make_input(&runtime, &values)?, Quantization::Q8)?;
		let scales = quantized.scale().read_f32()?;
		assert_eq!(scales.len(), 2);
		expect_near(&scales, &[1.0, 1.0e-9], 1.0e-6);
		let output = dequantize(&quantized)?.read_f32()?;
		assert_eq!(output[0], 0.0);
		assert_eq!(output[1], 0.0);
		assert_eq!(output[2], 0.0);
		assert_eq!(output[3], -127.0);
		assert!((output[32] - 127.0e-9).abs() <= 1.0e-12);
		assert!(output.iter().all(|value| value.is_finite()));
		Ok(())
	}
);

test_vk!(
	q8_invalid_scale_values_produce_deterministic_zero_blocks,
	runtime,
	{
		let payload = oa::Matrix::new_from_slice([96], &[0_u8; 96])?;
		let scale = oa::Matrix::new_from_f32([3], &[f32::NAN, 0.0, -1.0])?;
		let weight = QuantMatrix::from_planes(payload, scale, vec![96], Quantization::Q8)?;
		assert_eq!(dequantize(&weight)?.read_f32()?, vec![0.0; 96]);
		Ok(())
	}
);

test_vk!(q8_repeated_quantization_is_byte_deterministic, runtime, {
	let values: Vec<f32> = (0..129).map(|i| (i as f32 * 0.031).cos() * 200.0).collect();
	let input = make_input(&runtime, &values)?;
	let first = quantize(&input, Quantization::Q8)?;
	let second = quantize(&input, Quantization::Q8)?;
	assert_eq!(
		first.payload().read::<u8>()?,
		second.payload().read::<u8>()?
	);
	Ok(())
});

test_vk!(
	matmul_nt_q4_fuses_dequantization_across_odd_weight_rows_and_k_tail,
	runtime,
	{
		let (rows, output_features, input_features) = (6_usize, 5_usize, 35_usize);
		let input_values: Vec<f32> = (0..rows * input_features)
			.map(|i| (i as f32 * 0.37).sin() * 3.0)
			.collect();
		let weight_values: Vec<f32> = (0..output_features * input_features)
			.map(|i| (i as f32 * 0.23).cos() * 7.0)
			.collect();
		let input = oa::Matrix::new_from_f32([rows, input_features], &input_values)?;
		let weight = quantize(
			&oa::Matrix::new_from_f32([output_features, input_features], &weight_values)?,
			Quantization::Q4,
		)?;
		let expected = reference_matmul_nt(
			&input_values,
			&dequantize_q4_oracle(
				&weight.payload().read::<u8>()?,
				&weight.scale().read_f32()?,
				weight_values.len(),
			),
			rows,
			output_features,
			input_features,
		);
		let output = matmul(&input, &weight)?;
		assert_eq!(output.shape(), &[rows, output_features]);
		expect_near(&output.read_f32()?, &expected, 2.0e-4);
		Ok(())
	}
);

test_vk!(
	matmul_nt_q8_fuses_dequantization_across_odd_weight_rows_and_k_tail,
	runtime,
	{
		let (rows, output_features, input_features) = (6_usize, 5_usize, 35_usize);
		let input_values: Vec<f32> = (0..rows * input_features)
			.map(|i| (i as f32 * 0.37).sin() * 3.0)
			.collect();
		let weight_values: Vec<f32> = (0..output_features * input_features)
			.map(|i| (i as f32 * 0.23).cos() * 127.0)
			.collect();
		let input = oa::Matrix::new_from_f32([rows, input_features], &input_values)?;
		let weight = quantize(
			&oa::Matrix::new_from_f32([output_features, input_features], &weight_values)?,
			Quantization::Q8,
		)?;
		let expected = reference_matmul_nt(
			&input_values,
			&dequantize_q8_oracle(
				&weight.payload().read::<u8>()?,
				&weight.scale().read_f32()?,
				weight_values.len(),
			),
			rows,
			output_features,
			input_features,
		);
		let output = matmul(&input, &weight)?;
		assert_eq!(output.shape(), &[rows, output_features]);
		expect_near(&output.read_f32()?, &expected, 2.0e-4);
		Ok(())
	}
);

test_vk!(
	quantized_matmul_handles_batched_leading_dimensions,
	runtime,
	{
		let (batches, rows, output_features, input_features) = (2_usize, 4_usize, 3_usize, 33_usize);
		let input_values: Vec<f32> = (0..batches * rows * input_features)
			.map(|i| (i as f32 * 0.11).sin() * 2.0)
			.collect();
		let weight_values: Vec<f32> = (0..output_features * input_features)
			.map(|i| (i as f32 * 0.29).cos() * 5.0)
			.collect();
		let input = oa::Matrix::new_from_f32([batches, rows, input_features], &input_values)?;
		let weight = quantize(
			&oa::Matrix::new_from_f32([output_features, input_features], &weight_values)?,
			Quantization::Q8,
		)?;
		let dequantized = dequantize_q8_oracle(
			&weight.payload().read::<u8>()?,
			&weight.scale().read_f32()?,
			weight_values.len(),
		);
		let expected = reference_matmul_nt(
			&input_values,
			&dequantized,
			batches * rows,
			output_features,
			input_features,
		);
		let output = matmul(&input, &weight)?;
		assert_eq!(output.shape(), &[batches, rows, output_features]);
		expect_near(&output.read_f32()?, &expected, 2.0e-4);
		Ok(())
	}
);

test_vk!(
	quantized_value_owns_format_shape_and_private_planes,
	runtime,
	{
		let input = oa::Matrix::new_from_f32([2, 3], &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0])?;
		let weight = quantize(&input, Quantization::Q4)?;
		assert_eq!(weight.quantization(), Quantization::Q4);
		assert_eq!(weight.shape(), &[2, 3]);
		assert_eq!(weight.num_elements(), 6);
		assert_eq!(weight.payload().dtype(), oa::DType::U8);
		assert_eq!(weight.payload().shape(), &[16]);
		assert_eq!(weight.scale().dtype(), oa::DType::F32);
		assert_eq!(weight.scale().shape(), &[1]);
		Ok(())
	}
);

test_vk!(dequantize_restores_the_logical_shape, runtime, {
	let values: Vec<f32> = (0..99).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
	let input = oa::Matrix::new_from_f32([3, 33], &values)?;
	let output = dequantize(&quantize(&input, Quantization::Q8)?)?;
	assert_eq!(output.shape(), &[3, 33]);
	expect_round_trip(&runtime, &values, Quantization::Q8)
});

test_vk!(
	quantized_surface_fails_closed_on_invalid_contracts,
	runtime,
	{
		let f32_matrix = oa::Matrix::new_from_f32([32], &[0.0; 32])?;
		let byte_matrix = oa::Matrix::new_from_slice([32], &[0_u8; 32])?;
		let empty = oa::Matrix::new_from_f32([0], &[])?;

		assert!(quantize(&empty, Quantization::Q4).is_err());
		assert!(quantize(&byte_matrix, Quantization::Q4).is_err());
		assert!(quantize(&byte_matrix, Quantization::Q8).is_err());

		let weight = quantize(
			&oa::Matrix::new_from_f32([2, 32], &[0.0; 64])?,
			Quantization::Q8,
		)?;
		let bad_scale = oa::Matrix::new_from_f32([3], &[1.0; 3])?;
		assert!(
			QuantMatrix::from_planes(
				weight.payload().clone(),
				bad_scale,
				vec![2, 32],
				Quantization::Q8
			)
			.is_err()
		);
		assert!(
			QuantMatrix::from_planes(
				f32_matrix.clone(),
				weight.scale().clone(),
				vec![32],
				Quantization::Q8
			)
			.is_err()
		);

		let rank3_weight = quantize(
			&oa::Matrix::new_from_f32([3, 4, 5], &[0.0; 60])?,
			Quantization::Q8,
		)?;
		let rank2_input = oa::Matrix::new_from_f32([2, 5], &[0.0; 10])?;
		assert!(matmul(&rank2_input, &rank3_weight).is_err());

		let matrix_input = oa::Matrix::new_from_f32([2, 32], &[0.0; 64])?;
		let wrong_k = oa::Matrix::new_from_f32([2, 31], &[0.0; 62])?;
		assert!(matmul(&wrong_k, &weight).is_err());
		assert!(matmul(&matrix_input, &weight).is_ok());
		Ok(())
	}
);

test_vk!(quantized_matmul_rejects_invalid_contracts, runtime, {
	let weight_matrix = oa::Matrix::new_from_f32([4, 32], &[0.0; 128])?;
	let weight = quantize(&weight_matrix, Quantization::Q4)?;
	let rank1_input = oa::Matrix::new_from_f32([32], &[0.0; 32])?;
	assert!(matmul(&rank1_input, &weight).is_err());
	let mismatched = oa::Matrix::new_from_f32([2, 31], &[0.0; 62])?;
	assert!(matmul(&mismatched, &weight).is_err());
	let valid = oa::Matrix::new_from_f32([2, 32], &[0.0; 64])?;
	assert_eq!(matmul(&valid, &weight)?.shape(), &[2, 4]);
	Ok(())
});

test_vk!(quantized_matmul_zero_rows_do_not_dispatch, runtime, {
	let weight = quantize(
		&oa::Matrix::new_from_f32([4, 32], &[1.0; 128])?,
		Quantization::Q8,
	)?;
	let input = oa::Matrix::new_from_f32([0, 32], &[])?;
	let output = matmul(&input, &weight)?;
	assert_eq!(output.shape(), &[0, 4]);
	assert_eq!(output.num_elements(), 0);
	Ok(())
});

test_vk!(
	quantize_dequantize_matmul_own_one_semantic_operation_each,
	runtime,
	{
		let (capture_plan, weight) = runtime.capture(|| {
			quantize(
				&oa::Matrix::new_from_f32([4, 32], &[1.0; 128])?,
				Quantization::Q4,
			)
		})?;
		let semantic = capture_plan.semantic_graph();
		assert_eq!(semantic.operations().len(), 1);
		let operation = &semantic.operations()[0];
		assert_eq!(operation.name(), "oa::FnMatrix::quantize");
		let output_value = semantic
			.find_value(operation.outputs()[0])
			.expect("quantized output value must be admitted");
		assert_eq!(output_value.kind(), oa::OpValueKind::QuantMatrix);
		assert_eq!(output_value.shape(), &[4, 32]);
		assert_eq!(output_value.dtype(), oa::DType::F32);
		assert!(!output_value.is_external());
		assert!(capture_plan.captured_resources().len() >= 2);
		runtime.submit(&capture_plan)?.wait()?;

		let input = oa::Matrix::new_from_f32([2, 32], &[0.5; 64])?;
		let (matmul_plan, _) = runtime.capture(|| matmul(&input, &weight))?;
		let semantic = matmul_plan.semantic_graph();
		assert_eq!(semantic.operations().len(), 1);
		assert_eq!(
			semantic.operations()[0].name(),
			"oa::FnMatrix::matmulNtQuantized"
		);
		Ok(())
	}
);

test_vk!(
	captured_quantized_matmul_matches_eager_oracle_and_replays,
	runtime,
	{
		let input_values: Vec<f32> = (0..96).map(|i| (i as f32 * 0.19).sin() * 4.0).collect();
		let input = oa::Matrix::new_from_f32([3, 32], &input_values)?;
		let weight = quantize(
			&oa::Matrix::new_from_f32(
				[4, 32],
				&(0..128)
					.map(|i| (i as f32 * 0.07).cos() * 9.0)
					.collect::<Vec<f32>>(),
			)?,
			Quantization::Q8,
		)?;
		let eager = matmul(&input, &weight)?.read_f32()?;
		let (plan, _) = runtime.capture(|| matmul(&input, &weight))?;
		for _ in 0..2 {
			runtime.submit(&plan)?.wait()?;
			assert_eq!(matmul(&input, &weight)?.read_f32()?, eager);
		}
		Ok(())
	}
);

// ---------------------------------------------------------------------------
// QuantizedLinear — model-level quantized-inference contract (ML-5).

use oa::ml::Module;
use oa::ml::nn::{Linear, QuantizedLinear};

fn dense_linear(with_bias: bool) -> oa::Result<Linear> {
	let weight = oa::Matrix::new_from_f32(
		[4, 32],
		&(0..128)
			.map(|i| (i as f32 * 0.07).cos() * 9.0)
			.collect::<Vec<f32>>(),
	)?;
	if with_bias {
		let bias = oa::Matrix::new_from_f32(
			[4],
			&(0..4).map(|i| i as f32 * 0.25 - 0.5).collect::<Vec<f32>>(),
		)?;
		Linear::new_from_matrices(weight, bias)
	} else {
		Linear::new_from_weight(weight)
	}
}

test_vk!(
	quantized_linear_q8_matches_dense_forward_within_tolerance,
	runtime,
	{
		let dense = dense_linear(true)?;
		let quantized = QuantizedLinear::from_linear(&dense, Quantization::Q8)?;
		assert_eq!(quantized.input_features(), 32);
		assert_eq!(quantized.output_features(), 4);
		assert_eq!(quantized.quantization(), Quantization::Q8);
		assert_eq!(quantized.num_parameters()?, 0);

		let input_values: Vec<f32> = (0..96).map(|i| (i as f32 * 0.19).sin() * 4.0).collect();
		let input = oa::Matrix::new_from_f32([3, 32], &input_values)?;
		let expected = dense.forward(&input)?.read_f32()?;
		let actual = quantized.forward(&input)?.read_f32()?;
		assert_eq!(expected.len(), actual.len());
		for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
			// Q8 error budget: block scale ~ max|w|/127, per-element weight
			// error <= scale/2, amplified by |x| over K=32 products.
			assert!(
				(e - a).abs() <= 0.05 + 0.01 * e.abs(),
				"Q8 QuantizedLinear deviates at {i}: expected {e}, found {a}"
			);
		}
		Ok(())
	}
);

test_vk!(
	quantized_linear_q4_without_bias_matches_dequantized_oracle,
	runtime,
	{
		let dense = dense_linear(false)?;
		let quantized = QuantizedLinear::from_linear(&dense, Quantization::Q4)?;
		assert!(quantized.bias().is_none());

		let input_values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).cos() * 2.0).collect();
		let input = oa::Matrix::new_from_f32([2, 32], &input_values)?;
		// Independent host oracle: dequantize the packed weight, then x @ W^T.
		let weight = dequantize(quantized.weight())?.read_f32()?;
		let expected: Vec<f32> = (0..2)
			.flat_map(|r| (0..4).map(move |n| (r, n)))
			.map(|(r, n)| {
				(0..32)
					.map(|k| input_values[r * 32 + k] * weight[n * 32 + k])
					.sum::<f32>()
			})
			.collect();
		let actual = quantized.forward(&input)?.read_f32()?;
		for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
			assert!(
				(e - a).abs() <= 0.05,
				"Q4 QuantizedLinear deviates from dequantized oracle at {i}: expected {e}, found {a}"
			);
		}
		Ok(())
	}
);

test_vk!(
	quantized_linear_registers_bias_as_persistent_buffer_not_parameter,
	runtime,
	{
		let dense = dense_linear(true)?;
		let quantized = QuantizedLinear::from_linear(&dense, Quantization::Q8)?;
		assert!(quantized.all_named_parameters()?.is_empty());
		let buffers = quantized.all_named_buffers()?;
		assert_eq!(buffers.len(), 1);
		assert_eq!(buffers[0].path(), "bias");
		assert!(buffers[0].persistent());
		Ok(())
	}
);

test_vk!(quantized_linear_rejects_rank_and_bias_mismatch, runtime, {
	let rank1 = quantize(
		&oa::Matrix::new_from_f32([32], &[1.0; 32])?,
		Quantization::Q8,
	)?;
	assert!(QuantizedLinear::from_quantized(rank1, None).is_err());
	let weight = quantize(
		&oa::Matrix::new_from_f32([4, 32], &[0.5; 128])?,
		Quantization::Q8,
	)?;
	let bad_bias = oa::Matrix::new_from_f32([3], &[0.0; 3])?;
	assert!(QuantizedLinear::from_quantized(weight, Some(bad_bias)).is_err());

	let dense = dense_linear(false)?;
	let quantized = QuantizedLinear::from_linear(&dense, Quantization::Q8)?;
	let bad_input = oa::Matrix::new_from_f32([2, 16], &[0.0; 32])?;
	assert!(quantized.forward(&bad_input).is_err());
	Ok(())
});
