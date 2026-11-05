//! Independent adjoint oracles across the numerical-provider/tape boundary.

fn assert_adjoint(parameter: &oa::ml::Parameter, expected: &[f32]) -> oa::Result<()> {
	let actual = parameter
		.gradient()
		.expect("adjoint did not reach the parameter")
		.read_f32()?;
	assert_eq!(actual.len(), expected.len());
	for (actual, expected) in actual.iter().zip(expected) {
		assert!(
			(actual - expected).abs() <= 2.0e-5,
			"{actual} != {expected}"
		);
	}
	Ok(())
}

fn scalar_sum(input: &oa::Matrix) -> oa::Result<oa::Matrix> {
	let mut result = input.clone();
	for axis in 0..input.shape().len() {
		result = oa::matrix::sum(&result, i32::try_from(axis).expect("test rank fits i32"))?;
	}
	result.reshape([])
}

test_vk!(matrix_unary_adjoints_match_analytic_oracles, engine, {
	let values = [0.5_f32, 1.5, 2.5];
	for operation in [
		"copy",
		"scale",
		"abs",
		"log",
		"sqrt",
		"exp",
		"reciprocal",
		"clamp_min",
		"clamp_max",
	] {
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, 1], &[0.0; 3])?,
			oa::Matrix::new_from_f32(&engine, [3], &values)?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?;
		let output = match operation {
			"copy" => oa::matrix::copy(&value)?,
			"scale" => oa::matrix::scale(&value, -2.0)?,
			"abs" => oa::matrix::abs(&value)?,
			"log" => oa::matrix::log(&value)?,
			"sqrt" => oa::matrix::sqrt(&value)?,
			"exp" => oa::matrix::exp(&value)?,
			"reciprocal" => oa::matrix::reciprocal(&value)?,
			"clamp_min" => oa::matrix::clamp_min(&value, 1.0)?,
			"clamp_max" => oa::matrix::clamp_max(&value, 2.0)?,
			_ => unreachable!(),
		};
		tape.backward(&scalar_sum(&output)?)?;
		let expected = values.map(|value| match operation {
			"copy" | "abs" => 1.0,
			"scale" => -2.0,
			"log" => 1.0 / value,
			"sqrt" => 0.5 / value.sqrt(),
			"exp" => value.exp(),
			"reciprocal" => -1.0 / (value * value),
			"clamp_min" => {
				if value >= 1.0 {
					1.0
				} else {
					0.0
				}
			}
			"clamp_max" => {
				if value <= 2.0 {
					1.0
				} else {
					0.0
				}
			}
			_ => unreachable!(),
		});
		assert_adjoint(&layer.bias().expect("test bias"), &expected)?;
	}
	Ok(())
});

test_vk!(
	matrix_binary_adjoints_reduce_broadcast_axes_and_shared_inputs,
	engine,
	{
		for operation in ["add", "sub", "mul", "div"] {
			let left = oa::ml::nn::Linear::new_from_matrices(
				oa::Matrix::new_from_f32(&engine, [1, 1], &[1.0])?,
				oa::Matrix::new_from_f32(&engine, [1], &[0.0])?,
			)?;
			let right = oa::ml::nn::Linear::new_from_matrices(
				oa::Matrix::new_from_f32(&engine, [3, 1], &[0.0; 3])?,
				oa::Matrix::new_from_f32(&engine, [3], &[1.0, 2.0, 4.0])?,
			)?;
			let left_input = oa::Matrix::new_from_f32(&engine, [2, 1], &[2.0, 3.0])?;
			let right_input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
			let tape = oa::ml::GradientTape::new();
			let a = left.forward(&left_input)?;
			let b = right.forward(&right_input)?;
			let (output, left_expected, right_expected) = match operation {
				"add" => (oa::matrix::add(&a, &b)?, 15.0, [2.0; 3]),
				"sub" => (oa::matrix::sub(&a, &b)?, 15.0, [-2.0; 3]),
				"mul" => (oa::matrix::mul(&a, &b)?, 35.0, [5.0; 3]),
				"div" => (oa::matrix::div(&a, &b)?, 8.75, [-5.0, -1.25, -0.3125]),
				_ => unreachable!(),
			};
			tape.backward(&scalar_sum(&output)?)?;
			assert_adjoint(&left.weight(), &[left_expected])?;
			assert_adjoint(&right.bias().expect("test bias"), &right_expected)?;
		}
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, 1], &[0.0; 3])?,
			oa::Matrix::new_from_f32(&engine, [3], &[-1.0, 0.0, 2.0])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?;
		let square = oa::matrix::mul(&value, &value)?;
		let magnitude = oa::matrix::abs(&value)?;
		let output = oa::matrix::add(&square, &magnitude)?;
		tape.backward(&scalar_sum(&output)?)?;
		assert_adjoint(&layer.bias().expect("test bias"), &[-3.0, 0.0, 5.0])?;
		Ok(())
	}
);

test_vk!(
	matrix_mean_and_max_adjoints_match_analytic_oracles,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, 1], &[0.0; 3])?,
			oa::Matrix::new_from_f32(&engine, [3], &[-1.0, 0.0, 2.0])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?;
		let output = oa::matrix::mean(&value, -1)?.reshape([])?;
		tape.backward(&output)?;
		assert_adjoint(&layer.bias().expect("test bias"), &[1.0 / 3.0; 3])?;
		Ok(())
	}
);

test_vk!(
	matrix_max_adjoint_routes_the_upstream_scalar_to_the_maximum,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, 1], &[0.0; 3])?,
			oa::Matrix::new_from_f32(&engine, [3], &[-1.0, 0.0, 2.0])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?;
		let output = oa::matrix::max(&value, -1)?.reshape([])?;
		tape.backward(&output)?;
		assert_adjoint(&layer.bias().expect("test bias"), &[0.0, 0.0, 1.0])?;
		Ok(())
	}
);

test_vk!(
	matrix_causal_mask_adjoint_zeroes_above_diagonal_gradients,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [6, 1], &[0.0; 6])?,
			oa::Matrix::new_from_f32(&engine, [6], &[1.0; 6])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?.reshape([2, 3])?;
		let output = oa::matrix::causal_mask(&value)?;
		tape.backward(&scalar_sum(&output)?)?;
		assert_adjoint(
			&layer.bias().expect("test bias"),
			&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
		)?;
		Ok(())
	}
);

test_vk!(
	matrix_split_adjoint_routes_gradients_to_the_consumed_segment,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [6, 1], &[0.0; 6])?,
			oa::Matrix::new_from_f32(&engine, [6], &[1.0; 6])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?.reshape([2, 3])?;
		let parts = oa::matrix::split(&value, &[1, 2], 1)?;
		tape.backward(&scalar_sum(&parts[1])?)?;
		assert_adjoint(
			&layer.bias().expect("test bias"),
			&[0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
		)?;
		Ok(())
	}
);

test_vk!(
	matrix_matmul_concat_transpose_and_view_adjoints_preserve_lineage,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?,
			oa::Matrix::new_from_f32(&engine, [3], &[0.0; 3])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let right = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, -1.0, 2.0, 0.5, 2.0, 1.0])?;
		let empty = oa::Matrix::new_from_f32(&engine, [0, 2], &[])?;
		let tape = oa::ml::GradientTape::new();
		let left = layer.forward(&input)?;
		let product = oa::matrix::mat_mul_nt(&left, &right)?;
		let combined = oa::matrix::concat(&[product.clone(), empty, product], 0)?;
		let transposed = oa::matrix::transpose(&combined, 0, 1)?;
		let flattened = transposed.reshape([8])?;
		tape.backward(&scalar_sum(&flattened)?)?;
		assert_adjoint(&layer.weight(), &[3.0, 3.0, 2.0, 2.0, 6.0, 6.0])?;
		assert_adjoint(&layer.bias().expect("test bias"), &[6.0, 4.0, 12.0])?;
		Ok(())
	}
);

test_vk!(
	matrix_compact_rows_adjoint_reconstructs_the_source_rows,
	engine,
	{
		let layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [8, 1], &[0.0; 8])?,
			oa::Matrix::new_from_f32(&engine, [8], &[1.0; 8])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let mask = oa::Matrix::new_from_f32(&engine, [4], &[1.0, 0.0, 1.0, 1.0])?;
		let tape = oa::ml::GradientTape::new();
		let value = layer.forward(&input)?.reshape([4, 2])?;
		let packed = oa::matrix::compact_rows(&value, &mask)?;
		tape.backward(&scalar_sum(&packed.values)?)?;
		assert_adjoint(
			&layer.bias().expect("test bias"),
			&[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
		)?;
		Ok(())
	}
);

test_vk!(
	matrix_scatter_rows_adjoint_gathers_source_and_flows_base,
	engine,
	{
		let base_layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [8, 1], &[0.0; 8])?,
			oa::Matrix::new_from_f32(&engine, [8], &[0.0; 8])?,
		)?;
		let source_layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [8, 1], &[0.0; 8])?,
			oa::Matrix::new_from_f32(&engine, [8], &[0.0; 8])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let mask = oa::Matrix::new_from_f32(&engine, [4], &[1.0, 0.0, 1.0, 0.0])?;
		let keys = oa::matrix::compact_rows(
			&oa::Matrix::new_from_f32(&engine, [4, 2], &[0.0; 8])?,
			&mask,
		)?;
		let tape = oa::ml::GradientTape::new();
		let base = base_layer.forward(&input)?.reshape([4, 2])?;
		let source = source_layer.forward(&input)?.reshape([4, 2])?;
		let output = oa::matrix::scatter_rows(&base, &source, &keys.row_map, &keys.count)?;
		tape.backward(&scalar_sum(&output)?)?;
		assert_adjoint(&base_layer.bias().expect("test bias"), &[1.0; 8])?;
		assert_adjoint(
			&source_layer.bias().expect("test bias"),
			&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		)?;
		Ok(())
	}
);

test_vk!(
	matrix_scatter_rows_indirect_adjoint_uses_the_gpu_authored_plan,
	engine,
	{
		let base_layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [8, 1], &[0.0; 8])?,
			oa::Matrix::new_from_f32(&engine, [8], &[0.0; 8])?,
		)?;
		let source_layer = oa::ml::nn::Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [8, 1], &[0.0; 8])?,
			oa::Matrix::new_from_f32(&engine, [8], &[0.0; 8])?,
		)?;
		let input = oa::Matrix::new_from_f32(&engine, [1, 1], &[0.0])?;
		let mask = oa::Matrix::new_from_f32(&engine, [4], &[1.0, 0.0, 1.0, 0.0])?;
		let keys = oa::matrix::compact_rows(
			&oa::Matrix::new_from_f32(&engine, [4, 2], &[0.0; 8])?,
			&mask,
		)?;
		let tape = oa::ml::GradientTape::new();
		let base = base_layer.forward(&input)?.reshape([4, 2])?;
		let source = source_layer.forward(&input)?.reshape([4, 2])?;
		let output = oa::matrix::scatter_rows_indirect(&base, &source, &keys)?;
		tape.backward(&scalar_sum(&output)?)?;
		assert_adjoint(&base_layer.bias().expect("test bias"), &[1.0; 8])?;
		assert_adjoint(
			&source_layer.bias().expect("test bias"),
			&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
		)?;
		Ok(())
	}
);
