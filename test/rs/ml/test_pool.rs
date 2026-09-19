use oa::ml::Module as _;

fn host_avg_pool_2d(
	input: &[f32],
	shape: [usize; 4],
	kernel_size: usize,
	stride: usize,
	padding: usize,
) -> ([usize; 4], Vec<f32>) {
	let [batch, channels, input_height, input_width] = shape;
	let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
	let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
	let output_shape = [batch, channels, output_height, output_width];
	let mut output = vec![0.0; batch * channels * output_height * output_width];
	for batch_index in 0..batch {
		for channel in 0..channels {
			for output_y in 0..output_height {
				for output_x in 0..output_width {
					let input_y_start = output_y as isize * stride as isize - padding as isize;
					let input_x_start = output_x as isize * stride as isize - padding as isize;
					let mut sum = 0.0;
					let mut count = 0_usize;
					for kernel_y in 0..kernel_size {
						for kernel_x in 0..kernel_size {
							let input_y = input_y_start + kernel_y as isize;
							let input_x = input_x_start + kernel_x as isize;
							if input_y >= 0
								&& input_y < input_height as isize
								&& input_x >= 0
								&& input_x < input_width as isize
							{
								let input_index = ((batch_index * channels + channel) * input_height
									+ input_y as usize)
									* input_width
									+ input_x as usize;
								sum += input[input_index];
								count += 1;
							}
						}
					}
					let output_index = ((batch_index * channels + channel) * output_height + output_y)
						* output_width
						+ output_x;
					output[output_index] = sum / count as f32;
				}
			}
		}
	}
	(output_shape, output)
}

fn host_pool_mse(
	input: &[f32],
	shape: [usize; 4],
	kernel_size: usize,
	stride: usize,
	padding: usize,
	target: &[f32],
) -> f32 {
	let (_, output) = host_avg_pool_2d(input, shape, kernel_size, stride, padding);
	output
		.iter()
		.zip(target)
		.map(|(actual, target)| (actual - target).powi(2))
		.sum::<f32>()
		/ output.len() as f32
}

fn host_adaptive_avg_pool_2d(
	input: &[f32],
	shape: [usize; 4],
	output_height: usize,
	output_width: usize,
) -> Vec<f32> {
	let [batch, channels, input_height, input_width] = shape;
	let mut output = vec![0.0; batch * channels * output_height * output_width];
	for batch_index in 0..batch {
		for channel in 0..channels {
			for output_y in 0..output_height {
				let input_y_start = output_y * input_height / output_height;
				let input_y_end = ((output_y + 1) * input_height).div_ceil(output_height);
				for output_x in 0..output_width {
					let input_x_start = output_x * input_width / output_width;
					let input_x_end = ((output_x + 1) * input_width).div_ceil(output_width);
					let mut sum = 0.0;
					for input_y in input_y_start..input_y_end {
						for input_x in input_x_start..input_x_end {
							let input_index = ((batch_index * channels + channel) * input_height + input_y)
								* input_width
								+ input_x;
							sum += input[input_index];
						}
					}
					let count = (input_y_end - input_y_start) * (input_x_end - input_x_start);
					let output_index = ((batch_index * channels + channel) * output_height + output_y)
						* output_width
						+ output_x;
					output[output_index] = sum / count as f32;
				}
			}
		}
	}
	output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(avg_pool_2d_matches_donor_and_odd_padding_oracles, engine, {
	let donor_input = oa::Matrix::from_f32(
		&engine,
		[1, 1, 4, 4],
		&[
			1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
		],
	)?;
	let donor_output = oa::ml::matrix::avg_pool_2d(&donor_input, 2, 2, 0)?;
	assert_eq!(donor_output.shape(), [1, 1, 2, 2]);
	assert_eq!(donor_output.read_f32()?, [3.5, 5.5, 11.5, 13.5]);

	let shape = [1, 2, 4, 5];
	let values = (0..40)
		.map(|index| index as f32 * 0.25 - 3.0)
		.collect::<Vec<_>>();
	let input = oa::Matrix::from_f32(&engine, shape, &values)?;
	let (expected_shape, expected) = host_avg_pool_2d(&values, shape, 3, 2, 1);
	let (plan, output) = engine.capture(|| oa::ml::matrix::avg_pool_2d(&input, 3, 2, 1))?;
	assert_eq!(output.shape(), expected_shape);
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	let operation = &plan.semantic_graph().operations()[0];
	assert_eq!(operation.name(), "oa::ml::matrix::avg_pool_2d");
	assert_eq!(operation.attributes().len(), 3);
	assert_eq!(operation.attributes()[0].name(), "kernel_size");
	assert_eq!(operation.attributes()[1].name(), "stride");
	assert_eq!(operation.attributes()[2].name(), "padding");
	engine.submit(&plan)?.wait()?;
	assert_close(&output.read_f32()?, &expected, 1.0e-5);
	Ok(())
});

test_vk!(avg_pool_2d_module_and_adjoint_match_host_oracle, engine, {
	const SHAPE: [usize; 4] = [1, 1, 3, 4];
	let input_values = [
		0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, -1.2, 0.5, 0.9, 1.4, -0.2, 0.1,
	];
	let target_values = [-0.5_f32, 0.2, 0.9, -0.1, 0.7, -0.8];
	let embedding =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 4], &input_values)?)?;
	let indices = oa::Matrix::from_slice(&engine, [1, 1, 3], &[0_u32, 1, 2])?;
	let target = oa::Matrix::from_f32(&engine, [1, 1, 2, 3], &target_values)?;
	let pool = oa::ml::nn::AvgPool2d::with_options(2, 2, 1)?;
	assert_eq!(pool.kernel_size(), 2);
	assert_eq!(pool.stride(), 2);
	assert_eq!(pool.padding(), 1);
	assert!(pool.parameters().is_empty());

	let tape = oa::ml::GradientTape::new();
	let output = pool.forward(&embedding.forward(&indices)?)?;
	assert_eq!(output.shape(), [1, 1, 2, 3]);
	let loss = oa::ml::loss::mse(&output, &target)?;
	tape.backward(&loss)?;
	let actual = embedding
		.weight()
		.gradient()
		.expect("AvgPool2d input adjoint is missing")
		.read_f32()?;

	const DELTA: f32 = 1.0e-3;
	let expected = (0..input_values.len())
		.map(|index| {
			let mut below = input_values;
			let mut above = input_values;
			below[index] -= DELTA;
			above[index] += DELTA;
			(host_pool_mse(&above, SHAPE, 2, 2, 1, &target_values)
				- host_pool_mse(&below, SHAPE, 2, 2, 1, &target_values))
				/ (2.0 * DELTA)
		})
		.collect::<Vec<_>>();
	assert_close(&actual, &expected, 2.0e-4);
	Ok(())
});

test_vk!(avg_pool_2d_rejects_invalid_contracts, engine, {
	let rank_two = oa::Matrix::from_f32(&engine, [2, 2], &[1.0; 4])?;
	let integer = oa::Matrix::from_slice(&engine, [1, 1, 2, 2], &[1_i32; 4])?;
	let input = oa::Matrix::from_f32(&engine, [1, 1, 2, 2], &[1.0; 4])?;
	for result in [
		oa::ml::matrix::avg_pool_2d(&rank_two, 2, 2, 0),
		oa::ml::matrix::avg_pool_2d(&integer, 2, 2, 0),
		oa::ml::matrix::avg_pool_2d(&input, 0, 1, 0),
		oa::ml::matrix::avg_pool_2d(&input, 2, 0, 0),
		oa::ml::matrix::avg_pool_2d(&input, 3, 1, 0),
	] {
		let error = match result {
			Ok(_) => panic!("invalid AvgPool2d contract was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	assert_eq!(oa::ml::nn::AvgPool2d::new(2)?.stride(), 2);
	assert_eq!(
		oa::ml::nn::AvgPool2d::with_options(1, 0, 0)
			.err()
			.expect("zero-stride AvgPool2d module was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(
	max_pool_2d_values_indices_and_adjoint_match_donor,
	engine,
	{
		let input_values = [
			1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
		];
		let input = oa::Matrix::from_f32(&engine, [1, 1, 4, 4], &input_values)?;
		let result = oa::ml::matrix::max_pool_2d(&input, 2, 2, 0)?;
		assert_eq!(result.output.shape(), [1, 1, 2, 2]);
		assert_eq!(result.indices.dtype(), oa::DType::U32);
		assert_eq!(result.output.read_f32()?, [6.0, 8.0, 14.0, 16.0]);
		assert_eq!(result.indices.read::<u32>()?, [5, 7, 13, 15]);
		let tie_input = oa::Matrix::from_f32(&engine, [1, 1, 2, 2], &[5.0, 5.0, 1.0, 0.0])?;
		let tie = oa::ml::matrix::max_pool_2d(&tie_input, 2, 2, 0)?;
		assert_eq!(tie.output.read_f32()?, [5.0]);
		assert_eq!(tie.indices.read::<u32>()?, [0]);

		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [4, 4], &input_values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 1, 4], &[0_u32, 1, 2, 3])?;
		let target = oa::Matrix::from_f32(&engine, [1, 1, 2, 2], &[0.0; 4])?;
		let module = oa::ml::nn::MaxPool2d::new(2)?;
		assert_eq!(module.kernel_size(), 2);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 0);
		assert!(module.parameters().is_empty());
		let tape = oa::ml::GradientTape::new();
		let pooled = module.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&pooled, &target)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("MaxPool2d input adjoint is missing")
			.read_f32()?;
		let mut expected = [0.0_f32; 16];
		expected[5] = 3.0;
		expected[7] = 4.0;
		expected[13] = 7.0;
		expected[15] = 8.0;
		assert_eq!(gradient, expected);
		Ok(())
	}
);

test_vk!(
	adaptive_avg_pool_2d_rectangular_forward_and_adjoint_match_oracles,
	engine,
	{
		let shape = [1, 1, 3, 4];
		let input_values = [
			0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, -1.2, 0.5, 0.9, 1.4, -0.2, 0.1,
		];
		let target_values = [-0.5_f32, 0.2, 0.9, -0.1, 0.7, -0.8];
		let input = oa::Matrix::from_f32(&engine, shape, &input_values)?;
		let output = oa::ml::matrix::adaptive_avg_pool_2d(&input, 2, 3)?;
		assert_eq!(output.shape(), [1, 1, 2, 3]);
		assert_close(
			&output.read_f32()?,
			&host_adaptive_avg_pool_2d(&input_values, shape, 2, 3),
			1.0e-6,
		);

		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 4], &input_values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 1, 3], &[0_u32, 1, 2])?;
		let target = oa::Matrix::from_f32(&engine, [1, 1, 2, 3], &target_values)?;
		let pool = oa::ml::nn::AdaptiveAvgPool2d::with_output_size(2, 3)?;
		assert_eq!(pool.output_height(), 2);
		assert_eq!(pool.output_width(), 3);
		assert!(pool.parameters().is_empty());
		let tape = oa::ml::GradientTape::new();
		let pooled = pool.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&pooled, &target)?;
		tape.backward(&loss)?;
		let actual = embedding
			.weight()
			.gradient()
			.expect("AdaptiveAvgPool2d input adjoint is missing")
			.read_f32()?;
		const DELTA: f32 = 1.0e-3;
		let expected = (0..input_values.len())
			.map(|index| {
				let mut below = input_values;
				let mut above = input_values;
				below[index] -= DELTA;
				above[index] += DELTA;
				let below = host_adaptive_avg_pool_2d(&below, shape, 2, 3);
				let above = host_adaptive_avg_pool_2d(&above, shape, 2, 3);
				let mse = |values: &[f32]| {
					values
						.iter()
						.zip(target_values)
						.map(|(actual, target)| (actual - target).powi(2))
						.sum::<f32>()
						/ target_values.len() as f32
				};
				(mse(&above) - mse(&below)) / (2.0 * DELTA)
			})
			.collect::<Vec<_>>();
		assert_close(&actual, &expected, 2.0e-4);
		assert_eq!(oa::ml::nn::AdaptiveAvgPool2d::new(1)?.output_width(), 1);
		Ok(())
	}
);
