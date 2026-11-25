use oa::ml::Module as _;

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps convolution geometry explicit"
)]
fn host_conv_1d(
	input: &[f32],
	input_shape: [usize; 3],
	weight: &[f32],
	weight_shape: [usize; 3],
	bias: &[f32],
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Vec<f32> {
	let [batch_size, input_channels, input_length] = input_shape;
	let [output_channels, weight_input_channels, kernel_size] = weight_shape;
	assert_eq!(weight_input_channels, input_channels);
	let effective_kernel = dilation * (kernel_size - 1) + 1;
	let output_length = (input_length + 2 * padding - effective_kernel) / stride + 1;
	let mut output = vec![0.0; batch_size * output_channels * output_length];
	for batch in 0..batch_size {
		for (output_channel, &bias_value) in bias.iter().enumerate().take(output_channels) {
			for output_position in 0..output_length {
				let mut value = f64::from(bias_value);
				for input_channel in 0..input_channels {
					for kernel in 0..kernel_size {
						let padded_position = output_position * stride + kernel * dilation;
						if padded_position < padding {
							continue;
						}
						let input_position = padded_position - padding;
						if input_position >= input_length {
							continue;
						}
						let input_offset =
							(batch * input_channels + input_channel) * input_length + input_position;
						let weight_offset =
							(output_channel * input_channels + input_channel) * kernel_size + kernel;
						value += f64::from(input[input_offset]) * f64::from(weight[weight_offset]);
					}
				}
				let output_offset =
					(batch * output_channels + output_channel) * output_length + output_position;
				output[output_offset] = value as f32;
			}
		}
	}
	output
}

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps transposed-convolution geometry explicit"
)]
fn host_conv_1d_transpose(
	input: &[f32],
	input_shape: [usize; 3],
	weight: &[f32],
	weight_shape: [usize; 3],
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Vec<f32> {
	let [batch_size, input_channels, input_length] = input_shape;
	let [weight_input_channels, output_channels, kernel_size] = weight_shape;
	assert_eq!(weight_input_channels, input_channels);
	let effective_kernel = dilation * (kernel_size - 1) + 1;
	let output_length = (input_length - 1) * stride + effective_kernel - 2 * padding;
	let mut output = vec![0.0; batch_size * output_channels * output_length];
	for batch in 0..batch_size {
		for input_channel in 0..input_channels {
			for input_position in 0..input_length {
				let input_offset = (batch * input_channels + input_channel) * input_length + input_position;
				for output_channel in 0..output_channels {
					for kernel in 0..kernel_size {
						let padded_position = input_position * stride + kernel * dilation;
						if padded_position < padding {
							continue;
						}
						let output_position = padded_position - padding;
						if output_position >= output_length {
							continue;
						}
						let weight_offset =
							(input_channel * output_channels + output_channel) * kernel_size + kernel;
						let output_offset =
							(batch * output_channels + output_channel) * output_length + output_position;
						output[output_offset] += input[input_offset] * weight[weight_offset];
					}
				}
			}
		}
	}
	output
}

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps transposed-convolution geometry explicit"
)]
fn host_conv_2d_transpose(
	input: &[f32],
	input_shape: [usize; 4],
	weight: &[f32],
	weight_shape: [usize; 4],
	bias: &[f32],
	stride: usize,
	padding: usize,
) -> Vec<f32> {
	let [batch_size, input_channels, input_height, input_width] = input_shape;
	let [
		weight_input_channels,
		output_channels,
		kernel_size,
		kernel_width,
	] = weight_shape;
	assert_eq!(weight_input_channels, input_channels);
	assert_eq!(kernel_size, kernel_width);
	let output_height = (input_height - 1) * stride + kernel_size - 2 * padding;
	let output_width = (input_width - 1) * stride + kernel_size - 2 * padding;
	let mut output = vec![0.0_f64; batch_size * output_channels * output_height * output_width];
	for batch in 0..batch_size {
		for (output_channel, &bias_value) in bias.iter().enumerate().take(output_channels) {
			for output_y in 0..output_height {
				for output_x in 0..output_width {
					let output_offset =
						((batch * output_channels + output_channel) * output_height + output_y) * output_width
							+ output_x;
					output[output_offset] = f64::from(bias_value);
				}
			}
		}
		for input_channel in 0..input_channels {
			for input_y in 0..input_height {
				for input_x in 0..input_width {
					let input_offset = ((batch * input_channels + input_channel) * input_height + input_y)
						* input_width
						+ input_x;
					for output_channel in 0..output_channels {
						for kernel_y in 0..kernel_size {
							let padded_y = input_y * stride + kernel_y;
							if padded_y < padding {
								continue;
							}
							let output_y = padded_y - padding;
							if output_y >= output_height {
								continue;
							}
							for kernel_x in 0..kernel_size {
								let padded_x = input_x * stride + kernel_x;
								if padded_x < padding {
									continue;
								}
								let output_x = padded_x - padding;
								if output_x >= output_width {
									continue;
								}
								let weight_offset =
									((input_channel * output_channels + output_channel) * kernel_size + kernel_y)
										* kernel_size
										+ kernel_x;
								let output_offset = ((batch * output_channels + output_channel) * output_height
									+ output_y)
									* output_width
									+ output_x;
								output[output_offset] +=
									f64::from(input[input_offset]) * f64::from(weight[weight_offset]);
							}
						}
					}
				}
			}
		}
	}
	output.into_iter().map(|value| value as f32).collect()
}

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps convolution geometry explicit"
)]
fn host_conv_2d(
	input: &[f32],
	input_shape: [usize; 4],
	weight: &[f32],
	weight_shape: [usize; 4],
	bias: &[f32],
	stride: usize,
	padding: usize,
	groups: usize,
) -> Vec<f32> {
	let [batch_size, input_channels, input_height, input_width] = input_shape;
	let [
		output_channels,
		input_channels_per_group,
		kernel_size,
		kernel_width,
	] = weight_shape;
	assert_eq!(kernel_size, kernel_width);
	assert_eq!(input_channels_per_group, input_channels / groups);
	let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
	let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
	let output_channels_per_group = output_channels / groups;
	let mut output = vec![0.0; batch_size * output_channels * output_height * output_width];
	for batch in 0..batch_size {
		for (output_channel, &bias_value) in bias.iter().enumerate().take(output_channels) {
			let group = output_channel / output_channels_per_group;
			for output_y in 0..output_height {
				for output_x in 0..output_width {
					let mut value = f64::from(bias_value);
					for local_input_channel in 0..input_channels_per_group {
						let input_channel = group * input_channels_per_group + local_input_channel;
						for kernel_y in 0..kernel_size {
							for kernel_x in 0..kernel_size {
								let input_y = output_y * stride + kernel_y;
								let input_x = output_x * stride + kernel_x;
								if input_y < padding || input_x < padding {
									continue;
								}
								let input_y = input_y - padding;
								let input_x = input_x - padding;
								if input_y >= input_height || input_x >= input_width {
									continue;
								}
								let input_offset =
									((batch * input_channels + input_channel) * input_height + input_y) * input_width
										+ input_x;
								let weight_offset =
									((output_channel * input_channels_per_group + local_input_channel) * kernel_size
										+ kernel_y)
										* kernel_size
										+ kernel_x;
								value += f64::from(input[input_offset]) * f64::from(weight[weight_offset]);
							}
						}
					}
					let output_offset =
						((batch * output_channels + output_channel) * output_height + output_y) * output_width
							+ output_x;
					output[output_offset] = value as f32;
				}
			}
		}
	}
	output
}

fn mse(values: &[f32], target: &[f32]) -> f32 {
	values
		.iter()
		.zip(target)
		.map(|(actual, target)| (actual - target).powi(2))
		.sum::<f32>()
		/ values.len() as f32
}

fn numerical_gradient(values: &[f32], loss: impl Fn(&[f32]) -> f32) -> Vec<f32> {
	const DELTA: f32 = 1.0e-3;
	(0..values.len())
		.map(|index| {
			let mut below = values.to_vec();
			let mut above = values.to_vec();
			below[index] -= DELTA;
			above[index] += DELTA;
			(loss(&above) - loss(&below)) / (2.0 * DELTA)
		})
		.collect()
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

test_vk!(conv_1d_gemm_forward_matches_independent_oracle, runtime, {
	let input_shape = [2, 3, 9];
	let weight_shape = [4, 3, 3];
	let input_values = (0..54)
		.map(|index| ((index * 5 % 23) as f32 - 11.0) * 0.04)
		.collect::<Vec<_>>();
	let weight_values = (0..36)
		.map(|index| ((index * 7 % 19) as f32 - 9.0) * 0.03)
		.collect::<Vec<_>>();
	let bias_values = [0.1_f32, -0.2, 0.3, -0.4];
	let expected = host_conv_1d(
		&input_values,
		input_shape,
		&weight_values,
		weight_shape,
		&bias_values,
		2,
		2,
		2,
	);
	let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
	let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
	let bias = oa::Matrix::new_from_f32([4], &bias_values)?;
	let (plan, output) =
		runtime.capture(|| oa::ml::matrix::conv_1d(&input, &weight, &bias, 2, 2, 2))?;
	assert_eq!(output.shape(), [2, 4, 5]);
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::matrix::conv_1d"
	);
	assert_eq!(plan.semantic_lowering().maximum_nodes_per_op(), 3);
	runtime.submit(&plan)?.wait()?;
	assert_close(&output.read_f32()?, &expected, 3.0e-5);
	Ok(())
});

test_vk!(
	conv_1d_dilated_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 6];
		let weight_shape = [2, 2, 3];
		let input_values = [
			0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 1.2, -0.1, 0.5, -0.8, 0.6, 0.9,
		];
		let weight_values = [
			0.3_f32, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.3, 0.25, -0.15, 0.45, -0.35,
		];
		let bias_values = [0.15_f32, -0.25];
		let target_values = [0.4_f32, -0.3, 0.2, 0.6, -0.5, 0.7];
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			mse(
				&host_conv_1d(input, input_shape, weight, weight_shape, bias, 2, 2, 2),
				&target_values,
			)
		};
		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(values, &weight_values, &bias_values)
		});
		let expected_weight = numerical_gradient(&weight_values, |values| {
			host_loss(&input_values, values, &bias_values)
		});
		let expected_bias = numerical_gradient(&bias_values, |values| {
			host_loss(&input_values, &weight_values, values)
		});

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 6], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2], &[0_u32, 1])?;
		let module = oa::ml::nn::Conv1d::new_from_matrices(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			oa::Matrix::new_from_f32([2], &bias_values)?,
			2,
			2,
			2,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 3], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("Conv1d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("Conv1d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("Conv1d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_1d_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::Conv1d::new_with_seed(3, 5, 3, 2, 1, 2, 0x434f_4e31)?;
		assert_eq!(module.input_channels(), 3);
		assert_eq!(module.output_channels(), 5);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.dilation(), 2);
		assert_eq!(module.weight().data().shape(), [5, 3, 3]);
		assert_eq!(module.bias().data().shape(), [5]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		assert!(oa::ml::nn::Conv1d::new_with_seed(2, 2, 3, 0, 0, 1, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 2, 3], &[0.0; 6])?;
		let weight = oa::Matrix::new_from_f32([2, 2, 5], &[0.0; 20])?;
		let bias = oa::Matrix::new_from_f32([2], &[0.0; 2])?;
		assert!(oa::ml::matrix::conv_1d(&input, &weight, &bias, 1, 0, 1).is_err());
		Ok(())
	}
);

test_vk!(
	conv_1d_transpose_forward_matches_independent_oracle,
	runtime,
	{
		let input_shape = [2, 2, 4];
		let weight_shape = [2, 3, 3];
		let input_values = (0..16)
			.map(|index| ((index * 5 % 17) as f32 - 8.0) * 0.08)
			.collect::<Vec<_>>();
		let weight_values = (0..18)
			.map(|index| ((index * 7 % 13) as f32 - 6.0) * 0.05)
			.collect::<Vec<_>>();
		let expected = host_conv_1d_transpose(
			&input_values,
			input_shape,
			&weight_values,
			weight_shape,
			2,
			1,
			2,
		);
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
		let (plan, output) =
			runtime.capture(|| oa::ml::matrix::conv_1d_transpose(&input, &weight, 2, 1, 2))?;
		assert_eq!(output.shape(), [2, 3, 9]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::conv_1d_transpose"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 3.0e-5);
		Ok(())
	}
);

test_vk!(
	conv_1d_transpose_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 3];
		let weight_shape = [2, 2, 3];
		let input_values = [0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4];
		let weight_values = [
			0.3_f32, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.3, 0.25, -0.15, 0.45, -0.35,
		];
		let target_values = [0.4_f32, -0.3, 0.2, 0.6, -0.5, 0.7, 0.1, -0.2, 0.8, -0.6];
		let host_loss = |input: &[f32], weight: &[f32]| {
			mse(
				&host_conv_1d_transpose(input, input_shape, weight, weight_shape, 2, 1, 1),
				&target_values,
			)
		};
		let expected_input =
			numerical_gradient(&input_values, |values| host_loss(values, &weight_values));
		let expected_weight =
			numerical_gradient(&weight_values, |values| host_loss(&input_values, values));

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 3], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2], &[0_u32, 1])?;
		let module = oa::ml::nn::ConvTranspose1d::new_from_matrix(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			2,
			1,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 5], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("ConvTranspose1d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("ConvTranspose1d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_1d_transpose_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::ConvTranspose1d::new_with_seed(3, 5, 3, 2, 1, 0x4354_4e31)?;
		assert_eq!(module.input_channels(), 3);
		assert_eq!(module.output_channels(), 5);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.weight().data().shape(), [3, 5, 3]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight"]
		);
		assert!(oa::ml::nn::ConvTranspose1d::new_with_seed(2, 2, 3, 0, 0, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 2, 1], &[0.0; 2])?;
		let weight = oa::Matrix::new_from_f32([2, 2, 1], &[0.0; 4])?;
		assert!(oa::ml::matrix::conv_1d_transpose(&input, &weight, 1, 1, 1).is_err());
		Ok(())
	}
);

test_vk!(
	conv_2d_transpose_forward_matches_independent_oracle,
	runtime,
	{
		let input_shape = [1, 2, 2, 3];
		let weight_shape = [2, 3, 2, 2];
		let input_values = (0..12)
			.map(|index| ((index * 5 % 17) as f32 - 8.0) * 0.08)
			.collect::<Vec<_>>();
		let weight_values = (0..24)
			.map(|index| ((index * 7 % 19) as f32 - 9.0) * 0.04)
			.collect::<Vec<_>>();
		let bias_values = [0.1_f32, -0.2, 0.3];
		let expected = host_conv_2d_transpose(
			&input_values,
			input_shape,
			&weight_values,
			weight_shape,
			&bias_values,
			2,
			1,
		);
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
		let bias = oa::Matrix::new_from_f32([3], &bias_values)?;
		let (plan, output) =
			runtime.capture(|| oa::ml::matrix::conv_2d_transpose(&input, &weight, &bias, 2, 1))?;
		assert_eq!(output.shape(), [1, 3, 2, 4]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::conv_2d_transpose"
		);
		assert_eq!(plan.semantic_lowering().maximum_nodes_per_op(), 2);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 3.0e-5);
		Ok(())
	}
);

test_vk!(
	conv_2d_transpose_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 2, 2];
		let weight_shape = [2, 2, 2, 2];
		let input_values = [0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 0.6, 0.9];
		let weight_values = [
			0.3_f32, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.3, 0.25, -0.15, 0.45, -0.35, 0.2, 0.4, -0.1, 0.55,
		];
		let bias_values = [0.15_f32, -0.25];
		let target_values = (0..32)
			.map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.03)
			.collect::<Vec<_>>();
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			mse(
				&host_conv_2d_transpose(input, input_shape, weight, weight_shape, bias, 2, 0),
				&target_values,
			)
		};
		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(values, &weight_values, &bias_values)
		});
		let expected_weight = numerical_gradient(&weight_values, |values| {
			host_loss(&input_values, values, &bias_values)
		});
		let expected_bias = numerical_gradient(&bias_values, |values| {
			host_loss(&input_values, &weight_values, values)
		});

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([4, 2], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2, 2], &[0_u32, 1, 2, 3])?;
		let module = oa::ml::nn::ConvTranspose2d::new_from_matrices(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			oa::Matrix::new_from_f32([2], &bias_values)?,
			2,
			0,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 4, 4], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("ConvTranspose2d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("ConvTranspose2d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("ConvTranspose2d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_2d_transpose_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::ConvTranspose2d::new_with_seed(3, 5, 3, 2, 1, 0x4354_4e32)?;
		assert_eq!(module.input_channels(), 3);
		assert_eq!(module.output_channels(), 5);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.weight().data().shape(), [3, 5, 3, 3]);
		assert_eq!(module.bias().data().shape(), [5]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		assert!(oa::ml::nn::ConvTranspose2d::new_with_seed(2, 2, 3, 0, 0, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 2, 1, 1], &[0.0; 2])?;
		let weight = oa::Matrix::new_from_f32([2, 2, 1, 1], &[0.0; 4])?;
		let bias = oa::Matrix::new_from_f32([2], &[0.0; 2])?;
		assert!(oa::ml::matrix::conv_2d_transpose(&input, &weight, &bias, 1, 1).is_err());
		Ok(())
	}
);

test_vk!(
	conv_2d_grouped_forward_matches_independent_oracle,
	runtime,
	{
		let input_shape = [1, 4, 3, 4];
		let weight_shape = [4, 2, 2, 2];
		let input_values = (0..48)
			.map(|index| (index as f32 - 19.0) * 0.07)
			.collect::<Vec<_>>();
		let weight_values = (0..32)
			.map(|index| ((index * 7 % 17) as f32 - 8.0) * 0.05)
			.collect::<Vec<_>>();
		let bias_values = [0.1_f32, -0.2, 0.3, -0.4];
		let expected = host_conv_2d(
			&input_values,
			input_shape,
			&weight_values,
			weight_shape,
			&bias_values,
			2,
			1,
			2,
		);
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
		let bias = oa::Matrix::new_from_f32([4], &bias_values)?;
		let (plan, output) =
			runtime.capture(|| oa::ml::matrix::conv_2d(&input, &weight, &bias, 2, 1, 2))?;
		assert_eq!(output.shape(), [1, 4, 2, 3]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::conv_2d"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 2.0e-5);
		Ok(())
	}
);

test_vk!(
	conv_2d_depthwise_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 3, 3];
		let weight_shape = [2, 1, 2, 2];
		let input_values = [
			0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 1.2, -0.1, 0.5, -0.8, 0.6, 0.9, -0.3, 0.4, 1.5, -1.0,
			0.7, 0.1,
		];
		let weight_values = [0.3_f32, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.3];
		let bias_values = [0.15_f32, -0.25];
		let target_values = [0.4_f32, -0.3, 0.2, 0.6, -0.5, 0.7, 0.1, -0.2];
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			mse(
				&host_conv_2d(input, input_shape, weight, weight_shape, bias, 1, 0, 2),
				&target_values,
			)
		};
		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(values, &weight_values, &bias_values)
		});
		let expected_weight = numerical_gradient(&weight_values, |values| {
			host_loss(&input_values, values, &bias_values)
		});
		let expected_bias = numerical_gradient(&bias_values, |values| {
			host_loss(&input_values, &weight_values, values)
		});

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([6, 3], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2, 3], &[0_u32, 1, 2, 3, 4, 5])?;
		let module = oa::ml::nn::Conv2d::new_from_matrices(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			oa::Matrix::new_from_f32([2], &bias_values)?,
			1,
			0,
			2,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 2, 2], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let output = module.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("Conv2d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("Conv2d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("Conv2d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_2d_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::Conv2d::new_with_seed(4, 6, 3, 2, 1, 2, 0x434f_4e56)?;
		assert_eq!(module.input_channels(), 4);
		assert_eq!(module.output_channels(), 6);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.groups(), 2);
		assert_eq!(module.weight().data().shape(), [6, 2, 3, 3]);
		assert_eq!(module.bias().data().shape(), [6]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		assert!(oa::ml::nn::Conv2d::new_with_seed(3, 4, 3, 1, 0, 2, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 4, 2, 2], &[0.0; 16])?;
		let weight = oa::Matrix::new_from_f32([4, 2, 5, 5], &[0.0; 200])?;
		let bias = oa::Matrix::new_from_f32([4], &[0.0; 4])?;
		assert!(oa::ml::matrix::conv_2d(&input, &weight, &bias, 1, 0, 2).is_err());
		Ok(())
	}
);

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps convolution geometry explicit"
)]
fn host_conv_3d(
	input: &[f32],
	input_shape: [usize; 5],
	weight: &[f32],
	weight_shape: [usize; 5],
	bias: &[f32],
	stride: usize,
	padding: usize,
	groups: usize,
) -> Vec<f32> {
	let [
		batch_size,
		input_channels,
		input_depth,
		input_height,
		input_width,
	] = input_shape;
	let [
		output_channels,
		input_channels_per_group,
		kernel_size,
		kernel_height,
		kernel_width,
	] = weight_shape;
	assert_eq!(kernel_size, kernel_height);
	assert_eq!(kernel_size, kernel_width);
	assert_eq!(input_channels_per_group, input_channels / groups);
	let output_depth = (input_depth + 2 * padding - kernel_size) / stride + 1;
	let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
	let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
	let output_channels_per_group = output_channels / groups;
	let mut output =
		vec![0.0; batch_size * output_channels * output_depth * output_height * output_width];
	for batch in 0..batch_size {
		for (output_channel, &bias_value) in bias.iter().enumerate().take(output_channels) {
			let group = output_channel / output_channels_per_group;
			for output_z in 0..output_depth {
				for output_y in 0..output_height {
					for output_x in 0..output_width {
						let mut value = f64::from(bias_value);
						for local_input_channel in 0..input_channels_per_group {
							let input_channel = group * input_channels_per_group + local_input_channel;
							for kernel_z in 0..kernel_size {
								for kernel_y in 0..kernel_size {
									for kernel_x in 0..kernel_size {
										let input_z = output_z * stride + kernel_z;
										let input_y = output_y * stride + kernel_y;
										let input_x = output_x * stride + kernel_x;
										if input_z < padding || input_y < padding || input_x < padding {
											continue;
										}
										let input_z = input_z - padding;
										let input_y = input_y - padding;
										let input_x = input_x - padding;
										if input_z >= input_depth || input_y >= input_height || input_x >= input_width {
											continue;
										}
										let input_offset = (((batch * input_channels + input_channel) * input_depth
											+ input_z) * input_height
											+ input_y) * input_width
											+ input_x;
										let weight_offset = (((output_channel * input_channels_per_group
											+ local_input_channel)
											* kernel_size
											+ kernel_z) * kernel_size
											+ kernel_y) * kernel_size
											+ kernel_x;
										value += f64::from(input[input_offset]) * f64::from(weight[weight_offset]);
									}
								}
							}
						}
						let output_offset = (((batch * output_channels + output_channel) * output_depth
							+ output_z)
							* output_height
							+ output_y)
							* output_width
							+ output_x;
						output[output_offset] = value as f32;
					}
				}
			}
		}
	}
	output
}

test_vk!(
	conv_3d_grouped_forward_matches_independent_oracle,
	runtime,
	{
		let input_shape = [1, 4, 3, 4, 4];
		let weight_shape = [4, 2, 2, 2, 2];
		let input_values = (0..192)
			.map(|index| (index as f32 - 79.0) * 0.03)
			.collect::<Vec<_>>();
		let weight_values = (0..64)
			.map(|index| ((index * 7 % 17) as f32 - 8.0) * 0.05)
			.collect::<Vec<_>>();
		let bias_values = [0.1_f32, -0.2, 0.3, -0.4];
		let expected = host_conv_3d(
			&input_values,
			input_shape,
			&weight_values,
			weight_shape,
			&bias_values,
			2,
			1,
			2,
		);
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
		let bias = oa::Matrix::new_from_f32([4], &bias_values)?;
		let (plan, output) =
			runtime.capture(|| oa::ml::matrix::conv_3d(&input, &weight, &bias, 2, 1, 2))?;
		assert_eq!(output.shape(), [1, 4, 2, 3, 3]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::conv_3d"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 2.0e-5);
		Ok(())
	}
);

test_vk!(
	conv_3d_depthwise_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 3, 3, 3];
		let weight_shape = [2, 1, 2, 2, 2];
		let input_values = (0..54)
			.map(|index| ((index * 5 % 23) as f32 - 11.0) * 0.09)
			.collect::<Vec<_>>();
		let weight_values = [
			0.3_f32, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.3, -0.5, 0.2, -0.1, 0.4, 0.6, -0.7, 0.3, -0.2,
		];
		let bias_values = [0.15_f32, -0.25];
		let target_values = [
			0.4_f32, -0.3, 0.2, 0.6, -0.5, 0.7, 0.1, -0.2, -0.6, 0.3, -0.4, 0.5, 0.2, -0.1, 0.8, -0.7,
		];
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			mse(
				&host_conv_3d(input, input_shape, weight, weight_shape, bias, 1, 0, 2),
				&target_values,
			)
		};
		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(values, &weight_values, &bias_values)
		});
		let expected_weight = numerical_gradient(&weight_values, |values| {
			host_loss(&input_values, values, &bias_values)
		});
		let expected_bias = numerical_gradient(&bias_values, |values| {
			host_loss(&input_values, &weight_values, values)
		});

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([18, 3], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2, 3, 3], &(0_u32..18).collect::<Vec<_>>())?;
		let module = oa::ml::nn::Conv3d::new_from_matrices(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			oa::Matrix::new_from_f32([2], &bias_values)?,
			1,
			0,
			2,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 2, 2, 2], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let output = module.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("Conv3d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("Conv3d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("Conv3d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_3d_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::Conv3d::new_with_seed(4, 6, 3, 2, 1, 2, 0x434f_4e56)?;
		assert_eq!(module.input_channels(), 4);
		assert_eq!(module.output_channels(), 6);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.groups(), 2);
		assert_eq!(module.weight().data().shape(), [6, 2, 3, 3, 3]);
		assert_eq!(module.bias().data().shape(), [6]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		assert!(oa::ml::nn::Conv3d::new_with_seed(3, 4, 3, 1, 0, 2, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 4, 2, 2, 2], &[0.0; 32])?;
		let weight = oa::Matrix::new_from_f32([4, 2, 5, 5, 5], &[0.0; 1000])?;
		let bias = oa::Matrix::new_from_f32([4], &[0.0; 4])?;
		assert!(oa::ml::matrix::conv_3d(&input, &weight, &bias, 1, 0, 2).is_err());
		Ok(())
	}
);

fn host_unfold(
	input: &[f32],
	input_shape: [usize; 4],
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> (Vec<f32>, usize, usize) {
	let [batch_size, channels, input_height, input_width] = input_shape;
	let effective = dilation * (kernel_size - 1) + 1;
	let output_height = (input_height + 2 * padding - effective) / stride + 1;
	let output_width = (input_width + 2 * padding - effective) / stride + 1;
	let output_length = output_height * output_width;
	let column_size = channels * kernel_size * kernel_size;
	let mut columns = vec![0.0_f32; batch_size * column_size * output_length];
	for batch in 0..batch_size {
		for channel in 0..channels {
			for kernel_y in 0..kernel_size {
				for kernel_x in 0..kernel_size {
					let column = channel * kernel_size * kernel_size + kernel_y * kernel_size + kernel_x;
					for output_y in 0..output_height {
						for output_x in 0..output_width {
							let input_y = output_y * stride + kernel_y * dilation;
							let input_x = output_x * stride + kernel_x * dilation;
							let mut value = 0.0;
							if input_y >= padding
								&& input_x >= padding
								&& input_y - padding < input_height
								&& input_x - padding < input_width
							{
								value = input[((batch * channels + channel) * input_height + input_y - padding)
									* input_width
									+ input_x - padding];
							}
							columns[(batch * column_size + column) * output_length
								+ output_y * output_width
								+ output_x] = value;
						}
					}
				}
			}
		}
	}
	(columns, output_height, output_width)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps fold geometry explicit"
)]
fn host_fold(
	columns: &[f32],
	column_shape: [usize; 3],
	output_height: usize,
	output_width: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
) -> Vec<f32> {
	let [batch_size, column_size, output_length] = column_shape;
	let channels = column_size / (kernel_size * kernel_size);
	let effective = dilation * (kernel_size - 1) + 1;
	let positions_height = (output_height + 2 * padding - effective) / stride + 1;
	let positions_width = (output_width + 2 * padding - effective) / stride + 1;
	assert_eq!(positions_height * positions_width, output_length);
	let mut image = vec![0.0_f32; batch_size * channels * output_height * output_width];
	for batch in 0..batch_size {
		for channel in 0..channels {
			for kernel_y in 0..kernel_size {
				for kernel_x in 0..kernel_size {
					let column = channel * kernel_size * kernel_size + kernel_y * kernel_size + kernel_x;
					for output_y in 0..positions_height {
						for output_x in 0..positions_width {
							let input_y = output_y * stride + kernel_y * dilation;
							let input_x = output_x * stride + kernel_x * dilation;
							if input_y < padding
								|| input_x < padding
								|| input_y - padding >= output_height
								|| input_x - padding >= output_width
							{
								continue;
							}
							let value = columns[(batch * column_size + column) * output_length
								+ output_y * positions_width
								+ output_x];
							image[((batch * channels + channel) * output_height + input_y - padding)
								* output_width
								+ input_x
								- padding] += value;
						}
					}
				}
			}
		}
	}
	image
}

test_vk!(
	unfold_matches_independent_oracle_and_records_semantic_op,
	runtime,
	{
		let input_shape = [1, 2, 4, 5];
		let input_values = (0..40)
			.map(|index| (index as f32 - 17.0) * 0.11)
			.collect::<Vec<_>>();
		let (expected, positions_height, positions_width) =
			host_unfold(&input_values, input_shape, 3, 2, 1, 1);
		assert_eq!((positions_height, positions_width), (2, 3));
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let (plan, output) = runtime.capture(|| oa::ml::matrix::unfold(&input, 3, 2, 1, 1))?;
		assert_eq!(output.shape(), [1, 18, 6]);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::unfold"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 0.0);
		Ok(())
	}
);

test_vk!(
	fold_reconstructs_unfolded_windows_with_overlap_sums,
	runtime,
	{
		let input_shape = [1, 1, 5, 5];
		let input_values = (0..25)
			.map(|index| index as f32 * 0.25 - 3.0)
			.collect::<Vec<_>>();
		let (columns, _, _) = host_unfold(&input_values, input_shape, 2, 1, 0, 1);
		let expected = host_fold(&columns, [1, 4, 16], 5, 5, 2, 1, 0, 1);
		let columns_matrix = oa::Matrix::new_from_f32([1, 4, 16], &columns)?;
		let output = oa::ml::matrix::fold(&columns_matrix, 5, 5, 2, 1, 0, 1)?;
		assert_eq!(output.shape(), [1, 1, 5, 5]);
		assert_close(&output.read_f32()?, &expected, 1.0e-6);
		Ok(())
	}
);

test_vk!(unfold_fold_adjoints_match_host_oracles, runtime, {
	let input_shape = [1, 1, 4, 4];
	let input_values = [
		0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 1.2, -0.1, 0.5, -0.8, 0.6, 0.9, -0.3, 0.4, 1.5, -1.0,
	];
	let column_shape = [1, 4, 9];
	let target_columns = (0..36)
		.map(|index| ((index * 3 % 13) as f32 - 6.0) * 0.1)
		.collect::<Vec<_>>();
	let embedding =
		oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([4, 4], &input_values)?)?;
	let indices = oa::Matrix::new_from_slice([1, 1, 4], &[0_u32, 1, 2, 3])?;
	let module = oa::ml::nn::Unfold::new(2, 1, 0, 1)?;
	let target = oa::Matrix::new_from_f32(column_shape, &target_columns)?;
	let tape = oa::ml::GradientTape::new();
	let columns = module.forward(&embedding.forward(&indices)?)?;
	let loss = oa::ml::loss::mse(&columns, &target)?;
	tape.backward(&loss)?;
	// dL/dx = fold(2 * (columns - target) / n) accumulated per image tap.
	let (expected_columns, _, _) = host_unfold(&input_values, input_shape, 2, 1, 0, 1);
	let grad_columns: Vec<f32> = expected_columns
		.iter()
		.zip(&target_columns)
		.map(|(actual, target)| 2.0 * (actual - target) / 36.0)
		.collect();
	let expected_input = host_fold(&grad_columns, column_shape, 4, 4, 2, 1, 0, 1);
	assert_close(
		&embedding
			.weight()
			.gradient()
			.expect("Unfold input adjoint is missing")
			.read_f32()?,
		&expected_input,
		1.0e-5,
	);
	Ok(())
});

test_vk!(
	fold_adjoint_is_unfold_and_contracts_are_explicit,
	runtime,
	{
		let column_shape = [1, 4, 9];
		let column_values = (0..36)
			.map(|index| ((index * 5 % 17) as f32 - 8.0) * 0.08)
			.collect::<Vec<_>>();
		let target_image = (0..16)
			.map(|index| (index as f32 - 7.0) * 0.13)
			.collect::<Vec<_>>();
		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([4, 9], &column_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 4], &[0_u32, 1, 2, 3])?;
		let module = oa::ml::nn::Fold::new(4, 4, 2, 1, 0, 1)?;
		let target = oa::Matrix::new_from_f32([1, 1, 4, 4], &target_image)?;
		let tape = oa::ml::GradientTape::new();
		let image = module.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&image, &target)?;
		tape.backward(&loss)?;
		// dL/dcolumns = unfold(2 * (image - target) / n).
		let expected_image = host_fold(&column_values, column_shape, 4, 4, 2, 1, 0, 1);
		let grad_image: Vec<f32> = expected_image
			.iter()
			.zip(&target_image)
			.map(|(actual, target)| 2.0 * (actual - target) / 16.0)
			.collect();
		let (expected_columns, _, _) = host_unfold(&grad_image, [1, 1, 4, 4], 2, 1, 0, 1);
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("Fold input adjoint is missing")
				.read_f32()?,
			&expected_columns,
			1.0e-5,
		);
		assert!(oa::ml::nn::Unfold::new(0, 1, 0, 1).is_err());
		assert!(oa::ml::nn::Fold::new(4, 4, 2, 0, 0, 1).is_err());
		let bad = oa::Matrix::new_from_f32([1, 5, 9], &[0.0; 45])?;
		assert!(oa::ml::matrix::fold(&bad, 4, 4, 2, 1, 0, 1).is_err());
		let short = oa::Matrix::new_from_f32([1, 4, 8], &[0.0; 32])?;
		assert!(oa::ml::matrix::fold(&short, 4, 4, 2, 1, 0, 1).is_err());
		Ok(())
	}
);

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps transposed-convolution geometry explicit"
)]
fn host_conv_3d_transpose(
	input: &[f32],
	input_shape: [usize; 5],
	weight: &[f32],
	weight_shape: [usize; 5],
	bias: &[f32],
	stride: usize,
	padding: usize,
) -> Vec<f32> {
	let [
		batch_size,
		input_channels,
		input_depth,
		input_height,
		input_width,
	] = input_shape;
	let [
		weight_input_channels,
		output_channels,
		kernel_depth,
		kernel_height,
		kernel_width,
	] = weight_shape;
	assert_eq!(weight_input_channels, input_channels);
	assert_eq!(kernel_depth, kernel_height);
	assert_eq!(kernel_height, kernel_width);
	let output_depth = (input_depth - 1) * stride + kernel_depth - 2 * padding;
	let output_height = (input_height - 1) * stride + kernel_height - 2 * padding;
	let output_width = (input_width - 1) * stride + kernel_width - 2 * padding;
	let mut output =
		vec![0.0_f64; batch_size * output_channels * output_depth * output_height * output_width];
	for batch in 0..batch_size {
		for (output_channel, &bias_value) in bias.iter().enumerate().take(output_channels) {
			for output_z in 0..output_depth {
				for output_y in 0..output_height {
					for output_x in 0..output_width {
						let output_offset = (((batch * output_channels + output_channel) * output_depth
							+ output_z)
							* output_height
							+ output_y)
							* output_width
							+ output_x;
						output[output_offset] = f64::from(bias_value);
					}
				}
			}
		}
		for input_channel in 0..input_channels {
			for input_z in 0..input_depth {
				for input_y in 0..input_height {
					for input_x in 0..input_width {
						let input_offset = (((batch * input_channels + input_channel) * input_depth + input_z)
							* input_height
							+ input_y)
							* input_width
							+ input_x;
						for output_channel in 0..output_channels {
							for kernel_z in 0..kernel_depth {
								for kernel_y in 0..kernel_height {
									for kernel_x in 0..kernel_width {
										let padded_z = input_z * stride + kernel_z;
										let padded_y = input_y * stride + kernel_y;
										let padded_x = input_x * stride + kernel_x;
										if padded_z < padding || padded_y < padding || padded_x < padding {
											continue;
										}
										let output_z = padded_z - padding;
										let output_y = padded_y - padding;
										let output_x = padded_x - padding;
										if output_z >= output_depth
											|| output_y >= output_height
											|| output_x >= output_width
										{
											continue;
										}
										let weight_offset = (((input_channel * output_channels + output_channel)
											* kernel_depth
											+ kernel_z) * kernel_height
											+ kernel_y) * kernel_width
											+ kernel_x;
										let output_offset = (((batch * output_channels + output_channel)
											* output_depth
											+ output_z) * output_height
											+ output_y) * output_width
											+ output_x;
										output[output_offset] +=
											f64::from(input[input_offset]) * f64::from(weight[weight_offset]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	output.iter().map(|value| *value as f32).collect()
}

test_vk!(
	conv_3d_transpose_forward_matches_independent_oracle,
	runtime,
	{
		let input_shape = [1, 2, 2, 2, 3];
		let weight_shape = [2, 3, 2, 2, 2];
		let input_values = (0..24)
			.map(|index| ((index * 5 % 17) as f32 - 8.0) * 0.08)
			.collect::<Vec<_>>();
		let weight_values = (0..48)
			.map(|index| ((index * 7 % 19) as f32 - 9.0) * 0.04)
			.collect::<Vec<_>>();
		let bias_values = [0.1_f32, -0.2, 0.3];
		let expected = host_conv_3d_transpose(
			&input_values,
			input_shape,
			&weight_values,
			weight_shape,
			&bias_values,
			2,
			1,
		);
		let input = oa::Matrix::new_from_f32(input_shape, &input_values)?;
		let weight = oa::Matrix::new_from_f32(weight_shape, &weight_values)?;
		let bias = oa::Matrix::new_from_f32([3], &bias_values)?;
		let (plan, output) =
			runtime.capture(|| oa::ml::matrix::conv_3d_transpose(&input, &weight, &bias, 2, 1))?;
		assert_eq!(output.shape(), [1, 3, 2, 2, 4]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::conv_3d_transpose"
		);
		runtime.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 3.0e-5);
		Ok(())
	}
);

test_vk!(
	conv_3d_transpose_adjoint_matches_finite_differences,
	runtime,
	{
		let input_shape = [1, 2, 2, 2, 2];
		let weight_shape = [2, 2, 2, 2, 2];
		let input_values = (0..16)
			.map(|index| ((index * 5 % 13) as f32 - 6.0) * 0.07)
			.collect::<Vec<_>>();
		let weight_values = (0..32)
			.map(|index| ((index * 3 % 11) as f32 - 5.0) * 0.06)
			.collect::<Vec<_>>();
		let bias_values = [0.15_f32, -0.25];
		let target_values = (0..128)
			.map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.03)
			.collect::<Vec<_>>();
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			mse(
				&host_conv_3d_transpose(input, input_shape, weight, weight_shape, bias, 2, 0),
				&target_values,
			)
		};
		let expected_input = numerical_gradient(&input_values, |values| {
			host_loss(values, &weight_values, &bias_values)
		});
		let expected_weight = numerical_gradient(&weight_values, |values| {
			host_loss(&input_values, values, &bias_values)
		});
		let expected_bias = numerical_gradient(&bias_values, |values| {
			host_loss(&input_values, &weight_values, values)
		});

		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([8, 2], &input_values)?)?;
		let indices = oa::Matrix::new_from_slice([1, 2, 2, 2], &(0_u32..8).collect::<Vec<_>>())?;
		let module = oa::ml::nn::ConvTranspose3d::new_from_matrices(
			oa::Matrix::new_from_f32(weight_shape, &weight_values)?,
			oa::Matrix::new_from_f32([2], &bias_values)?,
			2,
			0,
		)?;
		let target = oa::Matrix::new_from_f32([1, 2, 4, 4, 4], &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("ConvTranspose3d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			8.0e-4,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("ConvTranspose3d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			8.0e-4,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("ConvTranspose3d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			8.0e-4,
		);
		Ok(())
	}
);

test_vk!(
	conv_3d_transpose_module_contract_and_validation_are_explicit,
	runtime,
	{
		let module = oa::ml::nn::ConvTranspose3d::new_with_seed(3, 5, 3, 2, 1, 0x4354_4e33)?;
		assert_eq!(module.input_channels(), 3);
		assert_eq!(module.output_channels(), 5);
		assert_eq!(module.kernel_size(), 3);
		assert_eq!(module.stride(), 2);
		assert_eq!(module.padding(), 1);
		assert_eq!(module.weight().data().shape(), [3, 5, 3, 3, 3]);
		assert_eq!(module.bias().data().shape(), [5]);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		assert!(oa::ml::nn::ConvTranspose3d::new_with_seed(2, 2, 3, 0, 0, 1).is_err());
		let input = oa::Matrix::new_from_f32([1, 2, 1, 1, 1], &[0.0; 2])?;
		let weight = oa::Matrix::new_from_f32([2, 2, 1, 1, 1], &[0.0; 4])?;
		let bias = oa::Matrix::new_from_f32([2], &[0.0; 2])?;
		assert!(oa::ml::matrix::conv_3d_transpose(&input, &weight, &bias, 1, 1).is_err());
		Ok(())
	}
);
