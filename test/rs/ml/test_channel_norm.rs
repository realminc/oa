const EPSILON: f32 = 1.0e-5;

fn host_forward(
	input: &[f32],
	shape: [usize; 3],
	weight: &[f32],
	bias: &[f32],
	epsilon: f32,
	relu: bool,
) -> Vec<f32> {
	let [batch, channels, sequence] = shape;
	let mut output = vec![0.0; input.len()];
	for batch_index in 0..batch {
		for time in 0..sequence {
			let mean = (0..channels)
				.map(|channel| {
					f64::from(input[(batch_index * channels + channel) * sequence + time])
				})
				.sum::<f64>()
				/ channels as f64;
			let variance = (0..channels)
				.map(|channel| {
					let value =
						f64::from(input[(batch_index * channels + channel) * sequence + time])
							- mean;
					value * value
				})
				.sum::<f64>()
				/ channels as f64;
			let inverse_stddev = (variance + f64::from(epsilon)).sqrt().recip();
			for channel in 0..channels {
				let index = (batch_index * channels + channel) * sequence + time;
				let normalized = (f64::from(input[index]) - mean) * inverse_stddev;
				let value = normalized * f64::from(weight[channel]) + f64::from(bias[channel]);
				output[index] = if relu {
					value.max(0.0) as f32
				} else {
					value as f32
				};
			}
		}
	}
	output
}

fn host_backward(
	input: &[f32],
	shape: [usize; 3],
	weight: &[f32],
	bias: &[f32],
	output_gradient: &[f32],
	epsilon: f32,
	relu: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
	let [batch, channels, sequence] = shape;
	let forward = host_forward(input, shape, weight, bias, epsilon, relu);
	let mut input_gradient = vec![0.0; input.len()];
	let mut weight_gradient = vec![0.0; channels];
	let mut bias_gradient = vec![0.0; channels];
	for batch_index in 0..batch {
		for time in 0..sequence {
			let indices = (0..channels)
				.map(|channel| (batch_index * channels + channel) * sequence + time)
				.collect::<Vec<_>>();
			let mean = indices
				.iter()
				.map(|index| f64::from(input[*index]))
				.sum::<f64>()
				/ channels as f64;
			let variance = indices
				.iter()
				.map(|index| (f64::from(input[*index]) - mean).powi(2))
				.sum::<f64>()
				/ channels as f64;
			let inverse_stddev = (variance + f64::from(epsilon)).sqrt().recip();
			let normalized = indices
				.iter()
				.map(|index| (f64::from(input[*index]) - mean) * inverse_stddev)
				.collect::<Vec<_>>();
			let gradients = indices
				.iter()
				.map(|index| {
					if relu && forward[*index] <= 0.0 {
						0.0
					} else {
						f64::from(output_gradient[*index])
					}
				})
				.collect::<Vec<_>>();
			let dxhat = (0..channels)
				.map(|channel| gradients[channel] * f64::from(weight[channel]))
				.collect::<Vec<_>>();
			let mean_dxhat = dxhat.iter().sum::<f64>() / channels as f64;
			let mean_normalized_dxhat = normalized
				.iter()
				.zip(&dxhat)
				.map(|(normalized, dxhat)| normalized * dxhat)
				.sum::<f64>()
				/ channels as f64;
			for channel in 0..channels {
				let index = indices[channel];
				input_gradient[index] = (inverse_stddev
					* (dxhat[channel] - mean_dxhat - normalized[channel] * mean_normalized_dxhat))
					as f32;
				weight_gradient[channel] += (gradients[channel] * normalized[channel]) as f32;
				bias_gradient[channel] += gradients[channel] as f32;
			}
		}
	}
	(input_gradient, weight_gradient, bias_gradient)
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

test_vk!(channel_norm_forward_and_relu_match_host_oracle, engine, {
	let shape = [2, 5, 3];
	let input_values = (0..30)
		.map(|index| index as f32 * 0.13 - 1.7)
		.collect::<Vec<_>>();
	let weight_values = [0.7_f32, -1.2, 0.4, 1.5, -0.8];
	let bias_values = [0.1_f32, 0.3, -0.2, 0.0, 0.25];
	let input = oa::Matrix::from_f32(&engine, shape, &input_values)?;
	let weight = oa::Matrix::from_f32(&engine, [5], &weight_values)?;
	let bias = oa::Matrix::from_f32(&engine, [5], &bias_values)?;
	let (plan, output) =
		engine.capture(|| oa::ml::matrix::channel_norm(&input, &weight, &bias, EPSILON))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::matrix::channel_norm"
	);
	engine.submit(&plan)?.wait()?;
	assert_close(
		&output.read_f32()?,
		&host_forward(
			&input_values,
			shape,
			&weight_values,
			&bias_values,
			EPSILON,
			false,
		),
		2.0e-5,
	);
	let relu = oa::ml::matrix::channel_norm_relu(&input, &weight, &bias, EPSILON)?;
	assert_close(
		&relu.read_f32()?,
		&host_forward(
			&input_values,
			shape,
			&weight_values,
			&bias_values,
			EPSILON,
			true,
		),
		2.0e-5,
	);
	Ok(())
});

test_vk!(channel_norm_explicit_adjoints_match_host_oracle, engine, {
	let shape = [1, 5, 3];
	let input_values = [
		-1.1_f32, 0.2, 0.7, 1.4, -0.8, 0.3, 0.9, -0.4, 1.7, -0.3, 0.6, -1.5, 0.5, 1.1, -0.2,
	];
	let weight_values = [0.7_f32, -1.2, 0.4, 1.5, -0.8];
	let bias_values = [0.1_f32, 0.3, -0.2, 0.0, 0.25];
	let gradient_values = [
		0.3_f32, -0.7, 0.1, 0.8, -0.4, 0.2, -0.9, 0.5, 0.6, -0.2, 0.4, -0.1, 0.7, -0.6, 0.9,
	];
	let input = oa::Matrix::from_f32(&engine, shape, &input_values)?;
	let weight = oa::Matrix::from_f32(&engine, [5], &weight_values)?;
	let bias = oa::Matrix::from_f32(&engine, [5], &bias_values)?;
	let output_gradient = oa::Matrix::from_f32(&engine, shape, &gradient_values)?;
	let expected = host_backward(
		&input_values,
		shape,
		&weight_values,
		&bias_values,
		&gradient_values,
		EPSILON,
		false,
	);
	let result = oa::ml::matrix::channel_norm_backward(&input, &weight, &output_gradient, EPSILON)?;
	assert_close(&result.input.read_f32()?, &expected.0, 3.0e-5);
	assert_close(&result.weight.read_f32()?, &expected.1, 3.0e-5);
	assert_close(&result.bias.read_f32()?, &expected.2, 3.0e-5);

	let forward = oa::ml::matrix::channel_norm_relu(&input, &weight, &bias, EPSILON)?;
	let expected_relu = host_backward(
		&input_values,
		shape,
		&weight_values,
		&bias_values,
		&gradient_values,
		EPSILON,
		true,
	);
	let result = oa::ml::matrix::channel_norm_relu_backward(
		&input,
		&weight,
		&forward,
		&output_gradient,
		EPSILON,
	)?;
	assert_close(&result.input.read_f32()?, &expected_relu.0, 3.0e-5);
	assert_close(&result.weight.read_f32()?, &expected_relu.1, 3.0e-5);
	assert_close(&result.bias.read_f32()?, &expected_relu.2, 3.0e-5);
	Ok(())
});

test_vk!(channel_norm_module_accumulates_complete_adjoint, engine, {
	let shape = [1, 3, 2];
	let input_values = [-1.1_f32, 0.2, 1.4, -0.8, 0.9, -0.4];
	let weight_values = [0.7_f32, -1.2, 0.4];
	let bias_values = [0.1_f32, 0.3, -0.2];
	let target_values = [0.3_f32, 0.7, 0.1, 0.8, 0.4, 0.2];
	let upstream = host_forward(
		&input_values,
		shape,
		&weight_values,
		&bias_values,
		EPSILON,
		true,
	)
	.into_iter()
	.zip(&target_values)
	.map(|(output, target)| 2.0 * (output - target) / input_values.len() as f32)
	.collect::<Vec<_>>();
	let expected = host_backward(
		&input_values,
		shape,
		&weight_values,
		&bias_values,
		&upstream,
		EPSILON,
		true,
	);
	let embedding =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [6, 1], &input_values)?)?;
	let indices = oa::Matrix::from_slice(&engine, [6], &[0_u32, 1, 2, 3, 4, 5])?;
	let norm = oa::ml::nn::LayerNorm::from_matrices(
		oa::Matrix::from_f32(&engine, [3], &weight_values)?,
		oa::Matrix::from_f32(&engine, [3], &bias_values)?,
		EPSILON,
	)?;
	let target = oa::Matrix::from_f32(&engine, shape, &target_values)?;
	let tape = oa::ml::GradientTape::new();
	let input = embedding.forward(&indices)?.reshape(shape)?;
	let loss = oa::ml::loss::mse(&norm.forward_channel_relu(&input)?, &target)?;
	tape.backward(&loss)?;
	assert_close(
		&embedding
			.weight()
			.gradient()
			.expect("input adjoint")
			.read_f32()?,
		&expected.0,
		5.0e-5,
	);
	assert_close(
		&norm
			.weight()
			.gradient()
			.expect("weight adjoint")
			.read_f32()?,
		&expected.1,
		5.0e-5,
	);
	assert_close(
		&norm.bias().gradient().expect("bias adjoint").read_f32()?,
		&expected.2,
		5.0e-5,
	);
	Ok(())
});

test_vk!(channel_norm_rejects_unadmitted_geometry, engine, {
	let input = oa::Matrix::from_f32(&engine, [1, 3, 2], &[0.0; 6])?;
	let weight = oa::Matrix::from_f32(&engine, [3], &[1.0; 3])?;
	let bias = oa::Matrix::from_f32(&engine, [3], &[0.0; 3])?;
	for epsilon in [0.0_f32, -1.0, f32::NAN] {
		assert_eq!(
			oa::ml::matrix::channel_norm(&input, &weight, &bias, epsilon)
				.err()
				.expect("invalid epsilon accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let wide = oa::Matrix::from_f32(&engine, [1, 1025, 1], &[0.0; 1025])?;
	let wide_weight = oa::Matrix::from_f32(&engine, [1025], &[1.0; 1025])?;
	let wide_bias = oa::Matrix::from_f32(&engine, [1025], &[0.0; 1025])?;
	assert_eq!(
		oa::ml::matrix::channel_norm(&wide, &wide_weight, &wide_bias, EPSILON)
			.err()
			.expect("shader register bound accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
