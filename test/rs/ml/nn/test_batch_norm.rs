use oa::ml::Module as _;

const EPSILON: f32 = 1.0e-5;

fn host_statistics(input: &[f32], shape: [usize; 4]) -> (Vec<f32>, Vec<f32>) {
	let [batch, channels, height, width] = shape;
	let spatial = height * width;
	let sample_count = batch * spatial;
	let mut mean = vec![0.0; channels];
	let mut variance = vec![0.0; channels];
	for channel in 0..channels {
		let sum = (0..batch)
			.flat_map(|batch_index| {
				(0..spatial).map(move |spatial_index| {
					f64::from(input[(batch_index * channels + channel) * spatial + spatial_index])
				})
			})
			.sum::<f64>();
		let channel_mean = sum / sample_count as f64;
		let square_sum = (0..batch)
			.flat_map(|batch_index| {
				(0..spatial).map(move |spatial_index| {
					let value =
						f64::from(input[(batch_index * channels + channel) * spatial + spatial_index]);
					(value - channel_mean).powi(2)
				})
			})
			.sum::<f64>();
		mean[channel] = channel_mean as f32;
		variance[channel] = (square_sum / sample_count as f64) as f32;
	}
	(mean, variance)
}

fn host_batch_norm(
	input: &[f32],
	shape: [usize; 4],
	weight: &[f32],
	bias: &[f32],
	mean: &[f32],
	variance: &[f32],
	epsilon: f32,
) -> Vec<f32> {
	let [batch, channels, height, width] = shape;
	let spatial = height * width;
	let mut output = vec![0.0; input.len()];
	for batch_index in 0..batch {
		for channel in 0..channels {
			let inverse_stddev = 1.0_f64 / (f64::from(variance[channel]) + f64::from(epsilon)).sqrt();
			for spatial_index in 0..spatial {
				let index = (batch_index * channels + channel) * spatial + spatial_index;
				output[index] = ((f64::from(input[index]) - f64::from(mean[channel]))
					* inverse_stddev
					* f64::from(weight[channel])
					+ f64::from(bias[channel])) as f32;
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

test_vk!(batch_norm_2d_training_matches_independent_oracle, engine, {
	let shape = [2, 2, 2, 3];
	let input_values = [
		0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 1.2, -0.1, 0.5, -0.8, 0.6, 0.9, -0.3, 0.4, 1.5, -1.0, 0.7,
		0.1, 0.25, -0.55, 0.35, 1.25, -0.2, 0.75,
	];
	let weight_values = [0.75_f32, -1.25];
	let bias_values = [0.1_f32, -0.2];
	let (expected_mean, expected_variance) = host_statistics(&input_values, shape);
	let expected = host_batch_norm(
		&input_values,
		shape,
		&weight_values,
		&bias_values,
		&expected_mean,
		&expected_variance,
		EPSILON,
	);
	let input = oa::Matrix::from_f32(&engine, shape, &input_values)?;
	let weight = oa::Matrix::from_f32(&engine, [2], &weight_values)?;
	let bias = oa::Matrix::from_f32(&engine, [2], &bias_values)?;
	let (plan, result) =
		engine.capture(|| oa::ml::matrix::batch_norm_2d(&input, &weight, &bias, EPSILON))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::matrix::batch_norm_2d"
	);
	assert_eq!(plan.semantic_lowering().maximum_nodes_per_op(), 2);
	engine.submit(&plan)?.wait()?;
	assert_close(&result.mean.read_f32()?, &expected_mean, 2.0e-6);
	assert_close(&result.variance.read_f32()?, &expected_variance, 2.0e-6);
	assert_close(&result.output.read_f32()?, &expected, 2.0e-5);

	let large = oa::Matrix::from_f32(&engine, [1, 1, 1, 513], &[10_000.0; 513])?;
	let unit = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
	let zero = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
	let stable = oa::ml::matrix::batch_norm_2d(&large, &unit, &zero, EPSILON)?;
	assert!(stable.variance.read_f32()?[0].is_finite());
	assert!(
		stable
			.output
			.read_f32()?
			.iter()
			.all(|value| value.is_finite())
	);
	Ok(())
});

test_vk!(
	batch_norm_2d_module_updates_and_uses_persistent_state,
	engine,
	{
		let module = oa::ml::nn::BatchNorm2d::with_options(&engine, 1, EPSILON, 0.1)?;
		let input = oa::Matrix::from_f32(&engine, [1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
		let output = module.forward(&input)?;
		assert_close(
			&output.read_f32()?,
			&[-1.341_635_5, -0.447_211_83, 0.447_211_83, 1.341_635_5],
			2.0e-5,
		);
		assert_close(&module.running_mean().read_f32()?, &[0.25], 1.0e-6);
		assert_close(&module.running_variance().read_f32()?, &[1.025], 1.0e-6);
		assert_eq!(module.num_features(), 1);
		assert_eq!(module.epsilon(), EPSILON);
		assert_eq!(module.momentum(), 0.1);
		assert_eq!(
			module
				.named_parameters()
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["weight", "bias"]
		);
		let named_buffers = module.named_buffers();
		assert_eq!(
			named_buffers
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["running_mean", "running_variance"]
		);
		assert!(named_buffers.iter().all(|entry| entry.persistent()));

		module.eval();
		let before_mean = module.running_mean();
		let before_variance = module.running_variance();
		let eval_output = module.forward(&input)?;
		let expected = host_batch_norm(
			&[1.0, 2.0, 3.0, 4.0],
			[1, 1, 2, 2],
			&[1.0],
			&[0.0],
			&[0.25],
			&[1.025],
			EPSILON,
		);
		assert_close(&eval_output.read_f32()?, &expected, 2.0e-5);
		assert_close(
			&module.running_mean().read_f32()?,
			&before_mean.read_f32()?,
			0.0,
		);
		assert_close(
			&module.running_variance().read_f32()?,
			&before_variance.read_f32()?,
			0.0,
		);
		Ok(())
	}
);

test_vk!(
	batch_norm_2d_training_adjoint_matches_finite_differences,
	engine,
	{
		let shape = [1, 2, 2, 3];
		let input_values = [
			0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4, 1.2, -0.1, 0.5, -0.8, 0.6, 0.9,
		];
		let weight_values = [0.75_f32, -1.25];
		let bias_values = [0.1_f32, -0.2];
		let target_values = [
			0.3_f32, -0.2, 0.1, -0.4, 0.7, 0.5, -0.6, 0.8, 0.2, 0.4, -0.3, 0.9,
		];
		let host_loss = |input: &[f32], weight: &[f32], bias: &[f32]| {
			let (mean, variance) = host_statistics(input, shape);
			mse(
				&host_batch_norm(input, shape, weight, bias, &mean, &variance, EPSILON),
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
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [4, 3], &input_values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 2, 2], &[0_u32, 1, 2, 3])?;
		let module = oa::ml::nn::BatchNorm2d::from_matrices(
			oa::Matrix::from_f32(&engine, [2], &weight_values)?,
			oa::Matrix::from_f32(&engine, [2], &bias_values)?,
			oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?,
			oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?,
			EPSILON,
			0.1,
		)?;
		let target = oa::Matrix::from_f32(&engine, shape, &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let output = module.forward(&embedding.forward(&indices)?)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("BatchNorm2d input adjoint is missing")
				.read_f32()?,
			&expected_input,
			1.5e-3,
		);
		assert_close(
			&module
				.weight()
				.gradient()
				.expect("BatchNorm2d weight adjoint is missing")
				.read_f32()?,
			&expected_weight,
			1.5e-3,
		);
		assert_close(
			&module
				.bias()
				.gradient()
				.expect("BatchNorm2d bias adjoint is missing")
				.read_f32()?,
			&expected_bias,
			1.5e-3,
		);
		Ok(())
	}
);

test_vk!(
	batch_norm_2d_eval_input_adjoint_uses_fixed_statistics,
	engine,
	{
		let shape = [1, 1, 1, 4];
		let input_values = [0.2_f32, -0.7, 1.1, 0.8];
		let target_values = [0.3_f32, -0.2, 0.1, -0.4];
		let mean = [0.25_f32];
		let variance = [1.5_f32];
		let weight = [0.75_f32];
		let bias = [0.1_f32];
		let expected = numerical_gradient(&input_values, |values| {
			mse(
				&host_batch_norm(values, shape, &weight, &bias, &mean, &variance, EPSILON),
				&target_values,
			)
		});
		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 4], &input_values)?)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 1, 1], &[0_u32])?;
		let module = oa::ml::nn::BatchNorm2d::from_matrices(
			oa::Matrix::from_f32(&engine, [1], &weight)?,
			oa::Matrix::from_f32(&engine, [1], &bias)?,
			oa::Matrix::from_f32(&engine, [1], &mean)?,
			oa::Matrix::from_f32(&engine, [1], &variance)?,
			EPSILON,
			0.1,
		)?;
		module.eval();
		let target = oa::Matrix::from_f32(&engine, shape, &target_values)?;
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("BatchNorm2d eval input adjoint is missing")
				.read_f32()?,
			&expected,
			3.0e-4,
		);
		Ok(())
	}
);
