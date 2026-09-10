use oa::ml::{Module as _, UpsampleMode};

fn host_upsample_2d(
	input: &[f32],
	shape: [usize; 4],
	scale_factor: usize,
	mode: UpsampleMode,
) -> Vec<f32> {
	let [batch, channels, input_height, input_width] = shape;
	let output_height = input_height * scale_factor;
	let output_width = input_width * scale_factor;
	let mut output = vec![0.0; batch * channels * output_height * output_width];
	for batch_index in 0..batch {
		for channel in 0..channels {
			for output_y in 0..output_height {
				for output_x in 0..output_width {
					let output_index = ((batch_index * channels + channel) * output_height
						+ output_y) * output_width
						+ output_x;
					output[output_index] = match mode {
						UpsampleMode::Nearest => {
							let input_y = output_y / scale_factor;
							let input_x = output_x / scale_factor;
							input[((batch_index * channels + channel) * input_height + input_y)
								* input_width + input_x]
						}
						UpsampleMode::Bilinear => {
							let source_y = ((output_y as f64 + 0.5) / scale_factor as f64 - 0.5)
								.clamp(0.0, (input_height - 1) as f64);
							let source_x = ((output_x as f64 + 0.5) / scale_factor as f64 - 0.5)
								.clamp(0.0, (input_width - 1) as f64);
							let y0 = source_y.floor() as usize;
							let x0 = source_x.floor() as usize;
							let y1 = (y0 + 1).min(input_height - 1);
							let x1 = (x0 + 1).min(input_width - 1);
							let fy = source_y - y0 as f64;
							let fx = source_x - x0 as f64;
							let value = |y: usize, x: usize| {
								f64::from(
									input[((batch_index * channels + channel) * input_height + y)
										* input_width + x],
								)
							};
							let top = value(y0, x0) * (1.0 - fx) + value(y0, x1) * fx;
							let bottom = value(y1, x0) * (1.0 - fx) + value(y1, x1) * fx;
							(top * (1.0 - fy) + bottom * fy) as f32
						}
						_ => unreachable!("test covers every admitted Upsample mode"),
					};
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

test_vk!(
	upsample_2d_forward_modes_match_independent_oracles,
	engine,
	{
		let shape = [1, 1, 2, 3];
		let values = [1.0_f32, 2.0, 4.0, -1.0, 0.5, 3.0];
		let input = oa::Matrix::from_f32(&engine, shape, &values)?;
		for mode in [UpsampleMode::Nearest, UpsampleMode::Bilinear] {
			let (plan, output) = engine.capture(|| oa::ml::matrix::upsample_2d(&input, 2, mode))?;
			assert_eq!(output.shape(), [1, 1, 4, 6]);
			assert_eq!(plan.semantic_graph().operations().len(), 1);
			assert_eq!(
				plan.semantic_graph().operations()[0].name(),
				"oa::ml::matrix::upsample_2d"
			);
			assert_eq!(plan.semantic_graph().operations()[0].attributes().len(), 2);
			engine.submit(&plan)?.wait()?;
			assert_close(
				&output.read_f32()?,
				&host_upsample_2d(&values, shape, 2, mode),
				1.0e-6,
			);
		}
		Ok(())
	}
);

test_vk!(
	upsample_2d_adjoint_modes_match_finite_differences,
	engine,
	{
		let shape = [1, 1, 2, 3];
		let values = [0.2_f32, -0.7, 1.1, 0.8, 0.3, -0.4];
		let target_values = (0..24)
			.map(|index| (index as f32 * 0.17).sin())
			.collect::<Vec<_>>();
		for mode in [UpsampleMode::Nearest, UpsampleMode::Bilinear] {
			let expected = numerical_gradient(&values, |candidate| {
				mse(&host_upsample_2d(candidate, shape, 2, mode), &target_values)
			});
			let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
				&engine,
				[2, 3],
				&values,
			)?)?;
			let indices = oa::Matrix::from_slice(&engine, [1, 1, 2], &[0_u32, 1])?;
			let target = oa::Matrix::from_f32(&engine, [1, 1, 4, 6], &target_values)?;
			let module = oa::ml::nn::Upsample::with_mode(2, mode)?;
			assert_eq!(module.scale_factor(), 2);
			assert_eq!(module.mode(), mode);
			assert!(module.parameters().is_empty());
			let tape = oa::ml::GradientTape::new();
			let loss = oa::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			let actual = embedding
				.weight()
				.gradient()
				.expect("Upsample input adjoint is missing")
				.read_f32()?;
			assert_close(&actual, &expected, 3.0e-4);
		}
		Ok(())
	}
);

test_vk!(
	upsample_2d_preserves_donor_nearest_backward_vector,
	engine,
	{
		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[2, 2],
			&[1.0, 2.0, 3.0, 4.0],
		)?)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 1, 2], &[0_u32, 1])?;
		let target = oa::Matrix::from_f32(&engine, [1, 1, 4, 4], &[0.0; 16])?;
		let tape = oa::ml::GradientTape::new();
		let output =
			oa::ml::matrix::upsample_2d(&embedding.forward(&indices)?, 2, UpsampleMode::Nearest)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("nearest donor adjoint is missing")
				.read_f32()?,
			&[0.5, 1.0, 1.5, 2.0],
			1.0e-6,
		);
		assert_eq!(oa::ml::nn::Upsample::new(2)?.mode(), UpsampleMode::Bilinear);
		assert_eq!(
			oa::ml::nn::Upsample::new(0)
				.err()
				.expect("zero scale was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);
