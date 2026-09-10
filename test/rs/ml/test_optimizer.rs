fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
		);
	}
}

test_vk!(
	clip_grad_norm_matches_the_donor_two_pass_operation,
	engine,
	{
		let first = oa::Matrix::from_f32(&engine, [3], &[3.0, 4.0, 0.0])?;
		let second = oa::Matrix::from_f32(&engine, [2, 2], &[0.0, 12.0, 0.0, 0.0])?;
		let empty = oa::Matrix::from_f32(&engine, [0], &[])?;

		oa::ml::optim::clip_grad_norm(&[first.clone(), empty, second.clone()], 6.5)?;
		assert_close(&first.read_f32()?, &[1.5, 2.0, 0.0], 1.0e-6);
		assert_close(&second.read_f32()?, &[0.0, 6.0, 0.0, 0.0], 1.0e-6);

		let below_limit = oa::Matrix::from_f32(&engine, [2], &[3.0, 4.0])?;
		oa::ml::optim::clip_grad_norm(std::slice::from_ref(&below_limit), 5.0)?;
		assert_close(&below_limit.read_f32()?, &[3.0, 4.0], 0.0);
		Ok(())
	}
);

test_vk!(
	clip_grad_norm_validates_collection_and_scalar_contracts,
	engine,
	{
		let gradient = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
		let error = oa::ml::optim::clip_grad_norm(std::slice::from_ref(&gradient), f32::NAN)
			.expect_err("non-finite max norm was accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);

		let error = oa::ml::optim::clip_grad_norm(&[gradient.clone(), gradient.clone()], 1.0)
			.expect_err("duplicate gradient storage was accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);

		let integer = oa::Matrix::from_slice(&engine, [1], &[1_i32])?;
		let error = oa::ml::optim::clip_grad_norm(&[integer], 1.0)
			.expect_err("integer gradient was accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);

		let gradients = (0..17)
			.map(|_| oa::Matrix::from_f32(&engine, [1], &[1.0]))
			.collect::<oa::Result<Vec<_>>>()?;
		let error = oa::ml::optim::clip_grad_norm(&gradients, 1.0)
			.expect_err("more than sixteen gradients were accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		Ok(())
	}
);

test_vk!(
	clip_grad_norm_capture_retains_one_variadic_semantic_operation,
	engine,
	{
		let gradients = (1..=10)
			.map(|value| oa::Matrix::from_f32(&engine, [1], &[value as f32]))
			.collect::<oa::Result<Vec<_>>>()?;
		let (plan, ()) = engine.capture(|| oa::ml::optim::clip_grad_norm(&gradients, 100.0))?;

		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(plan.diagnostics().node_count(), 2);
		assert_eq!(plan.semantic_lowering().decomposed_op_count(), 1);
		assert_eq!(plan.semantic_lowering().maximum_nodes_per_op(), 2);
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::optim::clip_grad_norm");
		assert_eq!(operation.inputs().len(), 10);
		assert_eq!(operation.outputs().len(), 10);
		assert_eq!(operation.mutated_inputs().len(), 10);
		assert_eq!(operation.aliases().len(), 10);

		engine.submit(&plan)?.wait()?;
		for (index, gradient) in gradients.iter().enumerate() {
			assert_close(&gradient.read_f32()?, &[(index + 1) as f32], 0.0);
		}
		Ok(())
	}
);

test_vk!(sgd_matches_the_donor_fp32_update_order, engine, {
	const LEARNING_RATE: f32 = 0.05;
	const WEIGHT_DECAY: f32 = 0.01;
	let initial = [0.2_f32, -0.1, -0.3, 0.4];
	let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, -1.0, 0.5])?;
	let targets = oa::Matrix::from_slice(&engine, [2], &[1_u32, 0])?;
	let weight = oa::Matrix::from_f32(&engine, [2, 2], &initial)?;
	let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
	let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
	let weight_parameter = layer.weight();
	let mut optimizer = oa::ml::Sgd::new(layer.parameters(), LEARNING_RATE, 0.0, WEIGHT_DECAY)?;

	let tape = oa::ml::GradientTape::new();
	let loss = oa::ml::loss::cross_entropy(&layer.forward(&input)?, &targets)?;
	tape.backward(&loss)?;
	let gradient = weight_parameter
		.gradient()
		.expect("backward produced a weight gradient")
		.read_f32()?;
	let expected = initial
		.iter()
		.zip(&gradient)
		.map(|(parameter, gradient)| {
			parameter - LEARNING_RATE * (gradient + WEIGHT_DECAY * parameter)
		})
		.collect::<Vec<_>>();
	optimizer.step()?;
	assert_eq!(optimizer.step_count(), 1);
	assert_close(&weight_parameter.data().read_f32()?, &expected, 1.0e-6);
	Ok(())
});

test_vk!(adam_matches_the_donor_first_step_bias_correction, engine, {
	const LEARNING_RATE: f32 = 0.01;
	const EPSILON: f32 = 1.0e-8;
	let initial = [0.2_f32, -0.1, -0.3, 0.4];
	let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, -1.0, 0.5])?;
	let targets = oa::Matrix::from_slice(&engine, [2], &[1_u32, 0])?;
	let weight = oa::Matrix::from_f32(&engine, [2, 2], &initial)?;
	let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
	let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
	let weight_parameter = layer.weight();
	let mut optimizer = oa::ml::Adam::new(layer.parameters(), LEARNING_RATE)?;

	let tape = oa::ml::GradientTape::new();
	let loss = oa::ml::loss::cross_entropy(&layer.forward(&input)?, &targets)?;
	tape.backward(&loss)?;
	let gradient = weight_parameter
		.gradient()
		.expect("backward produced a weight gradient")
		.read_f32()?;
	let expected = initial
		.iter()
		.zip(&gradient)
		.map(|(parameter, gradient)| {
			parameter - LEARNING_RATE * gradient / (gradient.abs() + EPSILON)
		})
		.collect::<Vec<_>>();
	optimizer.step()?;
	assert_eq!(optimizer.step_count(), 1);
	assert_close(&weight_parameter.data().read_f32()?, &expected, 2.0e-5);
	Ok(())
});

test_vk!(muon_matches_the_donor_matrix_and_vector_routes, engine, {
	const LEARNING_RATE: f32 = 0.01;
	const BETA: f32 = 0.95;
	const WEIGHT_DECAY: f32 = 0.1;
	const EPSILON: f32 = 1.0e-7;
	const ITERATIONS: u32 = 5;
	let initial_weight = [-0.15_f32, -0.14, -0.13, -0.12, -0.11, -0.10, -0.09, -0.08];
	let initial_bias = [0.03_f32, -0.02];
	let input = oa::Matrix::from_f32(
		&engine,
		[4, 4],
		&[
			1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.25, 0.75, 0.5, -0.5, 1.0, 0.0, -0.5, 0.25, 0.0, 1.0,
		],
	)?;
	let targets = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 0, 1])?;
	let layer = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(&engine, [2, 4], &initial_weight)?,
		oa::Matrix::from_f32(&engine, [2], &initial_bias)?,
	)?;
	let weight = layer.weight();
	let bias = layer.bias().expect("biased Linear is missing its bias");
	let mut optimizer = oa::ml::Muon::with_hyperparameters(
		layer.parameters(),
		LEARNING_RATE,
		BETA,
		WEIGHT_DECAY,
		EPSILON,
		ITERATIONS,
	)?;

	let tape = oa::ml::GradientTape::new();
	let loss = oa::ml::loss::cross_entropy(&layer.forward(&input)?, &targets)?;
	tape.backward(&loss)?;
	let weight_gradient = weight
		.gradient()
		.expect("backward produced a weight gradient")
		.read_f32()?;
	let bias_gradient = bias
		.gradient()
		.expect("backward produced a bias gradient")
		.read_f32()?;
	let mut expected_weight = initial_weight.to_vec();
	let mut expected_weight_momentum = vec![0.0; initial_weight.len()];
	muon_matrix_reference(
		&mut expected_weight,
		&mut expected_weight_momentum,
		&weight_gradient,
		2,
		4,
		LEARNING_RATE,
		BETA,
		WEIGHT_DECAY,
		EPSILON,
		ITERATIONS,
	);
	let mut expected_bias = initial_bias.to_vec();
	let mut expected_bias_momentum = vec![0.0; initial_bias.len()];
	muon_vector_reference(
		&mut expected_bias,
		&mut expected_bias_momentum,
		&bias_gradient,
		LEARNING_RATE,
		BETA,
		WEIGHT_DECAY,
	);

	optimizer.step()?;
	assert_eq!(optimizer.step_count(), 1);
	assert_close(&weight.data().read_f32()?, &expected_weight, 2.0e-3);
	assert_close(&bias.data().read_f32()?, &expected_bias, 1.0e-6);
	Ok(())
});

test_vk!(muon_handles_tall_and_large_matrix_orientations, engine, {
	run_muon_matrix_case(&engine, 5, 3)?;
	run_muon_matrix_case(&engine, 65, 67)?;
	Ok(())
});

test_vk!(
	eager_training_accepts_sgd_through_optimizer_policy,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let layer = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x5347_4401)?;
		let initial = layer.weight().data().read_f32()?;
		let mut optimizer = oa::ml::Sgd::new(layer.parameters(), 0.1, 0.9, 0.0)?;
		let mut training = oa::ml::ItTraining::new_eager(
			&engine,
			&mut optimizer,
			oa::ml::ItTrainingConfig {
				total_steps: 1,
				..Default::default()
			},
		)?;
		assert!(training.begin_step()?);
		training.zero_grad();
		let tape = oa::ml::GradientTape::new();
		let loss = oa::ml::loss::cross_entropy(&layer.forward(&input)?, &targets)?;
		tape.backward(&loss)?;
		training.complete_step(&loss)?;
		let snapshot = training.finish()?;
		assert_eq!(snapshot.step_count(), 1);
		assert_eq!(optimizer.step_count(), 1);
		assert_ne!(layer.weight().data().read_f32()?, initial);
		Ok(())
	}
);

#[allow(clippy::too_many_arguments)]
fn muon_matrix_reference(
	weights: &mut [f32],
	momentum: &mut [f32],
	gradients: &[f32],
	rows: usize,
	columns: usize,
	learning_rate: f32,
	beta: f32,
	weight_decay: f32,
	epsilon: f32,
	iterations: u32,
) {
	let mut update = vec![0.0_f32; weights.len()];
	for index in 0..weights.len() {
		let next_momentum = beta * momentum[index] + (1.0 - beta) * gradients[index];
		update[index] = (1.0 - beta) * gradients[index] + beta * next_momentum;
		momentum[index] = next_momentum;
	}
	let transposed = rows > columns;
	let (operation_rows, operation_columns) = if transposed {
		(columns, rows)
	} else {
		(rows, columns)
	};
	let mut z = if transposed {
		transpose_reference(&update, rows, columns)
	} else {
		update
	};
	let norm = (z
		.iter()
		.map(|value| f64::from(*value) * f64::from(*value))
		.sum::<f64>() as f32
		+ epsilon)
		.sqrt();
	for value in &mut z {
		*value /= norm;
	}
	for _ in 0..iterations {
		let a = mat_mul_nt_reference(&z, &z, operation_rows, operation_rows, operation_columns);
		let aa = mat_mul_nt_reference(&a, &a, operation_rows, operation_rows, operation_rows);
		let b = a
			.iter()
			.zip(aa)
			.map(|(a, aa)| -4.7750 * a + 2.0315 * aa)
			.collect::<Vec<_>>();
		let z_transpose = transpose_reference(&z, operation_rows, operation_columns);
		let bz = mat_mul_nt_reference(
			&b,
			&z_transpose,
			operation_rows,
			operation_columns,
			operation_rows,
		);
		z = z.iter().zip(bz).map(|(z, bz)| 3.4445 * z + bz).collect();
	}
	let orthogonal = if transposed {
		transpose_reference(&z, operation_rows, operation_columns)
	} else {
		z
	};
	let scale = 0.2 * (rows.max(columns) as f32).sqrt();
	for (weight, orthogonal) in weights.iter_mut().zip(orthogonal) {
		*weight =
			(1.0 - learning_rate * weight_decay) * *weight - learning_rate * scale * orthogonal;
	}
}

fn run_muon_matrix_case(engine: &oa::Engine, rows: usize, columns: usize) -> oa::Result<()> {
	const LEARNING_RATE: f32 = 0.01;
	const BETA: f32 = 0.95;
	const WEIGHT_DECAY: f32 = 0.1;
	const EPSILON: f32 = 1.0e-7;
	const ITERATIONS: u32 = 5;
	let weights = (0..rows * columns)
		.map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.002)
		.collect::<Vec<_>>();
	let input_values = (0..2 * columns)
		.map(|index| ((index * 7 % 29) as f32 - 14.0) * 0.01)
		.collect::<Vec<_>>();
	let layer = oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(engine, [rows, columns], &weights)?,
		oa::Matrix::from_f32(engine, [rows], &vec![0.0; rows])?,
	)?;
	let input = oa::Matrix::from_f32(engine, [2, columns], &input_values)?;
	let targets = oa::Matrix::from_slice(engine, [2], &[0_u32, (rows - 1) as u32])?;
	let weight = layer.weight();
	let mut optimizer = oa::ml::Muon::with_hyperparameters(
		layer.parameters(),
		LEARNING_RATE,
		BETA,
		WEIGHT_DECAY,
		EPSILON,
		ITERATIONS,
	)?;
	let tape = oa::ml::GradientTape::new();
	let loss = oa::ml::loss::cross_entropy(&layer.forward(&input)?, &targets)?;
	tape.backward(&loss)?;
	let gradient = weight
		.gradient()
		.expect("backward produced matrix gradient")
		.read_f32()?;
	let mut expected = weights.clone();
	muon_matrix_reference(
		&mut expected,
		&mut vec![0.0; weights.len()],
		&gradient,
		rows,
		columns,
		LEARNING_RATE,
		BETA,
		WEIGHT_DECAY,
		EPSILON,
		ITERATIONS,
	);
	optimizer.step()?;
	assert_close(&weight.data().read_f32()?, &expected, 2.0e-3);
	Ok(())
}

fn muon_vector_reference(
	weights: &mut [f32],
	momentum: &mut [f32],
	gradients: &[f32],
	learning_rate: f32,
	beta: f32,
	weight_decay: f32,
) {
	for index in 0..weights.len() {
		let next_momentum = beta * momentum[index] + (1.0 - beta) * gradients[index];
		let update = (1.0 - beta) * gradients[index] + beta * next_momentum;
		momentum[index] = next_momentum;
		weights[index] =
			(1.0 - learning_rate * weight_decay) * weights[index] - learning_rate * update;
	}
}

fn transpose_reference(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
	let mut output = vec![0.0; input.len()];
	for row in 0..rows {
		for column in 0..columns {
			output[column * rows + row] = input[row * columns + column];
		}
	}
	output
}

fn mat_mul_nt_reference(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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
