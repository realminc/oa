fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"policy mismatch at {index}: actual={actual}, expected={expected}"
		);
	}
}

fn categorical_oracle(
	logits: &[f32],
	rows: usize,
	columns: usize,
	action: &[i32],
) -> (Vec<f32>, Vec<f32>) {
	let mut selected = Vec::with_capacity(rows);
	let mut entropy = Vec::with_capacity(rows);
	for row in 0..rows {
		let values = &logits[row * columns..(row + 1) * columns];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let denominator = values
			.iter()
			.map(|value| (value - maximum).exp())
			.sum::<f32>();
		let log_denominator = maximum + denominator.ln();
		let log_probability = values
			.iter()
			.map(|value| value - log_denominator)
			.collect::<Vec<_>>();
		selected.push(log_probability[action[row] as usize]);
		entropy.push(
			-log_probability
				.iter()
				.map(|value| value.exp() * value)
				.sum::<f32>(),
		);
	}
	(selected, entropy)
}

fn continuous_oracle(
	mean: &[f32],
	log_stddev: &[f32],
	raw_action: &[f32],
	shape: [usize; 2],
	range: [f32; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
	const LOG_TWO_PI: f32 = 1.837_877_1;
	let [rows, columns] = shape;
	let [minimum, maximum, epsilon] = range;
	let scale = 0.5 * (maximum - minimum);
	let bias = 0.5 * (maximum + minimum);
	let mut action = Vec::with_capacity(mean.len());
	let mut log_probability = Vec::with_capacity(rows);
	let mut entropy = Vec::with_capacity(rows);
	for row in 0..rows {
		let mut row_log_probability = 0.0;
		let mut row_entropy = 0.0;
		for column in 0..columns {
			let index = row * columns + column;
			let log_stddev = log_stddev[index].clamp(-20.0, 2.0);
			let normalized = (raw_action[index] - mean[index]) / log_stddev.exp();
			let squashed = raw_action[index].tanh();
			action.push(squashed * scale + bias);
			let base = -0.5 * (normalized * normalized + 2.0 * log_stddev + LOG_TWO_PI);
			let jacobian = (1.0 + epsilon - squashed * squashed).ln();
			row_log_probability += base - jacobian - scale.ln();
			row_entropy += log_stddev + 0.5 * (1.0 + LOG_TWO_PI);
		}
		log_probability.push(row_log_probability);
		entropy.push(row_entropy);
	}
	(action, log_probability, entropy)
}

test_vk!(
	categorical_policy_matches_donor_and_owns_one_semantic_operation,
	engine,
	{
		let logits_host = [1.0_f32, 2.0, -1.0, 0.5, 0.5, 0.5];
		let action_host = [1_i32, 0];
		let value_host = [0.25_f32, -0.75];
		let logits = oa::Matrix::from_f32(&engine, [2, 3], &logits_host)?;
		let action = oa::Matrix::from_slice(&engine, [2], &action_host)?;
		let value = oa::Matrix::from_f32(&engine, [2], &value_host)?;
		let (plan, result) =
			engine.capture(|| oa::ml::policy::evaluate_categorical(&logits, &action, &value))?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert!(plan.diagnostics().node_count() > 1);
		assert_eq!(
			plan.diagnostics().schema_owned_node_count(),
			plan.diagnostics().node_count()
		);
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::policy::evaluate_categorical");
		assert_eq!(operation.aliases().len(), 2);
		assert!(operation.mutated_inputs().is_empty());
		engine.submit(&plan)?.wait()?;

		let (expected_log_probability, expected_entropy) =
			categorical_oracle(&logits_host, 2, 3, &action_host);
		assert_eq!(result.action.read::<i32>()?, action_host);
		assert_eq!(result.value.read_f32()?, value_host);
		assert_close(
			&result.log_probability.read_f32()?,
			&expected_log_probability,
			1.0e-6,
		);
		assert_close(&result.entropy.read_f32()?, &expected_entropy, 1.0e-6);
		Ok(())
	}
);

test_vk!(
	categorical_policy_sampling_is_seeded_and_self_consistent,
	engine,
	{
		let logits = oa::Matrix::from_f32(
			&engine,
			[4, 3],
			&[2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, -1.0, 3.0, 0.5],
		)?;
		let value = oa::Matrix::from_f32(&engine, [4], &[0.0, 1.0, 2.0, 3.0])?;
		let first = oa::ml::policy::sample_categorical(&logits, &value, 918_273)?;
		let second = oa::ml::policy::sample_categorical(&logits, &value, 918_273)?;
		let first_action = first.action.read::<i32>()?;
		assert_eq!(first_action, second.action.read::<i32>()?);
		assert!(first_action.iter().all(|action| (0..3).contains(action)));
		let evaluated = oa::ml::policy::evaluate_categorical(&logits, &first.action, &value)?;
		assert_close(
			&first.log_probability.read_f32()?,
			&evaluated.log_probability.read_f32()?,
			1.0e-7,
		);
		Ok(())
	}
);

test_vk!(categorical_policy_entropy_adjoint_matches_cpu, engine, {
	let logits_host = [1.0_f32, 2.0, -1.0];
	let embedding =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 3], &logits_host)?)?;
	let row = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
	let action = oa::Matrix::from_slice(&engine, [1], &[1_i32])?;
	let value = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
	let tape = oa::ml::GradientTape::new();
	let logits = embedding.forward(&row)?;
	let result = oa::ml::policy::evaluate_categorical(&logits, &action, &value)?;
	let loss = oa::matrix::reshape(
		&oa::matrix::neg(&oa::matrix::sum(&result.entropy, -1)?)?,
		Vec::new(),
	)?;
	tape.backward(&loss)?;

	let maximum = logits_host
		.iter()
		.copied()
		.fold(f32::NEG_INFINITY, f32::max);
	let denominator = logits_host
		.iter()
		.map(|value| (value - maximum).exp())
		.sum::<f32>();
	let probability = logits_host
		.iter()
		.map(|value| (value - maximum).exp() / denominator)
		.collect::<Vec<_>>();
	let entropy = -probability
		.iter()
		.map(|value| value * value.ln())
		.sum::<f32>();
	let expected = probability
		.iter()
		.map(|value| value * (value.ln() + entropy))
		.collect::<Vec<_>>();
	let gradient = embedding
		.weight()
		.gradient()
		.expect("policy backward omitted logits gradient")
		.read_f32()?;
	assert_close(&gradient, &expected, 2.0e-6);
	Ok(())
});

test_vk!(categorical_policy_rejects_invalid_shapes, engine, {
	let logits = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let short_action = oa::Matrix::from_slice(&engine, [1], &[0_i32])?;
	let short_value = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
	assert!(oa::ml::policy::evaluate_categorical(&logits, &short_action, &short_value).is_err());
	assert!(oa::ml::policy::sample_categorical(&logits, &short_value, 7).is_err());
	Ok(())
});

test_vk!(
	tanh_normal_policy_matches_donor_and_one_semantic_operation,
	engine,
	{
		let mean_host = [0.1_f32, -0.4, 1.2, 0.0];
		let log_stddev_host = [-0.7_f32, 0.2, -30.0, 3.0];
		let raw_action_host = [0.3_f32, -1.1, 0.8, -0.2];
		let value_host = [0.25_f32, -0.5];
		let mean = oa::Matrix::from_f32(&engine, [2, 2], &mean_host)?;
		let log_stddev = oa::Matrix::from_f32(&engine, [2, 2], &log_stddev_host)?;
		let raw_action = oa::Matrix::from_f32(&engine, [2, 2], &raw_action_host)?;
		let value = oa::Matrix::from_f32(&engine, [2], &value_host)?;
		let (plan, result) = engine.capture(|| {
			oa::ml::policy::evaluate_tanh_normal(
				&mean,
				&log_stddev,
				&raw_action,
				&value,
				-2.0,
				3.0,
				1.0e-6,
			)
		})?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert!(plan.diagnostics().node_count() > 10);
		assert_eq!(
			plan.diagnostics().schema_owned_node_count(),
			plan.diagnostics().node_count()
		);
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::policy::evaluate_tanh_normal");
		assert_eq!(operation.aliases().len(), 2);
		assert_eq!(operation.attributes().len(), 3);
		engine.submit(&plan)?.wait()?;

		let (expected_action, expected_log_probability, expected_entropy) = continuous_oracle(
			&mean_host,
			&log_stddev_host,
			&raw_action_host,
			[2, 2],
			[-2.0, 3.0, 1.0e-6],
		);
		assert_close(&result.action.read_f32()?, &expected_action, 2.0e-6);
		assert_eq!(result.raw_action.read_f32()?, raw_action_host);
		assert_close(
			&result.log_probability.read_f32()?,
			&expected_log_probability,
			3.0e-5,
		);
		assert_close(&result.entropy.read_f32()?, &expected_entropy, 2.0e-6);
		assert_eq!(result.value.read_f32()?, value_host);
		Ok(())
	}
);

test_vk!(
	tanh_normal_sampling_is_seeded_and_self_consistent,
	engine,
	{
		let mean = oa::Matrix::from_f32(&engine, [3, 2], &[0.1, -0.2, 0.4, 0.7, -0.5, 0.3])?;
		let log_stddev = oa::Matrix::from_f32(&engine, [3, 2], &[-0.5, 0.1, -1.0, 0.3, 0.0, -0.2])?;
		let value = oa::Matrix::from_f32(&engine, [3], &[1.0, 2.0, 3.0])?;
		let (plan, first) = engine.capture(|| {
			oa::ml::policy::sample_tanh_normal(&mean, &log_stddev, &value, -1.5, 2.5, 918_273, 1.0e-6)
		})?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::policy::sample_tanh_normal"
		);
		assert_eq!(plan.semantic_graph().operations()[0].attributes().len(), 4);
		engine.submit(&plan)?.wait()?;
		let second =
			oa::ml::policy::sample_tanh_normal(&mean, &log_stddev, &value, -1.5, 2.5, 918_273, 1.0e-6)?;
		assert_eq!(first.raw_action.read_f32()?, second.raw_action.read_f32()?);
		assert_eq!(first.action.read_f32()?, second.action.read_f32()?);
		let evaluated = oa::ml::policy::evaluate_tanh_normal(
			&mean,
			&log_stddev,
			&first.raw_action,
			&value,
			-1.5,
			2.5,
			1.0e-6,
		)?;
		assert_close(
			&first.log_probability.read_f32()?,
			&evaluated.log_probability.read_f32()?,
			1.0e-6,
		);
		assert_close(
			&first.entropy.read_f32()?,
			&evaluated.entropy.read_f32()?,
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(
	tanh_normal_clamp_adjoint_matches_donor_boundaries,
	engine,
	{
		let log_stddev = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 4],
			&[-30.0, -20.0, 2.0, 3.0],
		)?)?;
		let row = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let mean = oa::Matrix::from_f32(&engine, [1, 4], &[0.0; 4])?;
		let raw_action = oa::Matrix::from_f32(&engine, [1, 4], &[0.1, -0.2, 0.3, -0.4])?;
		let value = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let log_stddev_value = log_stddev.forward(&row)?;
		let result = oa::ml::policy::evaluate_tanh_normal(
			&mean,
			&log_stddev_value,
			&raw_action,
			&value,
			-1.0,
			1.0,
			1.0e-6,
		)?;
		let loss = oa::matrix::reshape(&oa::matrix::sum(&result.entropy, -1)?, Vec::new())?;
		tape.backward(&loss)?;
		let gradient = log_stddev
			.weight()
			.gradient()
			.expect("continuous policy omitted log-stddev gradient")
			.read_f32()?;
		assert_close(&gradient, &[0.0, 1.0, 1.0, 0.0], 1.0e-6);
		Ok(())
	}
);

test_vk!(tanh_normal_policy_rejects_invalid_contracts, engine, {
	let mean = oa::Matrix::from_f32(&engine, [1, 2], &[0.0, 0.0])?;
	let log_stddev = oa::Matrix::from_f32(&engine, [1, 2], &[0.0, 0.0])?;
	let raw_action = oa::Matrix::from_f32(&engine, [1, 1], &[0.0])?;
	let value = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
	assert!(
		oa::ml::policy::evaluate_tanh_normal(
			&mean,
			&log_stddev,
			&raw_action,
			&value,
			-1.0,
			1.0,
			1.0e-6,
		)
		.is_err()
	);
	assert!(
		oa::ml::policy::sample_tanh_normal(&mean, &log_stddev, &value, 1.0, -1.0, 7, 1.0e-6,).is_err()
	);
	Ok(())
});

test_vk!(
	tanh_normal_sampling_reparameterization_has_exact_adjoint,
	engine,
	{
		let mean = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 3],
			&[0.2, -0.3, 0.7],
		)?)?;
		let log_stddev = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[1, 3],
			&[-0.5, 0.1, -1.0],
		)?)?;
		let row = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let value = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let tape = oa::ml::GradientTape::new();
		let mean_value = mean.forward(&row)?;
		let log_stddev_value = log_stddev.forward(&row)?;
		let result = oa::ml::policy::sample_tanh_normal(
			&mean_value,
			&log_stddev_value,
			&value,
			-1.0,
			1.0,
			42_424,
			1.0e-6,
		)?;
		let loss = oa::matrix::reshape(&oa::matrix::sum(&result.raw_action, -1)?, Vec::new())?;
		tape.backward(&loss)?;
		let raw_action = result.raw_action.read_f32()?;
		let mean_host = mean_value.read_f32()?;
		assert_close(
			&mean
				.weight()
				.gradient()
				.expect("reparameterization omitted mean gradient")
				.read_f32()?,
			&[1.0; 3],
			1.0e-6,
		);
		let expected_log_stddev = raw_action
			.iter()
			.zip(mean_host)
			.map(|(raw, mean)| raw - mean)
			.collect::<Vec<_>>();
		assert_close(
			&log_stddev
				.weight()
				.gradient()
				.expect("reparameterization omitted log-stddev gradient")
				.read_f32()?,
			&expected_log_stddev,
			2.0e-6,
		);
		Ok(())
	}
);
