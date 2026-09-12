use oa::ml::advantage::{GaeConfig, gae, gae_into, normalize};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"value {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(
	normalize_matches_donor_and_one_semantic_operation,
	engine,
	{
		let values = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
		let (plan, normalized) = engine.capture(|| normalize(&values, 1.0e-8))?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert!(plan.diagnostics().node_count() > 1);
		assert_eq!(
			plan.diagnostics().schema_owned_node_count(),
			plan.diagnostics().node_count()
		);
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::advantage::normalize");
		assert_eq!(operation.attributes().len(), 1);
		engine.submit(&plan)?.wait()?;

		let mean = 2.5_f32;
		let variance = 1.25_f32;
		let denominator = (variance + 1.0e-8).sqrt();
		let expected = [
			(1.0 - mean) / denominator,
			(2.0 - mean) / denominator,
			(3.0 - mean) / denominator,
			(4.0 - mean) / denominator,
		];
		assert_close(&normalized.read_f32()?, &expected, 1.0e-6);

		let constant = oa::Matrix::from_f32(&engine, [4], &[7.0; 4])?;
		let constant = normalize(&constant, 1.0e-8)?.read_f32()?;
		assert!(constant.iter().all(|value| value.is_finite()));
		assert_close(&constant, &[0.0; 4], 1.0e-6);
		Ok(())
	}
);

test_vk!(normalize_adjoint_matches_independent_formula, engine, {
	let values_host = [1.0_f32, -2.0, 0.5, 4.0];
	let upstream_host = [0.5_f32, -1.0, 2.0, 0.25];
	let input =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 4], &values_host)?)?;
	let row = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
	let tape = oa::ml::GradientTape::new();
	let values = input.forward(&row)?;
	let normalized = normalize(&values, 1.0e-5)?;
	let upstream = oa::Matrix::from_f32(&engine, [1, 4], &upstream_host)?;
	let loss = oa::matrix::reshape(
		&oa::matrix::sum(&oa::matrix::mul(&normalized, &upstream)?, -1)?,
		Vec::new(),
	)?;
	tape.backward(&loss)?;

	let count = values_host.len() as f32;
	let mean = values_host.iter().sum::<f32>() / count;
	let centered = values_host.map(|value| value - mean);
	let variance = centered.iter().map(|value| value * value).sum::<f32>() / count;
	let inverse_stddev = 1.0 / (variance + 1.0e-5).sqrt();
	let upstream_mean = upstream_host.iter().sum::<f32>() / count;
	let projected_mean = upstream_host
		.iter()
		.zip(centered)
		.map(|(gradient, centered)| gradient * centered)
		.sum::<f32>()
		/ count;
	let expected = std::array::from_fn::<_, 4, _>(|index| {
		inverse_stddev
			* (upstream_host[index]
				- upstream_mean
				- centered[index] * projected_mean / (variance + 1.0e-5))
	});
	let gradient = input
		.weight()
		.gradient()
		.expect("advantage normalization omitted input gradient")
		.read_f32()?;
	assert_close(&gradient, &expected, 3.0e-6);
	Ok(())
});

test_vk!(normalize_rejects_invalid_contracts, engine, {
	let values = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
	let integer = oa::Matrix::from_slice(&engine, [2], &[1_i32, 2])?;
	for invalid in [
		normalize(&values, 0.0),
		normalize(&values, f32::NAN),
		normalize(&integer, 1.0e-8),
	] {
		let error = match invalid {
			Ok(_) => panic!("invalid normalize contract was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(
	gae_matches_the_donor_oracle_and_boundary_semantics,
	engine,
	{
		let reward =
			oa::Matrix::from_f32(&engine, [4, 2], &[1.0, 0.5, 0.2, 1.0, 2.0, 0.3, 0.7, 1.5])?;
		let value =
			oa::Matrix::from_f32(&engine, [4, 2], &[0.4, 0.1, 0.3, 0.5, 0.8, 0.2, 0.6, 0.9])?;
		let next_value =
			oa::Matrix::from_f32(&engine, [4, 2], &[0.3, 0.5, 99.0, 4.0, 0.6, 0.9, 0.2, 0.4])?;
		let terminated = oa::Matrix::from_slice(&engine, [4, 2], &[0_u8, 0, 1, 0, 0, 0, 0, 0])?;
		let truncated = oa::Matrix::from_slice(&engine, [4, 2], &[0_u8, 0, 0, 1, 0, 0, 0, 0])?;
		let config = GaeConfig {
			gamma: 0.9,
			lambda: 0.8,
		};

		let (plan, result) = engine.capture(|| {
			gae(
				&reward,
				&value,
				&next_value,
				&terminated,
				&truncated,
				config,
			)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::advantage::gae"
		);
		engine.submit(&plan)?.wait()?;

		let reward_host = reward.read_f32()?;
		let value_host = value.read_f32()?;
		let next_value_host = next_value.read_f32()?;
		let terminated_host = terminated.read::<u8>()?;
		let truncated_host = truncated.read::<u8>()?;
		let mut expected_advantage = vec![0.0_f32; 8];
		let mut expected_return = vec![0.0_f32; 8];
		for environment in 0..2 {
			let mut next_advantage = 0.0;
			for time in (0..4).rev() {
				let index = time * 2 + environment;
				let bootstrap = if terminated_host[index] == 0 {
					1.0
				} else {
					0.0
				};
				let trace = if terminated_host[index] == 0 && truncated_host[index] == 0 {
					1.0
				} else {
					0.0
				};
				let delta = reward_host[index] + config.gamma * bootstrap * next_value_host[index]
					- value_host[index];
				let advantage = delta + config.gamma * config.lambda * trace * next_advantage;
				expected_advantage[index] = advantage;
				expected_return[index] = advantage + value_host[index];
				next_advantage = advantage;
			}
		}
		assert_close(&result.advantage.read_f32()?, &expected_advantage, 1.0e-6);
		assert_close(&result.returns.read_f32()?, &expected_return, 1.0e-6);

		let mut reused_advantage = oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?;
		let mut reused_return = oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?;
		gae_into(
			&reward,
			&value,
			&next_value,
			&terminated,
			&truncated,
			&mut reused_advantage,
			&mut reused_return,
			config,
		)?;
		assert_close(&reused_advantage.read_f32()?, &expected_advantage, 1.0e-6);
		assert_close(&reused_return.read_f32()?, &expected_return, 1.0e-6);
		Ok(())
	}
);

test_vk!(gae_rejects_invalid_rollout_contracts, engine, {
	let values = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
	let mask = oa::Matrix::from_slice(&engine, [2, 2], &[0_u8; 4])?;
	let wrong_mask = oa::Matrix::from_slice(&engine, [2, 1], &[0_u8; 2])?;
	let wrong_value = oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?;
	for invalid in [
		gae(
			&values,
			&values,
			&values,
			&wrong_mask,
			&mask,
			GaeConfig::default(),
		),
		gae(
			&wrong_value,
			&wrong_value,
			&wrong_value,
			&mask,
			&mask,
			GaeConfig::default(),
		),
		gae(
			&values,
			&values,
			&values,
			&mask,
			&mask,
			GaeConfig {
				gamma: f32::NAN,
				lambda: 0.95,
			},
		),
	] {
		let error = match invalid {
			Ok(_) => panic!("invalid GAE contract was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	let mut aliased_advantage = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
	let mut aliased_return = aliased_advantage.clone();
	assert_eq!(
		gae_into(
			&values,
			&values,
			&values,
			&mask,
			&mask,
			&mut aliased_advantage,
			&mut aliased_return,
			GaeConfig::default(),
		)
		.expect_err("aliased GAE outputs were accepted")
		.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
