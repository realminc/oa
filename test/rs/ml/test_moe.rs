fn host_route_weights(
	probabilities: &[f32],
	expert_indices: &[i32],
	tokens: usize,
	experts: usize,
	routes_per_token: usize,
) -> Vec<f32> {
	let mut output = vec![0.0; tokens * routes_per_token];
	for token in 0..tokens {
		let denominator = (0..routes_per_token)
			.filter_map(|route_slot| {
				usize::try_from(expert_indices[token * routes_per_token + route_slot])
					.ok()
					.filter(|expert| *expert < experts)
					.map(|expert| probabilities[token * experts + expert])
			})
			.sum::<f32>();
		if denominator > 0.0 {
			for route_slot in 0..routes_per_token {
				if let Ok(expert) =
					usize::try_from(expert_indices[token * routes_per_token + route_slot])
					&& expert < experts
				{
					output[token * routes_per_token + route_slot] =
						probabilities[token * experts + expert] / denominator;
				}
			}
		}
	}
	output
}

fn host_mse(values: &[f32], target: &[f32]) -> f32 {
	values
		.iter()
		.zip(target)
		.map(|(value, target)| (value - target).powi(2))
		.sum::<f32>()
		/ values.len() as f32
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: found {actual}, expected {expected}, tolerance {tolerance}"
		);
	}
}

test_vk!(moe_route_weights_matches_the_donor_vector, engine, {
	let probabilities = [
		0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0,
	];
	let expert_indices = [3_i32, 1, 2, 0, -1, 7];
	let expected = host_route_weights(&probabilities, &expert_indices, 3, 4, 2);
	let probabilities = oa::Matrix::from_f32(&engine, [3, 4], &probabilities)?;
	let expert_indices = oa::Matrix::from_slice(&engine, [3, 2], &expert_indices)?;
	let (plan, output) =
		engine.capture(|| oa::ml::matrix::moe_route_weights(&probabilities, &expert_indices))?;
	assert_eq!(output.shape(), [3, 2]);
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::matrix::moe_route_weights"
	);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("moe_route"))
		.expect("MoE route report must be valid JSON");
	assert_eq!(
		report["nodes"][0]["kernel"],
		"ml.matrix.moe_route_weights.f32"
	);
	assert!(!report["nodes"][0]["physical_write"].is_null());
	engine.submit(&plan)?.wait()?;
	assert_close(&output.read_f32()?, &expected, 1.0e-6);
	Ok(())
});

test_vk!(
	moe_route_weights_reverse_matches_finite_differences,
	engine,
	{
		let probabilities = [0.15_f32, 0.2, 0.35, 0.3, 0.4, 0.1, 0.2, 0.3];
		let expert_indices = [2_i32, 0, 3, 1];
		let target = [0.6_f32, 0.4, 0.7, 0.3];
		let parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[2, 4],
			&probabilities,
		)?)?;
		let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let expert_indices_matrix = oa::Matrix::from_slice(&engine, [2, 2], &expert_indices)?;
		let tape = oa::ml::GradientTape::new();
		let probability_values = parameter.forward(&rows)?;
		let route_weights =
			oa::ml::matrix::moe_route_weights(&probability_values, &expert_indices_matrix)?;
		let target_matrix = oa::Matrix::from_f32(&engine, [2, 2], &target)?;
		let loss = oa::ml::loss::mse(&route_weights, &target_matrix)?;
		tape.backward(&loss)?;
		let actual = parameter
			.weight()
			.gradient()
			.expect("MoE route probability gradient is missing")
			.read_f32()?;

		const EPSILON: f32 = 1.0e-3;
		let mut expected = vec![0.0; probabilities.len()];
		for index in 0..probabilities.len() {
			let mut plus = probabilities;
			plus[index] += EPSILON;
			let plus_loss = host_mse(
				&host_route_weights(&plus, &expert_indices, 2, 4, 2),
				&target,
			);
			let mut minus = probabilities;
			minus[index] -= EPSILON;
			let minus_loss = host_mse(
				&host_route_weights(&minus, &expert_indices, 2, 4, 2),
				&target,
			);
			expected[index] = (plus_loss - minus_loss) / (2.0 * EPSILON);
		}
		assert_close(&actual, &expected, 3.0e-4);
		Ok(())
	}
);

test_vk!(moe_route_weights_rejects_invalid_contracts, engine, {
	let other_engine = oa::Engine::new()?;
	let probabilities = oa::Matrix::from_f32(&engine, [2, 4], &[0.25; 8])?;
	let rank_one = oa::Matrix::from_f32(&engine, [8], &[0.25; 8])?;
	let indices = oa::Matrix::from_slice(&engine, [2, 2], &[0_i32, 1, 2, 3])?;
	let too_many = oa::Matrix::from_slice(&engine, [2, 5], &[0_i32; 10])?;
	let unsigned = oa::Matrix::from_slice(&engine, [2, 2], &[0_u32, 1, 2, 3])?;
	let foreign = oa::Matrix::from_slice(&other_engine, [2, 2], &[0_i32, 1, 2, 3])?;
	for result in [
		oa::ml::matrix::moe_route_weights(&rank_one, &indices),
		oa::ml::matrix::moe_route_weights(&probabilities, &too_many),
		oa::ml::matrix::moe_route_weights(&probabilities, &unsigned),
		oa::ml::matrix::moe_route_weights(&probabilities, &foreign),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid MoE route contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});
