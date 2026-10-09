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
				if let Ok(expert) = usize::try_from(expert_indices[token * routes_per_token + route_slot])
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

fn host_grouped_linear_m(
	input: &[f32],
	weight: &[f32],
	bias: &[f32],
	offsets: &[u32],
	rows: usize,
	output_features: usize,
	input_features: usize,
) -> Vec<f32> {
	let mut output = vec![0.0; rows * output_features];
	for expert in 0..offsets.len() - 1 {
		for row in offsets[expert] as usize..offsets[expert + 1] as usize {
			for output_feature in 0..output_features {
				let mut value = bias[expert * output_features + output_feature];
				for input_feature in 0..input_features {
					value += input[row * input_features + input_feature]
						* weight[(expert * output_features + output_feature) * input_features + input_feature];
				}
				output[row * output_features + output_feature] = value;
			}
		}
	}
	output
}

fn host_grouped_gemm_m(
	input: &[f32],
	weight: &[f32],
	offsets: &[u32],
	rows: usize,
	output_features: usize,
	input_features: usize,
) -> Vec<f32> {
	host_grouped_linear_m(
		input,
		weight,
		&vec![0.0; (offsets.len() - 1) * output_features],
		offsets,
		rows,
		output_features,
		input_features,
	)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the independent oracle keeps every donor-layout parameter explicit"
)]
fn host_sparse_moe(
	input: &[f32],
	norm_weight: &[f32],
	router_weight: &[f32],
	router_bias: &[f32],
	gate_up_weight: &[f32],
	gate_up_bias: &[f32],
	down_weight: &[f32],
	down_bias: &[f32],
	tokens: usize,
	model_width: usize,
	hidden_width: usize,
	num_experts: usize,
	experts_per_token: usize,
	epsilon: f32,
) -> (Vec<f32>, Vec<f32>) {
	let mut output = input.to_vec();
	let mut selection_mask = vec![0.0; tokens * num_experts];
	for token in 0..tokens {
		let row = &input[token * model_width..(token + 1) * model_width];
		let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>() / model_width as f32
			+ epsilon)
			.sqrt()
			.recip();
		let normalized = (0..model_width)
			.map(|column| row[column] * inverse_rms * norm_weight[column])
			.collect::<Vec<_>>();
		let logits = (0..num_experts)
			.map(|expert| {
				router_bias[expert]
					+ (0..model_width)
						.map(|column| normalized[column] * router_weight[expert * model_width + column])
						.sum::<f32>()
			})
			.collect::<Vec<_>>();
		let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let exponentials = logits
			.iter()
			.map(|value| (value - maximum).exp())
			.collect::<Vec<_>>();
		let exponential_sum = exponentials.iter().sum::<f32>();
		let probabilities = exponentials
			.iter()
			.map(|value| value / exponential_sum)
			.collect::<Vec<_>>();
		let mut selected = (0..num_experts).collect::<Vec<_>>();
		selected.sort_by(|left, right| {
			logits[*right]
				.total_cmp(&logits[*left])
				.then_with(|| left.cmp(right))
		});
		selected.truncate(experts_per_token);
		let route_denominator = selected
			.iter()
			.map(|expert| probabilities[*expert])
			.sum::<f32>();
		for expert in selected {
			selection_mask[token * num_experts + expert] = 1.0;
			let route_gate = probabilities[expert] / route_denominator;
			let mut hidden = vec![0.0; hidden_width];
			for hidden_feature in 0..hidden_width {
				let mut gate = gate_up_bias[expert * 2 * hidden_width + hidden_feature];
				let mut up = gate_up_bias[expert * 2 * hidden_width + hidden_width + hidden_feature];
				for column in 0..model_width {
					gate += normalized[column]
						* gate_up_weight[(expert * 2 * hidden_width + hidden_feature) * model_width + column];
					up += normalized[column]
						* gate_up_weight
							[(expert * 2 * hidden_width + hidden_width + hidden_feature) * model_width + column];
				}
				hidden[hidden_feature] = gate / (1.0 + (-gate).exp()) * up;
			}
			for column in 0..model_width {
				let mut delta = down_bias[expert * model_width + column];
				for hidden_feature in 0..hidden_width {
					delta += hidden[hidden_feature]
						* down_weight[(expert * model_width + column) * hidden_width + hidden_feature];
				}
				output[token * model_width + column] += route_gate * delta;
			}
		}
	}
	(output, selection_mask)
}

fn finite_difference(values: &[f32], epsilon: f32, evaluate: impl Fn(&[f32]) -> f32) -> Vec<f32> {
	(0..values.len())
		.map(|index| {
			let mut plus = values.to_vec();
			plus[index] += epsilon;
			let mut minus = values.to_vec();
			minus[index] -= epsilon;
			(evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
		})
		.collect()
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

test_vk!(grouped_gemm_m_matches_variable_expert_row_oracle, engine, {
	let input = [1.0_f32, 2.0, -1.0, 3.0, 0.5, -2.0];
	let weight = [
		1.0_f32, 0.0, 0.0, 2.0, // expert 0
		3.0, -1.0, 0.5, 0.25, // expert 1 (empty)
		-2.0, 1.0, 1.5, -0.5, // expert 2
	];
	let offsets = [0_u32, 2, 2, 3];
	let expected = host_grouped_gemm_m(&input, &weight, &offsets, 3, 2, 2);
	let input = oa::Matrix::from_f32(&engine, [3, 2], &input)?;
	let weight = oa::Matrix::from_f32(&engine, [3, 2, 2], &weight)?;
	let offsets = oa::Matrix::from_slice(&engine, [4], &offsets)?;
	let (plan, output) =
		engine.capture(|| oa::ml::matrix::grouped_gemm_m(&input, &weight, &offsets))?;
	assert_eq!(output.shape(), [3, 2]);
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::matrix::grouped_gemm_m"
	);
	engine.submit(&plan)?.wait()?;
	assert_close(&output.read_f32()?, &expected, 1.0e-6);
	Ok(())
});

test_vk!(grouped_gemm_m_reverse_matches_closed_form, engine, {
	let input_values = [1.0_f32, 2.0, -1.0, 3.0, 0.5, -2.0];
	let weight_values = [
		1.0_f32, 0.0, 0.0, 2.0, 3.0, -1.0, 0.5, 0.25, -2.0, 1.0, 1.5, -0.5,
	];
	let offsets = [0_u32, 2, 2, 3];
	let target = [0.5_f32, -1.0, 1.0, 2.0, -0.25, 0.75];
	let input_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[1, input_values.len()],
		&input_values,
	)?)?;
	let weight_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[1, weight_values.len()],
		&weight_values,
	)?)?;
	let index = oa::Matrix::from_slice(&engine, [], &[0_u32])?;
	let offsets_matrix = oa::Matrix::from_slice(&engine, [4], &offsets)?;
	let tape = oa::ml::GradientTape::new();
	let input = input_parameter.forward(&index)?.reshape([3, 2])?;
	let weight = weight_parameter.forward(&index)?.reshape([3, 2, 2])?;
	let output = oa::ml::matrix::grouped_gemm_m(&input, &weight, &offsets_matrix)?;
	let target_matrix = oa::Matrix::from_f32(&engine, [3, 2], &target)?;
	let loss = oa::ml::loss::mse(&output, &target_matrix)?;
	tape.backward(&loss)?;

	let host_output = host_grouped_gemm_m(&input_values, &weight_values, &offsets, 3, 2, 2);
	let loss_scale = 2.0 / target.len() as f32;
	let output_gradient = host_output
		.iter()
		.zip(target)
		.map(|(output, target)| loss_scale * (output - target))
		.collect::<Vec<_>>();
	let mut expected_input = vec![0.0_f32; input_values.len()];
	let mut expected_weight = vec![0.0_f32; weight_values.len()];
	for expert in 0..3 {
		for row in offsets[expert] as usize..offsets[expert + 1] as usize {
			for output_feature in 0..2 {
				let gradient = output_gradient[row * 2 + output_feature];
				for input_feature in 0..2 {
					expected_input[row * 2 + input_feature] +=
						gradient * weight_values[(expert * 2 + output_feature) * 2 + input_feature];
					expected_weight[(expert * 2 + output_feature) * 2 + input_feature] +=
						gradient * input_values[row * 2 + input_feature];
				}
			}
		}
	}
	assert_close(
		&input_parameter
			.weight()
			.gradient()
			.expect("grouped GEMM input gradient is missing")
			.read_f32()?,
		&expected_input,
		2.0e-6,
	);
	assert_close(
		&weight_parameter
			.weight()
			.gradient()
			.expect("grouped GEMM weight gradient is missing")
			.read_f32()?,
		&expected_weight,
		2.0e-6,
	);
	Ok(())
});

test_vk!(grouped_gemm_m_rejects_invalid_contracts, engine, {
	let other_engine = oa::Engine::new()?;
	let input = oa::Matrix::from_f32(&engine, [3, 2], &[1.0; 6])?;
	let weight = oa::Matrix::from_f32(&engine, [3, 2, 2], &[1.0; 12])?;
	let offsets = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
	let foreign_offsets = oa::Matrix::from_slice(&other_engine, [4], &[0_u32, 1, 2, 3])?;
	let signed_offsets = oa::Matrix::from_slice(&engine, [4], &[0_i32, 1, 2, 3])?;
	let bad_weight = oa::Matrix::from_f32(&engine, [3, 2, 3], &[1.0; 18])?;
	for result in [
		oa::ml::matrix::grouped_gemm_m(&input, &weight, &foreign_offsets),
		oa::ml::matrix::grouped_gemm_m(&input, &weight, &signed_offsets),
		oa::ml::matrix::grouped_gemm_m(&input, &bad_weight, &offsets),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid grouped GEMM contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});

test_vk!(
	moe_route_weights_reverse_matches_finite_differences,
	engine,
	{
		let probabilities = [0.15_f32, 0.2, 0.35, 0.3, 0.4, 0.1, 0.2, 0.3];
		let expert_indices = [2_i32, 0, 3, 1];
		let target = [0.6_f32, 0.4, 0.7, 0.3];
		let parameter =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [2, 4], &probabilities)?)?;
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

test_vk!(moe_pack_and_combine_match_stable_route_oracle, engine, {
	let input = oa::Matrix::from_f32(&engine, [3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
	let expert_indices = oa::Matrix::from_slice(&engine, [3, 2], &[2_i32, 0, 1, 2, 0, 1])?;
	let plan = oa::matrix::moe_expert_plan(&expert_indices, 3)?;
	let packed = oa::ml::matrix::moe_gather(&input, &plan.packed_token, &plan.inverse)?;
	assert_eq!(packed.shape(), [6, 2]);
	assert_eq!(
		packed.read_f32()?,
		vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0]
	);
	let gates = oa::Matrix::from_f32(&engine, [3, 2], &[0.25, 0.75, 0.6, 0.4, 0.1, 0.9])?;
	let output = oa::ml::matrix::moe_combine(&packed, &gates, &plan.inverse, &plan.packed_slot)?;
	assert_eq!(output.shape(), [3, 2]);
	assert_close(&output.read_f32()?, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0e-6);
	Ok(())
});

test_vk!(moe_pack_and_combine_reverse_matches_closed_form, engine, {
	let input_values = [1.0_f32, -2.0, 0.5, 3.0, -1.0, 2.0];
	let gate_values = [0.2_f32, 0.8, 0.6, 0.4, 0.3, 0.7];
	let target = [0.0_f32, 1.0, -1.0, 2.0, 0.5, -0.5];
	let input_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 2], &input_values)?)?;
	let gate_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 2], &gate_values)?)?;
	let rows = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let expert_indices = oa::Matrix::from_slice(&engine, [3, 2], &[2_i32, 0, 1, 2, 0, 1])?;
	let plan = oa::matrix::moe_expert_plan(&expert_indices, 3)?;
	let tape = oa::ml::GradientTape::new();
	let input = input_parameter.forward(&rows)?;
	let gates = gate_parameter.forward(&rows)?;
	let packed = oa::ml::matrix::moe_gather(&input, &plan.packed_token, &plan.inverse)?;
	let output = oa::ml::matrix::moe_combine(&packed, &gates, &plan.inverse, &plan.packed_slot)?;
	let target_matrix = oa::Matrix::from_f32(&engine, [3, 2], &target)?;
	let loss = oa::ml::loss::mse(&output, &target_matrix)?;
	tape.backward(&loss)?;

	let scale = 2.0 / target.len() as f32;
	let mut expected_input = [0.0_f32; 6];
	let mut expected_gates = [0.0_f32; 6];
	for token in 0..3 {
		let gate_sum = gate_values[token * 2] + gate_values[token * 2 + 1];
		let mut route_gradient = 0.0;
		for column in 0..2 {
			let index = token * 2 + column;
			let output_value = input_values[index] * gate_sum;
			let output_gradient = scale * (output_value - target[index]);
			expected_input[index] = output_gradient * gate_sum;
			route_gradient += output_gradient * input_values[index];
		}
		expected_gates[token * 2] = route_gradient;
		expected_gates[token * 2 + 1] = route_gradient;
	}
	assert_close(
		&input_parameter
			.weight()
			.gradient()
			.expect("MoE packed-input gradient is missing")
			.read_f32()?,
		&expected_input,
		2.0e-6,
	);
	assert_close(
		&gate_parameter
			.weight()
			.gradient()
			.expect("MoE route-gate gradient is missing")
			.read_f32()?,
		&expected_gates,
		2.0e-6,
	);
	Ok(())
});

test_vk!(moe_pack_and_combine_reject_invalid_contracts, engine, {
	let input = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	let packed_token = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 0, 1])?;
	let inverse = oa::Matrix::from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let wrong_map = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let signed_map = oa::Matrix::from_slice(&engine, [4], &[0_i32, 1, 0, 1])?;
	for result in [
		oa::ml::matrix::moe_gather(&input, &signed_map, &inverse),
		oa::ml::matrix::moe_gather(&input, &packed_token, &wrong_map),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid MoE gather contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}

	let packed = oa::Matrix::from_f32(&engine, [4, 3], &[1.0; 12])?;
	let gates = oa::Matrix::from_f32(&engine, [2, 2], &[0.5; 4])?;
	let packed_slot = oa::Matrix::from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let bad_gates = oa::Matrix::from_f32(&engine, [2, 3], &[1.0 / 3.0; 6])?;
	for result in [
		oa::ml::matrix::moe_combine(&packed, &bad_gates, &inverse, &packed_slot),
		oa::ml::matrix::moe_combine(&packed, &gates, &wrong_map, &packed_slot),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid MoE combine contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});

test_vk!(
	grouped_linear_m_matches_variable_expert_row_oracle,
	engine,
	{
		let input = [
			1.0_f32, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, 0.25, -0.5, -1.5, 2.5, 1.0,
		];
		let weight = [
			0.2_f32, -0.3, 0.5, 0.1, -0.4, 0.7, 0.9, -0.2, 0.3, 0.6, -0.8, 0.4,
		];
		let bias = [0.1_f32, -0.2, 0.3, 0.0, 0.25, -0.5];
		let offsets = [0_u32, 2, 3, 6];
		let expected = host_grouped_linear_m(&input, &weight, &bias, &offsets, 6, 2, 2);
		let input = oa::Matrix::from_f32(&engine, [6, 2], &input)?;
		let weight = oa::Matrix::from_f32(&engine, [3, 2, 2], &weight)?;
		let bias = oa::Matrix::from_f32(&engine, [3, 2], &bias)?;
		let offsets = oa::Matrix::from_slice(&engine, [4], &offsets)?;
		let (plan, output) =
			engine.capture(|| oa::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets))?;
		assert_eq!(output.shape(), [6, 2]);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::ml::matrix::grouped_linear_m"
		);
		engine.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &expected, 2.0e-5);
		Ok(())
	}
);

test_vk!(grouped_linear_m_reverse_matches_closed_form, engine, {
	let input_values = [
		1.0_f32, -2.0, 0.5, 3.0, -1.0, 2.0, 4.0, 0.25, -0.5, -1.5, 2.5, 1.0,
	];
	let weight_values = [
		0.2_f32, -0.3, 0.5, 0.1, -0.4, 0.7, 0.9, -0.2, 0.3, 0.6, -0.8, 0.4,
	];
	let bias_values = [0.1_f32, -0.2, 0.3, 0.0, 0.25, -0.5];
	let target = [
		0.0_f32, 1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.25, -0.75, 0.5, 1.0, -1.0,
	];
	let offsets_values = [0_u32, 2, 3, 6];
	let input_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [6, 2], &input_values)?)?;
	let weight_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 4], &weight_values)?)?;
	let bias_parameter =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [3, 2], &bias_values)?)?;
	let row_indices = oa::Matrix::from_slice(&engine, [6], &[0_u32, 1, 2, 3, 4, 5])?;
	let expert_indices = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let offsets = oa::Matrix::from_slice(&engine, [4], &offsets_values)?;
	let tape = oa::ml::GradientTape::new();
	let input = input_parameter.forward(&row_indices)?;
	let weight = weight_parameter
		.forward(&expert_indices)?
		.reshape([3, 2, 2])?;
	let bias = bias_parameter.forward(&expert_indices)?;
	let output = oa::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)?;
	let target_matrix = oa::Matrix::from_f32(&engine, [6, 2], &target)?;
	let loss = oa::ml::loss::mse(&output, &target_matrix)?;
	tape.backward(&loss)?;

	let expected_output = host_grouped_linear_m(
		&input_values,
		&weight_values,
		&bias_values,
		&offsets_values,
		6,
		2,
		2,
	);
	let mut expected_input = [0.0_f32; 12];
	let mut expected_weight = [0.0_f32; 12];
	let mut expected_bias = [0.0_f32; 6];
	let loss_scale = 2.0 / target.len() as f32;
	for expert in 0..3 {
		for row in offsets_values[expert] as usize..offsets_values[expert + 1] as usize {
			for output_feature in 0..2 {
				let gradient = loss_scale
					* (expected_output[row * 2 + output_feature] - target[row * 2 + output_feature]);
				expected_bias[expert * 2 + output_feature] += gradient;
				for input_feature in 0..2 {
					expected_input[row * 2 + input_feature] +=
						gradient * weight_values[(expert * 2 + output_feature) * 2 + input_feature];
					expected_weight[(expert * 2 + output_feature) * 2 + input_feature] +=
						gradient * input_values[row * 2 + input_feature];
				}
			}
		}
	}
	assert_close(
		&input_parameter
			.weight()
			.gradient()
			.expect("grouped-Linear input gradient is missing")
			.read_f32()?,
		&expected_input,
		2.0e-5,
	);
	assert_close(
		&weight_parameter
			.weight()
			.gradient()
			.expect("grouped-Linear weight gradient is missing")
			.read_f32()?,
		&expected_weight,
		2.0e-5,
	);
	assert_close(
		&bias_parameter
			.weight()
			.gradient()
			.expect("grouped-Linear bias gradient is missing")
			.read_f32()?,
		&expected_bias,
		2.0e-5,
	);
	Ok(())
});

test_vk!(grouped_linear_m_rejects_invalid_contracts, engine, {
	let input = oa::Matrix::from_f32(&engine, [4, 2], &[1.0; 8])?;
	let weight = oa::Matrix::from_f32(&engine, [2, 3, 2], &[0.5; 12])?;
	let bias = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
	let offsets = oa::Matrix::from_slice(&engine, [3], &[0_u32, 2, 4])?;
	let wrong_weight = oa::Matrix::from_f32(&engine, [2, 3, 4], &[0.5; 24])?;
	let wrong_bias = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
	let signed_offsets = oa::Matrix::from_slice(&engine, [3], &[0_i32, 2, 4])?;
	for result in [
		oa::ml::matrix::grouped_linear_m(&input, &wrong_weight, &bias, &offsets),
		oa::ml::matrix::grouped_linear_m(&input, &weight, &wrong_bias, &offsets),
		oa::ml::matrix::grouped_linear_m(&input, &weight, &bias, &signed_offsets),
	] {
		assert_eq!(
			result
				.err()
				.expect("invalid grouped-Linear contract was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});

test_vk!(sparse_moe_matches_independent_cpu_oracle, engine, {
	const TOKENS: usize = 5;
	const MODEL_WIDTH: usize = 2;
	const HIDDEN_WIDTH: usize = 2;
	const EXPERTS: usize = 3;
	const ROUTES: usize = 2;
	const EPSILON: f32 = 1.0e-5;
	let input = [1.0_f32, -0.5, -1.5, 0.25, 0.2, 1.3, 2.0, 0.75, -0.8, -1.1];
	let norm_weight = [1.1_f32, 0.9];
	let router_weight = [0.8_f32, -0.2, -0.5, 0.7, 0.15, -0.9];
	let router_bias = [0.1_f32, -0.05, 0.2];
	let gate_up_weight = [
		0.2_f32, -0.3, 0.4, 0.1, -0.6, 0.5, 0.7, -0.2, -0.1, 0.8, 0.3, -0.4, 0.9, 0.2, -0.5, 0.6, 0.45,
		-0.25, -0.7, 0.35, 0.15, 0.55, -0.3, -0.8,
	];
	let gate_up_bias = [
		0.05_f32, -0.1, 0.2, 0.0, -0.15, 0.25, 0.1, -0.2, 0.3, -0.05, -0.1, 0.15,
	];
	let down_weight = [
		0.4_f32, -0.2, 0.1, 0.6, -0.3, 0.8, 0.7, -0.5, 0.25, 0.45, -0.65, 0.35,
	];
	let down_bias = [0.05_f32, -0.1, 0.2, 0.0, -0.15, 0.25];
	let (expected, expected_mask) = host_sparse_moe(
		&input,
		&norm_weight,
		&router_weight,
		&router_bias,
		&gate_up_weight,
		&gate_up_bias,
		&down_weight,
		&down_bias,
		TOKENS,
		MODEL_WIDTH,
		HIDDEN_WIDTH,
		EXPERTS,
		ROUTES,
		EPSILON,
	);
	let moe = oa::ml::nn::Moe::from_matrices(
		oa::Matrix::from_f32(&engine, [MODEL_WIDTH], &norm_weight)?,
		oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH], &router_weight)?,
		oa::Matrix::from_f32(&engine, [EXPERTS], &router_bias)?,
		oa::Matrix::from_f32(
			&engine,
			[EXPERTS, 2 * HIDDEN_WIDTH, MODEL_WIDTH],
			&gate_up_weight,
		)?,
		oa::Matrix::from_f32(&engine, [EXPERTS, 2 * HIDDEN_WIDTH], &gate_up_bias)?,
		oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH, HIDDEN_WIDTH], &down_weight)?,
		oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH], &down_bias)?,
		ROUTES,
		EPSILON,
	)?;
	let input = oa::Matrix::from_f32(&engine, [TOKENS, MODEL_WIDTH], &input)?;
	let output = moe.forward(&input)?;
	assert_close(&output.read_f32()?, &expected, 3.0e-5);
	let mask = moe
		.last_selection_mask()
		.expect("MoE selection mask must exist after forward");
	assert_eq!(mask.shape(), [TOKENS, EXPERTS]);
	assert_eq!(mask.read_f32()?, expected_mask);
	assert_eq!(
		mask.read_f32()?.iter().sum::<f32>(),
		(TOKENS * ROUTES) as f32
	);
	Ok(())
});

test_vk!(
	sparse_moe_reverse_matches_end_to_end_finite_differences,
	engine,
	{
		use oa::ml::Module as _;

		const TOKENS: usize = 3;
		const MODEL_WIDTH: usize = 2;
		const HIDDEN_WIDTH: usize = 1;
		const EXPERTS: usize = 2;
		const ROUTES: usize = 2;
		const EPSILON: f32 = 1.0e-5;
		const DIFFERENCE_EPSILON: f32 = 1.0e-3;
		let input_values = vec![1.0_f32, -0.5, -1.5, 0.25, 0.2, 1.3];
		let norm_weight = vec![1.1_f32, 0.9];
		let router_weight = vec![0.8_f32, -0.2, -0.5, 0.7];
		let router_bias = vec![0.1_f32, -0.05];
		let gate_up_weight = vec![0.2_f32, -0.3, -0.6, 0.5, -0.1, 0.8, 0.9, 0.2];
		let gate_up_bias = vec![0.05_f32, 0.2, -0.15, 0.1];
		let down_weight = vec![0.4_f32, -0.2, -0.3, 0.8];
		let down_bias = vec![0.05_f32, -0.1, 0.2, 0.0];
		let target = vec![0.0_f32, 0.8, -0.7, 0.4, 0.5, -0.3];
		let host_loss = |input: &[f32],
		                 norm: &[f32],
		                 router_w: &[f32],
		                 router_b: &[f32],
		                 gate_w: &[f32],
		                 gate_b: &[f32],
		                 down_w: &[f32],
		                 down_b: &[f32]| {
			let output = host_sparse_moe(
				input,
				norm,
				router_w,
				router_b,
				gate_w,
				gate_b,
				down_w,
				down_b,
				TOKENS,
				MODEL_WIDTH,
				HIDDEN_WIDTH,
				EXPERTS,
				ROUTES,
				EPSILON,
			)
			.0;
			host_mse(&output, &target)
		};
		let input_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[TOKENS, MODEL_WIDTH],
			&input_values,
		)?)?;
		let rows = oa::Matrix::from_slice(&engine, [TOKENS], &[0_u32, 1, 2])?;
		let moe = oa::ml::nn::Moe::from_matrices(
			oa::Matrix::from_f32(&engine, [MODEL_WIDTH], &norm_weight)?,
			oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH], &router_weight)?,
			oa::Matrix::from_f32(&engine, [EXPERTS], &router_bias)?,
			oa::Matrix::from_f32(
				&engine,
				[EXPERTS, 2 * HIDDEN_WIDTH, MODEL_WIDTH],
				&gate_up_weight,
			)?,
			oa::Matrix::from_f32(&engine, [EXPERTS, 2 * HIDDEN_WIDTH], &gate_up_bias)?,
			oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH, HIDDEN_WIDTH], &down_weight)?,
			oa::Matrix::from_f32(&engine, [EXPERTS, MODEL_WIDTH], &down_bias)?,
			ROUTES,
			EPSILON,
		)?;
		let tape = oa::ml::GradientTape::new();
		let input = input_parameter.forward(&rows)?;
		let output = moe.forward(&input)?;
		let target_matrix = oa::Matrix::from_f32(&engine, [TOKENS, MODEL_WIDTH], &target)?;
		let loss = oa::ml::loss::mse(&output, &target_matrix)?;
		tape.backward(&loss)?;

		let expected_input = finite_difference(&input_values, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				candidate,
				&norm_weight,
				&router_weight,
				&router_bias,
				&gate_up_weight,
				&gate_up_bias,
				&down_weight,
				&down_bias,
			)
		});
		let expected_norm = finite_difference(&norm_weight, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				&input_values,
				candidate,
				&router_weight,
				&router_bias,
				&gate_up_weight,
				&gate_up_bias,
				&down_weight,
				&down_bias,
			)
		});
		let expected_router_weight =
			finite_difference(&router_weight, DIFFERENCE_EPSILON, |candidate| {
				host_loss(
					&input_values,
					&norm_weight,
					candidate,
					&router_bias,
					&gate_up_weight,
					&gate_up_bias,
					&down_weight,
					&down_bias,
				)
			});
		let expected_router_bias = finite_difference(&router_bias, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				&input_values,
				&norm_weight,
				&router_weight,
				candidate,
				&gate_up_weight,
				&gate_up_bias,
				&down_weight,
				&down_bias,
			)
		});
		let expected_gate_weight =
			finite_difference(&gate_up_weight, DIFFERENCE_EPSILON, |candidate| {
				host_loss(
					&input_values,
					&norm_weight,
					&router_weight,
					&router_bias,
					candidate,
					&gate_up_bias,
					&down_weight,
					&down_bias,
				)
			});
		let expected_gate_bias = finite_difference(&gate_up_bias, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				&input_values,
				&norm_weight,
				&router_weight,
				&router_bias,
				&gate_up_weight,
				candidate,
				&down_weight,
				&down_bias,
			)
		});
		let expected_down_weight = finite_difference(&down_weight, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				&input_values,
				&norm_weight,
				&router_weight,
				&router_bias,
				&gate_up_weight,
				&gate_up_bias,
				candidate,
				&down_bias,
			)
		});
		let expected_down_bias = finite_difference(&down_bias, DIFFERENCE_EPSILON, |candidate| {
			host_loss(
				&input_values,
				&norm_weight,
				&router_weight,
				&router_bias,
				&gate_up_weight,
				&gate_up_bias,
				&down_weight,
				candidate,
			)
		});
		assert_close(
			&input_parameter
				.weight()
				.gradient()
				.expect("MoE input gradient is missing")
				.read_f32()?,
			&expected_input,
			1.5e-3,
		);
		let named = moe.all_named_parameters()?;
		let gradient = |path: &str| -> oa::Result<Vec<f32>> {
			named
				.iter()
				.find(|entry| entry.path() == path)
				.expect("expected MoE parameter path is missing")
				.parameter()
				.gradient()
				.expect("expected MoE parameter gradient is missing")
				.read_f32()
		};
		for (path, expected) in [
			("norm.weight", expected_norm),
			("router.weight", expected_router_weight),
			("router.bias", expected_router_bias),
			("expert_gate_up_weight", expected_gate_weight),
			("expert_gate_up_bias", expected_gate_bias),
			("expert_down_weight", expected_down_weight),
			("expert_down_bias", expected_down_bias),
		] {
			assert_close(&gradient(path)?, &expected, 1.5e-3);
		}
		Ok(())
	}
);

test_vk!(
	dense_and_sparse_moe_match_forward_and_expert_gradients,
	engine,
	{
		use oa::ml::Module as _;

		let moe = oa::ml::nn::Moe::with_seed(&engine, 4, 3, 3, 2, 1.0e-5, 731)?;
		assert!(moe.sparse_execution());
		let optimizer = oa::ml::Sgd::new(moe.all_parameters()?, 0.0, 0.0, 0.0)?;
		let input = oa::Matrix::from_f32(
			&engine,
			[7, 4],
			&(0..28)
				.map(|index| 0.07 * ((index % 11) as f32 - 5.0))
				.collect::<Vec<_>>(),
		)?;
		let target = oa::Matrix::from_f32(
			&engine,
			[7, 4],
			&(0..28)
				.map(|index| 0.09 * ((index % 13) as f32 - 6.0))
				.collect::<Vec<_>>(),
		)?;

		moe.set_sparse_execution(false);
		assert!(!moe.sparse_execution());
		let dense_tape = oa::ml::GradientTape::new();
		let dense_output = moe.forward(&input)?;
		let dense_loss = oa::ml::loss::mse(&dense_output, &target)?;
		dense_tape.backward(&dense_loss)?;
		let dense_values = dense_output.read_f32()?;
		let dense_parameters = moe.all_named_parameters()?;
		let mut dense_expert_gradients = Vec::new();
		for parameter in &dense_parameters {
			if parameter.path().starts_with("expert_") {
				dense_expert_gradients.push((
					parameter.path().to_owned(),
					parameter
						.parameter()
						.gradient()
						.expect("dense oracle omitted an expert gradient")
						.read_f32()?,
				));
			}
		}
		assert_eq!(dense_expert_gradients.len(), 4);

		optimizer.zero_grad();
		moe.set_sparse_execution(true);
		let sparse_tape = oa::ml::GradientTape::new();
		let sparse_output = moe.forward(&input)?;
		let sparse_loss = oa::ml::loss::mse(&sparse_output, &target)?;
		sparse_tape.backward(&sparse_loss)?;
		assert_close(&sparse_output.read_f32()?, &dense_values, 3.0e-5);
		let sparse_parameters = moe.all_named_parameters()?;
		for (path, dense_gradient) in dense_expert_gradients {
			let sparse_gradient = sparse_parameters
				.iter()
				.find(|parameter| parameter.path() == path)
				.expect("sparse path omitted an expert parameter")
				.parameter()
				.gradient()
				.expect("sparse path omitted an expert gradient")
				.read_f32()?;
			assert_close(&sparse_gradient, &dense_gradient, 1.0e-4);
		}
		Ok(())
	}
);

test_vk!(
	mat_mul_nt_and_reciprocal_reverse_match_closed_form,
	engine,
	{
		let left_values = [1.0_f32, 2.0, 3.0, 4.0];
		let right_values = [2.0_f32, 1.0, 1.0, 3.0];
		let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let left =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [2, 2], &left_values)?)?;
		let right =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [2, 2], &right_values)?)?;
		let tape = oa::ml::GradientTape::new();
		let left_matrix = left.forward(&rows)?;
		let right_matrix = right.forward(&rows)?;
		let product = oa::matrix::mat_mul_nt(&left_matrix, &right_matrix)?;
		let reciprocal = oa::matrix::reciprocal(&product)?;
		let loss = oa::matrix::sum(&oa::matrix::sum(&reciprocal, 0)?, 1)?.reshape([])?;
		tape.backward(&loss)?;

		let product_values = product.read_f32()?;
		let product_gradient = product_values
			.iter()
			.map(|value| -1.0 / (value * value))
			.collect::<Vec<_>>();
		let mut expected_left = vec![0.0_f32; 4];
		let mut expected_right = vec![0.0_f32; 4];
		for row in 0..2 {
			for column in 0..2 {
				let gradient = product_gradient[row * 2 + column];
				for inner in 0..2 {
					expected_left[row * 2 + inner] += gradient * right_values[column * 2 + inner];
					expected_right[column * 2 + inner] += gradient * left_values[row * 2 + inner];
				}
			}
		}
		assert_close(
			&left
				.weight()
				.gradient()
				.expect("left MatMulNt gradient is missing")
				.read_f32()?,
			&expected_left,
			1.0e-6,
		);
		assert_close(
			&right
				.weight()
				.gradient()
				.expect("right MatMulNt gradient is missing")
				.read_f32()?,
			&expected_right,
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(
	moe_configuration_clamps_routes_and_rejects_invalid_layouts,
	engine,
	{
		let low = oa::ml::nn::Moe::with_seed(&engine, 4, 3, 3, 0, 1.0e-5, 7)?;
		let high = oa::ml::nn::Moe::with_seed(&engine, 4, 3, 3, 99, 1.0e-5, 7)?;
		assert_eq!(low.experts_per_token(), 1);
		assert_eq!(high.experts_per_token(), 3);
		assert_eq!(
			oa::ml::nn::Moe::with_seed(&engine, 0, 3, 3, 1, 1.0e-5, 7)
				.err()
				.expect("zero-width MoE was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(moe_route_stats_reduce_only_expert_vectors, engine, {
	let moe = oa::ml::nn::Moe::from_matrices(
		oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?,
		oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
		oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?,
		oa::Matrix::from_f32(&engine, [4, 4, 2], &[0.0; 32])?,
		oa::Matrix::from_f32(&engine, [4, 4], &[0.0; 16])?,
		oa::Matrix::from_f32(&engine, [4, 2, 2], &[0.0; 16])?,
		oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
		2,
		1.0e-5,
	)?;
	let empty = moe.route_stats()?;
	assert_eq!(empty.load_fraction(), &[0.0; 4]);
	assert_eq!(empty.mean_probability(), &[0.0; 4]);
	assert_eq!(empty.dead_experts(), 4);
	assert_eq!(empty.entropy(), 0.0);
	assert_eq!(empty.max_load_ratio(), 0.0);

	let input = oa::Matrix::from_f32(&engine, [3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 2.0])?;
	let _output = moe.forward(&input)?;
	let stats = moe.route_stats()?;
	assert_close(stats.load_fraction(), &[0.5, 0.5, 0.0, 0.0], 1.0e-7);
	assert_close(stats.mean_probability(), &[0.25; 4], 1.0e-7);
	assert!((stats.entropy() - 0.5).abs() <= 1.0e-6);
	assert!((stats.max_load_ratio() - 2.0).abs() <= 1.0e-6);
	assert_eq!(stats.dead_experts(), 2);
	Ok(())
});

test_vk!(
	moe_routing_bias_is_persistent_and_changes_selection_only,
	engine,
	{
		use oa::ml::Module as _;

		let moe = oa::ml::nn::Moe::from_matrices(
			oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?,
			oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
			oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?,
			oa::Matrix::from_f32(&engine, [4, 4, 2], &[0.0; 32])?,
			oa::Matrix::from_f32(&engine, [4, 4], &[0.0; 16])?,
			oa::Matrix::from_f32(&engine, [4, 2, 2], &[0.0; 16])?,
			oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
			2,
			1.0e-5,
		)?;
		let buffers = moe.all_named_buffers()?;
		let routing_bias = buffers
			.iter()
			.find(|buffer| buffer.path() == "routing_bias")
			.expect("persistent routing-bias buffer is missing");
		assert!(routing_bias.persistent());
		assert_eq!(routing_bias.data().shape(), [1, 4]);
		assert_eq!(routing_bias.data().read_f32()?, vec![0.0; 4]);
		assert_eq!(moe.balance_rate(), 0.0);

		moe.set_balance_rate(0.1);
		let input = oa::Matrix::from_f32(&engine, [3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 2.0])?;
		let _first = moe.forward(&input)?;
		assert_eq!(
			moe
				.last_selection_mask()
				.expect("first selection mask is missing")
				.read_f32()?,
			vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
		);
		moe.update_routing_bias()?;
		assert_close(
			&moe.routing_bias().read_f32()?,
			&[-0.1, -0.1, 0.1, 0.1],
			1.0e-7,
		);

		let _second = moe.forward(&input)?;
		assert_eq!(
			moe
				.last_selection_mask()
				.expect("second selection mask is missing")
				.read_f32()?,
			vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
		);
		assert_close(
			&moe
				.last_gate_probabilities()
				.expect("unbiased gate probabilities are missing")
				.read_f32()?,
			&[0.25; 12],
			1.0e-7,
		);
		Ok(())
	}
);

test_vk!(broadcast_add_adjoint_reduces_to_each_input_shape, engine, {
	let left_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[2, 3],
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
	)?)?;
	let right_parameter = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[1, 3],
		&[10.0, 20.0, 30.0],
	)?)?;
	let left_rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
	let right_rows = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
	let tape = oa::ml::GradientTape::new();
	let left = left_parameter.forward(&left_rows)?;
	let right = right_parameter.forward(&right_rows)?;
	let output = oa::matrix::add(&left, &right)?;
	let target = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
	let loss = oa::ml::loss::mse(&output, &target)?;
	tape.backward(&loss)?;

	let expected_left = output
		.read_f32()?
		.into_iter()
		.map(|value| value / 3.0)
		.collect::<Vec<_>>();
	let expected_right = vec![
		expected_left[0] + expected_left[3],
		expected_left[1] + expected_left[4],
		expected_left[2] + expected_left[5],
	];
	assert_close(
		&left_parameter
			.weight()
			.gradient()
			.expect("left broadcast gradient is missing")
			.read_f32()?,
		&expected_left,
		3.0e-6,
	);
	assert_close(
		&right_parameter
			.weight()
			.gradient()
			.expect("right broadcast gradient is missing")
			.read_f32()?,
		&expected_right,
		3.0e-6,
	);
	Ok(())
});

test_vk!(
	moe_balancing_losses_are_normalized_differentiable_and_reset,
	engine,
	{
		use oa::ml::Module as _;

		let moe = oa::ml::nn::Moe::from_matrices(
			oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?,
			oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
			oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?,
			oa::Matrix::from_f32(&engine, [4, 4, 2], &[0.0; 32])?,
			oa::Matrix::from_f32(&engine, [4, 4], &[0.0; 16])?,
			oa::Matrix::from_f32(&engine, [4, 2, 2], &[0.0; 16])?,
			oa::Matrix::from_f32(&engine, [4, 2], &[0.0; 8])?,
			2,
			1.0e-5,
		)?;
		assert_eq!(moe.aux_loss_alpha(), 0.0);
		assert_eq!(moe.router_z_loss_beta(), 0.0);
		assert!(moe.aux_loss().shape().is_empty());
		assert_eq!(moe.aux_loss().read_f32()?, vec![0.0]);

		moe.set_aux_loss_alpha(0.01);
		moe.set_router_z_loss_beta(0.001);
		let input = oa::Matrix::from_f32(&engine, [3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 2.0])?;
		let tape = oa::ml::GradientTape::new();
		let _output = moe.forward(&input)?;
		let aux_loss = moe.aux_loss();
		tape.backward(&aux_loss)?;
		let log_experts = 4.0_f32.ln();
		assert_close(
			&aux_loss.read_f32()?,
			&[0.01 + 0.001 * log_experts * log_experts],
			1.0e-7,
		);
		let router_bias_gradient = moe
			.all_named_parameters()?
			.into_iter()
			.find(|parameter| parameter.path() == "router.bias")
			.expect("router bias parameter is missing")
			.parameter()
			.gradient()
			.expect("Switch auxiliary loss did not reach the router bias")
			.read_f32()?;
		let z_gradient = 0.001 * log_experts / 2.0;
		assert_close(
			&router_bias_gradient,
			&[
				0.0025 + z_gradient,
				0.0025 + z_gradient,
				-0.0025 + z_gradient,
				-0.0025 + z_gradient,
			],
			1.0e-7,
		);

		moe.set_aux_loss_alpha(f32::NAN);
		moe.set_router_z_loss_beta(-1.0);
		let _output = moe.forward(&input)?;
		assert_eq!(moe.aux_loss_alpha(), 0.0);
		assert_eq!(moe.router_z_loss_beta(), 0.0);
		assert_eq!(moe.aux_loss().read_f32()?, vec![0.0]);
		Ok(())
	}
);

test_vk!(
	moe_shared_expert_is_registered_always_on_and_trainable,
	engine,
	{
		use oa::ml::Module as _;

		let base = oa::ml::nn::Moe::with_seed(&engine, 4, 6, 4, 1, 1.0e-5, 303)?;
		let shared =
			oa::ml::nn::Moe::with_seed_and_shared_experts(&engine, 4, 6, 4, 1, 1.0e-5, 1, 303)?;
		assert_eq!(base.num_shared_experts(), 0);
		assert_eq!(shared.num_shared_experts(), 1);
		let named = shared.all_named_parameters()?;
		for path in [
			"shared_expert_0.gate_weight",
			"shared_expert_0.up_weight",
			"shared_expert_0.down_weight",
			"shared_expert_0.gate_bias",
			"shared_expert_0.up_bias",
			"shared_expert_0.down_bias",
		] {
			assert!(
				named.iter().any(|entry| entry.path() == path),
				"missing shared-expert parameter {path}"
			);
		}
		let input = oa::Matrix::from_f32(
			&engine,
			[4, 4],
			&[
				0.2, -0.4, 0.7, 1.1, -0.3, 0.8, 0.1, -0.9, 1.0, 0.5, -0.2, 0.4, -0.7, 0.3, 0.9, -0.1,
			],
		)?;
		let base_output = base.forward(&input)?;
		let oracle_norm = oa::ml::nn::RmsNorm::new(&engine, 4, 1.0e-5)?;
		let oracle_expert = oa::ml::nn::Swiglu::with_seed(&engine, 4, 6, true, 306)?;
		let oracle_output = oa::matrix::add(
			&base_output,
			&oracle_expert.forward(&oracle_norm.forward(&input)?)?,
		)?;
		let tape = oa::ml::GradientTape::new();
		let shared_output = shared.forward(&input)?;
		let target = oa::Matrix::from_f32(&engine, [4, 4], &[0.0; 16])?;
		let loss = oa::ml::loss::mse(&shared_output, &target)?;
		tape.backward(&loss)?;
		let base_values = base_output.read_f32()?;
		let shared_values = shared_output.read_f32()?;
		assert_close(&shared_values, &oracle_output.read_f32()?, 1.0e-6);
		assert!(
			base_values
				.iter()
				.zip(&shared_values)
				.any(|(base, shared)| (base - shared).abs() > 1.0e-6),
			"always-on shared expert contributed no output"
		);
		let shared_gate_gradient = named
			.iter()
			.find(|entry| entry.path() == "shared_expert_0.gate_weight")
			.expect("shared gate weight is missing")
			.parameter()
			.gradient()
			.expect("shared gate weight gradient is missing")
			.read_f32()?;
		assert!(
			shared_gate_gradient
				.iter()
				.any(|value| value.abs() > 1.0e-6),
			"always-on shared expert received no gradient"
		);
		Ok(())
	}
);
