fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: found {actual}, expected {expected}, tolerance {tolerance}"
		);
	}
}

test_vk!(split_and_merge_heads_match_donor_permutation, runtime, {
	let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
	let input = oa::Matrix::new_from_f32([6, 4], &values)?;
	let (plan, (split, merged)) = runtime.capture(|| {
		let split = oa::ml::matrix::split_heads(&input, 2, 3, 2)?;
		let merged = oa::ml::matrix::merge_heads(&split, 2, 3, 2)?;
		Ok((split, merged))
	})?;
	assert_eq!(split.shape(), [4, 3, 2]);
	assert_eq!(merged.shape(), [6, 4]);
	assert_eq!(plan.semantic_graph().operations().len(), 2);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::FnMatrix::splitHeads"
	);
	assert_eq!(
		plan.semantic_graph().operations()[1].name(),
		"oa::FnMatrix::mergeHeads"
	);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("heads"))
		.expect("head-transform report must be valid JSON");
	assert_eq!(report["nodes"][0]["kernel"], "ml.matrix.split_heads.f32");
	assert_eq!(report["nodes"][1]["kernel"], "ml.matrix.merge_heads.f32");

	let _event = runtime.submit(&plan)?;
	plan.wait()?;
	assert_eq!(
		split.read_f32()?,
		[
			0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 16.0, 17.0, 20.0,
			21.0, 14.0, 15.0, 18.0, 19.0, 22.0, 23.0,
		]
	);
	assert_eq!(merged.read_f32()?, values);
	Ok(())
});

test_vk!(one_head_transform_is_a_differentiable_view, runtime, {
	let values = [0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
	let input = oa::Matrix::new_from_f32([2, 4], &values)?;
	let split = oa::ml::matrix::split_heads(&input, 1, 2, 1)?;
	let merged = oa::ml::matrix::merge_heads(&split, 1, 2, 1)?;
	assert_eq!(split.shape(), [1, 2, 4]);
	assert_eq!(merged.shape(), [2, 4]);
	assert_eq!(merged.read_f32()?, values);
	Ok(())
});

test_vk!(head_transform_reverse_is_the_exact_inverse, runtime, {
	let values = [0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
	for num_heads in [1, 2] {
		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 4], &values)?)?;
		let indices = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let input = embedding.forward(&indices)?;
		let split = oa::ml::matrix::split_heads(&input, 1, 2, num_heads)?;
		let merged = oa::ml::matrix::merge_heads(&split, 1, 2, num_heads)?;
		let target = oa::Matrix::new_from_f32([2, 4], &[0.0; 8])?;
		let loss = oa::ml::loss::mse(&merged, &target)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("head-transform input gradient is missing")
			.read_f32()?;
		let expected = values.map(|value| 2.0 * value / values.len() as f32);
		assert_close(&gradient, &expected, 1.0e-6);
	}
	Ok(())
});

test_vk!(head_transforms_reject_invalid_geometry, runtime, {
	let input = oa::Matrix::new_from_f32([6, 4], &[0.0; 24])?;
	let split = oa::ml::matrix::split_heads(&input, 2, 4, 2)
		.err()
		.expect("incompatible B*S was accepted");
	assert_eq!(split.kind(), oa::ErrorKind::InvalidArgument);
	let split = oa::ml::matrix::split_heads(&input, 2, 3, 3)
		.err()
		.expect("indivisible head width was accepted");
	assert_eq!(split.kind(), oa::ErrorKind::InvalidArgument);
	let rank_two = oa::Matrix::new_from_f32([6, 4], &[0.0; 24])?;
	let merge = oa::ml::matrix::merge_heads(&rank_two, 2, 3, 2)
		.err()
		.expect("rank-two merge input was accepted");
	assert_eq!(merge.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
});

fn host_softmax_scaled_masked(
	scores: &[f32],
	mask: &[f32],
	rows: usize,
	columns: usize,
	scale: f32,
) -> Vec<f32> {
	let mut output = vec![0.0; scores.len()];
	for row in 0..rows {
		let base = row * columns;
		let maximum = (0..columns)
			.map(|column| scores[base + column] * scale + mask[base + column])
			.fold(f32::NEG_INFINITY, f32::max);
		let sum = (0..columns)
			.map(|column| (scores[base + column] * scale + mask[base + column] - maximum).exp())
			.sum::<f32>();
		for column in 0..columns {
			output[base + column] =
				(scores[base + column] * scale + mask[base + column] - maximum).exp() / sum;
		}
	}
	output
}

test_vk!(
	scaled_masked_softmax_matches_both_donor_schedules,
	runtime,
	{
		for (rows, columns, expected_kernel) in [
			(3, 13, "ml.softmax_scaled_masked_n32.f32"),
			(2, 37, "ml.matrix.softmax_scaled_masked.f32"),
		] {
			let count = rows * columns;
			let scores = (0..count)
				.map(|index| ((index * 7 % 23) as f32 - 11.0) / 9.0)
				.collect::<Vec<_>>();
			let mask = (0..count)
				.map(|index| {
					if index % columns > columns / 2 {
						-1.5
					} else {
						0.0
					}
				})
				.collect::<Vec<_>>();
			let expected = host_softmax_scaled_masked(&scores, &mask, rows, columns, 0.7);
			let scores = oa::Matrix::new_from_f32([rows, columns], &scores)?;
			let mask = oa::Matrix::new_from_f32([rows, columns], &mask)?;
			let (plan, output) =
				runtime.capture(|| oa::ml::matrix::softmax_scaled_masked(&scores, &mask, 0.7))?;
			let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("softmax"))
				.expect("scaled-masked Softmax report must be valid JSON");
			assert_eq!(report["nodes"][0]["kernel"], expected_kernel);
			let _event = runtime.submit(&plan)?;
			plan.wait()?;
			assert_close(&output.read_f32()?, &expected, 2.0e-5);
		}
		Ok(())
	}
);

test_vk!(
	scaled_masked_softmax_reverse_matches_host_and_detaches_mask,
	runtime,
	{
		let scores = [0.2_f32, -0.3, 0.7, 0.1, -0.6, 0.5, 0.4, -0.2];
		let mask = [0.0_f32, 0.0, -1.0, -1.0, 0.0, -0.5, 0.0, -0.5];
		let target = [0.1_f32, 0.3, 0.2, 0.4, 0.25, 0.25, 0.25, 0.25];
		let scale = 0.65_f32;
		let score_parameter =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 4], &scores)?)?;
		let mask_parameter =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 4], &mask)?)?;
		let indices = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let score_values = score_parameter.forward(&indices)?;
		let mask_values = mask_parameter.forward(&indices)?;
		let probability = oa::ml::matrix::softmax_scaled_masked(&score_values, &mask_values, scale)?;
		let target_matrix = oa::Matrix::new_from_f32([2, 4], &target)?;
		let loss = oa::ml::loss::mse(&probability, &target_matrix)?;
		tape.backward(&loss)?;

		let probability = host_softmax_scaled_masked(&scores, &mask, 2, 4, scale);
		let output_gradient = probability
			.iter()
			.zip(target)
			.map(|(value, target)| 2.0 * (value - target) / probability.len() as f32)
			.collect::<Vec<_>>();
		let mut expected = vec![0.0; scores.len()];
		for row in 0..2 {
			let base = row * 4;
			let dot = (0..4)
				.map(|column| output_gradient[base + column] * probability[base + column])
				.sum::<f32>();
			for column in 0..4 {
				expected[base + column] =
					probability[base + column] * (output_gradient[base + column] - dot) * scale;
			}
		}
		let actual = score_parameter
			.weight()
			.gradient()
			.expect("scaled-masked Softmax score gradient is missing")
			.read_f32()?;
		assert_close(&actual, &expected, 3.0e-6);
		assert!(mask_parameter.weight().gradient().is_none());
		Ok(())
	}
);

#[derive(Clone, Copy)]
struct HostSdpa {
	batch_heads: usize,
	sequence_length: usize,
	head_dim: usize,
	scale: f32,
	causal: bool,
}

fn host_sdpa(
	query: &[f32],
	key: &[f32],
	value: &[f32],
	mask: Option<&[f32]>,
	config: HostSdpa,
) -> Vec<f32> {
	let HostSdpa {
		batch_heads,
		sequence_length,
		head_dim,
		scale,
		causal,
	} = config;
	let mut output = vec![0.0; query.len()];
	for batch_head in 0..batch_heads {
		for query_row in 0..sequence_length {
			let score_base = (batch_head * sequence_length + query_row) * sequence_length;
			let mut scores = vec![0.0; sequence_length];
			for key_row in 0..sequence_length {
				let mut score = 0.0;
				for feature in 0..head_dim {
					score += query[(batch_head * sequence_length + query_row) * head_dim + feature]
						* key[(batch_head * sequence_length + key_row) * head_dim + feature];
				}
				score *= scale;
				if let Some(mask) = mask {
					score += mask[score_base + key_row];
				}
				if causal && key_row > query_row {
					score = f32::NEG_INFINITY;
				}
				scores[key_row] = score;
			}
			let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
			let sum = scores
				.iter()
				.map(|score| (score - maximum).exp())
				.sum::<f32>();
			for key_row in 0..sequence_length {
				let probability = (scores[key_row] - maximum).exp() / sum;
				for feature in 0..head_dim {
					output[(batch_head * sequence_length + query_row) * head_dim + feature] +=
						probability * value[(batch_head * sequence_length + key_row) * head_dim + feature];
				}
			}
		}
	}
	output
}

test_vk!(
	standard_sdpa_matches_causal_and_additive_mask_oracles,
	runtime,
	{
		let query = [
			0.2_f32, -0.1, 0.4, 0.6, 0.3, -0.2, -0.5, 0.7, 0.1, 0.8, -0.4, 0.2,
		];
		let key = [
			0.3_f32, 0.2, -0.2, 0.1, -0.6, 0.5, 0.7, -0.3, 0.4, -0.1, 0.8, 0.2,
		];
		let value = [
			0.5_f32, -0.2, 0.1, 0.4, 0.7, -0.5, -0.3, 0.8, 0.6, 0.2, -0.1, 0.9,
		];
		let mask_values = [
			0.0_f32, -0.2, -1.0, 0.0, 0.0, -0.4, -0.8, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, -0.6, -0.1, 0.0,
			0.0, -0.5,
		];
		let query_matrix = oa::Matrix::new_from_f32([2, 3, 2], &query)?;
		let key_matrix = oa::Matrix::new_from_f32([2, 3, 2], &key)?;
		let value_matrix = oa::Matrix::new_from_f32([2, 3, 2], &value)?;
		let mask_matrix = oa::Matrix::new_from_f32([6, 3], &mask_values)?;
		for (additive_mask, causal) in [(None, true), (Some(&mask_matrix), false)] {
			let expected = host_sdpa(
				&query,
				&key,
				&value,
				additive_mask.map(|_| mask_values.as_slice()),
				HostSdpa {
					batch_heads: 2,
					sequence_length: 3,
					head_dim: 2,
					scale: 0.75,
					causal,
				},
			);
			let (plan, output) = runtime.capture(|| {
				oa::ml::matrix::scaled_dot_product_attention(
					&query_matrix,
					&key_matrix,
					&value_matrix,
					additive_mask,
					0.75,
					causal,
				)
			})?;
			assert_eq!(plan.semantic_graph().operations().len(), 1);
			let operation = &plan.semantic_graph().operations()[0];
			assert_eq!(operation.name(), "oa::FnMatrix::scaledDotProductAttention");
			assert_eq!(operation.inputs()[3].is_some(), additive_mask.is_some());
			let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("sdpa"))
				.expect("SDPA report must be valid JSON");
			let kernels = report["nodes"]
				.as_array()
				.expect("SDPA nodes must be an array")
				.iter()
				.map(|node| node["kernel"].as_str().expect("kernel must be text"))
				.collect::<Vec<_>>();
			assert_eq!(
				kernels,
				[
					"ml.matrix.bmm_nt.f32",
					"ml.sdpa_softmax_n32.f32",
					"ml.matrix.bmm.f32",
				]
			);
			let _event = runtime.submit(&plan)?;
			plan.wait()?;
			assert_close(&output.read_f32()?, &expected, 4.0e-5);
		}
		Ok(())
	}
);

fn host_mse(values: &[f32], target: &[f32]) -> f32 {
	values
		.iter()
		.zip(target)
		.map(|(value, target)| (value - target).powi(2))
		.sum::<f32>()
		/ values.len() as f32
}

fn numerical_gradient(mut values: Vec<f32>, loss: impl Fn(&[f32]) -> f32) -> Vec<f32> {
	const EPSILON: f32 = 1.0e-3;
	(0..values.len())
		.map(|index| {
			values[index] += EPSILON;
			let above = loss(&values);
			values[index] -= 2.0 * EPSILON;
			let below = loss(&values);
			values[index] += EPSILON;
			(above - below) / (2.0 * EPSILON)
		})
		.collect()
}

test_vk!(standard_sdpa_reverse_matches_finite_differences, runtime, {
	let query = vec![0.2_f32, -0.1, 0.4, 0.6];
	let key = vec![0.3_f32, 0.2, -0.2, 0.1];
	let value = vec![0.5_f32, -0.2, 0.1, 0.4];
	let mask = vec![0.0_f32, -0.3, 0.0, 0.0];
	let target = [0.15_f32, -0.05, 0.2, 0.3];
	let scale = 0.7_f32;
	let config = HostSdpa {
		batch_heads: 1,
		sequence_length: 2,
		head_dim: 2,
		scale,
		causal: true,
	};
	let q_expected = numerical_gradient(query.clone(), |candidate| {
		host_mse(
			&host_sdpa(candidate, &key, &value, Some(&mask), config),
			&target,
		)
	});
	let k_expected = numerical_gradient(key.clone(), |candidate| {
		host_mse(
			&host_sdpa(&query, candidate, &value, Some(&mask), config),
			&target,
		)
	});
	let v_expected = numerical_gradient(value.clone(), |candidate| {
		host_mse(
			&host_sdpa(&query, &key, candidate, Some(&mask), config),
			&target,
		)
	});

	let parameter = |values: &[f32]| {
		oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 2], values)?)
	};
	let query_parameter = parameter(&query)?;
	let key_parameter = parameter(&key)?;
	let value_parameter = parameter(&value)?;
	let mask_parameter = parameter(&mask)?;
	let indices = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
	let tape = oa::ml::GradientTape::new();
	let query_matrix = query_parameter.forward(&indices)?.reshape([1, 2, 2])?;
	let key_matrix = key_parameter.forward(&indices)?.reshape([1, 2, 2])?;
	let value_matrix = value_parameter.forward(&indices)?.reshape([1, 2, 2])?;
	let mask_matrix = mask_parameter.forward(&indices)?;
	let output = oa::ml::matrix::scaled_dot_product_attention(
		&query_matrix,
		&key_matrix,
		&value_matrix,
		Some(&mask_matrix),
		scale,
		true,
	)?;
	let loss = oa::ml::loss::mse(&output, &oa::Matrix::new_from_f32([1, 2, 2], &target)?)?;
	tape.backward(&loss)?;
	for (parameter, expected) in [
		(&query_parameter, q_expected),
		(&key_parameter, k_expected),
		(&value_parameter, v_expected),
	] {
		let actual = parameter
			.weight()
			.gradient()
			.expect("SDPA operand gradient is missing")
			.read_f32()?;
		assert_close(&actual, &expected, 8.0e-4);
	}
	assert!(mask_parameter.weight().gradient().is_none());
	Ok(())
});

test_vk!(flash_sdpa_matches_the_standard_causal_route, runtime, {
	let batch_heads = 2;
	let sequence_length = 5;
	let head_dim = 4;
	let count = batch_heads * sequence_length * head_dim;
	let query = (0..count)
		.map(|index| ((index * 7 % 29) as f32 - 14.0) / 13.0)
		.collect::<Vec<_>>();
	let key = (0..count)
		.map(|index| ((index * 11 % 31) as f32 - 15.0) / 14.0)
		.collect::<Vec<_>>();
	let value = (0..count)
		.map(|index| ((index * 5 % 23) as f32 - 11.0) / 9.0)
		.collect::<Vec<_>>();
	let scale = 1.0 / (head_dim as f32).sqrt();
	let expected = host_sdpa(
		&query,
		&key,
		&value,
		None,
		HostSdpa {
			batch_heads,
			sequence_length,
			head_dim,
			scale,
			causal: true,
		},
	);
	let query = oa::Matrix::new_from_f32([batch_heads, sequence_length, head_dim], &query)?;
	let key = oa::Matrix::new_from_f32([batch_heads, sequence_length, head_dim], &key)?;
	let value = oa::Matrix::new_from_f32([batch_heads, sequence_length, head_dim], &value)?;
	let (plan, output) =
		runtime.capture(|| oa::ml::matrix::flash_attention_causal(&query, &key, &value, scale))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	let operation = &plan.semantic_graph().operations()[0];
	assert_eq!(operation.name(), "oa::FnMatrix::scaledDotProductAttention");
	assert!(operation.inputs()[3].is_none());
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("flash_sdpa"))
		.expect("Flash SDPA report must be valid JSON");
	assert_eq!(
		report["nodes"][0]["kernel"],
		"ml.flash_attention_causal.f32"
	);
	let _event = runtime.submit(&plan)?;
	plan.wait()?;
	assert_close(&output.read_f32()?, &expected, 2.0e-5);
	Ok(())
});

test_vk!(flash_sdpa_reverse_matches_finite_differences, runtime, {
	let query = vec![0.2_f32, -0.1, 0.4, 0.6];
	let key = vec![0.3_f32, 0.2, -0.2, 0.1];
	let value = vec![0.5_f32, -0.2, 0.1, 0.4];
	let target = [0.15_f32, -0.05, 0.2, 0.3];
	let scale = 0.7_f32;
	let config = HostSdpa {
		batch_heads: 1,
		sequence_length: 2,
		head_dim: 2,
		scale,
		causal: true,
	};
	let expected = [
		numerical_gradient(query.clone(), |candidate| {
			host_mse(&host_sdpa(candidate, &key, &value, None, config), &target)
		}),
		numerical_gradient(key.clone(), |candidate| {
			host_mse(&host_sdpa(&query, candidate, &value, None, config), &target)
		}),
		numerical_gradient(value.clone(), |candidate| {
			host_mse(&host_sdpa(&query, &key, candidate, None, config), &target)
		}),
	];
	let parameter = |values: &[f32]| {
		oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 2], values)?)
	};
	let query_parameter = parameter(&query)?;
	let key_parameter = parameter(&key)?;
	let value_parameter = parameter(&value)?;
	let indices = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
	let target = oa::Matrix::new_from_f32([1, 2, 2], &target)?;
	let (plan, _) = runtime.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let query = query_parameter.forward(&indices)?.reshape([1, 2, 2])?;
		let key = key_parameter.forward(&indices)?.reshape([1, 2, 2])?;
		let value = value_parameter.forward(&indices)?.reshape([1, 2, 2])?;
		let output = oa::ml::matrix::flash_attention_causal(&query, &key, &value, scale)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("flash_reverse"))
		.expect("Flash reverse report must be valid JSON");
	let kernels = report["nodes"]
		.as_array()
		.expect("Flash reverse nodes must be an array")
		.iter()
		.filter_map(|node| node["kernel"].as_str())
		.collect::<Vec<_>>();
	assert!(kernels.contains(&"ml.matrix.scaled_dot_product_attention_backward.f32"));
	assert!(kernels.contains(&"ml.flash_attention_causal_backward_kv.f32"));
	let _event = runtime.submit(&plan)?;
	plan.wait()?;
	for (parameter, expected) in [
		(&query_parameter, &expected[0]),
		(&key_parameter, &expected[1]),
		(&value_parameter, &expected[2]),
	] {
		let actual = parameter
			.weight()
			.gradient()
			.expect("Flash SDPA operand gradient is missing")
			.read_f32()?;
		assert_close(&actual, expected, 8.0e-4);
	}
	Ok(())
});

test_vk!(flash_sdpa_rejects_the_unverified_long_sequence, runtime, {
	let values = vec![0.0_f32; 1025];
	let input = oa::Matrix::new_from_f32([1, 1025, 1], &values)?;
	let error = oa::ml::matrix::flash_attention_causal(&input, &input, &input, 1.0)
		.err()
		.expect("Flash SDPA accepted a sequence longer than 1024");
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(multi_head_attention_backend_policy_is_explicit, runtime, {
	use oa::ml::nn::{AttentionBackend, MultiHeadAttention};

	let input_values = (0..64)
		.map(|index| ((index * 7 % 31) as f32 - 15.0) / 17.0)
		.collect::<Vec<_>>();
	let input = oa::Matrix::new_from_f32([8, 8], &input_values)?;
	let attention = MultiHeadAttention::new_with_seed(8, 2, 4, 0x4154_544e)?;
	assert_eq!(attention.backend(), AttentionBackend::Auto);
	let standard = attention.forward(&input)?.read_f32()?;
	assert_eq!(attention.last_backend(), AttentionBackend::Standard);

	attention.set_backend(AttentionBackend::Flash);
	let flash = attention.forward(&input)?.read_f32()?;
	assert_eq!(attention.last_backend(), AttentionBackend::Flash);
	assert_close(&flash, &standard, 5.0e-5);

	attention.set_backend(AttentionBackend::Standard);
	let explicit_standard = attention.forward(&input)?.read_f32()?;
	assert_eq!(attention.last_backend(), AttentionBackend::Standard);
	assert_close(&explicit_standard, &standard, 1.0e-6);
	Ok(())
});

test_vk!(
	multi_head_attention_visibility_and_masks_are_explicit,
	runtime,
	{
		use oa::ml::nn::{AttentionBackend, AttentionMode, MultiHeadAttention};

		let input_values = (0..16)
			.map(|index| ((index * 5 % 17) as f32 - 8.0) / 9.0)
			.collect::<Vec<_>>();
		let input = oa::Matrix::new_from_f32([4, 4], &input_values)?;
		let attention = MultiHeadAttention::new_with_seed(4, 2, 4, 0x4d41_534b)?;
		assert_eq!(attention.mode(), AttentionMode::Causal);
		let causal = attention.forward(&input)?.read_f32()?;

		attention.set_mode(AttentionMode::Bidirectional);
		assert_eq!(attention.mode(), AttentionMode::Bidirectional);
		let bidirectional = attention.forward(&input)?.read_f32()?;
		assert!(
			causal
				.iter()
				.zip(&bidirectional)
				.any(|(left, right)| (left - right).abs() > 1.0e-6)
		);

		let zero_mask = oa::Matrix::new_from_f32([8, 4], &[0.0; 32])?;
		let zero_masked = attention.forward_masked(&input, &zero_mask)?.read_f32()?;
		assert_eq!(attention.last_backend(), AttentionBackend::Standard);
		assert_close(&zero_masked, &bidirectional, 1.0e-6);

		let mask_values = (0..32)
			.map(|index| if index % 4 == 3 { -1.0e4 } else { 0.0 })
			.collect::<Vec<_>>();
		let mask = oa::Matrix::new_from_f32([8, 4], &mask_values)?;
		let masked = attention.forward_masked(&input, &mask)?.read_f32()?;
		assert!(
			masked
				.iter()
				.zip(&bidirectional)
				.any(|(left, right)| (left - right).abs() > 1.0e-6)
		);

		attention.set_backend(AttentionBackend::Flash);
		assert_eq!(
			attention
				.forward(&input)
				.err()
				.expect("bidirectional Flash request was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert_eq!(
			attention
				.forward_masked(&input, &mask)
				.err()
				.expect("masked Flash request was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(
	transformer_block_propagates_attention_visibility,
	runtime,
	{
		use oa::ml::nn::{AttentionMode, TransformerBlock};

		let block = TransformerBlock::new_with_seed(8, 16, 3, 2, 1.0e-5, 0x424c_4f43)?;
		assert_eq!(block.attention_mode(), AttentionMode::Causal);
		block.set_attention_mode(AttentionMode::Bidirectional);
		assert_eq!(block.attention_mode(), AttentionMode::Bidirectional);
		let input = oa::Matrix::new_from_f32([6, 8], &[0.25; 48])?;
		assert_eq!(block.forward(&input)?.shape(), [6, 8]);
		block.set_sequence_length(2)?;
		assert_eq!(block.sequence_length(), 2);
		let shorter = oa::Matrix::new_from_f32([4, 8], &[0.25; 32])?;
		let unmasked = block.forward(&shorter)?.read_f32()?;
		let zero_mask = oa::Matrix::new_from_f32([8, 2], &[0.0; 16])?;
		let masked = block.forward_masked(&shorter, &zero_mask)?.read_f32()?;
		assert_close(&masked, &unmasked, 1.0e-6);
		assert_eq!(
			block
				.set_sequence_length(0)
				.expect_err("zero sequence length was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert_eq!(block.sequence_length(), 2);
		Ok(())
	}
);

test_vk!(
	bias_free_attention_registers_only_projection_weights,
	runtime,
	{
		use oa::ml::{Module, nn::MultiHeadAttention};

		let attention = MultiHeadAttention::new_with_seed_and_bias(4, 2, 3, false, 0x4e4f_4249)?;
		assert!(!attention.has_bias());
		let named = attention.all_named_parameters()?;
		assert_eq!(named.len(), 4);
		assert_eq!(named[0].path(), "q_proj.weight");
		assert_eq!(named[1].path(), "k_proj.weight");
		assert_eq!(named[2].path(), "v_proj.weight");
		assert_eq!(named[3].path(), "out_proj.weight");
		let input = oa::Matrix::new_from_f32([3, 4], &[0.2; 12])?;
		let tape = oa::ml::GradientTape::new();
		let output = attention.forward(&input)?;
		let target = oa::Matrix::new_from_f32([3, 4], &[0.0; 12])?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		for parameter in attention.all_parameters()? {
			assert!(parameter.gradient().is_some());
		}
		Ok(())
	}
);

test_vk!(
	attention_dropout_obeys_train_eval_and_reverse_contracts,
	runtime,
	{
		use oa::ml::{
			Module,
			nn::{AttentionBackend, MultiHeadAttention},
		};

		let attention =
			MultiHeadAttention::new_with_seed_and_options(4, 2, 3, 0.25, false, 0x4452_4f50)?;
		assert_eq!(attention.dropout_probability(), 0.25);
		let input = oa::Matrix::new_from_f32(
			[3, 4],
			&[
				0.2, -0.1, 0.4, 0.3, -0.3, 0.5, 0.1, -0.2, 0.6, 0.2, -0.4, 0.7,
			],
		)?;
		let target = oa::Matrix::new_from_f32([3, 4], &[0.0; 12])?;
		let (training_plan, _) = runtime.capture(|| {
			let tape = oa::ml::GradientTape::new();
			let output = attention.forward(&input)?;
			let loss = oa::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let training_report: serde_json::Value =
			serde_json::from_str(&training_plan.debug_report_json("attention_dropout"))
				.expect("attention-dropout report must be valid JSON");
		let training_kernels = training_report["nodes"]
			.as_array()
			.expect("attention-dropout nodes must be an array")
			.iter()
			.filter_map(|node| node["kernel"].as_str())
			.collect::<Vec<_>>();
		assert!(training_kernels.contains(&"matrix.dropout_replay.f32"));
		assert!(training_kernels.contains(&"matrix.dropout_backward_replay.f32"));
		runtime.submit(&training_plan)?.wait()?;
		for parameter in attention.all_parameters()? {
			assert!(parameter.gradient().is_some());
		}

		attention.eval();
		let (eval_plan, _) = runtime.capture(|| attention.forward(&input))?;
		let eval_report: serde_json::Value =
			serde_json::from_str(&eval_plan.debug_report_json("attention_eval"))
				.expect("attention-eval report must be valid JSON");
		assert!(
			eval_report["nodes"]
				.as_array()
				.expect("attention-eval nodes must be an array")
				.iter()
				.all(|node| !node["kernel"]
					.as_str()
					.is_some_and(|name| name.contains("dropout")))
		);

		attention.set_backend(AttentionBackend::Flash);
		assert_eq!(
			attention
				.forward(&input)
				.err()
				.expect("Flash accepted a module configured with dropout")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(
	multi_head_attention_step_matches_the_causal_scan,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		let attention = MultiHeadAttention::new_with_seed(8, 2, 4, 0x4b56_4348)?;
		let host = (0..64)
			.map(|index| 0.35 * (0.37 * index as f32 + 0.11).sin())
			.collect::<Vec<f32>>();
		// Two packed length-4 sequences: batch 0 occupies rows 0..4, batch 1 rows 4..8.
		let packed = oa::Matrix::new_from_f32([8, 8], &host)?;
		let scan = attention.forward(&packed)?.read_f32()?;

		let mut cache = attention.new_cache(2)?;
		assert_eq!(cache.batch_size(), 2);
		assert!(cache.is_empty());
		for time in 0..4 {
			let step_input = oa::Matrix::new_from_f32(
				[2, 8],
				&[
					&host[time * 8..time * 8 + 8],
					&host[32 + time * 8..32 + time * 8 + 8],
				]
				.concat(),
			)?;
			let stepped = attention.step(&step_input, &mut cache)?.read_f32()?;
			assert_eq!(cache.len(), time + 1);
			let mut expected = Vec::with_capacity(16);
			expected.extend_from_slice(&scan[time * 8..time * 8 + 8]);
			expected.extend_from_slice(&scan[32 + time * 8..32 + time * 8 + 8]);
			assert_close(&stepped, &expected, 2.0e-4);
		}
		Ok(())
	}
);

test_vk!(
	multi_head_attention_cache_enforces_ownership_and_contracts,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		let attention = MultiHeadAttention::new_with_seed(8, 2, 4, 0x4b56_4348)?;
		let other = MultiHeadAttention::new_with_seed(8, 2, 4, 0x4f54_4852)?;
		let input = oa::Matrix::new_from_f32([2, 8], &[0.1; 16])?;

		assert_eq!(
			attention
				.new_cache(0)
				.err()
				.expect("attention accepted a zero-batch cache")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let mut foreign = other.new_cache(2)?;
		assert_eq!(
			attention
				.step(&input, &mut foreign)
				.err()
				.expect("attention accepted a foreign cache")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let mut cache = attention.new_cache(2)?;
		let wrong_batch = oa::Matrix::new_from_f32([1, 8], &[0.1; 8])?;
		assert_eq!(
			attention
				.step(&wrong_batch, &mut cache)
				.err()
				.expect("attention accepted a mismatched cache batch")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_dtype = oa::Matrix::new_from_slice([2, 8], &[1_u32; 16])?;
		assert_eq!(
			attention
				.step(&wrong_dtype, &mut cache)
				.err()
				.expect("attention accepted a non-F32 step input")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let tape = oa::ml::GradientTape::new();
		assert_eq!(
			attention
				.step(&input, &mut cache)
				.err()
				.expect("attention step ran inside a GradientTape")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		drop(tape);
		Ok(())
	}
);

test_vk!(
	grouped_query_attention_rejects_invalid_head_grouping,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		// Zero, non-dividing, and oversized key/value head counts are invalid.
		for num_kv_heads in [0_usize, 3, 5] {
			assert_eq!(
				MultiHeadAttention::new_with_seed_and_options_grouped_on(
					&runtime,
					8,
					4,
					num_kv_heads,
					4,
					0.0,
					true,
					0x4751_4142,
				)
				.err()
				.expect("grouped attention accepted an invalid key/value head count")
				.kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		Ok(())
	}
);

test_vk!(
	grouped_query_attention_degenerates_to_multi_head_when_counts_match,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		let host = (0..64)
			.map(|index| 0.3 * (0.29 * index as f32 + 0.07).sin())
			.collect::<Vec<f32>>();
		let input = oa::Matrix::new_from_f32([8, 8], &host)?;
		let plain = MultiHeadAttention::new_with_seed_on(&runtime, 8, 4, 4, 0x4d48_4145)?;
		let grouped = MultiHeadAttention::new_with_seed_and_options_grouped_on(
			&runtime,
			8,
			4,
			4,
			4,
			0.0,
			true,
			0x4d48_4145,
		)?;
		assert_eq!(grouped.num_heads(), 4);
		assert_eq!(grouped.num_kv_heads(), 4);
		assert_close(
			&grouped.forward(&input)?.read_f32()?,
			&plain.forward(&input)?.read_f32()?,
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(
	grouped_and_multi_query_attention_own_compact_key_value_projections,
	runtime,
	{
		use oa::ml::{Module, nn::MultiHeadAttention};

		let input = oa::Matrix::new_from_f32([4, 8], &[0.15; 32])?;
		// GQA: four query heads share two key/value heads (K/V width 4).
		let gqa = MultiHeadAttention::new_with_seed_and_options_grouped_on(
			&runtime,
			8,
			4,
			2,
			4,
			0.0,
			true,
			0x4751_413f,
		)?;
		assert_eq!(gqa.num_kv_heads(), 2);
		let gqa_named = gqa.all_named_parameters()?;
		let shape_of = |named: &[oa::ml::NamedParameter], path: &str| {
			named
				.iter()
				.find(|parameter| parameter.path() == path)
				.map(|parameter| parameter.parameter().data().shape().to_vec())
		};
		assert_eq!(
			shape_of(&gqa_named, "k_proj.weight").as_deref(),
			Some(&[4, 8][..])
		);
		assert_eq!(
			shape_of(&gqa_named, "v_proj.weight").as_deref(),
			Some(&[4, 8][..])
		);
		assert_eq!(
			shape_of(&gqa_named, "q_proj.weight").as_deref(),
			Some(&[8, 8][..])
		);
		assert_eq!(gqa.forward(&input)?.shape(), [4, 8]);

		// MQA: one shared key/value head (K/V width 2).
		let mqa = MultiHeadAttention::new_with_seed_and_options_grouped_on(
			&runtime,
			8,
			4,
			1,
			4,
			0.0,
			true,
			0x4d51_413f,
		)?;
		assert_eq!(mqa.num_kv_heads(), 1);
		let mqa_named = mqa.all_named_parameters()?;
		assert_eq!(
			shape_of(&mqa_named, "k_proj.weight").as_deref(),
			Some(&[2, 8][..])
		);
		assert_eq!(
			shape_of(&mqa_named, "v_proj.weight").as_deref(),
			Some(&[2, 8][..])
		);
		assert_eq!(mqa.forward(&input)?.shape(), [4, 8]);
		Ok(())
	}
);

test_vk!(
	grouped_query_attention_cross_attends_unequal_source_lengths,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		let attention = MultiHeadAttention::new_with_seed_and_options_grouped_on(
			&runtime,
			8,
			4,
			2,
			4,
			0.0,
			true,
			0x4352_4f53,
		)?;
		let queries = oa::Matrix::new_from_f32([4, 8], &[0.2; 32])?;
		let keys_values = oa::Matrix::new_from_f32([7, 8], &[0.1; 56])?;
		let output = attention.forward_kv(&queries, &keys_values)?;
		assert_eq!(output.shape(), [4, 8]);
		Ok(())
	}
);

test_vk!(
	grouped_query_attention_step_matches_the_causal_scan,
	runtime,
	{
		use oa::ml::nn::MultiHeadAttention;

		for num_kv_heads in [2_usize, 1] {
			let attention = MultiHeadAttention::new_with_seed_and_options_grouped_on(
				&runtime,
				8,
				4,
				num_kv_heads,
				4,
				0.0,
				true,
				0x4b56_4741,
			)?;
			let host = (0..64)
				.map(|index| 0.35 * (0.37 * index as f32 + 0.11).sin())
				.collect::<Vec<f32>>();
			let packed = oa::Matrix::new_from_f32([8, 8], &host)?;
			let scan = attention.forward(&packed)?.read_f32()?;

			let mut cache = attention.new_cache(2)?;
			for time in 0..4 {
				let step_input = oa::Matrix::new_from_f32(
					[2, 8],
					&[
						&host[time * 8..time * 8 + 8],
						&host[32 + time * 8..32 + time * 8 + 8],
					]
					.concat(),
				)?;
				let stepped = attention.step(&step_input, &mut cache)?.read_f32()?;
				assert_eq!(cache.len(), time + 1);
				let mut expected = Vec::with_capacity(16);
				expected.extend_from_slice(&scan[time * 8..time * 8 + 8]);
				expected.extend_from_slice(&scan[32 + time * 8..32 + time * 8 + 8]);
				assert_close(&stepped, &expected, 2.0e-4);
			}
		}
		Ok(())
	}
);

test_vk!(
	grouped_query_attention_backpropagates_through_head_sharing,
	runtime,
	{
		use oa::ml::{Module, nn::MultiHeadAttention};

		let attention = MultiHeadAttention::new_with_seed_and_options_grouped_on(
			&runtime,
			8,
			4,
			2,
			4,
			0.0,
			true,
			0x4752_4144,
		)?;
		let input = oa::Matrix::new_from_f32([4, 8], &[0.2; 32])?;
		let tape = oa::ml::GradientTape::new();
		let output = attention.forward(&input)?;
		let target = oa::Matrix::new_from_f32([4, 8], &[0.0; 32])?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		let mut checked = 0_usize;
		for named in attention.all_named_parameters()? {
			let gradient = named
				.parameter()
				.gradient()
				.expect("grouped attention parameter missed a gradient");
			assert!(
				gradient.read_f32()?.iter().any(|value| *value != 0.0),
				"{} produced a zero gradient through grouped attention",
				named.path(),
			);
			checked += 1;
		}
		assert_eq!(checked, 8);
		Ok(())
	}
);
