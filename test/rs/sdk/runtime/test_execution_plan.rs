use oa::ml::Module;

fn kernel_histogram(
	plan: &oa::ExecutionPlan,
	name: &str,
) -> (usize, std::collections::BTreeMap<String, usize>) {
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json(name)).expect("plan report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("nodes");
	let mut histogram = std::collections::BTreeMap::new();
	for node in nodes {
		*histogram
			.entry(node["kernel"].as_str().expect("kernel").to_owned())
			.or_default() += 1;
	}
	(nodes.len(), histogram)
}

test_vk!(alm_tokenizer_plan_keeps_its_conv_tail_fusions, engine, {
	// Pins the fused ALM training plan recorded in jit_fusion_census.md
	// (202 nodes vs 248 unfused): every Conv1d lowering tail sits inside a
	// fused variant and only the bias-free ConvTranspose1d backward
	// transposes remain.
	let config = sdk::ml::alm::AlmTokenizerConfig {
		input_dim: 16,
		width: 32,
		code_dim: 16,
		num_codes: 64,
		downsample_stages: 2,
		depth: 2,
		commitment_beta: 0.25,
		ema_decay: 0.99,
		ema_epsilon: 1e-5,
		dead_threshold: 2.0,
	};
	let model = sdk::ml::alm::AlmTokenizer::new(&engine, config)?;
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 32, 16],
		&vec![0.05; 2 * 32 * 16]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_f32(&engine, [2, 32, 16], &vec![0.0; 2 * 32 * 16])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let reconstruction = model.forward(&input)?;
		let loss = oa::ml::loss::mse(&reconstruction, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let (total, census) = kernel_histogram(&plan, "alm");
	assert_eq!(
		total, 202,
		"fused ALM training plan size changed: {census:?}"
	);
	assert!(
		!census.contains_key("ml.conv_1d_bias_add.f32"),
		"no unfused Conv1d bias add may remain"
	);
	assert_eq!(
		census.get("ml.conv_1d_transpose.f32").copied().unwrap_or(0),
		2,
		"only the bias-free ConvTranspose1d backward transposes may remain"
	);
	assert!(
		census.contains_key("ml.conv_1d_bias_transpose_residual_relu.f32"),
		"the residual Conv1d tail must stay fused"
	);
	Ok(())
});

test_vk!(commands_cnn_plan_keeps_its_conv_and_pool_fusions, engine, {
	// Pins the fused commands-CNN training plan recorded in
	// jit_fusion_census.md (23 nodes vs 27 unfused): each Conv2d block
	// applies ReLU at the conv store and folds the ReLU mask into the pool
	// adjoint.
	let model = sdk::ml::audio::commands_cnn(&engine, 40, 62, 8)?;
	let input = oa::Matrix::new_from_f32(
		&engine,
		[4, 1, 40, 62],
		&vec![0.05; 4 * 40 * 62]
			.iter()
			.enumerate()
			.map(|(i, v)| v * ((i * 37 % 23) as f32 - 11.0))
			.collect::<Vec<f32>>(),
	)?;
	let target = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 2, 1, 3])?;
	let (plan, _) = engine.capture(|| {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(())
	})?;
	let (total, census) = kernel_histogram(&plan, "cnn");
	assert_eq!(
		total, 23,
		"fused commands-CNN training plan size changed: {census:?}"
	);
	for (kernel, count) in [
		("ml.conv_2d_relu.f32", 2),
		("ml.max_pool_2d_relu_backward.f32", 2),
		("ml.matrix.conv_2d.f32", 0),
		("ml.matrix.relu_backward.f32", 0),
	] {
		assert_eq!(
			census.get(kernel).copied().unwrap_or(0),
			count,
			"{kernel}: {census:?}"
		);
	}
	Ok(())
});
