use oa::ml::Module;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"value {index}: expected {expected}, found {actual}"
		);
	}
}

fn transformer_config(moe: bool) -> oa::ml::nn::FlowTransformerConfig {
	oa::ml::nn::FlowTransformerConfig {
		model_width: 4,
		hidden_width: if moe { 4 } else { 8 },
		sequence_length: 2,
		num_layers: 1,
		num_heads: 1,
		num_experts: if moe { 2 } else { 0 },
		experts_per_token: if moe { 1 } else { 0 },
		..Default::default()
	}
}

test_vk!(flow_time_embedding_matches_the_donor_cpu_oracle, engine, {
	let embedding = oa::ml::nn::FlowTimeEmbedding::new(&engine, 4, 100.0, 10.0)?;
	assert_eq!(embedding.embedding_dim(), 4);
	assert_eq!(embedding.max_period(), 100.0);
	assert_eq!(embedding.time_scale(), 10.0);
	let buffers = embedding.all_named_buffers()?;
	assert_eq!(buffers.len(), 1);
	assert_eq!(buffers[0].path(), "frequencies");
	assert!(!buffers[0].persistent());
	assert_close(&buffers[0].data().read_f32()?, &[10.0, 1.0], 1.0e-6);

	let times = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.5])?;
	let output = embedding.forward(&times)?;
	let expected = [
		0.0_f32.sin(),
		0.0_f32.sin(),
		0.0_f32.cos(),
		0.0_f32.cos(),
		5.0_f32.sin(),
		0.5_f32.sin(),
		5.0_f32.cos(),
		0.5_f32.cos(),
	];
	assert_eq!(output.shape(), [2, 4]);
	assert_close(&output.read_f32()?, &expected, 2.0e-6);
	Ok(())
});

test_vk!(flow_time_embedding_rejects_invalid_contracts, engine, {
	for (dimension, period, scale) in [
		(0, 100.0, 1.0),
		(3, 100.0, 1.0),
		(4, 1.0, 1.0),
		(4, f32::NAN, 1.0),
		(4, 100.0, 0.0),
	] {
		assert_eq!(
			oa::ml::nn::FlowTimeEmbedding::new(&engine, dimension, period, scale)
				.err()
				.expect("invalid FlowTimeEmbedding configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let embedding = oa::ml::nn::FlowTimeEmbedding::new(&engine, 4, 100.0, 1.0)?;
	let wrong_shape = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
	assert_eq!(
		embedding
			.forward(&wrong_shape)
			.err()
			.expect("invalid time shape was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(flow_objective_and_euler_match_the_donor_oracle, engine, {
	let clean = oa::Matrix::from_f32(&engine, [2, 3], &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0])?;
	let noise = oa::Matrix::from_f32(&engine, [2, 3], &[10.0, 12.0, 14.0, 16.0, 18.0, 20.0])?;
	let time = oa::Matrix::from_f32(&engine, [2], &[0.25, 0.75])?;
	let matched = oa::ml::flow::linear_match(&clean, &noise, &time)?;
	assert_eq!(
		matched.state.read_f32()?,
		vec![2.5, 4.5, 6.5, 13.5, 15.5, 17.5]
	);
	assert_eq!(matched.velocity.read_f32()?, vec![10.0; 6]);
	let reconstructed = oa::ml::flow::euler_step(&matched.state, &matched.velocity, -0.25)?;
	assert_close(&reconstructed.read_f32()?[..3], &[0.0, 2.0, 4.0], 1.0e-6);
	Ok(())
});

test_vk!(
	flow_linear_match_and_euler_reverse_reach_both_endpoints,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [1, 2], &[1.0, 2.0])?;
		let identity = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let clean = oa::ml::nn::Linear::from_matrices(
			identity.clone(),
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
		)?;
		let noise = oa::ml::nn::Linear::from_matrices(
			identity,
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
		)?;
		let time = oa::Matrix::from_f32(&engine, [1], &[0.25])?;
		let tape = oa::ml::GradientTape::new();
		let matched =
			oa::ml::flow::linear_match(&clean.forward(&input)?, &noise.forward(&input)?, &time)?;
		let reconstructed = oa::ml::flow::euler_step(&matched.state, &matched.velocity, -0.25)?;
		let loss = oa::matrix::reshape(&oa::matrix::sum(&reconstructed, -1)?, Vec::new())?;
		tape.backward(&loss)?;
		assert_close(
			&clean
				.weight()
				.gradient()
				.expect("clean endpoint weight gradient is missing")
				.read_f32()?,
			&[1.0, 2.0, 1.0, 2.0],
			1.0e-6,
		);
		assert_close(
			&clean
				.bias()
				.expect("clean endpoint bias is missing")
				.gradient()
				.expect("clean endpoint bias gradient is missing")
				.read_f32()?,
			&[1.0, 1.0],
			1.0e-6,
		);
		assert_close(
			&noise
				.weight()
				.gradient()
				.expect("noise endpoint weight gradient is missing")
				.read_f32()?,
			&[0.0; 4],
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(
	flow_masked_mse_excludes_padding_and_preserves_reverse,
	engine,
	{
		let input_values = [1.0, 3.0, 5.0, 7.0, 100.0, 100.0, 100.0, 100.0];
		let input = oa::Matrix::from_f32(&engine, [1, 2, 4], &input_values)?;
		let prediction = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(
				&engine,
				[4, 4],
				&[
					1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
				],
			)?,
			oa::Matrix::from_f32(&engine, [4], &[0.0; 4])?,
		)?;
		let target = oa::Matrix::from_f32(
			&engine,
			[1, 2, 4],
			&[0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
		)?;
		let mask = oa::Matrix::from_f32(&engine, [1, 2, 1], &[1.0, 0.0])?;
		let tape = oa::ml::GradientTape::new();
		let values = prediction.forward(&input)?;
		let loss = oa::ml::flow::masked_mse(&values, &target, &mask)?;
		assert_close(&loss.read_f32()?, &[7.5], 1.0e-5);
		tape.backward(&loss)?;
		assert_close(
			&prediction
				.bias()
				.expect("prediction bias is missing")
				.gradient()
				.expect("masked MSE bias gradient is missing")
				.read_f32()?,
			&[0.5, 1.0, 1.5, 2.0],
			1.0e-5,
		);

		let zero_mask = oa::Matrix::from_f32(&engine, [1, 2, 1], &[0.0, 0.0])?;
		let zero = oa::ml::flow::masked_mse(&values, &target, &zero_mask)?;
		assert_eq!(zero.read_f32()?, vec![0.0]);
		Ok(())
	}
);

test_vk!(flow_operations_keep_one_semantic_identity_each, engine, {
	let clean = oa::Matrix::from_f32(&engine, [1, 2], &[1.0, 2.0])?;
	let noise = oa::Matrix::from_f32(&engine, [1, 2], &[3.0, 4.0])?;
	let time = oa::Matrix::from_f32(&engine, [1], &[0.5])?;
	let (match_plan, matched) =
		engine.capture(|| oa::ml::flow::linear_match(&clean, &noise, &time))?;
	assert_eq!(match_plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		match_plan.semantic_graph().operations()[0].name(),
		"oa::ml::flow::linear_match"
	);
	engine.submit(&match_plan)?.wait()?;

	let (euler_plan, _) =
		engine.capture(|| oa::ml::flow::euler_step(&matched.state, &matched.velocity, -0.5))?;
	assert_eq!(euler_plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		euler_plan.semantic_graph().operations()[0].name(),
		"oa::ml::flow::euler_step"
	);
	engine.submit(&euler_plan)?.wait()?;

	let mask = oa::Matrix::from_f32(&engine, [1, 1], &[1.0])?;
	let (loss_plan, _) = engine.capture(|| oa::ml::flow::masked_mse(&clean, &noise, &mask))?;
	assert_eq!(loss_plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		loss_plan.semantic_graph().operations()[0].name(),
		"oa::ml::flow::masked_mse"
	);
	engine.submit(&loss_plan)?.wait()?;
	Ok(())
});

test_vk!(
	dense_and_moe_flow_transformers_share_the_donor_contract,
	engine,
	{
		let input = oa::Matrix::from_f32(
			&engine,
			[2, 2, 4],
			&[
				1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
				-8.0,
			],
		)?;
		let condition = oa::Matrix::from_f32(&engine, [2, 4], &[0.5; 8])?;
		for moe in [false, true] {
			let model = oa::ml::nn::FlowTransformer::with_seed(
				&engine,
				transformer_config(moe),
				0x464c_4f57,
			)?;
			assert_eq!(model.is_moe(), moe);
			assert_eq!(model.num_layers(), 1);
			assert_eq!(
				model
					.block(0)
					.expect("Flow block is missing")
					.attention_mode(),
				oa::ml::nn::AttentionMode::Bidirectional
			);
			let output = model.forward_conditioned(&input, &condition, None)?;
			assert_eq!(output.shape(), input.shape());
			assert!(output.read_f32()?.iter().all(|value| value.is_finite()));
		}
		Ok(())
	}
);

test_vk!(
	flow_padding_mask_prevents_invalid_keys_changing_valid_tokens,
	engine,
	{
		let first = oa::Matrix::from_f32(
			&engine,
			[1, 2, 4],
			&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
		)?;
		let changed = oa::Matrix::from_f32(
			&engine,
			[1, 2, 4],
			&[1.0, 2.0, 3.0, 4.0, 500.0, -600.0, 700.0, -800.0],
		)?;
		let mask = oa::Matrix::from_f32(&engine, [1, 2, 1], &[1.0, 0.0])?;
		for moe in [false, true] {
			let mut config = transformer_config(moe);
			config.adaptive_conditioning = false;
			let model = oa::ml::nn::FlowTransformer::with_seed(&engine, config, 0x4d41_534b)?;
			let a = model.forward_masked(&first, &mask)?.read_f32()?;
			let b = model.forward_masked(&changed, &mask)?.read_f32()?;
			assert_close(&a[..4], &b[..4], 2.0e-5);
		}
		Ok(())
	}
);

test_vk!(
	flow_denoiser_dense_moe_cfg_and_reverse_match_one_contract,
	engine,
	{
		for moe in [false, true] {
			let config = oa::ml::nn::FlowDenoiserConfig {
				input_dim: 2,
				condition_dim: 3,
				backbone: transformer_config(moe),
				time_scale: 1.0,
				condition_dropout_probability: 0.25,
				..Default::default()
			};
			let model = oa::ml::nn::FlowDenoiser::with_seed(&engine, config, 0x444e_4f49)?;
			assert_eq!(model.is_moe(), moe);
			let sample = oa::Matrix::from_f32(
				&engine,
				[2, 2, 2],
				&[0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
			)?;
			let time = oa::Matrix::from_f32(&engine, [2, 1], &[0.25, 0.75])?;
			let condition =
				oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 0.0, -1.0, 0.5, 0.25, -0.5])?;

			let tape = oa::ml::GradientTape::new();
			let output = model.forward_conditioned(&sample, &time, Some(&condition), None)?;
			assert_eq!(output.shape(), sample.shape());
			let loss = oa::matrix::reshape(&oa::matrix::sum(&output, -1)?, Vec::new())?;
			tape.backward(&loss)?;
			for parameter in model.all_named_parameters()? {
				let gradient = parameter
					.parameter()
					.gradient()
					.unwrap_or_else(|| panic!("missing gradient for {}", parameter.path()));
				assert!(
					gradient.read_f32()?.iter().all(|value| value.is_finite()),
					"non-finite gradient for {}",
					parameter.path()
				);
			}
			assert!(model.position().gradient().is_some());

			model.eval();
			let zero_condition = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
			let unconditional =
				model.forward_conditioned(&sample, &time, Some(&zero_condition), None)?;
			let conditional = model.forward_conditioned(&sample, &time, Some(&condition), None)?;
			let guidance_zero = model.forward_guided(&sample, &time, &condition, 0.0, None)?;
			let guidance_one = model.forward_guided(&sample, &time, &condition, 1.0, None)?;
			assert_close(
				&guidance_zero.read_f32()?,
				&unconditional.read_f32()?,
				2.0e-5,
			);
			assert_close(&guidance_one.read_f32()?, &conditional.read_f32()?, 2.0e-5);
		}
		Ok(())
	}
);

test_vk!(flow_models_reject_invalid_geometry_and_guidance, engine, {
	let invalid = oa::ml::nn::FlowTransformerConfig {
		model_width: 4,
		hidden_width: 8,
		sequence_length: 2,
		num_experts: 2,
		experts_per_token: 3,
		..Default::default()
	};
	assert_eq!(
		oa::ml::nn::FlowTransformer::with_seed(&engine, invalid, 1)
			.err()
			.expect("invalid Flow Transformer config was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	let config = oa::ml::nn::FlowDenoiserConfig {
		input_dim: 2,
		condition_dim: 3,
		backbone: transformer_config(false),
		condition_dropout_probability: 1.0,
		..Default::default()
	};
	assert_eq!(
		oa::ml::nn::FlowDenoiser::with_seed(&engine, config, 1)
			.err()
			.expect("invalid FlowDenoiser dropout was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	let left = oa::Matrix::from_f32(&engine, [2, 2], &[1.0; 4])?;
	let wrong_shape = oa::Matrix::from_f32(&engine, [2, 1], &[1.0; 2])?;
	let bad_time = oa::Matrix::from_f32(&engine, [3], &[0.5; 3])?;
	let bad_mask = oa::Matrix::from_f32(&engine, [3, 1], &[1.0; 3])?;
	for error in [
		oa::ml::flow::linear_match(&left, &wrong_shape, &bad_time)
			.err()
			.expect("mismatched flow endpoints were accepted"),
		oa::ml::flow::linear_match(&left, &left, &bad_time)
			.err()
			.expect("non-broadcast flow time was accepted"),
		oa::ml::flow::euler_step(&left, &wrong_shape, 0.1)
			.err()
			.expect("mismatched Euler state was accepted"),
		oa::ml::flow::euler_step(&left, &left, f32::NAN)
			.err()
			.expect("non-finite Euler step was accepted"),
		oa::ml::flow::masked_mse(&left, &left, &bad_mask)
			.err()
			.expect("non-broadcast masked-MSE mask was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
