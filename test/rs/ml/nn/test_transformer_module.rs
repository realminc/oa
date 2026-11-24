use oa::ml::Module;

test_vk!(
	donor_transformer_model_owns_and_differentiates_its_stack,
	engine,
	{
		let model = oa::ml::nn::Transformer::new_with_seed(5, 3, 4, 7, 2, 2, 1.0e-5, 0x5452_4e53)?;
		assert_eq!(model.vocab_size(), 5);
		assert_eq!(model.context_length(), 3);
		assert_eq!(model.model_width(), 4);
		assert_eq!(model.hidden_width(), 7);
		assert_eq!(model.num_layers(), 2);
		assert_eq!(model.num_heads(), 2);
		assert_eq!(model.num_parameters()?, 391);
		let named = model.all_named_parameters()?;
		assert_eq!(named.len(), 38);
		assert_eq!(named[0].path(), "token_embedding.weight");
		assert_eq!(named[1].path(), "position_embedding.weight");
		assert_eq!(named[2].path(), "block_0.ln_attn.weight");
		assert_eq!(named[17].path(), "block_0.ffn2.bias");
		assert_eq!(named[18].path(), "block_1.ln_attn.weight");
		assert_eq!(named[33].path(), "block_1.ffn2.bias");
		assert_eq!(named[34].path(), "final_norm.weight");
		assert_eq!(named[36].path(), "head.weight");
		assert_eq!(named[37].path(), "head.bias");

		let tokens = oa::Matrix::new_from_slice([1, 3], &[0_u32, 1, 2])?;
		let targets = oa::Matrix::new_from_slice([3], &[1_u32, 2, 3])?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&tokens)?;
		assert_eq!(logits.shape(), [3, 5]);
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;
		for parameter in model.all_parameters()? {
			assert!(
				parameter
					.gradient()
					.expect("Transformer parameter gradient is missing")
					.read_f32()?
					.iter()
					.all(|value| value.is_finite())
			);
		}
		Ok(())
	}
);

test_vk!(transformer_model_rejects_invalid_contracts, engine, {
	for dimensions in [
		(0, 3, 4, 7, 2, 2),
		(5, 0, 4, 7, 2, 2),
		(5, 3, 0, 7, 2, 2),
		(5, 3, 4, 0, 2, 2),
		(5, 3, 4, 7, 0, 2),
		(5, 3, 4, 7, 2, 0),
		(5, 3, 5, 7, 2, 2),
	] {
		let (vocab, context, model, hidden, layers, heads) = dimensions;
		assert_eq!(
			oa::ml::nn::Transformer::new_with_seed(
				vocab, context, model, hidden, layers, heads, 1.0e-5, 1,
			)
			.err()
			.expect("invalid Transformer configuration was accepted")
			.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let model = oa::ml::nn::Transformer::new_with_seed(5, 3, 4, 7, 1, 2, 1.0e-5, 1)?;
	let wrong_length = oa::Matrix::new_from_slice([1, 2], &[0_u32, 1])?;
	assert_eq!(
		model
			.forward(&wrong_length)
			.err()
			.expect("wrong Transformer context length was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(
	moe_transformer_owns_and_differentiates_donor_topology,
	engine,
	{
		let model =
			oa::ml::nn::Transformer::new_with_seed_moe(5, 3, 4, 3, 1, 2, 2, 1, 1.0e-5, 0x4d4f_4554)?;
		assert!(model.is_moe());
		let block = model.block(0).expect("MoE Transformer lost its block");
		assert!(block.is_moe());
		let moe = block.moe().expect("MoE Transformer lost its expert module");
		assert_eq!(moe.model_width(), 4);
		assert_eq!(moe.hidden_width(), 3);
		assert_eq!(moe.num_experts(), 2);
		assert_eq!(moe.experts_per_token(), 1);

		let named = model.all_named_parameters()?;
		assert!(
			named
				.iter()
				.any(|value| value.path() == "block_0.moe.norm.weight")
		);
		assert!(
			named
				.iter()
				.any(|value| value.path() == "block_0.moe.expert_gate_up_weight")
		);
		assert!(
			named
				.iter()
				.all(|value| !value.path().contains("ln_ffn") && !value.path().contains("ffn1"))
		);

		let tokens = oa::Matrix::new_from_slice([1, 3], &[0_u32, 1, 2])?;
		let targets = oa::Matrix::new_from_slice([3], &[1_u32, 2, 3])?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&tokens)?;
		assert_eq!(logits.shape(), [3, 5]);
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;
		for parameter in model.all_parameters()? {
			assert!(
				parameter
					.gradient()
					.expect("MoE Transformer parameter gradient is missing")
					.read_f32()?
					.iter()
					.all(|value| value.is_finite())
			);
		}
		Ok(())
	}
);

test_vk!(
	adaptive_transformer_blocks_preserve_adaln_zero_identity_and_gradients,
	engine,
	{
		let dense =
			oa::ml::nn::TransformerBlock::new_with_seed_conditioned(4, 7, 2, 2, 3, 1.0e-5, 0x4144_414c)?;
		let moe = oa::ml::nn::TransformerBlock::new_with_seed_moe_conditioned(
			4,
			3,
			2,
			2,
			2,
			1,
			3,
			1.0e-5,
			0x4144_4d4f,
		)?;
		for block in [&dense, &moe] {
			assert!(block.is_adaptively_conditioned());
			assert_eq!(block.condition_dim(), Some(3));
			let adaptive = block
				.all_named_parameters()?
				.into_iter()
				.filter(|parameter| parameter.path().starts_with("adaptive_modulation."))
				.collect::<Vec<_>>();
			assert_eq!(adaptive.len(), 2);
			for parameter in adaptive {
				assert!(
					parameter
						.parameter()
						.data()
						.read_f32()?
						.iter()
						.all(|value| *value == 0.0)
				);
			}
		}

		let input = oa::Matrix::new_from_f32(
			[4, 4],
			&[
				0.2, -0.3, 0.5, 0.7, -0.1, 0.4, -0.6, 0.8, 0.9, -0.2, 0.3, -0.5, 0.6, 0.1, -0.7, 0.4,
			],
		)?;
		let condition = oa::Matrix::new_from_f32([2, 3], &[0.25, -0.5, 0.75, -0.2, 0.4, 0.8])?;
		let mask = oa::Matrix::new_from_f32([8, 2], &[0.0; 16])?;

		let tape = oa::ml::GradientTape::new();
		let dense_output = dense.forward_conditioned(&input, &condition)?;
		let moe_output = moe.forward_conditioned_masked(&input, &condition, &mask)?;
		assert_eq!(dense_output.read_f32()?, input.read_f32()?);
		assert_eq!(moe_output.read_f32()?, input.read_f32()?);
		let target = oa::Matrix::new_from_f32([4, 4], &[0.0; 16])?;
		let loss = oa::ml::loss::mse(&dense_output, &target)?;
		tape.backward(&loss)?;

		let adaptive = dense
			.all_named_parameters()?
			.into_iter()
			.filter(|parameter| parameter.path().starts_with("adaptive_modulation."))
			.collect::<Vec<_>>();
		assert_eq!(adaptive.len(), 2);
		for parameter in adaptive {
			let gradient = parameter
				.parameter()
				.gradient()
				.expect("AdaLN-Zero parameter gradient is missing")
				.read_f32()?;
			assert!(gradient.iter().all(|value| value.is_finite()));
			assert!(gradient.iter().any(|value| value.abs() > 1.0e-7));
		}
		Ok(())
	}
);

test_vk!(adaptive_transformer_rejects_invalid_contracts, engine, {
	assert_eq!(
		oa::ml::nn::TransformerBlock::new_with_seed_conditioned(4, 7, 2, 2, 0, 1.0e-5, 1,)
			.err()
			.expect("zero adaptive condition width was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let ordinary = oa::ml::nn::TransformerBlock::new_with_seed(4, 7, 2, 2, 1.0e-5, 1)?;
	let input = oa::Matrix::new_from_f32([2, 4], &[0.0; 8])?;
	let condition = oa::Matrix::new_from_f32([1, 3], &[0.0; 3])?;
	assert_eq!(
		ordinary
			.forward_conditioned(&input, &condition)
			.err()
			.expect("ordinary block accepted conditioned forward")
			.kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let adaptive = oa::ml::nn::TransformerBlock::new_with_seed_conditioned(4, 7, 2, 2, 3, 1.0e-5, 1)?;
	for error in [
		adaptive
			.forward_conditioned(&oa::Matrix::new_from_f32([2, 3], &[0.0; 6])?, &condition)
			.err()
			.expect("wrong model width was accepted"),
		adaptive
			.forward_conditioned(&oa::Matrix::new_from_f32([3, 4], &[0.0; 12])?, &condition)
			.err()
			.expect("non-divisible sequence geometry was accepted"),
		adaptive
			.forward_conditioned(&input, &oa::Matrix::new_from_f32([1, 2], &[0.0; 2])?)
			.err()
			.expect("wrong condition width was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(
	transformer_incremental_decode_matches_full_forward,
	engine,
	{
		let model = oa::ml::nn::Transformer::new_with_seed(8, 4, 8, 16, 2, 2, 1.0e-5, 7)?;
		let token_values = [1_u32, 3, 0, 5];
		let tokens = oa::Matrix::new_from_slice([1, 4], &token_values)?;
		let full = model.forward(&tokens)?;
		assert_eq!(full.shape(), [4, 8]);
		let full_logits = full.read_f32()?;

		let mut cache = model.new_cache(1)?;
		assert!(cache.is_empty());
		let mut step_logits = Vec::new();
		for (position, &value) in token_values.iter().enumerate() {
			let token = oa::Matrix::new_from_slice([1], &[value])?;
			let logits = model.step(&token, &mut cache)?;
			assert_eq!(logits.shape(), [1, 8]);
			assert_eq!(cache.len(), position + 1);
			step_logits.extend(logits.read_f32()?);
		}
		for (index, (stepped, expected)) in step_logits.iter().zip(&full_logits).enumerate() {
			assert!(
				(stepped - expected).abs() <= 1.0e-4,
				"decode logit {index} differs: stepped={stepped}, forward={expected}"
			);
		}

		let extra = oa::Matrix::new_from_slice([1], &[2_u32])?;
		assert!(model.step(&extra, &mut cache).is_err());
		Ok(())
	}
);

test_vk!(transformer_step_rejects_contract_violations, engine, {
	let model = oa::ml::nn::Transformer::new_with_seed(8, 4, 8, 16, 1, 2, 1.0e-5, 3)?;
	let mut cache = model.new_cache(1)?;
	let pair = oa::Matrix::new_from_slice([2], &[1_u32, 2])?;
	assert!(model.step(&pair, &mut cache).is_err());
	let f32_token = oa::Matrix::new_from_f32([1], &[0.5])?;
	assert!(model.step(&f32_token, &mut cache).is_err());
	assert!(model.new_cache(0).is_err());
	Ok(())
});

test_vk!(
	transformer_grouped_query_attention_builds_and_decodes,
	engine,
	{
		use oa::ml::nn::Transformer;

		let grouped = Transformer::new_with_seed_gqa(32, 8, 8, 16, 2, 4, 2, 1.0e-5, 0x4751_5452)?;
		assert_eq!(grouped.num_heads(), 4);
		assert_eq!(grouped.num_kv_heads(), 2);
		assert_eq!(
			Transformer::new_with_seed_gqa(32, 8, 8, 16, 2, 4, 3, 1.0e-5, 0x4751_5452)
				.err()
				.expect("transformer accepted a non-dividing key/value head count")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let tokens = oa::Matrix::new_from_slice([1, 8], &[1_u32, 2, 3, 4, 5, 6, 7, 8])?;
		let logits = grouped.forward(&tokens)?;
		assert_eq!(logits.shape(), [8, 32]);
		let mut cache = grouped.new_cache(1)?;
		let stepped = grouped.step(&oa::Matrix::new_from_slice([1], &[9_u32])?, &mut cache)?;
		assert_eq!(stepped.shape(), [1, 32]);
		assert_eq!(cache.len(), 1);
		Ok(())
	}
);
