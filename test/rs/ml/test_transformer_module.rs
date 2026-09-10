use oa::ml::Module;

test_vk!(
	donor_transformer_model_owns_and_differentiates_its_stack,
	engine,
	{
		let model =
			oa::ml::nn::Transformer::with_seed(&engine, 5, 3, 4, 7, 2, 2, 1.0e-5, 0x5452_4e53)?;
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

		let tokens = oa::Matrix::from_slice(&engine, [1, 3], &[0_u32, 1, 2])?;
		let targets = oa::Matrix::from_slice(&engine, [3], &[1_u32, 2, 3])?;
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
			oa::ml::nn::Transformer::with_seed(
				&engine, vocab, context, model, hidden, layers, heads, 1.0e-5, 1,
			)
			.err()
			.expect("invalid Transformer configuration was accepted")
			.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let model = oa::ml::nn::Transformer::with_seed(&engine, 5, 3, 4, 7, 1, 2, 1.0e-5, 1)?;
	let wrong_length = oa::Matrix::from_slice(&engine, [1, 2], &[0_u32, 1])?;
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
