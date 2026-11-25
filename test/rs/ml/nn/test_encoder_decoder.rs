use oa::ml::Module;

test_vk!(
	encoder_decoder_constructs_and_enumerates_its_tree,
	runtime,
	{
		let model =
			oa::ml::nn::EncoderDecoder::new_with_seed(11, 13, 4, 3, 8, 16, 2, 2, 1.0e-5, 0x4543_4f44)?;
		assert_eq!(model.source_vocab_size(), 11);
		assert_eq!(model.target_vocab_size(), 13);
		assert_eq!(model.source_context_length(), 4);
		assert_eq!(model.target_context_length(), 3);
		assert_eq!(model.model_width(), 8);
		assert_eq!(model.hidden_width(), 16);
		assert_eq!(model.num_layers(), 2);
		assert_eq!(model.num_heads(), 2);
		assert!(model.encoder_block(0).is_some());
		assert!(model.decoder_block(1).is_some());
		assert!(model.encoder_block(2).is_none());
		let named = model.all_named_parameters()?;
		let paths = named.iter().map(|entry| entry.path()).collect::<Vec<_>>();
		assert_eq!(paths[0], "src_token_embedding.weight");
		assert_eq!(paths[1], "src_position_embedding.weight");
		assert_eq!(paths[2], "tgt_token_embedding.weight");
		assert_eq!(paths[3], "tgt_position_embedding.weight");
		assert!(paths.contains(&"encoder_0.ln_attn.weight"));
		assert!(paths.contains(&"encoder_1.ffn2.bias"));
		assert!(paths.contains(&"decoder_0.self_attn.q_proj.weight"));
		assert!(paths.contains(&"decoder_0.cross_attn.out_proj.weight"));
		assert!(paths.contains(&"decoder_1.ffn2.bias"));
		assert!(paths.contains(&"encoder_norm.weight"));
		assert!(paths.contains(&"decoder_norm.weight"));
		assert_eq!(paths.last().copied(), Some("head.bias"));
		Ok(())
	}
);

test_vk!(
	encoder_decoder_forward_flows_gradients_into_the_encoder,
	runtime,
	{
		let model =
			oa::ml::nn::EncoderDecoder::new_with_seed(7, 9, 4, 3, 8, 16, 1, 2, 1.0e-5, 0x5351_3251)?;
		let source = oa::Matrix::new_from_slice([2, 4], &[1_u32, 2, 3, 4, 5, 6, 0, 1])?;
		let target = oa::Matrix::new_from_slice([2, 3], &[0_u32, 1, 2, 3, 4, 5])?;
		let labels = oa::Matrix::new_from_slice([6], &[1_u32, 2, 3, 4, 5, 6])?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&source, &target)?;
		assert_eq!(logits.shape(), [6, 9]);
		let loss = oa::ml::loss::cross_entropy(&logits, &labels)?;
		tape.backward(&loss)?;
		for parameter in model.all_parameters()? {
			let gradient = parameter
				.gradient()
				.expect("EncoderDecoder parameter gradient is missing");
			assert!(
				gradient.read_f32()?.iter().all(|value| value.is_finite()),
				"non-finite gradient in EncoderDecoder parameter",
			);
		}
		// Cross-attention must carry loss back through the encoder: source
		// embeddings and encoder block parameters cannot receive nonzero
		// gradients unless the decoder's forward_kv stage differentiates.
		for parameter in model.all_named_parameters()? {
			if !parameter.path().starts_with("src_") && !parameter.path().starts_with("encoder_0") {
				continue;
			}
			let gradient = parameter
				.parameter()
				.gradient()
				.expect("encoder-side parameter gradient is missing");
			assert!(
				gradient.read_f32()?.iter().any(|value| *value != 0.0),
				"{} received an all-zero gradient; cross-attention did not differentiate",
				parameter.path(),
			);
		}
		Ok(())
	}
);

test_vk!(
	multi_head_attention_forward_kv_matches_self_attention_identity,
	runtime,
	{
		// With an identical key/value source and bidirectional visibility,
		// forward_kv reduces to bidirectional self-attention over the same
		// weights; the composed score/softmax/context route must agree with
		// the fused provider within float tolerance.
		let attention = oa::ml::nn::MultiHeadAttention::new_with_seed(8, 2, 3, 0x4b56_4d41)?;
		attention.set_mode(oa::ml::nn::AttentionMode::Bidirectional);
		attention.eval();
		let values = (0..48)
			.map(|index| ((index * 37 % 97) as f32 - 48.0) / 53.0)
			.collect::<Vec<_>>();
		let input = oa::Matrix::new_from_f32([6, 8], &values)?;
		let expected = attention.forward(&input)?;
		let actual = attention.forward_kv(&input, &input)?;
		assert_eq!(actual.shape(), expected.shape());
		let expected = expected.read_f32()?;
		let actual = actual.read_f32()?;
		for (index, (left, right)) in expected.iter().zip(actual.iter()).enumerate() {
			assert!(
				(left - right).abs() <= 1.0e-5,
				"forward_kv diverged from bidirectional self-attention at {index}: {left} vs {right}",
			);
		}
		Ok(())
	}
);

test_vk!(encoder_decoder_rejects_invalid_contracts, runtime, {
	let model = oa::ml::nn::EncoderDecoder::new_with_seed(7, 9, 4, 3, 8, 16, 1, 2, 1.0e-5, 1)?;
	let source = oa::Matrix::new_from_slice([1, 4], &[0_u32, 1, 2, 3])?;
	let target = oa::Matrix::new_from_slice([1, 3], &[0_u32, 1, 2])?;
	let short_source = oa::Matrix::new_from_slice([1, 3], &[0_u32, 1, 2])?;
	assert_eq!(
		model
			.forward(&short_source, &target)
			.err()
			.expect("short source was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let f32_source = oa::Matrix::new_from_f32([1, 4], &[0.0, 1.0, 2.0, 3.0])?;
	assert_eq!(
		model
			.forward(&f32_source, &target)
			.err()
			.expect("F32 source tokens were accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let encoder_output = model.encode(&source)?;
	let wrong_batch = oa::Matrix::new_from_slice([2, 3], &[0_u32; 6])?;
	assert_eq!(
		model
			.decode(&wrong_batch, &encoder_output)
			.err()
			.expect("mismatched decode batch was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let wrong_width = oa::Matrix::new_from_f32([4, 7], &[0.0; 28])?;
	assert_eq!(
		model
			.decode(&target, &wrong_width)
			.err()
			.expect("wrong-width encoder output was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
