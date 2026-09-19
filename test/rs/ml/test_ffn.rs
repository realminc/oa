use oa::ml::Module;

test_vk!(donor_ffn_composes_and_differentiates_every_child, engine, {
	let ffn = oa::ml::nn::Ffn::with_seed(&engine, 4, 7, 1.0e-5, 0x4646_4e00)?;
	assert_eq!(ffn.model_width(), 4);
	assert_eq!(ffn.hidden_width(), 7);
	assert_eq!(ffn.epsilon(), 1.0e-5);

	let named = ffn.all_named_parameters()?;
	assert_eq!(
		named.iter().map(|entry| entry.path()).collect::<Vec<_>>(),
		[
			"norm.weight",
			"gate.weight",
			"gate.bias",
			"up.weight",
			"up.bias",
			"down.weight",
			"down.bias",
		]
	);
	assert_eq!(ffn.num_parameters()?, 106);

	let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[3, 4],
		&[
			0.2, -0.4, 0.8, 1.1, -0.7, 0.3, 0.5, -0.2, 0.9, -1.0, 0.1, 0.6,
		],
	)?)?;
	let indices = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 2])?;
	let targets = oa::Matrix::from_slice(&engine, [3], &[1_u32, 3, 0])?;
	let tape = oa::ml::GradientTape::new();
	let output = ffn.forward(&embedding.forward(&indices)?)?;
	assert_eq!(output.shape(), [3, 4]);
	assert!(output.read_f32()?.iter().all(|value| value.is_finite()));
	let loss = oa::ml::loss::cross_entropy(&output, &targets)?;
	tape.backward(&loss)?;

	assert!(
		embedding
			.weight()
			.gradient()
			.expect("FFN input gradient is missing")
			.read_f32()?
			.iter()
			.all(|value| value.is_finite())
	);
	for parameter in ffn.all_parameters()? {
		assert!(
			parameter
				.gradient()
				.expect("FFN child gradient is missing")
				.read_f32()?
				.iter()
				.all(|value| value.is_finite())
		);
	}
	Ok(())
});

test_vk!(ffn_rejects_invalid_configuration_and_input, engine, {
	for (model_width, hidden_width, epsilon) in [
		(0, 4, 1.0e-5),
		(4, 0, 1.0e-5),
		(4, 4, 0.0),
		(4, 4, f32::NAN),
	] {
		assert_eq!(
			oa::ml::nn::Ffn::with_seed(&engine, model_width, hidden_width, epsilon, 1)
				.err()
				.expect("invalid FFN configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let ffn = oa::ml::nn::Ffn::with_seed(&engine, 4, 7, 1.0e-5, 1)?;
	let input = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	assert_eq!(
		ffn
			.forward(&input)
			.err()
			.expect("wrong-width FFN input was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
