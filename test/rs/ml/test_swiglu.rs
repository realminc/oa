use oa::ml::Module;

test_vk!(
	swiglu_module_preserves_donor_bias_contract_and_gradients,
	engine,
	{
		for has_bias in [false, true] {
			let module = oa::ml::nn::Swiglu::with_seed(&engine, 4, 7, has_bias, 0x5357_474c)?;
			assert_eq!(module.input_features(), 4);
			assert_eq!(module.intermediate_size(), 7);
			assert_eq!(module.has_bias(), has_bias);
			let expected_paths: &[&str] = if has_bias {
				&[
					"gate_weight",
					"up_weight",
					"down_weight",
					"gate_bias",
					"up_bias",
					"down_bias",
				]
			} else {
				&["gate_weight", "up_weight", "down_weight"]
			};
			assert_eq!(
				module
					.all_named_parameters()?
					.iter()
					.map(|entry| entry.path())
					.collect::<Vec<_>>(),
				expected_paths
			);
			assert_eq!(module.num_parameters()?, if has_bias { 102 } else { 84 });

			let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
				&engine,
				[3, 4],
				&[
					0.2, -0.4, 0.8, 1.1, -0.7, 0.3, 0.5, -0.2, 0.9, -1.0, 0.1, 0.6,
				],
			)?)?;
			let indices = oa::Matrix::from_slice(&engine, [1, 3], &[0_u32, 1, 2])?;
			let targets = oa::Matrix::from_slice(&engine, [3], &[1_u32, 3, 0])?;
			let tape = oa::ml::GradientTape::new();
			let output = module.forward(&embedding.forward(&indices)?)?;
			assert_eq!(output.shape(), [1, 3, 4]);
			let loss = oa::ml::loss::cross_entropy(&output.reshape([3, 4])?, &targets)?;
			tape.backward(&loss)?;
			assert!(embedding.weight().gradient().is_some());
			for parameter in module.all_parameters()? {
				assert!(
					parameter
						.gradient()
						.expect("SwiGLU parameter gradient is missing")
						.read_f32()?
						.iter()
						.all(|value| value.is_finite())
				);
			}
		}
		Ok(())
	}
);

test_vk!(swiglu_module_rejects_invalid_contracts, engine, {
	for (input_features, intermediate_size) in [(0, 4), (4, 0)] {
		assert_eq!(
			oa::ml::nn::Swiglu::with_seed(&engine, input_features, intermediate_size, false, 1,)
				.err()
				.expect("invalid SwiGLU configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	let module = oa::ml::nn::Swiglu::with_seed(&engine, 4, 7, false, 1)?;
	let wrong_width = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	assert_eq!(
		module
			.forward(&wrong_width)
			.err()
			.expect("wrong-width SwiGLU input was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});
