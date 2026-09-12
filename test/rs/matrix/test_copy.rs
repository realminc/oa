test_vk!(
	copy_owns_independent_storage_and_preserves_reverse_mode,
	engine,
	{
		let source = oa::Matrix::from_f32(&engine, [2], &[3.0, 4.0])?;
		let copied = oa::matrix::copy(&source)?;
		oa::ml::optim::clip_grad_norm(std::slice::from_ref(&copied), 1.0)?;
		assert_eq!(source.read_f32()?, [3.0, 4.0]);
		let copied_values = copied.read_f32()?;
		assert!((copied_values[0] - 0.6).abs() < 1.0e-6);
		assert!((copied_values[1] - 0.8).abs() < 1.0e-6);

		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[2, 1],
			&[2.0, -3.0],
		)?)?;
		let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let value = embedding.forward(&rows)?;
		let copied = oa::matrix::copy(&value)?;
		let loss = oa::matrix::reshape(
			&oa::matrix::sum(&oa::matrix::reshape(&copied, [2])?, -1)?,
			[],
		)?;
		tape.backward(&loss)?;
		assert_eq!(
			embedding
				.weight()
				.gradient()
				.expect("copy dropped its identity adjoint")
				.read_f32()?,
			[1.0, 1.0]
		);
		Ok(())
	}
);
