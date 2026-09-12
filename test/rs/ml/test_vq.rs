use oa::ml::Module as _;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: expected {expected}, found {actual}"
		);
	}
}

fn small_config() -> oa::ml::nn::VectorQuantizerConfig {
	oa::ml::nn::VectorQuantizerConfig {
		num_codes: 3,
		code_dim: 2,
		commitment_beta: 0.25,
		ema_decay: 0.5,
		ema_epsilon: 1.0e-5,
		dead_threshold: 0.0,
		normalize_codes: false,
	}
}

test_vk!(
	vq_assignment_ste_ema_lookup_and_seed_match_donor_contract,
	engine,
	{
		let config = small_config();
		let quantizer = oa::ml::nn::VectorQuantizer::with_seed(&engine, config, 7)?;
		quantizer.seed(&oa::Matrix::from_f32(
			&engine,
			[4, 2],
			&[0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.25, 0.25],
		)?)?;
		assert_eq!(
			quantizer.codebook().read_f32()?,
			[2.0, 0.0, 0.0, 2.0, 0.25, 0.25]
		);
		let latent_values = [1.8_f32, 0.1, 0.2, 1.7, 0.1, 0.2, 1.0, 1.0];
		let latent = oa::Matrix::from_f32(&engine, [4, 2], &latent_values)?;
		let result = quantizer.quantize(&latent)?;
		assert_eq!(result.indices.read::<i32>()?, [0, 1, 2, 2]);
		assert_eq!(
			result.quantized.read_f32()?,
			[2.0, 0.0, 0.0, 2.0, 0.25, 0.25, 0.25, 0.25]
		);
		let expected_loss = latent_values
			.iter()
			.zip(result.quantized.read_f32()?)
			.map(|(latent, code)| (latent - code) * (latent - code))
			.sum::<f32>()
			/ latent_values.len() as f32
			* config.commitment_beta;
		assert_close(
			&result.commitment_loss.read_f32()?,
			&[expected_loss],
			1.0e-6,
		);

		quantizer.ema_update(&latent, &result.indices)?;
		assert_eq!(quantizer.ema_step(), 1);
		let buffers = quantizer.all_named_buffers()?;
		assert_eq!(
			buffers
				.iter()
				.map(|buffer| buffer.path())
				.collect::<Vec<_>>(),
			["codebook", "embed_sum", "cluster_size"]
		);
		assert!(buffers.iter().all(|buffer| buffer.persistent()));
		let codebook = quantizer.codebook().read_f32()?;
		assert_close(&codebook, &[1.9, 0.05, 0.1, 1.85, 0.45, 0.48333332], 1.0e-6);
		let decoded = quantizer.lookup(&oa::Matrix::from_slice(&engine, [3], &[2_i32, 0, 1])?)?;
		assert_close(
			&decoded.read_f32()?,
			&[0.45, 0.48333332, 1.9, 0.05, 0.1, 1.85],
			1.0e-6,
		);

		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[4, 2],
			&latent_values,
		)?)?;
		let row_ids = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
		let tape = oa::ml::GradientTape::new();
		let encoded = embedding.forward(&row_ids)?;
		let quantized = quantizer.quantize(&encoded)?.quantized;
		let loss = oa::matrix::sum(&quantized, -1)?.reshape([])?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("STE gradient is missing");
		assert_eq!(gradient.read_f32()?, vec![1.0; 8]);
		Ok(())
	}
);

test_vk!(
	residual_vq_round_trips_tokens_and_persistent_level_tree,
	engine,
	{
		let config = oa::ml::nn::VectorQuantizerConfig {
			num_codes: 2,
			code_dim: 2,
			commitment_beta: 0.25,
			ema_decay: 0.9,
			ema_epsilon: 1.0e-5,
			dead_threshold: 0.0,
			normalize_codes: false,
		};
		let quantizer = oa::ml::nn::ResidualVectorQuantizer::with_seed(&engine, config, 2, 11)?;
		let latent = oa::Matrix::from_f32(&engine, [3, 2], &[3.0, 0.0, 0.0, 3.0, 2.0, 2.0])?;
		quantizer.seed(&latent)?;
		let result = quantizer.quantize(&latent)?;
		assert_eq!(result.indices.len(), 2);
		assert_eq!(result.residuals.len(), 2);
		let decoded = quantizer.lookup(&result.indices)?;
		assert_eq!(decoded.read_f32()?, result.quantized.read_f32()?);
		quantizer.ema_update(&result)?;
		assert_eq!(quantizer.num_levels(), 2);
		assert_eq!(quantizer.all_named_buffers()?.len(), 6);
		assert!(quantizer.all_named_parameters()?.is_empty());
		Ok(())
	}
);

test_vk!(
	vq_ties_dead_code_revival_and_normalization_match_donor,
	engine,
	{
		let tie = oa::ml::matrix::vq_assign(
			&oa::Matrix::from_f32(&engine, [1, 2], &[1.0, 0.0])?,
			&oa::Matrix::from_f32(&engine, [2, 2], &[0.0, 0.0, 2.0, 0.0])?,
		)?;
		assert_eq!(tie.indices.read::<i32>()?, [0]);

		let latent_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
		let latent = oa::Matrix::from_f32(&engine, [4, 2], &latent_values)?;
		let indices = oa::Matrix::from_slice(&engine, [4], &[0_i32; 4])?;
		let zeros = oa::Matrix::from_f32(&engine, [3, 2], &[0.0; 6])?;
		let counts = oa::Matrix::from_f32(&engine, [3], &[0.0; 3])?;
		let state = oa::ml::matrix::vq_ema_update(
			&latent, &indices, &zeros, &counts, &zeros, 0.9, 1.0e-5, 1.0, 7, false,
		)?;
		let codebook = state.codebook.read_f32()?;
		let populations = state.cluster_size.read_f32()?;
		for code in 1_u32..3 {
			let mut hash =
				(code + 1).wrapping_mul(2_654_435_761) ^ 7_u32.wrapping_mul(2_246_822_519);
			hash ^= hash >> 13;
			hash = hash.wrapping_mul(3_266_489_917);
			hash ^= hash >> 16;
			let row = (hash % 4) as usize;
			assert_eq!(populations[code as usize], 1.0);
			assert_eq!(
				&codebook[code as usize * 2..code as usize * 2 + 2],
				&latent_values[row * 2..row * 2 + 2]
			);
		}

		let normalized = oa::ml::matrix::vq_ema_update(
			&oa::Matrix::from_f32(&engine, [2, 2], &[3.0, 4.0, 5.0, 12.0])?,
			&oa::Matrix::from_slice(&engine, [2], &[0_i32, 1])?,
			&oa::Matrix::from_f32(&engine, [2, 2], &[3.0, 4.0, 5.0, 12.0])?,
			&oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?,
			&oa::Matrix::from_f32(&engine, [2, 2], &[3.0, 4.0, 5.0, 12.0])?,
			0.5,
			1.0e-5,
			0.0,
			0,
			true,
		)?;
		let normalized_codebook = normalized.codebook.read_f32()?;
		for row in normalized_codebook.as_chunks::<2>().0 {
			let rms = ((row[0] * row[0] + row[1] * row[1]) / 2.0).sqrt();
			assert!((rms - 1.0).abs() <= 1.0e-6);
		}
		Ok(())
	}
);
