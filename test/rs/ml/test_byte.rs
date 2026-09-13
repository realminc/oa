use oa::{Matrix, ml};

test_vk!(
	byte_upload_embedding_and_adjoint_match_donor_contract,
	engine,
	{
		let bytes = [65_u8, 66, 65];
		let encoded = ml::byte::encode_batched(&engine, &bytes)?;
		assert_eq!(encoded.shape(), [1, 3]);
		assert_eq!(encoded.read::<u8>()?, bytes);

		let mut weights = vec![0.0_f32; ml::byte::VOCAB_SIZE * 2];
		weights[65 * 2..65 * 2 + 2].copy_from_slice(&[1.0, 2.0]);
		weights[66 * 2..66 * 2 + 2].copy_from_slice(&[-1.0, 3.0]);
		let embedding = ml::nn::ByteEmbedding::from_matrix(Matrix::from_f32(
			&engine,
			[ml::byte::VOCAB_SIZE, 2],
			&weights,
		)?)?;
		let tape = ml::GradientTape::new();
		let output = embedding.forward(&encoded)?;
		assert_eq!(output.shape(), [1, 3, 2]);
		assert_eq!(output.read_f32()?, [1.0, 2.0, -1.0, 3.0, 1.0, 2.0]);
		let zero = Matrix::from_f32(&engine, [1, 3, 2], &[0.0; 6])?;
		let loss = ml::loss::mse(&output, &zero)?;
		tape.backward(&loss)?;

		let gradient = embedding
			.weight()
			.gradient()
			.expect("byte embedding adjoint did not reach its table")
			.read_f32()?;
		assert_eq!(&gradient[0..65 * 2], &[0.0; 65 * 2]);
		for (actual, expected) in
			gradient[65 * 2..65 * 2 + 4]
				.iter()
				.zip([2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, 1.0])
		{
			assert!(
				(actual - expected).abs() <= 1.0e-6,
				"expected {expected}, found {actual}"
			);
		}
		assert!(gradient[67 * 2..].iter().all(|value| *value == 0.0));
		Ok(())
	}
);

test_vk!(byte_logit_decode_is_exact_and_checked, engine, {
	let mut logits = vec![-10.0_f32; 2 * ml::byte::VOCAB_SIZE];
	logits[65] = 3.0;
	logits[ml::byte::VOCAB_SIZE + 255] = 4.0;
	let logits = Matrix::from_f32(&engine, [2, ml::byte::VOCAB_SIZE], &logits)?;
	assert_eq!(ml::byte::decode(&logits)?, [65, 255]);

	let invalid = Matrix::from_f32(&engine, [2, 255], &[0.0; 510])?;
	assert_eq!(
		ml::byte::decode(&invalid)
			.expect_err("non-byte vocabulary was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

#[test]
fn byte_control_tokens_and_vocabulary_are_stable() {
	assert_eq!(ml::byte::VOCAB_SIZE, 256);
	assert_eq!(
		[ml::byte::PAD, ml::byte::BOS, ml::byte::EOS, ml::byte::SEP],
		[0, 1, 2, 3]
	);
}
