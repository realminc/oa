//! Misc module contracts: Bilinear, Glu, Unflatten, ChannelShuffle,
//! LocalResponseNorm, PairwiseDistance, CosineSimilarity.

use oa::ml::Module as _;

test_vk!(
	bilinear_matches_host_oracle_and_registers_parameters,
	runtime,
	{
		let _ = runtime;
		// W [2,2,2]: W[0] = [[1,0],[0,1]], W[1] = [[0,1],[1,0]]
		let weight = oa::Matrix::new_from_f32([2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])?;
		let bias = oa::Matrix::new_from_f32([2], &[0.5, -0.5])?;
		let module = oa::ml::nn::Bilinear::new_from_matrices(weight, bias)?;
		let x1 = oa::Matrix::new_from_f32([1, 2], &[2.0, 3.0])?;
		let x2 = oa::Matrix::new_from_f32([1, 2], &[4.0, 5.0])?;
		let output = module.forward2(&x1, &x2)?;
		assert_eq!(output.shape(), [1, 2]);
		// out0 = 2*4 + 3*5 + 0.5 = 23.5; out1 = 2*5 + 3*4 - 0.5 = 21.5
		assert_eq!(output.read_f32()?, [23.5, 21.5]);
		assert_eq!(module.parameters().len(), 2);
		Ok(())
	}
);

test_vk!(
	glu_unflatten_channel_shuffle_match_shape_oracles,
	runtime,
	{
		let _ = runtime;
		let input = oa::Matrix::new_from_f32([1, 4], &[1.0, 2.0, 0.0, 2.0])?;
		let glu = oa::ml::nn::Glu::new();
		let output = glu.forward(&input)?;
		assert_eq!(output.shape(), [1, 2]);
		let sig0 = 1.0 / (1.0 + (-0.0_f32).exp());
		let sig2 = 1.0 / (1.0 + (-2.0_f32).exp());
		assert!((output.read_f32()?[0] - sig0).abs() < 1e-6);
		assert!((output.read_f32()?[1] - 2.0 * sig2).abs() < 1e-6);

		let unflatten = oa::ml::nn::Unflatten::new(1, &[2, 2])?;
		let expanded = unflatten.forward(&input)?;
		assert_eq!(expanded.shape(), [1, 2, 2]);
		assert_eq!(expanded.read_f32()?, input.read_f32()?);
		assert!(
			unflatten
				.forward(&oa::Matrix::new_from_f32([1, 3], &[0.0; 3])?)
				.is_err()
		);

		// [1,4,1]: channels 0..3 -> groups of 2 -> out = [c0, c2, c1, c3]
		let channels = oa::Matrix::new_from_f32([1, 4, 1], &[10.0, 20.0, 30.0, 40.0])?;
		let shuffled = oa::ml::nn::ChannelShuffle::new(2)?.forward(&channels)?;
		assert_eq!(shuffled.shape(), [1, 4, 1]);
		assert_eq!(shuffled.read_f32()?, [10.0, 30.0, 20.0, 40.0]);
		assert!(
			oa::ml::nn::ChannelShuffle::new(3)
				.and_then(|module| module.forward(&channels))
				.is_err()
		);
		Ok(())
	}
);

test_vk!(local_response_norm_matches_host_oracle, runtime, {
	let _ = runtime;
	// size=3, alpha=3e-4*3? donor: alpha/size * sum; alpha=0.9, beta=1, k=0
	// x = [1,2,3], squares [1,4,9]
	// c0: (0 + 0.3*(1+4)) = 1.5 -> 1/1.5
	// c1: (0 + 0.3*(1+4+9)) = 4.2 -> 2/4.2
	// c2: (0 + 0.3*(4+9)) = 3.9 -> 3/3.9
	let module = oa::ml::nn::LocalResponseNorm::new_with_options(3, 0.9, 1.0, 0.0)?;
	let input = oa::Matrix::new_from_f32([1, 3, 1], &[1.0, 2.0, 3.0])?;
	let output = module.forward(&input)?.read_f32()?;
	let expected = [1.0 / 1.5, 2.0 / 4.2, 3.0 / 3.9];
	for (actual, want) in output.iter().zip(expected) {
		assert!((actual - want).abs() < 1e-6, "lrn {actual} != {want}");
	}
	Ok(())
});

test_vk!(pairwise_and_cosine_match_host_oracles, runtime, {
	let _ = runtime;
	let x1 = oa::Matrix::new_from_f32([2, 2], &[3.0, 4.0, 1.0, 0.0])?;
	let x2 = oa::Matrix::new_from_f32([2, 2], &[0.0, 0.0, 0.0, 0.0])?;
	let distance = oa::ml::nn::PairwiseDistance::new_with_options(2.0, 1, 0.0).forward2(&x1, &x2)?;
	assert_eq!(distance.shape(), [2, 1]);
	let values = distance.read_f32()?;
	assert!((values[0] - 5.0).abs() < 1e-6);
	assert!((values[1] - 1.0).abs() < 1e-6);

	let y2 = oa::Matrix::new_from_f32([2, 2], &[3.0, 4.0, 0.0, 1.0])?;
	let similarity = oa::ml::nn::CosineSimilarity::new().forward2(&x1, &y2)?;
	let values = similarity.read_f32()?;
	assert!((values[0] - 1.0).abs() < 1e-6);
	assert!((values[1] - 0.0).abs() < 1e-6);
	Ok(())
});
