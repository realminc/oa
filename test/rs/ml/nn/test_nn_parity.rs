//! nn-parity module contracts: PReLU/ELU, group/instance norm,
//! BatchNorm1d/3d, Pool1d, PixelShuffle, AttentionPool.

use oa::ml::nn::{
	AdaptiveMaxPool1d, AttentionPool, AvgPool1d, BatchNorm1d, BatchNorm3d, Elu, GroupNorm,
	InstanceNorm, MaxPool1d, PixelShuffle, PixelUnshuffle, Prelu,
};
use oa::ml::{GradientTape, Module};

test_vk!(prelu_learns_slope_and_matches_piecewise_oracle, runtime, {
	// per-channel slopes [C=2]: slope[0]=0.25, slope[1]=0.25 (init)
	let prelu = Prelu::new(2)?;
	assert_eq!(prelu.num_parameters()?, 2);
	let input = oa::Matrix::new_from_f32([1, 2, 2], &[2.0, -4.0, -8.0, 1.0])?;
	let output = prelu.forward(&input)?.read_f32()?;
	// [2, -4*0.25, -8*0.25, 1] = [2, -1, -2, 1]
	let expected = [2.0, -1.0, -2.0, 1.0];
	for (a, e) in output.iter().zip(expected.iter()) {
		assert!((a - e).abs() <= 1.0e-6, "prelu deviates: {a} vs {e}");
	}
	// gradient reaches the slope parameter
	let tape = GradientTape::new();
	let output = prelu.forward(&input)?;
	let loss = oa::ml::loss::mse(&output, &oa::Matrix::new_from_f32([1, 2, 2], &[0.0; 4])?)?;
	tape.backward(&loss)?;
	let gradient = prelu.slope().gradient();
	assert!(gradient.is_some());
	assert_eq!(gradient.expect("slope gradient").shape(), &[2]);
	Ok(())
});

test_vk!(elu_matches_the_alpha_oracle, runtime, {
	let elu = Elu::new(1.5)?;
	let input = oa::Matrix::new_from_f32([4], &[2.0, -1.0, 0.0, -3.0])?;
	let output = elu.forward(&input)?.read_f32()?;
	let expected = [
		2.0,
		1.5 * ((-1.0_f32).exp() - 1.0),
		0.0,
		1.5 * ((-3.0_f32).exp() - 1.0),
	];
	for (i, (a, e)) in output.iter().zip(expected.iter()).enumerate() {
		assert!((a - e).abs() <= 1.0e-5, "elu deviates at {i}: {a} vs {e}");
	}
	assert!(Elu::new(f32::NAN).is_err());
	Ok(())
});

test_vk!(
	group_and_instance_norm_normalize_like_the_host_oracle,
	runtime,
	{
		// [N=1, C=4, S=2], groups=2 → two groups of 2 channels.
		let values = [1.0_f32, 3.0, 10.0, 20.0, -5.0, 5.0, 0.0, 8.0];
		let input = oa::Matrix::new_from_f32([1, 4, 2], &values)?;
		let norm = GroupNorm::new(2, 4)?;
		assert_eq!(norm.num_parameters()?, 8);
		let output = norm.forward(&input)?.read_f32()?;
		// group 0 = values[0..4] (channels 0,1), group 1 = values[4..8]
		let mut expected = [0.0_f32; 8];
		for (offset, group) in [(0, &values[0..4]), (4, &values[4..8])] {
			let mean = group.iter().sum::<f32>() / 4.0;
			let var = group.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
			let std = (var + 1e-5).sqrt();
			for (i, v) in group.iter().enumerate() {
				expected[offset + i] = (v - mean) / std;
			}
		}
		for (i, (a, e)) in output.iter().zip(expected.iter()).enumerate() {
			assert!(
				(a - e).abs() <= 1.0e-4,
				"group norm deviates at {i}: {a} vs {e}"
			);
		}
		// instance norm = per-channel groups
		let instance = InstanceNorm::new(4)?;
		let output = instance.forward(&input)?.read_f32()?;
		// each channel pair [v0,v1] normalizes to ±~1
		assert!((output[0] - -1.0).abs() <= 1.0e-3);
		assert!((output[1] - 1.0).abs() <= 1.0e-3);
		assert!(GroupNorm::new(3, 4).is_err());
		Ok(())
	}
);

test_vk!(batch_norm_1d_and_3d_match_the_2d_path, runtime, {
	let bn1 = BatchNorm1d::new_on(&runtime, 2)?;
	let input1 = oa::Matrix::new_from_f32([2, 2, 4], &[0.0; 16])?;
	let out1 = bn1.forward(&input1)?;
	assert_eq!(out1.shape(), &[2, 2, 4]);
	assert_eq!(bn1.num_parameters()?, 4); // weight + bias
	let bn3 = BatchNorm3d::new_on(&runtime, 2)?;
	let input3 = oa::Matrix::new_from_f32([1, 2, 2, 2, 2], &[1.0; 16])?;
	let out3 = bn3.forward(&input3)?;
	assert_eq!(out3.shape(), &[1, 2, 2, 2, 2]);
	// parameters surface under the child path
	let names: Vec<String> = bn3
		.all_named_parameters()?
		.iter()
		.map(|p| p.path().to_string())
		.collect();
	assert_eq!(names.len(), 2);
	assert!(names.iter().all(|n| n.starts_with("inner.")));
	assert!(
		bn1
			.forward(&oa::Matrix::new_from_f32([2, 2], &[0.0; 4])?)
			.is_err()
	);
	Ok(())
});

test_vk!(avg_pool_1d_matches_the_window_oracle, runtime, {
	let input = oa::Matrix::new_from_f32([1, 1, 6], &[1.0, 5.0, 2.0, 8.0, 3.0, 4.0])?;
	let avg1 = AvgPool1d::new(2)?;
	let out = avg1.forward(&input)?.read_f32()?;
	assert_eq!(out, [3.0, 5.0, 3.5]);
	Ok(())
});

test_vk!(pixel_shuffle_rearranges_channels_into_space, runtime, {
	// C=1, r=2: packed channels [N,4,H,W] = values 1..8
	let input = oa::Matrix::new_from_f32(
		[1, 4, 2, 2],
		&(1..=16).map(|v| v as f32).collect::<Vec<f32>>(),
	)?;
	let shuffle = PixelShuffle::new(2)?;
	let output = shuffle.forward(&input)?;
	assert_eq!(output.shape(), &[1, 1, 4, 4]);
	let values = output.read_f32()?;
	// torch ordering: out[h*r+i, w*r+j] = in[c=r*i+j, h, w]
	// channel 0 → top-left quads: [[1,2],[3,4]]... wait c indexing:

	// out[b,c,h,w] = in[b, c*(r*r) + (h%r)*r + (w%r), h/r, w/r]
	let input_values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
	for h in 0..4_usize {
		for w in 0..4_usize {
			let c = (h % 2) * 2 + (w % 2);
			let expected = input_values[c * 4 + (h / 2) * 2 + (w / 2)];
			let actual = values[h * 4 + w];
			assert!(
				(actual - expected).abs() <= 1.0e-6,
				"pixel shuffle [{h},{w}]: {actual} vs {expected}"
			);
		}
	}
	assert!(PixelShuffle::new(1).is_err());
	assert!(
		shuffle
			.forward(&oa::Matrix::new_from_f32([1, 5, 2, 2], &[0.0; 20])?)
			.is_err()
	);
	Ok(())
});

test_vk!(attention_pool_pools_sequence_with_learned_query, runtime, {
	let pool = AttentionPool::new_with_seed_on(&runtime, 8, 2, 0xCAFE)?;
	// query + attention projections are parameters
	assert!(pool.num_parameters()? > 0);
	let input = oa::Matrix::new_from_f32(
		[6, 8],
		&(0..48)
			.map(|i| (i as f32 * 0.1).sin())
			.collect::<Vec<f32>>(),
	)?;
	let output = pool.forward(&input, 2)?;
	assert_eq!(output.shape(), &[2, 8]);
	// deterministic: same input → same output
	let again = pool.forward(&input, 2)?.read_f32()?;
	assert_eq!(output.read_f32()?, again);
	assert!(pool.forward(&input, 4).is_err()); // 6 % 4 != 0
	assert!(pool.forward(&input, 0).is_err());
	Ok(())
});

test_vk!(max_pool_1d_uses_the_kernel_with_argmax_backward, runtime, {
	let pool = MaxPool1d::new_with_options(3, 2, 1)?;
	// input routed through an embedding so reverse mode lands on a parameter
	let embedding = oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
		[2, 7],
		&[
			1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0, 9.0, 1.0, 5.0, 2.0, 7.0,
		],
	)?)?;
	let indices = oa::Matrix::new_from_slice([1, 2], &[0_u32, 1])?;
	// padded [0]x|x|x|x|x|x|x|x[0]; windows at starts -1,1,3,5:
	// c0: max(0,1,5)=5, max(3,7,2)=7, max(2,8,4)=8, max(8,4,0)=8
	// c1: max(0,6,0)=6, max(9,1,5)=9, max(1,5,2)=5, max(2,7,0)=7
	let output = pool.forward(&embedding.forward(&indices)?)?.read_f32()?;
	assert_eq!(output, [5.0, 7.0, 8.0, 8.0, 6.0, 9.0, 5.0, 7.0]);
	// reverse mode reaches the input through the argmax path
	let tape = GradientTape::new();
	let output = pool.forward(&embedding.forward(&indices)?)?;
	let loss = oa::ml::loss::mse(&output, &oa::Matrix::new_from_f32([1, 2, 4], &[0.0; 8])?)?;
	tape.backward(&loss)?;
	let gradient = embedding
		.weight()
		.gradient()
		.expect("input gradient")
		.read_f32()?;
	// mse adjoint 2y/n with n=8; argmax idx: c0 1,3,5,5 | c1 7,9,11,13
	let expected = [
		0.0, 1.25, 0.0, 1.75, 0.0, 4.0, 0.0, 1.5, 0.0, 2.25, 0.0, 1.25, 0.0, 1.75,
	];
	for (index, value) in gradient.iter().enumerate() {
		assert!(
			(value - expected[index]).abs() <= 1.0e-5,
			"max-pool adjoint deviates at {index}: {value} vs {}",
			expected[index]
		);
	}
	Ok(())
});

test_vk!(adaptive_max_pool_1d_pools_to_exact_bins, runtime, {
	let pool = AdaptiveMaxPool1d::new(3)?;
	// L=5, out=3: bins [0,2), [1,4), [3,5)
	let embedding = oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
		[1, 5],
		&[1.0, 5.0, 3.0, 7.0, 2.0],
	)?)?;
	let indices = oa::Matrix::new_from_slice([1, 1], &[0_u32])?;
	let output = pool.forward(&embedding.forward(&indices)?)?.read_f32()?;
	assert_eq!(output, [5.0, 7.0, 7.0]);
	let tape = GradientTape::new();
	let output = pool.forward(&embedding.forward(&indices)?)?;
	let loss = oa::ml::loss::mse(&output, &oa::Matrix::new_from_f32([1, 1, 3], &[0.0; 3])?)?;
	tape.backward(&loss)?;
	let gradient = embedding
		.weight()
		.gradient()
		.expect("input gradient")
		.read_f32()?;
	// mse adjoint 2y/3: argmax idx1 gets 10/3; bins 1 and 2 share idx3 -> 28/3
	let expected = [0.0, 10.0 / 3.0, 0.0, 28.0 / 3.0, 0.0];
	for (index, value) in gradient.iter().enumerate() {
		assert!(
			(value - expected[index]).abs() <= 1.0e-5,
			"adaptive adjoint deviates at {index}: {value} vs {}",
			expected[index]
		);
	}
	Ok(())
});

test_vk!(
	pixel_unshuffle_inverts_pixel_shuffle_channel_ordering,
	runtime,
	{
		// C=1, r=2 round trip: shuffle packs C*r^2 spatial blocks, unshuffle
		// must restore the exact source element order.
		let input = oa::Matrix::new_from_f32(
			[1, 4, 2, 2],
			&(1..=16).map(|v| v as f32).collect::<Vec<f32>>(),
		)?;
		let shuffled = PixelShuffle::new(2)?.forward(&input)?;
		assert_eq!(shuffled.shape(), &[1, 1, 4, 4]);
		let unshuffle = PixelUnshuffle::new(2)?;
		assert_eq!(unshuffle.downscale_factor(), 2);
		let restored = unshuffle.forward(&shuffled)?;
		assert_eq!(restored.shape(), &[1, 4, 2, 2]);
		assert_eq!(
			restored.read_f32()?,
			(1..=16).map(|v| v as f32).collect::<Vec<f32>>()
		);

		// torch ordering oracle: out[b, c*r^2 + i*r + j, h, w] = in[b, c, h*r + i, w*r + j]
		let image_values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
		let image = oa::Matrix::new_from_f32([1, 1, 4, 4], &image_values)?;
		let packed = unshuffle.forward(&image)?;
		assert_eq!(packed.shape(), &[1, 4, 2, 2]);
		let packed_values = packed.read_f32()?;
		for c in 0..4_usize {
			for h in 0..2_usize {
				for w in 0..2_usize {
					let (i, j) = (c / 2, c % 2);
					let expected = image_values[(h * 2 + i) * 4 + (w * 2 + j)];
					let actual = packed_values[(c * 2 + h) * 2 + w];
					assert!(
						(actual - expected).abs() <= 1.0e-6,
						"pixel unshuffle [c={c},h={h},w={w}]: {actual} vs {expected}"
					);
				}
			}
		}
		assert!(PixelUnshuffle::new(1).is_err());
		assert!(
			unshuffle
				.forward(&oa::Matrix::new_from_f32([1, 1, 3, 4], &[0.0; 12])?)
				.is_err()
		);
		Ok(())
	}
);
