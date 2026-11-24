//! Channel-masked and alpha-preserving dropout variant contracts.

use oa::ml::Module as _;

test_vk!(channel_dropout_shares_one_mask_per_channel, engine, {
	let _ = engine;
	let input = oa::Matrix::new_from_f32(
		[2, 3, 2, 2],
		&(0..24).map(|v| v as f32 + 1.0).collect::<Vec<_>>(),
	)?;
	for (module, name) in [(oa::ml::nn::Dropout2d::new_with_seed(0.5, 7)?, "Dropout2d")] {
		let output = module.forward(&input)?;
		assert_eq!(output.shape(), input.shape(), "{name} must preserve shape");
		let values = output.read_f32()?;
		let input_values = input.read_f32()?;
		for batch in 0..2 {
			for channel in 0..3 {
				let dropped = (0..4).all(|s| values[(batch * 3 + channel) * 4 + s] == 0.0);
				let kept = (0..4).all(|s| {
					values[(batch * 3 + channel) * 4 + s] == input_values[(batch * 3 + channel) * 4 + s] * 2.0
				});
				assert!(dropped || kept, "{name} must share one mask per channel");
			}
		}
	}
	let flat = oa::Matrix::new_from_f32([2, 3, 4], &(0..24).map(|v| v as f32).collect::<Vec<_>>())?;
	let out1d = oa::ml::nn::Dropout1d::new_with_seed(0.5, 7)?.forward(&flat)?;
	assert_eq!(out1d.shape(), flat.shape());
	let volume = oa::Matrix::new_from_f32(
		[1, 2, 2, 2, 2],
		&(0..16).map(|v| v as f32).collect::<Vec<_>>(),
	)?;
	let out3d = oa::ml::nn::Dropout3d::new_with_seed(0.5, 7)?.forward(&volume)?;
	assert_eq!(out3d.shape(), volume.shape());
	Ok(())
});

test_vk!(alpha_dropouts_apply_the_selu_affine_contract, engine, {
	let _ = engine;
	let input = oa::Matrix::new_from_f32([2, 4], &[0.5; 8])?;
	let alpha = oa::ml::nn::AlphaDropout::new_with_seed(0.4, 11)?;
	let output = alpha.forward(&input)?.read_f32()?;
	// alpha' = -1.7581, a = (1 + alpha'^2 * 0.4 * 0.6)^-0.5, b = -a * alpha' * 0.4
	let alpha_prime = -1.758_099_3_f32;
	let scale_a = (1.0 + alpha_prime * alpha_prime * 0.4 * 0.6).sqrt().recip();
	let scale_b = -scale_a * alpha_prime * 0.4;
	let kept = scale_a * (0.5 + 0.0) + scale_b;
	let dropped = scale_a * alpha_prime + scale_b;
	for value in output {
		assert!(
			(value - kept).abs() < 1e-5 || (value - dropped).abs() < 1e-5,
			"alpha dropout emitted {value}, expected {kept} or {dropped}"
		);
	}
	let channels = oa::Matrix::new_from_f32([1, 4, 2], &[0.5; 8])?;
	let feature = oa::ml::nn::FeatureAlphaDropout::new_with_seed(0.4, 11)?;
	let out = feature.forward(&channels)?.read_f32()?;
	for channel in 0..4 {
		assert!(
			(out[channel * 2] - out[channel * 2 + 1]).abs() < 1e-7,
			"FeatureAlphaDropout must share one mask per channel"
		);
	}
	Ok(())
});

test_vk!(
	dropout_variants_identity_in_eval_and_train_mode_contracts,
	engine,
	{
		let _ = engine;
		let input =
			oa::Matrix::new_from_f32([2, 3, 2, 2], &(0..24).map(|v| v as f32).collect::<Vec<_>>())?;
		let module = oa::ml::nn::Dropout2d::new_with_seed(0.9, 3)?;
		{
			let _eval = module.scoped_eval();
			assert_eq!(module.forward(&input)?.read_f32()?, input.read_f32()?);
		}
		let trained = module.forward(&input)?;
		assert_eq!(trained.shape(), input.shape());
		assert!(oa::ml::nn::Dropout2d::new(1.5).is_err());
		Ok(())
	}
);
