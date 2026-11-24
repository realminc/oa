//! Padding module contracts over `matrix::pad`.

test_vk!(constant_pad_modules_match_constant_mode_oracles, engine, {
	let _ = engine;
	let input =
		oa::Matrix::new_from_f32([1, 1, 2, 3], &(0..6).map(|v| v as f32).collect::<Vec<_>>())?;
	let pad1d = oa::ml::nn::ConstantPad1d::new(1, 7.0);
	assert_eq!(pad1d.forward(&input)?.shape(), [1, 1, 2, 5]);
	assert_eq!(
		oa::ml::nn::ConstantPad1d::new_with_padding([1, 2], -1.0)
			.forward(&input)?
			.read_f32()?,
		[
			-1.0, 0.0, 1.0, 2.0, -1.0, -1.0, -1.0, 3.0, 4.0, 5.0, -1.0, -1.0
		]
	);
	let pad2d = oa::ml::nn::ConstantPad2d::new_with_padding([1, 1, 1, 0], 9.0);
	let output = pad2d.forward(&input)?;
	assert_eq!(output.shape(), [1, 1, 3, 5]);
	assert_eq!(
		output.read_f32()?,
		[
			9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 1.0, 2.0, 9.0, 9.0, 3.0, 4.0, 5.0, 9.0,
		]
	);
	let zero = oa::ml::nn::ZeroPad2d::new(1);
	let zeroed = zero.forward(&input)?;
	assert_eq!(zeroed.shape(), [1, 1, 4, 5]);
	assert_eq!(
		zeroed.read_f32()?,
		[
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0,
		]
	);
	Ok(())
});

test_vk!(border_pad_modules_match_mode_oracles, engine, {
	let _ = engine;
	let input = oa::Matrix::new_from_f32([1, 1, 1, 3], &[1.0, 2.0, 3.0])?;
	assert_eq!(
		oa::ml::nn::ReplicationPad1d::new_with_padding([1, 2])
			.forward(&input)?
			.read_f32()?,
		[1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
	);
	assert_eq!(
		oa::ml::nn::ReflectionPad1d::new_with_padding([1, 1])
			.forward(&input)?
			.read_f32()?,
		[2.0, 1.0, 2.0, 3.0, 2.0]
	);
	assert_eq!(
		oa::ml::nn::CircularPad1d::new_with_padding([1, 0])
			.forward(&input)?
			.read_f32()?,
		[3.0, 1.0, 2.0, 3.0]
	);
	let plane = oa::Matrix::new_from_f32([1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let replicated = oa::ml::nn::ReplicationPad2d::new(1).forward(&plane)?;
	assert_eq!(replicated.shape(), [1, 1, 4, 4]);
	assert_eq!(
		replicated.read_f32()?,
		[
			1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
		]
	);
	let plane3d = oa::Matrix::new_from_f32([1, 1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let padded3d =
		oa::ml::nn::ReplicationPad3d::new_with_padding([0, 0, 0, 0, 1, 0]).forward(&plane3d)?;
	assert_eq!(padded3d.shape(), [1, 1, 2, 2, 2]);
	assert_eq!(
		padded3d.read_f32()?,
		[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
	);
	Ok(())
});

test_vk!(pad_modules_reject_rank_and_reflect_contracts, engine, {
	let _ = engine;
	let vector = oa::Matrix::new_from_f32([4], &[1.0, 2.0, 3.0, 4.0])?;
	assert!(
		oa::ml::nn::ConstantPad2d::new(1, 0.0)
			.forward(&vector)
			.is_err()
	);
	let plane = oa::Matrix::new_from_f32([1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	assert!(oa::ml::nn::ReflectionPad2d::new(2).forward(&plane).is_err());
	Ok(())
});
