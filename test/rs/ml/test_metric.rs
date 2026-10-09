test_vk!(metric_host_observations_match_the_device_scalars, engine, {
	let loss = oa::Matrix::from_f32(&engine, [1], &[0.375])?;
	assert_eq!(oa::ml::metric::scalar_loss(&loss)?, 0.375);

	let logits = oa::Matrix::from_f32(
		&engine,
		[4, 3],
		&[4.0, 4.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 3.0, 1.0, 0.0],
	)?;
	let labels = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 0, 0])?;
	assert_eq!(oa::ml::metric::accuracy(&logits, &labels)?, 0.75);
	Ok(())
});

test_vk!(metric_rejects_non_scalar_loss, engine, {
	let loss = oa::Matrix::from_f32(&engine, [2], &[0.25, 0.5])?;
	let error = oa::ml::metric::scalar_loss(&loss).expect_err("vector loss was accepted");
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
});
