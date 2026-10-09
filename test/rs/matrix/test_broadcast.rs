//! Donor-differential multidirectional Matrix broadcasting tests.

fn rejected<T>(result: oa::Result<T>, message: &str) -> oa::Error {
	result.err().unwrap_or_else(|| panic!("{message}"))
}

test_vk!(add_broadcast_matches_rank_aligned_fp32_oracle, engine, {
	let left = oa::Matrix::from_f32(&engine, [2, 1, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0])?;
	let right = oa::Matrix::from_f32(&engine, [1, 4, 1], &[100.0, 200.0, 300.0, 400.0])?;
	let output = oa::matrix::add(&left, &right)?;
	assert_eq!(output.shape(), [2, 4, 3]);
	assert_eq!(
		output.read_f32()?,
		vec![
			101.0, 102.0, 103.0, 201.0, 202.0, 203.0, 301.0, 302.0, 303.0, 401.0, 402.0, 403.0, 110.0,
			120.0, 130.0, 210.0, 220.0, 230.0, 310.0, 320.0, 330.0, 410.0, 420.0, 430.0,
		]
	);
	Ok(())
});

test_vk!(add_broadcast_preserves_i32_wrapping_arithmetic, engine, {
	let left = oa::Matrix::from_slice(&engine, [2, 1], &[i32::MAX, -2_i32])?;
	let right = oa::Matrix::from_slice(&engine, [1, 2], &[1_i32, i32::MIN])?;
	let output = oa::matrix::add(&left, &right)?;
	assert_eq!(output.shape(), [2, 2]);
	assert_eq!(output.read::<i32>()?, vec![i32::MIN, -1, -1, 2_147_483_646]);
	Ok(())
});

test_vk!(mul_broadcast_matches_rank_aligned_fp32_oracle, engine, {
	let left = oa::Matrix::from_f32(&engine, [2, 1], &[2.0, -1.0])?;
	let right = oa::Matrix::from_f32(&engine, [1, 3], &[3.0, 0.5, -2.0])?;
	let output = oa::matrix::mul(&left, &right)?;
	assert_eq!(output.shape(), [2, 3]);
	assert_eq!(output.read_f32()?, vec![6.0, 1.0, -4.0, -3.0, -0.5, 2.0]);
	Ok(())
});

test_vk!(sub_and_div_broadcast_match_donor_oracles, engine, {
	let left = oa::Matrix::from_f32(&engine, [2, 1], &[12.0, -6.0])?;
	let right = oa::Matrix::from_f32(&engine, [1, 3], &[3.0, 2.0, -4.0])?;
	let difference = oa::matrix::sub(&left, &right)?;
	let quotient = oa::matrix::div(&left, &right)?;
	assert_eq!(difference.shape(), [2, 3]);
	assert_eq!(
		difference.read_f32()?,
		vec![9.0, 10.0, 16.0, -9.0, -8.0, -2.0]
	);
	assert_eq!(quotient.shape(), [2, 3]);
	assert_eq!(quotient.read_f32()?, vec![4.0, 6.0, -3.0, -2.0, -3.0, 1.5]);
	Ok(())
});

test_vk!(add_broadcast_capture_keeps_add_semantic_identity, engine, {
	let left = oa::Matrix::from_f32(&engine, [2, 3], &[1.0; 6])?;
	let right = oa::Matrix::from_f32(&engine, [1, 3], &[2.0; 3])?;
	let (plan, output) = engine.capture(|| oa::matrix::add(&left, &right))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::matrix::add"
	);
	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, vec![3.0; 6]);
	Ok(())
});

test_vk!(
	add_broadcast_rejects_incompatible_or_excessive_shapes,
	engine,
	{
		let left = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
		let right = oa::Matrix::from_f32(&engine, [4, 3], &[0.0; 12])?;
		let rank_nine = oa::Matrix::from_f32(&engine, [1, 1, 1, 1, 1, 1, 1, 1, 2], &[0.0; 2])?;
		let scalar = oa::Matrix::from_f32(&engine, [], &[0.0])?;
		for error in [
			rejected(
				oa::matrix::add(&left, &right),
				"incompatible broadcast was accepted",
			),
			rejected(
				oa::matrix::add(&rank_nine, &scalar),
				"rank-nine broadcast was accepted",
			),
		] {
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);
