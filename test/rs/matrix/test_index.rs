//! Donor-differential Matrix selection and sparse-routing tests.

fn rejected<T>(result: oa::Result<T>, message: &str) -> oa::Error {
	match result {
		Err(error) => error,
		Ok(_) => panic!("{message}"),
	}
}

test_vk!(equal_matches_fp32_scalar_mask_oracle, engine, {
	let input = oa::Matrix::from_f32(
		&engine,
		[6],
		&[-1.0, 0.0, -0.0, 1.0, f32::NAN, f32::INFINITY],
	)?;
	assert_eq!(
		oa::matrix::equal(&input, 0.0)?.read_f32()?,
		[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
	);
	let integers = oa::Matrix::from_slice(&engine, [1], &[0_i32])?;
	assert_eq!(
		rejected(
			oa::matrix::equal(&integers, 0.0),
			"I32 equality exceeded the admitted FP32 slice",
		)
		.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(top_k_matches_deterministic_last_axis_oracle, engine, {
	let input = oa::Matrix::from_f32(
		&engine,
		[3, 5],
		&[
			3.0, 5.0, 5.0, 1.0, 4.0, // ties select index 1 before 2
			-2.0, -1.0, -3.0, -1.0, -4.0, 9.0, 7.0, 8.0, 6.0, 5.0,
		],
	)?;
	let result = oa::matrix::top_k(&input, 3, -1)?;
	assert_eq!(result.values.shape(), [3, 3]);
	assert_eq!(result.indices.shape(), [3, 3]);
	assert_eq!(result.values.dtype(), oa::DType::F32);
	assert_eq!(result.indices.dtype(), oa::DType::I32);
	assert_eq!(
		result.values.read_f32()?,
		vec![5.0, 5.0, 4.0, -1.0, -1.0, -2.0, 9.0, 8.0, 7.0]
	);
	assert_eq!(
		result.indices.read::<i32>()?,
		vec![1, 2, 4, 1, 3, 0, 0, 2, 1]
	);

	let rank_one = oa::Matrix::from_f32(&engine, [3], &[2.0, 4.0, 1.0])?;
	let clamped = oa::matrix::top_k(&rank_one, 10, 0)?;
	assert_eq!(clamped.values.shape(), [3]);
	assert_eq!(clamped.values.read_f32()?, vec![4.0, 2.0, 1.0]);
	assert_eq!(clamped.indices.read::<i32>()?, vec![1, 0, 2]);
	Ok(())
});

test_vk!(
	transpose_matches_tiled_rank_two_rank_three_and_adjoint_oracles,
	engine,
	{
		let rank_two = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
		let transposed = oa::matrix::transpose(&rank_two, -2, -1)?;
		assert_eq!(transposed.shape(), [3, 2]);
		assert_eq!(transposed.read_f32()?, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[2, 6],
			&(0..12).map(|value| value as f32).collect::<Vec<_>>(),
		)?)?;
		let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let input = embedding.forward(&rows)?.reshape([2, 2, 3])?;
		let output = oa::matrix::transpose(&input, 1, 2)?;
		assert_eq!(output.shape(), [2, 3, 2]);
		assert_eq!(
			output.read_f32()?,
			[0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0]
		);
		let target = oa::Matrix::from_f32(&engine, [2, 3, 2], &[0.0; 12])?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("transpose input gradient is missing")
			.read_f32()?;
		for (index, actual) in gradient.iter().enumerate() {
			assert!((*actual - index as f32 / 6.0).abs() <= 1.0e-6);
		}

		let integers = oa::Matrix::from_slice(&engine, [2, 2], &[0_i32; 4])?;
		assert!(oa::matrix::transpose(&integers, 0, 1).is_err());
		assert!(oa::matrix::transpose(&rank_two, 0, 0).is_err());
		assert!(oa::matrix::transpose(&rank_two, -3, -1).is_err());
		Ok(())
	}
);

test_vk!(
	gather_matches_i32_bounds_and_repeated_row_adjoint,
	engine,
	{
		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[4, 3],
			&[
				1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
			],
		)?)?;
		let table_ids = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
		let tape = oa::ml::GradientTape::new();
		let table = embedding.forward(&table_ids)?;
		let indices = oa::Matrix::from_slice(&engine, [3], &[2_i32, 0, 2])?;
		let gathered = oa::matrix::gather(&table, &indices)?;
		assert_eq!(gathered.shape(), [3, 3]);
		assert_eq!(
			gathered.read_f32()?,
			[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
		);
		let loss = oa::matrix::sum(&oa::matrix::sum(&gathered, -1)?, -1)?.reshape([])?;
		tape.backward(&loss)?;
		assert_eq!(
			embedding
				.weight()
				.gradient()
				.expect("gather adjoint did not reach table")
				.read_f32()?,
			[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
		);

		let invalid =
			oa::matrix::gather(&table, &oa::Matrix::from_slice(&engine, [2], &[-1_i32, 4])?)?
				.read_f32()?;
		assert!(invalid.into_iter().all(f32::is_nan));
		Ok(())
	}
);

test_vk!(top_k_mask_matches_exact_membership_oracle, engine, {
	let indices = oa::Matrix::from_slice(&engine, [3, 2], &[2_i32, 0, 1, 2, 0, 1])?;
	let mask = oa::matrix::top_k_mask(&indices, 4)?;
	assert_eq!(mask.shape(), [3, 4]);
	assert_eq!(
		mask.read_f32()?,
		vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
	);
	Ok(())
});

test_vk!(gather_last_dim_matches_donor_and_reverse_oracles, engine, {
	let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
		&engine,
		[2, 4],
		&[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
	)?)?;
	let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
	let indices = oa::Matrix::from_slice(&engine, [2, 3], &[1_i32, 1, 3, 0, 2, 9])?;
	let tape = oa::ml::GradientTape::new();
	let input = embedding.forward(&rows)?;
	let gathered = oa::matrix::gather_last_dim(&input, &indices)?;
	assert_eq!(gathered.shape(), [2, 3]);
	assert_eq!(gathered.read_f32()?, vec![2.0, 2.0, 4.0, -1.0, -3.0, 0.0]);
	let target = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
	let loss = oa::ml::loss::mse(&gathered, &target)?;
	tape.backward(&loss)?;
	let gradient = embedding
		.weight()
		.gradient()
		.expect("gather input gradient is missing")
		.read_f32()?;
	let expected = [0.0, 4.0 / 3.0, 0.0, 4.0 / 3.0, -1.0 / 3.0, 0.0, -1.0, 0.0];
	for (actual, expected) in gradient.iter().zip(expected) {
		assert!((actual - expected).abs() <= 1.0e-6);
	}
	Ok(())
});

test_vk!(moe_routing_bias_update_matches_cpp_oracle, engine, {
	let mask = oa::Matrix::from_f32(
		&engine,
		[4, 4],
		&[
			1.0, 1.0, 0.0, 0.0, // expert counts: [4, 3, 1, 0]
			1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
		],
	)?;
	let bias = oa::Matrix::from_f32(&engine, [1, 4], &[0.1, 0.2, 0.3, 0.4])?;
	oa::matrix::moe_routing_bias_update(&mask, &bias, 2, 0.05)?;
	let actual = bias.read_f32()?;
	for (actual, expected) in actual.iter().zip([0.05, 0.15, 0.35, 0.45]) {
		assert!((actual - expected).abs() <= 1.0e-6);
	}
	Ok(())
});

test_vk!(
	slice_matches_copy_region_and_zero_padded_adjoint_oracles,
	engine,
	{
		let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
			&engine,
			[2, 4],
			&[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
		)?)?;
		let rows = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let input = embedding.forward(&rows)?;
		let output = oa::matrix::slice(&input, 1, 1, 3)?;
		assert_eq!(output.shape(), [2, 2]);
		assert_eq!(output.read_f32()?, vec![2.0, 3.0, -2.0, -3.0]);
		let target = oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		assert_eq!(
			embedding
				.weight()
				.gradient()
				.expect("slice input gradient is missing")
				.read_f32()?,
			vec![0.0, 1.0, 1.5, 0.0, 0.0, -1.0, -1.5, 0.0]
		);
		Ok(())
	}
);

test_vk!(
	repeat_interleave_matches_axis_oracles_and_nonleading_axis_adjoint,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
		let rows = oa::matrix::repeat_interleave(&input, 2, 0)?;
		assert_eq!(rows.shape(), [4, 3]);
		assert_eq!(
			rows.read_f32()?,
			vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
		);

		let columns = oa::matrix::repeat_interleave(&input, 2, 1)?;
		assert_eq!(columns.shape(), [2, 6]);
		assert_eq!(
			columns.read_f32()?,
			vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
		);

		let embedding = oa::ml::nn::Embedding::from_matrix(input)?;
		let indices = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let gathered = embedding.forward(&indices)?;
		let repeated = oa::matrix::repeat_interleave(&gathered, 2, 1)?;
		let target = oa::Matrix::from_f32(&engine, [2, 6], &[0.0; 12])?;
		let loss = oa::ml::loss::mse(&repeated, &target)?;
		tape.backward(&loss)?;
		let actual = embedding
			.weight()
			.gradient()
			.expect("repeat-interleave input gradient is missing")
			.read_f32()?;
		for (actual, expected) in
			actual
				.iter()
				.zip([1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0])
		{
			assert!((actual - expected).abs() <= 1.0e-6);
		}
		Ok(())
	}
);

test_vk!(concat_matches_variadic_copy_and_reverse_oracles, engine, {
	let left = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let right = oa::Matrix::from_f32(&engine, [2, 1], &[5.0, 6.0])?;
	let columns = oa::matrix::concat(&[left, right], 1)?;
	assert_eq!(columns.shape(), [2, 3]);
	assert_eq!(columns.read_f32()?, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);

	let top = oa::Matrix::from_f32(&engine, [1, 2], &[7.0, 8.0])?;
	let bottom = oa::Matrix::from_f32(&engine, [2, 2], &[9.0, 10.0, 11.0, 12.0])?;
	let rows = oa::matrix::concat(&[top, bottom], 0)?;
	assert_eq!(rows.shape(), [3, 2]);
	assert_eq!(rows.read_f32()?, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

	let first =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 2], &[1.5, -3.0])?)?;
	let second =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [1, 1], &[4.5])?)?;
	let index = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
	let tape = oa::ml::GradientTape::new();
	let first_value = first.forward(&index)?;
	let second_value = second.forward(&index)?;
	let joined = oa::matrix::concat(&[first_value, second_value], 1)?;
	let target = oa::Matrix::from_f32(&engine, [1, 3], &[0.0; 3])?;
	let loss = oa::ml::loss::mse(&joined, &target)?;
	tape.backward(&loss)?;
	assert_eq!(
		first
			.weight()
			.gradient()
			.expect("first Concat gradient is missing")
			.read_f32()?,
		vec![1.0, -2.0]
	);
	assert_eq!(
		second
			.weight()
			.gradient()
			.expect("second Concat gradient is missing")
			.read_f32()?,
		vec![3.0]
	);
	Ok(())
});

test_vk!(
	moe_expert_plan_matches_stable_expert_major_oracle,
	engine,
	{
		let indices = oa::Matrix::from_slice(&engine, [3, 2], &[2_i32, 0, 1, 2, 0, 1])?;
		let plan = oa::matrix::moe_expert_plan(&indices, 3)?;
		assert_eq!(plan.counts.read::<u32>()?, vec![2, 2, 2]);
		assert_eq!(plan.offsets.read::<u32>()?, vec![0, 2, 4, 6]);
		assert_eq!(plan.packed_token.read::<u32>()?, vec![0, 2, 1, 2, 0, 1]);
		assert_eq!(plan.packed_expert.read::<u32>()?, vec![0, 0, 1, 1, 2, 2]);
		assert_eq!(plan.packed_slot.read::<u32>()?, vec![1, 4, 2, 5, 0, 3]);
		assert_eq!(plan.inverse.read::<u32>()?, vec![4, 0, 2, 5, 1, 3]);
		Ok(())
	}
);

test_vk!(
	matrix_index_capture_preserves_one_semantic_operation_per_call,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 3.0, 2.0, 6.0, 4.0, 5.0])?;
		let (top_k_plan, selected) = engine.capture(|| oa::matrix::top_k(&input, 2, -1))?;
		assert_eq!(top_k_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			top_k_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::top_k"
		);
		engine.submit(&top_k_plan)?.wait()?;
		assert_eq!(selected.indices.read::<i32>()?, vec![1, 2, 0, 2]);

		let (route_plan, packed) =
			engine.capture(|| oa::matrix::moe_expert_plan(&selected.indices, 3))?;
		assert_eq!(route_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			route_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::moe_expert_plan"
		);
		engine.submit(&route_plan)?.wait()?;
		assert_eq!(packed.offsets.read::<u32>()?, vec![0, 1, 2, 4]);

		let mask = oa::Matrix::from_f32(&engine, [2, 3], &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0])?;
		let bias = oa::Matrix::from_f32(&engine, [3], &[0.0, 0.0, 0.0])?;
		let (bias_plan, ()) =
			engine.capture(|| oa::matrix::moe_routing_bias_update(&mask, &bias, 2, 0.01))?;
		assert_eq!(bias_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			bias_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::moe_routing_bias_update"
		);
		engine.submit(&bias_plan)?.wait()?;

		let (slice_plan, slice) = engine.capture(|| oa::matrix::slice(&input, 1, 1, 3))?;
		assert_eq!(slice_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			slice_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::slice"
		);
		engine.submit(&slice_plan)?.wait()?;
		assert_eq!(slice.read_f32()?, vec![3.0, 2.0, 4.0, 5.0]);

		let (repeat_plan, repeated) =
			engine.capture(|| oa::matrix::repeat_interleave(&input, 2, 1))?;
		assert_eq!(repeat_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			repeat_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::repeat_interleave"
		);
		engine.submit(&repeat_plan)?.wait()?;
		assert_eq!(
			repeated.read_f32()?,
			vec![1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 6.0, 6.0, 4.0, 4.0, 5.0, 5.0]
		);

		let tail = oa::Matrix::from_f32(&engine, [2, 1], &[7.0, 8.0])?;
		let (concat_plan, concatenated) =
			engine.capture(|| oa::matrix::concat(&[input.clone(), tail.clone()], 1))?;
		assert_eq!(concat_plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			concat_plan.semantic_graph().operations()[0].name(),
			"oa::matrix::concat"
		);
		engine.submit(&concat_plan)?.wait()?;
		assert_eq!(
			concatenated.read_f32()?,
			vec![1.0, 3.0, 2.0, 7.0, 6.0, 4.0, 5.0, 8.0]
		);
		Ok(())
	}
);

test_vk!(matrix_index_rejects_invalid_public_contracts, engine, {
	let f32_rank_two = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let i32_rank_two = oa::Matrix::from_slice(&engine, [2, 1], &[0_i32, 1])?;
	let f32_rank_three = oa::Matrix::from_f32(&engine, [1, 1, 1], &[1.0])?;
	for error in [
		rejected(
			oa::matrix::top_k(&f32_rank_two, -1, -1),
			"negative k was accepted",
		),
		rejected(
			oa::matrix::top_k(&f32_rank_two, 1, 0),
			"non-last axis was accepted",
		),
		rejected(
			oa::matrix::top_k(&f32_rank_three, 1, -1),
			"rank three was accepted",
		),
		rejected(
			oa::matrix::top_k_mask(&f32_rank_two, 2),
			"F32 indices were accepted",
		),
		rejected(
			oa::matrix::top_k_mask(&i32_rank_two, 0),
			"zero experts were accepted",
		),
		rejected(
			oa::matrix::moe_expert_plan(&i32_rank_two, 257),
			"too many experts were accepted",
		),
		rejected(
			oa::matrix::moe_routing_bias_update(&f32_rank_two, &f32_rank_two, 1, 0.1),
			"wrong-size bias was accepted",
		),
		rejected(
			oa::matrix::slice(&f32_rank_two, 2, 0, 1),
			"out-of-range slice axis was accepted",
		),
		rejected(
			oa::matrix::slice(&f32_rank_two, 1, 1, 1),
			"empty slice range was accepted",
		),
		rejected(
			oa::matrix::slice(&i32_rank_two, 1, 0, 1),
			"non-F32 slice was accepted",
		),
		rejected(
			oa::matrix::repeat_interleave(&f32_rank_two, 0, 1),
			"zero repeats were accepted",
		),
		rejected(
			oa::matrix::repeat_interleave(&f32_rank_two, 2, -1),
			"negative repeat axis was accepted",
		),
		rejected(
			oa::matrix::repeat_interleave(&f32_rank_two, 2, 2),
			"out-of-range repeat axis was accepted",
		),
		rejected(
			oa::matrix::repeat_interleave(&i32_rank_two, 2, 1),
			"non-F32 repeat was accepted",
		),
		rejected(oa::matrix::concat(&[], 0), "empty Concat was accepted"),
		rejected(
			oa::matrix::concat(std::slice::from_ref(&f32_rank_two), -1),
			"negative Concat axis was accepted",
		),
		rejected(
			oa::matrix::concat(&[f32_rank_two.clone(), f32_rank_three], 0),
			"mixed-rank Concat was accepted",
		),
		rejected(
			oa::matrix::concat(&[f32_rank_two.clone(), i32_rank_two.clone()], 1),
			"mixed-dtype Concat was accepted",
		),
		rejected(
			oa::matrix::concat(
				&[f32_rank_two, oa::Matrix::from_f32(&engine, [1, 1], &[0.0])?],
				1,
			),
			"mismatched non-Concat extent was accepted",
		),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
