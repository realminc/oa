fn assert_exact(left: &[f32], right: &[f32]) {
	assert_eq!(
		left.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
		right
			.iter()
			.map(|value| value.to_bits())
			.collect::<Vec<_>>()
	);
}

fn cpu_philox(counter: u32, seed: u64) -> [u32; 4] {
	let mut state = [counter, 0, 0, 0];
	let mut key = [seed as u32, (seed >> 32) as u32];
	for _ in 0..10 {
		let product0 = u64::from(0xd251_1f53_u32) * u64::from(state[0]);
		let product1 = u64::from(0xcd9e_8d57_u32) * u64::from(state[2]);
		state = [
			(product1 >> 32) as u32 ^ state[1] ^ key[0],
			product1 as u32,
			(product0 >> 32) as u32 ^ state[3] ^ key[1],
			product0 as u32,
		];
		key[0] = key[0].wrapping_add(0x9e37_79b9);
		key[1] = key[1].wrapping_add(0xbb67_ae85);
	}
	state
}

fn cpu_uniform(count: usize, low: f32, high: f32, seed: u64) -> Vec<f32> {
	(0..count)
		.map(|index| {
			let bits = cpu_philox((index / 4) as u32, seed)[index % 4];
			(bits as f32) * 2.328_306_4e-10_f32 * (high - low) + low
		})
		.collect()
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
	assert_eq!(left.len(), right.len());
	for (index, (left, right)) in left.iter().zip(right).enumerate() {
		assert!(
			(left - right).abs() <= tolerance,
			"mismatch at {index}: actual={left}, expected={right}, tolerance={tolerance}"
		);
	}
}

fn cpu_sample_dense(
	logits: &[f32],
	rows: usize,
	vocabulary: usize,
	temperature: f32,
	seed: u64,
) -> Vec<i32> {
	let random = cpu_uniform(rows, 0.0, 1.0, seed);
	(0..rows)
		.map(|row| {
			let values = &logits[row * vocabulary..(row + 1) * vocabulary];
			let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
			let total = values
				.iter()
				.map(|value| ((value - maximum) / temperature).exp())
				.sum::<f32>();
			let threshold = random[row] * total;
			let mut cumulative = 0.0;
			for (column, value) in values.iter().enumerate() {
				cumulative += ((value - maximum) / temperature).exp();
				if cumulative >= threshold {
					return column as i32;
				}
			}
			(vocabulary - 1) as i32
		})
		.collect()
}

test_vk!(
	philox_eager_preserves_seed_and_distribution_contracts,
	engine,
	{
		let shape = oa::Matrix::from_f32(&engine, [1031], &[0.0; 1031])?;
		let first = oa::matrix::philox_uniform(&shape, -2.0, 4.0, 0x1234_5678_9abc_def0)?;
		let repeated = oa::matrix::philox_uniform(&shape, -2.0, 4.0, 0x1234_5678_9abc_def0)?;
		let different_high = oa::matrix::philox_uniform(&shape, -2.0, 4.0, 0x2234_5678_9abc_def0)?;
		let first = first.read_f32()?;
		let repeated = repeated.read_f32()?;
		let different_high = different_high.read_f32()?;
		assert_exact(&first, &repeated);
		assert_close(
			&first,
			&cpu_uniform(1031, -2.0, 4.0, 0x1234_5678_9abc_def0),
			2.0e-6,
		);
		assert_ne!(first, different_high);
		assert!(first.iter().all(|value| (-2.0..4.0).contains(value)));

		let normal = oa::matrix::philox_normal(&shape, 1.5, 0.25, 77)?;
		let normal_repeated = oa::matrix::philox_normal(&shape, 1.5, 0.25, 77)?;
		let normal = normal.read_f32()?;
		let normal_repeated = normal_repeated.read_f32()?;
		assert_exact(&normal, &normal_repeated);
		assert!(normal.iter().all(|value| value.is_finite()));
		let mean = normal.iter().sum::<f32>() / normal.len() as f32;
		let stddev = (normal
			.iter()
			.map(|value| (value - mean) * (value - mean))
			.sum::<f32>()
			/ normal.len() as f32)
			.sqrt();
		assert!((mean - 1.5).abs() < 0.03, "normal mean was {mean}");
		assert!(
			(stddev - 0.25).abs() < 0.03,
			"normal standard deviation was {stddev}"
		);
		Ok(())
	}
);

test_vk!(
	philox_replay_advances_without_freezing_or_losing_abi_fields,
	engine,
	{
		let shape = oa::Matrix::from_f32(&engine, [1031], &[0.0; 1031])?;
		let (plan, random) =
			engine.capture(|| oa::matrix::philox_uniform(&shape, -3.0, 7.0, 0xfedc_ba98_7654_3210))?;
		assert_eq!(plan.diagnostics().compatibility_node_count(), 0);
		assert_eq!(plan.diagnostics().semantic_operation_count(), 2);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::matrix::philox_uniform"
		);
		assert_eq!(
			plan.semantic_graph().operations()[1].name(),
			"oa::matrix::philox_replay_advance"
		);

		engine.submit(&plan)?.wait()?;
		let replay0 = random.read_f32()?;
		engine.submit(&plan)?.wait()?;
		let replay1 = random.read_f32()?;
		assert_ne!(replay0, replay1);
		assert!(replay0.iter().all(|value| (-3.0..7.0).contains(value)));
		assert!(replay1.iter().all(|value| (-3.0..7.0).contains(value)));

		let (fresh_plan, fresh_random) =
			engine.capture(|| oa::matrix::philox_uniform(&shape, -3.0, 7.0, 0xfedc_ba98_7654_3210))?;
		engine.submit(&fresh_plan)?.wait()?;
		assert_exact(&replay0, &fresh_random.read_f32()?);
		Ok(())
	}
);

test_vk!(
	philox_rejects_invalid_public_contracts_before_recording,
	engine,
	{
		let float = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let integer = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		for error in [
			oa::matrix::philox_uniform(&float, 1.0, 1.0, 1)
				.err()
				.expect("equal uniform bounds were accepted"),
			oa::matrix::philox_uniform(&integer, 0.0, 1.0, 1)
				.err()
				.expect("integer uniform input was accepted"),
			oa::matrix::philox_normal(&float, 0.0, -1.0, 1)
				.err()
				.expect("negative normal deviation was accepted"),
			oa::matrix::philox_normal(&integer, 0.0, 1.0, 1)
				.err()
				.expect("integer normal input was accepted"),
		] {
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

test_vk!(
	dropout_eager_and_replay_preserve_random_contracts,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [1031], &[1.0; 1031])?;
		let first = oa::matrix::dropout(&input, 0.5, 0x1234_5678_9abc_def0)?;
		let repeated = oa::matrix::dropout(&input, 0.5, 0x1234_5678_9abc_def0)?;
		let first = first.read_f32()?;
		let repeated = repeated.read_f32()?;
		assert_exact(&first, &repeated);
		let expected = cpu_uniform(1031, 0.0, 1.0, 0x1234_5678_9abc_def0)
			.into_iter()
			.map(|value| if value >= 0.5 { 2.0 } else { 0.0 })
			.collect::<Vec<_>>();
		assert_exact(&first, &expected);
		assert!(first.contains(&0.0));
		assert!(first.contains(&2.0));
		assert!(first.iter().all(|value| *value == 0.0 || *value == 2.0));

		let (plan, dropped) =
			engine.capture(|| oa::matrix::dropout(&input, 0.5, 0x1234_5678_9abc_def0))?;
		assert_eq!(plan.diagnostics().compatibility_node_count(), 0);
		assert_eq!(plan.diagnostics().semantic_operation_count(), 2);
		engine.submit(&plan)?.wait()?;
		let replay0 = dropped.read_f32()?;
		engine.submit(&plan)?.wait()?;
		let replay1 = dropped.read_f32()?;
		assert_ne!(replay0, replay1);
		assert_exact(&first, &replay0);
		assert!(replay1.iter().all(|value| *value == 0.0 || *value == 2.0));
		Ok(())
	}
);

test_vk!(dropout_rejects_invalid_public_contracts, engine, {
	let float = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
	let integer = oa::Matrix::from_slice(&engine, [1], &[1_u32])?;
	for error in [
		oa::matrix::dropout(&float, -0.1, 1)
			.err()
			.expect("negative dropout probability was accepted"),
		oa::matrix::dropout(&float, 1.0, 1)
			.err()
			.expect("unit dropout probability was accepted"),
		oa::matrix::dropout(&integer, 0.5, 1)
			.err()
			.expect("integer dropout input was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(sample_logits_preserves_all_donor_routes, engine, {
	let greedy_logits = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 4.0, 2.0, -1.0, 3.0, 3.0])?;
	assert_eq!(
		oa::matrix::sample_logits(&greedy_logits, 0.0, 0, 1.0, 0)?.read::<i32>()?,
		vec![1, 1]
	);

	let dense_host = [2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0];
	let dense_logits = oa::Matrix::from_f32(&engine, [3, 4], &dense_host)?;
	let seed = 0x1234_5678_9abc_def0;
	let dense = oa::matrix::sample_logits(&dense_logits, 0.75, 0, 1.0, seed)?;
	assert_eq!(
		dense.read::<i32>()?,
		cpu_sample_dense(&dense_host, 3, 4, 0.75, seed)
	);

	let sorted = oa::matrix::sample_logits(&dense_logits, 0.8, 1, 0.9, 123)?;
	assert_eq!(sorted.read::<i32>()?, vec![0, 3, 0]);
	let nucleus = oa::matrix::sample_logits(&dense_logits, 0.8, 0, 1.0e-7, 456)?;
	assert_eq!(nucleus.read::<i32>()?, vec![0, 3, 0]);

	let wide_dense = oa::Matrix::from_f32(&engine, [1025], &[0.0; 1025])?;
	let wide_sample = oa::matrix::sample_logits(&wide_dense, 1.0, 0, 1.0, 789)?;
	assert_eq!(wide_sample.read::<i32>()?.len(), 1);
	Ok(())
});

test_vk!(
	sample_logits_capture_advances_its_private_philox_state,
	engine,
	{
		let logits = oa::Matrix::from_f32(&engine, [257, 4], &[0.0; 1028])?;
		let seed = 0x4452_4f50_4c49_4359;
		let eager = oa::matrix::sample_logits(&logits, 1.0, 0, 1.0, seed)?.read::<i32>()?;
		let (plan, sampled) =
			engine.capture(|| oa::matrix::sample_logits(&logits, 1.0, 0, 1.0, seed))?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::matrix::sample_logits"
		);
		engine.submit(&plan)?.wait()?;
		let first = sampled.read::<i32>()?;
		engine.submit(&plan)?.wait()?;
		let second = sampled.read::<i32>()?;
		assert_eq!(first, eager);
		assert_ne!(second, first);
		Ok(())
	}
);

test_vk!(sample_logits_rejects_invalid_public_contracts, engine, {
	let integer = oa::Matrix::from_slice(&engine, [2], &[1_i32, 2])?;
	let rank_three = oa::Matrix::from_f32(&engine, [1, 1, 2], &[0.0, 1.0])?;
	let wide = oa::Matrix::from_f32(&engine, [1, 1025], &[0.0; 1025])?;
	for error in [
		oa::matrix::sample_logits(&integer, 1.0, 0, 1.0, 1)
			.err()
			.expect("integer logits were accepted"),
		oa::matrix::sample_logits(&rank_three, 1.0, 0, 1.0, 1)
			.err()
			.expect("rank-three logits were accepted"),
		oa::matrix::sample_logits(&wide, 1.0, 0, 0.9, 1)
			.err()
			.expect("oversized sorted vocabulary was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
