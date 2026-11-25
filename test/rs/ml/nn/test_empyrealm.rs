use oa::ml::Module;

fn config(attention_every: usize) -> oa::ml::nn::EmpyrealmConfig {
	oa::ml::nn::EmpyrealmConfig {
		layers: 4,
		attention_every,
		attention_heads: 2,
		sequence_length: 4,
		ffn_hidden: 16,
		moe_experts: 0,
		moe_experts_per_token: 0,
		epsilon: 1e-5,
		mamba: oa::ml::nn::Mamba3Config {
			model_width: 8,
			state_size: 8,
			expand: 2,
			head_dim: 8,
			num_groups: 1,
			mimo_rank: 1,
			rope_fraction: 0.5,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: true,
		},
	}
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(
	empyrealm_hybrid_schedule_interleaves_attention_blocks,
	runtime,
	{
		let model = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
		assert_eq!(model.layer_count(), 4);
		assert_eq!(model.attention_indices(), vec![3]);

		let dense = oa::ml::nn::Empyrealm::new_with_seed(7, config(2), 0x454d_5059)?;
		assert_eq!(dense.layer_count(), 4);
		assert_eq!(dense.attention_indices(), vec![1, 3]);
		Ok(())
	}
);

test_vk!(
	empyrealm_pure_mamba_schedule_has_no_attention_blocks,
	runtime,
	{
		let mut pure = config(0);
		pure.sequence_length = 4;
		let model = oa::ml::nn::Empyrealm::new_with_seed(7, pure, 0x454d_5059)?;
		assert_eq!(model.layer_count(), 4);
		assert!(model.attention_indices().is_empty());

		// Without attention blocks the packed sequence length is not fixed.
		let tokens = oa::Matrix::new_from_slice([2, 3], &[0_u32, 1, 2, 3, 4, 5])?;
		let output = model.forward(&tokens)?;
		assert_eq!(output.shape(), [2, 3, 8]);
		Ok(())
	}
);

test_vk!(empyrealm_rejects_invalid_configuration, runtime, {
	for (layers, ffn_hidden, sequence_length, model_width) in
		[(0, 16, 4, 8), (4, 0, 4, 8), (4, 16, 0, 8), (4, 16, 4, 0)]
	{
		let mut invalid = config(4);
		invalid.layers = layers;
		invalid.ffn_hidden = ffn_hidden;
		invalid.sequence_length = sequence_length;
		invalid.mamba.model_width = model_width;
		assert_eq!(
			oa::ml::nn::Empyrealm::new_with_seed(7, invalid, 1)
				.err()
				.expect("invalid Empyrealm configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	for epsilon in [0.0, -1.0e-5, f32::NAN, f32::INFINITY] {
		let mut invalid = config(4);
		invalid.epsilon = epsilon;
		assert_eq!(
			oa::ml::nn::Empyrealm::new_with_seed(7, invalid, 1)
				.err()
				.expect("invalid Empyrealm epsilon was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	for heads in [0, 3, 6] {
		let mut invalid = config(4);
		invalid.attention_heads = heads;
		assert_eq!(
			oa::ml::nn::Empyrealm::new_with_seed(7, invalid, 1)
				.err()
				.expect("invalid Empyrealm head count was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	for (experts, per_token) in [(2, 0), (2, 3)] {
		let mut invalid = config(4);
		invalid.moe_experts = experts;
		invalid.moe_experts_per_token = per_token;
		assert_eq!(
			oa::ml::nn::Empyrealm::new_with_seed(7, invalid, 1)
				.err()
				.expect("invalid Empyrealm MoE routing was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});

test_vk!(
	empyrealm_forward_is_deterministic_and_shape_checked,
	runtime,
	{
		let first = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
		let second = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
		let tokens = oa::Matrix::new_from_slice([2, 4], &[0_u32, 1, 2, 3, 4, 5, 6, 0])?;
		let output = first.forward(&tokens)?;
		assert_eq!(output.shape(), [2, 4, 8]);
		assert_eq!(
			output.read_f32()?,
			second.forward(&tokens)?.read_f32()?,
			"identical seeds must produce identical Empyrealm outputs"
		);

		let wrong_width = oa::Matrix::new_from_f32([2, 4, 4], &[0.0; 32])?;
		assert_eq!(
			first
				.forward_embedded(&wrong_width)
				.err()
				.expect("Empyrealm accepted a wrong-width embedded input")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_length = oa::Matrix::new_from_f32([2, 3, 8], &[0.0; 48])?;
		assert_eq!(
			first
				.forward_embedded(&wrong_length)
				.err()
				.expect("Empyrealm attention accepted a foreign sequence length")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(empyrealm_registers_one_stable_parameter_tree, runtime, {
	let model = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
	let named = model.all_named_parameters()?;
	let paths = named
		.iter()
		.map(|parameter| parameter.path())
		.collect::<Vec<_>>();
	assert!(paths.contains(&"embed.weight"));
	assert!(paths.contains(&"tail_norm.weight"));
	assert!(paths.contains(&"block_0.norm.weight"));
	assert!(paths.contains(&"block_0.mixer.D"));
	assert!(paths.contains(&"block_0.ffn.norm.weight"));
	assert!(paths.contains(&"block_3.mixer.q_proj.weight"));
	assert!(paths.contains(&"block_3.mixer.out_proj.weight"));
	assert!(
		paths.iter().all(|path| path.starts_with("embed.")
			|| path.starts_with("block_")
			|| path.starts_with("tail_norm.")),
		"unexpected Empyrealm parameter root: {paths:?}"
	);
	Ok(())
});

test_vk!(empyrealm_decode_steps_match_the_full_scan, runtime, {
	let model = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
	let token_values = [0_u32, 1, 2, 3, 4, 5, 6, 0];
	let tokens = oa::Matrix::new_from_slice([2, 4], &token_values)?;
	let scan = model.forward(&tokens)?.read_f32()?;

	let mut state = model.new_state(2)?;
	assert_eq!(state.batch_size(), 2);
	assert_eq!(state.layer_count(), 4);
	for time in 0..4 {
		let column = oa::Matrix::new_from_slice([2, 1], &[token_values[time], token_values[4 + time]])?;
		let stepped = model.step(&column, &mut state)?.read_f32()?;
		let mut expected = Vec::with_capacity(16);
		for batch in 0..2 {
			expected.extend_from_slice(&scan[(batch * 4 + time) * 8..(batch * 4 + time) * 8 + 8]);
		}
		assert_close(&stepped, &expected, 2.0e-4);
	}
	Ok(())
});

test_vk!(
	empyrealm_decode_state_enforces_ownership_batch_and_tape,
	runtime,
	{
		let model = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x454d_5059)?;
		let other = oa::ml::nn::Empyrealm::new_with_seed(7, config(4), 0x5448_4552)?;
		let column = oa::Matrix::new_from_slice([2, 1], &[1_u32, 2])?;

		assert_eq!(
			model
				.new_state(0)
				.err()
				.expect("Empyrealm accepted a zero-batch decode state")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let mut foreign = other.new_state(2)?;
		assert_eq!(
			model
				.step(&column, &mut foreign)
				.err()
				.expect("Empyrealm accepted a foreign decode state")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let mut state = model.new_state(2)?;
		let wrong_batch = oa::Matrix::new_from_slice([1, 1], &[1_u32])?;
		assert_eq!(
			model
				.step(&wrong_batch, &mut state)
				.err()
				.expect("Empyrealm accepted a mismatched decode batch")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let wrong_shape = oa::Matrix::new_from_f32([2, 2, 8], &[0.0; 32])?;
		assert_eq!(
			model
				.step_embedded(&wrong_shape, &mut state)
				.err()
				.expect("Empyrealm accepted a multi-token decode step")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);

		let tape = oa::ml::GradientTape::new();
		assert_eq!(
			model
				.step(&column, &mut state)
				.err()
				.expect("Empyrealm decode ran inside a GradientTape")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		drop(tape);
		Ok(())
	}
);

test_vk!(empyrealm_moe_tail_constructs_and_forwards, runtime, {
	let mut routed = config(4);
	routed.moe_experts = 2;
	routed.moe_experts_per_token = 1;
	let model = oa::ml::nn::Empyrealm::new_with_seed(7, routed, 0x4d4f_4521)?;
	assert_eq!(model.attention_indices(), vec![3]);
	let tokens = oa::Matrix::new_from_slice([1, 4], &[0_u32, 1, 2, 3])?;
	let output = model.forward(&tokens)?;
	assert_eq!(output.shape(), [1, 4, 8]);
	assert!(
		output.read_f32()?.iter().all(|value| value.is_finite()),
		"Empyrealm MoE tail produced a non-finite output"
	);
	Ok(())
});
