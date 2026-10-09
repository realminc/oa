use oa::ml::{ReplayBuffer, ReplayConfig, ReplayTransition};

fn transition(
	engine: &oa::Engine,
	observations: &[[f32; 2]],
	actions: &[i32],
	rewards: &[f32],
	terminated: &[u8],
	truncated: &[u8],
) -> oa::Result<ReplayTransition> {
	let batch = rewards.len();
	let observations = observations.iter().flatten().copied().collect::<Vec<_>>();
	let next_observations = observations
		.iter()
		.map(|value| value + 100.0)
		.collect::<Vec<_>>();
	Ok(ReplayTransition::new(
		oa::Matrix::from_f32(engine, [batch, 2], &observations)?,
		oa::Matrix::from_slice(engine, [batch], actions)?,
		oa::Matrix::from_f32(engine, [batch, 2], &next_observations)?,
		oa::Matrix::from_f32(engine, [batch], rewards)?,
		oa::Matrix::from_slice(engine, [batch], terminated)?,
		oa::Matrix::from_slice(engine, [batch], truncated)?,
	))
}

fn hash32(mut value: u32) -> u32 {
	value ^= value >> 16;
	value = value.wrapping_mul(0x7feb_352d);
	value ^= value >> 15;
	value = value.wrapping_mul(0x846c_a68b);
	value ^ (value >> 16)
}

fn source_index(seed: u64, lane: u32, size: u32) -> u32 {
	let low = seed as u32;
	let high = (seed >> 32) as u32;
	hash32(low ^ hash32(high.wrapping_add(lane.wrapping_mul(0x9e37_79b9)))) % size
}

test_vk!(
	replay_buffer_wraparound_and_seeded_sampling_match_donor_oracle,
	engine,
	{
		let mut replay = ReplayBuffer::new(
			&engine,
			ReplayConfig {
				capacity: 5,
				observation_shape: vec![2],
				action_shape: vec![],
				action_dtype: oa::DType::I32,
			},
		)?;
		let first = transition(
			&engine,
			&[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
			&[0, 1, 2],
			&[10.0, 11.0, 12.0],
			&[0, 0, 1],
			&[0, 1, 0],
		)?;
		let second = transition(
			&engine,
			&[[3.0, 3.5], [4.0, 4.5], [5.0, 5.5], [6.0, 6.5]],
			&[3, 4, 5, 6],
			&[13.0, 14.0, 15.0, 16.0],
			&[0, 1, 0, 1],
			&[1, 0, 1, 0],
		)?;
		let seed = 0x0123_4567_89ab_cdef;
		let (plan, sample) = engine.capture(|| {
			replay.append(&first)?;
			replay.append(&second)?;
			replay.sample(7, seed)
		})?;
		assert!(replay.is_full());
		assert_eq!(replay.len(), 5);
		assert_eq!(replay.cursor(), 2);
		assert_eq!(plan.semantic_graph().operations().len(), 3);
		for operation in &plan.semantic_graph().operations()[..2] {
			assert_eq!(operation.name(), "oa::ml::replay::append_batch");
			assert_eq!(operation.inputs().len(), 12);
			assert_eq!(operation.outputs().len(), 6);
			assert_eq!(operation.mutated_inputs().len(), 6);
			assert_eq!(operation.aliases().len(), 6);
		}
		assert_eq!(
			plan.semantic_graph().operations()[2].name(),
			"oa::ml::replay::sample"
		);
		engine.submit(&plan)?.wait()?;

		let physical_observation = [[5.0, 5.5], [6.0, 6.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]];
		let physical_action = [5_i32, 6, 2, 3, 4];
		let physical_reward = [15.0_f32, 16.0, 12.0, 13.0, 14.0];
		let physical_terminated = [0_u8, 1, 1, 0, 1];
		let physical_truncated = [1_u8, 0, 0, 1, 0];
		let indices = (0..7)
			.map(|lane| source_index(seed, lane, 5))
			.collect::<Vec<_>>();
		assert_eq!(sample.index().read::<u32>()?, indices);
		let expected_observation = indices
			.iter()
			.flat_map(|index| physical_observation[*index as usize])
			.collect::<Vec<_>>();
		let expected_next = expected_observation
			.iter()
			.map(|value| value + 100.0)
			.collect::<Vec<_>>();
		assert_eq!(sample.observation().read_f32()?, expected_observation);
		assert_eq!(sample.next_observation().read_f32()?, expected_next);
		assert_eq!(
			sample.action().read::<i32>()?,
			indices
				.iter()
				.map(|index| physical_action[*index as usize])
				.collect::<Vec<_>>()
		);
		assert_eq!(
			sample.reward().read_f32()?,
			indices
				.iter()
				.map(|index| physical_reward[*index as usize])
				.collect::<Vec<_>>()
		);
		assert_eq!(
			sample.terminated().read::<u8>()?,
			indices
				.iter()
				.map(|index| physical_terminated[*index as usize])
				.collect::<Vec<_>>()
		);
		assert_eq!(
			sample.truncated().read::<u8>()?,
			indices
				.iter()
				.map(|index| physical_truncated[*index as usize])
				.collect::<Vec<_>>()
		);
		Ok(())
	}
);

test_vk!(
	replay_buffer_reset_and_contract_failures_are_explicit,
	engine,
	{
		let mut replay = ReplayBuffer::new(
			&engine,
			ReplayConfig {
				capacity: 3,
				observation_shape: vec![2],
				action_shape: vec![],
				action_dtype: oa::DType::I32,
			},
		)?;
		assert_eq!(
			replay
				.sample(1, 1)
				.err()
				.expect("empty replay sampling was accepted")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		let valid = transition(&engine, &[[1.0, 2.0]], &[1], &[1.0], &[0], &[0])?;
		replay.append(&valid)?;
		assert_eq!(replay.len(), 1);
		assert_eq!(replay.cursor(), 1);
		replay.reset();
		assert!(replay.is_empty());
		assert_eq!(replay.cursor(), 0);
		assert_eq!(
			replay
				.sample(0, 1)
				.err()
				.expect("zero replay sample was accepted")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);

		let invalid = ReplayTransition::new(
			oa::Matrix::from_f32(&engine, [1, 3], &[0.0; 3])?,
			valid.action().clone(),
			valid.next_observation().clone(),
			valid.reward().clone(),
			valid.terminated().clone(),
			valid.truncated().clone(),
		);
		assert_eq!(
			replay
				.append(&invalid)
				.expect_err("invalid replay transition was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert!(replay.is_empty());
		Ok(())
	}
);

test_vk!(replay_buffer_rejects_invalid_configuration, engine, {
	for config in [
		ReplayConfig {
			capacity: 0,
			observation_shape: vec![2],
			action_shape: vec![],
			action_dtype: oa::DType::I32,
		},
		ReplayConfig {
			capacity: 2,
			observation_shape: vec![],
			action_shape: vec![],
			action_dtype: oa::DType::I32,
		},
		ReplayConfig {
			capacity: 2,
			observation_shape: vec![2, 0],
			action_shape: vec![],
			action_dtype: oa::DType::I32,
		},
		ReplayConfig {
			capacity: 2,
			observation_shape: vec![2],
			action_shape: vec![],
			action_dtype: oa::DType::U32,
		},
	] {
		assert_eq!(
			ReplayBuffer::new(&engine, config)
				.err()
				.expect("invalid replay configuration was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});
