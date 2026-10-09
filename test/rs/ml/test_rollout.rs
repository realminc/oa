use oa::ml::advantage::GaeConfig;
use oa::ml::{RolloutBuffer, RolloutConfig, RolloutTransition};

#[allow(
	clippy::too_many_arguments,
	reason = "the oracle fixture names every donor transition field at each time step"
)]
fn transition(
	engine: &oa::Engine,
	observation: &[f32; 4],
	action: &[i32; 2],
	reward: &[f32; 2],
	value: &[f32; 2],
	next_value: &[f32; 2],
	log_probability: &[f32; 2],
	terminated: &[u8; 2],
	truncated: &[u8; 2],
) -> oa::Result<RolloutTransition> {
	Ok(RolloutTransition::new(
		oa::Matrix::from_f32(engine, [2, 2], observation)?,
		oa::Matrix::from_slice(engine, [2], action)?,
		oa::Matrix::from_f32(engine, [2], reward)?,
		oa::Matrix::from_f32(engine, [2], value)?,
		oa::Matrix::from_f32(engine, [2], next_value)?,
		oa::Matrix::from_f32(engine, [2], log_probability)?,
		oa::Matrix::from_slice(engine, [2], terminated)?,
		oa::Matrix::from_slice(engine, [2], truncated)?,
	))
}

fn assert_close(actual: &[f32], expected: &[f32]) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= 1.0e-6,
			"value {index}: expected {expected}, found {actual}"
		);
	}
}

test_vk!(
	rollout_buffer_preserves_time_major_fields_and_finalizes_in_place,
	engine,
	{
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 3,
				environments: 2,
				observation_shape: vec![2],
			},
		)?;
		let transitions = [
			transition(
				&engine,
				&[0.0, 1.0, 10.0, 11.0],
				&[0, 1],
				&[1.0, 0.5],
				&[0.4, 0.1],
				&[0.3, 0.5],
				&[-0.2, -0.4],
				&[0, 0],
				&[0, 0],
			)?,
			transition(
				&engine,
				&[2.0, 3.0, 12.0, 13.0],
				&[1, 0],
				&[0.2, 1.0],
				&[0.3, 0.5],
				&[99.0, 4.0],
				&[-0.3, -0.1],
				&[1, 0],
				&[0, 1],
			)?,
			transition(
				&engine,
				&[4.0, 5.0, 14.0, 15.0],
				&[0, 1],
				&[2.0, 0.3],
				&[0.8, 0.2],
				&[0.6, 0.9],
				&[-0.5, -0.6],
				&[0, 0],
				&[0, 0],
			)?,
		];
		let config = GaeConfig {
			gamma: 0.9,
			lambda: 0.8,
		};
		let (plan, ()) = engine.capture(|| {
			for transition in &transitions {
				rollout.append(transition)?;
			}
			rollout.finalize(config)
		})?;
		assert!(rollout.is_full());
		assert!(rollout.is_finalized());
		assert_eq!(rollout.len(), 3);
		assert_eq!(rollout.capacity(), 3);
		assert_eq!(plan.semantic_graph().operations().len(), 4);
		for operation in &plan.semantic_graph().operations()[..3] {
			assert_eq!(operation.name(), "oa::ml::rollout::append");
			assert_eq!(operation.inputs().len(), 17);
			assert_eq!(operation.outputs().len(), 9);
			assert_eq!(operation.mutated_inputs().len(), 9);
			assert_eq!(operation.aliases().len(), 9);
		}
		assert_eq!(
			plan.semantic_graph().operations()[3].name(),
			"oa::ml::advantage::gae"
		);
		engine.submit(&plan)?.wait()?;

		let batch = rollout.batch();
		assert_eq!(batch.observation().shape(), [3, 2, 2]);
		assert_close(
			&batch.observation().read_f32()?,
			&[
				0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0, 4.0, 5.0, 14.0, 15.0,
			],
		);
		assert_eq!(batch.action().read::<i32>()?, [0, 1, 1, 0, 0, 1]);
		assert_close(&batch.reward().read_f32()?, &[1.0, 0.5, 0.2, 1.0, 2.0, 0.3]);
		assert_close(&batch.value().read_f32()?, &[0.4, 0.1, 0.3, 0.5, 0.8, 0.2]);
		assert_close(
			&batch.next_value().read_f32()?,
			&[0.3, 0.5, 99.0, 4.0, 0.6, 0.9],
		);
		assert_close(
			&batch.old_log_probability().read_f32()?,
			&[-0.2, -0.4, -0.3, -0.1, -0.5, -0.6],
		);
		assert_eq!(batch.terminated().read::<u8>()?, [0, 0, 1, 0, 0, 0]);
		assert_eq!(batch.truncated().read::<u8>()?, [0, 0, 0, 1, 0, 0]);
		assert_eq!(batch.valid().read::<u8>()?, [1, 1, 1, 1, 1, 1]);

		let reward = batch.reward().read_f32()?;
		let value = batch.value().read_f32()?;
		let next_value = batch.next_value().read_f32()?;
		let terminated = batch.terminated().read::<u8>()?;
		let truncated = batch.truncated().read::<u8>()?;
		let mut expected_advantage = vec![0.0_f32; 6];
		let mut expected_return = vec![0.0_f32; 6];
		for environment in 0..2 {
			let mut next_advantage = 0.0;
			for time in (0..3).rev() {
				let index = time * 2 + environment;
				let bootstrap = if terminated[index] == 0 { 1.0 } else { 0.0 };
				let trace = if terminated[index] == 0 && truncated[index] == 0 {
					1.0
				} else {
					0.0
				};
				let delta = reward[index] + config.gamma * bootstrap * next_value[index] - value[index];
				let advantage = delta + config.gamma * config.lambda * trace * next_advantage;
				expected_advantage[index] = advantage;
				expected_return[index] = advantage + value[index];
				next_advantage = advantage;
			}
		}
		assert_close(&batch.advantage().read_f32()?, &expected_advantage);
		assert_close(&batch.returns().read_f32()?, &expected_return);
		Ok(())
	}
);

test_vk!(
	rollout_buffer_reuses_storage_and_enforces_lifecycle,
	engine,
	{
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 2,
				environments: 2,
				observation_shape: vec![2],
			},
		)?;
		assert!(rollout.is_empty());
		assert_eq!(
			rollout
				.finalize(GaeConfig::default())
				.expect_err("incomplete rollout was finalized")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		let first = transition(
			&engine,
			&[1.0, 2.0, 3.0, 4.0],
			&[0, 1],
			&[1.0, 1.0],
			&[0.0, 0.0],
			&[0.0, 0.0],
			&[0.0, 0.0],
			&[0, 0],
			&[0, 0],
		)?;
		let second = transition(
			&engine,
			&[5.0, 6.0, 7.0, 8.0],
			&[1, 0],
			&[2.0, 3.0],
			&[0.1, 0.2],
			&[0.3, 0.4],
			&[-0.1, -0.2],
			&[0, 1],
			&[1, 0],
		)?;
		rollout.append(&first)?;
		rollout.append(&second)?;
		assert_eq!(
			rollout
				.append(&first)
				.expect_err("rollout capacity overflow was accepted")
				.kind(),
			oa::ErrorKind::ResourceExhausted
		);
		rollout.finalize(GaeConfig::default())?;
		rollout.finalize(GaeConfig::default())?;
		assert_eq!(
			rollout
				.append(&first)
				.expect_err("append after finalize was accepted")
				.kind(),
			oa::ErrorKind::FailedPrecondition
		);
		let observation_before = rollout.batch().observation().read_f32()?;
		rollout.batch().valid().read::<u8>()?;
		rollout.reset()?;
		assert!(rollout.is_empty());
		assert!(!rollout.is_finalized());
		rollout.append(&first)?;
		assert_eq!(rollout.batch().valid().read::<u8>()?, [1, 1, 0, 0]);
		assert_close(
			&rollout.batch().observation().read_f32()?,
			&observation_before,
		);

		let mut alias_rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 1,
				environments: 2,
				observation_shape: vec![2],
			},
		)?;
		let aliased_observation = alias_rollout.batch().observation().reshape([2, 2])?;
		let aliased = RolloutTransition::new(
			aliased_observation,
			first.action().clone(),
			first.reward().clone(),
			first.value().clone(),
			first.next_value().clone(),
			first.log_probability().clone(),
			first.terminated().clone(),
			first.truncated().clone(),
		);
		assert_eq!(
			alias_rollout
				.append(&aliased)
				.expect_err("transition aliasing rollout storage was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(
	rollout_buffer_rejects_invalid_configuration_and_transition,
	engine,
	{
		for config in [
			RolloutConfig {
				time: 0,
				environments: 2,
				observation_shape: vec![4],
			},
			RolloutConfig {
				time: 2,
				environments: 0,
				observation_shape: vec![4],
			},
			RolloutConfig {
				time: 2,
				environments: 2,
				observation_shape: vec![],
			},
			RolloutConfig {
				time: 2,
				environments: 2,
				observation_shape: vec![4, 0],
			},
		] {
			assert_eq!(
				RolloutBuffer::new(&engine, config)
					.err()
					.expect("invalid rollout configuration was accepted")
					.kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 2,
				environments: 2,
				observation_shape: vec![2],
			},
		)?;
		let invalid = RolloutTransition::new(
			oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?,
			oa::Matrix::from_slice(&engine, [2], &[0_i32; 2])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0; 2])?,
			oa::Matrix::from_slice(&engine, [2], &[0_u8; 2])?,
			oa::Matrix::from_slice(&engine, [2], &[0_u8; 2])?,
		);
		assert_eq!(
			rollout
				.append(&invalid)
				.expect_err("invalid transition was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert!(rollout.is_empty());
		Ok(())
	}
);
