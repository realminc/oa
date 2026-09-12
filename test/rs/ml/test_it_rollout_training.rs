use oa::ml::{
	NoOpOptimizer, RolloutBuffer, RolloutConfig, RolloutTransition,
	training::{ItRolloutTraining, ItRolloutTrainingConfig, RolloutTrainingPhase},
};

fn transition(engine: &oa::Engine, value: f32) -> oa::Result<RolloutTransition> {
	Ok(RolloutTransition::new(
		oa::Matrix::from_f32(engine, [1, 1], &[value])?,
		oa::Matrix::from_slice(engine, [1], &[0_i32])?,
		oa::Matrix::from_f32(engine, [1], &[1.0])?,
		oa::Matrix::from_f32(engine, [1], &[0.25])?,
		oa::Matrix::from_f32(engine, [1], &[0.5])?,
		oa::Matrix::from_f32(engine, [1], &[-0.2])?,
		oa::Matrix::from_slice(engine, [1], &[0_u8])?,
		oa::Matrix::from_slice(engine, [1], &[0_u8])?,
	))
}

test_vk!(
	rollout_training_alternates_collection_and_update_epochs,
	engine,
	{
		let mut optimizer = NoOpOptimizer::default();
		let mut training = ItRolloutTraining::new(
			&engine,
			&mut optimizer,
			ItRolloutTrainingConfig {
				rollouts: 1,
				horizon: 1,
				environments: 1,
				update_epochs: 2,
				timer_name: "test_rollout_update".into(),
				enable_gpu_timing: false,
			},
		)?;
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 1,
				environments: 1,
				observation_shape: vec![1],
			},
		)?;
		assert_eq!(training.phase(), RolloutTrainingPhase::Collect);
		training.begin_rollout(&mut rollout)?;
		rollout.append(&transition(&engine, 1.0)?)?;
		training.finalize_rollout(&mut rollout, oa::ml::advantage::GaeConfig::default())?;
		assert_eq!(training.phase(), RolloutTrainingPhase::Update);
		let loss = oa::Matrix::from_f32(&engine, [], &[0.25])?;
		for epoch in 1..=2 {
			assert!(training.begin_update()?);
			training.complete_update(&loss)?;
			assert_eq!(training.update_epoch(), epoch);
		}
		assert!(training.is_done());
		assert_eq!(training.phase(), RolloutTrainingPhase::Complete);
		assert_eq!(training.rollout_index(), 1);
		let snapshot = training.finish()?;
		assert_eq!(snapshot.step_count(), 2);
		Ok(())
	}
);

test_vk!(
	rollout_training_abort_restores_host_collection_state,
	engine,
	{
		let mut optimizer = NoOpOptimizer::default();
		let mut training = ItRolloutTraining::new(
			&engine,
			&mut optimizer,
			ItRolloutTrainingConfig {
				rollouts: 1,
				horizon: 1,
				environments: 1,
				update_epochs: 1,
				timer_name: "test_rollout_abort".into(),
				enable_gpu_timing: false,
			},
		)?;
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 1,
				environments: 1,
				observation_shape: vec![1],
			},
		)?;
		let result: oa::Result<()> = engine
			.capture(|| {
				training.begin_rollout(&mut rollout)?;
				rollout.append(&transition(&engine, 1.0)?)?;
				let integer = oa::Matrix::from_slice(&engine, [1], &[1_i32])?;
				oa::matrix::add(rollout.batch().reward(), &integer).map(|_| ())
			})
			.map(|_| ());
		assert!(result.is_err());
		training.abort_rollout(&mut rollout)?;
		assert_eq!(training.phase(), RolloutTrainingPhase::Collect);
		assert!(rollout.is_empty());
		training.begin_rollout(&mut rollout)?;
		Ok(())
	}
);
