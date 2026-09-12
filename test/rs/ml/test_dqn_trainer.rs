use oa::ml::{
	Module, ReplayBuffer, ReplayConfig, ReplayTransition, Sgd,
	training::{DqnTrainer, DqnTrainerConfig},
};

fn linear(engine: &oa::Engine, weight: [f32; 4], bias: [f32; 2]) -> oa::Result<oa::ml::nn::Linear> {
	oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(engine, [2, 2], &weight)?,
		oa::Matrix::from_f32(engine, [2], &bias)?,
	)
}

test_vk!(
	dqn_trainer_matches_donor_update_and_target_cadence,
	engine,
	{
		let mut replay = ReplayBuffer::new(
			&engine,
			ReplayConfig {
				capacity: 8,
				observation_shape: vec![2],
				action_shape: vec![],
				action_dtype: oa::DType::I32,
			},
		)?;
		replay.append(&ReplayTransition::new(
			oa::Matrix::from_f32(&engine, [4, 2], &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?,
			oa::Matrix::from_slice(&engine, [4], &[0_i32, 1, 0, 1])?,
			oa::Matrix::from_f32(&engine, [4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])?,
			oa::Matrix::from_f32(&engine, [4], &[0.0, 1.0, 1.0, 0.0])?,
			oa::Matrix::from_slice(&engine, [4], &[0_u8, 0, 1, 0])?,
			oa::Matrix::from_slice(&engine, [4], &[0_u8, 0, 0, 1])?,
		))?;

		let online = linear(&engine, [0.2, -0.1, -0.3, 0.4], [0.0, 0.0])?;
		let target = linear(&engine, [9.0; 4], [7.0; 2])?;
		let online_weight = online.weight();
		let target_weight = target.weight();
		let mut optimizer = Sgd::new(online.all_parameters()?, 1.0e-3, 0.0, 0.0)?;
		let mut trainer = DqnTrainer::new(
			&engine,
			&online,
			&target,
			&mut optimizer,
			&replay,
			DqnTrainerConfig {
				updates: 2,
				batch_size: 4,
				target_update_interval: 1,
				observation_shape: vec![2],
				seed: 77,
				loss: oa::ml::loss::DqnLossConfig { discount: 0.95 },
			},
		)?;
		assert_eq!(
			target_weight.data().read_f32()?,
			online_weight.data().read_f32()?
		);
		assert!(!target_weight.requires_grad());

		assert!(trainer.update()?);
		assert_eq!(trainer.metrics().update, 1);
		assert!(trainer.metrics().loss.is_finite());
		assert!(trainer.update()?);
		assert!(trainer.is_done());
		assert!(!trainer.update()?);
		assert_eq!(trainer.metrics().update, 2);
		assert!(trainer.metrics().loss.is_finite());
		assert_eq!(
			target_weight.data().read_f32()?,
			online_weight.data().read_f32()?
		);
		let summary = trainer.finish()?;
		assert_eq!(summary.step_count(), 2);
		assert_eq!(optimizer.step_count(), 2);
		Ok(())
	}
);

test_vk!(
	dqn_trainer_rejects_replay_and_module_contract_drift,
	engine,
	{
		let replay = ReplayBuffer::new(
			&engine,
			ReplayConfig {
				capacity: 4,
				observation_shape: vec![2],
				action_shape: vec![1],
				action_dtype: oa::DType::F32,
			},
		)?;
		let online = linear(&engine, [0.0; 4], [0.0; 2])?;
		let target = linear(&engine, [0.0; 4], [0.0; 2])?;
		let mut optimizer = Sgd::new(online.all_parameters()?, 1.0e-3, 0.0, 0.0)?;
		let error = match DqnTrainer::new(
			&engine,
			&online,
			&target,
			&mut optimizer,
			&replay,
			DqnTrainerConfig {
				updates: 1,
				batch_size: 1,
				target_update_interval: 1,
				observation_shape: vec![2],
				seed: 0,
				loss: oa::ml::loss::DqnLossConfig::default(),
			},
		) {
			Ok(_) => panic!("invalid DQN replay contract was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		Ok(())
	}
);
