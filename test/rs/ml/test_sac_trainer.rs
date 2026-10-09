use oa::ml::{
	Module, ReplayBuffer, ReplayConfig, ReplayTransition, Sgd,
	training::{SacTrainer, SacTrainerConfig},
};

fn linear(
	engine: &oa::Engine,
	input: usize,
	output: usize,
	weights: &[f32],
) -> oa::Result<oa::ml::nn::Linear> {
	oa::ml::nn::Linear::from_matrices(
		oa::Matrix::from_f32(engine, [output, input], weights)?,
		oa::Matrix::from_f32(engine, [output], &vec![0.0; output])?,
	)
}

test_vk!(sac_trainer_runs_donor_actor_twin_critic_update, engine, {
	let mut replay = ReplayBuffer::new(
		&engine,
		ReplayConfig {
			capacity: 8,
			observation_shape: vec![2],
			action_shape: vec![1],
			action_dtype: oa::DType::F32,
		},
	)?;
	replay.append(&ReplayTransition::new(
		oa::Matrix::from_f32(&engine, [4, 2], &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?,
		oa::Matrix::from_f32(&engine, [4, 1], &[-0.5, 0.25, 0.75, -0.25])?,
		oa::Matrix::from_f32(&engine, [4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])?,
		oa::Matrix::from_f32(&engine, [4], &[0.0, 1.0, 1.0, 0.0])?,
		oa::Matrix::from_slice(&engine, [4], &[0_u8, 0, 1, 0])?,
		oa::Matrix::from_slice(&engine, [4], &[0_u8, 0, 0, 1])?,
	))?;

	let actor = linear(&engine, 2, 2, &[0.1, -0.2, 0.05, 0.1])?;
	let critic1 = linear(&engine, 3, 1, &[0.2, -0.1, 0.3])?;
	let critic2 = linear(&engine, 3, 1, &[-0.1, 0.25, 0.15])?;
	let target_critic1 = linear(&engine, 3, 1, &[9.0; 3])?;
	let target_critic2 = linear(&engine, 3, 1, &[8.0; 3])?;
	let critic1_weight = critic1.weight();
	let critic2_weight = critic2.weight();
	let target1_weight = target_critic1.weight();
	let target2_weight = target_critic2.weight();
	let mut actor_optimizer = Sgd::new(actor.all_parameters()?, 1.0e-3, 0.0, 0.0)?;
	let mut critic_parameters = critic1.all_parameters()?;
	critic_parameters.extend(critic2.all_parameters()?);
	let mut critic_optimizer = Sgd::new(critic_parameters, 1.0e-3, 0.0, 0.0)?;

	let mut trainer = SacTrainer::new(
		&engine,
		&actor,
		&critic1,
		&critic2,
		&target_critic1,
		&target_critic2,
		&mut actor_optimizer,
		&mut critic_optimizer,
		&replay,
		SacTrainerConfig {
			updates: 1,
			batch_size: 4,
			action_dimensions: 1,
			target_update_interval: 1,
			observation_shape: vec![2],
			action_minimum: -1.0,
			action_maximum: 1.0,
			seed: 123,
			loss: oa::ml::loss::SacLossConfig::default(),
		},
	)?;
	assert_eq!(
		target1_weight.data().read_f32()?,
		critic1_weight.data().read_f32()?
	);
	assert_eq!(
		target2_weight.data().read_f32()?,
		critic2_weight.data().read_f32()?
	);
	assert!(!target1_weight.requires_grad());
	assert!(!target2_weight.requires_grad());

	assert!(trainer.update()?);
	assert!(trainer.is_done());
	assert!(!trainer.update()?);
	let metrics = trainer.metrics();
	assert_eq!(metrics.update, 1);
	assert!(metrics.actor_loss.is_finite());
	assert!(metrics.critic_loss.is_finite());
	assert_eq!(
		target1_weight.data().read_f32()?,
		critic1_weight.data().read_f32()?
	);
	assert_eq!(
		target2_weight.data().read_f32()?,
		critic2_weight.data().read_f32()?
	);
	let (critic_summary, actor_summary) = trainer.finish()?;
	assert_eq!(critic_summary.step_count(), 1);
	assert_eq!(actor_summary.step_count(), 1);
	assert_eq!(actor_optimizer.step_count(), 1);
	assert_eq!(critic_optimizer.step_count(), 1);
	Ok(())
});

test_vk!(sac_trainer_rejects_categorical_replay, engine, {
	let replay = ReplayBuffer::new(
		&engine,
		ReplayConfig {
			capacity: 4,
			observation_shape: vec![2],
			action_shape: vec![],
			action_dtype: oa::DType::I32,
		},
	)?;
	let actor = linear(&engine, 2, 2, &[0.0; 4])?;
	let critic1 = linear(&engine, 3, 1, &[0.0; 3])?;
	let critic2 = linear(&engine, 3, 1, &[0.0; 3])?;
	let target1 = linear(&engine, 3, 1, &[0.0; 3])?;
	let target2 = linear(&engine, 3, 1, &[0.0; 3])?;
	let mut actor_optimizer = Sgd::new(actor.all_parameters()?, 1.0e-3, 0.0, 0.0)?;
	let mut critic_parameters = critic1.all_parameters()?;
	critic_parameters.extend(critic2.all_parameters()?);
	let mut critic_optimizer = Sgd::new(critic_parameters, 1.0e-3, 0.0, 0.0)?;
	let error = match SacTrainer::new(
		&engine,
		&actor,
		&critic1,
		&critic2,
		&target1,
		&target2,
		&mut actor_optimizer,
		&mut critic_optimizer,
		&replay,
		SacTrainerConfig {
			updates: 1,
			batch_size: 1,
			action_dimensions: 1,
			target_update_interval: 1,
			observation_shape: vec![2],
			action_minimum: -1.0,
			action_maximum: 1.0,
			seed: 0,
			loss: oa::ml::loss::SacLossConfig::default(),
		},
	) {
		Ok(_) => panic!("categorical replay was accepted by SAC"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
});
