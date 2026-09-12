use oa::ml::{
	Adam, CategoricalActorCritic, CategoricalActorCriticConfig, Module,
	training::{PpoTrainer, PpoTrainerConfig, RolloutTrainingPhase},
};

fn model(engine: &oa::Engine, seed: u64) -> oa::Result<CategoricalActorCritic> {
	CategoricalActorCritic::new(
		engine,
		CategoricalActorCriticConfig {
			observation_size: 2,
			action_count: 2,
			hidden_size: 8,
			seed,
		},
	)
}

fn config(seed: u64) -> PpoTrainerConfig {
	PpoTrainerConfig {
		rollouts: 1,
		horizon: 1,
		environments: 2,
		update_epochs: 1,
		observation_shape: vec![2],
		seed,
		..PpoTrainerConfig::default()
	}
}

fn collect_one(engine: &oa::Engine, trainer: &mut PpoTrainer<'_, '_>) -> oa::Result<()> {
	trainer.begin_collection()?;
	let observation = oa::Matrix::from_f32(engine, [2, 2], &[0.1, -0.2, 0.3, 0.4])?;
	let next_observation = oa::Matrix::from_f32(engine, [2, 2], &[0.2, -0.1, 0.4, 0.5])?;
	let reward = oa::Matrix::from_f32(engine, [2], &[1.0, 0.5])?;
	let terminated = oa::Matrix::from_slice(engine, [2], &[0_u8, 0])?;
	let truncated = oa::Matrix::from_slice(engine, [2], &[0_u8, 0])?;
	let policy = trainer.act(&observation)?;
	assert_eq!(policy.action.shape(), [2]);
	assert_eq!(policy.value.shape(), [2]);
	trainer.observe(
		&observation,
		&next_observation,
		&reward,
		&terminated,
		&truncated,
		&policy,
	)?;
	trainer.end_collection()
}

test_vk!(
	ppo_trainer_completes_the_donor_environment_neutral_gate,
	engine,
	{
		let model = model(&engine, 17)?;
		let mut optimizer = Adam::new(model.all_parameters()?, 1.0e-3)?;
		let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, config(123))?;
		assert!(trainer.needs_collection());
		collect_one(&engine, &mut trainer)?;
		assert_eq!(trainer.phase(), RolloutTrainingPhase::Update);
		assert!(trainer.update()?);
		assert!(trainer.is_done());
		let metrics = trainer.metrics();
		assert_eq!(metrics.rollout, 1);
		assert_eq!(metrics.update_epoch, 1);
		assert!(metrics.total_loss.is_finite());
		assert!(metrics.policy_loss.is_finite());
		assert!(metrics.value_loss.is_finite());
		assert!(metrics.entropy.is_finite());
		assert_eq!(trainer.batch().observation().shape(), [1, 2, 2]);
		let snapshot = trainer.finish()?;
		assert_eq!(snapshot.step_count(), 1);
		assert_eq!(optimizer.step_count(), 1);
		assert!(
			model
				.all_parameters()?
				.into_iter()
				.any(|parameter| parameter.gradient().is_some())
		);
		Ok(())
	}
);

test_vk!(
	ppo_collection_abort_restores_capture_and_collect_phase,
	engine,
	{
		let model = model(&engine, 29)?;
		let mut optimizer = Adam::new(model.all_parameters()?, 1.0e-3)?;
		let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, config(321))?;
		assert!(trainer.abort_collection().is_err());
		let result: oa::Result<()> = engine
			.capture(|| {
				collect_one(&engine, &mut trainer)?;
				let integer = oa::Matrix::from_slice(&engine, [1], &[1_i32])?;
				oa::matrix::add(trainer.batch().reward(), &integer).map(|_| ())
			})
			.map(|_| ());
		assert!(result.is_err());
		trainer.abort_collection()?;
		assert!(trainer.needs_collection());
		assert_eq!(trainer.phase(), RolloutTrainingPhase::Collect);
		collect_one(&engine, &mut trainer)?;
		assert!(trainer.update()?);
		let snapshot = trainer.finish()?;
		assert_eq!(snapshot.step_count(), 1);
		assert_eq!(optimizer.step_count(), 1);
		Ok(())
	}
);
