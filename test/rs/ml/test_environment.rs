use oa::ml::{
	CategoricalActorCritic, CategoricalActorCriticConfig, EnvironmentSpace, EnvironmentSpaceKind,
	EnvironmentSpec, EnvironmentTransition, RolloutBuffer, RolloutCollector,
	RolloutCollectorConfig, RolloutConfig,
	environment::{Environment, EnvironmentExecution},
	evaluation::{PolicyEvaluationConfig, evaluate_categorical},
};

struct TestEnvironment<'engine> {
	engine: &'engine oa::Engine,
	execution: EnvironmentExecution,
	spec: EnvironmentSpec,
	environments: u32,
	observation: oa::Matrix,
	committed_observation: oa::Matrix,
	reward: oa::Matrix,
	boundary: oa::Matrix,
}

impl<'engine> TestEnvironment<'engine> {
	fn new(engine: &'engine oa::Engine, environments: u32) -> oa::Result<Self> {
		let spec = cart_pole_spec()?;
		let observation = oa::Matrix::from_f32(
			engine,
			[environments as usize, 4],
			&vec![0.0; environments as usize * 4],
		)?;
		Ok(Self {
			engine,
			execution: EnvironmentExecution::new(engine),
			spec,
			environments,
			committed_observation: observation.clone(),
			observation,
			reward: oa::Matrix::from_f32(
				engine,
				[environments as usize],
				&vec![1.0; environments as usize],
			)?,
			boundary: oa::Matrix::from_slice(
				engine,
				[environments as usize],
				&vec![0_u8; environments as usize],
			)?,
		})
	}
}

impl Environment for TestEnvironment<'_> {
	fn engine(&self) -> &oa::Engine {
		self.engine
	}

	fn spec(&self) -> &EnvironmentSpec {
		&self.spec
	}

	fn environments(&self) -> u32 {
		self.environments
	}

	fn observation(&self) -> &oa::Matrix {
		&self.observation
	}

	fn execution(&self) -> &EnvironmentExecution {
		&self.execution
	}

	fn execution_mut(&mut self) -> &mut EnvironmentExecution {
		&mut self.execution
	}

	fn record_reset(&mut self, seed: u64) -> oa::Result<()> {
		self.observation = oa::matrix::add_scalar(&self.observation, seed as f32)?;
		Ok(())
	}

	fn record_step(&mut self, _action: &oa::Matrix) -> oa::Result<EnvironmentTransition> {
		let observation = self.observation.clone();
		self.observation = oa::matrix::add_scalar(&self.observation, 1.0)?;
		Ok(EnvironmentTransition::new(
			observation,
			self.observation.clone(),
			self.reward.clone(),
			self.boundary.clone(),
			self.boundary.clone(),
		))
	}

	fn record_reset_completed(&mut self) -> oa::Result<()> {
		self.observation = oa::matrix::add_scalar(&self.observation, 0.0)?;
		Ok(())
	}

	fn commit_recorded_state(&mut self) {
		self.committed_observation = self.observation.clone();
	}

	fn rollback_recorded_state(&mut self) {
		self.observation = self.committed_observation.clone();
	}
}

fn cart_pole_spec() -> oa::Result<EnvironmentSpec> {
	EnvironmentSpec::new(
		EnvironmentSpace::continuous(
			"observation",
			[4],
			oa::DType::F32,
			f64::NEG_INFINITY,
			f64::INFINITY,
		)?,
		EnvironmentSpace::discrete("action", 2, oa::DType::I32)?,
		EnvironmentSpace::continuous(
			"reward",
			[],
			oa::DType::F32,
			f64::NEG_INFINITY,
			f64::INFINITY,
		)?,
		EnvironmentSpace::binary("terminated", [])?,
		EnvironmentSpace::binary("truncated", [])?,
	)
}

#[test]
fn environment_spaces_preserve_donor_shape_and_dtype_contracts() -> oa::Result<()> {
	let observation =
		EnvironmentSpace::continuous("observation", [4], oa::DType::F32, -10.0, 10.0)?;
	assert_eq!(observation.kind(), EnvironmentSpaceKind::Box);
	assert_eq!(observation.name(), "observation");
	assert_eq!(observation.shape(), [4]);
	assert_eq!(observation.elements_per_environment(), 4);
	assert_eq!(observation.batched_shape(8)?, [8, 4]);
	let action = EnvironmentSpace::discrete("action", 3, oa::DType::U32)?;
	assert_eq!(action.kind(), EnvironmentSpaceKind::Discrete);
	assert!(action.shape().is_empty());
	assert_eq!(action.minimum(), 0.0);
	assert_eq!(action.maximum(), 2.0);
	assert_eq!(action.cardinality(), 3);
	let terminated = EnvironmentSpace::binary("terminated", [])?;
	assert_eq!(terminated.dtype(), oa::DType::U8);
	assert_eq!(terminated.batched_shape(8)?, [8]);

	for invalid in [
		EnvironmentSpace::continuous("", [4], oa::DType::F32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [0], oa::DType::F32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::I32, -1.0, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::F32, f64::NAN, 1.0),
		EnvironmentSpace::continuous("box", [4], oa::DType::F32, 2.0, 1.0),
		EnvironmentSpace::discrete("action", 0, oa::DType::I32),
		EnvironmentSpace::discrete("action", 2, oa::DType::F32),
	] {
		assert_eq!(
			invalid
				.expect_err("invalid environment space was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	assert_eq!(
		observation
			.batched_shape(0)
			.expect_err("zero environment count was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	let shaped_reward = EnvironmentSpace::continuous(
		"reward",
		[1],
		oa::DType::F32,
		f64::NEG_INFINITY,
		f64::INFINITY,
	)?;
	assert_eq!(
		EnvironmentSpec::new(
			observation,
			action,
			shaped_reward,
			terminated.clone(),
			terminated,
		)
		.expect_err("vector reward was accepted")
		.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}

test_vk!(
	environment_spec_validates_complete_batched_transition,
	engine,
	{
		let spec = cart_pole_spec()?;
		let observation = oa::Matrix::from_f32(&engine, [3, 4], &[0.0; 12])?;
		let next_observation = oa::Matrix::from_f32(&engine, [3, 4], &[0.1; 12])?;
		let action = oa::Matrix::from_slice(&engine, [3], &[0_i32, 1, 0])?;
		let reward = oa::Matrix::from_f32(&engine, [3], &[1.0; 3])?;
		let terminated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 1, 0])?;
		let truncated = oa::Matrix::from_slice(&engine, [3], &[0_u8, 0, 1])?;
		let transition = EnvironmentTransition::new(
			observation.clone(),
			next_observation,
			reward,
			terminated,
			truncated,
		);
		spec.validate_reset(&observation, 3)?;
		spec.validate_action(&action, 3)?;
		spec.validate_transition(&action, &transition, 3)?;
		assert_eq!(transition.observation().shape(), [3, 4]);
		assert_eq!(transition.next_observation().shape(), [3, 4]);
		assert_eq!(transition.reward().shape(), [3]);
		assert_eq!(transition.terminated().read::<u8>()?, [0, 1, 0]);
		assert_eq!(transition.truncated().read::<u8>()?, [0, 0, 1]);

		let wrong_action = oa::Matrix::from_slice(&engine, [3], &[0_u32, 1, 0])?;
		assert_eq!(
			spec.validate_action(&wrong_action, 3)
				.expect_err("wrong action dtype was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_observation = oa::Matrix::from_f32(&engine, [2, 4], &[0.0; 8])?;
		assert_eq!(
			spec.validate_reset(&wrong_observation, 3)
				.expect_err("wrong observation batch was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

test_vk!(environment_transforms_match_donor_values, engine, {
	let observation = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 4.0, 5.0, -8.0])?;
	let mean = oa::Matrix::from_f32(&engine, [2], &[1.0, 0.0])?;
	let stddev = oa::Matrix::from_f32(&engine, [2], &[2.0, 2.0])?;
	let normalized =
		oa::ml::environment::normalize_observation(&observation, &mean, &stddev, 1.0e-6, 3.0)?;
	let action = oa::Matrix::from_f32(&engine, [3], &[-2.0, 0.0, 2.0])?;
	let scaled = oa::ml::environment::scale_action(&action, -1.0, 1.0, 0.0, 10.0, true)?;
	let reward = oa::Matrix::from_f32(&engine, [3], &[-3.0, 0.25, 4.0])?;
	let clipped = oa::ml::environment::clip_reward(&reward, -1.0, 1.0)?;
	for (actual, expected) in normalized
		.read_f32()?
		.into_iter()
		.zip([0.0, 2.0, 2.0, -3.0])
	{
		assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
	}
	assert_eq!(scaled.read_f32()?, [0.0, 5.0, 10.0]);
	assert_eq!(clipped.read_f32()?, [-1.0, 0.25, 1.0]);
	Ok(())
});

test_vk!(environment_transforms_reject_invalid_policy, engine, {
	let value = oa::Matrix::from_f32(&engine, [2], &[0.0, 1.0])?;
	assert!(oa::ml::environment::normalize_observation(&value, &value, &value, 0.0, 1.0).is_err());
	assert!(oa::ml::environment::scale_action(&value, 1.0, 1.0, 0.0, 1.0, true).is_err());
	assert!(oa::ml::environment::clip_reward(&value, 1.0, -1.0).is_err());
	Ok(())
});

test_vk!(
	environment_execution_owns_exact_transaction_lifecycle,
	engine,
	{
		let mut environment = TestEnvironment::new(&engine, 2)?;
		assert!(environment.execution().is_open());
		assert_eq!(environment.execution().submission_count(), 0);
		environment.reset(7)?;
		assert!(environment.execution().has_active_recording());
		let action = oa::Matrix::from_slice(&engine, [2], &[0_i32, 1])?;
		let transition = environment.step(&action)?;
		assert_eq!(transition.next_observation().shape(), [2, 4]);
		environment.reset_completed()?;
		let event = environment.submit()?;
		assert!(environment.execution().has_pending_event());
		assert_eq!(environment.execution().submission_count(), 1);
		assert!(environment.begin().is_err());
		environment.wait(&event)?;
		assert!(!environment.execution().has_pending_event());
		assert_eq!(environment.observation().read_f32()?, vec![8.0; 8]);

		environment.reset(3)?;
		assert!(environment.execution().has_active_recording());
		environment.cancel()?;
		assert!(!environment.execution().has_active_recording());
		assert_eq!(environment.observation().read_f32()?, vec![8.0; 8]);
		environment.close()?;
		environment.close()?;
		assert!(!environment.execution().is_open());
		assert!(environment.begin().is_err());
		Ok(())
	}
);

test_vk!(
	rollout_collector_submits_one_complete_environment_transaction,
	engine,
	{
		let mut environment = TestEnvironment::new(&engine, 2)?;
		let model = CategoricalActorCritic::new(
			&engine,
			CategoricalActorCriticConfig {
				observation_size: 4,
				action_count: 2,
				hidden_size: 8,
				seed: 19,
			},
		)?;
		let mut rollout = RolloutBuffer::new(
			&engine,
			RolloutConfig {
				time: 2,
				environments: 2,
				observation_shape: vec![4],
			},
		)?;
		let (event, metrics) = {
			let mut collector = RolloutCollector::new(
				&mut environment,
				&model,
				RolloutCollectorConfig {
					horizon: 2,
					seed: 5,
					..RolloutCollectorConfig::default()
				},
			)?;
			let event = collector.collect(&mut rollout)?;
			(event, collector.metrics())
		};
		assert_eq!(metrics.collections, 1);
		assert_eq!(metrics.environment_steps, 2);
		assert_eq!(metrics.transitions, 4);
		assert!(environment.execution().has_pending_event());
		environment.wait(&event)?;
		assert_eq!(rollout.batch().reward().read_f32()?, [1.0; 4]);
		assert_eq!(
			rollout.batch().observation().read_f32()?,
			[vec![5.0; 8], vec![6.0; 8]].concat()
		);
		assert_eq!(environment.observation().read_f32()?, vec![7.0; 8]);
		assert!(rollout.is_finalized());
		environment.close()?;
		Ok(())
	}
);

test_vk!(
	policy_evaluator_uses_one_exact_environment_completion,
	engine,
	{
		let mut environment = TestEnvironment::new(&engine, 2)?;
		let model = CategoricalActorCritic::new(
			&engine,
			CategoricalActorCriticConfig {
				observation_size: 4,
				action_count: 2,
				hidden_size: 8,
				seed: 23,
			},
		)?;
		let metrics = evaluate_categorical(
			&mut environment,
			&model,
			PolicyEvaluationConfig {
				horizon: 2,
				seed: 7,
			},
		)?;
		assert_eq!(metrics.environment_steps, 2);
		assert_eq!(metrics.transitions, 4);
		assert_eq!(metrics.completed_episodes, 0);
		assert_eq!(metrics.mean_completed_return, 0.0);
		assert_eq!(environment.execution().submission_count(), 1);
		assert!(!environment.execution().has_pending_event());
		environment.close()?;
		Ok(())
	}
);
