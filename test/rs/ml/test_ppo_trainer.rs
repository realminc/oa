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
	for _ in 0..trainer.config().horizon {
		let observation = oa::Matrix::new_from_f32(engine, [2, 2], &[0.1, -0.2, 0.3, 0.4])?;
		let next_observation = oa::Matrix::new_from_f32(engine, [2, 2], &[0.2, -0.1, 0.4, 0.5])?;
		let reward = oa::Matrix::new_from_f32(engine, [2], &[1.0, 0.5])?;
		let terminated = oa::Matrix::new_from_slice(engine, [2], &[0_u8, 0])?;
		let truncated = oa::Matrix::new_from_slice(engine, [2], &[0_u8, 0])?;
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
	}

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
				let integer = oa::Matrix::new_from_slice(&engine, [1], &[1_i32])?;
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

test_vk!(
	ppo_model_auxiliary_loss_is_composed_under_the_update_tape,
	engine,
	{
		struct WithAuxiliary {
			base: CategoricalActorCritic,
			calls: std::cell::Cell<u32>,
		}
		impl Module for WithAuxiliary {
			fn registry(&self) -> &oa::ml::ModuleRegistry {
				self.base.registry()
			}
			fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.base.forward(input)
			}
		}
		impl oa::ml::ActorCritic for WithAuxiliary {
			fn evaluate(&self, input: &oa::Matrix) -> oa::Result<oa::ml::ActorCriticOutput> {
				self.base.evaluate(input)
			}
			fn auxiliary_loss(&self, task: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.calls.set(self.calls.get() + 1);
				oa::matrix::add_scalar(task, 1.0)
			}
		}
		let model = WithAuxiliary {
			base: model(&engine, 17)?,
			calls: std::cell::Cell::new(0),
		};
		let mut optimizer = Adam::new(model.all_parameters()?, 1.0e-3)?;
		let c = config(123);
		let coefficients = c.loss;
		let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, c)?;
		collect_one(&engine, &mut trainer)?;
		trainer.update()?;
		let metrics = trainer.metrics();
		let expected = metrics.policy_loss + coefficients.value_coefficient * metrics.value_loss
			- coefficients.entropy_coefficient * metrics.entropy
			+ 1.0;
		assert!((metrics.total_loss - expected).abs() < 1e-5);
		assert_eq!(model.calls.get(), 1);
		trainer.finish()?;
		assert_eq!(optimizer.step_count(), 1);
		Ok(())
	}
);

test_vk!(
	ppo_minibatch_accumulation_matches_full_batch_with_tail,
	engine,
	{
		use oa::ml::Sgd;
		let mut outcomes = Vec::new();
		for minibatch in [0, 1, 4, 6] {
			let model = model(&engine, 72)?;
			let mut optimizer = Sgd::new(model.all_parameters()?, 0.005, 0.0, 0.0)?;
			let mut c = config(431);
			c.horizon = 3;
			c.update_epochs = 2;
			c.update_minibatch_size = minibatch;
			let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, c)?;
			collect_one(&engine, &mut trainer)?;
			while !trainer.is_done() {
				trainer.update()?;
			}
			let metrics = trainer.metrics();
			let snapshot = trainer.finish()?;
			assert_eq!(snapshot.step_count(), 2);
			assert!(metrics.total_loss.is_finite());
			let values = model
				.all_parameters()?
				.into_iter()
				.map(|p| p.data().read_f32())
				.collect::<oa::Result<Vec<_>>>()?
				.concat();
			outcomes.push((metrics, values));
		}
		for (metrics, values) in &outcomes[1..] {
			assert!((metrics.total_loss - outcomes[0].0.total_loss).abs() < 2e-5);
			for (a, b) in values.iter().zip(&outcomes[0].1) {
				assert!((a - b).abs() < 3e-6, "minibatch update differs: {a} vs {b}");
			}
		}
		Ok(())
	}
);

test_vk!(ppo_rejects_oversized_minibatch, engine, {
	let model = model(&engine, 17)?;
	let mut optimizer = Adam::new(model.all_parameters()?, 0.001)?;
	let mut c = config(7);
	c.update_minibatch_size = 3;
	assert!(PpoTrainer::new(&engine, &model, &mut optimizer, c.clone()).is_err());
	Ok(())
});

test_vk!(
	ppo_late_minibatch_failure_never_advances_optimizer,
	engine,
	{
		struct RejectTail {
			model: CategoricalActorCritic,
			reject: std::cell::Cell<bool>,
			completed_prefix: std::cell::Cell<u32>,
		}
		impl Module for RejectTail {
			fn registry(&self) -> &oa::ml::ModuleRegistry {
				self.model.registry()
			}
			fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.model.forward(input)
			}
		}
		impl oa::ml::ActorCritic for RejectTail {
			fn evaluate(&self, input: &oa::Matrix) -> oa::Result<oa::ml::ActorCriticOutput> {
				if self.reject.get() && input.shape()[0] == 2 {
					return Err(oa::Error::callback("synthetic tail rejection"));
				}
				if self.reject.get() {
					self.completed_prefix.set(self.completed_prefix.get() + 1);
				}
				oa::ml::ActorCritic::evaluate(&self.model, input)
			}
		}
		let model = RejectTail {
			model: model(&engine, 19)?,
			reject: std::cell::Cell::new(false),
			completed_prefix: std::cell::Cell::new(0),
		};
		let initial = model
			.all_parameters()?
			.into_iter()
			.map(|p| p.data().read_f32())
			.collect::<oa::Result<Vec<_>>>()?;
		let mut optimizer = Adam::new(model.all_parameters()?, 0.001)?;
		let mut c = config(421);
		c.horizon = 3;
		c.update_minibatch_size = 4;
		c.enable_gpu_timing = true;
		let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, c)?;
		collect_one(&engine, &mut trainer)?;
		model.reject.set(true);
		assert!(trainer.update().is_err());
		assert!(trainer.is_done());
		assert_eq!(
			trainer.training_loop().snapshot().gpu_timing_stats().count,
			0
		);
		assert_eq!(model.completed_prefix.get(), 1);
		drop(trainer);
		assert_eq!(optimizer.step_count(), 0);
		for (parameter, initial) in model.all_parameters()?.into_iter().zip(initial) {
			assert_eq!(parameter.data().read_f32()?, initial);
		}
		Ok(())
	}
);

test_vk!(
	ppo_timed_minibatches_preserve_updates_and_publish_complete_epoch_samples,
	engine,
	{
		struct Times(Vec<std::time::Duration>);
		impl oa::ml::TrainingCallback for Times {
			fn on_step_end(
				&mut self,
				context: &mut oa::ml::TrainingCallbackContext<'_>,
			) -> oa::Result<oa::ml::TrainingControl> {
				self.0.push(
					context
						.snapshot()
						.last_gpu_time()
						.expect("missing epoch GPU time"),
				);
				Ok(oa::ml::TrainingControl::Continue)
			}
		}
		let mut reference: Option<Vec<f32>> = None;
		for timed in [false, true] {
			for minibatch in [0, 1, 4, 6] {
				let model = model(&engine, 72)?;
				let mut optimizer = oa::ml::Sgd::new(model.all_parameters()?, 0.005, 0.0, 0.0)?;
				let mut c = config(431);
				c.horizon = 3;
				c.rollouts = 2;
				c.update_epochs = 2;
				c.update_minibatch_size = minibatch;
				c.enable_gpu_timing = timed;
				let mut times = Times(Vec::new());
				let mut trainer = PpoTrainer::new(&engine, &model, &mut optimizer, c)?;
				if timed {
					trainer.add_callback(&mut times);
				}
				while !trainer.is_done() {
					if trainer.needs_collection() {
						collect_one(&engine, &mut trainer)?;
					}
					trainer.update()?;
				}
				let snapshot = trainer.finish()?;
				assert_eq!(optimizer.step_count(), 4);
				assert_eq!(snapshot.gpu_timing_stats().count, if timed { 4 } else { 0 });
				if timed {
					assert_eq!(times.0.len(), 4);
					assert!(times.0.iter().all(|time| !time.is_zero()));
					let expected_mean = times
						.0
						.iter()
						.map(|time| time.as_secs_f64() * 1000.0)
						.sum::<f64>()
						/ 4.0;
					assert!((snapshot.gpu_timing_stats().mean_ms - expected_mean).abs() < 1e-9);
				}
				let values = model
					.all_parameters()?
					.into_iter()
					.map(|p| p.data().read_f32())
					.collect::<oa::Result<Vec<_>>>()?
					.concat();
				if let Some(reference) = &reference {
					for (actual, expected) in values.iter().zip(reference) {
						assert!((actual - expected).abs() < 3e-6);
					}
				} else {
					reference = Some(values);
				}
			}
		}
		Ok(())
	}
);

test_vk!(
	ppo_bootstrap_batches_only_row_independent_critics,
	engine,
	{
		use oa::ml::{ActorCritic, ActorCriticOutput, ModuleRegistry};
		// Each row's value is the batch-wide sum of all observation features.
		// Per-step bootstrap yields one total per step; coalescing the whole
		// rollout into one batch would put the rollout-wide total everywhere.
		struct BatchSum<'a> {
			inner: CategoricalActorCritic,
			engine: &'a oa::Engine,
		}
		impl Module for BatchSum<'_> {
			fn registry(&self) -> &ModuleRegistry {
				self.inner.registry()
			}
			fn forward(&self, x: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.inner.forward(x)
			}
		}
		impl ActorCritic for BatchSum<'_> {
			fn evaluate(&self, x: &oa::Matrix) -> oa::Result<ActorCriticOutput> {
				let &[batch, _] = x.shape() else {
					return Err(oa::Error::callback("batch-sum critic requires rank two"));
				};
				let row_sums = oa::matrix::reshape(&oa::matrix::sum(x, 1)?, [1, batch])?;
				let total =
					oa::matrix::mat_mul_nt(&oa::matrix::ones(self.engine, [batch, batch])?, &row_sums)?;
				Ok(ActorCriticOutput {
					logits: self.inner.evaluate(x)?.logits,
					value: oa::matrix::reshape(&total, [batch])?,
				})
			}
		}
		let collect = |trainer: &mut PpoTrainer<'_, '_>| -> oa::Result<Vec<oa::Matrix>> {
			trainer.begin_collection()?;
			let mut successors = Vec::new();
			for step in 0..trainer.config().horizon {
				let scale = (step + 1) as f32;
				let observation = oa::Matrix::new_from_f32(&engine, [2, 2], &[0.1, -0.2, 0.3, 0.4])?;
				let next = oa::Matrix::new_from_f32(&engine, [2, 2], &[scale, 0.5, -0.25, scale])?;
				let reward = oa::Matrix::new_from_f32(&engine, [2], &[1.0, 0.5])?;
				let done = oa::Matrix::new_from_slice(&engine, [2], &[0_u8, 0])?;
				let policy = trainer.act(&observation)?;
				trainer.observe(&observation, &next, &reward, &done, &done, &policy)?;
				successors.push(next);
			}
			trainer.end_collection()?;
			Ok(successors)
		};
		let c = PpoTrainerConfig {
			horizon: 2,
			..config(321)
		};

		let dependent = BatchSum {
			inner: model(&engine, 31)?,
			engine: &engine,
		};
		assert!(!dependent.evaluates_rows_independently());
		let mut optimizer = Adam::new(dependent.all_parameters()?, 1.0e-3)?;
		let mut trainer = PpoTrainer::new(&engine, &dependent, &mut optimizer, c.clone())?;
		collect(&mut trainer)?;
		// Step totals: 1 + 0.5 - 0.25 + 1 = 2.25 and 2 + 0.5 - 0.25 + 2 = 4.25.
		assert_eq!(
			trainer.batch().next_value().read_f32()?,
			[2.25, 2.25, 4.25, 4.25],
			"a batch-dependent critic must bootstrap from its per-step evaluation"
		);
		drop(trainer);

		let independent = model(&engine, 32)?;
		assert!(ActorCritic::evaluates_rows_independently(&independent));
		let mut optimizer = Adam::new(independent.all_parameters()?, 1.0e-3)?;
		let mut trainer = PpoTrainer::new(&engine, &independent, &mut optimizer, c)?;
		let successors = collect(&mut trainer)?;
		let batched = trainer.batch().next_value().read_f32()?;
		drop(trainer);
		let per_step = successors
			.iter()
			.map(|next| independent.evaluate(next)?.value.read_f32())
			.collect::<oa::Result<Vec<_>>>()?
			.concat();
		for (actual, expected) in batched.iter().zip(&per_step) {
			assert!(
				(actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs(),
				"coalesced bootstrap must match per-step evaluation: {actual} vs {expected}"
			);
		}
		Ok(())
	}
);

test_vk!(
	ppo_compiled_updates_match_eager_with_tail_across_rollout_boundaries,
	engine,
	{
		use oa::ml::{ActorCritic, ActorCriticOutput, AdamW, ModuleRegistry, PpoUpdateProgram};
		use std::cell::Cell;
		struct Counted {
			inner: CategoricalActorCritic,
			updates: Cell<usize>,
		}
		impl Module for Counted {
			fn registry(&self) -> &ModuleRegistry {
				self.inner.registry()
			}
			fn forward(&self, x: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.inner.forward(x)
			}
		}
		impl ActorCritic for Counted {
			fn evaluate(&self, x: &oa::Matrix) -> oa::Result<ActorCriticOutput> {
				self.inner.evaluate(x)
			}
			fn auxiliary_loss(&self, x: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.updates.set(self.updates.get() + 1);
				Ok(x.clone())
			}
		}
		let mut results = Vec::new();
		for compiled in [false, true] {
			let model = Counted {
				inner: model(&engine, 72)?,
				updates: Cell::new(0),
			};
			let mut optimizer = AdamW::new(model.all_parameters()?, 0.001)?;
			let mut program = PpoUpdateProgram::default();
			let mut metrics = Vec::new();
			for rollout in 0..2 {
				let c = PpoTrainerConfig {
					horizon: 3,
					update_epochs: 2,
					update_minibatch_size: 4,
					enable_gpu_timing: true,
					..config(123 + rollout * 4)
				};
				let mut trainer = if compiled {
					PpoTrainer::new_compiled(&engine, &model, &mut optimizer, &mut program, c)?
				} else {
					PpoTrainer::new(&engine, &model, &mut optimizer, c)?
				};
				collect_one(&engine, &mut trainer)?;
				while !trainer.is_done() {
					trainer.update()?;
					metrics.push(trainer.metrics());
				}
				let snapshot = trainer.finish()?;
				assert_eq!(snapshot.gpu_timing_stats().count, 2);
				assert_eq!(optimizer.step_count(), (rollout + 1) as u32 * 2);
			}
			assert_eq!(model.updates.get(), if compiled { 2 } else { 8 });
			if compiled {
				assert_eq!(program.replay_count(), 4);
				// normalize + optimizer record once; each minibatch plan records
				// once per distinct cursor staging-ring binding (depth 3).
				assert_eq!(program.command_recording_count(), 8);
			}
			let parameters = model
				.all_parameters()?
				.iter()
				.map(|p| p.data().read_f32())
				.collect::<oa::Result<Vec<_>>>()?;
			results.push((parameters, metrics));
		}
		for (a, b) in results[0]
			.0
			.iter()
			.flatten()
			.zip(results[1].0.iter().flatten())
		{
			assert!((a - b).abs() < 0.00002, "{a} vs {b}");
		}
		for (a, b) in results[0].1.iter().zip(&results[1].1) {
			assert!(
				(a.total_loss - b.total_loss).abs() < 0.00002,
				"{a:?} vs {b:?}"
			);
			assert!((a.entropy - b.entropy).abs() < 0.00002);
		}
		Ok(())
	}
);

test_vk!(
	ppo_complete_compiled_trading_matches_eager_and_reuses_all_phases,
	engine,
	{
		use oa::ml::environment::{Environment, TradingConfig, TradingEnvironment, TradingSeries};
		use oa::ml::{AdamW, PpoUpdateProgram, TradingRolloutProgram};
		let config = TradingConfig {
			environments: 2,
			window: 2,
			horizon: 3,
			seed: 123,
			capital: 100.,
			exposure: 0.1,
			taker_fee: 0.00045,
			slippage_bps: 2.,
			spread_bps: 2.,
			max_drawdown: 0.2,
			perpetual: true,
			liquidate_at_horizon: true,
		};
		let make_series = |n: usize, shift: f32| TradingSeries {
			timestamps: (0..n).map(|i| i as i64 * 1_000_000).collect(),
			market: (0..n)
				.map(|i| {
					let p = 100. + shift + i as f32 * 0.1;
					[p, p + 0.05, 0.]
				})
				.collect(),
			features: (0..n * 2).map(|i| (i as f32 * 0.2).sin()).collect(),
			feature_width: 2,
			books: Vec::new(),
			execution: oa::ml::environment::TradingExecution {
				quantity_step: 0.01,
				fee_per_unit: 0.012 + shift * 0.001,
				..Default::default()
			},
		};
		let mut outcomes = Vec::new();
		for compiled in [false, true] {
			let model = CategoricalActorCritic::new(
				&engine,
				CategoricalActorCriticConfig {
					observation_size: 16,
					action_count: 3,
					hidden_size: 8,
					seed: 77,
				},
			)?;
			let mut optimizer = AdamW::new(model.all_parameters()?, 0.001)?;
			let mut update = PpoUpdateProgram::default();
			let mut collection = TradingRolloutProgram::default();
			let mut env =
				TradingEnvironment::new_with_row_capacity(&engine, config, &[make_series(24, 0.)], 32)?;
			let mut rollouts = Vec::new();
			for rollout in 0..2 {
				if rollout > 0 {
					env.replace_series(&[make_series(29, 10.)])?;
				}
				// Cross the low-word boundary in both eager and replay Philox streams.
				let seed = (1u64 << 32) - 2 + rollout * 4;
				let c = PpoTrainerConfig {
					rollouts: 1,
					horizon: 3,
					environments: 2,
					update_epochs: 2,
					update_minibatch_size: 4,
					observation_shape: vec![16],
					seed,
					enable_gpu_timing: true,
					..PpoTrainerConfig::default()
				};
				let mut trainer = if compiled {
					PpoTrainer::new_compiled(&engine, &model, &mut optimizer, &mut update, c)?
				} else {
					PpoTrainer::new(&engine, &model, &mut optimizer, c)?
				};
				if compiled {
					assert!(
						!trainer
							.collect_trading(&mut env, &mut collection, seed, 2, 0.003)?
							.is_zero()
					);
				} else {
					trainer.begin_collection()?;
					engine.checkpoint()?.wait()?;
					env.reset(seed)?;
					for step in 0..3 {
						let observation = env.observation().clone();
						let policy = trainer.act(&observation)?;
						let transition = if step == 2 {
							env.step_terminal(&policy.action)?
						} else {
							env.step(&policy.action)?
						};
						let reward = oa::matrix::add_scalar(transition.reward(), 0.003)?;
						trainer.observe(
							transition.observation(),
							transition.next_observation(),
							&reward,
							transition.terminated(),
							transition.truncated(),
							&policy,
						)?;
						env.reset_completed()?;
					}
					trainer.end_collection()?;
					let event = env.submit()?;
					env.wait(&event)?;
					env.check_status()?;
				}
				let b = trainer.batch();
				rollouts.push((
					b.action().read::<i32>()?,
					[
						b.observation(),
						b.reward(),
						b.value(),
						b.next_value(),
						b.old_log_probability(),
						b.advantage(),
						b.returns(),
					]
					.iter()
					.map(|m| m.read_f32())
					.collect::<oa::Result<Vec<_>>>()?,
				));
				while !trainer.is_done() {
					trainer.update()?;
				}
				trainer.finish()?;
			}
			if compiled {
				assert_eq!(collection.replay_count(), 6);
				assert_eq!(collection.command_recording_count(), 3);
				assert_eq!(update.replay_count(), 4);
				// normalize + optimizer record once; each minibatch plan records
				// once per distinct cursor staging-ring binding (depth 3).
				assert_eq!(update.command_recording_count(), 8);
			}
			outcomes.push((
				rollouts,
				model
					.all_parameters()?
					.iter()
					.map(|p| p.data().read_f32())
					.collect::<oa::Result<Vec<_>>>()?,
			));
			env.close()?;
		}
		for (a, b) in outcomes[0].0.iter().zip(&outcomes[1].0) {
			assert_eq!(a.0, b.0);
			for (x, y) in a.1.iter().flatten().zip(b.1.iter().flatten()) {
				assert!((x - y).abs() < 0.00005, "{x} vs {y}");
			}
		}
		for (a, b) in outcomes[0]
			.1
			.iter()
			.flatten()
			.zip(outcomes[1].1.iter().flatten())
		{
			assert!((a - b).abs() < 0.00005, "{a} vs {b}");
		}
		Ok(())
	}
);

test_vk!(
	ppo_continuing_trading_retains_position_bootstrap_and_restores_state,
	engine,
	{
		use oa::ml::environment::{
			Environment, TRADING_ACCOUNT_FEATURES, TradingConfig, TradingEnvironment, TradingSeries,
			TradingState,
		};
		use oa::ml::{AdamW, PpoUpdateProgram, TradingRolloutProgram};
		let c = TradingConfig {
			environments: 1,
			window: 2,
			horizon: 7,
			seed: 3,
			capital: 1000.0,
			exposure: 1.0,
			taker_fee: 0.0,
			slippage_bps: 0.0,
			spread_bps: 0.0,
			max_drawdown: 0.9,
			perpetual: true,
			liquidate_at_horizon: true,
		};
		let source = TradingSeries {
			timestamps: (0..9).map(|i| i * 1_000_000).collect(),
			market: vec![[100.0, 100.0, 0.0]; 9],
			features: vec![0.0; 9],
			feature_width: 1,
			execution: Default::default(),
			books: Vec::new(),
		};
		// Exact constant initial policy through the public Linear constructor;
		// parameter storage remains private and optimizer-owned.
		use oa::ml::{ActorCritic, ActorCriticOutput, ModuleRegistry, nn::Linear};
		use std::rc::Rc;
		struct HoldingPolicy {
			policy: Rc<Linear>,
			value: Rc<Linear>,
			registry: ModuleRegistry,
		}
		impl Module for HoldingPolicy {
			fn registry(&self) -> &ModuleRegistry {
				&self.registry
			}
			fn forward(&self, x: &oa::Matrix) -> oa::Result<oa::Matrix> {
				self.policy.forward(x)
			}
		}
		impl ActorCritic for HoldingPolicy {
			fn evaluate(&self, x: &oa::Matrix) -> oa::Result<ActorCriticOutput> {
				Ok(ActorCriticOutput {
					logits: self.policy.forward(x)?,
					value: self.value.forward(x)?.reshape([x.shape()[0]])?,
				})
			}
		}
		let width = 2 + TRADING_ACCOUNT_FEATURES;
		let policy = Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [3, width], &vec![0.0; 3 * width])?,
			oa::Matrix::new_from_f32(&engine, [3], &[0.0, 0.0, 50.0])?,
		)?);
		let value = Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32(&engine, [1, width], &vec![0.0; width])?,
			oa::Matrix::new_from_f32(&engine, [1], &[2.0])?,
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("policy", policy.clone())?;
		registry.register_module("value", value.clone())?;
		let model = HoldingPolicy {
			policy,
			value,
			registry,
		};
		let mut optimizer = AdamW::new(model.all_parameters()?, 0.0001)?;
		let mut env = TradingEnvironment::new(&engine, c, std::slice::from_ref(&source))?;
		let mut updates = PpoUpdateProgram::default();
		let mut collection = TradingRolloutProgram::default();
		let mut previous_cursor = 0;
		for chunk in 0..3 {
			if chunk == 1 {
				// Change loss policy with inventory still open. The collection
				// plan must retain its rollout buffers and the optimizer its moments.
				updates.invalidate_updates();
				assert!(!updates.is_compiled());
				optimizer.set_learning_rate(0.00003)?;
			}
			let config = PpoTrainerConfig {
				rollouts: 1,
				horizon: 2,
				environments: 1,
				update_epochs: 1,
				update_minibatch_size: 2,
				observation_shape: vec![2 + TRADING_ACCOUNT_FEATURES],
				seed: (1_u64 << 32) - 2,
				enable_gpu_timing: true,
				loss: oa::ml::loss::PpoLossConfig {
					entropy_coefficient: if chunk == 0 { 0.01 } else { 0.001 },
					..Default::default()
				},
				..Default::default()
			};
			let gamma = config.gae.gamma;
			let mut trainer =
				PpoTrainer::new_compiled(&engine, &model, &mut optimizer, &mut updates, config)?;
			trainer.restore_action_index(previous_cursor)?;
			trainer.collect_trading_continuing(&mut env, &mut collection, 3, 2, 0.0)?;
			previous_cursor = trainer.action_index();
			assert_eq!(previous_cursor, (chunk + 1) * 2);
			let batch = trainer.batch();
			assert_eq!(batch.action().read::<i32>()?, [2, 2]);
			assert_eq!(batch.terminated().read::<u8>()?, [0, 0]);
			assert_eq!(batch.truncated().read::<u8>()?, [0, 0]);
			let rewards = batch.reward().read_f32()?;
			let values = batch.value().read_f32()?;
			let next = batch.next_value().read_f32()?;
			let advantage = batch.advantage().read_f32()?;
			assert!(
				(advantage[1] - (rewards[1] + gamma * next[1] - values[1])).abs() < 1e-6,
				"chunk boundary must bootstrap"
			);
			assert_eq!(env.ledger().read_f32()?[1], 10.0);
			assert_eq!(env.ledger().read_f32()?[7], 1.0);
			assert_eq!(env.metadata().read::<u32>()?[1], (chunk as u32 + 1) * 2);
			while !trainer.is_done() {
				trainer.update()?;
			}
			trainer.finish()?;
			if chunk == 1 {
				let state = TradingState::new_from_bytes(&env.snapshot()?.to_bytes())?;
				let mut restored = TradingEnvironment::new(&engine, c, std::slice::from_ref(&source))?;
				restored.restore(&state)?;
				env.close()?;
				env = restored;
				collection = TradingRolloutProgram::default();
			}
		}
		env.close()?;
		Ok(())
	}
);
