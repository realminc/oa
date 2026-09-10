use std::{
	cell::{Cell, RefCell},
	rc::Rc,
	time::SystemTime,
};

struct EpochEndRecorder {
	count: u64,
}

impl oa::ml::TrainingCallback for EpochEndRecorder {
	fn on_epoch_end(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		self.count += 1;
		Ok(oa::ml::TrainingControl::Continue)
	}
}

#[test]
fn phase_contract_rejects_invalid_descriptions() {
	assert!(oa::ml::TrainingPhase::new("", 1, 1).is_err());
	assert!(oa::ml::TrainingPhase::new("warmup", 0, 1).is_err());
	assert!(oa::ml::TrainingPhase::new("warmup", 2, 1).is_err());
}

test_vk!(
	validation_feeds_early_stop_and_does_not_suppress_later_callbacks,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let evaluations = Rc::new(Cell::new(0_u64));
		let evaluator_count = Rc::clone(&evaluations);
		let mut validation = oa::ml::Validation::new(
			move |_| {
				let index = evaluator_count.get();
				evaluator_count.set(index + 1);
				Ok(oa::ml::ValidationResult {
					loss: if index == 0 { 0.5 } else { 0.6 },
					batches: 1,
					samples: 2,
				})
			},
			"val_loss",
			0,
		)?;
		let validation_metric = validation.metric();
		let mut early_stop = oa::ml::EarlyStopping::with_validation_metric(
			1,
			0.0,
			oa::ml::EarlyStopMode::Min,
			validation_metric.clone(),
		)?;
		let mut trailing = EpochEndRecorder { count: 0 };
		let mut summary_callback = oa::ml::TrainingSummary::new(false);
		summary_callback.set_validation_metric(validation_metric.clone());

		let final_snapshot = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 3,
					steps_per_epoch: 1,
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut validation);
			training.add_callback(&mut early_stop);
			training.add_callback(&mut trailing);
			training.add_callback(&mut summary_callback);
			while training.begin_step()? {
				training.seal_replay_inputs()?;
				training.zero_grad();
				let tape = oa::ml::GradientTape::new();
				let logits = layer.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
				tape.backward(&loss)?;
				training.complete_step(&loss)?;
			}
			training.finish()?
		};

		assert_eq!(final_snapshot.step_count(), 2);
		assert_eq!(evaluations.get(), 2);
		assert!(early_stop.should_stop());
		assert_eq!(early_stop.best(), 0.5);
		assert_eq!(trailing.count, 2);
		assert_eq!(validation_metric.value(), Some(0.6));
		assert_eq!(
			validation.last_result(),
			oa::ml::ValidationResult {
				loss: 0.6,
				batches: 1,
				samples: 2,
			}
		);
		let report = summary_callback.render_report(
			final_snapshot,
			&oa::ml::ItTrainingConfig {
				batch_size: 2,
				..oa::ml::ItTrainingConfig::default()
			},
		);
		assert!(report.contains("Validation: val_loss 0.600000"));
		Ok(())
	}
);

test_vk!(
	step_only_validation_runs_at_interval_without_duplicate_final_evaluation,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let evaluations = Rc::new(Cell::new(0_u64));
		let evaluator_count = Rc::clone(&evaluations);
		let mut validation = oa::ml::Validation::new(
			move |_| {
				evaluator_count.set(evaluator_count.get() + 1);
				Ok(oa::ml::ValidationResult {
					loss: 0.25,
					batches: 1,
					samples: 2,
				})
			},
			"val_loss",
			2,
		)?;
		{
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 4,
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut validation);
			while training.begin_step()? {
				training.seal_replay_inputs()?;
				training.zero_grad();
				let tape = oa::ml::GradientTape::new();
				let logits = layer.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
				tape.backward(&loss)?;
				training.complete_step(&loss)?;
			}
			training.finish()?;
		}
		assert_eq!(evaluations.get(), 2);
		Ok(())
	}
);

test_vk!(
	phase_schedule_uses_the_existing_optimizer_and_epoch_map,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let transitions = Rc::new(RefCell::new(Vec::<(usize, String)>::new()));
		let recorded = Rc::clone(&transitions);
		let mut phases = oa::ml::PhaseSchedule::new();
		phases.add_phase("warmup", 1, 1)?;
		phases.add_phase("main", 2, 2)?;
		phases.set_on_phase_begin(move |index, phase, optimizer| {
			recorded.borrow_mut().push((index, phase.id().to_owned()));
			optimizer.set_learning_rate(if index == 0 { 0.02 } else { 0.01 })
		});

		{
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					epoch_steps: vec![1, 1, 1],
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut phases);
			while training.begin_step()? {
				training.seal_replay_inputs()?;
				training.zero_grad();
				let tape = oa::ml::GradientTape::new();
				let logits = layer.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
				tape.backward(&loss)?;
				training.complete_step(&loss)?;
			}
			training.finish()?;
		}

		assert_eq!(phases.current_phase(), Some(1));
		assert_eq!(
			*transitions.borrow(),
			[(0, "warmup".to_owned()), (1, "main".to_owned())]
		);
		assert_eq!(optimizer.learning_rate(), 0.01);

		let mut mismatched = oa::ml::PhaseSchedule::new();
		mismatched.add_phase("wrong", 1, 2)?;
		let error = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 1,
					steps_per_epoch: 1,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut mismatched);
			training.begin_step().unwrap_err()
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		assert!(error.message().contains("defines 2 steps"));
		Ok(())
	}
);

test_vk!(
	checkpoint_callback_saves_each_epoch_and_restores_the_best_state,
	engine,
	{
		let directory = std::env::temp_dir().join(format!(
			"oars-callback-checkpoint-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let validation_losses = [3.0, 2.0, 2.5];
		let mut validation = oa::ml::Validation::new(
			move |snapshot| {
				Ok(oa::ml::ValidationResult {
					loss: validation_losses[(snapshot.epoch() - 1) as usize],
					batches: 1,
					samples: 2,
				})
			},
			"val_loss",
			0,
		)?;
		let metric = validation.metric();
		let mut manager = oa::ml::CheckpointManager::new(
			&engine,
			oa::ml::CheckpointManagerConfig {
				directory: directory.clone(),
				model_name: "Linear".to_owned(),
				max_keep: 2,
				metric_name: "val_loss".to_owned(),
				..oa::ml::CheckpointManagerConfig::default()
			},
		)?;
		let mut checkpoint = oa::ml::CbCheckpoint::new(&mut manager, &layer);
		checkpoint.set_validation_metric(metric);
		checkpoint.set_verbose(false);
		let mut best_weight = None;
		let mut last_weight_before_restore = None;

		let final_snapshot = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					epoch_steps: vec![1, 1, 1],
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut validation);
			training.add_callback(&mut checkpoint);
			let mut completed_steps = 0_u64;
			while training.begin_step()? {
				training.seal_replay_inputs()?;
				training.zero_grad();
				let tape = oa::ml::GradientTape::new();
				let logits = layer.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
				tape.backward(&loss)?;
				training.complete_step(&loss)?;
				completed_steps += 1;
				match completed_steps {
					2 => best_weight = Some(layer.weight().data().read_f32()?),
					3 => last_weight_before_restore = Some(layer.weight().data().read_f32()?),
					_ => {}
				}
			}
			training.finish()?
		};

		let best_epoch = checkpoint.best_epoch();
		drop(checkpoint);
		assert_eq!(final_snapshot.step_count(), 3);
		assert_eq!(best_epoch, 2);
		assert_eq!(manager.best_metric(), 2.0);
		assert_eq!(optimizer.step_count(), 2);
		let best_weight = best_weight.expect("captured epoch-two weight");
		assert_ne!(
			last_weight_before_restore.expect("captured epoch-three weight"),
			best_weight
		);
		assert_eq!(layer.weight().data().read_f32()?, best_weight);
		assert!(manager.master_path().is_file());
		assert_eq!(
			std::fs::read_dir(manager.incremental_directory())
				.expect("read incremental checkpoints")
				.count(),
			2
		);

		std::fs::remove_dir_all(&directory).expect("remove callback checkpoint directory");
		Ok(())
	}
);
