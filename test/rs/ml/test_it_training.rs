struct LifecycleRecorder {
	train_begin: u32,
	epoch_begin: u32,
	step_end: u32,
	epoch_end: u32,
	train_end: u32,
	steps: Vec<(u64, u64)>,
}

impl LifecycleRecorder {
	fn new() -> Self {
		Self {
			train_begin: 0,
			epoch_begin: 0,
			step_end: 0,
			epoch_end: 0,
			train_end: 0,
			steps: Vec::new(),
		}
	}
}

impl oa::ml::TrainingCallback for LifecycleRecorder {
	fn on_train_begin(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		self.train_begin += 1;
		Ok(oa::ml::TrainingControl::Continue)
	}

	fn on_epoch_begin(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		self.epoch_begin += 1;
		Ok(oa::ml::TrainingControl::Continue)
	}

	fn on_step_end(
		&mut self,
		context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		let state = context.snapshot();
		self.step_end += 1;
		self.steps.push((state.epoch(), state.step_in_epoch()));
		Ok(oa::ml::TrainingControl::Continue)
	}

	fn on_epoch_end(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		self.epoch_end += 1;
		Ok(oa::ml::TrainingControl::Continue)
	}

	fn on_train_end(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		self.train_end += 1;
		Ok(oa::ml::TrainingControl::Continue)
	}
}

struct FailingCallback;

impl oa::ml::TrainingCallback for FailingCallback {
	fn on_train_begin(
		&mut self,
		_context: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		Err(oa::Error::callback("intentional callback failure"))
	}
}

test_vk!(
	training_session_applies_revisioned_commands_at_safe_points,
	engine,
	{
		use oa::ml::Optimizer as _;
		use std::{
			sync::{
				Arc,
				atomic::{AtomicU32, Ordering},
			},
			thread,
			time::Duration,
		};

		let checkpoint_count = Arc::new(AtomicU32::new(0));
		let evaluate_count = Arc::new(AtomicU32::new(0));
		let rebuild_count = Arc::new(AtomicU32::new(0));
		let checkpoint_counter = Arc::clone(&checkpoint_count);
		let evaluate_counter = Arc::clone(&evaluate_count);
		let rebuild_counter = Arc::clone(&rebuild_count);
		let handlers = oa::ml::TrainingSessionHandlers::default()
			.checkpoint(move || {
				checkpoint_counter.fetch_add(1, Ordering::Relaxed);
				Ok(())
			})
			.evaluate(move || {
				evaluate_counter.fetch_add(1, Ordering::Relaxed);
				Ok(())
			})
			.rebuild(move |_| {
				rebuild_counter.fetch_add(1, Ordering::Relaxed);
				Ok(())
			});
		let session = oa::ml::TrainingSession::new(oa::ml::TrainingSessionConfig {
			command_capacity: 16,
			result_capacity: 16,
			snapshot_capacity: 2,
		});
		let observer = session.clone();
		let mut optimizer = oa::ml::NoOpOptimizer::new(0.25)?;
		let loss = oa::Matrix::from_f32(&engine, [], &[0.5])?;
		let summary = {
			let mut training = oa::ml::ItTraining::new_eager(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 2,
					steps_per_epoch: 1,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.attach_session(&session, handlers)?;
			assert_eq!(session.revision(), 1);
			assert_eq!(
				session.parameter("learning_rate"),
				Some(oa::ml::TrainingValue::Float(0.25))
			);

			session.pause(0)?;
			assert!(!training.begin_step()?);
			assert_eq!(session.state(), oa::ml::TrainingState::Paused);
			let paused_revision = session.revision();

			let producer = thread::spawn(move || -> oa::Result<()> {
				thread::sleep(Duration::from_millis(20));
				observer.set_parameter(
					"learning_rate",
					oa::ml::TrainingValue::Float(0.125),
					paused_revision,
				)?;
				observer.checkpoint(0)?;
				observer.evaluate(0)?;
				observer.request_rebuild(oa::ml::TrainingValue::String("same-shape".into()), 0)?;
				observer.resume(0)?;
				Ok(())
			});
			assert!(training.wait_begin_step()?);
			producer.join().expect("command producer panicked")?;
			assert_eq!(session.state(), oa::ml::TrainingState::Running);
			assert_eq!(
				session.parameter("learning_rate"),
				Some(oa::ml::TrainingValue::Float(0.125))
			);
			session.publish_metric("accuracy", 0.75);
			training.complete_step(&loss)?;

			session.set_parameter(
				"learning_rate",
				oa::ml::TrainingValue::Float(0.5),
				paused_revision,
			)?;
			session.request_recapture(0)?;
			session.stop(0)?;
			assert!(!training.begin_step()?);
			assert_eq!(session.state(), oa::ml::TrainingState::Stopping);
			training.finish()?
		};

		assert_eq!(summary.step_count(), 1);
		assert_eq!(session.state(), oa::ml::TrainingState::Completed);
		assert_eq!(checkpoint_count.load(Ordering::Relaxed), 1);
		assert_eq!(evaluate_count.load(Ordering::Relaxed), 1);
		assert_eq!(rebuild_count.load(Ordering::Relaxed), 1);
		assert_eq!(optimizer.learning_rate(), 0.125);
		let results = session.results_after(0);
		assert_eq!(results.len(), 9);
		assert_eq!(
			results[0].disposition,
			oa::ml::TrainingCommandDisposition::Applied
		);
		assert_eq!(
			results[6].disposition,
			oa::ml::TrainingCommandDisposition::Rejected
		);
		assert_eq!(
			results[6].error.as_ref().map(|error| error.kind()),
			Some(oa::ErrorKind::Aborted)
		);
		assert_eq!(
			results[7].disposition,
			oa::ml::TrainingCommandDisposition::Rejected
		);
		assert_eq!(
			results[7].error.as_ref().map(|error| error.kind()),
			Some(oa::ErrorKind::FailedPrecondition)
		);
		assert_eq!(session.take_results(), results);
		assert!(session.take_results().is_empty());
		assert_eq!(session.results_after(0), results);
		let snapshot = session.latest_snapshot().expect("terminal snapshot");
		assert_eq!(snapshot.state, oa::ml::TrainingState::Completed);
		assert_eq!(snapshot.step, 1);
		assert_eq!(snapshot.loss, 0.5);
		assert_eq!(snapshot.metrics.len(), 1);
		assert_eq!(snapshot.metrics[0].name, "accuracy");
		assert_eq!(snapshot.metrics[0].value, 0.75);
		Ok(())
	}
);

test_vk!(
	eager_it_training_owns_exact_completion_metrics_and_callbacks,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let config = oa::ml::ItTrainingConfig {
			total_steps: 4,
			steps_per_epoch: 2,
			batch_size: 2,
			sequence_length: 3,
			source_units_per_sample: 4.0,
			enable_gpu_timing: true,
			..oa::ml::ItTrainingConfig::default()
		};
		let report_config = config.clone();
		let mut callback = LifecycleRecorder::new();
		let mut loss_metric = oa::ml::LossMetric::default();
		let summary = {
			let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
			training.add_callback(&mut callback);
			training.add_metric(&mut loss_metric);
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

		assert_eq!(optimizer.step_count(), 4);
		assert_eq!(callback.train_begin, 1);
		assert_eq!(callback.epoch_begin, 2);
		assert_eq!(callback.step_end, 4);
		assert_eq!(callback.epoch_end, 2);
		assert_eq!(callback.train_end, 1);
		assert_eq!(callback.steps, [(1, 1), (1, 2), (2, 1), (2, 2)]);
		assert_eq!(loss_metric.count(), 2);
		assert_eq!(summary.step_count(), 4);
		assert_eq!(summary.total_samples(), 8);
		assert_eq!(summary.total_units(), 24);
		assert_eq!(summary.total_source_units(), 32);
		assert!(summary.last_loss().is_some_and(f32::is_finite));
		assert!(summary.last_gpu_time().is_some_and(|time| !time.is_zero()));
		let gpu = summary.gpu_timing_stats();
		assert_eq!(gpu.count, 4);
		assert!(gpu.min_ms > 0.0);
		assert!(gpu.min_ms <= gpu.median_ms);
		assert!(gpu.median_ms <= gpu.max_ms);
		assert!(gpu.p95_ms <= gpu.max_ms);
		assert!(summary.gpu_samples_per_second() > 0.0);
		assert!(summary.gpu_units_per_second() > 0.0);
		assert!(summary.gpu_source_units_per_second() > 0.0);
		let progress = oa::ml::ProgressBar::default().render_line(summary, &report_config);
		assert!(progress.contains("2/2 |██████████|"));
		assert!(progress.contains("sample/s"));
		assert!(progress.contains("loss:"));
		let report = oa::ml::TrainingSummary::new(false).render_report(summary, &report_config);
		assert!(report.contains("Summary:"));
		assert!(report.contains("Wall:"));
		assert!(report.contains("GPU (training_step): mean"));
		assert!(report.contains("p50"));
		assert!(report.contains("p95"));
		assert!(report.contains("wall-GPU gap"));

		let mut failing = FailingCallback;
		let error = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 1,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut failing);
			training.begin_step().unwrap_err()
		};
		assert_eq!(error.kind(), oa::ErrorKind::CallbackFailure);
		assert_eq!(error.message(), "intentional callback failure");
		Ok(())
	}
);

test_vk!(
	captured_it_training_replays_one_program_without_reauthoring_the_graph,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = layer.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		assert_eq!(program.compilation_stages().len(), 11);
		assert_eq!(
			program.compilation_stages()[1].stage(),
			oa::ml::TrainingCompilationStage::ReplaySafety
		);
		assert_eq!(
			program.compilation_stages()[1].state(),
			oa::ml::TrainingCompilationState::Applied
		);
		assert_eq!(
			program.compilation_stages()[10].stage(),
			oa::ml::TrainingCompilationStage::CommandRecording
		);
		assert_eq!(
			program.compilation_stages()[10].state(),
			oa::ml::TrainingCompilationState::NotRun
		);
		let mut metric = oa::ml::LossMetric::default();
		let summary = {
			let config = oa::ml::ItTrainingConfig {
				total_steps: 3,
				batch_size: 2,
				enable_gpu_timing: true,
				..oa::ml::ItTrainingConfig::default()
			};
			let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
			training.add_metric(&mut metric);
			while training.begin_step()? {
				training.complete_program_step(&mut program)?;
			}
			training.finish()?
		};

		assert_eq!(program.replay_count(), 3);
		assert_eq!(program.diagnostics().command_recording_count(), 3);
		assert_eq!(optimizer.step_count(), 3);
		assert_eq!(metric.count(), 3);
		assert_eq!(summary.step_count(), 3);
		assert!(summary.last_gpu_time().is_some_and(|time| !time.is_zero()));
		program.wait()?;
		assert!(program.is_complete()?);
		program.reset()?;
		Ok(())
	}
);

test_vk!(
	automatic_it_training_prepares_every_step_and_records_only_capture_boundaries,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let config = oa::ml::ItTrainingConfig {
			total_steps: 4,
			batch_size: 2,
			..oa::ml::ItTrainingConfig::default()
		};
		let mut prepare_count = 0_u32;
		let mut record_count = 0_u32;
		let summary = {
			let mut training = oa::ml::ItTraining::new(&engine, &mut optimizer, config)?;
			loop {
				let ran = training.step(
					|| {
						prepare_count += 1;
						let scale = prepare_count as f32;
						let input = oa::Matrix::from_f32(&engine, [2, 2], &[scale, 0.0, 0.0, scale])?;
						let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
						Ok((input, targets))
					},
					|(input, targets)| {
						record_count += 1;
						let tape = oa::ml::GradientTape::new();
						let logits = layer.forward(&input)?;
						let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
						tape.backward(&loss)?;
						Ok(loss)
					},
				)?;
				if !ran {
					break;
				}
				if training.snapshot().step_count() == 2 {
					let first = training
						.training_program()
						.expect("step two must capture the first program");
					assert_eq!(first.replay_count(), 1);
					training.request_program_recapture()?;
				}
			}
			let program = training
				.training_program()
				.expect("recapture must leave an owned program");
			assert_eq!(program.replay_count(), 2);
			assert_eq!(program.diagnostics().command_recording_count(), 1);
			assert_eq!(program.diagnostics().command_cache_hit_count(), 2);
			assert_eq!(program.compilation_stages().len(), 11);
			assert_eq!(
				program.compilation_stages()[1].state(),
				oa::ml::TrainingCompilationState::Applied
			);
			assert_eq!(
				program.compilation_stages()[10].stage(),
				oa::ml::TrainingCompilationStage::CommandRecording
			);
			assert_eq!(
				program.compilation_stages()[10].state(),
				oa::ml::TrainingCompilationState::Applied
			);
			let report = program.compilation_debug_report_json("Training\n\"Step");
			let report: serde_json::Value =
				serde_json::from_str(&report).expect("compilation report must be valid JSON");
			assert_eq!(report["schema"], "oa.training_compilation.v2");
			assert_eq!(report["name"], "Training\n\"Step");
			assert_eq!(report["captured"], true);
			assert_eq!(
				report["stages"]
					.as_array()
					.expect("stages must be an array")
					.len(),
				11
			);
			assert_eq!(report["stages"][10]["state"], "applied");
			assert_eq!(report["dnn_plan"]["applied_partition_count"], 0);
			assert_eq!(
				report["dnn_plan"]["inherited_partition_count"],
				program.diagnostics().dnn_inherited_partition_count()
			);
			assert_eq!(
				report["dnn_plan"]["fallback_partition_count"],
				program.diagnostics().dnn_fallback_partition_count()
			);
			assert_eq!(
				report["dnn_plan"]["unexpected_fallback_count"],
				program.diagnostics().dnn_unexpected_fallback_count()
			);
			assert_eq!(
				report["dnn_plan"]["fallback_reasons"]
					.as_array()
					.expect("DNN fallback reasons must be an array")
					.len(),
				program.diagnostics().dnn_fallback_partition_count() as usize
			);
			assert_eq!(
				report["memory_analysis"]["resource_count"],
				program.diagnostics().captured_resource_count()
			);
			let semantic: serde_json::Value =
				serde_json::from_str(&program.semantic_debug_report_json("TrainingStep"))
					.expect("training semantic report must be valid JSON");
			assert_eq!(semantic["schema"], "oa.semantic_graph.v2");
			assert_eq!(
				semantic["operations"]
					.as_array()
					.expect("semantic operations must be an array")
					.len(),
				program.semantic_graph().operations().len()
			);
			let executable: serde_json::Value =
				serde_json::from_str(&program.debug_report_json("TrainingStep"))
					.expect("training executable report must be valid JSON");
			assert_eq!(executable["schema"], "oa.execution_graph.v3");
			assert_eq!(executable["name"], "TrainingStep");
			assert_eq!(executable["compiled"], true);
			assert!(
				executable["stats"]["war_barriers"]
					.as_u64()
					.is_some_and(|count| count != 0)
			);
			assert!(
				executable["barriers"]
					.as_array()
					.expect("executable barriers must be an array")
					.iter()
					.any(|barrier| barrier["reason"] == "write_after_read")
			);
			assert_eq!(
				executable["nodes"]
					.as_array()
					.expect("executable nodes must be an array")
					.len(),
				program.diagnostics().node_count()
			);
			training.finish()?
		};

		assert_eq!(prepare_count, 4);
		assert_eq!(record_count, 3);
		assert_eq!(optimizer.step_count(), 4);
		assert_eq!(summary.step_count(), 4);
		assert!(summary.last_loss().is_some_and(f32::is_finite));
		Ok(())
	}
);

test_vk!(
	automatic_it_training_rejects_executable_work_during_preparation,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [1, 1], &[1.0])?;
		let bias = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let error = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 1,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training
				.step(
					|| {
						let left = oa::matrix::ones(&engine, [1])?;
						let right = oa::matrix::ones(&engine, [1])?;
						let _unexpected = oa::matrix::add(&left, &right)?;
						Ok(())
					},
					|()| Err(oa::Error::callback("record must not run")),
				)
				.unwrap_err()
		};
		assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
		assert_eq!(
			error.message(),
			"automatic training preparation may only allocate or refresh inputs"
		);

		let left = oa::matrix::ones(&engine, [1])?;
		let right = oa::matrix::ones(&engine, [1])?;
		assert_eq!(oa::matrix::add(&left, &right)?.read_f32()?, [2.0]);
		Ok(())
	}
);

test_vk!(
	learning_rate_and_early_stop_callbacks_share_the_one_training_lifecycle,
	engine,
	{
		use oa::ml::LrScheduler as _;

		let weight = oa::Matrix::from_f32(&engine, [1, 1], &[1.0])?;
		let bias = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.25)?;
		let schedule = oa::ml::CosineScheduler::new(0.25, 0.05, 4)?;
		let mut scheduler = oa::ml::LearningRateScheduler::new(&schedule);
		let mut early_stop = oa::ml::EarlyStopping::new(1, 0.0, oa::ml::EarlyStopMode::Min)?;
		let loss = oa::Matrix::from_f32(&engine, [], &[1.0])?;
		let summary = {
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 5,
					steps_per_epoch: 1,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut scheduler);
			training.add_callback(&mut early_stop);
			while training.begin_step()? {
				training.complete_step(&loss)?;
			}
			training.finish()?
		};

		assert_eq!(summary.step_count(), 2);
		assert_eq!(optimizer.step_count(), 2);
		assert!(early_stop.should_stop());
		assert_eq!(early_stop.bad_epochs(), 1);
		assert_eq!(early_stop.best(), 1.0);
		assert_eq!(
			optimizer.learning_rate(),
			schedule.learning_rate(summary.step_count() + 1)
		);
		optimizer.set_learning_rate(0.0)?;
		assert_eq!(optimizer.learning_rate(), 0.0);
		assert!(optimizer.set_learning_rate(-0.1).is_err());
		Ok(())
	}
);

test_vk!(
	captured_learning_rate_schedule_matches_eager_parameter_updates,
	engine,
	{
		fn assert_close(left: &[f32], right: &[f32]) {
			assert_eq!(left.len(), right.len());
			for (index, (left, right)) in left.iter().zip(right).enumerate() {
				assert!(
					(left - right).abs() <= 2.0e-5,
					"parameter {index}: eager {left}, captured {right}"
				);
			}
		}

		let eager_layer = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?,
		)?;
		let eager_weight = eager_layer.weight();
		let eager_bias = eager_layer
			.bias()
			.expect("biased Linear is missing its bias");
		let mut eager_optimizer = oa::ml::AdamW::new(eager_layer.parameters(), 0.25)?;
		let eager_schedule = oa::ml::CosineScheduler::new(0.25, 0.05, 4)?;
		let mut eager_scheduler = oa::ml::LearningRateScheduler::new(&eager_schedule);
		{
			let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
			let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut eager_optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 4,
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut eager_scheduler);
			while training.begin_step()? {
				training.zero_grad();
				let tape = oa::ml::GradientTape::new();
				let logits = eager_layer.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
				tape.backward(&loss)?;
				training.complete_step(&loss)?;
			}
			training.finish()?;
		}

		let captured_layer = oa::ml::nn::Linear::from_matrices(
			oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?,
			oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?,
		)?;
		let captured_weight = captured_layer.weight();
		let captured_bias = captured_layer
			.bias()
			.expect("biased Linear is missing its bias");
		let mut captured_optimizer = oa::ml::AdamW::new(captured_layer.parameters(), 0.25)?;
		let captured_schedule = oa::ml::CosineScheduler::new(0.25, 0.05, 4)?;
		let mut captured_scheduler = oa::ml::LearningRateScheduler::new(&captured_schedule);
		{
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut captured_optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 4,
					batch_size: 2,
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut captured_scheduler);
			loop {
				let ran = training.step(
					|| {
						Ok((
							oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?,
							oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?,
						))
					},
					|(input, targets)| {
						let tape = oa::ml::GradientTape::new();
						let logits = captured_layer.forward(&input)?;
						let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
						tape.backward(&loss)?;
						Ok(loss)
					},
				)?;
				if !ran {
					break;
				}
			}
			assert!(training.training_program().is_some());
			training.finish()?;
		}

		assert_eq!(eager_optimizer.step_count(), 4);
		assert_eq!(captured_optimizer.step_count(), 4);
		assert_eq!(
			eager_optimizer.learning_rate(),
			captured_optimizer.learning_rate()
		);
		assert_close(
			&eager_weight.data().read_f32()?,
			&captured_weight.data().read_f32()?,
		);
		assert_close(
			&eager_bias.data().read_f32()?,
			&captured_bias.data().read_f32()?,
		);
		Ok(())
	}
);

test_vk!(
	csv_callback_writes_one_exact_row_per_completed_step,
	engine,
	{
		use std::time::{SystemTime, UNIX_EPOCH};

		let unique = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system clock precedes Unix epoch")
			.as_nanos();
		let path =
			std::env::temp_dir().join(format!("oars-training-{}-{unique}.csv", std::process::id()));
		let weight = oa::Matrix::from_f32(&engine, [1, 1], &[1.0])?;
		let bias = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.1)?;
		let mut logger = oa::ml::CsvLogger::new(&path);
		assert_eq!(logger.path(), path);
		let loss = oa::Matrix::from_f32(&engine, [], &[0.5])?;
		{
			let mut training = oa::ml::ItTraining::new(
				&engine,
				&mut optimizer,
				oa::ml::ItTrainingConfig {
					total_steps: 2,
					steps_per_epoch: 1,
					batch_size: 4,
					sequence_length: 8,
					sequence_unit: "token,utf8".into(),
					source_unit: "byte\"raw".into(),
					..oa::ml::ItTrainingConfig::default()
				},
			)?;
			training.add_callback(&mut logger);
			while training.begin_step()? {
				training.complete_step(&loss)?;
			}
			training.finish()?;
		}
		let text = std::fs::read_to_string(&path).expect("training CSV must be readable");
		std::fs::remove_file(&path).expect("training CSV cleanup must succeed");
		let lines = text.lines().collect::<Vec<_>>();
		assert_eq!(lines.len(), 3);
		assert!(lines[0].starts_with("epoch,step,batch_size,sequence_length"));
		assert!(lines[1].starts_with("1,1,4,8,"));
		assert!(lines[2].starts_with("2,2,4,8,"));
		assert!(lines[1].contains("\"token,utf8\""));
		assert!(lines[1].contains("\"byte\"\"raw\""));
		Ok(())
	}
);
