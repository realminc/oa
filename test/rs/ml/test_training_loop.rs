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
	eager_training_loop_owns_exact_completion_metrics_and_callbacks,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let config = oa::ml::TrainingLoopConfig {
			total_steps: 4,
			steps_per_epoch: 2,
			batch_size: 2,
			sequence_length: 3,
			source_units_per_sample: 4.0,
			enable_gpu_timing: true,
			..oa::ml::TrainingLoopConfig::default()
		};
		let mut callback = LifecycleRecorder::new();
		let mut loss_metric = oa::ml::LossMetric::default();
		let summary = {
			let mut training = oa::ml::TrainingLoop::new(&engine, &mut optimizer, config)?;
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

		let mut failing = FailingCallback;
		let error = {
			let mut training = oa::ml::TrainingLoop::new(
				&engine,
				&mut optimizer,
				oa::ml::TrainingLoopConfig {
					total_steps: 1,
					..oa::ml::TrainingLoopConfig::default()
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
	captured_training_loop_replays_one_program_without_reauthoring_the_graph,
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
			let config = oa::ml::TrainingLoopConfig {
				total_steps: 3,
				batch_size: 2,
				enable_gpu_timing: true,
				..oa::ml::TrainingLoopConfig::default()
			};
			let mut training = oa::ml::TrainingLoop::new(&engine, &mut optimizer, config)?;
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
	automatic_training_loop_prepares_every_step_and_records_only_capture_boundaries,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = oa::Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let config = oa::ml::TrainingLoopConfig {
			total_steps: 4,
			batch_size: 2,
			..oa::ml::TrainingLoopConfig::default()
		};
		let mut prepare_count = 0_u32;
		let mut record_count = 0_u32;
		let summary = {
			let mut training = oa::ml::TrainingLoop::new(&engine, &mut optimizer, config)?;
			loop {
				let ran = training.step(
					|| {
						prepare_count += 1;
						let scale = prepare_count as f32;
						let input =
							oa::Matrix::from_f32(&engine, [2, 2], &[scale, 0.0, 0.0, scale])?;
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
	automatic_training_loop_rejects_executable_work_during_preparation,
	engine,
	{
		let weight = oa::Matrix::from_f32(&engine, [1, 1], &[1.0])?;
		let bias = oa::Matrix::from_f32(&engine, [1], &[0.0])?;
		let layer = oa::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = oa::ml::AdamW::new(layer.parameters(), 0.01)?;
		let error = {
			let mut training = oa::ml::TrainingLoop::new(
				&engine,
				&mut optimizer,
				oa::ml::TrainingLoopConfig {
					total_steps: 1,
					..oa::ml::TrainingLoopConfig::default()
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
