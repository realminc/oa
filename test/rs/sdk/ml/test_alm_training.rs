use oa::sdk::ml::alm::training::{
	PriorSpecialTokens, PriorWindow, TokenizerWindow, build_prior_windows, build_tokenizer_windows,
	gather_prior_batch,
};

#[test]
fn tokenizer_windows_match_half_stride_and_exact_tail_contract() -> oa::Result<()> {
	let windows = build_tokenizer_windows([3, 4, 5, 8, 9], 4)?;
	assert_eq!(
		windows,
		vec![
			TokenizerWindow { clip: 1, start: 0 },
			TokenizerWindow { clip: 2, start: 0 },
			TokenizerWindow { clip: 2, start: 1 },
			TokenizerWindow { clip: 3, start: 0 },
			TokenizerWindow { clip: 3, start: 2 },
			TokenizerWindow { clip: 3, start: 4 },
			TokenizerWindow { clip: 4, start: 0 },
			TokenizerWindow { clip: 4, start: 2 },
			TokenizerWindow { clip: 4, start: 4 },
			TokenizerWindow { clip: 4, start: 5 },
		]
	);
	Ok(())
}

#[test]
fn prior_windows_preserve_true_boundaries_and_all_start_density() -> oa::Result<()> {
	let sequences = vec![vec![], vec![4, 5], vec![0, 1, 2, 3, 4]];
	let windows = build_prior_windows(&sequences, 3)?;
	assert_eq!(
		windows,
		vec![
			PriorWindow {
				sequence: 0,
				start: 0,
				valid: 1,
			},
			PriorWindow {
				sequence: 1,
				start: 0,
				valid: 3,
			},
			PriorWindow {
				sequence: 2,
				start: 0,
				valid: 3,
			},
			PriorWindow {
				sequence: 2,
				start: 1,
				valid: 3,
			},
			PriorWindow {
				sequence: 2,
				start: 2,
				valid: 3,
			},
			PriorWindow {
				sequence: 2,
				start: 3,
				valid: 3,
			},
		]
	);
	Ok(())
}

#[test]
fn prior_batch_matches_som_motion_eom_pad_and_mask_oracle() -> oa::Result<()> {
	let sequences = vec![vec![2, 4], vec![1, 3, 0, 2]];
	let windows = build_prior_windows(&sequences, 4)?;
	let special = PriorSpecialTokens::after_codebook(5)?;
	let batch = gather_prior_batch(&sequences, &windows, 0, 2, 4, special)?;
	assert_eq!(batch.input_ids, vec![5, 2, 4, 7, 5, 1, 3, 0]);
	assert_eq!(batch.target_ids, vec![2, 4, 6, 7, 1, 3, 0, 2]);
	assert_eq!(
		batch.loss_mask,
		vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
	);
	assert_eq!(batch.valid_count, 7);
	Ok(())
}

#[test]
fn malformed_windows_and_tokens_are_rejected() -> oa::Result<()> {
	assert!(build_tokenizer_windows([4], 0).is_err());
	assert!(build_prior_windows(&[vec![1]], 0).is_err());
	let special = PriorSpecialTokens::after_codebook(2)?;
	assert!(
		gather_prior_batch(
			&[vec![2]],
			&[PriorWindow {
				sequence: 0,
				start: 0,
				valid: 2,
			}],
			0,
			1,
			2,
			special,
		)
		.is_err()
	);
	Ok(())
}

test_vk!(
	tiny_prior_training_uses_shared_iterator_and_autograd_lifecycle,
	engine,
	{
		use oa::sdk::ml::alm::{
			AlmPrior, AlmPriorConfig,
			training::{PriorTrainingConfig, train_prior},
		};
		let mut model_config = AlmPriorConfig {
			model_width: 8,
			num_heads: 1,
			num_layers: 1,
			hidden_width: 16,
			batch_size: 2,
			sequence_length: 4,
			max_sequence_length: 8,
			..AlmPriorConfig::default()
		};
		model_config.sync_vocab(4)?;
		let prior = AlmPrior::with_seed(&engine, model_config, 7)?;
		let report = train_prior(
			&engine,
			&prior,
			&[vec![0, 1, 2], vec![1, 2, 3]],
			None,
			PriorTrainingConfig {
				epochs: 3,
				batch_size: 2,
				window_len: 4,
				learning_rate: 1.0e-3,
				minimum_learning_rate: 1.0e-4,
				warmup_steps: 1,
				weight_decay: 0.0,
				enable_gpu_timing: true,
				show_progress: false,
				checkpoint: None,
			},
		)?;
		assert_eq!(report.steps, 3);
		assert_eq!(report.steps_per_epoch, 1);
		assert!(report.initial_loss.is_finite());
		assert!(report.final_loss.is_finite());
		assert!(report.gpu_mean_ms > 0.0);
		Ok(())
	}
);

test_vk!(
	tiny_tokenizer_training_runs_reconstruction_velocity_and_ema_paths,
	engine,
	{
		use oa::sdk::ml::alm::{
			AlmTokenizer, AlmTokenizerConfig,
			training::{TokenizerTrainingConfig, train_tokenizer},
		};
		let tokenizer = AlmTokenizer::with_seed(
			&engine,
			AlmTokenizerConfig {
				input_dim: 3,
				width: 4,
				code_dim: 4,
				num_codes: 2,
				downsample_stages: 1,
				depth: 1,
				commitment_beta: 0.25,
				ema_decay: 0.9,
				ema_epsilon: 1.0e-5,
				dead_threshold: 0.0,
			},
			11,
		)?;
		let clips = vec![
			vec![0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8],
			vec![0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0],
		];
		let report = train_tokenizer(
			&engine,
			&tokenizer,
			&clips,
			TokenizerTrainingConfig {
				epochs: 3,
				batch_size: 2,
				sequence_len: 4,
				learning_rate: 1.0e-3,
				minimum_learning_rate: 1.0e-4,
				warmup_steps: 1,
				weight_decay: 0.0,
				seed_codebook: true,
				enable_gpu_timing: true,
				show_progress: false,
				checkpoint: None,
			},
		)?;
		assert_eq!(report.steps, 3);
		assert_eq!(report.steps_per_epoch, 1);
		assert_eq!(report.ema_steps, 3);
		assert!(report.initial_reconstruction_loss.is_finite());
		assert!(report.final_reconstruction_loss.is_finite());
		assert!(report.gpu_mean_ms > 0.0);
		Ok(())
	}
);

test_vk!(
	tokenizer_checkpoint_resume_matches_uninterrupted_parameters_and_ema_state,
	engine,
	{
		use std::time::SystemTime;

		use oa::{
			ml::Module,
			sdk::ml::alm::{
				AlmTokenizer, AlmTokenizerConfig,
				training::{StageCheckpointConfig, TokenizerTrainingConfig, train_tokenizer},
			},
		};
		let directory = std::env::temp_dir().join(format!(
			"oars-alm-resume-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		let model_config = AlmTokenizerConfig {
			input_dim: 3,
			width: 4,
			code_dim: 4,
			num_codes: 2,
			downsample_stages: 1,
			depth: 1,
			commitment_beta: 0.25,
			ema_decay: 0.9,
			ema_epsilon: 1.0e-5,
			dead_threshold: 0.0,
		};
		let clips = vec![
			vec![0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8],
			vec![0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0],
		];
		let checkpoint = StageCheckpointConfig {
			directory: directory.clone(),
			model_name: "TokenizerResume".into(),
			context: "gate".into(),
			max_keep: 8,
			save_every: 1,
			resume: false,
			restore_best: false,
			verbose: false,
		};
		let training_config = TokenizerTrainingConfig {
			epochs: 2,
			batch_size: 1,
			sequence_len: 4,
			learning_rate: 1.0e-3,
			minimum_learning_rate: 1.0e-4,
			warmup_steps: 1,
			weight_decay: 0.0,
			seed_codebook: true,
			enable_gpu_timing: false,
			show_progress: false,
			checkpoint: Some(checkpoint.clone()),
		};
		let uninterrupted = AlmTokenizer::with_seed(&engine, model_config, 29)?;
		let uninterrupted_report =
			train_tokenizer(&engine, &uninterrupted, &clips, training_config.clone())?;
		assert_eq!(uninterrupted_report.steps, 4);

		let incremental = directory.join("TokenizerResume").join("checkpoint_gate");
		for entry in std::fs::read_dir(&incremental).expect("read tokenizer checkpoints") {
			let path = entry.expect("read tokenizer checkpoint entry").path();
			if !path
				.file_name()
				.and_then(|name| name.to_str())
				.is_some_and(|name| name.contains("_step1_"))
			{
				std::fs::remove_file(path).expect("remove later synthetic checkpoint");
			}
		}
		let resumed = AlmTokenizer::with_seed(&engine, model_config, 991)?;
		let mut resumed_config = training_config;
		resumed_config
			.checkpoint
			.as_mut()
			.expect("checkpoint policy")
			.resume = true;
		let resumed_report = train_tokenizer(&engine, &resumed, &clips, resumed_config)?;
		assert_eq!(resumed_report.resumed_from_step, 1);
		assert_eq!(resumed_report.steps, 4);
		assert_eq!(resumed_report.ema_steps, 4);

		for (expected, actual) in uninterrupted
			.all_named_parameters()?
			.into_iter()
			.zip(resumed.all_named_parameters()?)
		{
			assert_eq!(expected.path(), actual.path());
			let expected_values = expected.parameter().data().read_f32()?;
			let actual_values = actual.parameter().data().read_f32()?;
			let maximum_error = expected_values
				.iter()
				.zip(&actual_values)
				.map(|(left, right)| (left - right).abs())
				.fold(0.0_f32, f32::max);
			// Restored execution rebuilds transient graphs and is numerically, not
			// bitwise, equivalent across the Vulkan reduction schedule.
			assert!(
				maximum_error <= 2.0e-3,
				"parameter {} diverged after resume",
				expected.path()
			);
		}
		for (expected, actual) in uninterrupted
			.all_named_buffers()?
			.into_iter()
			.filter(|buffer| buffer.persistent())
			.zip(
				resumed
					.all_named_buffers()?
					.into_iter()
					.filter(|buffer| buffer.persistent()),
			) {
			assert_eq!(expected.path(), actual.path());
			let expected_values = expected.data().read_f32()?;
			let actual_values = actual.data().read_f32()?;
			let maximum_error = expected_values
				.iter()
				.zip(&actual_values)
				.map(|(left, right)| (left - right).abs())
				.fold(0.0_f32, f32::max);
			assert_eq!(
				maximum_error,
				0.0,
				"buffer {} was not restored exactly",
				expected.path()
			);
		}
		assert_eq!(
			uninterrupted
				.all_named_state_u32()?
				.into_iter()
				.map(|state| (state.path().to_owned(), state.value()))
				.collect::<Vec<_>>(),
			resumed
				.all_named_state_u32()?
				.into_iter()
				.map(|state| (state.path().to_owned(), state.value()))
				.collect::<Vec<_>>()
		);
		std::fs::remove_dir_all(directory).expect("remove tokenizer resume fixture");
		Ok(())
	}
);

test_vk!(
	prior_checkpoint_resume_preserves_absolute_cursor_and_optimizer_state,
	engine,
	{
		use std::time::SystemTime;

		use oa::{
			ml::Module,
			sdk::ml::alm::{
				AlmPrior, AlmPriorConfig,
				training::{PriorTrainingConfig, StageCheckpointConfig, train_prior},
			},
		};
		let directory = std::env::temp_dir().join(format!(
			"oars-alm-prior-resume-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		let mut model_config = AlmPriorConfig {
			model_width: 8,
			num_heads: 1,
			num_layers: 1,
			hidden_width: 16,
			batch_size: 2,
			sequence_length: 4,
			max_sequence_length: 8,
			..AlmPriorConfig::default()
		};
		model_config.sync_vocab(4)?;
		let sequences = vec![vec![0, 1, 2], vec![1, 2, 3]];
		let checkpoint = StageCheckpointConfig {
			directory: directory.clone(),
			model_name: "PriorResume".into(),
			context: "gate".into(),
			max_keep: 8,
			save_every: 1,
			resume: false,
			restore_best: false,
			verbose: false,
		};
		let training_config = PriorTrainingConfig {
			epochs: 3,
			batch_size: 2,
			window_len: 4,
			learning_rate: 1.0e-3,
			minimum_learning_rate: 1.0e-4,
			warmup_steps: 1,
			weight_decay: 0.0,
			enable_gpu_timing: false,
			show_progress: false,
			checkpoint: Some(checkpoint),
		};
		let uninterrupted = AlmPrior::with_seed(&engine, model_config, 41)?;
		let uninterrupted_report = train_prior(
			&engine,
			&uninterrupted,
			&sequences,
			None,
			training_config.clone(),
		)?;
		assert_eq!(uninterrupted_report.steps, 3);
		let incremental = directory.join("PriorResume").join("checkpoint_gate");
		for entry in std::fs::read_dir(&incremental).expect("read prior checkpoints") {
			let path = entry.expect("read prior checkpoint entry").path();
			if !path
				.file_name()
				.and_then(|name| name.to_str())
				.is_some_and(|name| name.contains("_step1_"))
			{
				std::fs::remove_file(path).expect("remove later synthetic checkpoint");
			}
		}
		let resumed = AlmPrior::with_seed(&engine, model_config, 999)?;
		let mut resumed_config = training_config;
		resumed_config
			.checkpoint
			.as_mut()
			.expect("checkpoint policy")
			.resume = true;
		let resumed_report = train_prior(&engine, &resumed, &sequences, None, resumed_config)?;
		assert_eq!(resumed_report.resumed_from_step, 1);
		assert_eq!(resumed_report.steps, 3);
		for (expected, actual) in uninterrupted
			.all_named_parameters()?
			.into_iter()
			.zip(resumed.all_named_parameters()?)
		{
			assert_eq!(expected.path(), actual.path());
			let expected_values = expected.parameter().data().read_f32()?;
			let actual_values = actual.parameter().data().read_f32()?;
			let maximum_error = expected_values
				.iter()
				.zip(&actual_values)
				.map(|(left, right)| (left - right).abs())
				.fold(0.0_f32, f32::max);
			assert!(
				maximum_error <= 2.0e-3,
				"prior parameter {} diverged after resume",
				expected.path()
			);
		}
		std::fs::remove_dir_all(directory).expect("remove prior resume fixture");
		Ok(())
	}
);

test_vk!(
	synthetic_human_ml3d_trains_and_round_trips_one_complete_alm,
	engine,
	{
		use std::{rc::Rc, time::SystemTime};

		use oa::sdk::{
			data::HumanMl3dDataset,
			ml::alm::{
				Alm, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig,
				training::{
					AlmTrainingConfig, AlmValidation, PriorTrainingConfig, PriorValidationConfig,
					TokenizerTrainingConfig, TokenizerValidationConfig, evaluate_prior,
					evaluate_tokenizer, tokenize_corpus, train_alm_with_validation,
				},
			},
		};

		let directory = std::env::temp_dir().join(format!(
			"oars-alm-training-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(directory.join("new_joint_vecs"))
			.expect("create motion fixture directory");
		std::fs::create_dir_all(directory.join("texts")).expect("create caption fixture directory");
		write_npy(&directory.join("Mean.npy"), &[263], &vec![0.0; 263]);
		write_npy(&directory.join("Std.npy"), &[263], &vec![1.0; 263]);
		for (index, name) in ["first", "second"].into_iter().enumerate() {
			let values = (0..5 * 263)
				.map(|offset| ((offset + index * 17) as f32 * 0.013).sin() * 0.1)
				.collect::<Vec<_>>();
			write_npy(
				&directory.join("new_joint_vecs").join(format!("{name}.npy")),
				&[5, 263],
				&values,
			);
			std::fs::write(
				directory.join("texts").join(format!("{name}.txt")),
				format!("{name} motion#tags#0#0\n"),
			)
			.expect("write caption fixture");
		}
		std::fs::write(directory.join("train.txt"), "first\nsecond\n")
			.expect("write split fixture");
		let dataset = HumanMl3dDataset::open_cmp(&directory, "train", 0)?;
		let tokenizer = Rc::new(AlmTokenizer::with_seed(
			&engine,
			AlmTokenizerConfig {
				input_dim: 263,
				width: 4,
				code_dim: 4,
				num_codes: 2,
				downsample_stages: 1,
				depth: 1,
				commitment_beta: 0.25,
				ema_decay: 0.9,
				ema_epsilon: 1.0e-5,
				dead_threshold: 0.0,
			},
			17,
		)?);
		let mut prior_config = AlmPriorConfig {
			model_width: 8,
			num_heads: 1,
			num_layers: 1,
			hidden_width: 16,
			sequence_length: 3,
			max_sequence_length: 8,
			..AlmPriorConfig::default()
		};
		prior_config.sync_vocab(2)?;
		let prior = Rc::new(AlmPrior::with_seed(&engine, prior_config, 18)?);
		let report = train_alm_with_validation(
			&engine,
			&dataset,
			AlmValidation {
				dataset: &dataset,
				tokenizer: TokenizerValidationConfig {
					sequence_len: 4,
					batch_size: 2,
					max_batches: 1,
				},
				prior: PriorValidationConfig {
					window_len: 3,
					batch_size: 2,
					max_batches: 1,
				},
			},
			tokenizer,
			prior,
			AlmTrainingConfig {
				tokenizer: TokenizerTrainingConfig {
					epochs: 2,
					batch_size: 2,
					sequence_len: 4,
					learning_rate: 1.0e-3,
					minimum_learning_rate: 1.0e-4,
					warmup_steps: 1,
					weight_decay: 0.0,
					seed_codebook: true,
					enable_gpu_timing: false,
					show_progress: false,
					checkpoint: None,
				},
				prior: PriorTrainingConfig {
					epochs: 3,
					batch_size: 2,
					window_len: 3,
					learning_rate: 1.0e-3,
					minimum_learning_rate: 1.0e-4,
					warmup_steps: 1,
					weight_decay: 0.0,
					enable_gpu_timing: false,
					show_progress: false,
					checkpoint: None,
				},
				text_seed: 42,
			},
		)?;
		assert_eq!(report.corpus_tokens, 4);
		assert_eq!(report.tokenizer.steps, 4);
		assert_eq!(report.prior.steps, 3);
		assert_eq!(
			report
				.tokenizer
				.validation
				.expect("tokenizer validation")
				.samples,
			2
		);
		assert_eq!(
			report
				.prior
				.validation
				.expect("prior validation")
				.valid_tokens,
			6
		);
		let tokenizer_validation = evaluate_tokenizer(
			&engine,
			report.model.tokenizer(),
			&dataset,
			TokenizerValidationConfig {
				sequence_len: 4,
				batch_size: 2,
				max_batches: 0,
			},
		)?;
		assert_eq!(tokenizer_validation.samples, 4);
		assert_eq!(tokenizer_validation.batches, 2);
		assert_eq!(tokenizer_validation.tokens, 8);
		assert!(tokenizer_validation.reconstruction_loss.is_finite());
		assert!(tokenizer_validation.velocity_loss.is_finite());
		assert!(tokenizer_validation.mpjpe_cm.is_finite());
		assert!((0.0..=1.0).contains(&tokenizer_validation.contact_accuracy));
		assert!((1..=2).contains(&tokenizer_validation.live_codes));
		let sequences = tokenize_corpus(&engine, report.model.tokenizer(), &dataset)?;
		let prior_validation = evaluate_prior(
			&engine,
			report.model.prior(),
			&sequences,
			None,
			PriorValidationConfig {
				window_len: 3,
				batch_size: 2,
				max_batches: 0,
			},
		)?;
		assert_eq!(prior_validation.batches, 1);
		assert_eq!(prior_validation.valid_tokens, 6);
		assert_eq!(prior_validation.eos_tokens, 2);
		assert!(prior_validation.loss.is_finite());
		assert!(prior_validation.perplexity.is_finite());
		assert!((0.0..=1.0).contains(&prior_validation.token_accuracy));
		assert!((0.0..=1.0).contains(&prior_validation.eos_accuracy));
		let path = directory.join("Alm.oam");
		report.model.save_bundle(&path)?;
		let restored = Alm::load_bundle(&engine, &path)?;
		assert_eq!(restored.config(), report.model.config());
		std::fs::remove_dir_all(directory).expect("remove ALM fixture");
		Ok(())
	}
);

fn write_npy(path: &std::path::Path, shape: &[usize], values: &[f32]) {
	assert_eq!(shape.iter().product::<usize>(), values.len());
	let dimensions = shape
		.iter()
		.map(usize::to_string)
		.collect::<Vec<_>>()
		.join(", ");
	let comma = if shape.len() == 1 { "," } else { "" };
	let mut header =
		format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}{comma}), }}");
	let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
	header.extend(std::iter::repeat_n(' ', padding));
	header.push('\n');
	let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
	bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
	bytes.extend_from_slice(header.as_bytes());
	for value in values {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	std::fs::write(path, bytes).expect("write NumPy fixture");
}
