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
	synthetic_human_ml3d_trains_and_round_trips_one_complete_alm,
	engine,
	{
		use std::{rc::Rc, time::SystemTime};

		use oa::sdk::{
			data::HumanMl3dDataset,
			ml::alm::{
				Alm, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig,
				training::{
					AlmTrainingConfig, PriorTrainingConfig, TokenizerTrainingConfig, train_alm,
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
		let report = train_alm(
			&engine,
			&dataset,
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
				},
				text_seed: 42,
			},
		)?;
		assert_eq!(report.corpus_tokens, 4);
		assert_eq!(report.tokenizer.steps, 4);
		assert_eq!(report.prior.steps, 3);
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
