use std::time::SystemTime;

use oa::ml::{Adam, AdamW, Module, ModuleRegistry, Muon, Optimizer, Parameter, Sgd};

struct CheckpointModule {
	registry: ModuleRegistry,
	parameter: Parameter,
}

impl CheckpointModule {
	fn new(engine: &oa::Engine, weight: f32, persistent: u32, scratch: i32) -> oa::Result<Self> {
		let parameter = Parameter::new(
			"value",
			oa::Matrix::from_f32(engine, [1, 2], &[weight, weight + 1.0])?,
		)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("value", parameter.clone())?;
		registry.register_buffer(
			"alphabet",
			oa::Matrix::from_slice(engine, [1], &[persistent])?,
			true,
		)?;
		registry.register_buffer(
			"scratch",
			oa::Matrix::from_slice(engine, [1], &[scratch])?,
			false,
		)?;
		Ok(Self {
			registry,
			parameter,
		})
	}
}

impl Module for CheckpointModule {
	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		Ok(input.clone())
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

test_vk!(
	native_oam_roundtrip_integrity_and_legacy_checksums,
	engine,
	{
		let directory = std::env::temp_dir().join(format!(
			"oars-oam-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create .oam test directory");
		let path = directory.join("model.oam");

		let source = CheckpointModule::new(&engine, 3.0, 97, -7)?;
		let source_optimizer = AdamW::new(source.all_parameters()?, 0.025)?;
		oa::ml::save_checkpoint(&path, &source, &source_optimizer)?;
		let first = std::fs::read(&path).expect("read native .oam");
		assert_eq!(&first[..4], b"OAM\0");
		assert_eq!(u32_at(&first, 4), 3);
		assert_eq!(u64_at(&first, 16), first.len() as u64);
		assert_eq!(u64_at(&first, 16) % 4, 0);
		let section_count = u32_at(&first, 8) as usize;
		assert_eq!(section_count, 5);
		assert_eq!(u64_at(&first, 64 + 12), 4096);
		for index in 0..section_count {
			let base = 64 + index * 64;
			let kind = u32_at(&first, base);
			let offset = u64_at(&first, base + 12);
			let size = u64_at(&first, base + 20);
			if matches!(kind, 2 | 3) && index + 1 < section_count {
				let next_offset = u64_at(&first, 64 + (index + 1) * 64 + 12);
				assert_eq!(next_offset, offset + size.div_ceil(4096) * 4096);
			}
		}

		// Saving a second complete artifact to the same path exercises atomic
		// replacement rather than in-place mutation.
		let replacement = CheckpointModule::new(&engine, 9.0, 98, -8)?;
		let replacement_optimizer = AdamW::new(replacement.all_parameters()?, 0.05)?;
		oa::ml::save_checkpoint(&path, &replacement, &replacement_optimizer)?;
		let valid = std::fs::read(&path).expect("read replaced .oam");
		assert_ne!(first, valid);

		let destination = CheckpointModule::new(&engine, -1.0, 1, 123)?;
		let mut destination_optimizer = AdamW::new(destination.all_parameters()?, 1.0e-3)?;
		oa::ml::load_checkpoint(&engine, &path, &destination, &mut destination_optimizer)?;
		assert_eq!(destination.parameter.data().read_f32()?, [9.0, 10.0]);
		let buffers = destination.all_named_buffers()?;
		assert_eq!(buffers[0].path(), "alphabet");
		assert_eq!(buffers[0].data().read::<u32>()?, [98]);
		assert_eq!(buffers[1].path(), "scratch");
		assert_eq!(buffers[1].data().read::<i32>()?, [123]);
		assert_eq!(destination_optimizer.step_count(), 0);

		if let Ok(modelctl) = std::env::var("OA_CPP_MODELCTL") {
			let status = std::process::Command::new(&modelctl)
				.args(["verify", path.to_str().expect("UTF-8 temporary path")])
				.status()
				.expect("run OA C++ modelctl");
			assert!(status.success(), "OA C++ rejected Rust-written .oam");

			let cpp_path = directory.join("cpp-rewritten.oam");
			let status = std::process::Command::new(modelctl)
				.args([
					"quantize",
					path.to_str().expect("UTF-8 temporary path"),
					"--out",
					cpp_path.to_str().expect("UTF-8 temporary path"),
					"--dtype",
					"q4",
				])
				.status()
				.expect("run OA C++ modelctl quantize");
			assert!(status.success(), "OA C++ could not rewrite Rust .oam");
			let cpp_destination = CheckpointModule::new(&engine, 0.0, 0, 0)?;
			let mut cpp_optimizer = AdamW::new(cpp_destination.all_parameters()?, 0.01)?;
			let error = oa::ml::load_checkpoint(&engine, &cpp_path, &cpp_destination, &mut cpp_optimizer)
				.expect_err("inference-only C++ artifact retained optimizer state");
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
			assert_eq!(
				error.message(),
				"checkpoint restore requires dense tensor value"
			);
		}

		let weights = section(&valid, 2);
		let mut corrupt = valid.clone();
		corrupt[weights.end - 1] ^= 1;
		let corrupt_path = directory.join("corrupt.oam");
		std::fs::write(&corrupt_path, corrupt).expect("write corrupt .oam");
		assert_corrupt(&engine, &corrupt_path)?;

		let mut metadata_corrupt = valid.clone();
		metadata_corrupt[64 + 8] ^= 1;
		let metadata_path = directory.join("metadata-corrupt.oam");
		std::fs::write(&metadata_path, metadata_corrupt).expect("write metadata-corrupt .oam");
		assert_corrupt(&engine, &metadata_path)?;

		let truncated_path = directory.join("truncated.oam");
		std::fs::write(&truncated_path, &valid[..valid.len() - 7]).expect("write truncated .oam");
		assert_corrupt(&engine, &truncated_path)?;

		for version in [1_u32, 2] {
			let mut legacy = valid.clone();
			legacy[4..8].copy_from_slice(&version.to_le_bytes());
			legacy[24..32].fill(0);
			let checksum = if version == 1 {
				(0..section_count).fold(0_u64, |checksum, index| {
					checksum ^ u64_at(&legacy, 64 + index * 64 + 36)
				})
			} else {
				hash(&legacy[..64 + section_count * 64])
			};
			legacy[24..32].copy_from_slice(&checksum.to_le_bytes());
			let legacy_path = directory.join(format!("model-v{version}.oam"));
			std::fs::write(&legacy_path, legacy).expect("write legacy .oam");
			let legacy_destination = CheckpointModule::new(&engine, 0.0, 0, 0)?;
			let mut legacy_optimizer = AdamW::new(legacy_destination.all_parameters()?, 0.01)?;
			oa::ml::load_checkpoint(
				&engine,
				&legacy_path,
				&legacy_destination,
				&mut legacy_optimizer,
			)?;
			assert_eq!(legacy_destination.parameter.data().read_f32()?, [9.0, 10.0]);
		}

		std::fs::remove_dir_all(directory).expect("remove .oam test directory");
		Ok(())
	}
);

test_vk!(
	native_oam_roundtrips_sgd_adam_and_muon_optimizer_state,
	engine,
	{
		let directory = std::env::temp_dir().join(format!(
			"oars-optimizer-oam-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create optimizer checkpoint directory");
		let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;

		let adam_source = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x4144_414d)?;
		let mut adam = Adam::new(adam_source.all_parameters()?, 0.025)?;
		train_linear_once(&adam_source, &input, &targets, &mut adam)?;
		let adam_path = directory.join("adam.oam");
		oa::ml::save_checkpoint(&adam_path, &adam_source, &adam)?;
		let adam_destination = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 7)?;
		let mut restored_adam = Adam::new(adam_destination.all_parameters()?, 0.5)?;
		oa::ml::load_checkpoint(&engine, &adam_path, &adam_destination, &mut restored_adam)?;
		assert_eq!(restored_adam.step_count(), 1);
		assert_eq!(restored_adam.learning_rate(), 0.025);
		assert_module_parameters_equal(&adam_source, &adam_destination)?;

		let sgd_source = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x5347_4400)?;
		let mut sgd = Sgd::new(sgd_source.all_parameters()?, 0.05, 0.0, 0.01)?;
		train_linear_once(&sgd_source, &input, &targets, &mut sgd)?;
		let sgd_path = directory.join("sgd.oam");
		oa::ml::save_checkpoint(&sgd_path, &sgd_source, &sgd)?;
		let sgd_destination = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 11)?;
		let mut restored_sgd = Sgd::new(sgd_destination.all_parameters()?, 0.5, 0.0, 0.0)?;
		oa::ml::load_checkpoint(&engine, &sgd_path, &sgd_destination, &mut restored_sgd)?;
		assert_eq!(restored_sgd.step_count(), 1);
		assert_eq!(restored_sgd.learning_rate(), 0.05);
		assert_module_parameters_equal(&sgd_source, &sgd_destination)?;

		let muon_source = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x4d55_4f4e)?;
		let mut muon = Muon::new(muon_source.all_parameters()?, 0.02)?;
		train_linear_once(&muon_source, &input, &targets, &mut muon)?;
		let muon_path = directory.join("muon.oam");
		oa::ml::save_checkpoint(&muon_path, &muon_source, &muon)?;
		let muon_destination = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 13)?;
		let mut restored_muon = Muon::new(muon_destination.all_parameters()?, 0.5)?;
		oa::ml::load_checkpoint(&engine, &muon_path, &muon_destination, &mut restored_muon)?;
		assert_eq!(restored_muon.step_count(), 1);
		assert_eq!(restored_muon.learning_rate(), 0.02);
		assert_module_parameters_equal(&muon_source, &muon_destination)?;
		train_linear_once(&muon_source, &input, &targets, &mut muon)?;
		train_linear_once(&muon_destination, &input, &targets, &mut restored_muon)?;
		assert_module_parameters_equal(&muon_source, &muon_destination)?;

		let momentum_model = oa::ml::nn::Linear::with_seed(&engine, 2, 2, 0x4d4f_4d45)?;
		let mut momentum = Sgd::new(momentum_model.all_parameters()?, 0.05, 0.9, 0.0)?;
		train_linear_once(&momentum_model, &input, &targets, &mut momentum)?;
		let error = oa::ml::save_checkpoint(directory.join("momentum.oam"), &momentum_model, &momentum)
			.expect_err("live SGD momentum was silently omitted from .oam");
		assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);

		std::fs::remove_dir_all(directory).expect("remove optimizer checkpoint directory");
		Ok(())
	}
);

test_vk!(
	checkpoint_manager_tracks_best_latest_and_rotation,
	engine,
	{
		let directory = std::env::temp_dir().join(format!(
			"oars-checkpoint-manager-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		let config = oa::ml::CheckpointManagerConfig {
			directory: directory.clone(),
			model_name: "Tiny".to_owned(),
			context: "train".to_owned(),
			max_keep: 2,
			save_best: true,
			metric_name: "loss".to_owned(),
			lower_is_better: true,
		};
		let source = CheckpointModule::new(&engine, 4.0, 9, 0)?;
		let mut optimizer = AdamW::new(source.all_parameters()?, 0.01)?;
		let mut manager = oa::ml::CheckpointManager::new(&engine, config.clone())?;
		for (metric, expected_improvement) in [(3.0, true), (2.0, true), (2.5, false)] {
			optimizer.step()?;
			assert_eq!(
				manager.maybe_save(
					&source,
					&optimizer,
					u64::from(optimizer.step_count()),
					metric,
					true,
				)?,
				expected_improvement
			);
		}
		assert_eq!(manager.best_metric(), 2.0);
		assert!(manager.master_path().is_file());
		let incrementals = std::fs::read_dir(manager.incremental_directory())
			.expect("read incremental checkpoint directory")
			.collect::<std::io::Result<Vec<_>>>()
			.expect("read incremental checkpoint entries");
		assert_eq!(incrementals.len(), 2);

		let latest_model = CheckpointModule::new(&engine, -1.0, 0, 0)?;
		let mut latest_optimizer = AdamW::new(latest_model.all_parameters()?, 0.5)?;
		let scanning_manager = oa::ml::CheckpointManager::new(&engine, config.clone())?;
		scanning_manager.load_latest_into(&latest_model, &mut latest_optimizer)?;
		assert_eq!(latest_optimizer.step_count(), 3);
		let latest_path = incrementals
			.iter()
			.map(std::fs::DirEntry::path)
			.find(|path| {
				path
					.file_name()
					.and_then(|name| name.to_str())
					.is_some_and(|name| name.contains("_step3_"))
			})
			.expect("step-three checkpoint exists");
		let mismatched_path = latest_path.with_file_name("Tiny_train_step99_loss2.50.oam");
		std::fs::rename(&latest_path, &mismatched_path).expect("rename checkpoint step segment");
		let mismatched_manager = oa::ml::CheckpointManager::new(&engine, config.clone())?;
		let error = mismatched_manager
			.load_latest_into(&latest_model, &mut latest_optimizer)
			.expect_err("filename/progress step mismatch was accepted");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		assert!(error.message().contains("filename step"));
		assert!(
			manager
				.maybe_save(
					&source,
					&optimizer,
					u64::from(optimizer.step_count()),
					f64::MAX,
					true,
				)
				.is_err()
		);

		let best_model = CheckpointModule::new(&engine, -2.0, 0, 0)?;
		let mut best_optimizer = AdamW::new(best_model.all_parameters()?, 0.5)?;
		manager.load_best_into(&best_model, &mut best_optimizer)?;
		assert_eq!(best_optimizer.step_count(), 2);
		assert_eq!(best_model.parameter.data().read_f32()?, [4.0, 5.0]);

		let invalid = oa::ml::CheckpointManagerConfig {
			model_name: "../escape".to_owned(),
			..Default::default()
		};
		assert!(oa::ml::CheckpointManager::new(&engine, invalid).is_err());
		std::fs::remove_dir_all(directory).expect("remove checkpoint manager directory");
		Ok(())
	}
);

fn train_linear_once(
	model: &oa::ml::nn::Linear,
	input: &oa::Matrix,
	targets: &oa::Matrix,
	optimizer: &mut dyn Optimizer,
) -> oa::Result<()> {
	optimizer.zero_grad();
	let tape = oa::ml::GradientTape::new();
	let loss = oa::ml::loss::cross_entropy(&model.forward(input)?, targets)?;
	tape.backward(&loss)?;
	optimizer.step()?;
	Ok(())
}

fn assert_module_parameters_equal(left: &dyn Module, right: &dyn Module) -> oa::Result<()> {
	let left = left.all_named_parameters()?;
	let right = right.all_named_parameters()?;
	assert_eq!(left.len(), right.len());
	for (left, right) in left.iter().zip(&right) {
		assert_eq!(left.path(), right.path());
		assert_eq!(
			left.parameter().data().read_f32()?,
			right.parameter().data().read_f32()?
		);
	}
	Ok(())
}

fn section(bytes: &[u8], requested: u32) -> std::ops::Range<usize> {
	let count = u32_at(bytes, 8) as usize;
	for index in 0..count {
		let base = 64 + index * 64;
		if u32_at(bytes, base) == requested {
			let start = u64_at(bytes, base + 12) as usize;
			let size = u64_at(bytes, base + 20) as usize;
			return start..start + size;
		}
	}
	panic!("missing .oam section {requested}")
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
	u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
	u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn hash(bytes: &[u8]) -> u64 {
	bytes.iter().fold(0xcbf2_9ce4_8422_2325, |value, byte| {
		(value ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
	})
}

fn assert_corrupt(engine: &oa::Engine, path: &std::path::Path) -> oa::Result<()> {
	let destination = CheckpointModule::new(engine, 0.0, 0, 0)?;
	let mut optimizer = AdamW::new(destination.all_parameters()?, 0.01)?;
	assert_eq!(
		oa::ml::load_checkpoint(engine, path, &destination, &mut optimizer)
			.expect_err("corrupt .oam was accepted")
			.kind(),
		oa::ErrorKind::CheckpointCorrupt
	);
	Ok(())
}
