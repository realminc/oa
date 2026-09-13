use std::time::SystemTime;

use oa::ml::Module as _;

#[test]
fn clip_tokenizer_preserves_canonical_ids_padding_and_failure_state() -> oa::Result<()> {
	let mut merges = String::from("#version: 0.2\n");
	merges.push_str(&"a b\n".repeat(48_894));
	let mut tokenizer = oa::sdk::ml::alm::ClipTokenizer::new();
	tokenizer.load_merges(merges.as_bytes())?;
	assert!(tokenizer.is_loaded());
	assert_eq!(tokenizer.vocab_size(), 49_408);
	assert_eq!(tokenizer.bos_token(), 49_406);
	assert_eq!(tokenizer.eos_token(), 49_407);

	let encoded = tokenizer.encode(&["a!", ""], 6, true)?;
	assert_eq!(encoded.batch, 2);
	assert_eq!(encoded.context_length, 6);
	assert_eq!(
		encoded.token_ids,
		[
			49_406, 320, 256, 49_407, 49_407, 49_407, 49_406, 49_407, 49_407, 49_407, 49_407,
			49_407
		]
	);
	assert_eq!(encoded.flat_eos_rows, [3, 7]);
	assert!(tokenizer.encode(&["a a a"], 2, false).is_err());
	assert!(tokenizer.load_merges(b"bad").is_err());
	assert!(!tokenizer.is_loaded());
	Ok(())
}

#[test]
fn clip_tokenizer_matches_pinned_openai_ids_when_asset_is_available() -> oa::Result<()> {
	let Some(path) = std::env::var_os("OA_CLIP_MERGES") else {
		return Ok(());
	};
	let mut tokenizer = oa::sdk::ml::alm::ClipTokenizer::new();
	tokenizer.load_merges_file(path)?;
	let prompts = [
		"hello world",
		"a person walks forward, turns left, and raises both arms",
		"We're testing UTF-8 café — fast!",
		"",
	];
	let encoded = tokenizer.encode(&prompts, 77, true)?;
	let expected = [
		&[49_406, 3306, 1002, 49_407][..],
		&[
			49_406, 320, 2533, 8192, 2342, 267, 3185, 1823, 267, 537, 13_297, 2212, 5706, 49_407,
		][..],
		&[
			49_406, 649, 982, 4967, 1419, 325, 268, 279, 15_304, 2005, 1953, 256, 49_407,
		][..],
		&[49_406, 49_407][..],
	];
	for (row, expected) in expected.iter().enumerate() {
		assert_eq!(
			&encoded.token_ids[row * 77..row * 77 + expected.len()],
			*expected
		);
		assert_eq!(
			encoded.flat_eos_rows[row] as usize,
			row * 77 + expected.len() - 1
		);
	}
	Ok(())
}

fn tiny_config() -> oa::sdk::ml::alm::AlmTokenizerConfig {
	oa::sdk::ml::alm::AlmTokenizerConfig {
		input_dim: 3,
		width: 4,
		code_dim: 4,
		num_codes: 4,
		downsample_stages: 1,
		depth: 1,
		commitment_beta: 0.25,
		ema_decay: 0.5,
		ema_epsilon: 1.0e-5,
		dead_threshold: 0.0,
	}
}

fn tiny_prior_config(
	ffn_type: oa::sdk::ml::alm::AlmFfnType,
	text_feature_dim: usize,
) -> oa::sdk::ml::alm::AlmPriorConfig {
	oa::sdk::ml::alm::AlmPriorConfig {
		vocab_size: 7,
		num_codes: 4,
		som_token: 4,
		eom_token: 5,
		pad_token: 6,
		model_width: 4,
		num_heads: 1,
		num_layers: 2,
		hidden_width: 8,
		text_feature_dim,
		ffn_type,
		moe_num_experts: 2,
		moe_experts_per_token: 1,
		moe_every: 2,
		moe_balance_rate: 0.01,
		moe_aux_loss_alpha: 0.1,
		moe_router_z_loss_beta: 0.01,
		batch_size: 2,
		sequence_length: 4,
		max_sequence_length: 8,
		learning_rate: 1.0e-4,
		num_epochs: 2,
		temperature: 0.0,
		top_k: 0,
		top_p: 1.0,
		max_generation_length: 3,
	}
}

fn assert_module_state_equal(
	left: &dyn oa::ml::Module,
	right: &dyn oa::ml::Module,
) -> oa::Result<()> {
	let left_parameters = left.all_named_parameters()?;
	let right_parameters = right.all_named_parameters()?;
	assert_eq!(left_parameters.len(), right_parameters.len());
	for (left, right) in left_parameters.iter().zip(&right_parameters) {
		assert_eq!(left.path(), right.path());
		assert_eq!(
			left.parameter().data().read_f32()?,
			right.parameter().data().read_f32()?
		);
	}
	let left_buffers = left.all_named_buffers()?;
	let right_buffers = right.all_named_buffers()?;
	assert_eq!(left_buffers.len(), right_buffers.len());
	for (left, right) in left_buffers.iter().zip(&right_buffers) {
		assert_eq!(left.path(), right.path());
		assert_eq!(left.persistent(), right.persistent());
		if left.persistent() {
			assert_eq!(left.data().read_f32()?, right.data().read_f32()?);
		}
	}
	let left_state = left.all_named_state_u32()?;
	let right_state = right.all_named_state_u32()?;
	assert_eq!(left_state.len(), right_state.len());
	for (left, right) in left_state.iter().zip(&right_state) {
		assert_eq!(left.path(), right.path());
		assert_eq!(left.value(), right.value());
	}
	Ok(())
}

test_vk!(
	alm_tokenizer_trains_tokenizes_and_restores_complete_state,
	engine,
	{
		let config = tiny_config();
		let model = oa::sdk::ml::alm::AlmTokenizer::with_seed(&engine, config, 0xC0FFEE)?;
		assert_eq!(model.config(), &config);
		assert_eq!(model.downsample_factor(), 2);
		let parameter_paths = model
			.all_named_parameters()?
			.into_iter()
			.map(|entry| entry.path().to_owned())
			.collect::<Vec<_>>();
		assert!(parameter_paths.contains(&"enc_in.weight".to_owned()));
		assert!(parameter_paths.contains(&"enc_res0_0_a.weight".to_owned()));
		assert!(parameter_paths.contains(&"dec_up0.weight".to_owned()));
		assert!(parameter_paths.contains(&"dec_ln3.bias".to_owned()));
		assert!(parameter_paths.iter().all(|path| !path.starts_with("rvq.")));

		let input_values = (0..48)
			.map(|index| ((index as f32 * 0.173).sin() + index as f32 * 0.01) * 0.5)
			.collect::<Vec<_>>();
		let input = oa::Matrix::from_f32(&engine, [2, 8, 3], &input_values)?;
		let latent = model.encode(&input)?;
		assert_eq!(latent.shape(), [8, 4]);
		for row in latent.read_f32()?.as_chunks::<4>().0 {
			let rms = (row.iter().map(|value| value * value).sum::<f32>() / 4.0).sqrt();
			assert!((rms - 1.0).abs() <= 2.0e-4, "latent RMS was {rms}");
		}

		model.seed(&latent)?;
		let quantized = model.quantize(&latent)?;
		assert_eq!(quantized.indices.len(), 1);
		assert_eq!(quantized.indices[0].shape(), [8]);
		assert!(quantized.commitment_loss.shape().is_empty());
		assert!(quantized.commitment_loss.read_f32()?[0].is_finite());
		let reconstruction = model.decode(&quantized.quantized, 2)?;
		assert_eq!(reconstruction.shape(), [2, 8, 3]);
		assert!(reconstruction.read_f32()?.into_iter().all(f32::is_finite));

		let tokens = model.tokenize(&input)?;
		let decoded_tokens = model.detokenize(&tokens, 2)?;
		let decoded_lookup = model.decode(&model.rvq().lookup(&tokens)?, 2)?;
		assert_eq!(decoded_tokens.read_f32()?, decoded_lookup.read_f32()?);

		let tape = oa::ml::GradientTape::new();
		let output = model.forward(&input)?;
		assert_eq!(output.shape(), input.shape());
		let target = oa::matrix::full(&engine, [2, 8, 3], 0.0)?;
		let loss = oa::ml::loss::mse(&output, &target)?;
		tape.backward(&loss)?;
		let parameters = model.all_named_parameters()?;
		assert!(!parameters.is_empty());
		for parameter in &parameters {
			let gradient = parameter
				.parameter()
				.gradient()
				.unwrap_or_else(|| panic!("missing ALM gradient for {}", parameter.path()));
			assert_eq!(gradient.shape(), parameter.parameter().data().shape());
			assert!(gradient.read_f32()?.into_iter().all(f32::is_finite));
		}

		model.ema_update(&quantized)?;
		assert_eq!(model.rvq().level(0).expect("level zero").ema_step(), 1);
		let directory = std::env::temp_dir().join(format!(
			"oars-alm-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create ALM checkpoint directory");
		let checkpoint = directory.join("tokenizer.oam");
		let source_optimizer = oa::ml::AdamW::new(model.all_parameters()?, 2.0e-4)?;
		oa::ml::save_checkpoint(&checkpoint, &model, &source_optimizer)?;
		let restored = oa::sdk::ml::alm::AlmTokenizer::with_seed(&engine, config, 9)?;
		let mut restored_optimizer = oa::ml::AdamW::new(restored.all_parameters()?, 1.0e-3)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &restored, &mut restored_optimizer)?;
		assert_module_state_equal(&model, &restored)?;
		assert_eq!(
			model.forward(&input)?.read_f32()?,
			restored.forward(&input)?.read_f32()?
		);
		std::fs::remove_dir_all(directory).expect("remove ALM checkpoint directory");

		assert!(
			model
				.encode(&oa::Matrix::from_f32(&engine, [1, 7, 3], &[0.0; 21])?)
				.is_err()
		);
		assert!(
			model
				.encode(&oa::Matrix::from_f32(&engine, [1, 8, 2], &[0.0; 16])?)
				.is_err()
		);
		Ok(())
	}
);

test_vk!(
	alm_prior_dense_hybrid_conditioning_generation_and_motion_decode,
	engine,
	{
		use oa::sdk::ml::alm::{AlmFfnType, AlmGenerationOptions, AlmPrior};

		let tokens = oa::Matrix::from_slice(&engine, [2, 3], &[4_i32, 0, 1, 4, 2, 3])?;
		let dense = AlmPrior::with_seed(&engine, tiny_prior_config(AlmFfnType::Dense, 0), 17)?;
		let dense_logits = dense.forward(&tokens)?;
		assert_eq!(dense_logits.shape(), [2, 3, 7]);
		assert!(dense_logits.read_f32()?.into_iter().all(f32::is_finite));
		assert!(dense.moe_aux_loss()?.is_none());
		assert!(dense.moe_route_stats()?.is_empty());
		assert!(!dense.layer(0).expect("dense layer zero").is_moe());

		let options = AlmGenerationOptions {
			temperature: 0.0,
			top_k: 0,
			top_p: 1.0,
			max_length: 3,
			seed: 99,
			use_cache: true,
		};
		let first = dense.generate(2, options)?;
		let second = dense.generate(2, options)?;
		assert_eq!(first.read::<i32>()?, second.read::<i32>()?);
		assert_eq!(first.shape()[0], 2);
		assert!((2..=4).contains(&first.shape()[1]));
		for row in first.read::<i32>()?.chunks_exact(first.shape()[1]) {
			assert_eq!(row[0], 4);
			assert!(row.iter().all(|token| (0..7).contains(token)));
		}

		let hybrid = AlmPrior::with_seed(&engine, tiny_prior_config(AlmFfnType::Hybrid, 3), 23)?;
		assert!(!hybrid.layer(0).expect("hybrid layer zero").is_moe());
		assert!(hybrid.layer(1).expect("hybrid layer one").is_moe());
		assert!(hybrid.forward(&tokens).is_err());
		let text = oa::Matrix::from_f32(&engine, [2, 3], &[0.5, -0.25, 1.0, -0.5, 0.75, 0.25])?;
		let tape = oa::ml::GradientTape::new();
		let logits = hybrid.forward_conditioned(&tokens, &text)?;
		assert_eq!(logits.shape(), [2, 3, 7]);
		let task_loss = oa::ml::loss::mse(&logits, &oa::matrix::full(&engine, [2, 3, 7], 0.0)?)?;
		let total_loss = oa::matrix::add(
			&task_loss,
			&hybrid.moe_aux_loss()?.expect("hybrid auxiliary loss"),
		)?;
		tape.backward(&total_loss)?;
		for parameter in hybrid.all_named_parameters()? {
			let gradient = parameter
				.parameter()
				.gradient()
				.unwrap_or_else(|| panic!("missing ALM prior gradient for {}", parameter.path()));
			assert!(gradient.read_f32()?.into_iter().all(f32::is_finite));
		}
		hybrid.update_moe_routing_bias()?;
		let stats = hybrid.moe_route_stats()?;
		assert_eq!(stats.len(), 1);
		assert_eq!(stats[0].load_fraction().len(), 2);
		let generated = hybrid.generate_conditioned(&text, options)?;
		assert_eq!(generated.shape()[0], 2);

		let tokenizer = oa::sdk::ml::alm::AlmTokenizer::with_seed(&engine, tiny_config(), 31)?;
		let generated_codes =
			oa::Matrix::from_slice(&engine, [2, 4], &[4_i32, 0, 1, 5, 4, 2, 3, 5])?;
		let motion = dense
			.decode_to_motion(&generated_codes, &tokenizer)?
			.expect("two motion tokens per row");
		assert_eq!(motion.shape(), [2, 4, 3]);
		assert!(motion.read_f32()?.into_iter().all(f32::is_finite));
		assert!(
			dense
				.decode_to_motion(
					&oa::Matrix::from_slice(&engine, [1, 2], &[4_i32, 5])?,
					&tokenizer,
				)?
				.is_none()
		);

		let bundle = oa::sdk::ml::alm::Alm::with_seed(
			&engine,
			oa::sdk::ml::alm::AlmConfig {
				tokenizer: tiny_config(),
				prior: tiny_prior_config(AlmFfnType::Dense, 0),
				clip_text: None,
			},
			41,
		)?;
		let bundle_paths = bundle
			.all_named_parameters()?
			.into_iter()
			.map(|entry| entry.path().to_owned())
			.collect::<Vec<_>>();
		assert!(bundle_paths.contains(&"tokenizer.enc_in.weight".to_owned()));
		assert!(bundle_paths.contains(&"prior.token_embed.weight".to_owned()));
		assert_eq!(bundle.forward(&tokens)?.shape(), [2, 3, 7]);
		assert_eq!(
			bundle
				.tokenize(&oa::Matrix::from_f32(&engine, [2, 8, 3], &[0.1; 48])?)?
				.len(),
			1
		);
		let directory = std::env::temp_dir().join(format!(
			"oars-alm-bundle-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create ALM bundle test directory");
		let path = directory.join("alm.oam");
		bundle.save_bundle(&path)?;
		let bytes = std::fs::read(&path).expect("read ALM bundle");
		assert!(bytes.windows(7).any(|window| window == b"OaAlmAg"));
		assert!(
			bytes
				.windows(b"tokenizer.enc_in.weight".len())
				.any(|window| window == b"tokenizer.enc_in.weight")
		);
		assert!(
			bytes
				.windows(b"prior.token_embed.weight".len())
				.any(|window| window == b"prior.token_embed.weight")
		);
		assert!(!bytes.windows(8).any(|window| window == b"ema_step"));
		if let Ok(modelctl) = std::env::var("OA_CPP_MODELCTL") {
			let status = std::process::Command::new(modelctl)
				.args(["verify", path.to_str().expect("UTF-8 temporary path")])
				.status()
				.expect("run OA C++ modelctl");
			assert!(status.success(), "OA C++ rejected Rust-written ALM bundle");
		}
		let restored = oa::sdk::ml::alm::Alm::load_bundle(&engine, &path)?;
		assert_eq!(restored.config().tokenizer, bundle.config().tokenizer);
		assert_eq!(
			restored.config().prior.model_width,
			bundle.config().prior.model_width
		);
		assert_eq!(
			restored.config().prior.num_layers,
			bundle.config().prior.num_layers
		);
		assert_eq!(
			restored.config().prior.max_sequence_length,
			bundle.config().prior.max_sequence_length
		);
		assert_eq!(
			restored.config().prior.max_generation_length,
			bundle.config().prior.max_generation_length
		);
		assert_eq!(restored.text_encoder_identity(), None);
		assert!(!restored.has_native_text_encoder());
		assert_module_state_equal(&bundle, &restored)?;
		std::fs::remove_dir_all(directory).expect("remove ALM bundle test directory");
		Ok(())
	}
);

test_vk!(
	clip_text_matches_explicit_eos_gather_and_conditions_alm,
	engine,
	{
		let config = oa::sdk::ml::alm::ClipTextConfig {
			vocab_size: 8,
			context_length: 4,
			hidden_size: 4,
			intermediate_size: 8,
			num_heads: 1,
			num_layers: 2,
			projection_dim: 3,
			layer_norm_epsilon: 1.0e-5,
			quick_gelu_alpha: 1.702,
			bos_token: 6,
			eos_token: 7,
			pad_token: 7,
		};
		let clip = oa::sdk::ml::alm::ClipText::with_seed(&engine, config, 101)?;
		assert_eq!(clip.config(), &config);
		assert!(
			clip.all_parameters()?
				.into_iter()
				.all(|parameter| !parameter.requires_grad())
		);
		let token_ids = oa::Matrix::from_slice(&engine, [2, 4], &[6_i32, 7, 0, 0, 6, 2, 7, 0])?;
		let eos_rows = oa::Matrix::from_slice(&engine, [2], &[1_i32, 6])?;
		let explicit = clip.forward_tokens(&token_ids, &eos_rows)?;
		let fallback = clip.forward(&token_ids)?;
		assert_eq!(explicit.shape(), [2, 3]);
		assert_eq!(explicit.read_f32()?, fallback.read_f32()?);
		assert!(explicit.read_f32()?.into_iter().all(f32::is_finite));

		let prior = oa::sdk::ml::alm::AlmPrior::with_seed(
			&engine,
			tiny_prior_config(oa::sdk::ml::alm::AlmFfnType::Dense, 3),
			107,
		)?;
		let motion_tokens = oa::Matrix::from_slice(&engine, [2, 3], &[4_i32, 0, 1, 4, 2, 3])?;
		assert_eq!(
			prior
				.forward_conditioned(&motion_tokens, &explicit)?
				.shape(),
			[2, 3, 7]
		);

		let directory = std::env::temp_dir().join(format!(
			"oars-clip-model-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create CLIP model test directory");
		let path = directory.join("clip.oam");
		clip.save_model(&path)?;
		let bytes = std::fs::read(&path).expect("read CLIP model");
		assert!(
			bytes
				.windows(b"text_model.embeddings.token_embedding.weight".len())
				.any(|window| window == b"text_model.embeddings.token_embedding.weight")
		);
		assert!(
			bytes
				.windows(b"text_model.encoder.layers.0.mlp.fc1.weight".len())
				.any(|window| window == b"text_model.encoder.layers.0.mlp.fc1.weight")
		);
		let restored = oa::sdk::ml::alm::ClipText::load_model(&engine, &path)?;
		assert_module_state_equal(&clip, &restored)?;
		assert!(
			restored
				.all_parameters()?
				.into_iter()
				.all(|parameter| !parameter.requires_grad())
		);
		std::fs::remove_dir_all(directory).expect("remove CLIP model test directory");
		Ok(())
	}
);
