use oa::ml::Module;
use oa::sdk::ml::nlp;

struct BpeTrainingRun {
	optimizer: oa::ml::AdamW,
	input: oa::Matrix,
	target: oa::Matrix,
	initial_loss: f32,
	training_loss: f32,
	final_loss: f32,
	accuracy: f32,
	bytes_per_token: f64,
	generated: Vec<u8>,
}

fn train_bpe_model(
	engine: &oa::Engine,
	model: &impl Module,
	tokenizer: &oa::ml::BpeTokenizer,
	learning_rate: f32,
) -> oa::Result<BpeTrainingRun> {
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, learning_rate)?;
	let mut sampler = nlp::BpeSampler::new(nlp::BATCH_SIZE, tokenizer)?;
	let (initial_input, initial_target) = sampler.next_values()?;
	let input = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
		&initial_input,
	)?;
	let target = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&initial_target,
	)?;
	let mut program = oa::ml::TrainingProgram::capture(engine, &mut optimizer, || {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let mut initial_loss = 0.0_f32;
	let mut training_loss = 0.0_f32;
	for step in 0..nlp::TRAINING_STEPS {
		if step != 0 {
			let (next_input, next_target) = sampler.next_values()?;
			program.upload_input(&input, &next_input)?;
			program.upload_input(&target, &next_target)?;
		}
		training_loss = program.replay_and_wait(engine, &mut optimizer)?;
		if step == 0 {
			initial_loss = training_loss;
		}
	}
	let diagnostics = program.diagnostics();
	assert_eq!(diagnostics.command_recording_count(), 1);
	assert_eq!(diagnostics.command_cache_hit_count(), 299);
	assert_eq!(diagnostics.submission_count(), 300);
	assert_eq!(diagnostics.input_upload_count(), 598);
	let logits = model.forward(&input)?;
	let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&logits, &target)?;
	let generated = nlp::generate_bpe_greedy(
		engine,
		model,
		tokenizer,
		nlp::GENERATION_PROMPT.as_bytes(),
		nlp::GENERATION_LENGTH,
	)?;
	Ok(BpeTrainingRun {
		optimizer,
		input,
		target,
		initial_loss,
		training_loss,
		final_loss,
		accuracy,
		bytes_per_token: sampler.last_batch_bytes_per_token(),
		generated,
	})
}

struct ByteTrainingRun {
	optimizer: oa::ml::AdamW,
	input: oa::Matrix,
	target: oa::Matrix,
	initial_loss: f32,
	training_loss: f32,
	final_loss: f32,
	accuracy: f32,
	generated: Vec<u8>,
}

fn train_byte_model(
	engine: &oa::Engine,
	model: &impl Module,
	learning_rate: f32,
) -> oa::Result<ByteTrainingRun> {
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, learning_rate)?;
	let mut sampler = nlp::ByteSampler::new(nlp::BATCH_SIZE)?;
	let (initial_input, initial_target) = sampler.next_values()?;
	let input = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
		&initial_input,
	)?;
	let target = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&initial_target,
	)?;
	let mut program = oa::ml::TrainingProgram::capture(engine, &mut optimizer, || {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let mut initial_loss = 0.0_f32;
	let mut training_loss = 0.0_f32;
	for step in 0..nlp::TRAINING_STEPS {
		if step != 0 {
			let (next_input, next_target) = sampler.next_values()?;
			program.upload_input(&input, &next_input)?;
			program.upload_input(&target, &next_target)?;
		}
		training_loss = program.replay_and_wait(engine, &mut optimizer)?;
		if step == 0 {
			initial_loss = training_loss;
		}
	}
	let diagnostics = program.diagnostics();
	assert_eq!(diagnostics.command_recording_count(), 1);
	assert_eq!(diagnostics.command_cache_hit_count(), 299);
	assert_eq!(diagnostics.submission_count(), 300);
	assert_eq!(diagnostics.input_upload_count(), 598);
	let logits = model.forward(&input)?;
	let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&logits, &target)?;
	let generated = nlp::generate_bytes_greedy(
		engine,
		model,
		nlp::GENERATION_PROMPT.as_bytes(),
		nlp::GENERATION_LENGTH,
	)?;
	Ok(ByteTrainingRun {
		optimizer,
		input,
		target,
		initial_loss,
		training_loss,
		final_loss,
		accuracy,
		generated,
	})
}

fn verify_byte_checkpoint<M: Module>(
	engine: &oa::Engine,
	model: &M,
	run: &ByteTrainingRun,
	stem: &str,
	learning_rate: f32,
	construct: impl Fn(&oa::Engine) -> oa::Result<M>,
) -> oa::Result<()> {
	let checkpoint = std::env::temp_dir().join(format!("oars-{stem}-{}.oam", std::process::id()));
	oa::ml::save_checkpoint(&checkpoint, model, &run.optimizer)?;
	let reloaded = construct(engine)?;
	let mut optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, learning_rate)?;
	oa::ml::load_checkpoint(engine, &checkpoint, &reloaded, &mut optimizer)?;
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
		run.accuracy
	);
	assert_eq!(
		nlp::generate_bytes_greedy(
			engine,
			&reloaded,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?,
		run.generated
	);
	assert_eq!(optimizer.step_count(), run.optimizer.step_count());
	std::fs::remove_file(checkpoint).expect("remove completed byte checkpoint");
	Ok(())
}

struct CharTrainingRun {
	optimizer: oa::ml::AdamW,
	input: oa::Matrix,
	target: oa::Matrix,
	initial_loss: f32,
	training_loss: f32,
	final_loss: f32,
	accuracy: f32,
	generated: String,
}

fn train_char_model(
	engine: &oa::Engine,
	model: &impl Module,
	learning_rate: f32,
) -> oa::Result<CharTrainingRun> {
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, learning_rate)?;
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (initial_input, initial_target) = sampler.next_values()?;
	let input = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
		&initial_input,
	)?;
	let target = oa::Matrix::from_slice(
		engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&initial_target,
	)?;
	let mut program = oa::ml::TrainingProgram::capture(engine, &mut optimizer, || {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let mut initial_loss = 0.0_f32;
	let mut training_loss = 0.0_f32;
	for step in 0..nlp::TRAINING_STEPS {
		if step != 0 {
			let (next_input, next_target) = sampler.next_values()?;
			program.upload_input(&input, &next_input)?;
			program.upload_input(&target, &next_target)?;
		}
		training_loss = program.replay_and_wait(engine, &mut optimizer)?;
		if step == 0 {
			initial_loss = training_loss;
		}
	}
	let diagnostics = program.diagnostics();
	assert_eq!(diagnostics.command_recording_count(), 1);
	assert_eq!(diagnostics.command_cache_hit_count(), 299);
	assert_eq!(diagnostics.submission_count(), 300);
	assert_eq!(diagnostics.input_upload_count(), 598);
	let logits = model.forward(&input)?;
	let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&logits, &target)?;
	let generated = nlp::generate_greedy(
		engine,
		model,
		nlp::GENERATION_PROMPT,
		nlp::GENERATION_LENGTH,
	)?;
	Ok(CharTrainingRun {
		optimizer,
		input,
		target,
		initial_loss,
		training_loss,
		final_loss,
		accuracy,
		generated,
	})
}

fn verify_char_checkpoint<M: Module>(
	engine: &oa::Engine,
	model: &M,
	run: &CharTrainingRun,
	stem: &str,
	learning_rate: f32,
	construct: impl Fn(&oa::Engine) -> oa::Result<M>,
) -> oa::Result<()> {
	let checkpoint = std::env::temp_dir().join(format!("oars-{stem}-{}.oam", std::process::id()));
	oa::ml::save_checkpoint(&checkpoint, model, &run.optimizer)?;
	let reloaded = construct(engine)?;
	let mut optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, learning_rate)?;
	oa::ml::load_checkpoint(engine, &checkpoint, &reloaded, &mut optimizer)?;
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
		run.accuracy
	);
	assert_eq!(
		nlp::generate_greedy(
			engine,
			&reloaded,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?,
		run.generated
	);
	assert_eq!(optimizer.step_count(), run.optimizer.step_count());
	std::fs::remove_file(checkpoint).expect("remove completed character checkpoint");
	Ok(())
}

fn verify_bpe_checkpoint<M: Module>(
	engine: &oa::Engine,
	model: &M,
	tokenizer: &oa::ml::BpeTokenizer,
	run: &BpeTrainingRun,
	stem: &str,
	learning_rate: f32,
	construct: impl Fn(&oa::Engine) -> oa::Result<M>,
) -> oa::Result<()> {
	let base = std::env::temp_dir().join(format!("oars-{stem}-{}", std::process::id()));
	let checkpoint = base.with_extension("oam");
	let vocabulary = base.with_extension("bpe");
	oa::ml::save_checkpoint(&checkpoint, model, &run.optimizer)?;
	tokenizer.save(&vocabulary)?;
	let mut reloaded_tokenizer = oa::ml::BpeTokenizer::new(256);
	reloaded_tokenizer.load(&vocabulary)?;
	let reloaded = construct(engine)?;
	let mut optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, learning_rate)?;
	oa::ml::load_checkpoint(engine, &checkpoint, &reloaded, &mut optimizer)?;
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
		run.accuracy
	);
	assert_eq!(
		nlp::generate_bpe_greedy(
			engine,
			&reloaded,
			&reloaded_tokenizer,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?,
		run.generated
	);
	assert_eq!(optimizer.step_count(), run.optimizer.step_count());
	std::fs::remove_file(checkpoint).expect("remove completed BPE checkpoint");
	std::fs::remove_file(vocabulary).expect("remove completed BPE vocabulary");
	Ok(())
}

#[test]
fn canonical_character_contract_matches_oa_cpp() -> oa::Result<()> {
	assert_eq!(nlp::CORPUS.len(), 576);
	assert_eq!(nlp::encode(nlp::CORPUS).len(), 576);
	assert_eq!(nlp::decode(&nlp::encode(nlp::CORPUS)), nlp::CORPUS);
	assert_eq!(nlp::encode("aAzZ !"), [0, 0, 25, 25, 26, 26]);
	assert_eq!(nlp::CONTEXT_LENGTH, 16);
	assert_eq!(nlp::MODEL_WIDTH, 32);
	assert_eq!(nlp::HIDDEN_WIDTH, 64);
	assert_eq!(nlp::MOE_EXPERT_HIDDEN_WIDTH, 16);
	assert_eq!(nlp::MOE_NUM_EXPERTS, 4);
	assert_eq!(nlp::MOE_EXPERTS_PER_TOKEN, 2);
	assert_eq!(nlp::TRAINING_STEPS, 300);
	assert_eq!(nlp::BATCH_SIZE, 64);
	assert_eq!(nlp::RNG_SEED, 20_260_714);
	assert_eq!(nlp::GENERATION_PROMPT, "to be");
	assert_eq!(nlp::GENERATION_LENGTH, 80);
	assert_eq!(nlp::CHAR_VOCAB_SIZE, 27);
	assert_eq!(
		nlp::CHAR_RNN_REFERENCE_GENERATION.chars().count(),
		nlp::GENERATION_PROMPT.len() + nlp::GENERATION_LENGTH
	);
	assert_eq!(
		nlp::CHAR_TRANSFORMER_REFERENCE_GENERATION.chars().count(),
		nlp::GENERATION_PROMPT.len() + nlp::GENERATION_LENGTH
	);
	assert_eq!(
		nlp::CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION
			.chars()
			.count(),
		nlp::GENERATION_PROMPT.len() + nlp::GENERATION_LENGTH
	);
	assert_eq!(
		nlp::CHAR_MAMBA3_REFERENCE_GENERATION.chars().count(),
		nlp::GENERATION_PROMPT.len() + nlp::GENERATION_LENGTH
	);
	assert_eq!(
		nlp::CharSampler::new(0)
			.err()
			.expect("zero batch was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}

test_vk!(canonical_char_rnn_has_the_cpp_parameter_contract, engine, {
	let model = nlp::CharRnn::new(&engine)?;
	assert_eq!(model.num_parameters()?, 8_891);
	assert_eq!(model.all_parameters()?.len(), 7);
	assert_eq!(
		model
			.all_named_parameters()?
			.iter()
			.map(|entry| entry.path())
			.collect::<Vec<_>>(),
		[
			"embed.weight",
			"rnn.layer0.weight_ih",
			"rnn.layer0.weight_hh",
			"rnn.layer0.bias_ih",
			"rnn.layer0.bias_hh",
			"head.weight",
			"head.bias",
		]
	);
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.shape(), [64, 16]);
	assert_eq!(target.shape(), [64, 16]);
	let input_values = input.read::<u32>()?;
	let target_values = target.read::<u32>()?;
	assert_eq!(&nlp::decode(&input_values[..16]), "to be or not to ");
	assert_eq!(&nlp::decode(&target_values[..16]), "o be or not to b");
	assert_eq!(model.forward(&input)?.shape(), [1024, 27]);
	Ok(())
});

test_vk!(canonical_char_gru_has_the_cpp_parameter_contract, engine, {
	let model = nlp::CharGru::new(&engine)?;
	assert_eq!(model.num_parameters()?, 21_435);
	assert_eq!(model.all_parameters()?.len(), 7);
	let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.dtype(), oa::DType::U32);
	assert_eq!(target.dtype(), oa::DType::U32);
	assert_eq!(model.forward(&input)?.shape(), [1024, 27]);
	Ok(())
});

test_vk!(canonical_byte_rnn_has_the_cpp_parameter_contract, engine, {
	let model = nlp::ByteRnn::new(&engine)?;
	assert_eq!(model.num_parameters()?, 31_104);
	assert_eq!(model.all_parameters()?.len(), 7);
	assert_eq!(
		model
			.all_named_parameters()?
			.iter()
			.map(|entry| entry.path())
			.collect::<Vec<_>>(),
		[
			"embed.weight",
			"rnn.layer0.weight_ih",
			"rnn.layer0.weight_hh",
			"rnn.layer0.bias_ih",
			"rnn.layer0.bias_hh",
			"head.weight",
			"head.bias",
		]
	);
	let mut sampler = nlp::ByteSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.dtype(), oa::DType::U8);
	assert_eq!(input.shape(), [64, 16]);
	assert_eq!(target.dtype(), oa::DType::U32);
	assert_eq!(&input.read::<u8>()?[..16], b"to be or not to ");
	assert_eq!(
		&target.read::<u32>()?[..16],
		&[
			111, 32, 98, 101, 32, 111, 114, 32, 110, 111, 116, 32, 116, 111, 32, 98
		]
	);
	assert_eq!(model.forward(&input)?.shape(), [1024, 256]);
	Ok(())
});

test_vk!(canonical_byte_gru_has_the_cpp_parameter_contract, engine, {
	let model = nlp::ByteGru::new(&engine)?;
	assert_eq!(model.num_parameters()?, 43_648);
	assert_eq!(model.all_parameters()?.len(), 7);
	assert_eq!(
		model
			.all_named_parameters()?
			.iter()
			.map(|entry| entry.path())
			.collect::<Vec<_>>(),
		[
			"embed.weight",
			"gru.layer0.weight_ih",
			"gru.layer0.weight_hh",
			"gru.layer0.bias_ih",
			"gru.layer0.bias_hh",
			"head.weight",
			"head.bias",
		]
	);
	let mut sampler = nlp::ByteSampler::new(nlp::BATCH_SIZE)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.dtype(), oa::DType::U8);
	assert_eq!(target.dtype(), oa::DType::U32);
	assert_eq!(model.forward(&input)?.shape(), [1024, 256]);
	Ok(())
});

test_vk!(canonical_bpe_rnn_has_the_cpp_parameter_contract, engine, {
	let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	assert_eq!(tokenizer.vocab_size(), nlp::BPE_VOCAB_SIZE);
	assert_eq!(tokenizer.num_merges(), nlp::BPE_MERGES);
	let corpus_tokens = tokenizer.encode_text(nlp::CORPUS);
	assert_eq!(tokenizer.decode_text(&corpus_tokens)?, nlp::CORPUS);
	assert!(corpus_tokens.len() < nlp::CORPUS.len());

	let model = nlp::BpeRnn::new(&engine)?;
	assert_eq!(model.num_parameters()?, 37_312);
	assert_eq!(model.all_parameters()?.len(), 7);
	assert_eq!(
		model
			.all_named_parameters()?
			.iter()
			.map(|entry| entry.path())
			.collect::<Vec<_>>(),
		[
			"embed.weight",
			"rnn.layer0.weight_ih",
			"rnn.layer0.weight_hh",
			"rnn.layer0.bias_ih",
			"rnn.layer0.bias_hh",
			"head.weight",
			"head.bias",
		]
	);
	let mut sampler = nlp::BpeSampler::new(nlp::BATCH_SIZE, &tokenizer)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.dtype(), oa::DType::U32);
	assert_eq!(input.shape(), [64, 16]);
	assert_eq!(target.dtype(), oa::DType::U32);
	assert_eq!(target.shape(), [64, 16]);
	assert!(sampler.last_batch_bytes_per_token() > 1.0);
	assert_eq!(model.forward(&input)?.shape(), [1024, 320]);
	Ok(())
});

test_vk!(canonical_bpe_gru_has_the_cpp_parameter_contract, engine, {
	let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	let model = nlp::BpeGru::new(&engine)?;
	assert_eq!(model.num_parameters()?, 49_856);
	assert_eq!(model.all_parameters()?.len(), 7);
	assert_eq!(
		model
			.all_named_parameters()?
			.iter()
			.map(|entry| entry.path())
			.collect::<Vec<_>>(),
		[
			"embed.weight",
			"gru.layer0.weight_ih",
			"gru.layer0.weight_hh",
			"gru.layer0.bias_ih",
			"gru.layer0.bias_hh",
			"head.weight",
			"head.bias",
		]
	);
	let mut sampler = nlp::BpeSampler::new(nlp::BATCH_SIZE, &tokenizer)?;
	let (input, target) = sampler.next(&engine)?;
	assert_eq!(input.dtype(), oa::DType::U32);
	assert_eq!(target.dtype(), oa::DType::U32);
	assert_eq!(model.forward(&input)?.shape(), [1024, 320]);
	Ok(())
});

test_vk!(
	canonical_byte_and_bpe_transformers_reuse_the_nn_owner,
	engine,
	{
		let byte = nlp::ByteTransformer::new(&engine)?;
		assert_eq!(byte.num_parameters()?, 25_760);
		let byte_tokens = oa::Matrix::from_slice(&engine, [1, 16], b"to be or not to ")?;
		assert_eq!(byte.forward(&byte_tokens)?.shape(), [16, 256]);

		let bpe = nlp::BpeTransformer::new(&engine)?;
		assert_eq!(bpe.num_parameters()?, 29_920);
		let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
		tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
		let mut sampler = nlp::BpeSampler::new(1, &tokenizer)?;
		let (bpe_tokens, _) = sampler.next(&engine)?;
		assert_eq!(bpe.forward(&bpe_tokens)?.shape(), [16, 320]);
		Ok(())
	}
);

test_vk!(
	canonical_byte_and_bpe_moe_models_reuse_the_nn_owner,
	engine,
	{
		let byte = nlp::ByteMoeTransformer::new(&engine)?;
		assert_eq!(byte.num_parameters()?, 28_068);
		let byte_tokens = oa::Matrix::from_slice(&engine, [1, 16], b"to be or not to ")?;
		assert_eq!(byte.forward(&byte_tokens)?.shape(), [16, 256]);

		let bpe = nlp::BpeMoeTransformer::new(&engine)?;
		assert_eq!(bpe.num_parameters()?, 32_228);
		let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
		tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
		let mut sampler = nlp::BpeSampler::new(1, &tokenizer)?;
		let (bpe_tokens, _) = sampler.next(&engine)?;
		assert_eq!(bpe.forward(&bpe_tokens)?.shape(), [16, 320]);
		Ok(())
	}
);

test_vk!(
	canonical_byte_and_bpe_mamba3_models_reuse_the_nn_owner,
	engine,
	{
		let byte = nlp::ByteMamba3::new(&engine)?;
		assert_eq!(byte.num_parameters()?, 25_800);
		let byte_tokens = oa::Matrix::from_slice(&engine, [1, 16], b"to be or not to ")?;
		assert_eq!(byte.forward(&byte_tokens)?.shape(), [16, 256]);

		let bpe = nlp::BpeMamba3::new(&engine)?;
		assert_eq!(bpe.num_parameters()?, 29_960);
		let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
		tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
		let mut sampler = nlp::BpeSampler::new(1, &tokenizer)?;
		let (bpe_tokens, _) = sampler.next(&engine)?;
		assert_eq!(bpe.forward(&bpe_tokens)?.shape(), [16, 320]);
		Ok(())
	}
);

test_vk!(
	byte_empyrealm_core_preserves_the_donor_fidelity_tree,
	engine,
	{
		let model = nlp::ByteEmpyrealm::new(&engine)?;
		assert_eq!(model.num_parameters()?, 25_800);
		let paths = model
			.all_named_parameters()?
			.into_iter()
			.map(|parameter| parameter.path().to_owned())
			.collect::<Vec<_>>();
		assert_eq!(paths.first().map(String::as_str), Some("core.embed.weight"));
		assert!(paths.iter().any(|path| path == "core.mixer.in_proj"));
		assert!(paths.iter().any(|path| path == "core.mixer.out_proj"));
		assert_eq!(paths.last().map(String::as_str), Some("head.bias"));

		let tokens = oa::Matrix::from_slice(&engine, [1, 16], b"to be or not to ")?;
		let reference = nlp::ByteMamba3::new(&engine)?;
		assert_eq!(model.forward(&tokens)?.shape(), [16, 256]);
		assert_eq!(
			model.forward(&tokens)?.read_f32()?,
			reference.forward(&tokens)?.read_f32()?
		);
		Ok(())
	}
);

test_vk!(
	canonical_byte_rnn_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteRnn::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
		let mut sampler = nlp::ByteSampler::new(nlp::BATCH_SIZE)?;
		let (initial_input, initial_target) = sampler.next_values()?;
		let input = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
			&initial_input,
		)?;
		let target = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
			&initial_target,
		)?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let mut initial_loss = 0.0_f32;
		let mut training_loss = 0.0_f32;
		for step in 0..nlp::TRAINING_STEPS {
			if step != 0 {
				let (next_input, next_target) = sampler.next_values()?;
				program.upload_input(&input, &next_input)?;
				program.upload_input(&target, &next_target)?;
			}
			training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
			if step == 0 {
				initial_loss = training_loss;
			}
		}
		let diagnostics = program.diagnostics();
		assert_eq!(diagnostics.command_recording_count(), 1);
		assert_eq!(diagnostics.command_cache_hit_count(), 299);
		assert_eq!(diagnostics.submission_count(), 300);
		assert_eq!(diagnostics.input_upload_count(), 598);
		let logits = model.forward(&input)?;
		let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
		let accuracy = nlp::accuracy(&logits, &target)?;
		let generated = nlp::generate_bytes_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?;
		println!(
			"Byte RNN: initial={initial_loss:.6} training={training_loss:.6} final={final_loss:.6} accuracy={accuracy:.6} generated={:?}",
			String::from_utf8_lossy(&generated)
		);
		assert!(initial_loss > 5.0);
		assert!((final_loss - nlp::BYTE_RNN_FINAL_LOSS).abs() < 0.001);
		assert!((accuracy - nlp::BYTE_RNN_ACCURACY).abs() < 0.001);
		assert_eq!(generated, nlp::BYTE_RNN_REFERENCE_GENERATION);
		assert_eq!(optimizer.step_count(), 300);
		let checkpoint = std::env::temp_dir().join(format!("oars-byte-rnn-{}.oam", std::process::id()));
		oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
		let reloaded = nlp::ByteRnn::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		assert_eq!(
			nlp::accuracy(&reloaded.forward(&input)?, &target)?,
			accuracy
		);
		assert_eq!(
			nlp::generate_bytes_greedy(
				&engine,
				&reloaded,
				nlp::GENERATION_PROMPT.as_bytes(),
				nlp::GENERATION_LENGTH,
			)?,
			generated
		);
		assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
		std::fs::remove_file(checkpoint).expect("remove completed Byte RNN checkpoint");
		Ok(())
	}
);

test_vk!(
	canonical_byte_gru_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteGru::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
		let mut sampler = nlp::ByteSampler::new(nlp::BATCH_SIZE)?;
		let (initial_input, initial_target) = sampler.next_values()?;
		let input = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
			&initial_input,
		)?;
		let target = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
			&initial_target,
		)?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let mut initial_loss = 0.0_f32;
		let mut training_loss = 0.0_f32;
		for step in 0..nlp::TRAINING_STEPS {
			if step != 0 {
				let (next_input, next_target) = sampler.next_values()?;
				program.upload_input(&input, &next_input)?;
				program.upload_input(&target, &next_target)?;
			}
			training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
			if step == 0 {
				initial_loss = training_loss;
			}
		}
		let diagnostics = program.diagnostics();
		assert_eq!(diagnostics.command_recording_count(), 1);
		assert_eq!(diagnostics.command_cache_hit_count(), 299);
		assert_eq!(diagnostics.submission_count(), 300);
		assert_eq!(diagnostics.input_upload_count(), 598);
		let logits = model.forward(&input)?;
		let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
		let accuracy = nlp::accuracy(&logits, &target)?;
		let generated = nlp::generate_bytes_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?;
		println!(
			"Byte GRU: initial={initial_loss:.6} training={training_loss:.6} final={final_loss:.6} accuracy={accuracy:.6} generated={:?}",
			String::from_utf8_lossy(&generated)
		);
		assert!(initial_loss > 5.0);
		assert!((final_loss - nlp::BYTE_GRU_FINAL_LOSS).abs() < 0.001);
		assert!((accuracy - nlp::BYTE_GRU_ACCURACY).abs() < 0.001);
		assert_eq!(generated, nlp::BYTE_GRU_REFERENCE_GENERATION);
		assert_eq!(optimizer.step_count(), 300);
		let checkpoint = std::env::temp_dir().join(format!("oars-byte-gru-{}.oam", std::process::id()));
		oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
		let reloaded = nlp::ByteGru::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		assert_eq!(
			nlp::accuracy(&reloaded.forward(&input)?, &target)?,
			accuracy
		);
		assert_eq!(
			nlp::generate_bytes_greedy(
				&engine,
				&reloaded,
				nlp::GENERATION_PROMPT.as_bytes(),
				nlp::GENERATION_LENGTH,
			)?,
			generated
		);
		assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
		std::fs::remove_file(checkpoint).expect("remove completed Byte GRU checkpoint");
		Ok(())
	}
);

test_vk!(canonical_bpe_rnn_completes_the_cpp_300_step_gate, engine, {
	let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	let model = nlp::BpeRnn::new(&engine)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
	let mut sampler = nlp::BpeSampler::new(nlp::BATCH_SIZE, &tokenizer)?;
	let (initial_input, initial_target) = sampler.next_values()?;
	let input = oa::Matrix::from_slice(
		&engine,
		[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
		&initial_input,
	)?;
	let target = oa::Matrix::from_slice(
		&engine,
		[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
		&initial_target,
	)?;
	let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		Ok(loss)
	})?;
	let mut initial_loss = 0.0_f32;
	let mut training_loss = 0.0_f32;
	for step in 0..nlp::TRAINING_STEPS {
		if step != 0 {
			let (next_input, next_target) = sampler.next_values()?;
			program.upload_input(&input, &next_input)?;
			program.upload_input(&target, &next_target)?;
		}
		training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
		if step == 0 {
			initial_loss = training_loss;
		}
	}
	let diagnostics = program.diagnostics();
	assert_eq!(diagnostics.command_recording_count(), 1);
	assert_eq!(diagnostics.command_cache_hit_count(), 299);
	assert_eq!(diagnostics.submission_count(), 300);
	assert_eq!(diagnostics.input_upload_count(), 598);
	let logits = model.forward(&input)?;
	let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
	let accuracy = nlp::accuracy(&logits, &target)?;
	let generated = nlp::generate_bpe_greedy(
		&engine,
		&model,
		&tokenizer,
		nlp::GENERATION_PROMPT.as_bytes(),
		nlp::GENERATION_LENGTH,
	)?;
	println!(
		"BPE RNN: initial={initial_loss:.6} training={training_loss:.6} final={final_loss:.6} accuracy={accuracy:.6} bytes/token={:.6} generated={:?}",
		sampler.last_batch_bytes_per_token(),
		String::from_utf8_lossy(&generated)
	);
	assert!(initial_loss > 5.0);
	assert!((final_loss - nlp::BPE_RNN_FINAL_LOSS).abs() < 0.001);
	assert!((accuracy - nlp::BPE_RNN_ACCURACY).abs() < 0.001);
	assert_eq!(generated, nlp::BPE_RNN_REFERENCE_GENERATION);
	assert_eq!(optimizer.step_count(), 300);

	let base = std::env::temp_dir().join(format!("oars-bpe-rnn-{}", std::process::id()));
	let checkpoint = base.with_extension("oam");
	let vocabulary = base.with_extension("bpe");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	tokenizer.save(&vocabulary)?;
	let mut reloaded_tokenizer = oa::ml::BpeTokenizer::new(256);
	reloaded_tokenizer.load(&vocabulary)?;
	assert_eq!(reloaded_tokenizer.merges(), tokenizer.merges());
	let reloaded = nlp::BpeRnn::new(&engine)?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
	oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&input)?, &target)?,
		accuracy
	);
	assert_eq!(
		nlp::generate_bpe_greedy(
			&engine,
			&reloaded,
			&reloaded_tokenizer,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?,
		generated
	);
	assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
	std::fs::remove_file(checkpoint).expect("remove completed BPE RNN checkpoint");
	std::fs::remove_file(vocabulary).expect("remove completed BPE vocabulary");
	Ok(())
});

test_vk!(canonical_bpe_gru_completes_the_cpp_300_step_gate, engine, {
	let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	let model = nlp::BpeGru::new(&engine)?;
	let run = train_bpe_model(&engine, &model, &tokenizer, 0.01)?;
	println!(
		"BPE GRU: initial={:.6} training={:.6} final={:.6} accuracy={:.6} bytes/token={:.6} generated={:?}",
		run.initial_loss,
		run.training_loss,
		run.final_loss,
		run.accuracy,
		run.bytes_per_token,
		String::from_utf8_lossy(&run.generated)
	);
	assert!(run.initial_loss > 5.0);
	assert!((run.final_loss - nlp::BPE_GRU_FINAL_LOSS).abs() < 0.001);
	assert!((run.accuracy - nlp::BPE_GRU_ACCURACY).abs() < 0.001);
	assert_eq!(run.generated, nlp::BPE_GRU_REFERENCE_GENERATION);
	assert_eq!(run.optimizer.step_count(), 300);

	let base = std::env::temp_dir().join(format!("oars-bpe-gru-{}", std::process::id()));
	let checkpoint = base.with_extension("oam");
	let vocabulary = base.with_extension("bpe");
	oa::ml::save_checkpoint(&checkpoint, &model, &run.optimizer)?;
	tokenizer.save(&vocabulary)?;
	let mut reloaded_tokenizer = oa::ml::BpeTokenizer::new(256);
	reloaded_tokenizer.load(&vocabulary)?;
	let reloaded = nlp::BpeGru::new(&engine)?;
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
	oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(
		nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
		run.accuracy
	);
	assert_eq!(
		nlp::generate_bpe_greedy(
			&engine,
			&reloaded,
			&reloaded_tokenizer,
			nlp::GENERATION_PROMPT.as_bytes(),
			nlp::GENERATION_LENGTH,
		)?,
		run.generated
	);
	assert_eq!(reloaded_optimizer.step_count(), run.optimizer.step_count());
	std::fs::remove_file(checkpoint).expect("remove completed BPE GRU checkpoint");
	std::fs::remove_file(vocabulary).expect("remove completed BPE vocabulary");
	Ok(())
});

test_vk!(
	canonical_bpe_transformer_completes_the_cpp_300_step_gate,
	engine,
	{
		let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
		tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
		let model = nlp::BpeTransformer::new(&engine)?;
		let run = train_bpe_model(&engine, &model, &tokenizer, 0.01)?;
		println!(
			"BPE Transformer: initial={:.6} training={:.6} final={:.6} accuracy={:.6} bytes/token={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			run.bytes_per_token,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BPE_TRANSFORMER_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BPE_TRANSFORMER_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BPE_TRANSFORMER_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);

		let base = std::env::temp_dir().join(format!("oars-bpe-transformer-{}", std::process::id()));
		let checkpoint = base.with_extension("oam");
		let vocabulary = base.with_extension("bpe");
		oa::ml::save_checkpoint(&checkpoint, &model, &run.optimizer)?;
		tokenizer.save(&vocabulary)?;
		let mut reloaded_tokenizer = oa::ml::BpeTokenizer::new(256);
		reloaded_tokenizer.load(&vocabulary)?;
		let reloaded = nlp::BpeTransformer::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		assert_eq!(
			nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
			run.accuracy
		);
		assert_eq!(
			nlp::generate_bpe_greedy(
				&engine,
				&reloaded,
				&reloaded_tokenizer,
				nlp::GENERATION_PROMPT.as_bytes(),
				nlp::GENERATION_LENGTH,
			)?,
			run.generated
		);
		assert_eq!(reloaded_optimizer.step_count(), run.optimizer.step_count());
		std::fs::remove_file(checkpoint).expect("remove completed BPE Transformer checkpoint");
		std::fs::remove_file(vocabulary).expect("remove completed BPE vocabulary");
		Ok(())
	}
);

test_vk!(
	canonical_byte_transformer_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteTransformer::new(&engine)?;
		let run = train_byte_model(&engine, &model, 0.01)?;
		println!(
			"Byte Transformer: initial={:.6} training={:.6} final={:.6} accuracy={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BYTE_TRANSFORMER_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BYTE_TRANSFORMER_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BYTE_TRANSFORMER_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);

		let checkpoint =
			std::env::temp_dir().join(format!("oars-byte-transformer-{}.oam", std::process::id()));
		oa::ml::save_checkpoint(&checkpoint, &model, &run.optimizer)?;
		let reloaded = nlp::ByteTransformer::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		assert_eq!(
			nlp::accuracy(&reloaded.forward(&run.input)?, &run.target)?,
			run.accuracy
		);
		assert_eq!(
			nlp::generate_bytes_greedy(
				&engine,
				&reloaded,
				nlp::GENERATION_PROMPT.as_bytes(),
				nlp::GENERATION_LENGTH,
			)?,
			run.generated
		);
		assert_eq!(reloaded_optimizer.step_count(), run.optimizer.step_count());
		std::fs::remove_file(checkpoint).expect("remove completed Byte Transformer checkpoint");
		Ok(())
	}
);

test_vk!(
	canonical_byte_moe_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteMoeTransformer::new(&engine)?;
		let run = train_byte_model(&engine, &model, 0.01)?;
		println!(
			"Byte MoE: initial={:.6} training={:.6} final={:.6} accuracy={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BYTE_MOE_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BYTE_MOE_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BYTE_MOE_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);
		verify_byte_checkpoint(
			&engine,
			&model,
			&run,
			"byte-moe",
			0.01,
			nlp::ByteMoeTransformer::new,
		)?;
		Ok(())
	}
);

test_vk!(canonical_bpe_moe_completes_the_cpp_300_step_gate, engine, {
	let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
	tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
	let model = nlp::BpeMoeTransformer::new(&engine)?;
	let run = train_bpe_model(&engine, &model, &tokenizer, 0.01)?;
	println!(
		"BPE MoE: initial={:.6} training={:.6} final={:.6} accuracy={:.6} bytes/token={:.6} generated={:?}",
		run.initial_loss,
		run.training_loss,
		run.final_loss,
		run.accuracy,
		run.bytes_per_token,
		String::from_utf8_lossy(&run.generated)
	);
	assert!(run.initial_loss > 5.0);
	assert!((run.final_loss - nlp::BPE_MOE_FINAL_LOSS).abs() < 0.001);
	assert!((run.accuracy - nlp::BPE_MOE_ACCURACY).abs() < 0.001);
	assert_eq!(run.generated, nlp::BPE_MOE_REFERENCE_GENERATION);
	assert_eq!(run.optimizer.step_count(), 300);
	verify_bpe_checkpoint(
		&engine,
		&model,
		&tokenizer,
		&run,
		"bpe-moe",
		0.01,
		nlp::BpeMoeTransformer::new,
	)?;
	Ok(())
});

test_vk!(
	canonical_byte_mamba3_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteMamba3::new(&engine)?;
		let run = train_byte_model(&engine, &model, 0.003)?;
		println!(
			"Byte Mamba3: initial={:.6} training={:.6} final={:.6} accuracy={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BYTE_MAMBA3_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BYTE_MAMBA3_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BYTE_MAMBA3_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);
		verify_byte_checkpoint(
			&engine,
			&model,
			&run,
			"byte-mamba3",
			0.003,
			nlp::ByteMamba3::new,
		)?;
		Ok(())
	}
);

test_vk!(
	canonical_bpe_mamba3_completes_the_cpp_300_step_gate,
	engine,
	{
		let mut tokenizer = oa::ml::BpeTokenizer::new(nlp::BPE_VOCAB_SIZE);
		tokenizer.train_text(nlp::CORPUS, nlp::BPE_MERGES);
		let model = nlp::BpeMamba3::new(&engine)?;
		let run = train_bpe_model(&engine, &model, &tokenizer, 0.003)?;
		println!(
			"BPE Mamba3: initial={:.6} training={:.6} final={:.6} accuracy={:.6} bytes/token={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			run.bytes_per_token,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BPE_MAMBA3_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BPE_MAMBA3_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BPE_MAMBA3_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);
		verify_bpe_checkpoint(
			&engine,
			&model,
			&tokenizer,
			&run,
			"bpe-mamba3",
			0.003,
			nlp::BpeMamba3::new,
		)?;
		Ok(())
	}
);

test_vk!(
	canonical_byte_empyrealm_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::ByteEmpyrealm::new(&engine)?;
		let run = train_byte_model(&engine, &model, 0.003)?;
		println!(
			"Byte Empyrealm: initial={:.6} training={:.6} final={:.6} accuracy={:.6} generated={:?}",
			run.initial_loss,
			run.training_loss,
			run.final_loss,
			run.accuracy,
			String::from_utf8_lossy(&run.generated)
		);
		assert!(run.initial_loss > 5.0);
		assert!((run.final_loss - nlp::BYTE_EMPYREALM_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::BYTE_EMPYREALM_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::BYTE_EMPYREALM_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);
		verify_byte_checkpoint(
			&engine,
			&model,
			&run,
			"byte-empyrealm",
			0.003,
			nlp::ByteEmpyrealm::new,
		)?;
		Ok(())
	}
);

test_vk!(
	canonical_char_gru_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::CharGru::new(&engine)?;
		let run = train_char_model(&engine, &model, 0.01)?;
		println!(
			"Char GRU: initial={:.6} training={:.6} final={:.6} accuracy={:.6} generated={:?}",
			run.initial_loss, run.training_loss, run.final_loss, run.accuracy, run.generated
		);
		assert!(run.initial_loss > 3.0);
		assert!((run.final_loss - nlp::CHAR_GRU_FINAL_LOSS).abs() < 0.001);
		assert!((run.accuracy - nlp::CHAR_GRU_ACCURACY).abs() < 0.001);
		assert_eq!(run.generated, nlp::CHAR_GRU_REFERENCE_GENERATION);
		assert_eq!(run.optimizer.step_count(), 300);
		verify_char_checkpoint(&engine, &model, &run, "char-gru", 0.01, nlp::CharGru::new)?;
		Ok(())
	}
);

test_vk!(
	canonical_char_mamba3_has_the_cpp_parameter_contract,
	engine,
	{
		let model = nlp::CharMamba3::new(&engine)?;
		assert_eq!(model.num_parameters()?, 10_915);
		assert_eq!(model.all_parameters()?.len(), 10);
		assert_eq!(
			model
				.all_named_parameters()?
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			[
				"embed.weight",
				"mamba3.in_proj",
				"mamba3.dt_bias",
				"mamba3.B_bias",
				"mamba3.C_bias",
				"mamba3.D",
				"mamba3.out_proj",
				"mamba3.norm_weight",
				"head.weight",
				"head.bias",
			]
		);
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let (input, target) = sampler.next(&engine)?;
		assert_eq!(input.shape(), [64, 16]);
		assert_eq!(target.shape(), [64, 16]);
		assert_eq!(model.forward(&input)?.shape(), [1024, 27]);
		Ok(())
	}
);

test_vk!(
	canonical_char_mamba3_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::CharMamba3::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.003)?;
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let (initial_input, initial_target) = sampler.next_values()?;
		let input = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
			&initial_input,
		)?;
		let target = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
			&initial_target,
		)?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let mut initial_loss = 0.0_f32;
		let mut training_loss = 0.0_f32;
		for step in 0..nlp::TRAINING_STEPS {
			if step != 0 {
				let (next_input, next_target) = sampler.next_values()?;
				program.upload_input(&input, &next_input)?;
				program.upload_input(&target, &next_target)?;
			}
			training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
			if step == 0 {
				initial_loss = training_loss;
			}
		}
		let logits = model.forward(&input)?;
		let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
		let accuracy = nlp::accuracy(&logits, &target)?;
		let generated = nlp::generate_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?;
		println!(
			"Char Mamba3: initial={initial_loss:.6} training={training_loss:.6} final={final_loss:.6} accuracy={accuracy:.6} generated={generated:?}"
		);
		assert!(initial_loss > 3.0);
		assert!((final_loss - nlp::CHAR_MAMBA3_FINAL_LOSS).abs() < 0.000_01);
		assert!((accuracy - nlp::CHAR_MAMBA3_ACCURACY).abs() < 0.000_001);
		assert_eq!(generated, nlp::CHAR_MAMBA3_REFERENCE_GENERATION);
		assert_eq!(optimizer.step_count(), 300);
		Ok(())
	}
);

test_vk!(
	canonical_char_rnn_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::CharRnn::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let mut initial_loss = 0.0_f32;
		let mut final_loss = 0.0_f32;
		let mut final_batch = None;

		for step in 0..nlp::TRAINING_STEPS {
			let (input, target) = sampler.next(&engine)?;
			optimizer.zero_grad();
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let target = target.reshape([nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH])?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			final_loss = loss.read_f32()?[0];
			if step == 0 {
				initial_loss = final_loss;
			}
			optimizer.step()?;
			final_batch = Some((input, target));
		}

		let (input, target) = final_batch.expect("the fixed 300-step loop is nonempty");
		let accuracy = nlp::accuracy(&model.forward(&input)?, &target)?;
		let generated = nlp::generate_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?;
		assert!(
			initial_loss > 3.0,
			"initial loss {initial_loss} is not near the ln(27) baseline"
		);
		assert!(
			final_loss < 0.25,
			"final loss {final_loss} missed the C++ small-corpus regime"
		);
		assert!(
			accuracy > 0.9,
			"final accuracy {}% missed the C++ regime",
			accuracy * 100.0
		);
		assert_eq!(generated, nlp::CHAR_RNN_REFERENCE_GENERATION);
		assert_eq!(optimizer.step_count(), 300);
		Ok(())
	}
);

test_vk!(
	canonical_char_transformer_has_the_cpp_parameter_contract,
	engine,
	{
		let model = nlp::CharTransformer::new(&engine)?;
		assert_eq!(model.num_parameters()?, 10_875);
		assert_eq!(model.all_parameters()?.len(), 22);
		assert_eq!(
			model
				.all_named_parameters()?
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			[
				"token_embedding.weight",
				"position_embedding.weight",
				"block_0.ln_attn.weight",
				"block_0.ln_attn.bias",
				"block_0.attention.q_proj.weight",
				"block_0.attention.q_proj.bias",
				"block_0.attention.k_proj.weight",
				"block_0.attention.k_proj.bias",
				"block_0.attention.v_proj.weight",
				"block_0.attention.v_proj.bias",
				"block_0.attention.out_proj.weight",
				"block_0.attention.out_proj.bias",
				"block_0.ln_ffn.weight",
				"block_0.ln_ffn.bias",
				"block_0.ffn1.weight",
				"block_0.ffn1.bias",
				"block_0.ffn2.weight",
				"block_0.ffn2.bias",
				"final_norm.weight",
				"final_norm.bias",
				"head.weight",
				"head.bias",
			]
		);
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let (input, _) = sampler.next(&engine)?;
		let (plan, output) = engine.capture(|| model.forward(&input))?;
		assert_eq!(output.shape(), [1024, 27]);
		let diagnostics = plan.diagnostics();
		assert!(diagnostics.semantic_operation_count() > 0);
		assert_eq!(diagnostics.compatibility_node_count(), 0);
		assert!(diagnostics.dnn_recognized_partition_count() >= 3);
		assert!(diagnostics.dnn_portable_partition_count() > 0);
		assert_eq!(diagnostics.dnn_applied_partition_count(), 1);
		assert!(diagnostics.dnn_inherited_partition_count() > 0);
		assert_eq!(diagnostics.dnn_unexpected_fallback_count(), 0);
		let executable: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("CanonicalCharTransformer"))
				.expect("Transformer executable report must be valid JSON");
		assert!(
			executable["nodes"]
				.as_array()
				.expect("executable nodes must be an array")
				.iter()
				.any(|node| node["kernel"] == "ml.qkv_projection_bias.f32")
		);
		let attention = plan
			.semantic_graph()
			.operations()
			.iter()
			.find(|operation| operation.name() == "oa::ml::matrix::scaled_dot_product_attention")
			.expect("Transformer capture omitted semantic attention");
		assert_eq!(attention.attributes().len(), 2);
		assert_eq!(attention.attributes()[0].name(), "scale");
		assert_eq!(attention.attributes()[1].name(), "causal");
		engine.submit(&plan)?.wait()?;
		Ok(())
	}
);

test_vk!(
	canonical_char_transformer_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::CharTransformer::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let (initial_input, initial_target) = sampler.next_values()?;
		let input = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
			&initial_input,
		)?;
		let target = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
			&initial_target,
		)?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let mut initial_loss = 0.0_f32;
		let mut training_loss = 0.0_f32;
		for step in 0..nlp::TRAINING_STEPS {
			if step != 0 {
				let (next_input, next_target) = sampler.next_values()?;
				program.upload_input(&input, &next_input)?;
				program.upload_input(&target, &next_target)?;
			}
			training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
			if step == 0 {
				initial_loss = training_loss;
			}
		}
		let diagnostics = program.diagnostics();
		assert_eq!(diagnostics.command_recording_count(), 1);
		assert_eq!(diagnostics.command_cache_hit_count(), 299);
		assert_eq!(diagnostics.submission_count(), 300);
		assert_eq!(diagnostics.input_upload_count(), 598);
		assert!(diagnostics.semantic_fused_node_count() >= 5);
		assert!(diagnostics.semantic_fused_operation_count() >= 20);
		assert_eq!(diagnostics.maximum_semantic_operations_per_node(), 4);
		let logits = model.forward(&input)?;
		let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
		let accuracy = nlp::accuracy(&logits, &target)?;
		let generated = nlp::generate_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?;
		assert!(
			initial_loss > 3.0,
			"initial loss {initial_loss} is below the random regime"
		);
		assert!(
			(final_loss - nlp::CHAR_TRANSFORMER_CPP_FINAL_LOSS).abs() < 0.02,
			"final loss {final_loss} missed C++ {}; last training loss {training_loss}",
			nlp::CHAR_TRANSFORMER_CPP_FINAL_LOSS
		);
		assert!(
			(accuracy - nlp::CHAR_TRANSFORMER_CPP_ACCURACY).abs() < 0.01,
			"accuracy {}% missed C++ {}%",
			accuracy * 100.0,
			nlp::CHAR_TRANSFORMER_CPP_ACCURACY * 100.0
		);
		assert_eq!(generated, nlp::CHAR_TRANSFORMER_REFERENCE_GENERATION);
		assert_eq!(optimizer.step_count(), 300);
		let checkpoint =
			std::env::temp_dir().join(format!("oars-char-transformer-{}.oam", std::process::id()));
		oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
		let reloaded = nlp::CharTransformer::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		let reloaded_accuracy = nlp::accuracy(&reloaded.forward(&input)?, &target)?;
		let reloaded_generation = nlp::generate_greedy(
			&engine,
			&reloaded,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?;
		assert_eq!(reloaded_accuracy, accuracy);
		assert_eq!(reloaded_generation, generated);
		assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
		std::fs::remove_file(&checkpoint).expect("remove completed Transformer checkpoint");
		Ok(())
	}
);

test_vk!(
	canonical_char_moe_transformer_matches_the_cpp_recipe,
	engine,
	{
		let model = nlp::CharMoeTransformer::new(&engine)?;
		assert_eq!(model.num_parameters()?, 13_183);
		let named = model.all_named_parameters()?;
		assert_eq!(named.len(), 23);
		assert_eq!(named[0].path(), "token_embedding.weight");
		assert_eq!(named[1].path(), "position_embedding.weight");
		assert_eq!(named[2].path(), "block_0.ln_attn.weight");
		assert_eq!(named[12].path(), "block_0.moe.expert_gate_up_weight");
		assert_eq!(named[15].path(), "block_0.moe.expert_down_bias");
		assert_eq!(named[16].path(), "block_0.moe.norm.weight");
		assert_eq!(named[17].path(), "block_0.moe.router.weight");
		assert_eq!(named[19].path(), "final_norm.weight");
		assert_eq!(named[21].path(), "head.weight");
		assert_eq!(model.moe().num_experts(), 4);
		assert_eq!(model.moe().experts_per_token(), 2);

		let mut sampler = nlp::CharSampler::new(2)?;
		let (input, target) = sampler.next(&engine)?;
		let target = target.reshape([2 * nlp::CONTEXT_LENGTH])?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&input)?;
		assert_eq!(
			logits.shape(),
			[2 * nlp::CONTEXT_LENGTH, nlp::CHAR_VOCAB_SIZE]
		);
		let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
		tape.backward(&loss)?;
		for parameter in model.all_parameters()? {
			assert!(
				parameter
					.gradient()
					.expect("Char MoE Transformer parameter gradient is missing")
					.read_f32()?
					.iter()
					.all(|value| value.is_finite())
			);
		}
		Ok(())
	}
);

test_vk!(
	canonical_char_moe_transformer_completes_the_cpp_300_step_gate,
	engine,
	{
		let model = nlp::CharMoeTransformer::new(&engine)?;
		let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, 0.01)?;
		let mut sampler = nlp::CharSampler::new(nlp::BATCH_SIZE)?;
		let (initial_input, initial_target) = sampler.next_values()?;
		let input = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE, nlp::CONTEXT_LENGTH],
			&initial_input,
		)?;
		let target = oa::Matrix::from_slice(
			&engine,
			[nlp::BATCH_SIZE * nlp::CONTEXT_LENGTH],
			&initial_target,
		)?;
		let mut program = oa::ml::TrainingProgram::capture(&engine, &mut optimizer, || {
			let tape = oa::ml::GradientTape::new();
			let logits = model.forward(&input)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
			tape.backward(&loss)?;
			Ok(loss)
		})?;
		let mut initial_loss = 0.0_f32;
		let mut training_loss = 0.0_f32;
		for step in 0..nlp::TRAINING_STEPS {
			if step != 0 {
				let (next_input, next_target) = sampler.next_values()?;
				program.upload_input(&input, &next_input)?;
				program.upload_input(&target, &next_target)?;
			}
			training_loss = program.replay_and_wait(&engine, &mut optimizer)?;
			if step == 0 {
				initial_loss = training_loss;
			}
		}
		let diagnostics = program.diagnostics();
		assert_eq!(diagnostics.command_recording_count(), 1);
		assert_eq!(diagnostics.command_cache_hit_count(), 299);
		assert_eq!(diagnostics.submission_count(), 300);
		assert_eq!(diagnostics.input_upload_count(), 598);
		let logits = model.forward(&input)?;
		let final_loss = oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0];
		let accuracy = nlp::accuracy(&logits, &target)?;
		let generated = nlp::generate_greedy(
			&engine,
			&model,
			nlp::GENERATION_PROMPT,
			nlp::GENERATION_LENGTH,
		)?;
		eprintln!(
			"Char MoE: initial={initial_loss:.6} training={training_loss:.6} final={final_loss:.6} accuracy={:.3}% generated={generated:?}",
			accuracy * 100.0
		);
		assert!(initial_loss > 3.0);
		assert!(
			(final_loss - nlp::CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS).abs() < 0.02,
			"final loss {final_loss} missed C++ {}; last training loss {training_loss}",
			nlp::CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS
		);
		assert!(
			(accuracy - nlp::CHAR_MOE_TRANSFORMER_CPP_ACCURACY).abs() < 0.01,
			"accuracy {}% missed C++ {}%",
			accuracy * 100.0,
			nlp::CHAR_MOE_TRANSFORMER_CPP_ACCURACY * 100.0
		);
		assert_eq!(optimizer.step_count(), nlp::TRAINING_STEPS as u32);
		assert_eq!(generated, nlp::CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION);

		let checkpoint = std::env::temp_dir().join(format!(
			"oars-char-moe-transformer-{}.oam",
			std::process::id()
		));
		oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
		let reloaded = nlp::CharMoeTransformer::new(&engine)?;
		let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, 0.01)?;
		oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
		assert_eq!(
			nlp::accuracy(&reloaded.forward(&input)?, &target)?,
			accuracy
		);
		assert_eq!(
			nlp::generate_greedy(
				&engine,
				&reloaded,
				nlp::GENERATION_PROMPT,
				nlp::GENERATION_LENGTH,
			)?,
			generated
		);
		assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
		std::fs::remove_file(&checkpoint).expect("remove completed MoE checkpoint");
		Ok(())
	}
);
