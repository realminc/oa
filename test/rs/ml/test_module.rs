use std::rc::Rc;

use oa::ml::{Module, ModuleRegistry, Parameter};

struct CharRnn {
	registry: ModuleRegistry,
	embedding: Rc<oa::ml::nn::Embedding>,
	recurrent: Rc<oa::ml::nn::Rnn>,
	head: Rc<oa::ml::nn::Linear>,
	hidden_size: usize,
}

impl CharRnn {
	fn new(engine: &oa::Engine) -> oa::Result<Self> {
		let embedding = Rc::new(oa::ml::nn::Embedding::with_seed(engine, 3, 2, 11)?);
		let recurrent = Rc::new(oa::ml::nn::Rnn::with_seed(engine, 2, 2, 1, 12)?);
		let head = Rc::new(oa::ml::nn::Linear::with_seed(engine, 2, 3, 13)?);
		let alphabet = oa::Matrix::from_slice(
			engine,
			[3],
			&[u32::from(b'a'), u32::from(b'b'), u32::from(b'c')],
		)?;
		let mut registry = ModuleRegistry::new();
		registry.register_module("embedding", embedding.clone())?;
		registry.register_module("recurrent", recurrent.clone())?;
		registry.register_module("head", head.clone())?;
		registry.register_buffer("alphabet", alphabet, true)?;
		Ok(Self {
			registry,
			embedding,
			recurrent,
			head,
			hidden_size: 2,
		})
	}
}

impl Module for CharRnn {
	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		let embedded = self.embedding.forward(input)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		let [batch, sequence, _] = recurrent.shape() else {
			unreachable!("RNN guarantees a rank-three result")
		};
		// Matrix construction already proved the complete shape product fits usize.
		let rows = batch * sequence;
		self
			.head
			.forward(&recurrent.reshape([rows, self.hidden_size])?)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

struct Leaf {
	registry: ModuleRegistry,
}

impl Leaf {
	fn sharing(parameter: Parameter) -> oa::Result<Self> {
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("value", parameter)?;
		Ok(Self { registry })
	}
}

impl Module for Leaf {
	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		Ok(input.clone())
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

struct Branch {
	registry: ModuleRegistry,
}

impl Module for Branch {
	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		Ok(input.clone())
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

test_vk!(
	module_tree_drives_one_complete_character_rnn_step,
	engine,
	{
		let model = CharRnn::new(&engine)?;

		assert!(model.parameters().is_empty());
		let named = model.all_named_parameters()?;
		assert_eq!(
			named.iter().map(|entry| entry.path()).collect::<Vec<_>>(),
			[
				"embedding.weight",
				"recurrent.layer0.weight_ih",
				"recurrent.layer0.weight_hh",
				"recurrent.layer0.bias_ih",
				"recurrent.layer0.bias_hh",
				"head.weight",
				"head.bias",
			]
		);
		assert_eq!(model.num_parameters()?, 27);
		let buffers = model.all_named_buffers()?;
		assert_eq!(buffers.len(), 1);
		assert_eq!(buffers[0].path(), "alphabet");
		assert!(buffers[0].persistent());
		assert_eq!(buffers[0].data().read::<u32>()?, [97, 98, 99]);

		assert!(model.is_training());
		{
			let _eval = model.scoped_eval();
			assert!(!model.is_training());
			assert!(!model.embedding.is_training());
			assert!(!model.recurrent.is_training());
			assert!(!model.head.is_training());
		}
		assert!(model.is_training());
		assert!(model.embedding.is_training());
		assert!(model.recurrent.is_training());
		assert!(model.head.is_training());

		let parameters = model.all_parameters()?;
		let mut optimizer = oa::ml::AdamW::new(parameters.clone(), 0.01)?;
		let indices = oa::Matrix::from_slice(&engine, [1, 3], &[0_u32, 1, 2])?;
		let targets = oa::Matrix::from_slice(&engine, [3], &[1_u32, 2, 0])?;
		let tape = oa::ml::GradientTape::new();
		let logits = model.forward(&indices)?;
		let loss = oa::ml::loss::cross_entropy(&logits, &targets)?;
		tape.backward(&loss)?;
		assert_eq!(logits.shape(), [3, 3]);
		assert!(loss.read_f32()?[0].is_finite());
		assert!(
			parameters
				.iter()
				.all(|parameter| parameter.gradient().is_some())
		);
		optimizer.step()?;
		assert_eq!(optimizer.step_count(), 1);
		Ok(())
	}
);

test_vk!(module_registration_rejects_ambiguous_ownership, engine, {
	let parameter = Parameter::new("shared", oa::Matrix::from_f32(&engine, [1], &[1.0])?)?;
	let mut direct = ModuleRegistry::new();
	direct.register_parameter("value", parameter.clone())?;
	assert_eq!(
		direct
			.register_parameter("alias", parameter.clone())
			.expect_err("duplicate direct parameter was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	assert_eq!(
		direct
			.register_buffer(
				"invalid.name",
				oa::Matrix::from_f32(&engine, [1], &[0.0])?,
				false,
			)
			.expect_err("dotted local name was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	assert_eq!(
		oa::ml::AdamW::new([parameter.clone(), parameter.clone()], 0.01)
			.err()
			.expect("duplicate optimizer parameter was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	let left = Rc::new(Leaf::sharing(parameter.clone())?);
	let right = Rc::new(Leaf::sharing(parameter)?);
	let mut registry = ModuleRegistry::new();
	registry.register_module("left", left.clone())?;
	assert_eq!(
		registry
			.register_module("left_alias", left)
			.expect_err("duplicate child handle was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	registry.register_module("right", right)?;
	let branch = Branch { registry };
	let error = branch
		.all_named_parameters()
		.err()
		.expect("shared recursive parameter was accepted");
	assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition);
	assert!(error.message().contains("right.value"));
	Ok(())
});

test_vk!(dropout_module_obeys_train_and_eval_modes, engine, {
	let dropout = oa::ml::nn::Dropout::with_seed(0.5, 0x4452_4f50)?;
	let input = oa::Matrix::from_f32(&engine, [257], &[1.0; 257])?;
	let training = dropout.forward(&input)?.read_f32()?;
	assert!(training.contains(&0.0));
	assert!(training.contains(&2.0));
	assert!(dropout.parameters().is_empty());

	dropout.eval();
	assert_eq!(dropout.forward(&input)?.read_f32()?, vec![1.0; 257]);
	dropout.train(true);
	assert_eq!(dropout.forward(&input)?.read_f32()?, training);
	assert_eq!(dropout.probability(), 0.5);
	assert_eq!(
		oa::ml::nn::Dropout::new(f32::NAN)
			.err()
			.expect("NaN probability was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(
	utility_modules_preserve_views_and_validate_dimensions,
	engine,
	{
		let input = oa::Matrix::from_f32(
			&engine,
			[2, 3, 4],
			&(0..24).map(|value| value as f32).collect::<Vec<_>>(),
		)?;
		let identity = oa::ml::nn::Identity::new();
		let identity_output = identity.forward(&input)?;
		assert_eq!(identity_output.shape(), [2, 3, 4]);
		assert_eq!(identity_output.read_f32()?, input.read_f32()?);

		let default_flatten = oa::ml::nn::Flatten::default();
		assert_eq!(default_flatten.start_dim(), 1);
		assert_eq!(default_flatten.end_dim(), -1);
		assert_eq!(default_flatten.forward(&input)?.shape(), [2, 12]);
		assert_eq!(
			oa::ml::nn::Flatten::new(0, 1).forward(&input)?.shape(),
			[6, 4]
		);
		assert_eq!(
			oa::ml::nn::Flatten::new(-2, -1).forward(&input)?.shape(),
			[2, 12]
		);
		assert_eq!(
			oa::ml::nn::Flatten::new(3, -1).forward(&input)?.shape(),
			input.shape()
		);
		for flatten in [
			oa::ml::nn::Flatten::new(-4, -1),
			oa::ml::nn::Flatten::new(2, 1),
			oa::ml::nn::Flatten::new(0, 3),
		] {
			assert_eq!(
				flatten
					.forward(&input)
					.err()
					.expect("invalid Flatten dimensions were accepted")
					.kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		Ok(())
	}
);

test_vk!(
	sequential_owns_and_forwards_registered_children_in_order,
	engine,
	{
		let flatten = Rc::new(oa::ml::nn::Flatten::default());
		let head = Rc::new(oa::ml::nn::Linear::with_seed(&engine, 12, 2, 991)?);
		let relu = Rc::new(oa::ml::nn::Relu::new());
		let mut sequence = oa::ml::nn::Sequential::new();
		assert!(sequence.is_empty());
		sequence.add(flatten)?;
		sequence.add_named("head", head.clone())?;
		sequence.add(relu.clone())?;
		assert_eq!(sequence.len(), 3);
		assert_eq!(
			sequence
				.all_named_parameters()?
				.iter()
				.map(|entry| entry.path())
				.collect::<Vec<_>>(),
			["head.weight", "head.bias"]
		);

		let input = oa::Matrix::from_f32(&engine, [2, 3, 4], &[0.25; 24])?;
		let output = sequence.forward(&input)?;
		assert_eq!(output.shape(), [2, 2]);
		assert!(output.read_f32()?.iter().all(|value| *value >= 0.0));
		sequence.eval();
		assert!(!sequence.is_training());
		assert!(!head.is_training());
		assert!(!relu.is_training());

		let mut duplicate = oa::ml::nn::Sequential::new();
		duplicate.add(relu.clone())?;
		assert_eq!(
			duplicate
				.add(relu)
				.err()
				.expect("duplicate Sequential child was accepted")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);
