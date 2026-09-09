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
		self.head
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
