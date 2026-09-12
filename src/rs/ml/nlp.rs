//! Canonical controlled NLP workloads.

use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result};

use super::{Module, ModuleRegistry, nn};

/// Model-token context used by the OA NLP comparison suite.
pub const CONTEXT_LENGTH: usize = 16;
/// Embedding width used by the OA NLP comparison suite.
pub const MODEL_WIDTH: usize = 32;
/// Recurrent width used by the OA NLP comparison suite.
pub const HIDDEN_WIDTH: usize = 64;
/// Per-expert hidden width used by the canonical MoE Transformer row.
pub const MOE_EXPERT_HIDDEN_WIDTH: usize = 16;
/// Routed expert count used by the canonical MoE Transformer row.
pub const MOE_NUM_EXPERTS: usize = 4;
/// Experts selected for each token in the canonical MoE Transformer row.
pub const MOE_EXPERTS_PER_TOKEN: usize = 2;
/// Complete optimizer steps in one canonical tutorial run.
pub const TRAINING_STEPS: usize = 300;
/// Windows in one canonical training batch.
pub const BATCH_SIZE: usize = 64;
/// Deterministic model initialization seed.
pub const RNG_SEED: u64 = 20_260_714;
/// Fixed generation prompt shared by the suite.
pub const GENERATION_PROMPT: &str = "to be";
/// Source characters generated after the fixed prompt.
pub const GENERATION_LENGTH: usize = 80;
/// Character vocabulary size: `a` through `z`, plus space/unknown.
pub const CHAR_VOCAB_SIZE: usize = 27;
/// Reference greedy result produced by the accepted 300-step Char RNN run.
pub const CHAR_RNN_REFERENCE_GENERATION: &str =
	"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Reference greedy result produced by OA C++'s canonical Char Transformer.
pub const CHAR_TRANSFORMER_REFERENCE_GENERATION: &str =
	"to be or not to be that is the question whether tis nobler in the mind to suffer the ";
/// Reference greedy result produced by the accepted Char MoE Transformer run.
pub const CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION: &str =
	"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// OA C++ final-loss reference from the canonical local 300-step run.
pub const CHAR_TRANSFORMER_CPP_FINAL_LOSS: f32 = 0.190_238;
/// OA C++ final-batch accuracy reference from the canonical local 300-step run.
pub const CHAR_TRANSFORMER_CPP_ACCURACY: f32 = 0.927;
/// OA C++/Android Char MoE final-loss reference for the canonical workload.
pub const CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS: f32 = 0.190_7;
/// OA C++/Android Char MoE final-batch accuracy reference.
pub const CHAR_MOE_TRANSFORMER_CPP_ACCURACY: f32 = 0.932;

/// Exact 576-character teaching corpus shared with OA C++.
pub const CORPUS: &str = concat!(
	"to be or not to be that is the question whether tis nobler in the mind ",
	"to suffer the slings and arrows of outrageous fortune or to take arms ",
	"against a sea of troubles and by opposing end them ",
	"to be or not to be that is the question whether tis nobler in the mind ",
	"to suffer the slings and arrows of outrageous fortune or to take arms ",
	"against a sea of troubles and by opposing end them ",
	"to be or not to be that is the question whether tis nobler in the mind ",
	"to suffer the slings and arrows of outrageous fortune or to take arms ",
	"against a sea of troubles and by opposing end them ",
);

/// Encode one character using the canonical `a-z` plus space vocabulary.
pub const fn encode_char(character: char) -> u32 {
	if character >= 'a' && character <= 'z' {
		character as u32 - 'a' as u32
	} else if character >= 'A' && character <= 'Z' {
		character as u32 - 'A' as u32
	} else {
		26
	}
}

/// Decode one canonical character token. Invalid values decode to space.
pub const fn decode_char(token: u32) -> char {
	if token < 26 {
		match char::from_u32('a' as u32 + token) {
			Some(character) => character,
			None => ' ',
		}
	} else {
		' '
	}
}

/// Encode text into canonical character tokens.
pub fn encode(text: &str) -> Vec<u32> {
	text.chars().map(encode_char).collect()
}

/// Decode canonical character tokens.
pub fn decode(tokens: &[u32]) -> String {
	tokens.iter().copied().map(decode_char).collect()
}

/// Deterministic dense all-position next-character sampler.
pub struct CharSampler {
	tokens: Vec<u32>,
	batch_size: usize,
	cursor: usize,
}

impl CharSampler {
	/// Construct a sampler over the canonical corpus.
	///
	/// # Errors
	///
	/// Returns an error when `batch_size` is zero.
	pub fn new(batch_size: usize) -> Result<Self> {
		if batch_size == 0 {
			return Err(Error::invalid_argument(
				"NLP sampler batch size must be nonzero",
			));
		}
		Ok(Self {
			tokens: encode(CORPUS),
			batch_size,
			cursor: 0,
		})
	}

	/// Upload the next `[batch, 16]` input and shifted-target pair.
	///
	/// # Errors
	///
	/// Returns an error when shape arithmetic or Matrix upload fails.
	pub fn next(&mut self, engine: &Engine) -> Result<(Matrix, Matrix)> {
		let (input, target) = self.next_values()?;
		Ok((
			Matrix::from_slice(engine, [self.batch_size, CONTEXT_LENGTH], &input)?,
			Matrix::from_slice(engine, [self.batch_size, CONTEXT_LENGTH], &target)?,
		))
	}

	/// Produce the next host-side input and shifted-target token arrays.
	///
	/// This is the allocation-only sampler boundary used to upload new values into
	/// stable captured training-program slots without allocating new GPU buffers.
	///
	/// # Errors
	///
	/// Returns an error when batch shape arithmetic overflows.
	pub fn next_values(&mut self) -> Result<(Vec<u32>, Vec<u32>)> {
		let count = self
			.batch_size
			.checked_mul(CONTEXT_LENGTH)
			.ok_or_else(|| Error::invalid_argument("NLP batch size overflows usize"))?;
		let limit = self.tokens.len() - CONTEXT_LENGTH - 1;
		let mut input = vec![0_u32; count];
		let mut target = vec![0_u32; count];
		for batch in 0..self.batch_size {
			let start = (self.cursor + batch * 7) % limit;
			for position in 0..CONTEXT_LENGTH {
				let output = batch * CONTEXT_LENGTH + position;
				input[output] = self.tokens[start + position];
				target[output] = self.tokens[start + position + 1];
			}
		}
		self.cursor = (self.cursor + self.batch_size) % limit;
		Ok((input, target))
	}
}

/// Canonical character language model: Embedding → Elman RNN → Linear.
pub struct CharRnn {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	recurrent: Rc<nn::Rnn>,
	head: Rc<nn::Linear>,
}

impl CharRnn {
	/// Construct the canonical deterministic 8,891-parameter model.
	///
	/// # Errors
	///
	/// Returns an error when parameter allocation or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			CHAR_VOCAB_SIZE,
			MODEL_WIDTH,
			RNG_SEED,
		)?);
		let recurrent = Rc::new(nn::Rnn::with_seed(
			engine,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			RNG_SEED.wrapping_add(1),
		)?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			HIDDEN_WIDTH,
			CHAR_VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("rnn", recurrent.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			recurrent,
			head,
		})
	}

	/// Evaluate dense next-character logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless `tokens` is same-engine U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Char RNN tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"Char RNN tokens must be a nonempty U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Char RNN row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for CharRnn {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		CharRnn::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical 10,875-parameter character Transformer language model.
pub struct CharTransformer {
	inner: nn::Transformer,
}

impl CharTransformer {
	/// Construct the exact one-block, one-head OA C++ tutorial architecture.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let inner = nn::Transformer::with_seed(
			engine,
			CHAR_VOCAB_SIZE,
			CONTEXT_LENGTH,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			1,
			1.0e-5,
			RNG_SEED,
		)?;
		Ok(Self { inner })
	}

	/// Evaluate all-position next-character logits for `[B, 16]` tokens.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, 16]`, or recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		self.inner.forward(tokens)
	}
}

impl Module for CharTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		CharTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Canonical character Transformer with four sparse top-two experts.
pub struct CharMoeTransformer {
	inner: nn::Transformer,
}

impl CharMoeTransformer {
	/// Construct the donor one-block, one-head MoE Transformer recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let inner = nn::Transformer::with_seed_moe(
			engine,
			CHAR_VOCAB_SIZE,
			CONTEXT_LENGTH,
			MODEL_WIDTH,
			MOE_EXPERT_HIDDEN_WIDTH,
			1,
			1,
			MOE_NUM_EXPERTS,
			MOE_EXPERTS_PER_TOKEN,
			1.0e-5,
			RNG_SEED,
		)?;
		Ok(Self { inner })
	}

	/// Evaluate all-position next-character logits for `[B, 16]` tokens.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, 16]`, or recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		self.inner.forward(tokens)
	}

	/// Return the canonical routed expert module.
	pub fn moe(&self) -> &nn::Moe {
		self.inner
			.block(0)
			.and_then(nn::TransformerBlock::moe)
			.expect("CharMoeTransformer always owns one MoE block")
	}
}

impl Module for CharMoeTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		CharMoeTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Compute all-position argmax accuracy as a fraction in `[0, 1]`.
///
/// This is an explicit host-observation boundary.
///
/// # Errors
///
/// Returns an error unless logits are F32 `[N, C]`, targets contain `N` U32
/// values, or readback fails.
pub fn accuracy(logits: &Matrix, targets: &Matrix) -> Result<f32> {
	let [rows, classes] = logits.shape() else {
		return Err(Error::invalid_argument(
			"NLP accuracy logits must be [N, C]",
		));
	};
	if *rows == 0
		|| *classes == 0
		|| logits.dtype() != DType::F32
		|| targets.dtype() != DType::U32
		|| targets.shape() != [*rows]
	{
		return Err(Error::invalid_argument(
			"NLP accuracy requires nonempty F32 logits [N, C] and U32 targets [N]",
		));
	}
	super::metric::accuracy(logits, targets)
}

/// Greedily generate characters with the canonical left-filled/sliding window.
///
/// # Errors
///
/// Returns an error for an empty prompt, when model execution/readback fails,
/// or if the model violates its declared output shape.
pub fn generate_greedy(
	engine: &Engine,
	model: &impl Module,
	prompt: &str,
	count: usize,
) -> Result<String> {
	let prompt_tokens = encode(prompt);
	if prompt_tokens.is_empty() {
		return Err(Error::invalid_argument(
			"NLP generation prompt must not be empty",
		));
	}
	let mut context = vec![26_u32; CONTEXT_LENGTH];
	let copied = prompt_tokens.len().min(CONTEXT_LENGTH);
	context[..copied].copy_from_slice(&prompt_tokens[..copied]);
	let mut filled = copied;
	let mut logit_row = filled - 1;
	let mut output = prompt.to_owned();
	for _ in 0..count {
		let input = Matrix::from_slice(engine, [1, CONTEXT_LENGTH], &context)?;
		let logits = model.forward(&input)?;
		if logits.shape() != [CONTEXT_LENGTH, CHAR_VOCAB_SIZE] {
			return Err(Error::failed_precondition(
				"character model generation produced an invalid logit shape",
			));
		}
		let values = logits.read_f32()?;
		let begin = logit_row * CHAR_VOCAB_SIZE;
		let next = values[begin..begin + CHAR_VOCAB_SIZE]
			.iter()
			.enumerate()
			.max_by(|left, right| left.1.total_cmp(right.1))
			.map_or(0_u32, |(token, _)| token as u32);
		output.push(decode_char(next));
		if filled < CONTEXT_LENGTH {
			context[filled] = next;
			filled += 1;
			logit_row = filled - 1;
		} else {
			context.copy_within(1.., 0);
			context[CONTEXT_LENGTH - 1] = next;
			logit_row = CONTEXT_LENGTH - 1;
		}
	}
	Ok(output)
}
