//! Canonical controlled NLP workloads.

use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result};

use crate::ml::{BpeTokenizer, Module, ModuleRegistry, nn};

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
/// Canonical BPE vocabulary target: 256 bytes plus 64 learned merges.
pub const BPE_VOCAB_SIZE: usize = 320;
/// Learned merge budget used by every BPE suite row.
pub const BPE_MERGES: usize = BPE_VOCAB_SIZE - crate::ml::byte::VOCAB_SIZE;
/// Reference greedy result produced by the accepted 300-step Char RNN run.
pub const CHAR_RNN_REFERENCE_GENERATION: &str =
	"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Reference bytes reproduced by accepted fresh-process Byte RNN runs.
pub const BYTE_RNN_REFERENCE_GENERATION: &[u8] =
	b"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Accepted OARS final loss for the donor-equivalent Byte RNN workload.
pub const BYTE_RNN_FINAL_LOSS: f32 = 0.186_080;
/// Accepted OARS final-batch accuracy for the donor-equivalent Byte RNN workload.
pub const BYTE_RNN_ACCURACY: f32 = 0.921_875;
/// Reference bytes reproduced by accepted fresh-process Byte GRU runs.
pub const BYTE_GRU_REFERENCE_GENERATION: &[u8] =
	b"to be or to be ther the mind to suffer the slings and arroubles and by oppostioubles ";
/// Accepted OARS final loss for the donor-equivalent Byte GRU workload.
pub const BYTE_GRU_FINAL_LOSS: f32 = 0.499_343;
/// Accepted OARS final-batch accuracy for the donor-equivalent Byte GRU workload.
pub const BYTE_GRU_ACCURACY: f32 = 0.855_469;
/// Reference bytes reproduced by accepted fresh-process Byte Transformer runs.
pub const BYTE_TRANSFORMER_REFERENCE_GENERATION: &[u8] =
	b"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Accepted OARS final loss for the donor-equivalent Byte Transformer workload.
pub const BYTE_TRANSFORMER_FINAL_LOSS: f32 = 0.190_565;
/// Accepted OARS final-batch accuracy for the Byte Transformer workload.
pub const BYTE_TRANSFORMER_ACCURACY: f32 = 0.928_711;
/// Reference bytes reproduced by accepted fresh-process BPE RNN runs.
pub const BPE_RNN_REFERENCE_GENERATION: &[u8] =
	b"to be not to be that is the question whether tis nobler in the mind to suffer the sli";
/// Accepted OARS final loss for the donor-equivalent BPE RNN workload.
pub const BPE_RNN_FINAL_LOSS: f32 = 0.020_861;
/// Accepted OARS final-batch token accuracy for the BPE RNN workload.
pub const BPE_RNN_ACCURACY: f32 = 0.988_281;
/// Reference bytes reproduced by accepted fresh-process BPE GRU runs.
pub const BPE_GRU_REFERENCE_GENERATION: &[u8] =
	b"to bee arms against a sea of troubles and by opposing end them to be or not to be tha";
/// Accepted OARS final loss for the donor-equivalent BPE GRU workload.
pub const BPE_GRU_FINAL_LOSS: f32 = 0.021_224;
/// Accepted OARS final-batch token accuracy for the BPE GRU workload.
pub const BPE_GRU_ACCURACY: f32 = 0.988_281;
/// Reference bytes reproduced by accepted fresh-process BPE Transformer runs.
pub const BPE_TRANSFORMER_REFERENCE_GENERATION: &[u8] =
	b"to be end them to be or not to be that is the question whether tis nobler in the mind";
/// Accepted OARS final loss for the donor-equivalent BPE Transformer workload.
pub const BPE_TRANSFORMER_FINAL_LOSS: f32 = 0.020_028;
/// Accepted OARS final-batch token accuracy for the BPE Transformer workload.
pub const BPE_TRANSFORMER_ACCURACY: f32 = 0.989_258;
/// Reference bytes reproduced by accepted fresh-process Byte MoE runs.
pub const BYTE_MOE_REFERENCE_GENERATION: &[u8] =
	b"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Accepted OARS final loss for the donor-equivalent Byte MoE workload.
pub const BYTE_MOE_FINAL_LOSS: f32 = 0.193_275;
/// Accepted OARS final-batch accuracy for the Byte MoE workload.
pub const BYTE_MOE_ACCURACY: f32 = 0.925_781;
/// Reference bytes reproduced by accepted fresh-process BPE MoE runs.
pub const BPE_MOE_REFERENCE_GENERATION: &[u8] =
	b"to bee or not to be that is the question whether tis nobler in the mind to suffer the";
/// Accepted OARS final loss for the donor-equivalent BPE MoE workload.
pub const BPE_MOE_FINAL_LOSS: f32 = 0.019_962;
/// Accepted OARS final-batch accuracy for the BPE MoE workload.
pub const BPE_MOE_ACCURACY: f32 = 0.988_281;
/// Reference bytes reproduced by accepted fresh-process Byte Mamba-3 runs.
pub const BYTE_MAMBA3_REFERENCE_GENERATION: &[u8] =
	b"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Accepted OARS final loss for the donor-equivalent Byte Mamba-3 workload.
pub const BYTE_MAMBA3_FINAL_LOSS: f32 = 0.207_903;
/// Accepted OARS final-batch accuracy for the Byte Mamba-3 workload.
pub const BYTE_MAMBA3_ACCURACY: f32 = 0.929_688;
/// Reference bytes for the donor Empyrealm-Core fidelity row.
pub const BYTE_EMPYREALM_REFERENCE_GENERATION: &[u8] = BYTE_MAMBA3_REFERENCE_GENERATION;
/// Accepted OARS final loss for the donor Empyrealm-Core fidelity workload.
pub const BYTE_EMPYREALM_FINAL_LOSS: f32 = BYTE_MAMBA3_FINAL_LOSS;
/// Accepted OARS final-batch accuracy for the Empyrealm-Core fidelity workload.
pub const BYTE_EMPYREALM_ACCURACY: f32 = BYTE_MAMBA3_ACCURACY;
/// Reference bytes reproduced by accepted fresh-process BPE Mamba-3 runs.
pub const BPE_MAMBA3_REFERENCE_GENERATION: &[u8] =
	b"to beion wher tis nobler in the mind to suffer the slings and arrows of outrageous fo";
/// Accepted OARS final loss for the donor-equivalent BPE Mamba-3 workload.
pub const BPE_MAMBA3_FINAL_LOSS: f32 = 0.019_415;
/// Accepted OARS final-batch accuracy for the BPE Mamba-3 workload.
pub const BPE_MAMBA3_ACCURACY: f32 = 0.989_258;
/// Reference text reproduced by accepted fresh-process Char GRU runs.
pub const CHAR_GRU_REFERENCE_GENERATION: &str =
	"to be or not to be that is the question whether tis nobler in the mind to suffer the ";
/// Accepted OARS final loss for the donor-equivalent Char GRU workload.
pub const CHAR_GRU_FINAL_LOSS: f32 = 0.179_721;
/// Accepted OARS final-batch accuracy for the Char GRU workload.
pub const CHAR_GRU_ACCURACY: f32 = 0.927_734;
/// Reference greedy result produced by OA C++'s canonical Char Transformer.
pub const CHAR_TRANSFORMER_REFERENCE_GENERATION: &str =
	"to be or not to be that is the question whether tis nobler in the mind to suffer the ";
/// Reference greedy result produced by the accepted Char MoE Transformer run.
pub const CHAR_MOE_TRANSFORMER_REFERENCE_GENERATION: &str =
	"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// Reference greedy result produced by the accepted Char Mamba-3 run.
pub const CHAR_MAMBA3_REFERENCE_GENERATION: &str =
	"to be that is the question whether tis nobler in the mind to suffer the slings and ar";
/// OA C++ final-loss reference from the canonical local 300-step run.
pub const CHAR_TRANSFORMER_CPP_FINAL_LOSS: f32 = 0.190_238;
/// OA C++ final-batch accuracy reference from the canonical local 300-step run.
pub const CHAR_TRANSFORMER_CPP_ACCURACY: f32 = 0.927;
/// OA C++/Android Char MoE final-loss reference for the canonical workload.
pub const CHAR_MOE_TRANSFORMER_CPP_FINAL_LOSS: f32 = 0.190_7;
/// OA C++/Android Char MoE final-batch accuracy reference.
pub const CHAR_MOE_TRANSFORMER_CPP_ACCURACY: f32 = 0.932;
/// Accepted OARS final-loss reference for the donor-equivalent Char Mamba-3 workload.
pub const CHAR_MAMBA3_FINAL_LOSS: f32 = 0.190_578;
/// Accepted OARS final-batch accuracy for the donor-equivalent Char Mamba-3 workload.
pub const CHAR_MAMBA3_ACCURACY: f32 = 0.929_688;

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

/// Deterministic dense all-position next-byte sampler over the canonical corpus.
pub struct ByteSampler {
	tokens: Vec<u8>,
	batch_size: usize,
	cursor: usize,
}

impl ByteSampler {
	/// Construct the canonical raw-byte sampler.
	///
	/// # Errors
	///
	/// Returns an error when `batch_size` is zero.
	pub fn new(batch_size: usize) -> Result<Self> {
		if batch_size == 0 {
			return Err(Error::invalid_argument(
				"NLP byte sampler batch size must be nonzero",
			));
		}
		Ok(Self {
			tokens: CORPUS.as_bytes().to_vec(),
			batch_size,
			cursor: 0,
		})
	}

	/// Upload the next packed-U8 input and U32 shifted-target pair.
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

	/// Produce the next host values for stable captured-program input slots.
	///
	/// # Errors
	///
	/// Returns an error when batch shape arithmetic overflows.
	pub fn next_values(&mut self) -> Result<(Vec<u8>, Vec<u32>)> {
		let count = self
			.batch_size
			.checked_mul(CONTEXT_LENGTH)
			.ok_or_else(|| Error::invalid_argument("NLP byte batch size overflows usize"))?;
		let limit = self.tokens.len() - CONTEXT_LENGTH - 1;
		let mut input = vec![0_u8; count];
		let mut target = vec![0_u32; count];
		for batch in 0..self.batch_size {
			let start = (self.cursor + batch * 7) % limit;
			for position in 0..CONTEXT_LENGTH {
				let output = batch * CONTEXT_LENGTH + position;
				input[output] = self.tokens[start + position];
				target[output] = u32::from(self.tokens[start + position + 1]);
			}
		}
		self.cursor = (self.cursor + self.batch_size) % limit;
		Ok((input, target))
	}
}

/// Deterministic dense all-position sampler over canonical BPE tokens.
pub struct BpeSampler<'tokenizer> {
	tokenizer: &'tokenizer BpeTokenizer,
	tokens: Vec<u32>,
	batch_size: usize,
	cursor: usize,
	last_batch_bytes: usize,
}

impl<'tokenizer> BpeSampler<'tokenizer> {
	/// Construct a sampler over the canonical corpus and learned vocabulary.
	///
	/// # Errors
	///
	/// Returns an error for a zero batch or when tokenization is too short for
	/// one shifted context window.
	pub fn new(batch_size: usize, tokenizer: &'tokenizer BpeTokenizer) -> Result<Self> {
		if batch_size == 0 {
			return Err(Error::invalid_argument(
				"NLP BPE sampler batch size must be nonzero",
			));
		}
		let tokens = tokenizer.encode_text(CORPUS);
		if tokens.len() <= CONTEXT_LENGTH + 1 {
			return Err(Error::invalid_argument(
				"NLP BPE corpus must contain more than one shifted context",
			));
		}
		Ok(Self {
			tokenizer,
			tokens,
			batch_size,
			cursor: 0,
			last_batch_bytes: 0,
		})
	}

	/// Produce and upload the next U32 BPE input/target pair.
	///
	/// # Errors
	///
	/// Returns an error when host preparation or Matrix upload fails.
	pub fn next(&mut self, engine: &Engine) -> Result<(Matrix, Matrix)> {
		let (input, target) = self.next_values()?;
		Ok((
			Matrix::from_slice(engine, [self.batch_size, CONTEXT_LENGTH], &input)?,
			Matrix::from_slice(engine, [self.batch_size, CONTEXT_LENGTH], &target)?,
		))
	}

	/// Produce the next host values for captured-program input slots.
	///
	/// # Errors
	///
	/// Returns an error when shape, byte accounting, or token expansion overflows.
	pub fn next_values(&mut self) -> Result<(Vec<u32>, Vec<u32>)> {
		let count = self
			.batch_size
			.checked_mul(CONTEXT_LENGTH)
			.ok_or_else(|| Error::invalid_argument("NLP BPE batch size overflows usize"))?;
		let limit = self.tokens.len() - CONTEXT_LENGTH - 1;
		let mut input = vec![0_u32; count];
		let mut target = vec![0_u32; count];
		self.last_batch_bytes = 0;
		for batch in 0..self.batch_size {
			let start = (self.cursor + batch * 7) % limit;
			for position in 0..CONTEXT_LENGTH {
				let output = batch * CONTEXT_LENGTH + position;
				input[output] = self.tokens[start + position];
				target[output] = self.tokens[start + position + 1];
				self.last_batch_bytes = self
					.last_batch_bytes
					.checked_add(self.tokenizer.token_byte_len(target[output])?)
					.ok_or_else(|| Error::resource_exhausted("NLP BPE byte count overflows usize"))?;
			}
		}
		self.cursor = (self.cursor + self.batch_size) % limit;
		Ok((input, target))
	}

	/// Exact source bytes represented by the most recent shifted targets.
	pub const fn last_batch_bytes(&self) -> usize {
		self.last_batch_bytes
	}

	/// Mean source bytes represented by one target token in the last batch.
	pub fn last_batch_bytes_per_token(&self) -> f64 {
		self.last_batch_bytes as f64 / (self.batch_size * CONTEXT_LENGTH) as f64
	}

	/// Return exact source bytes represented by the next target batch without
	/// advancing the sampler.
	///
	/// # Errors
	///
	/// Returns an error when token expansion or byte accounting overflows.
	pub fn next_batch_bytes(&self) -> Result<usize> {
		let limit = self.tokens.len() - CONTEXT_LENGTH - 1;
		let mut bytes = 0_usize;
		for batch in 0..self.batch_size {
			let start = (self.cursor + batch * 7) % limit;
			for position in 0..CONTEXT_LENGTH {
				bytes = bytes
					.checked_add(
						self
							.tokenizer
							.token_byte_len(self.tokens[start + position + 1])?,
					)
					.ok_or_else(|| Error::resource_exhausted("NLP BPE byte count overflows usize"))?;
			}
		}
		Ok(bytes)
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

/// Canonical byte language model: ByteEmbedding → Elman RNN → ByteHead.
pub struct ByteRnn {
	registry: ModuleRegistry,
	embedding: Rc<nn::ByteEmbedding>,
	recurrent: Rc<nn::Rnn>,
	head: Rc<nn::ByteHead>,
}

impl ByteRnn {
	/// Construct the deterministic donor-shaped 31,104-parameter model.
	///
	/// # Errors
	///
	/// Returns an error when parameter allocation or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::ByteEmbedding::with_seed(engine, MODEL_WIDTH, RNG_SEED)?);
		let recurrent = Rc::new(nn::Rnn::with_seed(
			engine,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			RNG_SEED.wrapping_add(1),
		)?);
		let head = Rc::new(nn::ByteHead::with_seed(
			engine,
			HIDDEN_WIDTH,
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

	/// Evaluate dense next-byte logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U8/U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Byte RNN tokens must have shape [B, S]",
			));
		};
		if !matches!(tokens.dtype(), DType::U8 | DType::U32) || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"Byte RNN tokens must be a nonempty U8 or U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Byte RNN row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for ByteRnn {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteRnn::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical byte language model: ByteEmbedding → GRU → ByteHead.
pub struct ByteGru {
	registry: ModuleRegistry,
	embedding: Rc<nn::ByteEmbedding>,
	recurrent: Rc<nn::Gru>,
	head: Rc<nn::ByteHead>,
}

impl ByteGru {
	/// Construct the deterministic donor-shaped 43,648-parameter model.
	///
	/// # Errors
	///
	/// Returns an error when parameter allocation or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::ByteEmbedding::with_seed(engine, MODEL_WIDTH, RNG_SEED)?);
		let recurrent = Rc::new(nn::Gru::with_seed(
			engine,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			true,
			RNG_SEED.wrapping_add(1),
		)?);
		let head = Rc::new(nn::ByteHead::with_seed(
			engine,
			HIDDEN_WIDTH,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("gru", recurrent.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			recurrent,
			head,
		})
	}

	/// Evaluate dense next-byte logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U8/U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Byte GRU tokens must have shape [B, S]",
			));
		};
		if !matches!(tokens.dtype(), DType::U8 | DType::U32) || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"Byte GRU tokens must be a nonempty U8 or U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Byte GRU row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for ByteGru {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteGru::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical character language model: Embedding → GRU → Linear.
pub struct CharGru {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	recurrent: Rc<nn::Gru>,
	head: Rc<nn::Linear>,
}

impl CharGru {
	/// Construct the deterministic donor-shaped 21,435-parameter model.
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
		let recurrent = Rc::new(nn::Gru::with_seed(
			engine,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			true,
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
		registry.register_module("gru", recurrent.clone())?;
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
	/// Returns an error unless tokens are nonempty U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Char GRU tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"Char GRU tokens must be a nonempty U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Char GRU row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for CharGru {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		CharGru::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical BPE language model: Embedding → Elman RNN → Linear.
pub struct BpeRnn {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	recurrent: Rc<nn::Rnn>,
	head: Rc<nn::Linear>,
}

impl BpeRnn {
	/// Construct the deterministic donor-shaped 37,312-parameter model.
	///
	/// # Errors
	///
	/// Returns an error when parameter allocation or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			BPE_VOCAB_SIZE,
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
			BPE_VOCAB_SIZE,
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

	/// Evaluate dense next-token logits for every BPE position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"BPE RNN tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"BPE RNN tokens must be a nonempty U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("BPE RNN row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for BpeRnn {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BpeRnn::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical BPE language model: Embedding → GRU → Linear.
pub struct BpeGru {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	recurrent: Rc<nn::Gru>,
	head: Rc<nn::Linear>,
}

impl BpeGru {
	/// Construct the deterministic donor-shaped 49,856-parameter model.
	///
	/// # Errors
	///
	/// Returns an error when parameter allocation or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			BPE_VOCAB_SIZE,
			MODEL_WIDTH,
			RNG_SEED,
		)?);
		let recurrent = Rc::new(nn::Gru::with_seed(
			engine,
			MODEL_WIDTH,
			HIDDEN_WIDTH,
			1,
			true,
			RNG_SEED.wrapping_add(1),
		)?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			HIDDEN_WIDTH,
			BPE_VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("gru", recurrent.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			recurrent,
			head,
		})
	}

	/// Evaluate dense next-token logits for every BPE position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, S]`, or when
	/// forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"BPE GRU tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 {
			return Err(Error::invalid_argument(
				"BPE GRU tokens must be a nonempty U32 Matrix [B, S]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("BPE GRU row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let recurrent = self.recurrent.forward(&embedded)?;
		self.head.forward(&recurrent.reshape([rows, HIDDEN_WIDTH])?)
	}
}

impl Module for BpeGru {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BpeGru::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical byte Transformer language model over the shared NN owner.
pub struct ByteTransformer {
	inner: nn::Transformer,
}

impl ByteTransformer {
	/// Construct the deterministic donor one-block, one-head byte model.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		Ok(Self {
			inner: nn::Transformer::with_seed(
				engine,
				crate::ml::byte::VOCAB_SIZE,
				CONTEXT_LENGTH,
				MODEL_WIDTH,
				HIDDEN_WIDTH,
				1,
				1,
				1.0e-5,
				RNG_SEED,
			)?,
		})
	}

	/// Evaluate all-position logits for packed-U8 or U32 byte tokens.
	///
	/// # Errors
	///
	/// Returns an error for an invalid token contract or failed child operation.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		self.inner.forward(tokens)
	}
}

impl Module for ByteTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Canonical BPE Transformer language model over the shared NN owner.
pub struct BpeTransformer {
	inner: nn::Transformer,
}

impl BpeTransformer {
	/// Construct the deterministic donor one-block, one-head BPE model.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		Ok(Self {
			inner: nn::Transformer::with_seed(
				engine,
				BPE_VOCAB_SIZE,
				CONTEXT_LENGTH,
				MODEL_WIDTH,
				HIDDEN_WIDTH,
				1,
				1,
				1.0e-5,
				RNG_SEED,
			)?,
		})
	}

	/// Evaluate all-position logits for U32 BPE tokens.
	///
	/// # Errors
	///
	/// Returns an error for an invalid token contract or failed child operation.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		if tokens.dtype() != DType::U32 {
			return Err(Error::invalid_argument(
				"BPE Transformer tokens must use U32 storage",
			));
		}
		self.inner.forward(tokens)
	}
}

impl Module for BpeTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BpeTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Canonical byte Transformer with four sparse top-two experts.
pub struct ByteMoeTransformer {
	inner: nn::Transformer,
}

impl ByteMoeTransformer {
	/// Construct the donor one-block, one-head byte MoE recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		Ok(Self {
			inner: nn::Transformer::with_seed_moe(
				engine,
				crate::ml::byte::VOCAB_SIZE,
				CONTEXT_LENGTH,
				MODEL_WIDTH,
				MOE_EXPERT_HIDDEN_WIDTH,
				1,
				1,
				MOE_NUM_EXPERTS,
				MOE_EXPERTS_PER_TOKEN,
				1.0e-5,
				RNG_SEED,
			)?,
		})
	}

	/// Evaluate all-position logits for packed-U8 or U32 byte tokens.
	///
	/// # Errors
	///
	/// Returns an error for an invalid token contract or failed child operation.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		self.inner.forward(tokens)
	}
}

impl Module for ByteMoeTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteMoeTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Canonical BPE Transformer with four sparse top-two experts.
pub struct BpeMoeTransformer {
	inner: nn::Transformer,
}

impl BpeMoeTransformer {
	/// Construct the donor one-block, one-head BPE MoE recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		Ok(Self {
			inner: nn::Transformer::with_seed_moe(
				engine,
				BPE_VOCAB_SIZE,
				CONTEXT_LENGTH,
				MODEL_WIDTH,
				MOE_EXPERT_HIDDEN_WIDTH,
				1,
				1,
				MOE_NUM_EXPERTS,
				MOE_EXPERTS_PER_TOKEN,
				1.0e-5,
				RNG_SEED,
			)?,
		})
	}

	/// Evaluate all-position logits for U32 BPE tokens.
	///
	/// # Errors
	///
	/// Returns an error for an invalid token contract or failed child operation.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		if tokens.dtype() != DType::U32 {
			return Err(Error::invalid_argument(
				"BPE MoE Transformer tokens must use U32 storage",
			));
		}
		self.inner.forward(tokens)
	}
}

impl Module for BpeMoeTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BpeMoeTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

fn canonical_mamba3(engine: &Engine, seed: u64) -> Result<nn::Mamba3> {
	nn::Mamba3::with_seed(
		engine,
		nn::Mamba3Config {
			model_width: MODEL_WIDTH,
			state_size: 32,
			expand: 2,
			head_dim: 16,
			num_groups: 1,
			mimo_rank: 1,
			rope_fraction: 0.5,
			dt_min: 0.001,
			dt_max: 0.1,
			dt_init_floor: 0.0001,
			a_floor: 0.0001,
			output_norm: true,
		},
		seed,
	)
}

/// Canonical raw-byte Mamba-3 language model with a flat residual head.
pub struct ByteMamba3 {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	mamba: Rc<nn::Mamba3>,
	head: Rc<nn::Linear>,
}

impl ByteMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` byte recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			crate::ml::byte::VOCAB_SIZE,
			MODEL_WIDTH,
			RNG_SEED,
		)?);
		let mamba = Rc::new(canonical_mamba3(engine, RNG_SEED.wrapping_add(1))?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			MODEL_WIDTH,
			crate::ml::byte::VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("mamba3", mamba.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			mamba,
			head,
		})
	}

	/// Evaluate dense next-byte logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U8/U32 `[B, S]` with
	/// `S <= 16`, or when forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Byte Mamba3 tokens must have shape [B, S]",
			));
		};
		if !matches!(tokens.dtype(), DType::U8 | DType::U32)
			|| *batch == 0
			|| *sequence == 0
			|| *sequence > CONTEXT_LENGTH
		{
			return Err(Error::invalid_argument(
				"Byte Mamba3 tokens must be nonempty U8/U32 [B, S] with S <= 16",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Byte Mamba3 row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let sequence_output = self.mamba.forward(&embedded)?;
		let residual = crate::matrix::add(
			&sequence_output.reshape([rows, MODEL_WIDTH])?,
			&embedded.reshape([rows, MODEL_WIDTH])?,
		)?;
		self.head.forward(&residual)
	}
}

impl Module for ByteMamba3 {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteMamba3::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical byte language model using the donor Empyrealm-Core ownership tree.
pub struct ByteEmpyrealm {
	registry: ModuleRegistry,
	core: Rc<nn::EmpyrealmCore>,
	head: Rc<nn::Linear>,
}

impl ByteEmpyrealm {
	/// Construct the deterministic donor fidelity model.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let core = Rc::new(nn::EmpyrealmCore::with_seed(
			engine,
			crate::ml::byte::VOCAB_SIZE,
			nn::Mamba3Config {
				model_width: MODEL_WIDTH,
				state_size: 32,
				expand: 2,
				head_dim: 16,
				num_groups: 1,
				mimo_rank: 1,
				rope_fraction: 0.5,
				dt_min: 0.001,
				dt_max: 0.1,
				dt_init_floor: 0.0001,
				a_floor: 0.0001,
				output_norm: true,
			},
			RNG_SEED,
		)?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			MODEL_WIDTH,
			crate::ml::byte::VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("core", core.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			core,
			head,
		})
	}

	/// Evaluate dense next-byte logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens satisfy the core contract, or when a child
	/// operation cannot be recorded.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let mixed = self.core.forward(tokens)?;
		let rows = tokens.element_count();
		self.head.forward(&mixed.reshape([rows, MODEL_WIDTH])?)
	}
}

impl Module for ByteEmpyrealm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteEmpyrealm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Canonical BPE Mamba-3 language model with a flat residual head.
pub struct BpeMamba3 {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	mamba: Rc<nn::Mamba3>,
	head: Rc<nn::Linear>,
}

impl BpeMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` BPE recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			BPE_VOCAB_SIZE,
			MODEL_WIDTH,
			RNG_SEED,
		)?);
		let mamba = Rc::new(canonical_mamba3(engine, RNG_SEED.wrapping_add(1))?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			MODEL_WIDTH,
			BPE_VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("mamba3", mamba.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			mamba,
			head,
		})
	}

	/// Evaluate dense next-token logits for every BPE position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, S]` with `S <= 16`,
	/// or when forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"BPE Mamba3 tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 || *sequence > CONTEXT_LENGTH {
			return Err(Error::invalid_argument(
				"BPE Mamba3 tokens must be nonempty U32 [B, S] with S <= 16",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("BPE Mamba3 row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let sequence_output = self.mamba.forward(&embedded)?;
		let residual = crate::matrix::add(
			&sequence_output.reshape([rows, MODEL_WIDTH])?,
			&embedded.reshape([rows, MODEL_WIDTH])?,
		)?;
		self.head.forward(&residual)
	}
}

impl Module for BpeMamba3 {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		BpeMamba3::forward(self, input)
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
		self
			.inner
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

/// Canonical character Mamba-3 language model with a flat residual head.
pub struct CharMamba3 {
	registry: ModuleRegistry,
	embedding: Rc<nn::Embedding>,
	mamba: Rc<nn::Mamba3>,
	head: Rc<nn::Linear>,
}

impl CharMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` SISO recipe.
	///
	/// # Errors
	///
	/// Returns an error when parameter construction or registration fails.
	pub fn new(engine: &Engine) -> Result<Self> {
		let embedding = Rc::new(nn::Embedding::with_seed(
			engine,
			CHAR_VOCAB_SIZE,
			MODEL_WIDTH,
			RNG_SEED,
		)?);
		let mamba = Rc::new(canonical_mamba3(engine, RNG_SEED.wrapping_add(1))?);
		let head = Rc::new(nn::Linear::with_seed(
			engine,
			MODEL_WIDTH,
			CHAR_VOCAB_SIZE,
			RNG_SEED.wrapping_add(2),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("embed", embedding.clone())?;
		registry.register_module("mamba3", mamba.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			registry,
			embedding,
			mamba,
			head,
		})
	}

	/// Evaluate dense next-character logits for every token position.
	///
	/// # Errors
	///
	/// Returns an error unless tokens are nonempty U32 `[B, S]` with
	/// `S <= 16`, or when forward recording fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Char Mamba3 tokens must have shape [B, S]",
			));
		};
		if tokens.dtype() != DType::U32 || *batch == 0 || *sequence == 0 || *sequence > 16 {
			return Err(Error::invalid_argument(
				"Char Mamba3 tokens must be nonempty U32 [B, S] with S <= 16",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Char Mamba3 row count overflows usize"))?;
		let embedded = self.embedding.forward(tokens)?;
		let sequence_output = self.mamba.forward(&embedded)?;
		let residual = crate::matrix::add(
			&sequence_output.reshape([rows, MODEL_WIDTH])?,
			&embedded.reshape([rows, MODEL_WIDTH])?,
		)?;
		self.head.forward(&residual)
	}

	/// Return the selective-state block.
	pub fn mamba(&self) -> &nn::Mamba3 {
		&self.mamba
	}
}

impl Module for CharMamba3 {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		CharMamba3::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
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
	crate::ml::metric::accuracy(logits, targets)
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

/// Greedily generate exact bytes with the canonical left-filled/sliding window.
///
/// # Errors
///
/// Returns an error for an empty prompt, invalid model output, or failed
/// execution/readback.
pub fn generate_bytes_greedy(
	engine: &Engine,
	model: &impl Module,
	prompt: &[u8],
	count: usize,
) -> Result<Vec<u8>> {
	if prompt.is_empty() {
		return Err(Error::invalid_argument(
			"NLP byte generation prompt must not be empty",
		));
	}
	let mut context = vec![0_u8; CONTEXT_LENGTH];
	let copied = prompt.len().min(CONTEXT_LENGTH);
	context[..copied].copy_from_slice(&prompt[..copied]);
	let mut filled = copied;
	let mut logit_row = filled - 1;
	let mut output = prompt.to_vec();
	for _ in 0..count {
		let input = Matrix::from_slice(engine, [1, CONTEXT_LENGTH], &context)?;
		let logits = model.forward(&input)?;
		if logits.shape() != [CONTEXT_LENGTH, crate::ml::byte::VOCAB_SIZE] {
			return Err(Error::failed_precondition(
				"byte model generation produced an invalid logit shape",
			));
		}
		let row = crate::matrix::slice(&logits, 0, logit_row as i64, (logit_row + 1) as i64)?
			.reshape([crate::ml::byte::VOCAB_SIZE])?;
		let next = crate::ml::byte::decode(&row)?[0];
		output.push(next);
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

/// Greedily generate BPE tokens until `byte_count` decoded source bytes.
///
/// The decoded result is truncated to `prompt.len() + byte_count`, matching the
/// donor when the last learned token crosses the requested byte boundary.
///
/// # Errors
///
/// Returns an error for an empty prompt, invalid model output/token IDs, byte
/// accounting overflow, or failed execution/readback.
pub fn generate_bpe_greedy(
	engine: &Engine,
	model: &impl Module,
	tokenizer: &BpeTokenizer,
	prompt: &[u8],
	byte_count: usize,
) -> Result<Vec<u8>> {
	if prompt.is_empty() {
		return Err(Error::invalid_argument(
			"NLP BPE generation prompt must not be empty",
		));
	}
	let prompt_tokens = tokenizer.encode(prompt);
	let copied = prompt_tokens.len().min(CONTEXT_LENGTH);
	let mut context = vec![0_u32; CONTEXT_LENGTH];
	context[..copied].copy_from_slice(&prompt_tokens[..copied]);
	let mut filled = copied.max(1);
	let mut logit_row = filled - 1;
	let mut generated = prompt_tokens;
	let mut generated_bytes = 0_usize;
	for _ in 0..byte_count {
		if generated_bytes >= byte_count {
			break;
		}
		let input = Matrix::from_slice(engine, [1, CONTEXT_LENGTH], &context)?;
		let logits = model.forward(&input)?;
		if logits.shape() != [CONTEXT_LENGTH, tokenizer.vocab_size()] {
			return Err(Error::failed_precondition(
				"BPE model generation produced an invalid logit shape",
			));
		}
		let row = crate::matrix::slice(&logits, 0, logit_row as i64, (logit_row + 1) as i64)?
			.reshape([tokenizer.vocab_size()])?;
		let sampled = crate::matrix::sample_logits(&row, 0.0, 0, 1.0, 0)?.read::<i32>()?;
		let next = u32::try_from(sampled[0])
			.map_err(|_| Error::failed_precondition("BPE model generated a negative token ID"))?;
		let next_index = usize::try_from(next)
			.map_err(|_| Error::failed_precondition("BPE token ID exceeds usize"))?;
		if next_index >= tokenizer.vocab_size() {
			return Err(Error::failed_precondition(
				"BPE model generated an out-of-range token ID",
			));
		}
		generated.push(next);
		generated_bytes = generated_bytes
			.checked_add(tokenizer.token_byte_len(next)?)
			.ok_or_else(|| Error::resource_exhausted("BPE generation byte count overflows usize"))?;
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
	let mut output = tokenizer.decode(&generated)?;
	output.truncate(prompt.len().saturating_add(byte_count));
	Ok(output)
}
