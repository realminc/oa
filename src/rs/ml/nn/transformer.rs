use std::{cell::Cell, rc::Rc};

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, matrix::gelu};
use super::{AttentionMode, LayerNorm, Linear, MultiHeadAttention};

/// One pre-normalized causal Transformer block.
pub struct TransformerBlock {
	model_width: usize,
	hidden_width: usize,
	sequence_length: Cell<usize>,
	num_heads: usize,
	norm_attention: Rc<LayerNorm>,
	attention: Rc<MultiHeadAttention>,
	norm_feed_forward: Rc<LayerNorm>,
	feed_forward_in: Rc<Linear>,
	feed_forward_out: Rc<Linear>,
	registry: ModuleRegistry,
}

impl TransformerBlock {
	/// Construct the canonical pre-norm attention and GELU feed-forward block.
	///
	/// # Errors
	///
	/// Returns an error for invalid dimensions or failed child construction.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		if hidden_width == 0 {
			return Err(Error::invalid_argument(
				"Transformer FFN width must be nonzero",
			));
		}
		let norm_attention = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let attention = Rc::new(MultiHeadAttention::with_seed(
			engine,
			model_width,
			num_heads,
			sequence_length,
			seed,
		)?);
		let norm_feed_forward = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let feed_forward_in = Rc::new(Linear::with_seed(
			engine,
			model_width,
			hidden_width,
			seed.wrapping_add(4),
		)?);
		let feed_forward_out = Rc::new(Linear::with_seed(
			engine,
			hidden_width,
			model_width,
			seed.wrapping_add(5),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("ln_attn", norm_attention.clone())?;
		registry.register_module("attention", attention.clone())?;
		registry.register_module("ln_ffn", norm_feed_forward.clone())?;
		registry.register_module("ffn1", feed_forward_in.clone())?;
		registry.register_module("ffn2", feed_forward_out.clone())?;
		Ok(Self {
			model_width,
			hidden_width,
			sequence_length: Cell::new(sequence_length),
			num_heads,
			norm_attention,
			attention,
			norm_feed_forward,
			feed_forward_in,
			feed_forward_out,
			registry,
		})
	}

	/// Apply pre-norm attention/residual followed by pre-norm GELU FFN/residual.
	///
	/// # Errors
	///
	/// Returns an error unless input is F32 `[B*S, D]` or child recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		self.forward_impl(input, None)
	}

	/// Apply this block with an explicit bidirectional additive attention mask.
	///
	/// # Errors
	///
	/// Returns an error unless the mask is F32 `[B*H*S, S]` for the current
	/// sequence geometry, or a child operation fails.
	pub fn forward_masked(&self, input: &Matrix, additive_mask: &Matrix) -> Result<Matrix> {
		self.forward_impl(input, Some(additive_mask))
	}

	fn forward_impl(&self, input: &Matrix, additive_mask: Option<&Matrix>) -> Result<Matrix> {
		let normalized = self.norm_attention.forward(input)?;
		let attended = match additive_mask {
			Some(mask) => self.attention.forward_masked(&normalized, mask)?,
			None => self.attention.forward(&normalized)?,
		};
		let residual = matrix::add(input, &attended)?;
		let normalized = self.norm_feed_forward.forward(&residual)?;
		let hidden = gelu(&self.feed_forward_in.forward(&normalized)?)?;
		let projected = self.feed_forward_out.forward(&hidden)?;
		matrix::add(&residual, &projected)
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}
	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}
	pub fn sequence_length(&self) -> usize {
		self.sequence_length.get()
	}
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}

	/// Select causal or bidirectional attention for this block.
	pub fn set_attention_mode(&self, mode: AttentionMode) {
		self.attention.set_mode(mode);
	}

	/// Return this block's attention visibility contract.
	pub fn attention_mode(&self) -> AttentionMode {
		self.attention.mode()
	}

	/// Change runtime sequence geometry without rebuilding block parameters.
	///
	/// # Errors
	///
	/// Returns an error when `sequence_length` is zero.
	pub fn set_sequence_length(&self, sequence_length: usize) -> Result<()> {
		self.attention.set_sequence_length(sequence_length)?;
		self.sequence_length.set(sequence_length);
		Ok(())
	}
}

impl Module for TransformerBlock {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		TransformerBlock::forward(self, input)
	}
	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Ready-to-train causal Transformer language model.
///
/// This is the idiomatic Rust spelling of donor `oa::NnTransformer`: token and
/// position embeddings, registered Transformer blocks, final normalization,
/// and a vocabulary projection.
pub struct Transformer {
	vocab_size: usize,
	context_length: usize,
	model_width: usize,
	hidden_width: usize,
	num_heads: usize,
	token_embedding: Rc<super::Embedding>,
	position_embedding: Rc<super::Embedding>,
	blocks: Vec<Rc<TransformerBlock>>,
	final_norm: Rc<LayerNorm>,
	head: Rc<Linear>,
	registry: ModuleRegistry,
}

impl Transformer {
	/// Construct a deterministic causal language model.
	///
	/// # Errors
	///
	/// Returns an error unless all dimensions are positive, the model width is
	/// divisible by the head count, positions fit U32, and every child can be
	/// constructed and registered.
	#[allow(clippy::too_many_arguments)]
	pub fn with_seed(
		engine: &Engine,
		vocab_size: usize,
		context_length: usize,
		model_width: usize,
		hidden_width: usize,
		num_layers: usize,
		num_heads: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		if vocab_size == 0
			|| context_length == 0
			|| model_width == 0
			|| hidden_width == 0
			|| num_layers == 0
			|| num_heads == 0
			|| !model_width.is_multiple_of(num_heads)
			|| u32::try_from(context_length - 1).is_err()
		{
			return Err(Error::invalid_argument(
				"Transformer requires positive dimensions, D divisible by H, and U32 positions",
			));
		}
		let token_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			vocab_size,
			model_width,
			seed,
		)?);
		let position_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			context_length,
			model_width,
			seed.wrapping_add(1),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("token_embedding", token_embedding.clone())?;
		registry.register_module("position_embedding", position_embedding.clone())?;
		let mut blocks = Vec::with_capacity(num_layers);
		for index in 0..num_layers {
			let seed_offset = (index as u64).wrapping_mul(6).wrapping_add(2);
			let block = Rc::new(TransformerBlock::with_seed(
				engine,
				model_width,
				hidden_width,
				context_length,
				num_heads,
				epsilon,
				seed.wrapping_add(seed_offset),
			)?);
			registry.register_module(format!("block_{index}"), block.clone())?;
			blocks.push(block);
		}
		let final_norm = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let head_seed = seed.wrapping_add((num_layers as u64).wrapping_mul(6).wrapping_add(2));
		let head = Rc::new(Linear::with_seed(
			engine,
			model_width,
			vocab_size,
			head_seed,
		)?);
		registry.register_module("final_norm", final_norm.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			vocab_size,
			context_length,
			model_width,
			hidden_width,
			num_heads,
			token_embedding,
			position_embedding,
			blocks,
			final_norm,
			head,
			registry,
		})
	}

	/// Evaluate all-position vocabulary logits for U32 tokens `[B, S]`.
	///
	/// # Errors
	///
	/// Returns an error unless `S` equals the configured context length and the
	/// input is nonempty U32, or a child operation fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Transformer tokens must have shape [batch, context_length]",
			));
		};
		if *batch == 0 || *sequence != self.context_length || tokens.dtype() != DType::U32 {
			return Err(Error::invalid_argument(
				"Transformer tokens must be nonempty U32 [batch, context_length]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Transformer row count overflows usize"))?;
		let position_values = (0..rows)
			.map(|index| (index % self.context_length) as u32)
			.collect::<Vec<_>>();
		let position_ids =
			Matrix::from_slice_handle(tokens.engine_handle(), vec![rows], &position_values)?;
		let token_values = self
			.token_embedding
			.forward(tokens)?
			.reshape([rows, self.model_width])?;
		let position_values = self.position_embedding.forward(&position_ids)?;
		let mut value = matrix::add(&token_values, &position_values)?;
		for block in &self.blocks {
			value = block.forward(&value)?;
		}
		self.head.forward(&self.final_norm.forward(&value)?)
	}

	pub const fn vocab_size(&self) -> usize {
		self.vocab_size
	}

	pub const fn context_length(&self) -> usize {
		self.context_length
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}

	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}

	pub fn num_layers(&self) -> usize {
		self.blocks.len()
	}

	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}
}

impl Module for Transformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Transformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
