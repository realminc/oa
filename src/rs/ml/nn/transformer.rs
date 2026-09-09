use std::rc::Rc;

use crate::{Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, gelu};
use super::{LayerNorm, Linear, MultiHeadAttention};

/// One pre-normalized causal Transformer block.
pub struct TransformerBlock {
	model_width: usize,
	hidden_width: usize,
	sequence_length: usize,
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
			sequence_length,
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
		let normalized = self.norm_attention.forward(input)?;
		let attended = self.attention.forward(&normalized)?;
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
	pub const fn sequence_length(&self) -> usize {
		self.sequence_length
	}
	pub const fn num_heads(&self) -> usize {
		self.num_heads
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
