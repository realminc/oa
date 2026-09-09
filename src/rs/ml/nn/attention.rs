use std::rc::Rc;

use crate::{Engine, Error, Matrix, Result};

use super::super::{Module, ModuleRegistry, scaled_dot_product_attention_causal};
use super::Linear;

/// Multi-head causal self-attention over packed `[B*S, D]` values.
pub struct MultiHeadAttention {
	model_width: usize,
	num_heads: usize,
	sequence_length: usize,
	query_projection: Rc<Linear>,
	key_projection: Rc<Linear>,
	value_projection: Rc<Linear>,
	output_projection: Rc<Linear>,
	registry: ModuleRegistry,
}

impl MultiHeadAttention {
	/// Construct the four projections for causal self-attention.
	///
	/// # Errors
	///
	/// Returns an error unless dimensions are positive and `model_width` is
	/// divisible by `num_heads`, or child construction/registration fails.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		num_heads: usize,
		sequence_length: usize,
		seed: u64,
	) -> Result<Self> {
		if model_width == 0
			|| num_heads == 0
			|| sequence_length == 0
			|| !model_width.is_multiple_of(num_heads)
		{
			return Err(Error::invalid_argument(
				"attention requires positive D/H/S and D divisible by H",
			));
		}
		let query_projection = Rc::new(Linear::with_seed(engine, model_width, model_width, seed)?);
		let key_projection = Rc::new(Linear::with_seed(
			engine,
			model_width,
			model_width,
			seed.wrapping_add(1),
		)?);
		let value_projection = Rc::new(Linear::with_seed(
			engine,
			model_width,
			model_width,
			seed.wrapping_add(2),
		)?);
		let output_projection = Rc::new(Linear::with_seed(
			engine,
			model_width,
			model_width,
			seed.wrapping_add(3),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("q_proj", query_projection.clone())?;
		registry.register_module("k_proj", key_projection.clone())?;
		registry.register_module("v_proj", value_projection.clone())?;
		registry.register_module("out_proj", output_projection.clone())?;
		Ok(Self {
			model_width,
			num_heads,
			sequence_length,
			query_projection,
			key_projection,
			value_projection,
			output_projection,
			registry,
		})
	}

	/// Apply projected causal self-attention.
	///
	/// # Errors
	///
	/// Returns an error unless input is F32 `[B*S, D]` for this configuration,
	/// or operation recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let query = self.query_projection.forward(input)?;
		let key = self.key_projection.forward(input)?;
		let value = self.value_projection.forward(input)?;
		let context = scaled_dot_product_attention_causal(
			&query,
			&key,
			&value,
			self.sequence_length,
			self.num_heads,
		)?;
		self.output_projection.forward(&context)
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}
	pub const fn sequence_length(&self) -> usize {
		self.sequence_length
	}
}

impl Module for MultiHeadAttention {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		MultiHeadAttention::forward(self, input)
	}
	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
