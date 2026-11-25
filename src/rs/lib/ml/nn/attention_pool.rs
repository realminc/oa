//! Attention pooling: a learned query attending over a sequence.

use std::rc::Rc;

use crate::{DType, Error, Matrix, Result, Runtime, matrix};

use super::super::{Module, ModuleRegistry, Parameter, autograd};
use super::MultiHeadAttention;

/// Pool a `[B*S, D]` sequence to `[B, D]` through a single learned query.
///
/// The query is a trainable `[1, D]` parameter broadcast to one query row
/// per batch element, then routed through [`MultiHeadAttention::forward_kv`]
/// — the same cross-attention path encoder-decoder uses, with the sequence
/// as key/value source. Equivalent to `nn.MultiheadAttention` pooling blocks
/// used by ViT-style attention poolers.
pub struct AttentionPool {
	model_width: usize,
	query: Parameter,
	attention: Rc<MultiHeadAttention>,
	registry: ModuleRegistry,
}

impl AttentionPool {
	/// Construct with a seeded single-query attention block and a
	/// zero-initialized query.
	///
	/// The internal attention uses query sequence length one; the key/value
	/// extent is taken from the input at each forward.
	///
	/// # Errors
	///
	/// Returns an error under the attention constructor contract or when
	/// allocation/registration fails.
	pub fn new_with_seed_on(
		runtime: &Runtime,
		model_width: usize,
		num_heads: usize,
		seed: u64,
	) -> Result<Self> {
		if model_width == 0 {
			return Err(Error::invalid_argument(
				"attention pool requires a nonzero model width",
			));
		}
		let attention = Rc::new(MultiHeadAttention::new_with_seed_on(
			runtime,
			model_width,
			num_heads,
			1,
			seed,
		)?);
		let query = Parameter::new(
			"query",
			Matrix::new_from_f32_on(runtime, [1, model_width], &vec![0.0_f32; model_width])?,
		)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("query", query.clone())?;
		registry.register_module("attention", Rc::clone(&attention) as Rc<dyn Module>)?;
		Ok(Self {
			model_width,
			query,
			attention,
			registry,
		})
	}

	/// Construct on the current runtime.
	///
	/// # Errors
	///
	/// Returns an error under the same contract as [`Self::new_with_seed_on`].
	pub fn new_with_seed(model_width: usize, num_heads: usize, seed: u64) -> Result<Self> {
		Self::new_with_seed_on(&Runtime::current()?, model_width, num_heads, seed)
	}

	/// Pool `input` shaped `[B*S, D]` to `[B, D]` for `batch` sequences.
	///
	/// # Errors
	///
	/// Returns an error unless `input` is F32 `[B*S, D]` with `B == batch`,
	/// or the attention forward fails.
	pub fn forward(&self, input: &Matrix, batch: usize) -> Result<Matrix> {
		let [rows, width] = input.shape() else {
			return Err(Error::invalid_argument(format!(
				"attention pool requires [B*S, D]; found {:?}",
				input.shape(),
			)));
		};
		if input.dtype() != DType::F32 || *width != self.model_width {
			return Err(Error::invalid_argument(format!(
				"attention pool requires F32 width {}; found {:?} {}",
				self.model_width,
				input.shape(),
				input.dtype().token(),
			)));
		}
		if batch == 0 || !rows.is_multiple_of(batch) {
			return Err(Error::invalid_argument(format!(
				"attention pool rows {rows} must be a multiple of batch {batch}"
			)));
		}
		autograd::record_parameter_leaf(&self.query)?;
		let query = matrix::repeat_interleave(&self.query.data(), batch, 0)?;
		self.attention.forward_kv(&query, input)
	}
}

impl Module for AttentionPool {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"AttentionPool requires the batch count; call AttentionPool::forward(input, batch)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
