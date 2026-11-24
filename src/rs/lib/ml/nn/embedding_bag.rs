//! Inference-and-training bag-of-embeddings lookup over per-bag offsets.

use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::random;
use super::super::{Module, ModuleRegistry};
use super::Embedding;

/// Reduction applied to each bag of looked-up embeddings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingBagMode {
	/// Elementwise mean of each bag's embeddings (`nn.EmbeddingBag` default).
	#[default]
	Mean,
	/// Elementwise sum of each bag's embeddings.
	Sum,
}

/// Bag-of-embeddings lookup: `embedding` lookup followed by a per-bag
/// reduction, covering the `nn.EmbeddingBag` contract.
///
/// `forward` takes the flattened index matrix plus host `offsets` delimiting
/// bags — the same boundary choice as `torch.nn.EmbeddingBag(mode='mean'|'sum')`,
/// with offsets kept on the host side of the CPU/GPU contract because bag
/// boundaries are data-shaping metadata, not device state.
///
/// The bag loop composes `slice` + `sum`/`mean` + `concat`, so gradients flow
/// through `Embedding`'s own adjoint into the shared weight — a composed
/// contract, not a fused kernel.
pub struct EmbeddingBag {
	embedding: Rc<Embedding>,
	mode: EmbeddingBagMode,
	registry: ModuleRegistry,
}

impl EmbeddingBag {
	/// Construct with a uniformly initialized `[V, D]` table matching
	/// [`Embedding::new_with_seed_on`].
	///
	/// # Errors
	///
	/// Returns an error for zero dimensions or table allocation failure.
	pub fn new_with_seed_on(
		engine: &Engine,
		num_embeddings: usize,
		embedding_dim: usize,
		mode: EmbeddingBagMode,
		seed: u64,
	) -> Result<Self> {
		let embedding = Embedding::new_with_seed_on(engine, num_embeddings, embedding_dim, seed)?;
		Self::new_from_embedding(embedding, mode)
	}

	/// Construct with a standard-normal table scaled by `stddev`.
	///
	/// # Errors
	///
	/// Returns an error for zero dimensions, non-finite or negative `stddev`,
	/// or table allocation failure.
	pub fn new_with_seed_normal_on(
		engine: &Engine,
		num_embeddings: usize,
		embedding_dim: usize,
		stddev: f32,
		mode: EmbeddingBagMode,
		seed: u64,
	) -> Result<Self> {
		if !stddev.is_finite() || stddev < 0.0 {
			return Err(Error::invalid_argument(
				"embedding bag stddev must be finite and non-negative",
			));
		}
		let count = num_embeddings
			.checked_mul(embedding_dim)
			.ok_or_else(|| Error::invalid_argument("embedding bag weight size overflows usize"))?;
		if count == 0 {
			return Err(Error::invalid_argument(
				"embedding bag dimensions must be nonzero",
			));
		}
		let values = random::normal(count, stddev, seed);
		let weight = Matrix::new_from_f32_on(engine, [num_embeddings, embedding_dim], &values)?;
		Self::new_from_embedding(Embedding::new_from_matrix(weight)?, mode)
	}

	/// Wrap an existing embedding table.
	///
	/// # Errors
	///
	/// Returns an error when child registration fails.
	pub fn new_from_embedding(embedding: Embedding, mode: EmbeddingBagMode) -> Result<Self> {
		let mut registry = ModuleRegistry::new();
		let embedding = Rc::new(embedding);
		registry.register_module("embedding", Rc::clone(&embedding) as Rc<dyn Module>)?;
		Ok(Self {
			embedding,
			mode,
			registry,
		})
	}

	/// Wrap an exact FP32 `[V, D]` table.
	///
	/// # Errors
	///
	/// Returns an error unless `weight` passes [`Embedding::new_from_matrix`].
	pub fn new_from_matrix(weight: Matrix, mode: EmbeddingBagMode) -> Result<Self> {
		Self::new_from_embedding(Embedding::new_from_matrix(weight)?, mode)
	}

	/// Return the configured per-bag reduction.
	pub const fn mode(&self) -> EmbeddingBagMode {
		self.mode
	}

	/// Return the wrapped lookup table module.
	pub fn embedding(&self) -> &Embedding {
		&self.embedding
	}

	/// Look up `indices` and reduce each `[offsets[b], offsets[b+1])` bag.
	///
	/// `offsets` holds `num_bags + 1` boundaries over the flattened index
	/// stream. Empty bags (`offsets[b] == offsets[b+1]`) are rejected rather
	/// than silently producing NaN under `Mean`; empty output requires
	/// `num_bags > 0`.
	///
	/// # Errors
	///
	/// Returns an error when `indices` is empty, `offsets` is malformed
	/// (fewer than two entries, non-monotonic, or not ending at the index
	/// count), any bag is empty, or the lookup/reduction dispatch fails.
	pub fn forward(&self, indices: &Matrix, offsets: &[usize]) -> Result<Matrix> {
		let index_count = indices.num_elements();
		if matches!(indices.dtype(), DType::F32 | DType::F16) {
			return Err(Error::invalid_argument(format!(
				"embedding bag indices must be integer; found {}",
				indices.dtype().token(),
			)));
		}
		if index_count == 0 {
			return Err(Error::invalid_argument("embedding bag requires indices"));
		}
		if offsets.len() < 2 {
			return Err(Error::invalid_argument(
				"embedding bag offsets must carry num_bags + 1 boundaries",
			));
		}
		if offsets[0] != 0
			|| !offsets.windows(2).all(|pair| pair[0] <= pair[1])
			|| *offsets.last().unwrap_or(&0) != index_count
		{
			return Err(Error::invalid_argument(format!(
				"embedding bag offsets must start at 0, be non-decreasing, and end at the index count {index_count}"
			)));
		}
		if offsets.windows(2).any(|pair| pair[0] == pair[1]) {
			return Err(Error::invalid_argument(
				"embedding bag does not admit empty bags",
			));
		}
		let embeddings = self.embedding.forward(indices)?;
		let mut rows = Vec::with_capacity(offsets.len() - 1);
		for pair in offsets.windows(2) {
			let bag = matrix::slice(&embeddings, 0, pair[0] as i64, pair[1] as i64)?;
			let reduced = match self.mode {
				EmbeddingBagMode::Mean => matrix::mean(&bag, 0)?,
				EmbeddingBagMode::Sum => matrix::sum(&bag, 0)?,
			};
			// The reduce ops keep the bag axis as a size-1 row.
			rows.push(reduced);
		}
		matrix::concat(&rows, 0)
	}
}

impl Module for EmbeddingBag {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"EmbeddingBag requires indices plus host offsets; call EmbeddingBag::forward(indices, offsets)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
