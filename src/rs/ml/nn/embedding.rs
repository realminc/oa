use crate::{DType, Engine, Error, Matrix, Result};

use super::super::{
	Module, ModuleRegistry, Parameter, autograd, lowering::matrix as dispatch, random,
};

/// Trainable FP32 lookup table indexed by an arbitrary-shape U32 matrix.
pub struct Embedding {
	num_embeddings: usize,
	embedding_dim: usize,
	weight: Parameter,
	registry: ModuleRegistry,
}

impl Embedding {
	/// Construct a deterministically initialized embedding table.
	///
	/// # Errors
	///
	/// Returns an error when either dimension is zero, shape arithmetic overflows,
	/// or allocation/upload fails.
	pub fn with_seed(
		engine: &Engine,
		num_embeddings: usize,
		embedding_dim: usize,
		seed: u64,
	) -> Result<Self> {
		if num_embeddings == 0 || embedding_dim == 0 {
			return Err(Error::invalid_argument(
				"embedding dimensions must be nonzero",
			));
		}
		let count = num_embeddings
			.checked_mul(embedding_dim)
			.ok_or_else(|| Error::invalid_argument("embedding weight size overflows usize"))?;
		let limit = 1.0_f32 / (embedding_dim as f32).sqrt();
		let values = random::symmetric_uniform(count, limit, seed);
		let weight = Matrix::from_f32(engine, [num_embeddings, embedding_dim], &values)?;
		Self::from_matrix(weight)
	}

	/// Construct an embedding from an exact FP32 table shaped `[V, D]`.
	///
	/// # Errors
	///
	/// Returns an error unless `weight` is a nonempty rank-two FP32 matrix.
	pub fn from_matrix(weight: Matrix) -> Result<Self> {
		let [num_embeddings, embedding_dim] = weight.shape() else {
			return Err(Error::invalid_argument(
				"embedding weight must have rank two",
			));
		};
		if *num_embeddings == 0 || *embedding_dim == 0 || weight.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"embedding requires a nonempty FP32 weight [V, D]",
			));
		}
		let num_embeddings = *num_embeddings;
		let embedding_dim = *embedding_dim;
		let weight = Parameter::new("weight", weight)?;
		let mut registry = ModuleRegistry::new();
		registry.register_parameter("weight", weight.clone())?;
		Ok(Self {
			num_embeddings,
			embedding_dim,
			weight,
			registry,
		})
	}

	/// Gather one embedding row per U32 index.
	///
	/// The output shape is the complete index shape followed by `embedding_dim`.
	/// Out-of-range indices produce NaN rows without reading outside the table.
	///
	/// # Errors
	///
	/// Returns an error when indices are not U32, the engines differ, shape or ABI
	/// arithmetic overflows, or runtime recording fails.
	pub fn forward(&self, indices: &Matrix) -> Result<Matrix> {
		let (weight, version, requires_grad) = self.weight.snapshot();
		let output = dispatch::embedding(&weight, indices)?;
		if requires_grad {
			autograd::record_embedding(indices, &output, self.weight.clone(), weight, version)?;
		}
		Ok(output)
	}

	pub const fn num_embeddings(&self) -> usize {
		self.num_embeddings
	}

	pub const fn embedding_dim(&self) -> usize {
		self.embedding_dim
	}

	pub fn weight(&self) -> Parameter {
		self.weight.clone()
	}

	pub fn parameters(&self) -> [Parameter; 1] {
		[self.weight.clone()]
	}
}

impl Module for Embedding {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Embedding::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
