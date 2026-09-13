use crate::{Engine, Error, Matrix, Result};

use super::{Embedding, Linear};
use crate::ml::{Module, ModuleRegistry, Parameter, byte};

/// Fixed 256-row trainable embedding for packed U8 or U32 byte IDs.
pub struct ByteEmbedding {
	inner: Embedding,
}

impl ByteEmbedding {
	/// Construct a deterministically initialized byte embedding.
	///
	/// # Errors
	///
	/// Returns an error when `embedding_dim` is zero or initialization fails.
	pub fn with_seed(engine: &Engine, embedding_dim: usize, seed: u64) -> Result<Self> {
		Ok(Self {
			inner: Embedding::with_seed(engine, byte::VOCAB_SIZE, embedding_dim, seed)?,
		})
	}

	/// Construct a byte embedding from an exact FP32 `[256, D]` table.
	///
	/// # Errors
	///
	/// Returns an error unless the table has exactly 256 nonempty rows.
	pub fn from_matrix(weight: Matrix) -> Result<Self> {
		if weight.shape().first() != Some(&byte::VOCAB_SIZE) {
			return Err(Error::invalid_argument(
				"byte embedding weight must have shape [256, D]",
			));
		}
		Ok(Self {
			inner: Embedding::from_matrix(weight)?,
		})
	}

	/// Gather one row per packed U8 or U32 byte ID.
	///
	/// # Errors
	///
	/// Returns an error for an invalid input or runtime recording failure.
	pub fn forward(&self, byte_ids: &Matrix) -> Result<Matrix> {
		self.inner.forward(byte_ids)
	}

	/// Embedding width.
	pub const fn embedding_dim(&self) -> usize {
		self.inner.embedding_dim()
	}

	/// Stable trainable table handle.
	pub fn weight(&self) -> Parameter {
		self.inner.weight()
	}
}

impl Module for ByteEmbedding {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteEmbedding::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}

/// Trainable affine output head from hidden rows to 256 byte logits.
pub struct ByteHead {
	inner: Linear,
}

impl ByteHead {
	/// Construct a deterministically initialized byte head.
	///
	/// # Errors
	///
	/// Returns an error when `input_features` is zero or initialization fails.
	pub fn with_seed(engine: &Engine, input_features: usize, seed: u64) -> Result<Self> {
		Ok(Self {
			inner: Linear::with_seed(engine, input_features, byte::VOCAB_SIZE, seed)?,
		})
	}

	/// Project hidden states to 256 byte logits.
	///
	/// # Errors
	///
	/// Returns an error for an invalid input or runtime recording failure.
	pub fn forward(&self, hidden: &Matrix) -> Result<Matrix> {
		self.inner.forward(hidden)
	}

	/// Hidden input width.
	pub const fn input_features(&self) -> usize {
		self.inner.input_features()
	}

	/// Output weight handle.
	pub fn weight(&self) -> Parameter {
		self.inner.weight()
	}

	/// Output bias handle.
	pub fn bias(&self) -> Parameter {
		self.inner
			.bias()
			.expect("byte head always constructs a trainable bias")
	}
}

impl Module for ByteHead {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		ByteHead::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		self.inner.registry()
	}
}
