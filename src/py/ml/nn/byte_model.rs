use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── ByteEmbedding ─────────────────────────────────────────────────────────────

/// Fixed 256-row trainable embedding for packed U8 or U32 byte IDs.
#[pyclass(name = "ByteEmbedding", unsendable)]
pub(crate) struct PythonByteEmbedding {
	pub(crate) inner: oa::ml::nn::ByteEmbedding,
}

#[pymethods]
impl PythonByteEmbedding {
	/// Construct a deterministically initialized byte embedding.
	#[new]
	#[pyo3(signature = (engine, embedding_dim, seed = 0))]
	pub fn new(engine: &PythonEngine, embedding_dim: usize, seed: u64) -> PyResult<Self> {
		oa::ml::nn::ByteEmbedding::with_seed(&engine.inner, embedding_dim, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Gather one row per packed U8 or U32 byte ID.
	pub fn forward(&self, byte_ids: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&byte_ids.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn embedding_dim(&self) -> usize {
		self.inner.embedding_dim()
	}

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ByteEmbedding(embedding_dim={})",
			self.inner.embedding_dim()
		)
	}
}

// ── ByteHead ──────────────────────────────────────────────────────────────────

/// Trainable affine output head from hidden rows to 256 byte logits.
#[pyclass(name = "ByteHead", unsendable)]
pub(crate) struct PythonByteHead {
	pub(crate) inner: oa::ml::nn::ByteHead,
}

#[pymethods]
impl PythonByteHead {
	/// Construct a deterministically initialized byte head.
	#[new]
	#[pyo3(signature = (engine, input_features, seed = 0))]
	pub fn new(engine: &PythonEngine, input_features: usize, seed: u64) -> PyResult<Self> {
		oa::ml::nn::ByteHead::with_seed(&engine.inner, input_features, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Project hidden states to 256 byte logits.
	pub fn forward(&self, hidden: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&hidden.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn input_features(&self) -> usize {
		self.inner.input_features()
	}

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.bias(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!("ByteHead(input_features={})", self.inner.input_features())
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonByteEmbedding>()?;
	module.add_class::<PythonByteHead>()?;
	Ok(())
}
