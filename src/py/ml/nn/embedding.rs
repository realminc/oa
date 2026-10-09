use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Trainable FP32 lookup table indexed by U8, U32, or I32 indices.
#[pyclass(name = "Embedding", unsendable)]
pub(crate) struct PythonEmbedding {
	pub(crate) inner: oa::ml::nn::Embedding,
}

#[pymethods]
impl PythonEmbedding {
	/// Construct a deterministically initialized embedding table.
	#[new]
	pub fn new(
		engine: &PythonEngine,
		num_embeddings: usize,
		embedding_dim: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Embedding::with_seed(&engine.inner, num_embeddings, embedding_dim, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Gather one embedding row per index.
	pub fn forward(&self, indices: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&indices.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn __call__(&self, indices: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.forward(indices)
	}

	#[getter]
	pub fn num_embeddings(&self) -> usize {
		self.inner.num_embeddings()
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

	pub fn parameters(&self) -> Vec<PythonParameter> {
		self
			.inner
			.parameters()
			.into_iter()
			.map(|inner| PythonParameter { inner })
			.collect()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Embedding(num_embeddings={}, embedding_dim={})",
			self.inner.num_embeddings(),
			self.inner.embedding_dim(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonEmbedding>()?;
	Ok(())
}
