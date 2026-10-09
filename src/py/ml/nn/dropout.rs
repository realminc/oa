use pyo3::prelude::*;

use oa::ml::Module as _;

use crate::{error::python_error, matrix::PythonMatrix};

/// Inverted Dropout with explicit train/eval behavior.
#[pyclass(name = "Dropout", unsendable)]
pub(crate) struct PythonDropout {
	pub(crate) inner: oa::ml::nn::Dropout,
}

#[pymethods]
impl PythonDropout {
	/// Construct Dropout.  `seed=0` uses the thread-local seed source.
	#[new]
	#[pyo3(signature = (probability, seed=0))]
	pub fn new(probability: f32, seed: u64) -> PyResult<Self> {
		oa::ml::nn::Dropout::with_seed(probability, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Apply Dropout in training mode or return input identity in eval mode.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn probability(&self) -> f32 {
		self.inner.probability()
	}

	#[getter]
	pub fn training(&self) -> bool {
		self.inner.is_training()
	}

	pub fn train(&self) {
		self.inner.train(true);
	}

	pub fn eval(&self) {
		self.inner.eval();
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Dropout(p={}, training={})",
			self.inner.probability(),
			self.inner.is_training(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonDropout>()?;
	Ok(())
}
