use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Trainable FP32 RMSNorm over the final input dimension.
#[pyclass(name = "RmsNorm", unsendable)]
pub(crate) struct PythonRmsNorm {
	pub(crate) inner: oa::ml::nn::RmsNorm,
}

#[pymethods]
impl PythonRmsNorm {
	/// Construct RMSNorm with unit weight.
	///
	/// Accepts `features` as an alias for `dimension`.  The `seed` keyword
	/// is accepted for API compatibility but ignored.
	#[new]
	#[pyo3(signature = (engine, dimension=None, epsilon=1e-5, features=None, seed=None))]
	pub fn new(
		engine: &PythonEngine,
		dimension: Option<usize>,
		epsilon: f32,
		features: Option<usize>,
		seed: Option<u64>,
	) -> PyResult<Self> {
		let _ = seed;
		let dim = dimension.or(features).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("RmsNorm requires dimension or features")
		})?;
		oa::ml::nn::RmsNorm::new(&engine.inner, dim, epsilon)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Normalize each row by its root mean square and apply weight.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn __call__(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.forward(input)
	}

	#[getter]
	pub fn dimension(&self) -> usize {
		self.inner.dimension()
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon()
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
			"RmsNorm(dimension={}, epsilon={})",
			self.inner.dimension(),
			self.inner.epsilon()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRmsNorm>()?;
	Ok(())
}
