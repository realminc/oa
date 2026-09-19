use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

/// Return the input value unchanged.
#[pyclass(name = "Identity", unsendable)]
pub(crate) struct PythonIdentity {
	inner: oa::ml::nn::Identity,
}

#[pymethods]
impl PythonIdentity {
	#[new]
	pub fn new() -> Self {
		Self {
			inner: oa::ml::nn::Identity::new(),
		}
	}

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

	pub fn __repr__(&self) -> &str {
		"Identity"
	}
}

/// Flatten an inclusive range of Matrix dimensions into one dimension.
#[pyclass(name = "Flatten", unsendable)]
pub(crate) struct PythonFlatten {
	inner: oa::ml::nn::Flatten,
}

#[pymethods]
impl PythonFlatten {
	/// Construct with PyTorch-style dimension indices. Defaults `(1, -1)`.
	#[new]
	#[pyo3(signature = (start_dim = 1, end_dim = -1))]
	pub fn new(start_dim: isize, end_dim: isize) -> Self {
		Self {
			inner: oa::ml::nn::Flatten::new(start_dim, end_dim),
		}
	}

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
	pub fn start_dim(&self) -> isize {
		self.inner.start_dim()
	}

	#[getter]
	pub fn end_dim(&self) -> isize {
		self.inner.end_dim()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Flatten(start_dim={}, end_dim={})",
			self.inner.start_dim(),
			self.inner.end_dim()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonIdentity>()?;
	module.add_class::<PythonFlatten>()?;
	Ok(())
}
