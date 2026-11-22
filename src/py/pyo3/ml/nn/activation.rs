use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

macro_rules! activation_module {
	($(#[$meta:meta])* $pyname:literal, $rust_type:path, $struct_name:ident) => {
		$(#[$meta])*
		#[pyclass(name = $pyname, unsendable)]
		pub(crate) struct $struct_name {
			inner: $rust_type,
		}

		#[pymethods]
		impl $struct_name {
			#[new]
			pub fn new() -> Self {
				Self { inner: <$rust_type>::new() }
			}

			pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
				self.inner
					.forward(&input.inner)
					.map(PythonMatrix::wrap)
					.map_err(python_error)
			}

			pub fn __call__(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
				self.forward(input)
			}

			pub fn __repr__(&self) -> &str {
				$pyname
			}
		}
	};
}

activation_module!(
	/// Rectified linear unit module.
	"Relu",
	oa::ml::nn::Relu,
	PythonRelu
);

activation_module!(
	/// Gaussian error linear unit module.
	"Gelu",
	oa::ml::nn::Gelu,
	PythonGelu
);

activation_module!(
	/// Sigmoid linear unit module.
	"Silu",
	oa::ml::nn::Silu,
	PythonSilu
);

// ── Softmax / LogSoftmax ──────────────────────────────────────────────────────

/// Parameterless axis-aware Softmax module.
#[pyclass(name = "Softmax", unsendable)]
pub(crate) struct PythonSoftmax {
	inner: oa::ml::nn::Softmax,
}

#[pymethods]
impl PythonSoftmax {
	#[new]
	#[pyo3(signature = (dim = -1))]
	pub fn new(dim: i32) -> Self {
		Self {
			inner: oa::ml::nn::Softmax::new(dim),
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
	pub fn dim(&self) -> i32 {
		self.inner.dim()
	}

	pub fn __repr__(&self) -> String {
		format!("Softmax(dim={})", self.inner.dim())
	}
}

/// Parameterless axis-aware LogSoftmax module.
#[pyclass(name = "LogSoftmax", unsendable)]
pub(crate) struct PythonLogSoftmax {
	inner: oa::ml::nn::LogSoftmax,
}

#[pymethods]
impl PythonLogSoftmax {
	#[new]
	#[pyo3(signature = (dim = -1))]
	pub fn new(dim: i32) -> Self {
		Self {
			inner: oa::ml::nn::LogSoftmax::new(dim),
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
	pub fn dim(&self) -> i32 {
		self.inner.dim()
	}

	pub fn __repr__(&self) -> String {
		format!("LogSoftmax(dim={})", self.inner.dim())
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRelu>()?;
	module.add_class::<PythonGelu>()?;
	module.add_class::<PythonSilu>()?;
	module.add_class::<PythonSoftmax>()?;
	module.add_class::<PythonLogSoftmax>()?;
	Ok(())
}
