use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Pre-normalized SwiGLU feed-forward network with a residual connection.
#[pyclass(name = "Ffn", unsendable)]
pub(crate) struct PythonFfn {
	pub(crate) inner: oa::ml::nn::Ffn,
}

#[pymethods]
impl PythonFfn {
	/// Construct a deterministically initialized donor-compatible FFN.
	///
	/// Accepts `dim`/`hidden_dim` as aliases for `model_width`/`hidden_width`.
	#[new]
	#[pyo3(signature = (engine, model_width=None, hidden_width=None, epsilon = 1e-5, seed = 0, dim=None, hidden_dim=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		model_width: Option<usize>,
		hidden_width: Option<usize>,
		epsilon: f32,
		seed: u64,
		dim: Option<usize>,
		hidden_dim: Option<usize>,
	) -> PyResult<Self> {
		let mw = model_width
			.or(dim)
			.ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("Ffn requires model_width or dim"))?;
		let hw = hidden_width.or(hidden_dim).unwrap_or(mw * 4);
		oa::ml::nn::Ffn::with_seed(&engine.inner, mw, hw, epsilon, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Apply the pre-normalized gated projection and residual connection.
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
	pub fn model_width(&self) -> usize {
		self.inner.model_width()
	}

	#[getter]
	pub fn hidden_width(&self) -> usize {
		self.inner.hidden_width()
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon()
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
			"Ffn(model_width={}, hidden_width={}, epsilon={})",
			self.inner.model_width(),
			self.inner.hidden_width(),
			self.inner.epsilon(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonFfn>()?;
	Ok(())
}
