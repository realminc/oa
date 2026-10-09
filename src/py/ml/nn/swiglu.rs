use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Donor-compatible SwiGLU MLP with optional projection biases.
#[pyclass(name = "Swiglu", unsendable)]
pub(crate) struct PythonSwiglu {
	pub(crate) inner: oa::ml::nn::Swiglu,
}

#[pymethods]
impl PythonSwiglu {
	/// Construct gate, up, and down projections with deterministic Xavier weights.
	///
	/// Accepts `input_dim`/`hidden_dim`/`output_dim` as aliases for
	/// `input_features`/`intermediate_size` (output_dim is ignored — the Rust
	/// implementation uses intermediate→input projection).
	#[new]
	#[pyo3(signature = (engine, input_features=None, intermediate_size=None, bias=true, seed=0, input_dim=None, hidden_dim=None, output_dim=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		input_features: Option<usize>,
		intermediate_size: Option<usize>,
		bias: bool,
		seed: u64,
		input_dim: Option<usize>,
		hidden_dim: Option<usize>,
		output_dim: Option<usize>,
	) -> PyResult<Self> {
		let _ = output_dim;
		let inf = input_features.or(input_dim).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("Swiglu requires input_features or input_dim")
		})?;
		let is = intermediate_size.or(hidden_dim).unwrap_or(inf * 2);
		oa::ml::nn::Swiglu::with_seed(&engine.inner, inf, is, bias, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Apply gate/up projections, fused SwiGLU activation, and down projection.
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
	pub fn input_features(&self) -> usize {
		self.inner.input_features()
	}

	#[getter]
	pub fn intermediate_size(&self) -> usize {
		self.inner.intermediate_size()
	}

	#[getter]
	pub fn has_bias(&self) -> bool {
		self.inner.has_bias()
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
			"Swiglu(input_features={}, intermediate_size={}, bias={})",
			self.inner.input_features(),
			self.inner.intermediate_size(),
			self.inner.has_bias(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonSwiglu>()?;
	Ok(())
}
