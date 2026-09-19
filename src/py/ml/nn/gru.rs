use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── GruCell ──────────────────────────────────────────────────────────────────

/// One gated recurrent unit cell with an explicit caller-owned hidden state.
#[pyclass(name = "GruCell", unsendable)]
pub(crate) struct PythonGruCell {
	pub(crate) inner: oa::ml::nn::GruCell,
}

#[pymethods]
impl PythonGruCell {
	/// Construct a deterministically initialized GRU cell.
	#[new]
	#[pyo3(signature = (engine, input_size, hidden_size, bias=true, seed=0))]
	pub fn new(
		engine: &PythonEngine,
		input_size: usize,
		hidden_size: usize,
		bias: bool,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::GruCell::with_seed(&engine.inner, input_size, hidden_size, bias, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Create a zero hidden state `[batch, hidden]`.
	pub fn zero_state(&self, batch: usize) -> PyResult<PythonMatrix> {
		self
			.inner
			.zero_state(batch)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply one recurrent step to `[B, I]` input and `[B, H]` hidden state.
	pub fn step(&self, input: &PythonMatrix, hidden: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.step(&input.inner, &hidden.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn __call__(&self, input: &PythonMatrix, hidden: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.step(input, hidden)
	}

	#[getter]
	pub fn input_size(&self) -> usize {
		self.inner.input_size()
	}

	#[getter]
	pub fn hidden_size(&self) -> usize {
		self.inner.hidden_size()
	}

	#[getter]
	pub fn has_bias(&self) -> bool {
		self.inner.has_bias()
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
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
			"GruCell(input_size={}, hidden_size={}, bias={})",
			self.inner.input_size(),
			self.inner.hidden_size(),
			self.inner.has_bias(),
		)
	}
}

// ── Gru ──────────────────────────────────────────────────────────────────────

/// Stacked batch-first gated recurrent unit.
#[pyclass(name = "Gru", unsendable)]
pub(crate) struct PythonGru {
	pub(crate) inner: oa::ml::nn::Gru,
}

#[pymethods]
impl PythonGru {
	/// Construct a deterministically initialized stacked GRU.
	#[new]
	#[pyo3(signature = (engine, input_size, hidden_size, num_layers=1, bias=true, seed=0))]
	pub fn new(
		engine: &PythonEngine,
		input_size: usize,
		hidden_size: usize,
		num_layers: usize,
		bias: bool,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Gru::with_seed(
			&engine.inner,
			input_size,
			hidden_size,
			num_layers,
			bias,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Evaluate the complete `[B, S, I]` sequence.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[pyo3(signature = (input, hidden=None))]
	pub fn __call__(
		&self,
		input: &PythonMatrix,
		hidden: Option<&PythonMatrix>,
	) -> PyResult<PythonMatrix> {
		let _ = hidden;
		self.forward(input)
	}

	#[getter]
	pub fn input_size(&self) -> usize {
		self.inner.input_size()
	}

	#[getter]
	pub fn hidden_size(&self) -> usize {
		self.inner.hidden_size()
	}

	#[getter]
	pub fn num_layers(&self) -> usize {
		self.inner.num_layers()
	}

	#[getter]
	pub fn has_bias(&self) -> bool {
		self.inner.has_bias()
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
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

	pub fn layer_parameters(&self, layer: usize) -> Option<Vec<PythonParameter>> {
		self.inner.layer_parameters(layer).map(|params| {
			params
				.into_iter()
				.map(|inner| PythonParameter { inner })
				.collect()
		})
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Gru(input_size={}, hidden_size={}, num_layers={}, bias={})",
			self.inner.input_size(),
			self.inner.hidden_size(),
			self.inner.num_layers(),
			self.inner.has_bias(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonGruCell>()?;
	module.add_class::<PythonGru>()?;
	Ok(())
}
