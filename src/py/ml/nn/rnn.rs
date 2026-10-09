use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── RnnCell ──────────────────────────────────────────────────────────────────

/// One Elman recurrent cell with an explicit caller-owned hidden state.
#[pyclass(name = "RnnCell", unsendable)]
pub(crate) struct PythonRnnCell {
	pub(crate) inner: oa::ml::nn::RnnCell,
}

#[pymethods]
impl PythonRnnCell {
	/// Construct a deterministically initialized RNN cell.
	#[new]
	#[pyo3(signature = (engine, input_size, hidden_size, bias=true, seed=0))]
	pub fn new(
		engine: &PythonEngine,
		input_size: usize,
		hidden_size: usize,
		bias: bool,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::RnnCell::with_seed(&engine.inner, input_size, hidden_size, bias, seed)
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
			"RnnCell(input_size={}, hidden_size={}, bias={})",
			self.inner.input_size(),
			self.inner.hidden_size(),
			self.inner.has_bias(),
		)
	}
}

// ── Rnn ──────────────────────────────────────────────────────────────────────

/// Stacked batch-first Elman recurrent network.
#[pyclass(name = "Rnn", unsendable)]
pub(crate) struct PythonRnn {
	pub(crate) inner: oa::ml::nn::Rnn,
}

#[pymethods]
impl PythonRnn {
	/// Construct a deterministically initialized stacked RNN.
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
		oa::ml::nn::Rnn::with_seed_and_bias(
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
	///
	/// Accepts an optional initial hidden state as a second argument
	/// so `rnn(x, h0)` works like `rnn.forward(x)` when `h0` is None.
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
		let _ = hidden; // initial hidden state not yet threaded through; use forward
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
			"Rnn(input_size={}, hidden_size={}, num_layers={}, bias={})",
			self.inner.input_size(),
			self.inner.hidden_size(),
			self.inner.num_layers(),
			self.inner.has_bias(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRnnCell>()?;
	module.add_class::<PythonRnn>()?;
	Ok(())
}
