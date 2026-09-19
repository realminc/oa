use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;
use super::mamba3::PythonMamba3State;

/// OA embedding-plus-Mamba-3 core with canonical Rust execution path.
#[pyclass(name = "EmpyrealmCore", unsendable)]
pub(crate) struct PythonEmpyrealmCore {
	pub(crate) inner: oa::ml::nn::EmpyrealmCore,
}

#[pymethods]
impl PythonEmpyrealmCore {
	/// Construct a deterministic token embedding and Mamba-3 mixer.
	#[new]
	pub fn new(
		engine: &PythonEngine,
		vocab_size: usize,
		config: &super::mamba3::PythonMamba3Config,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::EmpyrealmCore::with_seed(&engine.inner, vocab_size, config.inner, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Embed a U8/U32 `[batch, sequence]` token grid and apply the residual mixer.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply the residual mixer over pre-embedded `[B, S, D]` features.
	pub fn forward_embedded(&self, embedded: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_embedded(&embedded.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Allocate a recurrent step state for `batch_size` sequences.
	pub fn new_state(&self, batch_size: usize) -> PyResult<PythonMamba3State> {
		self
			.inner
			.new_state(batch_size)
			.map(|inner| PythonMamba3State { inner })
			.map_err(python_error)
	}

	/// Evaluate one pre-embedded recurrent step plus residual.
	pub fn step_embedded(
		&self,
		embedded: &PythonMatrix,
		state: &mut PythonMamba3State,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.step_embedded(&embedded.inner, &mut state.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn train(&self) {
		use oa::ml::Module as _;
		self.inner.train(true);
	}

	pub fn eval(&self) {
		use oa::ml::Module as _;
		self.inner.train(false);
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

	pub fn __repr__(&self) -> &str {
		"EmpyrealmCore"
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonEmpyrealmCore>()?;
	Ok(())
}
