use pyo3::prelude::*;

use crate::error::python_error;

use super::autograd::PythonParameter;

fn extract_parameters(params: Vec<PyRef<'_, PythonParameter>>) -> Vec<oa::ml::Parameter> {
	params.iter().map(|p| p.inner.clone()).collect()
}

// ── Sgd ──────────────────────────────────────────────────────────────────────

/// Stochastic gradient descent over a stable parameter set.
#[pyclass(name = "Sgd", unsendable)]
pub(crate) struct PythonSgd {
	pub(crate) inner: oa::ml::Sgd,
}

#[pymethods]
impl PythonSgd {
	/// Bind SGD to the given parameters.
	#[new]
	#[pyo3(signature = (parameters, learning_rate, momentum=0.0, weight_decay=0.0))]
	pub fn new(
		parameters: Vec<PyRef<'_, PythonParameter>>,
		learning_rate: f32,
		momentum: f32,
		weight_decay: f32,
	) -> PyResult<Self> {
		oa::ml::Sgd::new(
			extract_parameters(parameters),
			learning_rate,
			momentum,
			weight_decay,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Discard every accumulated gradient.
	pub fn zero_grad(&self) {
		self.inner.zero_grad();
	}

	/// Record one SGD update step.
	pub fn step(&mut self) -> PyResult<()> {
		self.inner.step().map_err(python_error)
	}

	#[getter]
	pub fn learning_rate(&self) -> f32 {
		self.inner.learning_rate()
	}

	pub fn set_learning_rate(&mut self, lr: f32) -> PyResult<()> {
		self.inner.set_learning_rate(lr).map_err(python_error)
	}

	#[getter]
	pub fn step_count(&self) -> u64 {
		self.inner.step_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Sgd(lr={}, step={})",
			self.inner.learning_rate(),
			self.inner.step_count()
		)
	}
}

// ── Adam ─────────────────────────────────────────────────────────────────────

/// Adam optimizer over a stable parameter set.
#[pyclass(name = "Adam", unsendable)]
pub(crate) struct PythonAdam {
	pub(crate) inner: oa::ml::Adam,
}

#[pymethods]
impl PythonAdam {
	/// Bind Adam with default beta1=0.9, beta2=0.999, epsilon=1e-8.
	#[new]
	#[pyo3(signature = (parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8))]
	pub fn new(
		parameters: Vec<PyRef<'_, PythonParameter>>,
		learning_rate: f32,
		beta1: f32,
		beta2: f32,
		epsilon: f32,
	) -> PyResult<Self> {
		oa::ml::Adam::with_hyperparameters(
			extract_parameters(parameters),
			learning_rate,
			beta1,
			beta2,
			epsilon,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Discard every accumulated gradient.
	pub fn zero_grad(&self) {
		self.inner.zero_grad();
	}

	/// Record one Adam update step.
	pub fn step(&mut self) -> PyResult<()> {
		self.inner.step().map_err(python_error)
	}

	#[getter]
	pub fn learning_rate(&self) -> f32 {
		self.inner.learning_rate()
	}

	pub fn set_learning_rate(&mut self, lr: f32) -> PyResult<()> {
		self.inner.set_learning_rate(lr).map_err(python_error)
	}

	#[getter]
	pub fn step_count(&self) -> u32 {
		self.inner.step_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Adam(lr={}, step={})",
			self.inner.learning_rate(),
			self.inner.step_count()
		)
	}
}

// ── AdamW ────────────────────────────────────────────────────────────────────

/// AdamW optimizer over a stable parameter set.
#[pyclass(name = "AdamW", unsendable)]
pub(crate) struct PythonAdamW {
	pub(crate) inner: oa::ml::AdamW,
}

#[pymethods]
impl PythonAdamW {
	/// Bind AdamW with default hyperparameters.
	#[new]
	#[pyo3(signature = (parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01))]
	pub fn new(
		parameters: Vec<PyRef<'_, PythonParameter>>,
		learning_rate: f32,
		beta1: f32,
		beta2: f32,
		epsilon: f32,
		weight_decay: f32,
	) -> PyResult<Self> {
		oa::ml::AdamW::with_hyperparameters(
			extract_parameters(parameters),
			learning_rate,
			beta1,
			beta2,
			epsilon,
			weight_decay,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Discard every accumulated gradient.
	pub fn zero_grad(&self) {
		self.inner.zero_grad();
	}

	/// Record one AdamW update step.
	pub fn step(&mut self) -> PyResult<()> {
		self.inner.step().map_err(python_error)
	}

	#[getter]
	pub fn learning_rate(&self) -> f32 {
		self.inner.learning_rate()
	}

	pub fn set_learning_rate(&mut self, lr: f32) -> PyResult<()> {
		self.inner.set_learning_rate(lr).map_err(python_error)
	}

	#[getter]
	pub fn step_count(&self) -> u32 {
		self.inner.step_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AdamW(lr={}, step={})",
			self.inner.learning_rate(),
			self.inner.step_count()
		)
	}
}

// ── Muon ─────────────────────────────────────────────────────────────────────

/// Muon optimizer over a stable parameter set.
#[pyclass(name = "Muon", unsendable)]
pub(crate) struct PythonMuon {
	pub(crate) inner: oa::ml::Muon,
}

#[pymethods]
impl PythonMuon {
	/// Bind Muon with default beta=0.95, weight_decay=0.1, epsilon=1e-7, ns5_iterations=5.
	#[new]
	#[pyo3(signature = (parameters, learning_rate, beta=0.95, weight_decay=0.1, epsilon=1e-7, ns5_iterations=5))]
	pub fn new(
		parameters: Vec<PyRef<'_, PythonParameter>>,
		learning_rate: f32,
		beta: f32,
		weight_decay: f32,
		epsilon: f32,
		ns5_iterations: u32,
	) -> PyResult<Self> {
		oa::ml::Muon::with_hyperparameters(
			extract_parameters(parameters),
			learning_rate,
			beta,
			weight_decay,
			epsilon,
			ns5_iterations,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Discard every accumulated gradient.
	pub fn zero_grad(&self) {
		self.inner.zero_grad();
	}

	/// Record one Muon update step.
	pub fn step(&mut self) -> PyResult<()> {
		self.inner.step().map_err(python_error)
	}

	#[getter]
	pub fn learning_rate(&self) -> f32 {
		self.inner.learning_rate()
	}

	pub fn set_learning_rate(&mut self, lr: f32) -> PyResult<()> {
		self.inner.set_learning_rate(lr).map_err(python_error)
	}

	#[getter]
	pub fn step_count(&self) -> u64 {
		self.inner.step_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Muon(lr={}, step={})",
			self.inner.learning_rate(),
			self.inner.step_count()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonSgd>()?;
	module.add_class::<PythonAdam>()?;
	module.add_class::<PythonAdamW>()?;
	module.add_class::<PythonMuon>()?;
	// clip_grad_norm is registered by autograd::register (accepts Matrix or Parameter lists).
	Ok(())
}
