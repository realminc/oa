use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

// ── dtype helper (local copy to avoid cross-module dep on environment.rs) ─────

fn parse_dtype(token: &str) -> PyResult<oa::DType> {
	match token {
		"f32" => Ok(oa::DType::F32),
		"i32" => Ok(oa::DType::I32),
		"u8" => Ok(oa::DType::U8),
		"u32" => Ok(oa::DType::U32),
		_ => Err(pyo3::exceptions::PyValueError::new_err(format!(
			"unknown dtype {:?}; expected f32, i32, u8, or u32",
			token
		))),
	}
}

// ── ReplayConfig ──────────────────────────────────────────────────────────────

/// Shape, capacity, and action representation of one replay buffer.
#[pyclass(name = "ReplayConfig")]
#[derive(Clone)]
pub(crate) struct PythonReplayConfig {
	pub(crate) inner: oa::ml::ReplayConfig,
}

#[pymethods]
impl PythonReplayConfig {
	#[new]
	#[pyo3(signature = (capacity, observation_shape, action_shape = None, action_dtype = "f32"))]
	pub fn new(
		capacity: usize,
		observation_shape: Vec<usize>,
		action_shape: Option<Vec<usize>>,
		action_dtype: &str,
	) -> PyResult<Self> {
		let action_dtype = parse_dtype(action_dtype)?;
		Ok(Self {
			inner: oa::ml::ReplayConfig {
				capacity,
				observation_shape,
				action_shape: action_shape.unwrap_or_default(),
				action_dtype,
			},
		})
	}

	#[getter]
	pub fn capacity(&self) -> usize {
		self.inner.capacity
	}

	#[getter]
	pub fn observation_shape(&self) -> Vec<usize> {
		self.inner.observation_shape.clone()
	}

	#[getter]
	pub fn action_shape(&self) -> Vec<usize> {
		self.inner.action_shape.clone()
	}

	#[getter]
	pub fn action_dtype(&self) -> &'static str {
		self.inner.action_dtype.token()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ReplayConfig(capacity={}, observation_shape={:?}, action_dtype={})",
			self.inner.capacity,
			self.inner.observation_shape,
			self.inner.action_dtype.token(),
		)
	}
}

// ── ReplayTransition ──────────────────────────────────────────────────────────

/// One batch of transitions to append to a [`ReplayBuffer`].
#[pyclass(name = "ReplayTransition", unsendable)]
pub(crate) struct PythonReplayTransition {
	pub(crate) inner: oa::ml::ReplayTransition,
}

#[pymethods]
impl PythonReplayTransition {
	#[new]
	pub fn new(
		observation: &PythonMatrix,
		action: &PythonMatrix,
		next_observation: &PythonMatrix,
		reward: &PythonMatrix,
		terminated: &PythonMatrix,
		truncated: &PythonMatrix,
	) -> Self {
		Self {
			inner: oa::ml::ReplayTransition::new(
				observation.inner.clone(),
				action.inner.clone(),
				next_observation.inner.clone(),
				reward.inner.clone(),
				terminated.inner.clone(),
				truncated.inner.clone(),
			),
		}
	}

	#[getter]
	pub fn observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.observation().clone())
	}

	#[getter]
	pub fn action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.action().clone())
	}

	#[getter]
	pub fn next_observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.next_observation().clone())
	}

	#[getter]
	pub fn reward(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.reward().clone())
	}

	#[getter]
	pub fn terminated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.terminated().clone())
	}

	#[getter]
	pub fn truncated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.truncated().clone())
	}

	pub fn __repr__(&self) -> &str {
		"ReplayTransition"
	}
}

// ── ReplayBatch ───────────────────────────────────────────────────────────────

/// Device-resident replay values returned by [`ReplayBuffer::sample`].
#[pyclass(name = "ReplayBatch", unsendable)]
pub(crate) struct PythonReplayBatch {
	pub(crate) inner: oa::ml::ReplayBatch,
}

#[pymethods]
impl PythonReplayBatch {
	#[getter]
	pub fn observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.observation().clone())
	}

	#[getter]
	pub fn action(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.action().clone())
	}

	#[getter]
	pub fn next_observation(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.next_observation().clone())
	}

	#[getter]
	pub fn reward(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.reward().clone())
	}

	#[getter]
	pub fn terminated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.terminated().clone())
	}

	#[getter]
	pub fn truncated(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.truncated().clone())
	}

	#[getter]
	pub fn index(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.index().clone())
	}

	pub fn __repr__(&self) -> &str {
		"ReplayBatch"
	}
}

// ── ReplayBuffer ──────────────────────────────────────────────────────────────

/// Circular fixed-capacity owner for off-policy transitions.
#[pyclass(name = "ReplayBuffer", unsendable)]
pub(crate) struct PythonReplayBuffer {
	pub(crate) inner: oa::ml::ReplayBuffer,
}

#[pymethods]
impl PythonReplayBuffer {
	/// Allocate empty circular replay storage on `engine`.
	#[new]
	pub fn new(engine: &PythonEngine, config: &PythonReplayConfig) -> PyResult<Self> {
		oa::ml::ReplayBuffer::new(&engine.inner, config.inner.clone())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Record one batched circular append.
	pub fn append(&mut self, transition: &PythonReplayTransition) -> PyResult<()> {
		self.inner.append(&transition.inner).map_err(python_error)
	}

	/// Record deterministic uniform sampling with replacement.
	pub fn sample(&self, batch_size: usize, seed: u64) -> PyResult<PythonReplayBatch> {
		self
			.inner
			.sample(batch_size, seed)
			.map(|inner| PythonReplayBatch { inner })
			.map_err(python_error)
	}

	/// Forget all retained transitions without recording device work.
	pub fn reset(&mut self) {
		self.inner.reset();
	}

	pub fn is_full(&self) -> bool {
		self.inner.is_full()
	}

	pub fn len(&self) -> usize {
		self.inner.len()
	}

	pub fn is_empty(&self) -> bool {
		self.inner.is_empty()
	}

	pub fn capacity(&self) -> usize {
		self.inner.capacity()
	}

	pub fn cursor(&self) -> usize {
		self.inner.cursor()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ReplayBuffer(len={}, capacity={}, cursor={})",
			self.inner.len(),
			self.inner.capacity(),
			self.inner.cursor(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonReplayConfig>()?;
	module.add_class::<PythonReplayTransition>()?;
	module.add_class::<PythonReplayBatch>()?;
	module.add_class::<PythonReplayBuffer>()?;
	Ok(())
}
