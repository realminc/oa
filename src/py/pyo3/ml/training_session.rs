use pyo3::prelude::*;

use crate::error::python_error;

// ── TrainingState ─────────────────────────────────────────────────────────────

/// Live state of an attached training lifecycle.
#[pyclass(name = "TrainingState", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonTrainingState {
	Running = 0,
	Paused = 1,
	Stopping = 2,
	Completed = 3,
	Failed = 4,
}

impl From<oa::ml::TrainingState> for PythonTrainingState {
	fn from(value: oa::ml::TrainingState) -> Self {
		match value {
			oa::ml::TrainingState::Running => PythonTrainingState::Running,
			oa::ml::TrainingState::Paused => PythonTrainingState::Paused,
			oa::ml::TrainingState::Stopping => PythonTrainingState::Stopping,
			oa::ml::TrainingState::Completed => PythonTrainingState::Completed,
			oa::ml::TrainingState::Failed => PythonTrainingState::Failed,
		}
	}
}

// ── TrainingSessionSnapshot ───────────────────────────────────────────────────

/// Point-in-time evidence published by one training step.
#[pyclass(name = "TrainingSessionSnapshot")]
#[derive(Clone)]
pub(crate) struct PythonTrainingSessionSnapshot {
	inner: oa::ml::TrainingSessionSnapshot,
}

#[pymethods]
impl PythonTrainingSessionSnapshot {
	#[getter]
	pub fn revision(&self) -> u64 {
		self.inner.revision
	}

	#[getter]
	pub fn state(&self) -> PythonTrainingState {
		self.inner.state.into()
	}

	#[getter]
	pub fn step(&self) -> u64 {
		self.inner.step
	}

	#[getter]
	pub fn epoch(&self) -> u64 {
		self.inner.epoch
	}

	#[getter]
	pub fn learning_rate(&self) -> f32 {
		self.inner.learning_rate
	}

	#[getter]
	pub fn loss(&self) -> f32 {
		self.inner.loss
	}

	#[getter]
	pub fn gpu_ms(&self) -> f64 {
		self.inner.gpu_ms
	}

	#[getter]
	pub fn wall_ms(&self) -> f64 {
		self.inner.wall_ms
	}

	/// Additional named scalar metrics as a list of (name, value) pairs.
	pub fn metrics(&self) -> Vec<(String, f64)> {
		self
			.inner
			.metrics
			.iter()
			.map(|m| (m.name.clone(), m.value))
			.collect()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingSessionSnapshot(step={}, epoch={}, loss={:.4}, lr={:.2e})",
			self.inner.step, self.inner.epoch, self.inner.loss, self.inner.learning_rate,
		)
	}
}

// ── TrainingSessionConfig ─────────────────────────────────────────────────────

/// Configuration for one training session handle.
#[pyclass(name = "TrainingSessionConfig")]
#[derive(Clone)]
pub(crate) struct PythonTrainingSessionConfig {
	pub(crate) inner: oa::ml::TrainingSessionConfig,
}

#[pymethods]
impl PythonTrainingSessionConfig {
	#[new]
	#[pyo3(signature = (command_capacity = 64, result_capacity = 128, snapshot_capacity = 32))]
	pub fn new(command_capacity: usize, result_capacity: usize, snapshot_capacity: usize) -> Self {
		Self {
			inner: oa::ml::TrainingSessionConfig {
				command_capacity,
				result_capacity,
				snapshot_capacity,
			},
		}
	}

	#[getter]
	pub fn command_capacity(&self) -> usize {
		self.inner.command_capacity
	}

	#[getter]
	pub fn result_capacity(&self) -> usize {
		self.inner.result_capacity
	}

	#[getter]
	pub fn snapshot_capacity(&self) -> usize {
		self.inner.snapshot_capacity
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingSessionConfig(cmd={}, result={}, snapshot={})",
			self.inner.command_capacity, self.inner.result_capacity, self.inner.snapshot_capacity,
		)
	}
}

// ── TrainingSession ───────────────────────────────────────────────────────────

/// Cloneable, thread-safe command and observation handle for one training loop.
#[pyclass(name = "TrainingSession")]
#[derive(Clone)]
pub(crate) struct PythonTrainingSession {
	pub(crate) inner: oa::ml::TrainingSession,
}

#[pymethods]
impl PythonTrainingSession {
	/// Construct an unattached live-control handle.
	#[new]
	#[pyo3(signature = (config = None))]
	pub fn new(config: Option<&PythonTrainingSessionConfig>) -> Self {
		let cfg = config.map(|c| c.inner).unwrap_or_default();
		Self {
			inner: oa::ml::TrainingSession::new(cfg),
		}
	}

	/// Enqueue a pause request.
	pub fn pause(&self, expected_revision: u64) -> PyResult<u64> {
		self.inner.pause(expected_revision).map_err(python_error)
	}

	/// Enqueue a resume request.
	pub fn resume(&self, expected_revision: u64) -> PyResult<u64> {
		self.inner.resume(expected_revision).map_err(python_error)
	}

	/// Enqueue a stop request.
	pub fn stop(&self, expected_revision: u64) -> PyResult<u64> {
		self.inner.stop(expected_revision).map_err(python_error)
	}

	/// Enqueue a checkpoint request.
	pub fn checkpoint(&self, expected_revision: u64) -> PyResult<u64> {
		self
			.inner
			.checkpoint(expected_revision)
			.map_err(python_error)
	}

	/// Enqueue an evaluation request.
	pub fn evaluate(&self, expected_revision: u64) -> PyResult<u64> {
		self.inner.evaluate(expected_revision).map_err(python_error)
	}

	/// Enqueue a parameter recapture request.
	pub fn request_recapture(&self, expected_revision: u64) -> PyResult<u64> {
		self
			.inner
			.request_recapture(expected_revision)
			.map_err(python_error)
	}

	/// Publish or replace a named finite scalar for subsequent snapshots.
	pub fn publish_metric(&self, name: &str, value: f64) {
		self.inner.publish_metric(name, value);
	}

	/// Return the current live state.
	pub fn state(&self) -> PythonTrainingState {
		self.inner.state().into()
	}

	/// Return the current optimistic-concurrency revision.
	pub fn revision(&self) -> u64 {
		self.inner.revision()
	}

	/// Return the combined current state snapshot.
	pub fn current_snapshot(&self) -> PythonTrainingSessionSnapshot {
		PythonTrainingSessionSnapshot {
			inner: self.inner.current_snapshot(),
		}
	}

	/// Return the latest published lifecycle snapshot, or None.
	pub fn latest_snapshot(&self) -> Option<PythonTrainingSessionSnapshot> {
		self
			.inner
			.latest_snapshot()
			.map(|inner| PythonTrainingSessionSnapshot { inner })
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingSession(state={:?}, revision={})",
			self.inner.state(),
			self.inner.revision(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonTrainingState>()?;
	module.add_class::<PythonTrainingSessionSnapshot>()?;
	module.add_class::<PythonTrainingSessionConfig>()?;
	module.add_class::<PythonTrainingSession>()?;
	Ok(())
}
