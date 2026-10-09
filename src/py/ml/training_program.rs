//! Captured, replayable training-program binding.

use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::optim::PythonAdamW;

// ── TrainingCompilationStage ───────────────────────────────────────────────────

/// Ordered compilation stage for a captured training program.
#[pyclass(name = "TrainingCompilationStage", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonTrainingCompilationStage {
	SemanticValidation = 0,
	ReplaySafety = 1,
	Decomposition = 2,
	Fusion = 3,
	Placement = 4,
	Precision = 5,
	KernelSelection = 6,
	LoweringValidation = 7,
	MemoryPlanning = 8,
	SynchronizationPlanning = 9,
	CommandRecording = 10,
}

impl From<oa::ml::TrainingCompilationStage> for PythonTrainingCompilationStage {
	fn from(value: oa::ml::TrainingCompilationStage) -> Self {
		use oa::ml::TrainingCompilationStage as S;
		match value {
			S::SemanticValidation => Self::SemanticValidation,
			S::ReplaySafety => Self::ReplaySafety,
			S::Decomposition => Self::Decomposition,
			S::Fusion => Self::Fusion,
			S::Placement => Self::Placement,
			S::Precision => Self::Precision,
			S::KernelSelection => Self::KernelSelection,
			S::LoweringValidation => Self::LoweringValidation,
			S::MemoryPlanning => Self::MemoryPlanning,
			S::SynchronizationPlanning => Self::SynchronizationPlanning,
			S::CommandRecording => Self::CommandRecording,
		}
	}
}

#[pymethods]
impl PythonTrainingCompilationStage {
	/// Return the stable report token for this stage.
	pub fn token(&self) -> &'static str {
		let inner: oa::ml::TrainingCompilationStage = (*self).into();
		inner.token()
	}
}

impl From<PythonTrainingCompilationStage> for oa::ml::TrainingCompilationStage {
	fn from(value: PythonTrainingCompilationStage) -> Self {
		use PythonTrainingCompilationStage as P;
		match value {
			P::SemanticValidation => Self::SemanticValidation,
			P::ReplaySafety => Self::ReplaySafety,
			P::Decomposition => Self::Decomposition,
			P::Fusion => Self::Fusion,
			P::Placement => Self::Placement,
			P::Precision => Self::Precision,
			P::KernelSelection => Self::KernelSelection,
			P::LoweringValidation => Self::LoweringValidation,
			P::MemoryPlanning => Self::MemoryPlanning,
			P::SynchronizationPlanning => Self::SynchronizationPlanning,
			P::CommandRecording => Self::CommandRecording,
		}
	}
}

// ── TrainingCompilationState ───────────────────────────────────────────────────

/// Outcome of one training-program compilation stage.
#[pyclass(name = "TrainingCompilationState", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonTrainingCompilationState {
	NotRun = 0,
	Inherited = 1,
	Analyzed = 2,
	Applied = 3,
	Failed = 4,
}

impl From<oa::ml::TrainingCompilationState> for PythonTrainingCompilationState {
	fn from(value: oa::ml::TrainingCompilationState) -> Self {
		use oa::ml::TrainingCompilationState as S;
		match value {
			S::NotRun => Self::NotRun,
			S::Inherited => Self::Inherited,
			S::Analyzed => Self::Analyzed,
			S::Applied => Self::Applied,
			S::Failed => Self::Failed,
		}
	}
}

#[pymethods]
impl PythonTrainingCompilationState {
	/// Return the stable report token for this state.
	pub fn token(&self) -> &'static str {
		let inner: oa::ml::TrainingCompilationState = (*self).into();
		inner.token()
	}
}

impl From<PythonTrainingCompilationState> for oa::ml::TrainingCompilationState {
	fn from(value: PythonTrainingCompilationState) -> Self {
		use PythonTrainingCompilationState as P;
		match value {
			P::NotRun => Self::NotRun,
			P::Inherited => Self::Inherited,
			P::Analyzed => Self::Analyzed,
			P::Applied => Self::Applied,
			P::Failed => Self::Failed,
		}
	}
}

// ── TrainingCompilationStageRecord ─────────────────────────────────────────────

/// Immutable evidence for one training-program compilation stage.
#[pyclass(name = "TrainingCompilationStageRecord")]
#[derive(Clone, Copy)]
pub(crate) struct PythonTrainingCompilationStageRecord {
	inner: oa::ml::TrainingCompilationStageRecord,
}

#[pymethods]
impl PythonTrainingCompilationStageRecord {
	#[getter]
	pub fn stage(&self) -> PythonTrainingCompilationStage {
		self.inner.stage().into()
	}

	#[getter]
	pub fn state(&self) -> PythonTrainingCompilationState {
		self.inner.state().into()
	}

	#[getter]
	pub fn input_count(&self) -> usize {
		self.inner.input_count()
	}

	#[getter]
	pub fn output_count(&self) -> usize {
		self.inner.output_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingCompilationStageRecord(stage={:?}, state={:?}, in={}, out={})",
			self.inner.stage().token(),
			self.inner.state().token(),
			self.inner.input_count(),
			self.inner.output_count(),
		)
	}
}

// ── TrainingProgram ────────────────────────────────────────────────────────────

/// Fixed-shape forward, backward, and AdamW program captured for Vulkan replay.
///
/// Capture is triggered via `TrainingProgram.capture(engine, optimizer, step_fn)`.
/// `step_fn` must be a Python callable that builds and returns a scalar `Matrix`
/// loss (autograd backward is run inside the call). The optimizer must be `AdamW`.
///
/// The engine and optimizer must outlive this object — keep them as locals or
/// attributes so Python reference counting keeps them alive.
#[pyclass(name = "TrainingProgram", unsendable)]
pub(crate) struct PythonTrainingProgram {
	inner: oa::ml::TrainingProgram,
}

#[pymethods]
impl PythonTrainingProgram {
	/// Capture one complete forward/backward/AdamW step.
	///
	/// `step_fn` is a Python callable with no arguments that must build and
	/// return a scalar `Matrix` loss; its backward pass is recorded
	/// automatically inside the capture. The optimizer must be `AdamW`.
	#[staticmethod]
	pub fn capture(
		py: Python<'_>,
		engine: &PythonEngine,
		optimizer: &mut PythonAdamW,
		step_fn: Py<PyAny>,
	) -> PyResult<Self> {
		let engine_ref: &oa::Engine = &engine.inner;
		let inner = oa::ml::TrainingProgram::capture(engine_ref, &mut optimizer.inner, || {
			Python::attach(|py2| {
				let result = step_fn
					.bind(py2)
					.call0()
					.map_err(|e| oa::Error::callback(format!("step_fn raised: {e}")))?;
				let matrix = result
					.extract::<PyRef<PythonMatrix>>()
					.map_err(|e| oa::Error::callback(format!("step_fn must return a Matrix loss: {e}")))?;
				Ok(matrix.inner.clone())
			})
		})
		.map_err(python_error)?;
		let _ = py;
		Ok(Self { inner })
	}

	/// Upload a new host batch of `f32` values into a captured input slot.
	pub fn upload_input_f32(&mut self, captured: &PythonMatrix, values: Vec<f32>) -> PyResult<()> {
		self
			.inner
			.upload_input(&captured.inner, &values)
			.map_err(python_error)
	}

	/// Upload a new host batch of `u32` values into a captured input slot.
	pub fn upload_input_u32(&mut self, captured: &PythonMatrix, values: Vec<u32>) -> PyResult<()> {
		self
			.inner
			.upload_input(&captured.inner, &values)
			.map_err(python_error)
	}

	/// Submit one replay without waiting. Returns an `Event`.
	pub fn replay(
		&mut self,
		engine: &PythonEngine,
		optimizer: &mut PythonAdamW,
	) -> PyResult<crate::runtime::PythonEvent> {
		self
			.inner
			.replay(&engine.inner, &mut optimizer.inner)
			.map(|e| crate::runtime::PythonEvent { inner: e })
			.map_err(python_error)
	}

	/// Submit one timed replay without waiting. Returns an `Event`.
	pub fn replay_timed(
		&mut self,
		engine: &PythonEngine,
		optimizer: &mut PythonAdamW,
	) -> PyResult<crate::runtime::PythonEvent> {
		self
			.inner
			.replay_timed(&engine.inner, &mut optimizer.inner)
			.map(|e| crate::runtime::PythonEvent { inner: e })
			.map_err(python_error)
	}

	/// Submit one replay, wait for completion, and return the scalar loss.
	pub fn replay_and_wait(
		&mut self,
		engine: &PythonEngine,
		optimizer: &mut PythonAdamW,
	) -> PyResult<f32> {
		self
			.inner
			.replay_and_wait(&engine.inner, &mut optimizer.inner)
			.map_err(python_error)
	}

	/// Submit one timed replay, wait, and return `(loss, device_secs)`.
	pub fn replay_timed_and_wait(
		&mut self,
		engine: &PythonEngine,
		optimizer: &mut PythonAdamW,
	) -> PyResult<(f32, f64)> {
		self
			.inner
			.replay_timed_and_wait(&engine.inner, &mut optimizer.inner)
			.map(|(loss, dur)| (loss, dur.as_secs_f64()))
			.map_err(python_error)
	}

	/// Return the captured scalar loss `Matrix`.
	pub fn loss(&self) -> PythonMatrix {
		PythonMatrix {
			inner: self.inner.loss().clone(),
		}
	}

	/// Return how many training steps this program has submitted.
	pub fn replay_count(&self) -> u32 {
		self.inner.replay_count()
	}

	/// Return whether the most recently submitted replay has completed.
	pub fn is_complete(&self) -> PyResult<bool> {
		self.inner.is_complete().map_err(python_error)
	}

	/// Wait for the most recently submitted replay.
	pub fn wait(&self) -> PyResult<()> {
		self.inner.wait().map_err(python_error)
	}

	/// Return ordered stage evidence from the capture transaction.
	pub fn compilation_stages(&self) -> Vec<PythonTrainingCompilationStageRecord> {
		self
			.inner
			.compilation_stages()
			.iter()
			.map(|r| PythonTrainingCompilationStageRecord { inner: *r })
			.collect()
	}

	pub fn __repr__(&self) -> String {
		format!("TrainingProgram(replays={})", self.inner.replay_count())
	}
}

// ── register ──────────────────────────────────────────────────────────────────

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonTrainingCompilationStage>()?;
	module.add_class::<PythonTrainingCompilationState>()?;
	module.add_class::<PythonTrainingCompilationStageRecord>()?;
	module.add_class::<PythonTrainingProgram>()?;
	Ok(())
}
