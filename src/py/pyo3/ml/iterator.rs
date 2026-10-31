//! `ItTraining` stepwise training driver binding.
//!
//! `ItTraining<'engine, 'hooks>` borrows the engine, optimizer, metrics, and
//! callbacks. PyO3 cannot express those borrows directly, so this module uses
//! the same lifetime adapter as `trainer.rs`: the returned pyclass stores the
//! `Py` handles it borrows from, and hook installs either borrow the `inner`
//! Rust object of a native wrapper pyclass or box a `Py` adapter that
//! re-enters Python when a lifecycle hook fires.

use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::{
	optim::PythonAdamW,
	trainer::{box_callback, box_metric},
	trainer_data::{
		PythonCsvLogger, PythonEarlyStopping, PythonItTrainingConfig, PythonLearningRateScheduler,
		PythonLossMetric, PythonProgressBar, PythonTrainingSnapshot, PythonTrainingSummary,
		PythonValidation,
	},
	training_program::PythonExecutionPlanDiagnostics,
};

/// Exact eager or captured training-step lifecycle.
///
/// `ItTraining::new` requires an `AdamW` optimizer — whole-program capture is
/// implemented for AdamW only; other optimizers need the manual
/// `begin_step`/`complete_step` surface that is not yet bound.
///
/// The engine and optimizer must outlive this object — keep them as locals or
/// attributes so Python reference counting keeps them alive.
#[pyclass(name = "ItTraining", unsendable)]
pub(crate) struct PythonItTraining {
	_engine: Py<PythonEngine>,
	_optimizer: Py<PyAny>,
	// Native wrapper objects whose `inner` fields `inner` borrows mutably.
	_keep_alive: Vec<Py<PyAny>>,
	// Boxed Python adapters. `ItTraining::drop` only closes the stable-resource
	// frame — it never dereferences hooks — so the `'static` borrows below are
	// dead by field-drop time and never observed.
	_callbacks: Vec<Box<dyn oa::ml::TrainingCallback>>,
	_metrics: Vec<Box<dyn oa::ml::TrainingMetric>>,
	// The trainer holds 'static borrows into the owned fields above.
	inner: Option<oa::ml::ItTraining<'static, 'static>>,
	config: oa::ml::ItTrainingConfig,
}

/// Return a mutable borrow of the wrapped Rust callback for the six native
/// callback pyclasses, or `None` when `object` is an arbitrary Python object.
fn native_callback(object: &Bound<'_, PyAny>) -> Option<*mut dyn oa::ml::TrainingCallback> {
	macro_rules! borrow_inner {
		($($ty:ty),+ $(,)?) => {
			$(
				if let Ok(mut wrapped) = object.extract::<PyRefMut<$ty>>() {
					return Some(&mut wrapped.inner as *mut dyn oa::ml::TrainingCallback);
				}
			)+
		};
	}
	borrow_inner!(
		PythonProgressBar,
		PythonTrainingSummary,
		PythonCsvLogger,
		PythonEarlyStopping,
		PythonValidation,
		PythonLearningRateScheduler,
	);
	None
}

fn call_prepare(prepare: &Py<PyAny>) -> oa::Result<Py<PyAny>> {
	Python::attach(|py| {
		prepare
			.bind(py)
			.call0()
			.map(|result| result.unbind())
			.map_err(|e| oa::Error::callback(format!("training prepare raised: {e}")))
	})
}

fn call_record(record: &Py<PyAny>, prepared: Py<PyAny>) -> oa::Result<oa::Matrix> {
	Python::attach(|py| {
		let result = record
			.bind(py)
			.call1((prepared,))
			.map_err(|e| oa::Error::callback(format!("training record raised: {e}")))?;
		let matrix = result.extract::<PyRef<PythonMatrix>>().map_err(|e| {
			oa::Error::callback(format!("training record must return a Matrix loss: {e}"))
		})?;
		Ok(matrix.inner.clone())
	})
}

#[pymethods]
impl PythonItTraining {
	/// Construct a training lifecycle around one engine and AdamW optimizer.
	#[new]
	pub fn new(
		py: Python<'_>,
		engine: Py<PythonEngine>,
		optimizer: Py<PyAny>,
		config: &PythonItTrainingConfig,
	) -> PyResult<Self> {
		let cfg = config.inner.clone();
		let inner = {
			let eng = engine.bind(py).borrow();
			// SAFETY: `engine` and `optimizer` are stored in the returned struct,
			// so the pointees outlive `inner`.
			let engine_ptr: *const oa::Engine = &eng.inner;
			let mut opt = optimizer
				.bind(py)
				.extract::<PyRefMut<PythonAdamW>>()
				.map_err(|_| {
					pyo3::exceptions::PyTypeError::new_err("ItTraining requires an AdamW optimizer")
				})?;
			let opt_ptr: *mut oa::ml::AdamW = &mut opt.inner;
			unsafe {
				std::mem::transmute::<
					oa::Result<oa::ml::ItTraining<'_, '_>>,
					oa::Result<oa::ml::ItTraining<'static, 'static>>,
				>(oa::ml::ItTraining::new(
					&*engine_ptr,
					&mut *opt_ptr,
					cfg.clone(),
				))
			}
			.map_err(python_error)?
		};
		Ok(Self {
			_engine: engine,
			_optimizer: optimizer,
			_keep_alive: Vec::new(),
			_callbacks: Vec::new(),
			_metrics: Vec::new(),
			inner: Some(inner),
			config: cfg,
		})
	}

	/// Install a training callback. Accepts the native `ProgressBar`,
	/// `TrainingSummary`, `CsvLogger`, `EarlyStopping`, `Validation`, and
	/// `LearningRateScheduler` objects directly; any other Python object is
	/// adapted through its `on_*` hook methods (return `True` to stop).
	pub fn add_callback(&mut self, callback: &Bound<'_, PyAny>) -> PyResult<()> {
		if let Some(ptr) = native_callback(callback) {
			// SAFETY: `callback` is kept alive in `_keep_alive` for as long as
			// `inner` may dereference the borrow; the pyclass is single-threaded.
			let reference = unsafe { &mut *ptr };
			self.training_mut()?.add_callback(reference);
			self._keep_alive.push(callback.clone().unbind());
			return Ok(());
		}
		let mut boxed = box_callback(callback);
		let reference: &'static mut dyn oa::ml::TrainingCallback =
			unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingCallback) };
		self.training_mut()?.add_callback(reference);
		self._callbacks.push(boxed);
		Ok(())
	}

	/// Install a training metric. Accepts the native `LossMetric` object
	/// directly; any other Python object is adapted through `name`, `reset`,
	/// `update_step`, and `result`.
	pub fn add_metric(&mut self, metric: &Bound<'_, PyAny>) -> PyResult<()> {
		if let Ok(mut wrapped) = metric.extract::<PyRefMut<PythonLossMetric>>() {
			let reference: &'static mut dyn oa::ml::TrainingMetric =
				unsafe { &mut *(&mut wrapped.inner as *mut dyn oa::ml::TrainingMetric) };
			self.training_mut()?.add_metric(reference);
			self._keep_alive.push(metric.clone().unbind());
			return Ok(());
		}
		let mut boxed = box_metric(metric);
		let reference: &'static mut dyn oa::ml::TrainingMetric =
			unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingMetric) };
		self.training_mut()?.add_metric(reference);
		self._metrics.push(boxed);
		Ok(())
	}

	/// Run one automatic training step: `prepare` returns the step's inputs
	/// (e.g. a `(input, target)` Matrix pair) and `record` maps them to the
	/// scalar `Matrix` loss; backward runs inside `record`. Returns `False`
	/// when the step budget is already exhausted.
	pub fn step(&mut self, prepare: Py<PyAny>, record: Py<PyAny>) -> PyResult<bool> {
		self
			.training_mut()?
			.step(
				move || call_prepare(&prepare),
				move |prepared| call_record(&record, prepared),
			)
			.map_err(python_error)
	}

	/// Record non-finite-exempt source-unit accounting for the next step.
	pub fn record_source_units(&mut self, units: u64) -> PyResult<()> {
		self.training_mut()?.record_source_units(units);
		Ok(())
	}

	/// Request the loop to stop at the next lifecycle boundary.
	pub fn request_stop(&mut self) -> PyResult<()> {
		self.training_mut()?.request_stop();
		Ok(())
	}

	/// Return the immutable state snapshot.
	pub fn snapshot(&self) -> PyResult<PythonTrainingSnapshot> {
		Ok(PythonTrainingSnapshot::from_inner(
			self.training()?.snapshot(),
		))
	}

	/// Return the captured program's execution-plan diagnostics, or `None`
	/// when the automatic step contract did not produce a captured program.
	pub fn program_diagnostics(&self) -> Option<PythonExecutionPlanDiagnostics> {
		self
			.inner
			.as_ref()?
			.training_program()
			.map(|program| PythonExecutionPlanDiagnostics {
				inner: program.diagnostics(),
			})
	}

	/// Return how many whole-program capture rejections continued eagerly.
	pub fn program_capture_fallback_count(&self) -> u64 {
		self
			.inner
			.as_ref()
			.map(|training| training.program_capture_fallback_count())
			.unwrap_or_default()
	}

	/// Consume the trainer and report the terminal snapshot.
	pub fn finish(&mut self) -> PyResult<PythonTrainingSnapshot> {
		let inner = self
			.inner
			.take()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("training already finished"))?;
		inner
			.finish()
			.map(PythonTrainingSnapshot::from_inner)
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ItTraining(total_steps={}, batch_size={})",
			self.config.total_steps, self.config.batch_size,
		)
	}
}

impl PythonItTraining {
	fn training(&self) -> PyResult<&oa::ml::ItTraining<'static, 'static>> {
		self
			.inner
			.as_ref()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("training already finished"))
	}

	fn training_mut(&mut self) -> PyResult<&mut oa::ml::ItTraining<'static, 'static>> {
		self
			.inner
			.as_mut()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("training already finished"))
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonItTraining>()?;
	Ok(())
}
