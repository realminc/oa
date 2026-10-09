use pyo3::prelude::*;

use crate::{error::python_error, runtime::PythonEngine};

/// Filesystem and metric policy for checkpointing.
#[pyclass(name = "CheckpointManagerConfig")]
#[derive(Clone)]
pub(crate) struct PythonCheckpointManagerConfig {
	pub(crate) inner: oa::ml::CheckpointManagerConfig,
}

#[pymethods]
impl PythonCheckpointManagerConfig {
	/// Construct with the OA defaults.
	#[new]
	#[pyo3(
		signature = (
			directory = "var/model/dev",
			model_name = "Module",
			context = "",
			max_keep = 5,
			save_best = true,
			metric_name = "loss",
			lower_is_better = true,
		)
	)]
	pub fn new(
		directory: &str,
		model_name: &str,
		context: &str,
		max_keep: usize,
		save_best: bool,
		metric_name: &str,
		lower_is_better: bool,
	) -> Self {
		Self {
			inner: oa::ml::CheckpointManagerConfig {
				directory: directory.into(),
				model_name: model_name.to_owned(),
				context: context.to_owned(),
				max_keep,
				save_best,
				metric_name: metric_name.to_owned(),
				lower_is_better,
			},
		}
	}

	#[getter]
	pub fn directory(&self) -> String {
		self.inner.directory.to_string_lossy().into_owned()
	}

	#[getter]
	pub fn model_name(&self) -> &str {
		&self.inner.model_name
	}

	#[getter]
	pub fn context(&self) -> &str {
		&self.inner.context
	}

	#[getter]
	pub fn max_keep(&self) -> usize {
		self.inner.max_keep
	}

	#[getter]
	pub fn save_best(&self) -> bool {
		self.inner.save_best
	}

	#[getter]
	pub fn metric_name(&self) -> &str {
		&self.inner.metric_name
	}

	#[getter]
	pub fn lower_is_better(&self) -> bool {
		self.inner.lower_is_better
	}

	pub fn __repr__(&self) -> String {
		format!(
			"CheckpointManagerConfig(model={:?}, metric={:?}, max_keep={})",
			self.inner.model_name, self.inner.metric_name, self.inner.max_keep,
		)
	}
}

// ── CheckpointManager ─────────────────────────────────────────────────────────

/// Stateful checkpoint selection and bounded incremental-retention policy.
///
/// The engine is kept alive via a `Py<PythonEngine>` handle; an internal raw
/// pointer satisfies `CheckpointManager<'engine>`.  The engine object must not
/// be dropped while this manager is alive — normal Python reference counting
/// ensures this when you keep it as a local or attribute.
#[pyclass(name = "CheckpointManager", unsendable)]
pub(crate) struct PythonCheckpointManager {
	// Keep the engine allocation alive.
	_engine: Py<PythonEngine>,
	// 'static is sound because _engine pins the allocation.
	inner: oa::ml::CheckpointManager<'static>,
}

#[pymethods]
impl PythonCheckpointManager {
	/// Construct and validate the manager.
	#[new]
	pub fn new(
		py: Python<'_>,
		engine: Py<PythonEngine>,
		config: &PythonCheckpointManagerConfig,
	) -> PyResult<Self> {
		let cfg = config.inner.clone();
		let eng = engine.bind(py).borrow();
		// SAFETY: _engine keeps the allocation alive for the lifetime of this struct.
		let engine_ref: &'static oa::Engine = unsafe { &*(&eng.inner as *const oa::Engine) };
		let inner = oa::ml::CheckpointManager::new(engine_ref, cfg).map_err(python_error)?;
		drop(eng);
		Ok(Self {
			_engine: engine,
			inner,
		})
	}

	/// Return the directory containing master and incremental checkpoints.
	pub fn model_directory(&self) -> String {
		self.inner.model_directory().to_string_lossy().into_owned()
	}

	/// Return the incremental checkpoint directory.
	pub fn incremental_directory(&self) -> String {
		self
			.inner
			.incremental_directory()
			.to_string_lossy()
			.into_owned()
	}

	/// Return the master/best checkpoint path.
	pub fn master_path(&self) -> String {
		self.inner.master_path().to_string_lossy().into_owned()
	}

	/// Return whether `metric` improves the currently retained best value.
	pub fn is_better(&self, metric: f64) -> bool {
		self.inner.is_better(metric)
	}

	/// Return the best metric admitted in this manager session.
	#[getter]
	pub fn best_metric(&self) -> f64 {
		self.inner.best_metric()
	}

	/// Return the configured metric name.
	#[getter]
	pub fn metric_name(&self) -> &str {
		self.inner.metric_name()
	}

	/// Save an incremental checkpoint and update the master file on improvement.
	///
	/// `model` may be any recognized `Module` type.
	/// `optimizer` may be `Sgd`, `Adam`, `AdamW`, or `Muon`.
	/// Returns `True` when the metric improved and the master was updated.
	pub fn maybe_save(
		&mut self,
		py: Python<'_>,
		model: &Bound<'_, PyAny>,
		optimizer: &Bound<'_, PyAny>,
		step: u64,
		metric: f64,
		force: bool,
	) -> PyResult<bool> {
		with_module_shared_opt(model, optimizer, py, |m, opt| {
			self.inner.maybe_save(m, opt, step, metric, force)
		})
	}

	/// Save a resumable incremental checkpoint without changing best selection.
	///
	/// Returns the path of the written checkpoint file.
	pub fn save_incremental(
		&mut self,
		py: Python<'_>,
		model: &Bound<'_, PyAny>,
		optimizer: &Bound<'_, PyAny>,
		step: u64,
		metric: f64,
		metric_name: Option<&str>,
	) -> PyResult<String> {
		with_module_shared_opt(model, optimizer, py, |m, opt| {
			self
				.inner
				.save_incremental(m, opt, step, metric, metric_name)
				.map(|p| p.to_string_lossy().into_owned())
		})
	}

	/// Restore the master/best checkpoint into existing owners.
	pub fn load_best_into(
		&self,
		py: Python<'_>,
		model: &Bound<'_, PyAny>,
		optimizer: &Bound<'_, PyAny>,
	) -> PyResult<()> {
		with_module_mut_opt(model, optimizer, py, |m, opt| {
			self.inner.load_best_into(m, opt)
		})
	}

	/// Restore the highest-step incremental checkpoint. Returns the restored step.
	pub fn load_latest_into(
		&self,
		py: Python<'_>,
		model: &Bound<'_, PyAny>,
		optimizer: &Bound<'_, PyAny>,
	) -> PyResult<u64> {
		with_module_mut_opt(model, optimizer, py, |m, opt| {
			self.inner.load_latest_into(m, opt)
		})
	}

	/// Restore the latest incremental state and recover best-metric history.
	/// Returns the restored step.
	pub fn resume_latest_into(
		&mut self,
		py: Python<'_>,
		model: &Bound<'_, PyAny>,
		optimizer: &Bound<'_, PyAny>,
	) -> PyResult<u64> {
		with_module_mut_opt(model, optimizer, py, |m, opt| {
			self.inner.resume_latest_into(m, opt)
		})
	}

	pub fn __repr__(&self) -> String {
		format!(
			"CheckpointManager(metric={:?}, best={:.6})",
			self.inner.metric_name(),
			self.inner.best_metric(),
		)
	}
}

// ── Combined module + shared-optimizer dispatch ───────────────────────────────

fn with_module_shared_opt<R>(
	model: &Bound<'_, PyAny>,
	optimizer: &Bound<'_, PyAny>,
	py: Python<'_>,
	f: impl FnOnce(&dyn oa::ml::Module, &dyn oa::ml::CheckpointOptimizer) -> oa::Result<R>,
) -> PyResult<R> {
	use super::actor_critic::PythonCategoricalActorCritic;
	use super::nlp::{
		PythonBpeGru, PythonBpeMamba3, PythonBpeMoeTransformer, PythonBpeRnn, PythonBpeTransformer,
		PythonByteEmpyrealm, PythonByteGru, PythonByteMamba3, PythonByteMoeTransformer, PythonByteRnn,
		PythonByteTransformer, PythonCharGru, PythonCharMamba3, PythonCharMoeTransformer,
		PythonCharRnn, PythonCharTransformer,
	};
	use super::optim::{PythonAdam, PythonAdamW, PythonMuon, PythonSgd};

	macro_rules! dispatch_model_shared {
		([$($mty:ty),+ $(,)?]) => {
			$(if let Ok(m) = model.extract::<PyRef<$mty>>() {
				let m_ref: &dyn oa::ml::Module = &m.inner;
				if let Ok(o) = optimizer.extract::<PyRef<PythonAdamW>>() {
					return f(m_ref, &o.inner).map_err(python_error);
				}
				if let Ok(o) = optimizer.extract::<PyRef<PythonAdam>>() {
					return f(m_ref, &o.inner).map_err(python_error);
				}
				if let Ok(o) = optimizer.extract::<PyRef<PythonSgd>>() {
					return f(m_ref, &o.inner).map_err(python_error);
				}
				if let Ok(o) = optimizer.extract::<PyRef<PythonMuon>>() {
					return f(m_ref, &o.inner).map_err(python_error);
				}
				return Err(pyo3::exceptions::PyTypeError::new_err(
					"optimizer must be one of Sgd, Adam, AdamW, or Muon",
				));
			})+
		};
	}

	let _ = py;
	dispatch_model_shared!([
		PythonCategoricalActorCritic,
		PythonCharRnn,
		PythonCharGru,
		PythonCharTransformer,
		PythonCharMoeTransformer,
		PythonCharMamba3,
		PythonByteRnn,
		PythonByteGru,
		PythonByteTransformer,
		PythonByteMoeTransformer,
		PythonByteMamba3,
		PythonByteEmpyrealm,
		PythonBpeRnn,
		PythonBpeGru,
		PythonBpeTransformer,
		PythonBpeMoeTransformer,
		PythonBpeMamba3,
	]);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be a recognized Module type",
	))
}

// ── Combined module + mut-optimizer dispatch ─────────────────────────────────
//
// `with_module` requires `f: FnOnce(&dyn Module) -> oa::Result<R>`.  When the
// model and optimizer must *both* be dispatched and the Rust API takes
// `(&dyn Module, &mut dyn CheckpointOptimizer)` we cannot nest the two
// helpers (their closure types differ).  This helper does the two extractions
// sequentially and composes them cleanly.

fn with_module_mut_opt<R>(
	model: &Bound<'_, PyAny>,
	optimizer: &Bound<'_, PyAny>,
	py: Python<'_>,
	f: impl FnOnce(&dyn oa::ml::Module, &mut dyn oa::ml::CheckpointOptimizer) -> oa::Result<R>,
) -> PyResult<R> {
	use super::actor_critic::PythonCategoricalActorCritic;
	use super::nlp::{
		PythonBpeGru, PythonBpeMamba3, PythonBpeMoeTransformer, PythonBpeRnn, PythonBpeTransformer,
		PythonByteEmpyrealm, PythonByteGru, PythonByteMamba3, PythonByteMoeTransformer, PythonByteRnn,
		PythonByteTransformer, PythonCharGru, PythonCharMamba3, PythonCharMoeTransformer,
		PythonCharRnn, PythonCharTransformer,
	};
	use super::optim::{PythonAdam, PythonAdamW, PythonMuon, PythonSgd};

	macro_rules! dispatch_model {
		([$($mty:ty),+ $(,)?]) => {
			$(if let Ok(m) = model.extract::<PyRef<$mty>>() {
				let m_ref: &dyn oa::ml::Module = &m.inner;
				// Dispatch optimizer separately.
				if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonAdamW>>() {
					return f(m_ref, &mut o.inner).map_err(python_error);
				}
				if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonAdam>>() {
					return f(m_ref, &mut o.inner).map_err(python_error);
				}
				if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonSgd>>() {
					return f(m_ref, &mut o.inner).map_err(python_error);
				}
				if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonMuon>>() {
					return f(m_ref, &mut o.inner).map_err(python_error);
				}
				return Err(pyo3::exceptions::PyTypeError::new_err(
					"optimizer must be one of Sgd, Adam, AdamW, or Muon",
				));
			})+
		};
	}

	let _ = py;
	dispatch_model!([
		PythonCategoricalActorCritic,
		PythonCharRnn,
		PythonCharGru,
		PythonCharTransformer,
		PythonCharMoeTransformer,
		PythonCharMamba3,
		PythonByteRnn,
		PythonByteGru,
		PythonByteTransformer,
		PythonByteMoeTransformer,
		PythonByteMamba3,
		PythonByteEmpyrealm,
		PythonBpeRnn,
		PythonBpeGru,
		PythonBpeTransformer,
		PythonBpeMoeTransformer,
		PythonBpeMamba3,
	]);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be a recognized Module type",
	))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonCheckpointManagerConfig>()?;
	module.add_class::<PythonCheckpointManager>()?;
	Ok(())
}
