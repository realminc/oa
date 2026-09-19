//! Stateful trainer bindings for PPO, DQN, and SAC.
//!
//! # Lifetime adapter
//!
//! `ItTraining<'engine, 'hooks>` stores `&'hooks mut dyn TrainingCallback` and
//! `&'hooks mut dyn TrainingMetric` slices. PyO3 cannot express those borrows
//! directly. The solution:
//!
//! * Wrap each Python callback/metric in a `Box<dyn Trait + 'static>` adapter
//!   that holds a `Py<PyAny>` (GIL-independent reference).
//! * The adapter acquires the GIL when a hook fires and calls the Python method.
//! * Boxed adapters live inside the `Python*Trainer` struct, which satisfies
//!   `'hooks = 'static` — each Rust trainer struct owns its hook storage.
//! * The `unsendable` pyclass attribute enforces single-thread use.
//!
//! # RolloutCollector / evaluate_categorical
//!
//! Both require `&mut dyn Environment` from a Python-side environment object.
//! Since there is no Python-implemented `Environment` type in the current
//! binding surface (the trait is Rust-internal), these are exposed as config
//! and metrics structs only — the full runtime wiring is deferred.

use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::{
	autograd::PythonParameter,
	replay::PythonReplayBuffer,
	rl::PythonPolicyResult,
	trainer_data::{
		PythonDqnTrainerConfig, PythonDqnTrainerMetrics, PythonPpoTrainerConfig,
		PythonPpoTrainerMetrics, PythonRolloutTrainingPhase, PythonSacTrainerConfig,
		PythonSacTrainerMetrics, PythonTrainingSnapshot,
	},
};

// ── Python → dyn TrainingCallback adapter ────────────────────────────────────

/// Calls one Python object's lifecycle hook methods.
///
/// Only hooks the Python object has are called; all others return `Continue`.
/// The return value of each hook is interpreted as: `True` → `Stop`,
/// anything else (including `None`) → `Continue`.
struct PyCallbackAdapter {
	obj: Py<PyAny>,
}

impl oa::ml::TrainingCallback for PyCallbackAdapter {
	fn on_train_begin(
		&mut self,
		ctx: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		call_hook(&self.obj, "on_train_begin", ctx)
	}

	fn on_epoch_begin(
		&mut self,
		ctx: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		call_hook(&self.obj, "on_epoch_begin", ctx)
	}

	fn on_step_end(
		&mut self,
		ctx: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		call_hook(&self.obj, "on_step_end", ctx)
	}

	fn on_epoch_end(
		&mut self,
		ctx: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		call_hook(&self.obj, "on_epoch_end", ctx)
	}

	fn on_train_end(
		&mut self,
		ctx: &mut oa::ml::TrainingCallbackContext<'_>,
	) -> oa::Result<oa::ml::TrainingControl> {
		call_hook(&self.obj, "on_train_end", ctx)
	}
}

fn call_hook(
	obj: &Py<PyAny>,
	method: &str,
	ctx: &mut oa::ml::TrainingCallbackContext<'_>,
) -> oa::Result<oa::ml::TrainingControl> {
	Python::attach(|py| {
		let bound = obj.bind(py);
		if bound.hasattr(method).unwrap_or(false) {
			let snap = PythonTrainingSnapshot::from_inner(ctx.snapshot());
			let result = bound
				.call_method1(method, (snap,))
				.map_err(|e| oa::Error::callback(format!("{method} raised: {e}")))?;
			let stop = result.extract::<bool>().unwrap_or(false);
			if stop {
				return Ok(oa::ml::TrainingControl::Stop);
			}
		}
		Ok(oa::ml::TrainingControl::Continue)
	})
}

// ── Python → dyn TrainingMetric adapter ──────────────────────────────────────

struct PyMetricAdapter {
	obj: Py<PyAny>,
	name_cache: String,
	result_cache: f64,
}

impl PyMetricAdapter {
	fn new(obj: Py<PyAny>) -> Self {
		let name = Python::attach(|py| {
			obj
				.bind(py)
				.getattr("name")
				.and_then(|n| n.extract::<String>())
				.unwrap_or_else(|_| "metric".into())
		});
		Self {
			obj,
			name_cache: name,
			result_cache: 0.0,
		}
	}
}

impl oa::ml::TrainingMetric for PyMetricAdapter {
	fn name(&self) -> &str {
		&self.name_cache
	}

	fn reset(&mut self) {
		Python::attach(|py| {
			let bound = self.obj.bind(py);
			if bound.hasattr("reset").unwrap_or(false) {
				let _ = bound.call_method0("reset");
			}
		});
		self.result_cache = 0.0;
	}

	fn update_step(&mut self, state: oa::ml::TrainingSnapshot) {
		let snap = PythonTrainingSnapshot::from_inner(state);
		self.result_cache = Python::attach(|py| {
			let bound = self.obj.bind(py);
			if bound.hasattr("update_step").unwrap_or(false) {
				let _ = bound.call_method1("update_step", (snap,));
			}
			bound
				.call_method0("result")
				.and_then(|v| v.extract::<f64>())
				.unwrap_or(self.result_cache)
		});
	}

	fn result(&self) -> f64 {
		self.result_cache
	}
}

// ── Helper: box a Python object as a static callback/metric ──────────────────

fn box_callback(obj: &Bound<'_, PyAny>) -> Box<dyn oa::ml::TrainingCallback> {
	Box::new(PyCallbackAdapter {
		obj: obj.clone().unbind(),
	})
}

fn box_metric(obj: &Bound<'_, PyAny>) -> Box<dyn oa::ml::TrainingMetric> {
	Box::new(PyMetricAdapter::new(obj.clone().unbind()))
}

// ── Optimizer dispatch ────────────────────────────────────────────────────────

fn with_optimizer_mut<R>(
	optimizer: &Bound<'_, PyAny>,
	f: impl FnOnce(&mut dyn oa::ml::Optimizer) -> oa::Result<R>,
) -> PyResult<R> {
	use super::optim::{PythonAdam, PythonAdamW, PythonMuon, PythonSgd};
	if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonAdamW>>() {
		return f(&mut o.inner).map_err(python_error);
	}
	if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonAdam>>() {
		return f(&mut o.inner).map_err(python_error);
	}
	if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonSgd>>() {
		return f(&mut o.inner).map_err(python_error);
	}
	if let Ok(mut o) = optimizer.extract::<PyRefMut<PythonMuon>>() {
		return f(&mut o.inner).map_err(python_error);
	}
	Err(pyo3::exceptions::PyTypeError::new_err(
		"optimizer must be one of Sgd, Adam, AdamW, or Muon",
	))
}

// ── ActorCritic dispatch ──────────────────────────────────────────────────────

fn with_actor_critic<R>(
	model: &Bound<'_, PyAny>,
	f: impl FnOnce(&dyn oa::ml::ActorCritic) -> oa::Result<R>,
) -> PyResult<R> {
	use super::actor_critic::PythonCategoricalActorCritic;
	if let Ok(m) = model.extract::<PyRef<PythonCategoricalActorCritic>>() {
		return f(&m.inner).map_err(python_error);
	}
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be a CategoricalActorCritic",
	))
}

/// A heap-pinned type-erased module borrow.  Holds a `PyRef` on the heap so
/// the inner `*const dyn Module` raw pointer remains valid until this is dropped.
enum ModuleBorrow<'py> {
	Actor(PyRef<'py, super::actor_critic::PythonCategoricalActorCritic>),
	CharRnn(PyRef<'py, super::nlp::PythonCharRnn>),
	CharGru(PyRef<'py, super::nlp::PythonCharGru>),
	CharTransformer(PyRef<'py, super::nlp::PythonCharTransformer>),
	CharMoeTransformer(PyRef<'py, super::nlp::PythonCharMoeTransformer>),
	CharMamba3(PyRef<'py, super::nlp::PythonCharMamba3>),
	ByteRnn(PyRef<'py, super::nlp::PythonByteRnn>),
	ByteGru(PyRef<'py, super::nlp::PythonByteGru>),
	ByteTransformer(PyRef<'py, super::nlp::PythonByteTransformer>),
	ByteMoeTransformer(PyRef<'py, super::nlp::PythonByteMoeTransformer>),
	ByteMamba3(PyRef<'py, super::nlp::PythonByteMamba3>),
	ByteEmpyrealm(PyRef<'py, super::nlp::PythonByteEmpyrealm>),
	BpeRnn(PyRef<'py, super::nlp::PythonBpeRnn>),
	BpeGru(PyRef<'py, super::nlp::PythonBpeGru>),
	BpeTransformer(PyRef<'py, super::nlp::PythonBpeTransformer>),
	BpeMoeTransformer(PyRef<'py, super::nlp::PythonBpeMoeTransformer>),
	BpeMamba3(PyRef<'py, super::nlp::PythonBpeMamba3>),
}

impl<'py> ModuleBorrow<'py> {
	fn as_module_ptr(&self) -> *const dyn oa::ml::Module {
		match self {
			Self::Actor(m) => &m.inner as *const dyn oa::ml::Module,
			Self::CharRnn(m) => &m.inner as *const dyn oa::ml::Module,
			Self::CharGru(m) => &m.inner as *const dyn oa::ml::Module,
			Self::CharTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::CharMoeTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::CharMamba3(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteRnn(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteGru(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteMoeTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteMamba3(m) => &m.inner as *const dyn oa::ml::Module,
			Self::ByteEmpyrealm(m) => &m.inner as *const dyn oa::ml::Module,
			Self::BpeRnn(m) => &m.inner as *const dyn oa::ml::Module,
			Self::BpeGru(m) => &m.inner as *const dyn oa::ml::Module,
			Self::BpeTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::BpeMoeTransformer(m) => &m.inner as *const dyn oa::ml::Module,
			Self::BpeMamba3(m) => &m.inner as *const dyn oa::ml::Module,
		}
	}
}

fn extract_module_borrow<'py>(model: &Bound<'py, PyAny>) -> PyResult<ModuleBorrow<'py>> {
	use super::actor_critic::PythonCategoricalActorCritic;
	use super::nlp::{
		PythonBpeGru, PythonBpeMamba3, PythonBpeMoeTransformer, PythonBpeRnn, PythonBpeTransformer,
		PythonByteEmpyrealm, PythonByteGru, PythonByteMamba3, PythonByteMoeTransformer, PythonByteRnn,
		PythonByteTransformer, PythonCharGru, PythonCharMamba3, PythonCharMoeTransformer,
		PythonCharRnn, PythonCharTransformer,
	};
	macro_rules! try_m {
		($($ty:ty, $variant:ident),+ $(,)?) => {
			$(if let Ok(b) = model.extract::<PyRef<$ty>>() { return Ok(ModuleBorrow::$variant(b)); })+
		};
	}
	try_m!(
		PythonCategoricalActorCritic,
		Actor,
		PythonCharRnn,
		CharRnn,
		PythonCharGru,
		CharGru,
		PythonCharTransformer,
		CharTransformer,
		PythonCharMoeTransformer,
		CharMoeTransformer,
		PythonCharMamba3,
		CharMamba3,
		PythonByteRnn,
		ByteRnn,
		PythonByteGru,
		ByteGru,
		PythonByteTransformer,
		ByteTransformer,
		PythonByteMoeTransformer,
		ByteMoeTransformer,
		PythonByteMamba3,
		ByteMamba3,
		PythonByteEmpyrealm,
		ByteEmpyrealm,
		PythonBpeRnn,
		BpeRnn,
		PythonBpeGru,
		BpeGru,
		PythonBpeTransformer,
		BpeTransformer,
		PythonBpeMoeTransformer,
		BpeMoeTransformer,
		PythonBpeMamba3,
		BpeMamba3,
	);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be a recognized Module type",
	))
}

// ── PpoTrainer ────────────────────────────────────────────────────────────────

/// Complete caller-driven categorical PPO lifecycle.
///
/// The engine, model, and optimizer remain Python-owned; the trainer borrows
/// them through raw pointers pinned by the struct's field storage. All three
/// must outlive the `PpoTrainer` instance (i.e. not be garbage-collected while
/// this object is alive — normal Python reference counting ensures this when
/// you keep them as locals or attributes).
#[pyclass(name = "PpoTrainer", unsendable)]
pub(crate) struct PythonPpoTrainer {
	// Python-owned objects keep the underlying Rust values alive.
	_engine: Py<PythonEngine>,
	_model: Py<PyAny>,
	_optimizer: Py<PyAny>,
	// Boxed hook adapters — MUST be before `inner` in field order so they drop
	// after the trainer (Rust drops fields in declaration order).
	_callbacks: Vec<Box<dyn oa::ml::TrainingCallback>>,
	_metrics: Vec<Box<dyn oa::ml::TrainingMetric>>,
	// The trainer itself holds 'static borrows into the owned fields above.
	inner: Option<oa::ml::PpoTrainer<'static, 'static>>,
	config: oa::ml::PpoTrainerConfig,
}

#[pymethods]
impl PythonPpoTrainer {
	/// Construct and validate the PPO trainer.
	///
	/// `model` must be a `CategoricalActorCritic`.
	/// `optimizer` must be `Sgd`, `Adam`, `AdamW`, or `Muon`.
	#[new]
	pub fn new(
		py: Python<'_>,
		engine: Py<PythonEngine>,
		model: Py<PyAny>,
		optimizer: Py<PyAny>,
		config: &PythonPpoTrainerConfig,
	) -> PyResult<Self> {
		let cfg = config.inner.clone();
		// Build the trainer while we hold borrows on the Python objects.
		// The raw pointers remain valid as long as the Py<T> handles are alive,
		// which they are — they live in the returned struct.
		let inner = {
			let eng = engine.bind(py).borrow();
			// SAFETY: &eng.inner lives as long as the Python object pointed to by
			// `engine`. We store `engine` in the returned struct, so it lives at
			// least as long as `inner`.
			let engine_ptr: *const oa::Engine = &eng.inner as *const oa::Engine;
			// Use a type-erased closure return to let Rust unify the 'engine lifetime
			// with the raw-pointer based 'static. The trainer's invariant lifetimes
			// are satisfied by the Py<T> handles stored in the returned struct.
			let result: oa::ml::PpoTrainer<'static, 'static> = {
				let ac_borrow;
				let m_ptr: *const dyn oa::ml::ActorCritic = {
					use super::actor_critic::PythonCategoricalActorCritic;
					if let Ok(m) = model
						.bind(py)
						.extract::<PyRef<PythonCategoricalActorCritic>>()
					{
						ac_borrow = m;
						&ac_borrow.inner as *const dyn oa::ml::ActorCritic
					} else {
						return Err(pyo3::exceptions::PyTypeError::new_err(
							"model must be a CategoricalActorCritic",
						));
					}
				};

				with_optimizer_mut(optimizer.bind(py), |opt| {
					let opt_ptr: *mut dyn oa::ml::Optimizer = opt as *mut dyn oa::ml::Optimizer;
					// SAFETY: engine_ptr, m_ptr, opt_ptr are valid for the lifetime of
					// the returned struct (Py<T> handles keep them alive).
					unsafe {
						std::mem::transmute::<
							oa::Result<oa::ml::PpoTrainer<'_, '_>>,
							oa::Result<oa::ml::PpoTrainer<'static, 'static>>,
						>(oa::ml::PpoTrainer::new(
							&*engine_ptr,
							&*m_ptr,
							&mut *opt_ptr,
							cfg.clone(),
						))
					}
				})?
			};
			result
		};
		Ok(Self {
			_engine: engine,
			_model: model,
			_optimizer: optimizer,
			_callbacks: Vec::new(),
			_metrics: Vec::new(),
			inner: Some(inner),
			config: cfg,
		})
	}

	/// Add a Python callback object. Its hook methods are called at each
	/// training lifecycle boundary. Return `True` from a hook to stop training.
	pub fn add_callback(&mut self, callback: &Bound<'_, PyAny>) {
		let mut boxed = box_callback(callback);
		if let Some(trainer) = self.inner.as_mut() {
			// SAFETY: the Box lives in _callbacks which is in the same struct
			// and therefore outlives `inner` (fields drop in order).
			let cb_ref: &'static mut dyn oa::ml::TrainingCallback =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingCallback) };
			trainer.add_callback(cb_ref);
		}
		self._callbacks.push(boxed);
	}

	/// Add a Python metric object (`name`, `reset`, `update_step`, `result`).
	pub fn add_metric(&mut self, metric: &Bound<'_, PyAny>) {
		let mut boxed = box_metric(metric);
		if let Some(trainer) = self.inner.as_mut() {
			let m_ref: &'static mut dyn oa::ml::TrainingMetric =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingMetric) };
			trainer.add_metric(m_ref);
		}
		self._metrics.push(boxed);
	}

	/// Reset and open the next collection cycle.
	pub fn begin_collection(&mut self) -> PyResult<()> {
		self.trainer_mut()?.begin_collection().map_err(python_error)
	}

	/// Sample one categorical action for the given observation.
	pub fn act(&mut self, observation: &PythonMatrix) -> PyResult<PythonPolicyResult> {
		self
			.trainer_mut()?
			.act(&observation.inner)
			.map(|inner| PythonPolicyResult { inner })
			.map_err(python_error)
	}

	/// Append one step using the `PolicyResult` returned by `act`.
	pub fn observe(
		&mut self,
		observation: &PythonMatrix,
		next_observation: &PythonMatrix,
		reward: &PythonMatrix,
		terminated: &PythonMatrix,
		truncated: &PythonMatrix,
		policy_result: &PythonPolicyResult,
	) -> PyResult<()> {
		self
			.trainer_mut()?
			.observe(
				&observation.inner,
				&next_observation.inner,
				&reward.inner,
				&terminated.inner,
				&truncated.inner,
				&policy_result.inner,
			)
			.map_err(python_error)
	}

	/// Finalize the full collection with GAE and enter the Update phase.
	pub fn end_collection(&mut self) -> PyResult<()> {
		self.trainer_mut()?.end_collection().map_err(python_error)
	}

	/// Rewind collection state after a rejected engine transaction.
	pub fn abort_collection(&mut self) -> PyResult<()> {
		self.trainer_mut()?.abort_collection().map_err(python_error)
	}

	/// Perform one full-batch PPO update epoch.
	/// Returns `False` when paused, stopped, or all rollouts are complete.
	pub fn update(&mut self) -> PyResult<bool> {
		self.trainer_mut()?.update().map_err(python_error)
	}

	/// Return whether another collection must begin before the next update.
	pub fn needs_collection(&self) -> PyResult<bool> {
		Ok(self.trainer()?.needs_collection())
	}

	/// Return whether every configured rollout/update cycle has completed.
	pub fn is_done(&self) -> PyResult<bool> {
		Ok(self.trainer()?.is_done())
	}

	/// Return the current `Collect`/`Update`/`Complete` phase.
	pub fn phase(&self) -> PyResult<PythonRolloutTrainingPhase> {
		Ok(self.trainer()?.phase().into())
	}

	/// Return the last synchronized update metrics.
	pub fn metrics(&self) -> PyResult<PythonPpoTrainerMetrics> {
		Ok(PythonPpoTrainerMetrics {
			inner: self.trainer()?.metrics(),
		})
	}

	/// Return all trainable parameters.
	pub fn all_parameters(&self, py: Python<'_>) -> PyResult<Vec<PythonParameter>> {
		with_actor_critic(self._model.bind(py), |ac| ac.all_parameters()).map(|params| {
			params
				.into_iter()
				.map(|inner| PythonParameter { inner })
				.collect()
		})
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PpoTrainer(rollouts={}, horizon={}, envs={})",
			self.config.rollouts, self.config.horizon, self.config.environments,
		)
	}
}

impl PythonPpoTrainer {
	fn trainer(&self) -> PyResult<&oa::ml::PpoTrainer<'static, 'static>> {
		self
			.inner
			.as_ref()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("PpoTrainer already consumed"))
	}

	fn trainer_mut(&mut self) -> PyResult<&mut oa::ml::PpoTrainer<'static, 'static>> {
		self
			.inner
			.as_mut()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("PpoTrainer already consumed"))
	}
}

// ── DqnTrainer ────────────────────────────────────────────────────────────────

/// Stateful DQN training coordinator.
#[pyclass(name = "DqnTrainer", unsendable)]
pub(crate) struct PythonDqnTrainer {
	_engine: Py<PythonEngine>,
	_online: Py<PyAny>,
	_target: Py<PyAny>,
	_optimizer: Py<PyAny>,
	_replay: Py<PythonReplayBuffer>,
	_callbacks: Vec<Box<dyn oa::ml::TrainingCallback>>,
	_metrics: Vec<Box<dyn oa::ml::TrainingMetric>>,
	inner: Option<oa::ml::DqnTrainer<'static, 'static>>,
	config: oa::ml::DqnTrainerConfig,
}

#[pymethods]
impl PythonDqnTrainer {
	/// Construct and validate the DQN trainer, copying the online model into the target.
	///
	/// Both `online` and `target` must be recognized Module types.
	#[new]
	pub fn new(
		py: Python<'_>,
		engine: Py<PythonEngine>,
		online: Py<PyAny>,
		target: Py<PyAny>,
		optimizer: Py<PyAny>,
		replay: Py<PythonReplayBuffer>,
		config: &PythonDqnTrainerConfig,
	) -> PyResult<Self> {
		let cfg = config.inner.clone();
		let inner = {
			let eng = engine.bind(py).borrow();
			let engine_ptr: *const oa::Engine = &eng.inner as *const oa::Engine;
			let replay_ref = replay.bind(py).borrow();
			let replay_ptr: *const oa::ml::ReplayBuffer =
				&replay_ref.inner as *const oa::ml::ReplayBuffer;
			let result: oa::ml::DqnTrainer<'static, 'static> = {
				let on_borrow = extract_module_borrow(online.bind(py))?;
				let on_ptr = on_borrow.as_module_ptr();
				let tgt_borrow = extract_module_borrow(target.bind(py))?;
				let tgt_ptr = tgt_borrow.as_module_ptr();
				with_optimizer_mut(optimizer.bind(py), |opt| {
					let opt_ptr: *mut dyn oa::ml::Optimizer = opt as *mut dyn oa::ml::Optimizer;
					unsafe {
						std::mem::transmute::<
							oa::Result<oa::ml::DqnTrainer<'_, '_>>,
							oa::Result<oa::ml::DqnTrainer<'static, 'static>>,
						>(oa::ml::DqnTrainer::new(
							&*engine_ptr,
							&*on_ptr,
							&*tgt_ptr,
							&mut *opt_ptr,
							&*replay_ptr,
							cfg.clone(),
						))
					}
				})?
			};
			result
		};
		Ok(Self {
			_engine: engine,
			_online: online,
			_target: target,
			_optimizer: optimizer,
			_replay: replay,
			_callbacks: Vec::new(),
			_metrics: Vec::new(),
			inner: Some(inner),
			config: cfg,
		})
	}

	/// Add a training callback.
	pub fn add_callback(&mut self, callback: &Bound<'_, PyAny>) {
		let mut boxed = box_callback(callback);
		if let Some(trainer) = self.inner.as_mut() {
			let cb_ref: &'static mut dyn oa::ml::TrainingCallback =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingCallback) };
			trainer.add_callback(cb_ref);
		}
		self._callbacks.push(boxed);
	}

	/// Add a training metric.
	pub fn add_metric(&mut self, metric: &Bound<'_, PyAny>) {
		let mut boxed = box_metric(metric);
		if let Some(trainer) = self.inner.as_mut() {
			let m_ref: &'static mut dyn oa::ml::TrainingMetric =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingMetric) };
			trainer.add_metric(m_ref);
		}
		self._metrics.push(boxed);
	}

	/// Complete one DQN replay update. Returns `False` after the budget or a stop.
	pub fn update(&mut self) -> PyResult<bool> {
		self.trainer_mut()?.update().map_err(python_error)
	}

	/// Deep-copy every online parameter into the target module.
	pub fn sync_target(&self) -> PyResult<()> {
		self.trainer()?.sync_target().map_err(python_error)
	}

	/// Return whether the update budget or a callback stop has been reached.
	pub fn is_done(&self) -> PyResult<bool> {
		Ok(self.trainer()?.is_done())
	}

	/// Return the last completed update metrics.
	pub fn metrics(&self) -> PyResult<PythonDqnTrainerMetrics> {
		Ok(PythonDqnTrainerMetrics {
			inner: self.trainer()?.metrics(),
		})
	}

	pub fn __repr__(&self) -> String {
		format!(
			"DqnTrainer(updates={}, batch_size={})",
			self.config.updates, self.config.batch_size,
		)
	}
}

impl PythonDqnTrainer {
	fn trainer(&self) -> PyResult<&oa::ml::DqnTrainer<'static, 'static>> {
		self
			.inner
			.as_ref()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("DqnTrainer already consumed"))
	}

	fn trainer_mut(&mut self) -> PyResult<&mut oa::ml::DqnTrainer<'static, 'static>> {
		self
			.inner
			.as_mut()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("DqnTrainer already consumed"))
	}
}

// ── SacTrainer ────────────────────────────────────────────────────────────────

/// Stateful fixed-temperature SAC coordinator.
#[pyclass(name = "SacTrainer", unsendable)]
pub(crate) struct PythonSacTrainer {
	_engine: Py<PythonEngine>,
	_actor: Py<PyAny>,
	_critic1: Py<PyAny>,
	_critic2: Py<PyAny>,
	_target_critic1: Py<PyAny>,
	_target_critic2: Py<PyAny>,
	_actor_optimizer: Py<PyAny>,
	_critic_optimizer: Py<PyAny>,
	_replay: Py<PythonReplayBuffer>,
	_callbacks: Vec<Box<dyn oa::ml::TrainingCallback>>,
	_metrics: Vec<Box<dyn oa::ml::TrainingMetric>>,
	inner: Option<oa::ml::SacTrainer<'static, 'static>>,
	config: oa::ml::SacTrainerConfig,
}

#[pymethods]
impl PythonSacTrainer {
	/// Construct and validate the SAC trainer, synchronizing targets immediately.
	///
	/// All module arguments must be recognized Module types.
	/// Both `actor_optimizer` and `critic_optimizer` must be standard optimizer types.
	#[new]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		py: Python<'_>,
		engine: Py<PythonEngine>,
		actor: Py<PyAny>,
		critic1: Py<PyAny>,
		critic2: Py<PyAny>,
		target_critic1: Py<PyAny>,
		target_critic2: Py<PyAny>,
		actor_optimizer: Py<PyAny>,
		critic_optimizer: Py<PyAny>,
		replay: Py<PythonReplayBuffer>,
		config: &PythonSacTrainerConfig,
	) -> PyResult<Self> {
		let cfg = config.inner.clone();
		let inner = {
			let eng = engine.bind(py).borrow();
			let engine_ptr: *const oa::Engine = &eng.inner as *const oa::Engine;
			let replay_ref = replay.bind(py).borrow();
			let replay_ptr: *const oa::ml::ReplayBuffer =
				&replay_ref.inner as *const oa::ml::ReplayBuffer;
			let result: oa::ml::SacTrainer<'static, 'static> = {
				let a_borrow = extract_module_borrow(actor.bind(py))?;
				let a_ptr = a_borrow.as_module_ptr();
				let c1_borrow = extract_module_borrow(critic1.bind(py))?;
				let c1_ptr = c1_borrow.as_module_ptr();
				let c2_borrow = extract_module_borrow(critic2.bind(py))?;
				let c2_ptr = c2_borrow.as_module_ptr();
				let tc1_borrow = extract_module_borrow(target_critic1.bind(py))?;
				let tc1_ptr = tc1_borrow.as_module_ptr();
				let tc2_borrow = extract_module_borrow(target_critic2.bind(py))?;
				let tc2_ptr = tc2_borrow.as_module_ptr();
				with_optimizer_mut(actor_optimizer.bind(py), |aopt| {
					let aopt_ptr: *mut dyn oa::ml::Optimizer = aopt;
					let cfg2 = cfg.clone();
					with_optimizer_mut(critic_optimizer.bind(py), move |copt| {
						let copt_ptr: *mut dyn oa::ml::Optimizer = copt;
						unsafe {
							std::mem::transmute::<
								oa::Result<oa::ml::SacTrainer<'_, '_>>,
								oa::Result<oa::ml::SacTrainer<'static, 'static>>,
							>(oa::ml::SacTrainer::new(
								&*engine_ptr,
								&*a_ptr,
								&*c1_ptr,
								&*c2_ptr,
								&*tc1_ptr,
								&*tc2_ptr,
								&mut *aopt_ptr,
								&mut *copt_ptr,
								&*replay_ptr,
								cfg2,
							))
						}
					})
					.map_err(|e| oa::Error::callback(e.to_string()))
				})?
			};
			result
		};
		Ok(Self {
			_engine: engine,
			_actor: actor,
			_critic1: critic1,
			_critic2: critic2,
			_target_critic1: target_critic1,
			_target_critic2: target_critic2,
			_actor_optimizer: actor_optimizer,
			_critic_optimizer: critic_optimizer,
			_replay: replay,
			_callbacks: Vec::new(),
			_metrics: Vec::new(),
			inner: Some(inner),
			config: cfg,
		})
	}

	/// Add a callback to the critic training lifecycle.
	pub fn add_callback(&mut self, callback: &Bound<'_, PyAny>) {
		let mut boxed = box_callback(callback);
		if let Some(trainer) = self.inner.as_mut() {
			let cb_ref: &'static mut dyn oa::ml::TrainingCallback =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingCallback) };
			trainer.training_loop_mut().add_callback(cb_ref);
		}
		self._callbacks.push(boxed);
	}

	/// Add a metric to the critic training lifecycle.
	pub fn add_metric(&mut self, metric: &Bound<'_, PyAny>) {
		let mut boxed = box_metric(metric);
		if let Some(trainer) = self.inner.as_mut() {
			let m_ref: &'static mut dyn oa::ml::TrainingMetric =
				unsafe { &mut *(&mut *boxed as *mut dyn oa::ml::TrainingMetric) };
			trainer.training_loop_mut().add_metric(m_ref);
		}
		self._metrics.push(boxed);
	}

	/// Complete one critic update followed by one actor update.
	/// Returns `False` when either loop stops or reaches its budget.
	pub fn update(&mut self) -> PyResult<bool> {
		self.trainer_mut()?.update().map_err(python_error)
	}

	/// Exactly synchronize both online critics into their target peers.
	pub fn sync_targets(&self) -> PyResult<()> {
		self.trainer()?.sync_targets().map_err(python_error)
	}

	/// Return whether either coordinated loop has stopped or reached its budget.
	pub fn is_done(&self) -> PyResult<bool> {
		Ok(self.trainer()?.is_done())
	}

	/// Return the last completed update-pair metrics.
	pub fn metrics(&self) -> PyResult<PythonSacTrainerMetrics> {
		Ok(PythonSacTrainerMetrics {
			inner: self.trainer()?.metrics(),
		})
	}

	pub fn __repr__(&self) -> String {
		format!(
			"SacTrainer(updates={}, batch_size={}, action_dims={})",
			self.config.updates, self.config.batch_size, self.config.action_dimensions,
		)
	}
}

impl PythonSacTrainer {
	fn trainer(&self) -> PyResult<&oa::ml::SacTrainer<'static, 'static>> {
		self
			.inner
			.as_ref()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("SacTrainer already consumed"))
	}

	fn trainer_mut(&mut self) -> PyResult<&mut oa::ml::SacTrainer<'static, 'static>> {
		self
			.inner
			.as_mut()
			.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("SacTrainer already consumed"))
	}
}

// ── register ──────────────────────────────────────────────────────────────────

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonPpoTrainer>()?;
	module.add_class::<PythonDqnTrainer>()?;
	module.add_class::<PythonSacTrainer>()?;
	Ok(())
}
