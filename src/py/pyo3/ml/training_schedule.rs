use pyo3::prelude::*;

use crate::error::python_error;

// ── CosineScheduler ───────────────────────────────────────────────────────────

/// Cosine annealing from a maximum rate to a minimum rate.
#[pyclass(name = "CosineScheduler", unsendable)]
pub(crate) struct PythonCosineScheduler {
	pub(crate) inner: oa::ml::CosineScheduler,
}

#[pymethods]
impl PythonCosineScheduler {
	#[new]
	#[pyo3(signature = (initial_lr, final_lr = 0.0, total_steps = 1000))]
	pub fn new(initial_lr: f32, final_lr: f32, total_steps: u64) -> PyResult<Self> {
		oa::ml::CosineScheduler::new(initial_lr, final_lr, total_steps)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"CosineScheduler"
	}
}

// ── WarmupScheduler ───────────────────────────────────────────────────────────

/// Linear warmup that holds its target rate (or chains a delegated schedule).
#[pyclass(name = "WarmupScheduler", unsendable)]
pub(crate) struct PythonWarmupScheduler {
	pub(crate) inner: oa::ml::WarmupScheduler,
}

#[pymethods]
impl PythonWarmupScheduler {
	/// Construct a warmup that stays at `target_learning_rate` after the span.
	#[new]
	pub fn new(target_learning_rate: f32, warmup_steps: u64) -> PyResult<Self> {
		oa::ml::WarmupScheduler::new(target_learning_rate, warmup_steps)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"WarmupScheduler"
	}
}

// ── OneCycleScheduler ─────────────────────────────────────────────────────────

/// Smith one-cycle ramp-and-cosine-decay policy.
#[pyclass(name = "OneCycleScheduler", unsendable)]
pub(crate) struct PythonOneCycleScheduler {
	pub(crate) inner: oa::ml::OneCycleScheduler,
}

#[pymethods]
impl PythonOneCycleScheduler {
	/// Construct with OA defaults (percent_start=0.3, div=25, final_div=10000).
	#[new]
	#[pyo3(signature = (max_learning_rate, total_steps, percent_start = 0.3, division_factor = 25.0, final_division_factor = 10_000.0))]
	pub fn new(
		max_learning_rate: f32,
		total_steps: u64,
		percent_start: f32,
		division_factor: f32,
		final_division_factor: f32,
	) -> PyResult<Self> {
		oa::ml::OneCycleScheduler::new(
			max_learning_rate,
			total_steps,
			percent_start,
			division_factor,
			final_division_factor,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"OneCycleScheduler"
	}
}

// ── CyclicMode ────────────────────────────────────────────────────────────────

/// Amplitude policy for `CyclicScheduler`.
#[pyclass(name = "CyclicMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonCyclicMode {
	Triangular = 0,
	Triangular2 = 1,
	ExpRange = 2,
}

impl From<PythonCyclicMode> for oa::ml::CyclicMode {
	fn from(value: PythonCyclicMode) -> Self {
		match value {
			PythonCyclicMode::Triangular => oa::ml::CyclicMode::Triangular,
			PythonCyclicMode::Triangular2 => oa::ml::CyclicMode::Triangular2,
			PythonCyclicMode::ExpRange => oa::ml::CyclicMode::ExpRange,
		}
	}
}

// ── CyclicScheduler ───────────────────────────────────────────────────────────

/// Triangular cyclic learning-rate policy.
#[pyclass(name = "CyclicScheduler", unsendable)]
pub(crate) struct PythonCyclicScheduler {
	pub(crate) inner: oa::ml::CyclicScheduler,
}

#[pymethods]
impl PythonCyclicScheduler {
	#[new]
	#[pyo3(signature = (base_learning_rate, max_learning_rate, step_size_up, mode = PythonCyclicMode::Triangular, gamma = 1.0))]
	pub fn new(
		base_learning_rate: f32,
		max_learning_rate: f32,
		step_size_up: u64,
		mode: PythonCyclicMode,
		gamma: f32,
	) -> PyResult<Self> {
		oa::ml::CyclicScheduler::new(
			base_learning_rate,
			max_learning_rate,
			step_size_up,
			mode.into(),
			gamma,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"CyclicScheduler"
	}
}

// ── CosineWarmRestartsScheduler ───────────────────────────────────────────────

/// Cosine annealing with periodic warm restarts.
#[pyclass(name = "CosineWarmRestartsScheduler", unsendable)]
pub(crate) struct PythonCosineWarmRestartsScheduler {
	pub(crate) inner: oa::ml::CosineWarmRestartsScheduler,
}

#[pymethods]
impl PythonCosineWarmRestartsScheduler {
	#[new]
	#[pyo3(signature = (max_learning_rate, initial_period, period_multiplier = 1, min_learning_rate = 0.0))]
	pub fn new(
		max_learning_rate: f32,
		initial_period: u64,
		period_multiplier: u64,
		min_learning_rate: f32,
	) -> PyResult<Self> {
		oa::ml::CosineWarmRestartsScheduler::new(
			max_learning_rate,
			initial_period,
			period_multiplier,
			min_learning_rate,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"CosineWarmRestartsScheduler"
	}
}

// ── PlateauMode ───────────────────────────────────────────────────────────────

/// Improvement direction for `ReduceOnPlateauScheduler`.
#[pyclass(name = "PlateauMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonPlateauMode {
	Min = 0,
	Max = 1,
}

impl From<PythonPlateauMode> for oa::ml::PlateauMode {
	fn from(value: PythonPlateauMode) -> Self {
		match value {
			PythonPlateauMode::Min => oa::ml::PlateauMode::Min,
			PythonPlateauMode::Max => oa::ml::PlateauMode::Max,
		}
	}
}

// ── ReduceOnPlateauScheduler ──────────────────────────────────────────────────

/// Stateful schedule that lowers its rate after a monitored plateau.
#[pyclass(name = "ReduceOnPlateauScheduler", unsendable)]
pub(crate) struct PythonReduceOnPlateauScheduler {
	pub(crate) inner: oa::ml::ReduceOnPlateauScheduler,
}

#[pymethods]
impl PythonReduceOnPlateauScheduler {
	#[new]
	#[pyo3(signature = (initial_learning_rate, factor = 0.1, patience = 10, threshold = 1e-4, min_learning_rate = 0.0, mode = PythonPlateauMode::Min))]
	pub fn new(
		initial_learning_rate: f32,
		factor: f32,
		patience: u64,
		threshold: f32,
		min_learning_rate: f32,
		mode: PythonPlateauMode,
	) -> PyResult<Self> {
		oa::ml::ReduceOnPlateauScheduler::new(
			initial_learning_rate,
			factor,
			patience,
			threshold,
			min_learning_rate,
			mode.into(),
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Observe one completed metric value and update the schedule state.
	pub fn step(&mut self, metric: f32) -> PyResult<()> {
		self.inner.step(metric).map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"ReduceOnPlateauScheduler"
	}
}

// ── LinearWarmupCosineScheduler ───────────────────────────────────────────────

/// Linear warmup followed by cosine annealing.
#[pyclass(name = "LinearWarmupCosineScheduler", unsendable)]
pub(crate) struct PythonLinearWarmupCosineScheduler {
	pub(crate) inner: oa::ml::LinearWarmupCosineScheduler,
}

#[pymethods]
impl PythonLinearWarmupCosineScheduler {
	#[new]
	pub fn new(
		warmup_steps: u64,
		total_steps: u64,
		max_learning_rate: f32,
		min_learning_rate: f32,
	) -> PyResult<Self> {
		oa::ml::LinearWarmupCosineScheduler::new(
			warmup_steps,
			total_steps,
			max_learning_rate,
			min_learning_rate,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"LinearWarmupCosineScheduler"
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonCosineScheduler>()?;
	module.add_class::<PythonWarmupScheduler>()?;
	module.add_class::<PythonOneCycleScheduler>()?;
	module.add_class::<PythonCyclicMode>()?;
	module.add_class::<PythonCyclicScheduler>()?;
	module.add_class::<PythonCosineWarmRestartsScheduler>()?;
	module.add_class::<PythonPlateauMode>()?;
	module.add_class::<PythonReduceOnPlateauScheduler>()?;
	module.add_class::<PythonLinearWarmupCosineScheduler>()?;
	Ok(())
}
