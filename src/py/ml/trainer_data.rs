use pyo3::prelude::*;

use crate::error::python_error;

use super::rl::{PythonDqnLossConfig, PythonGaeConfig, PythonPpoLossConfig, PythonSacLossConfig};

// ── RolloutTrainingPhase ──────────────────────────────────────────────────────

/// Current phase of one synchronous on-policy schedule.
#[pyclass(name = "RolloutTrainingPhase", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonRolloutTrainingPhase {
	Collect = 0,
	Update = 1,
	Complete = 2,
}

impl From<oa::ml::RolloutTrainingPhase> for PythonRolloutTrainingPhase {
	fn from(value: oa::ml::RolloutTrainingPhase) -> Self {
		match value {
			oa::ml::RolloutTrainingPhase::Collect => PythonRolloutTrainingPhase::Collect,
			oa::ml::RolloutTrainingPhase::Update => PythonRolloutTrainingPhase::Update,
			oa::ml::RolloutTrainingPhase::Complete => PythonRolloutTrainingPhase::Complete,
		}
	}
}

// ── GpuTimingStats ────────────────────────────────────────────────────────────

/// Aggregate device time for completed, timestamped training steps.
#[pyclass(name = "GpuTimingStats")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonGpuTimingStats {
	inner: oa::ml::GpuTimingStats,
}

#[pymethods]
impl PythonGpuTimingStats {
	#[getter]
	pub fn count(&self) -> u64 {
		self.inner.count
	}

	#[getter]
	pub fn mean_ms(&self) -> f64 {
		self.inner.mean_ms
	}

	#[getter]
	pub fn min_ms(&self) -> f64 {
		self.inner.min_ms
	}

	#[getter]
	pub fn max_ms(&self) -> f64 {
		self.inner.max_ms
	}

	#[getter]
	pub fn median_ms(&self) -> f64 {
		self.inner.median_ms
	}

	#[getter]
	pub fn p95_ms(&self) -> f64 {
		self.inner.p95_ms
	}

	#[getter]
	pub fn last_ms(&self) -> f64 {
		self.inner.last_ms
	}

	pub fn __repr__(&self) -> String {
		format!(
			"GpuTimingStats(count={}, mean_ms={:.3}, p95_ms={:.3})",
			self.inner.count, self.inner.mean_ms, self.inner.p95_ms,
		)
	}
}

// ── TrainingSnapshot ──────────────────────────────────────────────────────────

/// Immutable state supplied to metrics and callbacks.
#[pyclass(name = "TrainingSnapshot")]
#[derive(Clone, Copy)]
pub(crate) struct PythonTrainingSnapshot {
	inner: oa::ml::TrainingSnapshot,
}

#[pymethods]
impl PythonTrainingSnapshot {
	#[getter]
	pub fn step_count(&self) -> u64 {
		self.inner.step_count()
	}

	#[getter]
	pub fn total_steps(&self) -> u64 {
		self.inner.total_steps()
	}

	#[getter]
	pub fn epoch(&self) -> u64 {
		self.inner.epoch()
	}

	#[getter]
	pub fn total_epochs(&self) -> u64 {
		self.inner.total_epochs()
	}

	#[getter]
	pub fn step_in_epoch(&self) -> u64 {
		self.inner.step_in_epoch()
	}

	#[getter]
	pub fn steps_in_epoch(&self) -> u64 {
		self.inner.steps_in_epoch()
	}

	#[getter]
	pub fn is_epoch_boundary(&self) -> bool {
		self.inner.is_epoch_boundary()
	}

	#[getter]
	pub fn is_last_step(&self) -> bool {
		self.inner.is_last_step()
	}

	#[getter]
	pub fn last_loss(&self) -> Option<f32> {
		self.inner.last_loss()
	}

	#[getter]
	pub fn total_samples(&self) -> u64 {
		self.inner.total_samples()
	}

	#[getter]
	pub fn total_units(&self) -> u64 {
		self.inner.total_units()
	}

	#[getter]
	pub fn total_source_units(&self) -> u64 {
		self.inner.total_source_units()
	}

	#[getter]
	pub fn elapsed_secs(&self) -> f64 {
		self.inner.elapsed().as_secs_f64()
	}

	#[getter]
	pub fn epoch_elapsed_secs(&self) -> f64 {
		self.inner.epoch_elapsed().as_secs_f64()
	}

	#[getter]
	pub fn training_mean_loss(&self) -> f64 {
		self.inner.training_mean_loss()
	}

	#[getter]
	pub fn epoch_mean_loss(&self) -> f64 {
		self.inner.epoch_mean_loss()
	}

	pub fn wall_ms_per_step(&self) -> f64 {
		self.inner.wall_ms_per_step()
	}

	pub fn wall_samples_per_second(&self) -> f64 {
		self.inner.wall_samples_per_second()
	}

	pub fn gpu_timing_stats(&self) -> PythonGpuTimingStats {
		PythonGpuTimingStats {
			inner: self.inner.gpu_timing_stats(),
		}
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingSnapshot(step={}, epoch={}, loss={:?})",
			self.inner.step_count(),
			self.inner.epoch(),
			self.inner.last_loss(),
		)
	}
}

impl PythonTrainingSnapshot {
	pub(crate) fn from_inner(inner: oa::ml::TrainingSnapshot) -> Self {
		Self { inner }
	}

	pub(crate) fn inner(&self) -> oa::ml::TrainingSnapshot {
		self.inner
	}
}

// ── ItTrainingConfig ──────────────────────────────────────────────────────────

/// Configuration for one explicit training loop.
#[pyclass(name = "ItTrainingConfig")]
#[derive(Clone)]
pub(crate) struct PythonItTrainingConfig {
	pub(crate) inner: oa::ml::ItTrainingConfig,
}

#[pymethods]
impl PythonItTrainingConfig {
	#[new]
	#[pyo3(signature = (
		total_steps = 0,
		initial_step = 0,
		steps_per_epoch = 0,
		batch_size = 1,
		sequence_length = 0,
		sequence_unit = "token",
		source_units_per_sample = 0.0,
		source_unit = "byte",
		timer_name = "training_step",
		enable_gpu_timing = false,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		total_steps: u64,
		initial_step: u64,
		steps_per_epoch: u64,
		batch_size: u64,
		sequence_length: u64,
		sequence_unit: &str,
		source_units_per_sample: f64,
		source_unit: &str,
		timer_name: &str,
		enable_gpu_timing: bool,
	) -> Self {
		Self {
			inner: oa::ml::ItTrainingConfig {
				total_steps,
				initial_step,
				steps_per_epoch,
				batch_size,
				sequence_length,
				sequence_unit: sequence_unit.into(),
				source_units_per_sample,
				source_unit: source_unit.into(),
				timer_name: timer_name.into(),
				enable_gpu_timing,
				..oa::ml::ItTrainingConfig::default()
			},
		}
	}

	#[getter]
	pub fn total_steps(&self) -> u64 {
		self.inner.total_steps
	}

	#[getter]
	pub fn initial_step(&self) -> u64 {
		self.inner.initial_step
	}

	#[getter]
	pub fn steps_per_epoch(&self) -> u64 {
		self.inner.steps_per_epoch
	}

	#[getter]
	pub fn batch_size(&self) -> u64 {
		self.inner.batch_size
	}

	#[getter]
	pub fn sequence_length(&self) -> u64 {
		self.inner.sequence_length
	}

	#[getter]
	pub fn sequence_unit(&self) -> &str {
		&self.inner.sequence_unit
	}

	#[getter]
	pub fn source_units_per_sample(&self) -> f64 {
		self.inner.source_units_per_sample
	}

	#[getter]
	pub fn source_unit(&self) -> &str {
		&self.inner.source_unit
	}

	#[getter]
	pub fn timer_name(&self) -> &str {
		&self.inner.timer_name
	}

	#[getter]
	pub fn enable_gpu_timing(&self) -> bool {
		self.inner.enable_gpu_timing
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ItTrainingConfig(total_steps={}, batch={}, epochs={})",
			self.inner.total_steps, self.inner.batch_size, self.inner.steps_per_epoch,
		)
	}
}

// ── LossAggregation ───────────────────────────────────────────────────────────

/// Aggregation policy for `LossMetric`.
#[pyclass(name = "LossAggregation", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonLossAggregation {
	Mean = 0,
	Last = 1,
}

impl From<PythonLossAggregation> for oa::ml::LossAggregation {
	fn from(value: PythonLossAggregation) -> Self {
		match value {
			PythonLossAggregation::Mean => oa::ml::LossAggregation::Mean,
			PythonLossAggregation::Last => oa::ml::LossAggregation::Last,
		}
	}
}

// ── LossMetric ────────────────────────────────────────────────────────────────

/// Running completed-step loss metric.
#[pyclass(name = "LossMetric", unsendable)]
pub(crate) struct PythonLossMetric {
	pub(crate) inner: oa::ml::LossMetric,
}

#[pymethods]
impl PythonLossMetric {
	#[new]
	#[pyo3(signature = (name = "loss", aggregation = PythonLossAggregation::Mean))]
	pub fn new(name: &str, aggregation: PythonLossAggregation) -> Self {
		Self {
			inner: oa::ml::LossMetric::new(name, aggregation.into()),
		}
	}

	pub fn count(&self) -> u64 {
		self.inner.count()
	}

	pub fn mean(&self) -> f64 {
		self.inner.mean()
	}

	pub fn last(&self) -> f64 {
		self.inner.last()
	}

	pub fn result(&self) -> f64 {
		use oa::ml::TrainingMetric as _;
		self.inner.result()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"LossMetric(count={}, mean={:.6})",
			self.inner.count(),
			self.inner.mean()
		)
	}
}

// ── PpoTrainerConfig ──────────────────────────────────────────────────────────

/// Complete categorical PPO collection and update policy.
#[pyclass(name = "PpoTrainerConfig")]
#[derive(Clone)]
pub(crate) struct PythonPpoTrainerConfig {
	pub(crate) inner: oa::ml::PpoTrainerConfig,
}

#[pymethods]
impl PythonPpoTrainerConfig {
	#[new]
	#[pyo3(signature = (
		rollouts,
		horizon,
		environments,
		update_epochs,
		observation_shape,
		seed = 0,
		enable_gpu_timing = false,
		gae = None,
		loss = None,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		rollouts: u64,
		horizon: usize,
		environments: usize,
		update_epochs: u64,
		observation_shape: Vec<usize>,
		seed: u64,
		enable_gpu_timing: bool,
		gae: Option<&PythonGaeConfig>,
		loss: Option<&PythonPpoLossConfig>,
	) -> Self {
		Self {
			inner: oa::ml::PpoTrainerConfig {
				rollouts,
				horizon,
				environments,
				update_epochs,
				observation_shape,
				seed,
				enable_gpu_timing,
				gae: gae.map(|g| g.inner).unwrap_or_default(),
				loss: loss.map(|l| l.inner).unwrap_or_default(),
			},
		}
	}

	#[getter]
	pub fn rollouts(&self) -> u64 {
		self.inner.rollouts
	}

	#[getter]
	pub fn horizon(&self) -> usize {
		self.inner.horizon
	}

	#[getter]
	pub fn environments(&self) -> usize {
		self.inner.environments
	}

	#[getter]
	pub fn update_epochs(&self) -> u64 {
		self.inner.update_epochs
	}

	#[getter]
	pub fn observation_shape(&self) -> Vec<usize> {
		self.inner.observation_shape.clone()
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	#[getter]
	pub fn enable_gpu_timing(&self) -> bool {
		self.inner.enable_gpu_timing
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PpoTrainerConfig(rollouts={}, horizon={}, envs={}, update_epochs={})",
			self.inner.rollouts, self.inner.horizon, self.inner.environments, self.inner.update_epochs,
		)
	}
}

// ── PpoTrainerMetrics ─────────────────────────────────────────────────────────

/// Scalar results from the last completed PPO update epoch.
#[pyclass(name = "PpoTrainerMetrics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonPpoTrainerMetrics {
	pub(crate) inner: oa::ml::PpoTrainerMetrics,
}

#[pymethods]
impl PythonPpoTrainerMetrics {
	#[getter]
	pub fn rollout(&self) -> u64 {
		self.inner.rollout
	}

	#[getter]
	pub fn update_epoch(&self) -> u64 {
		self.inner.update_epoch
	}

	#[getter]
	pub fn total_loss(&self) -> f32 {
		self.inner.total_loss
	}

	#[getter]
	pub fn policy_loss(&self) -> f32 {
		self.inner.policy_loss
	}

	#[getter]
	pub fn value_loss(&self) -> f32 {
		self.inner.value_loss
	}

	#[getter]
	pub fn entropy(&self) -> f32 {
		self.inner.entropy
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PpoTrainerMetrics(rollout={}, epoch={}, total_loss={:.4})",
			self.inner.rollout, self.inner.update_epoch, self.inner.total_loss,
		)
	}
}

// ── DqnTrainerConfig ──────────────────────────────────────────────────────────

/// Fixed update budget, replay batch, target cadence, and Bellman policy.
#[pyclass(name = "DqnTrainerConfig")]
#[derive(Clone)]
pub(crate) struct PythonDqnTrainerConfig {
	pub(crate) inner: oa::ml::DqnTrainerConfig,
}

#[pymethods]
impl PythonDqnTrainerConfig {
	#[new]
	#[pyo3(signature = (
		updates,
		batch_size,
		observation_shape,
		target_update_interval = 100,
		seed = 0,
		loss = None,
	))]
	pub fn new(
		updates: u64,
		batch_size: usize,
		observation_shape: Vec<usize>,
		target_update_interval: u64,
		seed: u64,
		loss: Option<&PythonDqnLossConfig>,
	) -> Self {
		Self {
			inner: oa::ml::DqnTrainerConfig {
				updates,
				batch_size,
				observation_shape,
				target_update_interval,
				seed,
				loss: loss.map(|l| l.inner).unwrap_or_default(),
			},
		}
	}

	#[getter]
	pub fn updates(&self) -> u64 {
		self.inner.updates
	}

	#[getter]
	pub fn batch_size(&self) -> usize {
		self.inner.batch_size
	}

	#[getter]
	pub fn observation_shape(&self) -> Vec<usize> {
		self.inner.observation_shape.clone()
	}

	#[getter]
	pub fn target_update_interval(&self) -> u64 {
		self.inner.target_update_interval
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	pub fn __repr__(&self) -> String {
		format!(
			"DqnTrainerConfig(updates={}, batch_size={}, target_interval={})",
			self.inner.updates, self.inner.batch_size, self.inner.target_update_interval,
		)
	}
}

// ── DqnTrainerMetrics ─────────────────────────────────────────────────────────

/// Last completed DQN update and its synchronized scalar loss.
#[pyclass(name = "DqnTrainerMetrics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonDqnTrainerMetrics {
	pub(crate) inner: oa::ml::DqnTrainerMetrics,
}

#[pymethods]
impl PythonDqnTrainerMetrics {
	#[getter]
	pub fn update(&self) -> u64 {
		self.inner.update
	}

	#[getter]
	pub fn loss(&self) -> f32 {
		self.inner.loss
	}

	pub fn __repr__(&self) -> String {
		format!(
			"DqnTrainerMetrics(update={}, loss={:.4})",
			self.inner.update, self.inner.loss,
		)
	}
}

// ── SacTrainerConfig ──────────────────────────────────────────────────────────

/// Fixed SAC update, shape, action-range, target, and loss policy.
#[pyclass(name = "SacTrainerConfig")]
#[derive(Clone)]
pub(crate) struct PythonSacTrainerConfig {
	pub(crate) inner: oa::ml::SacTrainerConfig,
}

#[pymethods]
impl PythonSacTrainerConfig {
	#[new]
	#[pyo3(signature = (
		updates,
		batch_size,
		action_dimensions,
		observation_shape,
		target_update_interval = 1,
		action_minimum = -1.0,
		action_maximum = 1.0,
		seed = 0,
		loss = None,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		updates: u64,
		batch_size: usize,
		action_dimensions: usize,
		observation_shape: Vec<usize>,
		target_update_interval: u64,
		action_minimum: f32,
		action_maximum: f32,
		seed: u64,
		loss: Option<&PythonSacLossConfig>,
	) -> Self {
		Self {
			inner: oa::ml::SacTrainerConfig {
				updates,
				batch_size,
				action_dimensions,
				observation_shape,
				target_update_interval,
				action_minimum,
				action_maximum,
				seed,
				loss: loss.map(|l| l.inner).unwrap_or_default(),
			},
		}
	}

	#[getter]
	pub fn updates(&self) -> u64 {
		self.inner.updates
	}

	#[getter]
	pub fn batch_size(&self) -> usize {
		self.inner.batch_size
	}

	#[getter]
	pub fn action_dimensions(&self) -> usize {
		self.inner.action_dimensions
	}

	#[getter]
	pub fn observation_shape(&self) -> Vec<usize> {
		self.inner.observation_shape.clone()
	}

	#[getter]
	pub fn target_update_interval(&self) -> u64 {
		self.inner.target_update_interval
	}

	#[getter]
	pub fn action_minimum(&self) -> f32 {
		self.inner.action_minimum
	}

	#[getter]
	pub fn action_maximum(&self) -> f32 {
		self.inner.action_maximum
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	pub fn __repr__(&self) -> String {
		format!(
			"SacTrainerConfig(updates={}, batch_size={}, action_dims={})",
			self.inner.updates, self.inner.batch_size, self.inner.action_dimensions,
		)
	}
}

// ── SacTrainerMetrics ─────────────────────────────────────────────────────────

/// Last completed SAC actor/critic update pair.
#[pyclass(name = "SacTrainerMetrics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonSacTrainerMetrics {
	pub(crate) inner: oa::ml::SacTrainerMetrics,
}

#[pymethods]
impl PythonSacTrainerMetrics {
	#[getter]
	pub fn update(&self) -> u64 {
		self.inner.update
	}

	#[getter]
	pub fn actor_loss(&self) -> f32 {
		self.inner.actor_loss
	}

	#[getter]
	pub fn critic_loss(&self) -> f32 {
		self.inner.critic_loss
	}

	pub fn __repr__(&self) -> String {
		format!(
			"SacTrainerMetrics(update={}, actor_loss={:.4}, critic_loss={:.4})",
			self.inner.update, self.inner.actor_loss, self.inner.critic_loss,
		)
	}
}

// ── RolloutCollectorConfig ────────────────────────────────────────────────────

/// Fixed categorical collection horizon and sampling policy.
#[pyclass(name = "RolloutCollectorConfig")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonRolloutCollectorConfig {
	pub(crate) inner: oa::ml::RolloutCollectorConfig,
}

#[pymethods]
impl PythonRolloutCollectorConfig {
	#[new]
	#[pyo3(signature = (horizon, seed = 0, gae = None))]
	pub fn new(horizon: usize, seed: u64, gae: Option<&PythonGaeConfig>) -> Self {
		Self {
			inner: oa::ml::RolloutCollectorConfig {
				horizon,
				seed,
				gae: gae.map(|g| g.inner).unwrap_or_default(),
			},
		}
	}

	#[getter]
	pub fn horizon(&self) -> usize {
		self.inner.horizon
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	pub fn __repr__(&self) -> String {
		format!(
			"RolloutCollectorConfig(horizon={}, seed={})",
			self.inner.horizon, self.inner.seed,
		)
	}
}

// ── RolloutCollectorMetrics ───────────────────────────────────────────────────

/// Accepted rollout and transition counts.
#[pyclass(name = "RolloutCollectorMetrics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonRolloutCollectorMetrics {
	pub(crate) inner: oa::ml::RolloutCollectorMetrics,
}

#[pymethods]
impl PythonRolloutCollectorMetrics {
	#[getter]
	pub fn collections(&self) -> u64 {
		self.inner.collections
	}

	#[getter]
	pub fn environment_steps(&self) -> u64 {
		self.inner.environment_steps
	}

	#[getter]
	pub fn transitions(&self) -> u64 {
		self.inner.transitions
	}

	pub fn __repr__(&self) -> String {
		format!(
			"RolloutCollectorMetrics(collections={}, env_steps={}, transitions={})",
			self.inner.collections, self.inner.environment_steps, self.inner.transitions,
		)
	}
}

// ── PolicyEvaluationConfig ────────────────────────────────────────────────────

/// Fixed deterministic categorical evaluation horizon.
#[pyclass(name = "PolicyEvaluationConfig")]
#[derive(Clone, Copy)]
pub(crate) struct PythonPolicyEvaluationConfig {
	pub(crate) inner: oa::ml::evaluation::PolicyEvaluationConfig,
}

#[pymethods]
impl PythonPolicyEvaluationConfig {
	#[new]
	#[pyo3(signature = (horizon = 1000, seed = 1))]
	pub fn new(horizon: usize, seed: u64) -> Self {
		Self {
			inner: oa::ml::evaluation::PolicyEvaluationConfig { horizon, seed },
		}
	}

	#[getter]
	pub fn horizon(&self) -> usize {
		self.inner.horizon
	}

	#[getter]
	pub fn seed(&self) -> u64 {
		self.inner.seed
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PolicyEvaluationConfig(horizon={}, seed={})",
			self.inner.horizon, self.inner.seed,
		)
	}
}

// ── PolicyEvaluationMetrics ───────────────────────────────────────────────────

/// Completed categorical evaluation telemetry.
#[pyclass(name = "PolicyEvaluationMetrics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonPolicyEvaluationMetrics {
	pub(crate) inner: oa::ml::evaluation::PolicyEvaluationMetrics,
}

#[pymethods]
impl PythonPolicyEvaluationMetrics {
	#[getter]
	pub fn environment_steps(&self) -> u64 {
		self.inner.environment_steps
	}

	#[getter]
	pub fn transitions(&self) -> u64 {
		self.inner.transitions
	}

	#[getter]
	pub fn completed_episodes(&self) -> u64 {
		self.inner.completed_episodes
	}

	#[getter]
	pub fn mean_completed_return(&self) -> f32 {
		self.inner.mean_completed_return
	}

	#[getter]
	pub fn minimum_completed_return(&self) -> f32 {
		self.inner.minimum_completed_return
	}

	#[getter]
	pub fn maximum_completed_return(&self) -> f32 {
		self.inner.maximum_completed_return
	}

	pub fn __repr__(&self) -> String {
		format!(
			"PolicyEvaluationMetrics(episodes={}, mean_return={:.4})",
			self.inner.completed_episodes, self.inner.mean_completed_return,
		)
	}
}

// ── EarlyStopMode ─────────────────────────────────────────────────────────────

/// Improvement direction for `EarlyStopping`.
#[pyclass(name = "EarlyStopMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonEarlyStopMode {
	Min = 0,
	Max = 1,
}

impl From<PythonEarlyStopMode> for oa::ml::EarlyStopMode {
	fn from(value: PythonEarlyStopMode) -> Self {
		match value {
			PythonEarlyStopMode::Min => oa::ml::EarlyStopMode::Min,
			PythonEarlyStopMode::Max => oa::ml::EarlyStopMode::Max,
		}
	}
}

// ── ProgressBar ───────────────────────────────────────────────────────────────

/// Throttled Keras/tqdm-style wall-throughput progress display.
#[pyclass(name = "ProgressBar", unsendable)]
pub(crate) struct PythonProgressBar {
	pub(crate) inner: oa::ml::ProgressBar,
}

#[pymethods]
impl PythonProgressBar {
	/// Construct with the given bar cell width.
	#[new]
	#[pyo3(signature = (width = 10))]
	pub fn new(width: usize) -> Self {
		Self {
			inner: oa::ml::ProgressBar::new(width),
		}
	}

	/// Enable or suppress the `epoch N/M` header line.
	pub fn set_show_epoch_header(&mut self, show: bool) {
		self.inner.set_show_epoch_header(show);
	}

	/// Format one progress line without writing it.
	pub fn render_line(
		&self,
		snapshot: &PythonTrainingSnapshot,
		config: &PythonItTrainingConfig,
	) -> String {
		self.inner.render_line(snapshot.inner(), &config.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ProgressBar"
	}
}

// ── TrainingSummary ───────────────────────────────────────────────────────────

/// Final loss, wall-time, device-time, throughput, and run summary.
#[pyclass(name = "TrainingSummary", unsendable)]
pub(crate) struct PythonTrainingSummary {
	pub(crate) inner: oa::ml::TrainingSummary,
}

#[pymethods]
impl PythonTrainingSummary {
	#[new]
	#[pyo3(signature = (track_initial_loss = true))]
	pub fn new(track_initial_loss: bool) -> Self {
		Self {
			inner: oa::ml::TrainingSummary::new(track_initial_loss),
		}
	}

	/// Format the complete summary without writing it.
	pub fn render_report(
		&self,
		snapshot: &PythonTrainingSnapshot,
		config: &PythonItTrainingConfig,
	) -> String {
		self.inner.render_report(snapshot.inner(), &config.inner)
	}

	pub fn __repr__(&self) -> &str {
		"TrainingSummary"
	}
}

// ── EarlyStopping ─────────────────────────────────────────────────────────────

/// Cooperative epoch-boundary early stopping.
#[pyclass(name = "EarlyStopping", unsendable)]
pub(crate) struct PythonEarlyStopping {
	pub(crate) inner: oa::ml::EarlyStopping,
}

#[pymethods]
impl PythonEarlyStopping {
	/// Monitor the completed epoch's mean training loss.
	#[new]
	#[pyo3(signature = (patience = 5, min_delta = 1e-4, mode = PythonEarlyStopMode::Min))]
	pub fn new(patience: u64, min_delta: f64, mode: PythonEarlyStopMode) -> PyResult<Self> {
		if patience == 0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"patience must be at least 1",
			));
		}
		oa::ml::EarlyStopping::new(patience, min_delta, mode.into())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	#[getter]
	pub fn should_stop(&self) -> bool {
		self.inner.should_stop()
	}

	#[getter]
	pub fn best(&self) -> f64 {
		self.inner.best()
	}

	#[getter]
	pub fn bad_epochs(&self) -> u64 {
		self.inner.bad_epochs()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EarlyStopping(best={:.6}, bad_epochs={})",
			self.inner.best(),
			self.inner.bad_epochs(),
		)
	}
}

// ── ItRolloutTrainingConfig ───────────────────────────────────────────────────

/// Fixed rollout and update schedule.
#[pyclass(name = "ItRolloutTrainingConfig")]
#[derive(Clone)]
pub(crate) struct PythonItRolloutTrainingConfig {
	pub(crate) inner: oa::ml::ItRolloutTrainingConfig,
}

#[pymethods]
impl PythonItRolloutTrainingConfig {
	#[new]
	#[pyo3(signature = (
		rollouts,
		horizon,
		environments,
		update_epochs,
		timer_name = "rl_update",
		enable_gpu_timing = false,
	))]
	pub fn new(
		rollouts: u64,
		horizon: usize,
		environments: usize,
		update_epochs: u64,
		timer_name: &str,
		enable_gpu_timing: bool,
	) -> Self {
		Self {
			inner: oa::ml::ItRolloutTrainingConfig {
				rollouts,
				horizon,
				environments,
				update_epochs,
				timer_name: timer_name.into(),
				enable_gpu_timing,
			},
		}
	}

	#[getter]
	pub fn rollouts(&self) -> u64 {
		self.inner.rollouts
	}

	#[getter]
	pub fn horizon(&self) -> usize {
		self.inner.horizon
	}

	#[getter]
	pub fn environments(&self) -> usize {
		self.inner.environments
	}

	#[getter]
	pub fn update_epochs(&self) -> u64 {
		self.inner.update_epochs
	}

	#[getter]
	pub fn timer_name(&self) -> &str {
		&self.inner.timer_name
	}

	#[getter]
	pub fn enable_gpu_timing(&self) -> bool {
		self.inner.enable_gpu_timing
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ItRolloutTrainingConfig(rollouts={}, horizon={}, envs={}, update_epochs={})",
			self.inner.rollouts, self.inner.horizon, self.inner.environments, self.inner.update_epochs,
		)
	}
}

// ── CsvLogger ─────────────────────────────────────────────────────────────────

/// Buffered one-row-per-completed-step CSV training logger.
///
/// Pass to `add_callback` on any trainer. The file is created when training
/// begins and flushed/closed at `on_train_end`.
#[pyclass(name = "CsvLogger", unsendable)]
pub(crate) struct PythonCsvLogger {
	pub(crate) inner: oa::ml::CsvLogger,
}

#[pymethods]
impl PythonCsvLogger {
	/// Construct a logger that will write (and truncate) `path` on training begin.
	#[new]
	pub fn new(path: &str) -> Self {
		Self {
			inner: oa::ml::CsvLogger::new(path),
		}
	}

	/// Return the configured output path.
	pub fn path(&self) -> String {
		self.inner.path().to_string_lossy().into_owned()
	}

	pub fn __repr__(&self) -> String {
		format!("CsvLogger(path={:?})", self.inner.path())
	}
}

// ── ValidationResult ──────────────────────────────────────────────────────────

/// Aggregate returned by one application-owned validation pass.
#[pyclass(name = "ValidationResult")]
#[derive(Clone, Copy, Default)]
pub(crate) struct PythonValidationResult {
	pub(crate) inner: oa::ml::ValidationResult,
}

#[pymethods]
impl PythonValidationResult {
	/// Construct from raw validation aggregates.
	#[new]
	#[pyo3(signature = (loss = f64::NAN, batches = 0, samples = 0))]
	pub fn new(loss: f64, batches: u64, samples: u64) -> Self {
		Self {
			inner: oa::ml::ValidationResult {
				loss,
				batches,
				samples,
			},
		}
	}

	#[getter]
	pub fn loss(&self) -> f64 {
		self.inner.loss
	}

	#[getter]
	pub fn batches(&self) -> u64 {
		self.inner.batches
	}

	#[getter]
	pub fn samples(&self) -> u64 {
		self.inner.samples
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ValidationResult(loss={:.6}, batches={}, samples={})",
			self.inner.loss, self.inner.batches, self.inner.samples,
		)
	}
}

// ── ValidationMetric ──────────────────────────────────────────────────────────

/// Cloneable observation handle for the latest completed validation loss.
///
/// Clone this before passing it to `Checkpoint` or `EarlyStopping` so
/// they share the same live value without running validation again.
#[pyclass(name = "ValidationMetric", unsendable)]
#[derive(Clone)]
pub(crate) struct PythonValidationMetric {
	pub(crate) inner: oa::ml::ValidationMetric,
}

#[pymethods]
impl PythonValidationMetric {
	/// Return the stable display name.
	#[getter]
	pub fn name(&self) -> &str {
		self.inner.name()
	}

	/// Return the latest finite validation loss, or `None` before a valid result.
	pub fn value(&self) -> Option<f64> {
		self.inner.value()
	}

	/// Return the latest validation loss, or `NaN` before a valid result.
	pub fn result(&self) -> f64 {
		self.inner.result()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ValidationMetric(name={:?}, value={:?})",
			self.inner.name(),
			self.inner.value(),
		)
	}
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Runs application-owned inference validation at completed lifecycle boundaries.
///
/// The `evaluate` callable receives a `TrainingSnapshot` and must return a
/// `ValidationResult`. Pass the `.metric()` handle to `Checkpoint` or
/// `EarlyStopping` before adding those callbacks.
#[pyclass(name = "Validation", unsendable)]
pub(crate) struct PythonValidation {
	// Validation<'eval> takes Box<dyn FnMut(...) + 'eval>.
	// We satisfy 'eval = 'static by storing a Py<PyAny> and acquiring the GIL.
	inner: oa::ml::Validation<'static>,
}

#[pymethods]
impl PythonValidation {
	/// Construct with a Python callable `evaluate(snapshot) -> ValidationResult`.
	///
	/// `metric_name` is the name used in logs and downstream callbacks.
	/// `step_interval` controls how often the callable is invoked for step-only
	/// training (0 = only at train end).
	#[new]
	#[pyo3(signature = (evaluate, metric_name = "val_loss", step_interval = 0))]
	pub fn new(
		py: Python<'_>,
		evaluate: Py<PyAny>,
		metric_name: &str,
		step_interval: u64,
	) -> PyResult<Self> {
		// Validate that evaluate is callable before handing it to Rust.
		if !evaluate.bind(py).is_callable() {
			return Err(pyo3::exceptions::PyTypeError::new_err(
				"evaluate must be a callable (got non-callable object)",
			));
		}
		let inner = oa::ml::Validation::new(
			move |snapshot: oa::ml::TrainingSnapshot| {
				Python::attach(|py| {
					let snap = PythonTrainingSnapshot::from_inner(snapshot);
					let result = evaluate
						.bind(py)
						.call1((snap,))
						.map_err(|e| oa::Error::callback(format!("validation callable raised: {e}")))?;
					let vr = result
						.extract::<PyRef<PythonValidationResult>>()
						.map_err(|e| {
							oa::Error::callback(format!("validation must return ValidationResult: {e}"))
						})?;
					Ok(vr.inner)
				})
			},
			metric_name,
			step_interval,
		)
		.map_err(python_error)?;
		Ok(Self { inner })
	}

	/// Clone the shared metric observation handle.
	pub fn metric(&self) -> PythonValidationMetric {
		PythonValidationMetric {
			inner: self.inner.metric(),
		}
	}

	/// Return the most recent raw evaluation result.
	pub fn last_result(&self) -> PythonValidationResult {
		PythonValidationResult {
			inner: self.inner.last_result(),
		}
	}

	pub fn __repr__(&self) -> &str {
		"Validation"
	}
}

// ── LearningRateScheduler ─────────────────────────────────────────────────────

/// Applies a learning-rate schedule after every completed optimizer step.
///
/// The schedule is borrowed via raw pointer (same pin contract as the trainers).
/// Pass the same scheduler Python object you constructed — it must outlive this
/// `LearningRateScheduler` object.
#[pyclass(name = "LearningRateScheduler", unsendable)]
pub(crate) struct PythonLearningRateScheduler {
	// Owns the schedule as a Py<PyAny> to keep it alive, and holds a static
	// borrow through a Box<dyn LrScheduler> adapter.
	_schedule: Py<PyAny>,
	_boxed: Box<PyLrSchedulerAdapter>,
	// Stored to keep the `LrScheduler` alive; not read directly.
	#[allow(dead_code)]
	inner: oa::ml::LearningRateScheduler<'static>,
}

struct PyLrSchedulerAdapter {
	obj: Py<PyAny>,
}

impl oa::ml::LrScheduler for PyLrSchedulerAdapter {
	fn learning_rate(&self, step: u64) -> f32 {
		Python::attach(|py| {
			self
				.obj
				.bind(py)
				.call_method1("learning_rate", (step,))
				.and_then(|v| v.extract::<f32>())
				.unwrap_or(0.0)
		})
	}
}

#[pymethods]
impl PythonLearningRateScheduler {
	/// Wrap any scheduler object that has a `learning_rate(step) -> float` method.
	#[new]
	pub fn new(py: Python<'_>, schedule: Py<PyAny>) -> Self {
		let boxed = Box::new(PyLrSchedulerAdapter {
			obj: schedule.clone_ref(py),
		});
		// SAFETY: boxed lives in _boxed which is in the same struct, dropped after inner.
		let sched_ref: &'static dyn oa::ml::LrScheduler =
			unsafe { &*(&*boxed as *const dyn oa::ml::LrScheduler) };
		let inner = oa::ml::LearningRateScheduler::new(sched_ref);
		Self {
			_schedule: schedule,
			_boxed: boxed,
			inner,
		}
	}

	pub fn __repr__(&self) -> &str {
		"LearningRateScheduler"
	}
}

// ── TrainingPhase ─────────────────────────────────────────────────────────────

/// One consecutive epoch range in a `PhaseSchedule`.
#[pyclass(name = "TrainingPhase")]
#[derive(Clone)]
pub(crate) struct PythonTrainingPhase {
	pub(crate) inner: oa::ml::TrainingPhase,
}

#[pymethods]
impl PythonTrainingPhase {
	/// Construct a named phase with aggregate epoch and step counts.
	#[new]
	pub fn new(id: &str, epochs: u64, steps: u64) -> PyResult<Self> {
		oa::ml::TrainingPhase::new(id, epochs, steps)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	#[getter]
	pub fn id(&self) -> &str {
		self.inner.id()
	}

	#[getter]
	pub fn epochs(&self) -> u64 {
		self.inner.epochs()
	}

	#[getter]
	pub fn steps(&self) -> u64 {
		self.inner.steps()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TrainingPhase(id={:?}, epochs={}, steps={})",
			self.inner.id(),
			self.inner.epochs(),
			self.inner.steps(),
		)
	}
}

// ── PhaseSchedule ─────────────────────────────────────────────────────────────

/// Consecutive phase policy over one `ItTraining` lifecycle.
///
/// The optional `on_phase_begin` callable receives `(phase_index: int,
/// phase: TrainingPhase)` and may update the optimizer's learning rate etc.
/// No `Optimizer` borrow is passed through Python — use `ReduceOnPlateau` or
/// `LearningRateScheduler` for programmatic LR changes.
#[pyclass(name = "PhaseSchedule", unsendable)]
pub(crate) struct PythonPhaseSchedule {
	// PhaseSchedule<'hook> can hold a mutable FnMut hook with 'hook lifetime.
	// We satisfy 'hook = 'static via a Py<PyAny>-based closure.
	inner: oa::ml::PhaseSchedule<'static>,
}

#[pymethods]
impl PythonPhaseSchedule {
	/// Construct an empty phase schedule.
	#[new]
	pub fn new() -> Self {
		Self {
			inner: oa::ml::PhaseSchedule::new(),
		}
	}

	/// Append a consecutive phase by id, epoch count, and aggregate step count.
	pub fn add_phase(&mut self, id: &str, epochs: u64, steps: u64) -> PyResult<()> {
		self
			.inner
			.add_phase(id, epochs, steps)
			.map_err(python_error)
	}

	/// Install a hook called once at the start of each phase.
	///
	/// The callable receives `(phase_index: int, phase: TrainingPhase)`.
	pub fn set_on_phase_begin(&mut self, hook: Py<PyAny>) {
		self
			.inner
			.set_on_phase_begin(move |index, phase, _optimizer| {
				Python::attach(|py| {
					let py_phase = PythonTrainingPhase {
						inner: phase.clone(),
					};
					hook
						.bind(py)
						.call1((index, py_phase))
						.map(|_| ())
						.map_err(|e| oa::Error::callback(format!("on_phase_begin raised: {e}")))
				})
			});
	}

	/// Return all configured phases in execution order.
	pub fn phases(&self) -> Vec<PythonTrainingPhase> {
		self
			.inner
			.phases()
			.iter()
			.map(|p| PythonTrainingPhase { inner: p.clone() })
			.collect()
	}

	/// Return the currently entered phase index, or `None`.
	pub fn current_phase(&self) -> Option<usize> {
		self.inner.current_phase()
	}

	/// Return the aggregate epoch count.
	pub fn total_epochs(&self) -> u64 {
		self.inner.total_epochs()
	}

	/// Return the aggregate step count.
	pub fn total_steps(&self) -> u64 {
		self.inner.total_steps()
	}

	pub fn __repr__(&self) -> String {
		format!("PhaseSchedule(phases={})", self.inner.phases().len())
	}
}

// ── SequentialScheduler ───────────────────────────────────────────────────────

/// Chain of schedulers separated by step milestones.
///
/// `schedulers` is a list of any recognized scheduler objects.
/// `milestones` is a list of `len(schedulers) - 1` step boundaries.
#[pyclass(name = "SequentialScheduler", unsendable)]
pub(crate) struct PythonSequentialScheduler {
	inner: oa::ml::SequentialScheduler,
}

fn extract_lr_scheduler(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn oa::ml::LrScheduler>> {
	// All scheduler types (both native and generic Python) are wrapped via the
	// PyLrSchedulerAdapter which calls `learning_rate(step)` on the Python object.
	// Native scheduler types are not Clone, so we cannot move them out; delegating
	// through the Python layer is the only safe path for boxing them.
	let adapter = PyLrSchedulerAdapter {
		obj: obj.clone().unbind(),
	};
	Ok(Box::new(adapter) as Box<dyn oa::ml::LrScheduler>)
}

#[pymethods]
impl PythonSequentialScheduler {
	/// Chain `schedulers` with the given step `milestones`.
	///
	/// `schedulers` may be any mix of recognized scheduler types or Python objects
	/// with a `learning_rate(step) -> float` method.
	/// `milestones` must have exactly `len(schedulers) - 1` entries, positive and
	/// strictly increasing.
	#[new]
	pub fn new(schedulers: Vec<Bound<'_, PyAny>>, milestones: Vec<u64>) -> PyResult<Self> {
		let boxed: Vec<Box<dyn oa::ml::LrScheduler>> = schedulers
			.iter()
			.map(|s| extract_lr_scheduler(s))
			.collect::<PyResult<_>>()?;
		oa::ml::SequentialScheduler::new(boxed, milestones)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	pub fn learning_rate(&self, step: u64) -> f32 {
		use oa::ml::LrScheduler as _;
		self.inner.learning_rate(step)
	}

	pub fn __repr__(&self) -> &str {
		"SequentialScheduler"
	}
}

// ── register ──────────────────────────────────────────────────────────────────

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRolloutTrainingPhase>()?;
	module.add_class::<PythonGpuTimingStats>()?;
	module.add_class::<PythonTrainingSnapshot>()?;
	module.add_class::<PythonItTrainingConfig>()?;
	module.add_class::<PythonLossAggregation>()?;
	module.add_class::<PythonLossMetric>()?;
	module.add_class::<PythonPpoTrainerConfig>()?;
	module.add_class::<PythonPpoTrainerMetrics>()?;
	module.add_class::<PythonDqnTrainerConfig>()?;
	module.add_class::<PythonDqnTrainerMetrics>()?;
	module.add_class::<PythonSacTrainerConfig>()?;
	module.add_class::<PythonSacTrainerMetrics>()?;
	module.add_class::<PythonRolloutCollectorConfig>()?;
	module.add_class::<PythonRolloutCollectorMetrics>()?;
	module.add_class::<PythonPolicyEvaluationConfig>()?;
	module.add_class::<PythonPolicyEvaluationMetrics>()?;
	module.add_class::<PythonEarlyStopMode>()?;
	module.add_class::<PythonProgressBar>()?;
	module.add_class::<PythonTrainingSummary>()?;
	module.add_class::<PythonEarlyStopping>()?;
	module.add_class::<PythonItRolloutTrainingConfig>()?;
	module.add_class::<PythonCsvLogger>()?;
	module.add_class::<PythonValidationResult>()?;
	module.add_class::<PythonValidationMetric>()?;
	module.add_class::<PythonValidation>()?;
	module.add_class::<PythonLearningRateScheduler>()?;
	module.add_class::<PythonTrainingPhase>()?;
	module.add_class::<PythonPhaseSchedule>()?;
	module.add_class::<PythonSequentialScheduler>()?;
	Ok(())
}
