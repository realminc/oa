//! Donor-backed training iteration, metric, and callback lifecycle.
//!
//! Port provenance: `oa/ml/itTraining.h`, `oa/ml/itTraining.cpp`, and
//! `oa/ml/metric.{h,cpp}`. Rust callbacks return control explicitly and are
//! borrowed by the loop instead of inheriting from the C++ callback root.

use std::time::{Duration, Instant};

use crate::{DType, Engine, Error, Matrix, Result};

use super::{AdamW, TrainingProgram, training_program::TrainingProgramCapture};

/// Configuration for one explicit training loop.
#[derive(Clone, Debug)]
pub struct TrainingLoopConfig {
	/// Completed-step budget; zero keeps the loop open-ended.
	pub total_steps: u64,
	/// Fixed steps per epoch; zero disables epoch boundaries.
	pub steps_per_epoch: u64,
	/// Variable epoch lengths. When nonempty, these replace `steps_per_epoch`
	/// and their sum becomes `total_steps`.
	pub epoch_steps: Vec<u64>,
	/// Samples completed by one step. Zero is normalized to one.
	pub batch_size: u64,
	/// Optional sequence work per sample for throughput accounting.
	pub sequence_length: u64,
	/// Optional fixed source units consumed by one sample.
	pub source_units_per_sample: f64,
	/// Collect exact device timestamps around submitted training work.
	pub enable_gpu_timing: bool,
}

impl Default for TrainingLoopConfig {
	fn default() -> Self {
		Self {
			total_steps: 0,
			steps_per_epoch: 0,
			epoch_steps: Vec::new(),
			batch_size: 1,
			sequence_length: 0,
			source_units_per_sample: 0.0,
			enable_gpu_timing: false,
		}
	}
}

/// Explicit callback decision at a lifecycle boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainingControl {
	/// Continue the configured lifecycle.
	#[default]
	Continue,
	/// Stop before admitting another step.
	Stop,
}

/// Immutable state supplied to metrics and callbacks.
#[derive(Clone, Copy, Debug)]
pub struct TrainingSnapshot {
	step_count: u64,
	total_steps: u64,
	epoch: u64,
	total_epochs: u64,
	step_in_epoch: u64,
	steps_in_epoch: u64,
	epoch_boundary: bool,
	last_step: bool,
	last_loss: Option<f32>,
	last_gpu_time: Option<Duration>,
	total_samples: u64,
	total_units: u64,
	total_source_units: u64,
	elapsed: Duration,
	epoch_elapsed: Duration,
	training_mean_loss: f64,
	epoch_mean_loss: f64,
}

impl TrainingSnapshot {
	pub const fn step_count(self) -> u64 {
		self.step_count
	}
	pub const fn total_steps(self) -> u64 {
		self.total_steps
	}
	pub const fn epoch(self) -> u64 {
		self.epoch
	}
	pub const fn total_epochs(self) -> u64 {
		self.total_epochs
	}
	pub const fn step_in_epoch(self) -> u64 {
		self.step_in_epoch
	}
	pub const fn steps_in_epoch(self) -> u64 {
		self.steps_in_epoch
	}
	pub const fn is_epoch_boundary(self) -> bool {
		self.epoch_boundary
	}
	pub const fn is_last_step(self) -> bool {
		self.last_step
	}
	pub const fn last_loss(self) -> Option<f32> {
		self.last_loss
	}
	pub const fn last_gpu_time(self) -> Option<Duration> {
		self.last_gpu_time
	}
	pub const fn total_samples(self) -> u64 {
		self.total_samples
	}
	pub const fn total_units(self) -> u64 {
		self.total_units
	}
	pub const fn total_source_units(self) -> u64 {
		self.total_source_units
	}
	pub const fn elapsed(self) -> Duration {
		self.elapsed
	}
	pub const fn epoch_elapsed(self) -> Duration {
		self.epoch_elapsed
	}
	pub const fn training_mean_loss(self) -> f64 {
		self.training_mean_loss
	}
	pub const fn epoch_mean_loss(self) -> f64 {
		self.epoch_mean_loss
	}

	/// Return completed samples per wall-clock second.
	pub fn wall_samples_per_second(self) -> f64 {
		let seconds = self.elapsed.as_secs_f64();
		if seconds > 0.0 {
			self.total_samples as f64 / seconds
		} else {
			0.0
		}
	}

	/// Return completed sequence units per wall-clock second.
	pub fn wall_units_per_second(self) -> f64 {
		let seconds = self.elapsed.as_secs_f64();
		if seconds > 0.0 {
			self.total_units as f64 / seconds
		} else {
			0.0
		}
	}

	/// Return completed source units per wall-clock second.
	pub fn wall_source_units_per_second(self) -> f64 {
		let seconds = self.elapsed.as_secs_f64();
		if seconds > 0.0 {
			self.total_source_units as f64 / seconds
		} else {
			0.0
		}
	}
}

/// Restricted mutable services available to one callback invocation.
pub struct TrainingCallbackContext<'a> {
	snapshot: TrainingSnapshot,
	optimizer: &'a mut AdamW,
	excluded_wall_time: Duration,
}

impl TrainingCallbackContext<'_> {
	/// Return the immutable state at this lifecycle boundary.
	pub const fn snapshot(&self) -> TrainingSnapshot {
		self.snapshot
	}

	/// Borrow the training optimizer for scheduling or checkpoint policy.
	pub fn optimizer(&mut self) -> &mut AdamW {
		self.optimizer
	}

	/// Exclude callback-owned work such as validation from throughput timing.
	pub fn exclude_wall_time(&mut self, duration: Duration) {
		self.excluded_wall_time = self.excluded_wall_time.saturating_add(duration);
	}
}

/// Stateful policy invoked in deterministic registration order.
pub trait TrainingCallback {
	fn on_train_begin(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_begin(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		Ok(TrainingControl::Continue)
	}

	fn on_step_end(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_end(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		Ok(TrainingControl::Continue)
	}
}

/// Stateful metric updated once after each completed step.
pub trait TrainingMetric {
	fn name(&self) -> &str;
	fn reset(&mut self);
	fn update_step(&mut self, state: TrainingSnapshot);
	fn result(&self) -> f64;
}

/// Aggregation policy for [`LossMetric`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LossAggregation {
	#[default]
	Mean,
	Last,
}

/// Running completed-step loss metric.
pub struct LossMetric {
	name: String,
	aggregation: LossAggregation,
	sum: f64,
	last: f64,
	count: u64,
}

impl LossMetric {
	pub fn new(name: impl Into<String>, aggregation: LossAggregation) -> Self {
		Self {
			name: name.into(),
			aggregation,
			sum: 0.0,
			last: 0.0,
			count: 0,
		}
	}

	pub const fn count(&self) -> u64 {
		self.count
	}
	pub fn mean(&self) -> f64 {
		if self.count == 0 {
			0.0
		} else {
			self.sum / self.count as f64
		}
	}
	pub fn last(&self) -> f64 {
		if self.count == 0 { 0.0 } else { self.last }
	}
}

impl Default for LossMetric {
	fn default() -> Self {
		Self::new("loss", LossAggregation::Mean)
	}
}

impl TrainingMetric for LossMetric {
	fn name(&self) -> &str {
		&self.name
	}
	fn reset(&mut self) {
		self.sum = 0.0;
		self.last = 0.0;
		self.count = 0;
	}
	fn update_step(&mut self, state: TrainingSnapshot) {
		if let Some(loss) = state.last_loss() {
			self.sum += f64::from(loss);
			self.last = f64::from(loss);
			self.count = self.count.saturating_add(1);
		}
	}
	fn result(&self) -> f64 {
		match self.aggregation {
			LossAggregation::Mean => self.mean(),
			LossAggregation::Last => self.last(),
		}
	}
}

#[derive(Clone, Copy)]
enum CallbackPhase {
	TrainBegin,
	EpochBegin,
	StepEnd,
	EpochEnd,
	TrainEnd,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExecutionMode {
	Eager,
	Captured,
	Automatic,
}

/// Exact eager or captured training-step lifecycle.
///
/// The loop borrows the engine, optimizer, metrics, and callbacks. Call
/// [`TrainingLoop::begin_step`] before authoring each body, then finish it with
/// [`TrainingLoop::complete_step`] or [`TrainingLoop::complete_program_step`].
/// [`TrainingLoop::finish`] is the explicit fallible terminal boundary.
#[must_use]
pub struct TrainingLoop<'engine, 'hooks> {
	engine: &'engine Engine,
	optimizer: &'engine mut AdamW,
	config: TrainingLoopConfig,
	epoch_offsets: Vec<u64>,
	metrics: Vec<&'hooks mut dyn TrainingMetric>,
	callbacks: Vec<&'hooks mut dyn TrainingCallback>,
	started: bool,
	body_pending: bool,
	stop_requested: bool,
	execution_mode: Option<ExecutionMode>,
	program: Option<TrainingProgram>,
	program_capture_disabled: bool,
	program_capture_fallback_count: u64,
	last_program_capture_fallback: Option<String>,
	stable_resource_frame_open: bool,
	stable_resource_inputs_sealed: bool,
	step_count: u64,
	last_epoch_started: u64,
	pending_source_units: Option<u64>,
	last_loss: Option<f32>,
	last_gpu_time: Option<Duration>,
	total_samples: u64,
	total_units: u64,
	total_source_units: u64,
	epoch_loss_sum: f64,
	epoch_loss_count: u64,
	training_loss_sum: f64,
	training_loss_count: u64,
	start: Instant,
	epoch_start: Instant,
	excluded_wall: Duration,
	excluded_epoch: Duration,
}

impl<'engine, 'hooks> TrainingLoop<'engine, 'hooks> {
	/// Construct a training lifecycle around one engine and optimizer.
	///
	/// # Errors
	///
	/// Returns an error when source-unit accounting is non-finite or overflows,
	/// or the variable epoch schedule overflows.
	pub fn new(
		engine: &'engine Engine,
		optimizer: &'engine mut AdamW,
		mut config: TrainingLoopConfig,
	) -> Result<Self> {
		if !optimizer.belongs_to(engine) {
			return Err(Error::invalid_argument(
				"training optimizer must belong to the selected engine",
			));
		}
		if !config.source_units_per_sample.is_finite() || config.source_units_per_sample < 0.0 {
			return Err(Error::invalid_argument(
				"source units per sample must be finite and non-negative",
			));
		}
		config.batch_size = config.batch_size.max(1);
		let epoch_offsets = normalize_epoch_schedule(&mut config)?;
		let now = Instant::now();
		Ok(Self {
			engine,
			optimizer,
			config,
			epoch_offsets,
			metrics: Vec::new(),
			callbacks: Vec::new(),
			started: false,
			body_pending: false,
			stop_requested: false,
			execution_mode: None,
			program: None,
			program_capture_disabled: false,
			program_capture_fallback_count: 0,
			last_program_capture_fallback: None,
			stable_resource_frame_open: false,
			stable_resource_inputs_sealed: false,
			step_count: 0,
			last_epoch_started: 0,
			pending_source_units: None,
			last_loss: None,
			last_gpu_time: None,
			total_samples: 0,
			total_units: 0,
			total_source_units: 0,
			epoch_loss_sum: 0.0,
			epoch_loss_count: 0,
			training_loss_sum: 0.0,
			training_loss_count: 0,
			start: now,
			epoch_start: now,
			excluded_wall: Duration::ZERO,
			excluded_epoch: Duration::ZERO,
		})
	}

	pub fn add_metric(&mut self, metric: &'hooks mut dyn TrainingMetric) {
		self.metrics.push(metric);
	}
	pub fn add_callback(&mut self, callback: &'hooks mut dyn TrainingCallback) {
		self.callbacks.push(callback);
	}
	pub fn zero_grad(&self) {
		self.optimizer.zero_grad();
	}
	pub fn request_stop(&mut self) {
		self.stop_requested = true;
	}
	pub const fn stop_requested(&self) -> bool {
		self.stop_requested
	}

	/// Begin the next step, firing lazy train/epoch begin hooks.
	///
	/// Returns `false` after the configured budget or a callback stop decision.
	///
	/// # Errors
	///
	/// Returns a callback failure or counter exhaustion.
	pub fn begin_step(&mut self) -> Result<bool> {
		if self.stop_requested {
			return Ok(false);
		}
		if self.body_pending {
			return Ok(true);
		}
		if !self.started {
			self.started = true;
			let now = Instant::now();
			self.start = now;
			self.epoch_start = now;
			for metric in &mut self.metrics {
				metric.reset();
			}
			if self.fire(CallbackPhase::TrainBegin)? == TrainingControl::Stop {
				return Ok(false);
			}
		}
		if self.config.total_steps > 0 && self.step_count >= self.config.total_steps {
			return Ok(false);
		}
		self.step_count = self
			.step_count
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("training step counter exhausted"))?;
		self.body_pending = true;
		let epoch = self.epoch();
		if self.has_epochs() && epoch != self.last_epoch_started {
			self.last_epoch_started = epoch;
			self.epoch_loss_sum = 0.0;
			self.epoch_loss_count = 0;
			self.epoch_start = Instant::now();
			self.excluded_epoch = Duration::ZERO;
			for metric in &mut self.metrics {
				metric.reset();
			}
			if self.fire(CallbackPhase::EpochBegin)? == TrainingControl::Stop {
				self.body_pending = false;
				return Ok(false);
			}
		}
		if self.step_count > 1
			&& matches!(
				self.execution_mode,
				Some(ExecutionMode::Eager | ExecutionMode::Automatic)
			) {
			if let Err(error) = self.engine.begin_stable_resource_frame() {
				return self.fail(error);
			}
			self.stable_resource_frame_open = true;
		}
		Ok(true)
	}

	/// Complete one eagerly authored forward/backward step.
	///
	/// This records AdamW, submits the complete eager batch, waits for its exact
	/// event, reads the scalar loss, updates metrics, and fires end hooks.
	///
	/// # Errors
	///
	/// Returns an error for an unmatched begin, invalid loss, optimizer/runtime
	/// failure, timestamp failure, metric accounting overflow, or callback failure.
	pub fn complete_step(&mut self, loss: &Matrix) -> Result<()> {
		self.require_pending()?;
		if let Err(error) = self.select_execution_mode(ExecutionMode::Eager) {
			return self.fail(error);
		}
		self.complete_eager_step(loss)
	}

	fn complete_eager_step(&mut self, loss: &Matrix) -> Result<()> {
		if let Err(error) = self.seal_stable_resources() {
			return self.fail(error);
		}
		if !loss.shape().is_empty() || loss.dtype() != DType::F32 {
			return self.fail(Error::invalid_argument(
				"training loss must be an FP32 scalar",
			));
		}
		if !self.engine.owns_matrix(loss) {
			return self.fail(Error::invalid_argument(
				"training loss must belong to the selected engine",
			));
		}
		if let Err(error) = self.optimizer.step() {
			return self.fail(error);
		}
		let event = if self.config.enable_gpu_timing {
			self.engine.checkpoint_timed()
		} else {
			self.engine.checkpoint()
		};
		let event = match event {
			Ok(event) => event,
			Err(error) => return self.fail(error),
		};
		let gpu_time = if self.config.enable_gpu_timing {
			match event.device_duration() {
				Ok(value) => Some(value),
				Err(error) => return self.fail(error),
			}
		} else {
			if let Err(error) = event.wait() {
				return self.fail(error);
			}
			None
		};
		let values = match loss.read_f32() {
			Ok(values) => values,
			Err(error) => return self.fail(error),
		};
		self.complete(values[0], gpu_time)
	}

	/// Execute one automatically captured fixed-shape training step.
	///
	/// `prepare` runs on every admitted step and should allocate or refresh the
	/// fixed-shape input batch. `record` runs for the eager warm-up and once more
	/// to capture forward, backward, and AdamW work; later steps replay the owned
	/// program without rebuilding that graph. The returned boolean is `false`
	/// after the configured budget or a cooperative stop decision.
	///
	/// # Errors
	///
	/// Returns lifecycle, preparation, recording, capture, replay, completion,
	/// metric-accounting, timestamp, or callback errors. A whole-program compile
	/// rejection is an observable optimization fallback: the recorded source step
	/// runs eagerly and later steps remain eager until recapture is requested.
	pub fn step<P>(
		&mut self,
		prepare: impl FnOnce() -> Result<P>,
		record: impl FnOnce(P) -> Result<Matrix>,
	) -> Result<bool> {
		if let Err(error) = self.select_execution_mode(ExecutionMode::Automatic) {
			return self.fail(error);
		}
		if !self.begin_step()? {
			return Ok(false);
		}
		if self.engine.has_pending_work() {
			return self.fail(Error::failed_precondition(
				"automatic training step requires an empty execution session",
			));
		}
		let prepared = match prepare() {
			Ok(prepared) => prepared,
			Err(error) => return self.fail(error),
		};
		if self.engine.has_pending_work() {
			self.engine.abort_pending_work();
			return self.fail(Error::failed_precondition(
				"automatic training preparation may only allocate or refresh inputs",
			));
		}

		if self.step_count == 1 {
			self.optimizer.zero_grad();
			let loss = match record(prepared) {
				Ok(loss) => loss,
				Err(error) => return self.fail(error),
			};
			self.complete_eager_step(&loss)?;
			return Ok(true);
		}

		self.seal_replay_inputs()?;
		if self.program_capture_disabled {
			self.optimizer.zero_grad();
			let loss = match record(prepared) {
				Ok(loss) => loss,
				Err(error) => return self.fail(error),
			};
			self.complete_eager_step(&loss)?;
			return Ok(true);
		}
		if self.program.is_none() {
			let capture =
				TrainingProgram::capture_preserving_recording(self.engine, self.optimizer, || {
					record(prepared)
				});
			match capture {
				Ok(TrainingProgramCapture::Captured(program)) => self.program = Some(*program),
				Ok(TrainingProgramCapture::Rejected {
					error,
					loss,
					optimizer,
				}) => {
					self.complete_capture_fallback(error, loss, &optimizer)?;
					return Ok(true);
				}
				Err(error) => return self.fail(error),
			}
		} else {
			drop(prepared);
		}
		self.complete_automatic_program_step()?;
		Ok(true)
	}

	fn complete_capture_fallback(
		&mut self,
		error: Error,
		loss: Matrix,
		optimizer: &super::optimizer::AdamWProgramSignature,
	) -> Result<()> {
		self.program_capture_disabled = true;
		self.program_capture_fallback_count = self.program_capture_fallback_count.saturating_add(1);
		self.last_program_capture_fallback = Some(error.to_string());
		crate::log_warn!(
			crate::LogComponent::ML,
			"training program capture unavailable at step {} ({}); continuing eagerly",
			self.step_count,
			error
		);
		let event = if self.config.enable_gpu_timing {
			self.engine.checkpoint_timed()
		} else {
			self.engine.checkpoint()
		};
		let event = match event {
			Ok(event) => event,
			Err(error) => return self.fail(error),
		};
		let gpu_time = if self.config.enable_gpu_timing {
			match event.device_duration() {
				Ok(duration) => Some(duration),
				Err(error) => return self.fail(error),
			}
		} else {
			if let Err(error) = event.wait() {
				return self.fail(error);
			}
			None
		};
		if let Err(error) = self.optimizer.complete_capture_fallback(optimizer) {
			return self.fail(error);
		}
		let values = match loss.read_f32() {
			Ok(values) => values,
			Err(error) => return self.fail(error),
		};
		self.complete(values[0], gpu_time)
	}

	fn complete_automatic_program_step(&mut self) -> Result<()> {
		let program = self
			.program
			.as_mut()
			.ok_or_else(|| Error::internal("automatic training program is missing"));
		let program = match program {
			Ok(program) => program,
			Err(error) => return self.fail(error),
		};
		let completed = if self.config.enable_gpu_timing {
			program
				.replay_timed_and_wait(self.engine, self.optimizer)
				.map(|(loss, time)| (loss, Some(time)))
		} else {
			program
				.replay_and_wait(self.engine, self.optimizer)
				.map(|loss| (loss, None))
		};
		match completed {
			Ok((loss, time)) => self.complete(loss, time),
			Err(error) => self.fail(error),
		}
	}

	/// Return the automatically owned program after its capture step.
	pub const fn training_program(&self) -> Option<&TrainingProgram> {
		self.program.as_ref()
	}

	/// Return how many whole-program capture rejections continued eagerly.
	pub const fn program_capture_fallback_count(&self) -> u64 {
		self.program_capture_fallback_count
	}

	/// Return the most recent whole-program capture rejection message.
	pub fn last_program_capture_fallback(&self) -> Option<&str> {
		self.last_program_capture_fallback.as_deref()
	}

	/// Discard the automatically captured program, or re-enable capture after a
	/// prior rejection, so the next step records a new fixed-shape program after
	/// preparation.
	///
	/// # Errors
	///
	/// Returns an error while a step body is pending or when this loop does not use
	/// [`TrainingLoop::step`].
	pub fn request_program_recapture(&mut self) -> Result<()> {
		if self.body_pending {
			return Err(Error::failed_precondition(
				"training program recapture requires a completed step",
			));
		}
		if self.execution_mode != Some(ExecutionMode::Automatic) {
			return Err(Error::failed_precondition(
				"training program recapture requires automatic capture",
			));
		}
		self.program = None;
		self.program_capture_disabled = false;
		Ok(())
	}

	/// Seal allocations authored during preparation as stable replay inputs.
	///
	/// On eager steps after the first warm-up, Matrix storage allocated after
	/// `begin_step` and before this call forms the externally retained input
	/// prefix. Later allocations in the same step are classified as stable
	/// capture-local temporaries with exact executable-graph lifetimes. Calling
	/// this on the warm-up step is a no-op so one loop body can be used unchanged.
	/// If this boundary is omitted, `complete_step` conservatively classifies every
	/// allocation in the frame as external.
	///
	/// # Errors
	///
	/// Returns an error without a pending step or when the boundary is repeated.
	pub fn seal_replay_inputs(&mut self) -> Result<()> {
		self.require_pending()?;
		if self.stable_resource_inputs_sealed {
			return self.fail(Error::failed_precondition(
				"training replay inputs may only be sealed once per step",
			));
		}
		if self.stable_resource_frame_open
			&& let Err(error) = self.engine.seal_stable_resource_inputs()
		{
			return self.fail(error);
		}
		self.stable_resource_inputs_sealed = true;
		Ok(())
	}

	/// Complete one replay of an already captured fixed-shape training program.
	///
	/// # Errors
	///
	/// Returns an unmatched-begin, replay, completion, accounting, or callback error.
	pub fn complete_program_step(&mut self, program: &mut TrainingProgram) -> Result<()> {
		self.require_pending()?;
		if let Err(error) = self.select_execution_mode(ExecutionMode::Captured) {
			return self.fail(error);
		}
		if let Err(error) = self.seal_stable_resources() {
			return self.fail(error);
		}
		let completed = if self.config.enable_gpu_timing {
			program
				.replay_timed_and_wait(self.engine, self.optimizer)
				.map(|(loss, time)| (loss, Some(time)))
		} else {
			program
				.replay_and_wait(self.engine, self.optimizer)
				.map(|loss| (loss, None))
		};
		match completed {
			Ok((loss, time)) => self.complete(loss, time),
			Err(error) => self.fail(error),
		}
	}

	pub fn record_source_units(&mut self, units: u64) {
		self.pending_source_units = Some(units);
	}

	/// Exclude application work such as validation from wall-rate denominators.
	pub fn exclude_wall_time(&mut self, duration: Duration) {
		self.excluded_wall = self.excluded_wall.saturating_add(duration);
		self.excluded_epoch = self.excluded_epoch.saturating_add(duration);
	}

	pub fn snapshot(&self) -> TrainingSnapshot {
		let epoch_loss = if self.epoch_loss_count == 0 {
			0.0
		} else {
			self.epoch_loss_sum / self.epoch_loss_count as f64
		};
		let training_loss = if self.training_loss_count == 0 {
			0.0
		} else {
			self.training_loss_sum / self.training_loss_count as f64
		};
		TrainingSnapshot {
			step_count: self.step_count,
			total_steps: self.config.total_steps,
			epoch: self.epoch(),
			total_epochs: self.total_epochs(),
			step_in_epoch: self.step_in_epoch(),
			steps_in_epoch: self.steps_in_current_epoch(),
			epoch_boundary: self.is_epoch_boundary(),
			last_step: self.is_last_step(),
			last_loss: self.last_loss,
			last_gpu_time: self.last_gpu_time,
			total_samples: self.total_samples,
			total_units: self.total_units,
			total_source_units: self.total_source_units,
			elapsed: self.start.elapsed().saturating_sub(self.excluded_wall),
			epoch_elapsed: self
				.epoch_start
				.elapsed()
				.saturating_sub(self.excluded_epoch),
			training_mean_loss: training_loss,
			epoch_mean_loss: epoch_loss,
		}
	}

	/// Close the lifecycle and fire `on_train_end` exactly once.
	///
	/// # Errors
	///
	/// Returns an error when a begun body was not completed or a callback fails.
	pub fn finish(mut self) -> Result<TrainingSnapshot> {
		if self.body_pending {
			return Err(Error::failed_precondition(
				"cannot finish with an incomplete training step",
			));
		}
		self.fire(CallbackPhase::TrainEnd)?;
		Ok(self.snapshot())
	}

	fn complete(&mut self, loss: f32, gpu_time: Option<Duration>) -> Result<()> {
		let counts = (|| {
			let total_samples = checked_add(
				self.total_samples,
				self.config.batch_size,
				"training sample",
			)?;
			let units = self
				.config
				.batch_size
				.checked_mul(self.config.sequence_length)
				.ok_or_else(|| {
					Error::resource_exhausted("training sequence-unit count overflows u64")
				})?;
			let total_units = checked_add(self.total_units, units, "training sequence-unit")?;
			let source = match self.pending_source_units {
				Some(value) => value,
				None => fixed_source_units(&self.config)?,
			};
			let total_source_units =
				checked_add(self.total_source_units, source, "training source-unit")?;
			Ok((total_samples, total_units, total_source_units))
		})();
		let (total_samples, total_units, total_source_units) = match counts {
			Ok(counts) => counts,
			Err(error) => return self.fail(error),
		};
		self.pending_source_units = None;
		self.last_loss = Some(loss);
		self.last_gpu_time = gpu_time;
		self.epoch_loss_sum += f64::from(loss);
		self.epoch_loss_count = self.epoch_loss_count.saturating_add(1);
		self.training_loss_sum += f64::from(loss);
		self.training_loss_count = self.training_loss_count.saturating_add(1);
		self.total_samples = total_samples;
		self.total_units = total_units;
		self.total_source_units = total_source_units;
		self.close_stable_resource_frame();
		let snapshot = self.snapshot();
		for metric in &mut self.metrics {
			metric.update_step(snapshot);
		}
		if self.fire(CallbackPhase::StepEnd)? == TrainingControl::Stop {
			self.stop_requested = true;
		}
		if self.is_epoch_boundary() && self.fire(CallbackPhase::EpochEnd)? == TrainingControl::Stop
		{
			self.stop_requested = true;
		}
		self.body_pending = false;
		Ok(())
	}

	fn fire(&mut self, phase: CallbackPhase) -> Result<TrainingControl> {
		let snapshot = self.snapshot();
		for callback in &mut self.callbacks {
			let mut context = TrainingCallbackContext {
				snapshot,
				optimizer: &mut *self.optimizer,
				excluded_wall_time: Duration::ZERO,
			};
			let result = match phase {
				CallbackPhase::TrainBegin => callback.on_train_begin(&mut context),
				CallbackPhase::EpochBegin => callback.on_epoch_begin(&mut context),
				CallbackPhase::StepEnd => callback.on_step_end(&mut context),
				CallbackPhase::EpochEnd => callback.on_epoch_end(&mut context),
				CallbackPhase::TrainEnd => callback.on_train_end(&mut context),
			};
			self.excluded_wall = self
				.excluded_wall
				.saturating_add(context.excluded_wall_time);
			self.excluded_epoch = self
				.excluded_epoch
				.saturating_add(context.excluded_wall_time);
			let control = match result {
				Ok(control) => control,
				Err(error) => {
					self.stop_requested = true;
					self.body_pending = false;
					return Err(error);
				}
			};
			if control == TrainingControl::Stop {
				self.stop_requested = true;
				return Ok(control);
			}
		}
		Ok(TrainingControl::Continue)
	}

	fn require_pending(&mut self) -> Result<()> {
		if self.body_pending {
			Ok(())
		} else {
			self.fail(Error::failed_precondition(
				"training step completion requires begin_step",
			))
		}
	}

	fn fail<T>(&mut self, error: Error) -> Result<T> {
		self.close_stable_resource_frame();
		self.stop_requested = true;
		self.body_pending = false;
		Err(error)
	}

	fn seal_stable_resources(&self) -> Result<()> {
		if self.stable_resource_frame_open && !self.stable_resource_inputs_sealed {
			self.engine.seal_all_stable_resources_external()?;
		}
		Ok(())
	}

	fn select_execution_mode(&mut self, mode: ExecutionMode) -> Result<()> {
		match self.execution_mode {
			Some(existing) if existing != mode => Err(Error::failed_precondition(
				"training loop cannot mix eager and captured step completion",
			)),
			Some(_) => Ok(()),
			None => {
				self.execution_mode = Some(mode);
				Ok(())
			}
		}
	}

	fn close_stable_resource_frame(&mut self) {
		if self.stable_resource_frame_open {
			self.engine.end_stable_resource_frame();
			self.stable_resource_frame_open = false;
		}
		self.stable_resource_inputs_sealed = false;
	}

	fn has_epochs(&self) -> bool {
		self.config.steps_per_epoch > 0 || !self.epoch_offsets.is_empty()
	}
	fn epoch(&self) -> u64 {
		if self.step_count == 0 || !self.has_epochs() {
			return 0;
		}
		if !self.epoch_offsets.is_empty() {
			return self
				.epoch_offsets
				.partition_point(|end| *end < self.step_count) as u64
				+ 1;
		}
		(self.step_count - 1) / self.config.steps_per_epoch + 1
	}
	fn total_epochs(&self) -> u64 {
		if !self.epoch_offsets.is_empty() {
			return self.epoch_offsets.len() as u64;
		}
		if self.config.steps_per_epoch == 0 || self.config.total_steps == 0 {
			return 0;
		}
		self.config
			.total_steps
			.div_ceil(self.config.steps_per_epoch)
	}
	fn step_in_epoch(&self) -> u64 {
		if self.step_count == 0 {
			return 0;
		}
		if !self.epoch_offsets.is_empty() {
			let index = self
				.epoch_offsets
				.partition_point(|end| *end < self.step_count);
			let begin = if index == 0 {
				0
			} else {
				self.epoch_offsets[index - 1]
			};
			return self.step_count - begin;
		}
		if self.config.steps_per_epoch == 0 {
			return self.step_count;
		}
		(self.step_count - 1) % self.config.steps_per_epoch + 1
	}
	fn steps_in_current_epoch(&self) -> u64 {
		if !self.config.epoch_steps.is_empty() {
			let index = self
				.epoch_offsets
				.partition_point(|end| *end < self.step_count.max(1));
			return self.config.epoch_steps[index.min(self.config.epoch_steps.len() - 1)];
		}
		if self.config.steps_per_epoch == 0 {
			return self.config.total_steps;
		}
		if self.config.total_steps > 0 && self.epoch() == self.total_epochs() {
			return self.config.total_steps - (self.epoch() - 1) * self.config.steps_per_epoch;
		}
		self.config.steps_per_epoch
	}
	fn is_last_step(&self) -> bool {
		self.config.total_steps > 0 && self.step_count == self.config.total_steps
	}
	fn is_epoch_boundary(&self) -> bool {
		if self.step_count == 0 || !self.has_epochs() {
			return false;
		}
		if self.is_last_step() {
			return true;
		}
		if !self.epoch_offsets.is_empty() {
			return self.epoch_offsets.binary_search(&self.step_count).is_ok();
		}
		self.step_count.is_multiple_of(self.config.steps_per_epoch)
	}
}

impl Drop for TrainingLoop<'_, '_> {
	fn drop(&mut self) {
		self.close_stable_resource_frame();
	}
}

fn checked_add(left: u64, right: u64, name: &'static str) -> Result<u64> {
	left.checked_add(right)
		.ok_or_else(|| Error::resource_exhausted(format!("{name} count overflows u64")))
}

fn fixed_source_units(config: &TrainingLoopConfig) -> Result<u64> {
	let value = config.batch_size as f64 * config.source_units_per_sample;
	if !value.is_finite() || value > u64::MAX as f64 {
		return Err(Error::resource_exhausted(
			"training source-unit count overflows u64",
		));
	}
	Ok(value.round() as u64)
}

fn normalize_epoch_schedule(config: &mut TrainingLoopConfig) -> Result<Vec<u64>> {
	let mut offsets = Vec::with_capacity(config.epoch_steps.len());
	if config.epoch_steps.is_empty() {
		return Ok(offsets);
	}
	let mut sum = 0_u64;
	for steps in &mut config.epoch_steps {
		*steps = (*steps).max(1);
		sum = sum
			.checked_add(*steps)
			.ok_or_else(|| Error::resource_exhausted("training epoch schedule overflows u64"))?;
		offsets.push(sum);
	}
	config.total_steps = sum;
	Ok(offsets)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn loss_metric_counts_each_completed_sample_once() {
		let mut metric = LossMetric::default();
		let mut state = snapshot(1, 1.0);
		metric.update_step(state);
		state.step_count = 2;
		state.last_loss = Some(3.0);
		metric.update_step(state);
		assert_eq!(metric.count(), 2);
		assert_eq!(metric.result(), 2.0);
	}

	#[test]
	fn variable_epoch_schedule_owns_total_and_boundaries() -> Result<()> {
		let mut config = TrainingLoopConfig {
			epoch_steps: vec![2, 0, 3],
			..TrainingLoopConfig::default()
		};
		let offsets = normalize_epoch_schedule(&mut config)?;
		assert_eq!(config.total_steps, 6);
		assert_eq!(config.epoch_steps, [2, 1, 3]);
		assert_eq!(offsets, [2, 3, 6]);
		Ok(())
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn rejected_program_compilation_executes_the_preserved_step_eagerly() -> Result<()> {
		let engine = Engine::new()?;
		let weight = Matrix::from_f32(&engine, [2, 2], &[0.2, -0.1, -0.3, 0.4])?;
		let bias = Matrix::from_f32(&engine, [2], &[0.0, 0.0])?;
		let layer = crate::ml::nn::Linear::from_matrices(weight, bias)?;
		let mut optimizer = AdamW::new(layer.parameters(), 0.01)?;
		let mut prepare_count = 0_u32;
		let mut record_count = 0_u32;
		let mut training = TrainingLoop::new(
			&engine,
			&mut optimizer,
			TrainingLoopConfig {
				total_steps: 4,
				batch_size: 2,
				..TrainingLoopConfig::default()
			},
		)?;
		loop {
			let ran = training.step(
				|| {
					prepare_count += 1;
					let input = Matrix::from_f32(&engine, [2, 2], &[1.0, 0.0, 0.0, 1.0])?;
					let targets = Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
					Ok((input, targets))
				},
				|(input, targets)| {
					record_count += 1;
					let tape = crate::ml::GradientTape::new();
					let logits = layer.forward(&input)?;
					let loss = crate::ml::loss::cross_entropy(&logits, &targets)?;
					tape.backward(&loss)?;
					Ok(loss)
				},
			)?;
			if !ran {
				break;
			}
			if training.snapshot().step_count() == 1 {
				engine.force_next_plan_compilation_failure();
			}
			if training.snapshot().step_count() == 3 {
				training.request_program_recapture()?;
			}
		}

		assert_eq!(prepare_count, 4);
		assert_eq!(record_count, 4);
		assert_eq!(training.program_capture_fallback_count(), 1);
		assert_eq!(
			training.last_program_capture_fallback(),
			Some("forced plan compilation failure")
		);
		let program = training
			.training_program()
			.expect("requested recapture must install a replacement program");
		assert_eq!(program.replay_count(), 1);
		assert_eq!(program.diagnostics().command_recording_count(), 1);
		assert_eq!(program.diagnostics().command_cache_hit_count(), 1);
		assert!(training.snapshot().last_loss().is_some_and(f32::is_finite));
		let summary = training.finish()?;
		assert_eq!(summary.step_count(), 4);
		assert_eq!(optimizer.step_count(), 4);
		Ok(())
	}

	fn snapshot(step: u64, loss: f32) -> TrainingSnapshot {
		TrainingSnapshot {
			step_count: step,
			total_steps: 2,
			epoch: 1,
			total_epochs: 1,
			step_in_epoch: step,
			steps_in_epoch: 2,
			epoch_boundary: step == 2,
			last_step: step == 2,
			last_loss: Some(loss),
			last_gpu_time: None,
			total_samples: step,
			total_units: 0,
			total_source_units: 0,
			elapsed: Duration::ZERO,
			epoch_elapsed: Duration::ZERO,
			training_mean_loss: f64::from(loss),
			epoch_mean_loss: f64::from(loss),
		}
	}
}
