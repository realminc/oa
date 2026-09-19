//! Donor-backed training-control and optimizer-policy callbacks.
//!
//! Port provenance: `oa/ml/callbacks.h`.

use crate::{Error, Result};

use super::super::{
	LrScheduler, TrainingCallback, TrainingCallbackContext, TrainingControl, TrainingSnapshot,
};
use super::validation::ValidationMetric;

/// Improvement direction for [`EarlyStopping`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EarlyStopMode {
	/// A lower monitored value is better.
	#[default]
	Min,
	/// A higher monitored value is better.
	Max,
}

/// Cooperative epoch-boundary early stopping.
pub struct EarlyStopping {
	patience: u64,
	min_delta: f64,
	mode: EarlyStopMode,
	metric_name: String,
	monitor: Box<dyn Fn(TrainingSnapshot) -> f64>,
	best: f64,
	bad_epochs: u64,
	stop: bool,
}

impl EarlyStopping {
	/// Monitor the completed epoch's mean training loss.
	///
	/// # Errors
	///
	/// Returns an error when `min_delta` is not finite and non-negative.
	pub fn new(patience: u64, min_delta: f64, mode: EarlyStopMode) -> Result<Self> {
		Self::with_monitor(patience, min_delta, mode, "loss", |snapshot| {
			snapshot.epoch_mean_loss()
		})
	}

	/// Monitor an application-owned metric sampled from the completed snapshot.
	///
	/// The closure can capture shared validation state; it is invoked only at an
	/// epoch boundary and never owns or re-enters the training loop.
	///
	/// # Errors
	///
	/// Returns an error when the metric name is empty or `min_delta` is not finite
	/// and non-negative.
	pub fn with_monitor(
		patience: u64,
		min_delta: f64,
		mode: EarlyStopMode,
		metric_name: impl Into<String>,
		monitor: impl Fn(TrainingSnapshot) -> f64 + 'static,
	) -> Result<Self> {
		if !min_delta.is_finite() || min_delta < 0.0 {
			return Err(Error::invalid_argument(
				"early-stop minimum delta must be finite and non-negative",
			));
		}
		let metric_name = metric_name.into();
		if metric_name.is_empty() {
			return Err(Error::invalid_argument(
				"early-stop metric name must not be empty",
			));
		}
		Ok(Self {
			patience,
			min_delta,
			mode,
			metric_name,
			monitor: Box::new(monitor),
			best: match mode {
				EarlyStopMode::Min => f64::INFINITY,
				EarlyStopMode::Max => f64::NEG_INFINITY,
			},
			bad_epochs: 0,
			stop: false,
		})
	}

	/// Monitor the result produced by a preceding [`Validation`](super::super::Validation)
	/// callback.
	///
	/// Register validation before this callback. A missing or non-finite
	/// validation result fails the lifecycle instead of silently acting as a
	/// non-improving epoch.
	///
	/// # Errors
	///
	/// Returns an error when `min_delta` is not finite and non-negative.
	pub fn with_validation_metric(
		patience: u64,
		min_delta: f64,
		mode: EarlyStopMode,
		metric: ValidationMetric,
	) -> Result<Self> {
		let name = metric.name().to_owned();
		Self::with_monitor(patience, min_delta, mode, name, move |_| metric.result())
	}

	/// Return whether this callback has requested termination.
	pub const fn should_stop(&self) -> bool {
		self.stop
	}

	/// Return the best monitored value observed so far.
	pub const fn best(&self) -> f64 {
		self.best
	}

	/// Return consecutive non-improving epochs since the last improvement.
	pub const fn bad_epochs(&self) -> u64 {
		self.bad_epochs
	}
}

impl Default for EarlyStopping {
	fn default() -> Self {
		Self::new(5, 1.0e-4, EarlyStopMode::Min).expect("the constant OA early-stop defaults are valid")
	}
}

impl TrainingCallback for EarlyStopping {
	fn on_epoch_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		let metric = (self.monitor)(snapshot);
		if !metric.is_finite() {
			return Err(Error::callback(format!(
				"early-stop metric {} is not finite",
				self.metric_name
			)));
		}
		let improved = match self.mode {
			EarlyStopMode::Min => metric < self.best - self.min_delta,
			EarlyStopMode::Max => metric > self.best + self.min_delta,
		};
		if improved {
			self.best = metric;
			self.bad_epochs = 0;
			return Ok(TrainingControl::Continue);
		}
		self.bad_epochs = self.bad_epochs.saturating_add(1);
		if self.bad_epochs >= self.patience {
			self.stop = true;
			crate::log_info!(
				crate::LogComponent::ML,
				"epoch {}: early stopping; {} did not improve for {} epochs (best {:.6})",
				snapshot.epoch(),
				self.metric_name,
				self.bad_epochs,
				self.best
			);
			Ok(TrainingControl::Stop)
		} else {
			Ok(TrainingControl::Continue)
		}
	}
}

/// Applies a learning-rate schedule after every completed optimizer step.
pub struct LearningRateScheduler<'schedule> {
	schedule: &'schedule dyn LrScheduler,
}

impl<'schedule> LearningRateScheduler<'schedule> {
	/// Borrow a schedule for the callback's lifetime.
	pub const fn new(schedule: &'schedule dyn LrScheduler) -> Self {
		Self { schedule }
	}
}

impl TrainingCallback for LearningRateScheduler<'_> {
	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let next_step = context.snapshot().step_count().saturating_add(1);
		let learning_rate = self.schedule.learning_rate(next_step);
		context.optimizer().set_learning_rate(learning_rate)?;
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`EarlyStopping`].
pub type CbEarlyStop = EarlyStopping;

/// OA C++ compatibility spelling for [`LearningRateScheduler`].
pub type CbLrScheduler<'schedule> = LearningRateScheduler<'schedule>;
