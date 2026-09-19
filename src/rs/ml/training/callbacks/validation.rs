//! Completed-boundary validation callback.
//!
//! Port provenance: `oa/ml/callbacks.h` (`CbValidation`).

use std::{
	cell::Cell,
	rc::Rc,
	time::{Duration, Instant},
};

use crate::{Error, Result};

use super::super::{TrainingCallback, TrainingCallbackContext, TrainingControl, TrainingSnapshot};
use super::write_stdout;

/// Aggregate returned by one application-owned validation pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidationResult {
	/// Sample-weighted mean validation loss, or NaN when no value is available.
	pub loss: f64,
	/// Number of validation batches consumed.
	pub batches: u64,
	/// Number of validation samples consumed.
	pub samples: u64,
}

impl Default for ValidationResult {
	fn default() -> Self {
		Self {
			loss: f64::NAN,
			batches: 0,
			samples: 0,
		}
	}
}

/// Cloneable observation handle for the latest completed validation loss.
///
/// Clones share one thread-affine value so checkpoint, summary, and early-stop
/// callbacks can observe the result without owning or rerunning validation.
#[derive(Clone)]
pub struct ValidationMetric {
	name: Rc<str>,
	value: Rc<Cell<Option<f64>>>,
}

impl ValidationMetric {
	fn new(name: String) -> Self {
		Self {
			name: Rc::from(name),
			value: Rc::new(Cell::new(None)),
		}
	}

	/// Return the stable display name.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Return the latest finite validation loss.
	pub fn value(&self) -> Option<f64> {
		self.value.get()
	}

	/// Return the latest validation loss, or NaN before a valid evaluation.
	pub fn result(&self) -> f64 {
		self.value().unwrap_or(f64::NAN)
	}

	fn replace(&self, value: Option<f64>) {
		self.value.set(value);
	}
}

/// Runs application-owned inference validation at completed lifecycle boundaries.
///
/// Epoch-based training evaluates once at every epoch end. Step-only training
/// evaluates at each non-zero configured interval and once at train end when
/// the final step was not already evaluated. Evaluation wall time is excluded
/// from training throughput and the callback never submits work on its own.
pub struct Validation<'eval> {
	evaluate: Box<dyn FnMut(TrainingSnapshot) -> Result<ValidationResult> + 'eval>,
	metric: ValidationMetric,
	step_interval: u64,
	last_result: ValidationResult,
	last_duration: Duration,
	last_eval_step: Option<u64>,
}

impl<'eval> Validation<'eval> {
	/// Construct a validation callback.
	///
	/// `step_interval` applies only to step-only training. Zero disables periodic
	/// evaluation while retaining the final train-end evaluation.
	///
	/// # Errors
	///
	/// Returns an error when `metric_name` is empty.
	pub fn new(
		evaluate: impl FnMut(TrainingSnapshot) -> Result<ValidationResult> + 'eval,
		metric_name: impl Into<String>,
		step_interval: u64,
	) -> Result<Self> {
		let metric_name = metric_name.into();
		if metric_name.is_empty() {
			return Err(Error::invalid_argument(
				"validation metric name must not be empty",
			));
		}
		Ok(Self {
			evaluate: Box::new(evaluate),
			metric: ValidationMetric::new(metric_name),
			step_interval,
			last_result: ValidationResult::default(),
			last_duration: Duration::ZERO,
			last_eval_step: None,
		})
	}

	/// Clone the shared metric observation handle.
	pub fn metric(&self) -> ValidationMetric {
		self.metric.clone()
	}

	/// Return the most recent raw evaluation result.
	pub const fn last_result(&self) -> ValidationResult {
		self.last_result
	}

	/// Return the wall duration of the most recent evaluation.
	pub const fn last_duration(&self) -> Duration {
		self.last_duration
	}

	fn run(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<()> {
		let snapshot = context.snapshot();
		let begin = Instant::now();
		let result = (self.evaluate)(snapshot);
		let duration = begin.elapsed();
		context.exclude_wall_time(duration);
		let result = result?;

		self.last_result = result;
		self.last_duration = duration;
		self.last_eval_step = Some(snapshot.step_count());
		let value = (result.loss.is_finite() && result.batches > 0).then_some(result.loss);
		self.metric.replace(value);

		if let Some(loss) = value {
			write_stdout(&format!(
				"Validation: {} {:.6} · {} batches · {} samples · {:.2}s\n",
				self.metric.name(),
				loss,
				result.batches,
				result.samples,
				duration.as_secs_f64(),
			))
		} else {
			write_stdout(&format!(
				"Validation: {} n/a · {:.2}s\n",
				self.metric.name(),
				duration.as_secs_f64(),
			))
		}
	}
}

impl TrainingCallback for Validation<'_> {
	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		if snapshot.total_epochs() == 0
			&& self.step_interval > 0
			&& snapshot.step_count().is_multiple_of(self.step_interval)
		{
			self.run(context)?;
		}
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		self.run(context)?;
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		if snapshot.total_epochs() == 0 && self.last_eval_step != Some(snapshot.step_count()) {
			self.run(context)?;
		}
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`Validation`].
pub type CbValidation<'eval> = Validation<'eval>;
