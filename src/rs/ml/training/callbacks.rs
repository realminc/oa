//! Donor-backed built-in training presentation callbacks.

use std::{
	io::{self, Write},
	time::{Duration, Instant},
};

use crate::{Error, Result};

use super::{
	ItTrainingConfig, TrainingCallback, TrainingCallbackContext, TrainingControl, TrainingSnapshot,
};

mod checkpoint;
mod csv;
mod phase;
mod policy;
mod validation;

pub use checkpoint::{CbCheckpoint, Checkpoint};
pub use csv::{CbCsvLogger, CsvLogger};
pub use phase::{CbPhase, PhaseSchedule, TrainingPhase};
pub use policy::{CbEarlyStop, CbLrScheduler, EarlyStopMode, EarlyStopping, LearningRateScheduler};
pub use validation::{CbValidation, Validation, ValidationMetric, ValidationResult};

/// Throttled Keras/tqdm-style wall-throughput progress display.
pub struct ProgressBar {
	width: usize,
	show_epoch_header: bool,
	last_print: Option<Instant>,
	last_finalized_step: u64,
}

impl ProgressBar {
	/// Construct a progress bar with the requested cell width.
	pub fn new(width: usize) -> Self {
		Self {
			width: width.max(1),
			show_epoch_header: true,
			last_print: None,
			last_finalized_step: 0,
		}
	}

	/// Enable or suppress the `epoch N/M` header.
	pub fn set_show_epoch_header(&mut self, show: bool) {
		self.show_epoch_header = show;
	}

	/// Format one deterministic progress line without writing it.
	pub fn render_line(&self, state: TrainingSnapshot, config: &ItTrainingConfig) -> String {
		let has_epochs = state.total_epochs() > 0;
		let step = if has_epochs {
			state.step_in_epoch()
		} else {
			state.step_count()
		};
		let total = if has_epochs {
			state.steps_in_epoch()
		} else {
			state.total_steps()
		};
		let elapsed = if has_epochs {
			state.epoch_elapsed()
		} else {
			state.elapsed()
		};
		let mut line = String::from("\r  ");
		if total > 0 {
			let eighths = ((step as u128 * self.width as u128 * 8 + u128::from(total / 2))
				/ u128::from(total)) as usize;
			let full = (eighths / 8).min(self.width);
			let partial = eighths % 8;
			line.push_str(&format!("{step}/{total} |"));
			line.push_str(&"█".repeat(full));
			if full < self.width && partial > 0 {
				line.push_str(["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"][partial]);
				line.push_str(&"░".repeat(self.width - full - 1));
			} else {
				line.push_str(&"░".repeat(self.width - full));
			}
			line.push_str("| ");
		} else {
			line.push_str(&format!("step {step} "));
		}
		let seconds = elapsed.as_secs_f64();
		let milliseconds = if step > 0 {
			seconds * 1000.0 / step as f64
		} else {
			0.0
		};
		let samples_per_second = if seconds > 0.0 {
			step as f64 * config.batch_size as f64 / seconds
		} else {
			0.0
		};
		line.push_str(&format!(
			"{seconds:.2}s · {milliseconds:.2} ms/step · {} sample/s",
			format_rate(samples_per_second)
		));
		if config.sequence_length > 0 {
			line.push_str(&format!(
				" · {} {}/s",
				format_rate(samples_per_second * config.sequence_length as f64),
				config.sequence_unit
			));
		}
		if state.total_source_units() > 0 {
			let rate = if has_epochs {
				state.epoch_source_units_per_second()
			} else {
				state.wall_source_units_per_second()
			};
			line.push_str(&format!(
				" · {} {}/s",
				format_rate(rate),
				config.source_unit
			));
		}
		if let Some(loss) = state.last_loss() {
			line.push_str(&format!(" · loss: {loss:.4}"));
		}
		line.push_str("        ");
		line
	}

	fn render(&self, context: &TrainingCallbackContext<'_>, newline: bool) -> Result<()> {
		let mut output = self.render_line(context.snapshot(), context.config());
		if newline {
			output.push('\n');
		}
		write_stdout(&output)
	}
}

impl Default for ProgressBar {
	fn default() -> Self {
		Self::new(10)
	}
}

impl TrainingCallback for ProgressBar {
	fn on_train_begin(
		&mut self,
		_context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		self.last_print = Some(Instant::now());
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_begin(
		&mut self,
		context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		let state = context.snapshot();
		if self.show_epoch_header && state.total_epochs() > 0 {
			write_stdout(&format!(
				"epoch {}/{}\n",
				state.epoch(),
				state.total_epochs()
			))?;
		}
		Ok(TrainingControl::Continue)
	}

	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let state = context.snapshot();
		if state.is_epoch_boundary() {
			return Ok(TrainingControl::Continue);
		}
		let now = Instant::now();
		let boundary = state.is_last_step();
		if !boundary
			&& self
				.last_print
				.is_some_and(|last| now.duration_since(last) < Duration::from_millis(33))
		{
			return Ok(TrainingControl::Continue);
		}
		self.last_print = Some(now);
		self.render(context, boundary)?;
		if boundary {
			self.last_finalized_step = state.step_count();
		}
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		self.render(context, true)?;
		self.last_finalized_step = context.snapshot().step_count();
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let state = context.snapshot();
		if self.last_finalized_step != state.step_count() {
			self.render(context, true)?;
		}
		Ok(TrainingControl::Continue)
	}
}

/// Final loss, wall-time, device-time, throughput, and run summary.
pub struct TrainingSummary {
	track_initial_loss: bool,
	initial_loss: Option<f32>,
	validation_metric: Option<ValidationMetric>,
}

impl TrainingSummary {
	/// Construct a summary callback, optionally retaining the first loss.
	pub const fn new(track_initial_loss: bool) -> Self {
		Self {
			track_initial_loss,
			initial_loss: None,
			validation_metric: None,
		}
	}

	/// Display the latest completed validation metric in the final report.
	pub fn set_validation_metric(&mut self, metric: ValidationMetric) {
		self.validation_metric = Some(metric);
	}

	/// Format the complete summary without writing it.
	pub fn render_report(&self, state: TrainingSnapshot, config: &ItTrainingConfig) -> String {
		let mut report = String::from("\nSummary:\n");
		match (self.initial_loss, state.last_loss()) {
			(_, None) => report.push_str("  loss: n/a (no loss recorded)\n"),
			(Some(initial), Some(final_loss)) if self.track_initial_loss => report.push_str(&format!(
				"  loss: initial {initial:.6} · final {final_loss:.6} · mean {:.6}\n",
				state.training_mean_loss()
			)),
			(_, Some(final_loss)) => report.push_str(&format!(
				"  loss: final {final_loss:.6} · mean {:.6}\n",
				state.training_mean_loss()
			)),
		}
		if let Some(metric) = &self.validation_metric
			&& let Some(value) = metric.value()
		{
			report.push_str(&format!("  Validation: {} {value:.6}\n", metric.name()));
		}
		report.push_str(&format!(
			"  Wall: {:.2} ms/step · {} sample/s",
			state.wall_ms_per_step(),
			format_rate(state.wall_samples_per_second())
		));
		if config.sequence_length > 0 {
			report.push_str(&format!(
				" · {} {}/s",
				format_rate(state.wall_units_per_second()),
				config.sequence_unit
			));
		}
		if state.total_source_units() > 0 {
			report.push_str(&format!(
				" · {} {}/s",
				format_rate(state.wall_source_units_per_second()),
				config.source_unit
			));
		}
		report.push('\n');

		let gpu = state.gpu_timing_stats();
		if gpu.count == 0 {
			report.push_str(&format!(
				"  GPU ({}): n/a (timer unavailable)\n",
				config.timer_name
			));
		} else {
			report.push_str(&format!(
				"  GPU ({}): mean {:.3} ms/step · p50 {:.3} · p95 {:.3} · min {:.3} · max {:.3} · {} sample/s",
				config.timer_name,
				gpu.mean_ms,
				gpu.median_ms,
				gpu.p95_ms,
				gpu.min_ms,
				gpu.max_ms,
				format_rate(state.gpu_samples_per_second())
			));
			if config.sequence_length > 0 {
				report.push_str(&format!(
					" · {} {}/s",
					format_rate(state.gpu_units_per_second()),
					config.sequence_unit
				));
			}
			if state.total_source_units() > 0 {
				report.push_str(&format!(
					" · {} {}/s",
					format_rate(state.gpu_source_units_per_second()),
					config.source_unit
				));
			}
			let gap = if state.gpu_samples_per_second() > 0.0 {
				(100.0 * (1.0 - state.wall_samples_per_second() / state.gpu_samples_per_second()))
					.clamp(0.0, 100.0)
			} else {
				0.0
			};
			report.push_str(&format!(" · wall-GPU gap {gap:.0}%\n"));
		}
		report.push_str(&format!(
			"  run: {:.2}s · {} steps · batch {}",
			state.elapsed().as_secs_f64(),
			state.step_count(),
			config.batch_size
		));
		if config.sequence_length > 0 {
			report.push_str(&format!(
				" · sequence {} {}/sample",
				config.sequence_length, config.sequence_unit
			));
		}
		report.push('\n');
		report
	}
}

impl Default for TrainingSummary {
	fn default() -> Self {
		Self::new(true)
	}
}

impl TrainingCallback for TrainingSummary {
	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		if self.track_initial_loss
			&& self.initial_loss.is_none()
			&& context.snapshot().step_count() == 1
		{
			self.initial_loss = context.snapshot().last_loss();
		}
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		write_stdout(&self.render_report(context.snapshot(), context.config()))?;
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`ProgressBar`].
pub type CbProgressBar = ProgressBar;

/// OA C++ compatibility spelling for [`TrainingSummary`].
pub type CbSummary = TrainingSummary;

fn format_rate(rate: f64) -> String {
	if rate >= 1.0e6 {
		format!("{:.2}M", rate / 1.0e6)
	} else if rate >= 1.0e3 {
		format!("{:.2}K", rate / 1.0e3)
	} else {
		format!("{rate:.0}")
	}
}

fn write_stdout(text: &str) -> Result<()> {
	let mut stdout = io::stdout().lock();
	stdout
		.write_all(text.as_bytes())
		.map_err(|error| Error::io("write training progress", error))?;
	stdout
		.flush()
		.map_err(|error| Error::io("flush training progress", error))
}
