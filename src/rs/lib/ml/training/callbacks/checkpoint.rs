//! Native `.oam` checkpoint callback.
//!
//! Port provenance: `oa/ml/callbacks.h` (`CbCheckpoint`).

use std::time::Instant;

use crate::{
	Result,
	ml::{CheckpointManager, Module},
};

use super::super::{TrainingCallback, TrainingCallbackContext, TrainingControl, TrainingSnapshot};
use super::{ValidationMetric, write_stdout};

/// Epoch, interval, best-model, and restore-best checkpoint policy.
pub struct Checkpoint<'owner, 'engine> {
	manager: &'owner mut CheckpointManager<'engine>,
	model: &'owner dyn Module,
	save_every: u64,
	validation_metric: Option<ValidationMetric>,
	restore_best: bool,
	verbose: bool,
	have_best: bool,
	best_epoch: u64,
	last_epoch: u64,
}

impl<'owner, 'engine> Checkpoint<'owner, 'engine> {
	/// Borrow the checkpoint manager and model for one training lifecycle.
	pub fn new(manager: &'owner mut CheckpointManager<'engine>, model: &'owner dyn Module) -> Self {
		Self {
			manager,
			model,
			save_every: 0,
			validation_metric: None,
			restore_best: true,
			verbose: true,
			have_best: false,
			best_epoch: 0,
			last_epoch: 0,
		}
	}

	/// Add a resumable mid-epoch checkpoint every `steps` completed steps.
	///
	/// Zero disables interval checkpoints.
	pub fn set_save_every(&mut self, steps: u64) {
		self.save_every = steps;
	}

	/// Select a preceding validation callback's completed metric.
	pub fn set_validation_metric(&mut self, metric: ValidationMetric) {
		self.validation_metric = Some(metric);
	}

	/// Enable or disable restoring the best epoch at training end.
	pub fn set_restore_best(&mut self, restore: bool) {
		self.restore_best = restore;
	}

	/// Enable or suppress checkpoint status output.
	pub fn set_verbose(&mut self, verbose: bool) {
		self.verbose = verbose;
	}

	/// Return the one-based best epoch observed by this callback.
	pub const fn best_epoch(&self) -> u64 {
		self.best_epoch
	}

	fn metric(&self, snapshot: TrainingSnapshot) -> f64 {
		self.validation_metric.as_ref().map_or_else(
			|| {
				if snapshot.total_epochs() > 0 {
					snapshot.epoch_mean_loss()
				} else {
					snapshot.training_mean_loss()
				}
			},
			ValidationMetric::result,
		)
	}

	fn metric_name(&self) -> &str {
		self
			.validation_metric
			.as_ref()
			.map_or_else(|| self.manager.metric_name(), ValidationMetric::name)
	}
}

impl TrainingCallback for Checkpoint<'_, '_> {
	fn on_step_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		if self.save_every == 0
			|| snapshot.is_epoch_boundary()
			|| !snapshot.step_count().is_multiple_of(self.save_every)
		{
			return Ok(TrainingControl::Continue);
		}
		let metric = f64::from(snapshot.last_loss().unwrap_or(f32::NAN));
		if self.verbose {
			write_stdout(&format!(
				"\nStep {}: loss = {metric:.6} — saving resumable checkpoint\n",
				snapshot.step_count()
			))?;
		}
		let start = Instant::now();
		let result = self.manager.save_incremental(
			self.model,
			context.checkpoint_optimizer()?,
			snapshot.step_count(),
			metric,
			Some("loss"),
		);
		context.exclude_wall_time(start.elapsed());
		result?;
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		let metric = self.metric(snapshot);
		let metric_name = self.metric_name().to_owned();
		let improved = self.manager.is_better(metric);
		let previous = self.manager.best_metric();
		if self.verbose {
			let message = if improved && !self.have_best {
				format!(
					"\nEpoch {}: {metric_name} = {metric:.6} — saving model\n",
					snapshot.epoch()
				)
			} else if improved {
				format!(
					"\nEpoch {}: {metric_name} improved from {previous:.6} to {metric:.6} — saving model\n",
					snapshot.epoch()
				)
			} else {
				format!(
					"\nEpoch {}: {metric_name} did not improve from {previous:.6}\n",
					snapshot.epoch()
				)
			};
			write_stdout(&message)?;
		}
		self.last_epoch = snapshot.epoch();
		let start = Instant::now();
		let result = self.manager.maybe_save(
			self.model,
			context.checkpoint_optimizer()?,
			snapshot.step_count(),
			metric,
			true,
		);
		context.exclude_wall_time(start.elapsed());
		if result? {
			self.have_best = true;
			self.best_epoch = snapshot.epoch();
		}
		Ok(TrainingControl::Continue)
	}

	fn on_train_end(&mut self, context: &mut TrainingCallbackContext<'_>) -> Result<TrainingControl> {
		let snapshot = context.snapshot();
		if snapshot.total_epochs() == 0 {
			let metric = self.metric(snapshot);
			let start = Instant::now();
			let result = self.manager.maybe_save(
				self.model,
				context.checkpoint_optimizer()?,
				snapshot.step_count(),
				metric,
				false,
			);
			context.exclude_wall_time(start.elapsed());
			result?;
			return Ok(TrainingControl::Continue);
		}
		if !self.restore_best {
			if !snapshot.is_epoch_boundary() {
				let metric = self.metric(snapshot);
				let start = Instant::now();
				let result = self.manager.maybe_save(
					self.model,
					context.checkpoint_optimizer()?,
					snapshot.step_count(),
					metric,
					true,
				);
				context.exclude_wall_time(start.elapsed());
				result?;
			}
			return Ok(TrainingControl::Continue);
		}
		if !self.have_best || (snapshot.is_epoch_boundary() && self.best_epoch == self.last_epoch) {
			return Ok(TrainingControl::Continue);
		}
		let start = Instant::now();
		let result = self
			.manager
			.load_best_into(self.model, context.checkpoint_optimizer()?);
		context.exclude_wall_time(start.elapsed());
		result?;
		if self.verbose {
			write_stdout(&format!(
				"Restoring model weights from the end of the best epoch: {} ({} {:.6})\n",
				self.best_epoch,
				self.metric_name(),
				self.manager.best_metric()
			))?;
		}
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`Checkpoint`].
pub type CbCheckpoint<'owner, 'engine> = Checkpoint<'owner, 'engine>;
