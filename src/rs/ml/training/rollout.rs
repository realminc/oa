//! Two-phase collection/update lifecycle for synchronous on-policy training.
//!
//! Port provenance: OA C++ `oa/ml/itRolloutTraining.{h,cpp}`. This coordinator
//! composes one [`ItTraining`] optimizer lifecycle and never owns environment
//! execution or another runtime.

use crate::{Engine, Error, Matrix, Result};

use super::{ItTraining, ItTrainingConfig, Optimizer, TrainingSnapshot};
use crate::ml::{RolloutBuffer, advantage::GaeConfig};

/// Fixed rollout and update schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItRolloutTrainingConfig {
	/// Number of complete collection cycles.
	pub rollouts: u64,
	/// Time steps stored by each rollout.
	pub horizon: usize,
	/// Vector environments represented by each time step.
	pub environments: usize,
	/// Full-batch optimizer passes after each collection.
	pub update_epochs: u64,
	/// Device timer label for update steps.
	pub timer_name: String,
	/// Whether each optimizer update records and resolves Vulkan timestamps.
	pub enable_gpu_timing: bool,
}

impl Default for ItRolloutTrainingConfig {
	fn default() -> Self {
		Self {
			rollouts: 0,
			horizon: 0,
			environments: 0,
			update_epochs: 0,
			timer_name: "rl_update".into(),
			enable_gpu_timing: false,
		}
	}
}

/// Current phase of one synchronous on-policy schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutTrainingPhase {
	/// Caller is collecting environment transitions.
	Collect,
	/// Caller is performing optimizer passes over the finalized rollout.
	Update,
	/// Every configured rollout/update pair has completed or stopped.
	Complete,
}

/// Coordinator for alternating fixed-capacity collection and optimizer phases.
#[must_use]
pub struct ItRolloutTraining<'engine, 'hooks> {
	config: ItRolloutTrainingConfig,
	updates: ItTraining<'engine, 'hooks>,
	phase: RolloutTrainingPhase,
	rollout_index: u64,
	update_epoch: u64,
	rollout_open: bool,
	update_body_pending: bool,
	active_rollout: Option<*const RolloutBuffer>,
}

impl<'engine, 'hooks> ItRolloutTraining<'engine, 'hooks> {
	/// Construct a checked rollout lifecycle around one optimizer.
	///
	/// # Errors
	///
	/// Returns an error for zero schedule dimensions, count overflow, or
	/// optimizer/engine ownership mismatch.
	pub fn new(
		engine: &'engine Engine,
		optimizer: &'engine mut dyn Optimizer,
		config: ItRolloutTrainingConfig,
	) -> Result<Self> {
		if config.rollouts == 0
			|| config.horizon == 0
			|| config.environments == 0
			|| config.update_epochs == 0
		{
			return Err(Error::invalid_argument(
				"rollout training requires nonzero rollouts, horizon, environments, and update epochs",
			));
		}
		let batch_size = config
			.horizon
			.checked_mul(config.environments)
			.ok_or_else(|| Error::resource_exhausted("rollout training batch size overflows"))?;
		let batch_size = u64::try_from(batch_size)
			.map_err(|_| Error::resource_exhausted("rollout training batch exceeds u64"))?;
		let total_steps = config
			.rollouts
			.checked_mul(config.update_epochs)
			.ok_or_else(|| Error::resource_exhausted("rollout training update count overflows"))?;
		let updates = ItTraining::new_eager(
			engine,
			optimizer,
			ItTrainingConfig {
				total_steps,
				steps_per_epoch: config.update_epochs,
				batch_size,
				timer_name: config.timer_name.clone(),
				enable_gpu_timing: config.enable_gpu_timing,
				..ItTrainingConfig::default()
			},
		)?;
		Ok(Self {
			config,
			updates,
			phase: RolloutTrainingPhase::Collect,
			rollout_index: 0,
			update_epoch: 0,
			rollout_open: false,
			update_body_pending: false,
			active_rollout: None,
		})
	}

	/// Reset and open one matching rollout buffer for collection.
	///
	/// # Errors
	///
	/// Returns an error outside the idle Collect phase, for mismatched geometry,
	/// or when device-side validity reset cannot be recorded.
	pub fn begin_rollout(&mut self, rollout: &mut RolloutBuffer) -> Result<()> {
		if self.phase != RolloutTrainingPhase::Collect
			|| self.rollout_open
			|| self.update_body_pending
		{
			return Err(Error::failed_precondition(
				"begin_rollout requires an idle Collect phase",
			));
		}
		if rollout.config().time != self.config.horizon
			|| rollout.config().environments != self.config.environments
		{
			return Err(Error::invalid_argument(
				"rollout buffer geometry does not match training configuration",
			));
		}
		rollout.reset()?;
		self.rollout_open = true;
		self.active_rollout = Some(std::ptr::from_ref(rollout));
		Ok(())
	}

	/// Finalize the full active rollout with GAE and enter Update.
	///
	/// # Errors
	///
	/// Returns an error for the wrong phase/buffer, incomplete collection, or
	/// GAE recording failure.
	pub fn finalize_rollout(
		&mut self,
		rollout: &mut RolloutBuffer,
		config: GaeConfig,
	) -> Result<()> {
		if self.phase != RolloutTrainingPhase::Collect
			|| !self.rollout_open
			|| self.update_body_pending
			|| !self.active_is(rollout)
		{
			return Err(Error::failed_precondition(
				"finalize_rollout requires the active Collect buffer",
			));
		}
		rollout.finalize(config)?;
		self.rollout_open = false;
		self.phase = RolloutTrainingPhase::Update;
		self.update_epoch = 0;
		Ok(())
	}

	/// Restore Collect control state after the enclosing command transaction aborts.
	///
	/// This is valid from `begin_rollout` through finalization until the first
	/// update begins. It changes host visibility only; the caller must already
	/// have rejected/aborted the unsubmitted Engine recording.
	///
	/// # Errors
	///
	/// Returns an error for another buffer or after any update began.
	pub fn abort_rollout(&mut self, rollout: &mut RolloutBuffer) -> Result<()> {
		let open = self.phase == RolloutTrainingPhase::Collect && self.rollout_open;
		let finalized = self.phase == RolloutTrainingPhase::Update
			&& !self.rollout_open
			&& self.update_epoch == 0;
		if (!open && !finalized) || self.update_body_pending || !self.active_is(rollout) {
			return Err(Error::failed_precondition(
				"abort_rollout requires an unsubmitted collection before its first update",
			));
		}
		rollout.abort_unsubmitted();
		self.rollout_open = false;
		self.update_body_pending = false;
		self.active_rollout = None;
		self.phase = RolloutTrainingPhase::Collect;
		self.update_epoch = 0;
		Ok(())
	}

	/// Begin one optimizer epoch over the active finalized rollout.
	///
	/// Returns `false` for a stopped/paused/completed underlying iterator.
	///
	/// # Errors
	///
	/// Returns an error outside Update, for a duplicate begin, or from lifecycle
	/// callbacks.
	pub fn begin_update(&mut self) -> Result<bool> {
		if self.phase != RolloutTrainingPhase::Update
			|| self.update_body_pending
			|| self.update_epoch >= self.config.update_epochs
		{
			return Err(Error::failed_precondition(
				"begin_update requires an idle Update phase",
			));
		}
		if !self.updates.begin_step()? {
			if self.updates.stop_requested() {
				self.phase = RolloutTrainingPhase::Complete;
			}
			return Ok(false);
		}
		self.update_body_pending = true;
		self.active_rollout = None;
		Ok(true)
	}

	/// Complete one previously begun update and advance the phase schedule.
	///
	/// # Errors
	///
	/// Returns an error without a matching begin or from optimizer, completion,
	/// metric, callback, and loss validation.
	pub fn complete_update(&mut self, loss: &Matrix) -> Result<()> {
		if self.phase != RolloutTrainingPhase::Update || !self.update_body_pending {
			return Err(Error::failed_precondition(
				"complete_update requires a preceding begin_update",
			));
		}
		if let Err(error) = self.updates.complete_step(loss) {
			self.update_body_pending = false;
			self.phase = RolloutTrainingPhase::Complete;
			return Err(error);
		}
		self.update_body_pending = false;
		self.update_epoch += 1;
		if self.update_epoch == self.config.update_epochs {
			self.rollout_index += 1;
			self.phase = if self.rollout_index == self.config.rollouts {
				RolloutTrainingPhase::Complete
			} else {
				RolloutTrainingPhase::Collect
			};
		}
		Ok(())
	}

	pub(super) fn fail_update<T>(&mut self, error: Error) -> Result<T> {
		if !self.update_body_pending {
			return Err(error);
		}
		self.update_body_pending = false;
		self.phase = RolloutTrainingPhase::Complete;
		self.updates.fail_composed_step(error)
	}

	/// Return whether collection/update work has completed or stopped.
	pub fn is_done(&self) -> bool {
		self.phase == RolloutTrainingPhase::Complete || self.updates.stop_requested()
	}

	/// Return the current two-phase state.
	pub const fn phase(&self) -> RolloutTrainingPhase {
		self.phase
	}

	/// Return the number of fully updated rollouts.
	pub const fn rollout_index(&self) -> u64 {
		self.rollout_index
	}

	/// Return completed update epochs in the current rollout.
	pub const fn update_epoch(&self) -> u64 {
		self.update_epoch
	}

	/// Return the immutable schedule.
	pub const fn config(&self) -> &ItRolloutTrainingConfig {
		&self.config
	}

	/// Borrow the ordinary optimizer-update lifecycle.
	pub const fn update_loop(&self) -> &ItTraining<'engine, 'hooks> {
		&self.updates
	}

	/// Mutably borrow the ordinary optimizer-update lifecycle between steps.
	pub fn update_loop_mut(&mut self) -> &mut ItTraining<'engine, 'hooks> {
		&mut self.updates
	}

	/// Finish the completed two-phase lifecycle and fire terminal callbacks.
	///
	/// # Errors
	///
	/// Returns an error before completion or with a pending update body.
	pub fn finish(self) -> Result<TrainingSnapshot> {
		if !self.is_done() || self.update_body_pending {
			return Err(Error::failed_precondition(
				"rollout training finish requires the Complete phase",
			));
		}
		self.updates.finish()
	}

	fn active_is(&self, rollout: &RolloutBuffer) -> bool {
		self.active_rollout
			.is_some_and(|active| std::ptr::eq(active, std::ptr::from_ref(rollout)))
	}
}
