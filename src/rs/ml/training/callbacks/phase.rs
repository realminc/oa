//! Multi-phase training callback.
//!
//! Port provenance: `oa/ml/callbacks.h` (`CbPhase`).

use crate::{Error, Result};

use super::super::{
	ItTrainingConfig, Optimizer, TrainingCallback, TrainingCallbackContext, TrainingControl,
};
use super::write_stdout;

/// One consecutive epoch range in a [`PhaseSchedule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingPhase {
	id: String,
	epochs: u64,
	steps: u64,
}

impl TrainingPhase {
	/// Construct a named phase with aggregate epoch and step counts.
	///
	/// # Errors
	///
	/// Returns an error for an empty name, zero epochs, or fewer than one step
	/// per epoch.
	pub fn new(id: impl Into<String>, epochs: u64, steps: u64) -> Result<Self> {
		let id = id.into();
		if id.is_empty() {
			return Err(Error::invalid_argument(
				"training phase name must not be empty",
			));
		}
		if epochs == 0 {
			return Err(Error::invalid_argument(
				"training phase must contain at least one epoch",
			));
		}
		if steps < epochs {
			return Err(Error::invalid_argument(
				"training phase must contain at least one step per epoch",
			));
		}
		Ok(Self { id, epochs, steps })
	}

	/// Return the stable phase identifier.
	pub fn id(&self) -> &str {
		&self.id
	}

	/// Return the number of epochs in this phase.
	pub const fn epochs(&self) -> u64 {
		self.epochs
	}

	/// Return the aggregate step count across this phase.
	pub const fn steps(&self) -> u64 {
		self.steps
	}
}

type PhaseBeginHook<'hook> =
	Box<dyn FnMut(usize, &TrainingPhase, &mut dyn Optimizer) -> Result<()> + 'hook>;

/// Consecutive phase policy over one [`super::super::ItTraining`] lifecycle.
///
/// The phase schedule does not own another iterator. Its epoch/step totals must
/// exactly match the iterator configuration, and its optional transition hook
/// receives restricted mutable access to the iterator's existing optimizer.
pub struct PhaseSchedule<'hook> {
	phases: Vec<TrainingPhase>,
	on_phase_begin: Option<PhaseBeginHook<'hook>>,
	current_phase: Option<usize>,
}

impl<'hook> PhaseSchedule<'hook> {
	/// Construct an empty phase schedule.
	pub const fn new() -> Self {
		Self {
			phases: Vec::new(),
			on_phase_begin: None,
			current_phase: None,
		}
	}

	/// Append a consecutive phase.
	///
	/// # Errors
	///
	/// Returns an error for invalid phase fields or aggregate count overflow.
	pub fn add_phase(&mut self, id: impl Into<String>, epochs: u64, steps: u64) -> Result<()> {
		let phase = TrainingPhase::new(id, epochs, steps)?;
		let _ = self
			.total_epochs()
			.checked_add(epochs)
			.ok_or_else(|| Error::resource_exhausted("training phase epoch count overflows u64"))?;
		let _ = self
			.total_steps()
			.checked_add(steps)
			.ok_or_else(|| Error::resource_exhausted("training phase step count overflows u64"))?;
		self.phases.push(phase);
		Ok(())
	}

	/// Install the hook invoked exactly once when each phase begins.
	pub fn set_on_phase_begin(
		&mut self,
		hook: impl FnMut(usize, &TrainingPhase, &mut dyn Optimizer) -> Result<()> + 'hook,
	) {
		self.on_phase_begin = Some(Box::new(hook));
	}

	/// Return all configured phases in execution order.
	pub fn phases(&self) -> &[TrainingPhase] {
		&self.phases
	}

	/// Return the currently entered phase index.
	pub const fn current_phase(&self) -> Option<usize> {
		self.current_phase
	}

	/// Return the aggregate epoch count.
	pub fn total_epochs(&self) -> u64 {
		self.phases.iter().map(TrainingPhase::epochs).sum()
	}

	/// Return the aggregate step count.
	pub fn total_steps(&self) -> u64 {
		self.phases.iter().map(TrainingPhase::steps).sum()
	}

	fn validate_config(&self, config: &ItTrainingConfig) -> Result<()> {
		if self.phases.is_empty() {
			return Ok(());
		}
		let actual = configured_epoch_steps(config)?;
		let expected_epochs = usize::try_from(self.total_epochs())
			.map_err(|_| Error::resource_exhausted("training phase epoch count exceeds usize"))?;
		if actual.len() != expected_epochs {
			return Err(Error::invalid_argument(format!(
				"training phase schedule defines {expected_epochs} epochs but iterator defines {}",
				actual.len()
			)));
		}

		let mut epoch = 0_usize;
		for phase in &self.phases {
			let phase_epochs = usize::try_from(phase.epochs).map_err(|_| {
				Error::resource_exhausted("training phase epoch count exceeds usize")
			})?;
			let end = epoch.checked_add(phase_epochs).ok_or_else(|| {
				Error::resource_exhausted("training phase epoch range overflows usize")
			})?;
			let actual_steps = actual[epoch..end].iter().try_fold(0_u64, |sum, steps| {
				sum.checked_add(*steps).ok_or_else(|| {
					Error::resource_exhausted("iterator phase step count overflows u64")
				})
			})?;
			if actual_steps != phase.steps {
				return Err(Error::invalid_argument(format!(
					"training phase {} defines {} steps but its iterator epochs contain {actual_steps}",
					phase.id, phase.steps
				)));
			}
			epoch = end;
		}
		Ok(())
	}

	fn phase_for_epoch(&self, epoch: u64) -> Result<(usize, u64)> {
		if epoch == 0 {
			return Err(Error::failed_precondition(
				"training phase callback requires a one-based epoch",
			));
		}
		let mut epochs_before = 0_u64;
		for (index, phase) in self.phases.iter().enumerate() {
			let end = epochs_before.checked_add(phase.epochs).ok_or_else(|| {
				Error::resource_exhausted("training phase epoch range overflows u64")
			})?;
			if epoch <= end {
				return Ok((index, epoch - epochs_before));
			}
			epochs_before = end;
		}
		Err(Error::failed_precondition(format!(
			"training epoch {epoch} is outside the configured phase schedule"
		)))
	}

	fn render_schedule(&self) -> String {
		let mut output = String::from("phase schedule:\n");
		for (index, phase) in self.phases.iter().enumerate() {
			let suffix = if phase.epochs == 1 { "" } else { "s" };
			output.push_str(&format!(
				"  {}. {:<8} — {} epoch{} ({} steps)\n",
				index + 1,
				phase.id,
				phase.epochs,
				suffix,
				phase.steps,
			));
		}
		output
	}
}

impl Default for PhaseSchedule<'_> {
	fn default() -> Self {
		Self::new()
	}
}

impl TrainingCallback for PhaseSchedule<'_> {
	fn on_train_begin(
		&mut self,
		context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		self.current_phase = None;
		self.validate_config(context.config())?;
		if !self.phases.is_empty() {
			write_stdout(&self.render_schedule())?;
		}
		Ok(TrainingControl::Continue)
	}

	fn on_epoch_begin(
		&mut self,
		context: &mut TrainingCallbackContext<'_>,
	) -> Result<TrainingControl> {
		if self.phases.is_empty() {
			return Ok(TrainingControl::Continue);
		}
		let (index, epoch_in_phase) = self.phase_for_epoch(context.snapshot().epoch())?;
		let phase = &self.phases[index];
		if self.current_phase != Some(index) {
			self.current_phase = Some(index);
			let suffix = if phase.epochs == 1 { "" } else { "s" };
			let steps_per_epoch = phase.steps / phase.epochs;
			write_stdout(&format!(
				"\nPhase {}/{} — {} · {} epoch{} × {} steps\n",
				index + 1,
				self.phases.len(),
				phase.id,
				phase.epochs,
				suffix,
				steps_per_epoch,
			))?;
			if let Some(hook) = &mut self.on_phase_begin {
				hook(index, phase, context.optimizer())?;
			}
		}
		write_stdout(&format!("epoch {epoch_in_phase}/{}\n", phase.epochs))?;
		Ok(TrainingControl::Continue)
	}
}

/// OA C++ compatibility spelling for [`PhaseSchedule`].
pub type CbPhase<'hook> = PhaseSchedule<'hook>;

fn configured_epoch_steps(config: &ItTrainingConfig) -> Result<Vec<u64>> {
	if !config.epoch_steps.is_empty() {
		return Ok(config.epoch_steps.clone());
	}
	if config.steps_per_epoch == 0 || config.total_steps == 0 {
		return Ok(Vec::new());
	}
	let epoch_count = config.total_steps.div_ceil(config.steps_per_epoch);
	let capacity = usize::try_from(epoch_count)
		.map_err(|_| Error::resource_exhausted("iterator epoch count exceeds usize"))?;
	let mut epochs = Vec::new();
	epochs
		.try_reserve_exact(capacity)
		.map_err(|_| Error::resource_exhausted("iterator epoch schedule allocation failed"))?;
	let mut remaining = config.total_steps;
	while remaining > 0 {
		let steps = remaining.min(config.steps_per_epoch);
		epochs.push(steps);
		remaining -= steps;
	}
	Ok(epochs)
}
