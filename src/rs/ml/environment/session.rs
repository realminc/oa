//! Explicit execution lifecycle shared by native OA environments.

use crate::{
	Engine, Error, Event, Matrix, Result,
	runtime::{EngineHandle, RecordingTransaction},
};

use super::{EnvironmentSpec, EnvironmentTransition};

/// Private command transaction and exact completion state for one environment.
///
/// Native environments embed this value and expose it through [`Environment`].
/// It borrows the canonical Engine recorder for each active transaction; it
/// does not own another queue, allocator, scheduler, or execution graph.
#[must_use]
pub struct EnvironmentExecution {
	engine: EngineHandle,
	recording: Option<RecordingTransaction>,
	pending: Option<Event>,
	closed: bool,
	submission_count: u64,
}

impl EnvironmentExecution {
	/// Construct an idle execution lifecycle over one Engine.
	pub fn new(engine: &Engine) -> Self {
		Self {
			engine: engine.handle(),
			recording: None,
			pending: None,
			closed: false,
			submission_count: 0,
		}
	}

	fn begin(&mut self) -> Result<()> {
		if self.closed {
			return Err(Error::failed_precondition(
				"environment execution session is closed",
			));
		}
		if self.pending.is_some() {
			return Err(Error::failed_precondition(
				"environment execution requires wait before another recording",
			));
		}
		if self.recording.is_none() {
			self.recording = Some(RecordingTransaction::begin(self.engine.clone())?);
		}
		Ok(())
	}

	fn submit(&mut self) -> Result<Event> {
		let recording = self.recording.take().ok_or_else(|| {
			Error::failed_precondition("environment submit requires an active recording")
		})?;
		let event = recording.submit()?;
		self.submission_count = self
			.submission_count
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("environment submission count exhausted"))?;
		self.pending = Some(event.clone());
		Ok(event)
	}

	fn wait(&mut self, event: &Event) -> Result<()> {
		let pending = self.pending.as_ref().ok_or_else(|| {
			Error::failed_precondition("environment wait requires a submitted event")
		})?;
		if !pending.same_as(event) {
			return Err(Error::invalid_argument(
				"environment wait event does not match its pending submission",
			));
		}
		event.wait()?;
		self.pending = None;
		Ok(())
	}

	fn cancel(&mut self) -> Result<()> {
		if self.closed {
			return Err(Error::failed_precondition(
				"environment execution session is closed",
			));
		}
		if self.pending.is_some() {
			return Err(Error::failed_precondition(
				"environment cannot cancel submitted work",
			));
		}
		self.recording = None;
		Ok(())
	}

	fn close(&mut self) -> Result<()> {
		if self.closed {
			return Ok(());
		}
		if let Some(event) = self.pending.clone() {
			self.wait(&event)?;
		}
		self.recording = None;
		self.closed = true;
		Ok(())
	}

	/// Return whether the session accepts lifecycle commands.
	pub const fn is_open(&self) -> bool {
		!self.closed
	}

	/// Return whether an unsubmitted command transaction is active.
	pub const fn has_active_recording(&self) -> bool {
		self.recording.is_some()
	}

	/// Return whether one exact completion must be waited.
	pub const fn has_pending_event(&self) -> bool {
		self.pending.is_some()
	}

	/// Return the number of accepted submissions.
	pub const fn submission_count(&self) -> u64 {
		self.submission_count
	}
}

/// Stateful native vector-environment behavior.
///
/// Implementations own semantic environment state and embed exactly one
/// [`EnvironmentExecution`]. Public default methods own transaction state,
/// validation boundaries, submission, and rollback; implementations only
/// author reset/step/reset-completed commands and transactional host metadata.
pub trait Environment {
	/// Return the sole Engine borrowed by this session.
	fn engine(&self) -> &Engine;

	/// Return the structural environment contract.
	fn spec(&self) -> &EnvironmentSpec;

	/// Return the positive vector-environment lane count.
	fn environments(&self) -> u32;

	/// Return the current device-resident observation.
	fn observation(&self) -> &Matrix;

	/// Return the embedded execution lifecycle.
	fn execution(&self) -> &EnvironmentExecution;

	/// Mutably return the embedded execution lifecycle.
	fn execution_mut(&mut self) -> &mut EnvironmentExecution;

	/// Record implementation-specific reset commands.
	fn record_reset(&mut self, seed: u64) -> Result<()>;

	/// Record one implementation-specific step.
	fn record_step(&mut self, action: &Matrix) -> Result<EnvironmentTransition>;

	/// Record reset of lanes that ended on the previous step.
	fn record_reset_completed(&mut self) -> Result<()>;

	/// Commit host metadata after queue submission accepts the transaction.
	fn commit_recorded_state(&mut self) {}

	/// Roll back host metadata after an unsubmitted transaction is discarded.
	fn rollback_recorded_state(&mut self) {}

	/// Begin or retain an idempotent command transaction.
	///
	/// # Errors
	///
	/// Returns an error when closed, awaiting completion, or the Engine recorder
	/// is unavailable.
	fn begin(&mut self) -> Result<()> {
		self.execution_mut().begin()
	}

	/// Record a full environment reset into the active transaction.
	///
	/// # Errors
	///
	/// Returns a lifecycle or implementation recording error. Failure cancels
	/// the entire unsubmitted transaction and rolls back host metadata.
	fn reset(&mut self, seed: u64) -> Result<()> {
		self.begin()?;
		if let Err(error) = self.record_reset(seed) {
			let _ = self.execution_mut().cancel();
			self.rollback_recorded_state();
			return Err(error);
		}
		if let Err(error) = self
			.spec()
			.validate_reset(self.observation(), self.environments())
		{
			let _ = self.execution_mut().cancel();
			self.rollback_recorded_state();
			return Err(error);
		}
		Ok(())
	}

	/// Record one environment step into the active transaction.
	///
	/// # Errors
	///
	/// Returns a lifecycle, action, transition, or implementation error. Failure
	/// cancels the complete transaction and rolls back host metadata.
	fn step(&mut self, action: &Matrix) -> Result<EnvironmentTransition> {
		self.begin()?;
		if let Err(error) = self.spec().validate_action(action, self.environments()) {
			let _ = self.execution_mut().cancel();
			self.rollback_recorded_state();
			return Err(error);
		}
		let transition = match self.record_step(action) {
			Ok(transition) => transition,
			Err(error) => {
				let _ = self.execution_mut().cancel();
				self.rollback_recorded_state();
				return Err(error);
			}
		};
		if let Err(error) =
			self.spec()
				.validate_transition(action, &transition, self.environments())
		{
			let _ = self.execution_mut().cancel();
			self.rollback_recorded_state();
			return Err(error);
		}
		Ok(transition)
	}

	/// Record reset of completed lanes into the active transaction.
	///
	/// # Errors
	///
	/// Returns a lifecycle or implementation error and cancels on failure.
	fn reset_completed(&mut self) -> Result<()> {
		self.begin()?;
		if let Err(error) = self.record_reset_completed() {
			let _ = self.execution_mut().cancel();
			self.rollback_recorded_state();
			return Err(error);
		}
		Ok(())
	}

	/// Submit the exact active command transaction without waiting.
	///
	/// # Errors
	///
	/// Returns an error without active work or when plan construction/queue
	/// submission fails. Host metadata is committed only after acceptance.
	fn submit(&mut self) -> Result<Event> {
		match self.execution_mut().submit() {
			Ok(event) => {
				self.commit_recorded_state();
				Ok(event)
			}
			Err(error) => {
				self.rollback_recorded_state();
				Err(error)
			}
		}
	}

	/// Wait for this environment's exact pending event.
	///
	/// # Errors
	///
	/// Returns an error for a foreign/stale event or completion failure.
	fn wait(&mut self, event: &Event) -> Result<()> {
		self.execution_mut().wait(event)
	}

	/// Discard only the unsubmitted command transaction.
	///
	/// # Errors
	///
	/// Returns an error when closed or after submission.
	fn cancel(&mut self) -> Result<()> {
		self.execution_mut().cancel()?;
		self.rollback_recorded_state();
		Ok(())
	}

	/// Explicitly complete submitted work, discard unsubmitted work, and close.
	///
	/// # Errors
	///
	/// Returns an exact completion failure. Repeated close is idempotent.
	fn close(&mut self) -> Result<()> {
		self.execution_mut().close()?;
		self.rollback_recorded_state();
		Ok(())
	}
}
