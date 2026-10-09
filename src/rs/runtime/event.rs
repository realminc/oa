use std::sync::Arc;

use std::{fmt, time::Duration};

use crate::Result;

use super::{Device, RetirementTicket, TimestampPair};

/// Exact completion point for work submitted to an OA engine.
///
/// Dropping an event never waits. The originating engine's retirement service
/// keeps submitted resources and their Vulkan device alive through completion.
#[derive(Clone)]
#[must_use]
pub struct Event {
	device: Arc<Device>,
	epoch: u64,
	retirement: RetirementTicket,
	timing: Option<TimestampPair>,
}

impl Event {
	pub(crate) fn same_as(&self, other: &Self) -> bool {
		self.epoch == other.epoch && self.device.same_as(&other.device)
	}

	pub(crate) fn follows_or_equals(&self, other: &Self) -> bool {
		self.epoch >= other.epoch && self.device.same_as(&other.device)
	}

	pub(super) const fn new(
		device: Arc<Device>,
		epoch: u64,
		retirement: RetirementTicket,
		timing: Option<TimestampPair>,
	) -> Self {
		Self {
			device,
			epoch,
			retirement,
			timing,
		}
	}

	pub(super) const fn epoch(&self) -> u64 {
		self.epoch
	}

	pub(in crate::runtime) fn comes_from(&self, device: &Arc<Device>) -> bool {
		self.device.same_as(device)
	}

	pub(in crate::runtime) fn retain_until_complete<T>(&self, value: T)
	where
		T: Send + 'static,
	{
		self.retirement.retain_until_complete(value);
	}

	/// Return whether this exact submission point has completed.
	///
	/// # Errors
	///
	/// Returns an error when retirement has failed or Vulkan cannot query the
	/// originating timeline.
	pub fn is_complete(&self) -> Result<bool> {
		self.retirement.check_failure()?;
		let complete = self.device.is_complete(self.epoch)?;
		self.retirement.check_failure()?;
		Ok(complete)
	}

	/// Wait until this exact submission point completes and its submitted command
	/// buffer has been retired on the host.
	///
	/// # Errors
	///
	/// Returns an error when the exact timeline wait or submission retirement fails.
	pub fn wait(&self) -> Result<()> {
		// The worker performs the exact device wait and retains every resource.
		// Observe its outcome instead of issuing a second Vulkan wait, including
		// after the execution domain has entered terminal failure.
		self.retirement.wait()
	}

	/// Return the complete device duration without waiting.
	///
	/// Returns `Ok(None)` while this event is incomplete.
	///
	/// # Errors
	///
	/// Returns an error when this event was not produced by a timed submission,
	/// completion cannot be queried, or Vulkan timestamp readback fails.
	pub fn try_device_duration(&self) -> Result<Option<Duration>> {
		let Some(timing) = &self.timing else {
			return Err(crate::Error::failed_precondition(
				"event was not produced by a timed submission",
			));
		};
		if !self.is_complete()? {
			return Ok(None);
		}
		Ok(Some(timing.duration()?))
	}

	/// Wait for this event and return its complete device duration.
	///
	/// # Errors
	///
	/// Returns an error when this event was not produced by a timed submission,
	/// completion waiting fails, or Vulkan timestamp readback fails.
	pub fn device_duration(&self) -> Result<Duration> {
		let Some(timing) = &self.timing else {
			return Err(crate::Error::failed_precondition(
				"event was not produced by a timed submission",
			));
		};
		self.wait()?;
		timing.duration()
	}
}

impl fmt::Debug for Event {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("Event")
			.field("epoch", &self.epoch)
			.finish_non_exhaustive()
	}
}
