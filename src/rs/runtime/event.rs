use std::fmt;

use crate::Result;

use super::vk::{Device, RetirementTicket};

/// Exact completion point for work submitted to an OA engine.
///
/// Dropping an event never waits. The originating engine's retirement service
/// keeps submitted resources and their Vulkan device alive through completion.
#[derive(Clone)]
#[must_use]
pub struct Event {
	device: Device,
	epoch: u64,
	retirement: RetirementTicket,
}

impl Event {
	pub(super) const fn new(device: Device, epoch: u64, retirement: RetirementTicket) -> Self {
		Self {
			device,
			epoch,
			retirement,
		}
	}

	/// Return whether this exact submission point has completed.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot query the originating timeline.
	pub fn is_complete(&self) -> Result<bool> {
		self.device.is_complete(self.epoch)
	}

	/// Wait until this exact submission point completes and its submitted command
	/// buffer has been retired on the host.
	///
	/// # Errors
	///
	/// Returns an error when the Vulkan timeline wait fails.
	pub fn wait(&self) -> Result<()> {
		self.device.wait(self.epoch)?;
		self.retirement.wait()
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
