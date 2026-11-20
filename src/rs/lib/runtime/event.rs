use std::sync::{Arc, Mutex};

use std::{fmt, time::Duration};

use crate::Result;

use super::{
	Device, RetirementTicket, TimestampPair,
	device::video::{DecodeStatus, EncodeReceipt},
};

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
	timing_start: Option<TimestampPair>,
	decode_statuses: Option<Arc<Mutex<Vec<Arc<DecodeStatus>>>>>,
	encode_receipts: Option<Arc<Mutex<Vec<Arc<EncodeReceipt>>>>>,
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
			timing_start: None,
			decode_statuses: None,
			encode_receipts: None,
		}
	}

	/// Engine supplies the first chunk of the same ordered timed plan. Keep
	/// its query pool alive while completion still names the final submission.
	pub(super) fn set_timing_start(&mut self, first: &Self) {
		self.timing_start = first.timing.clone();
	}

	pub(super) fn set_decode_statuses(&mut self, statuses: Vec<Arc<DecodeStatus>>) {
		if !statuses.is_empty() {
			self.decode_statuses = Some(Arc::new(Mutex::new(statuses)));
		}
	}

	pub(super) fn set_encode_receipts(&mut self, receipts: Vec<Arc<EncodeReceipt>>) {
		if !receipts.is_empty() {
			self.encode_receipts = Some(Arc::new(Mutex::new(receipts)));
		}
	}

	/// Encoded-packet receipts collected at this event's completion. The
	/// event first verifies feedback status, so a returned packet is safe to
	/// materialize from its slot's bitstream.
	pub(super) fn encode_receipts(&self) -> Vec<Arc<EncodeReceipt>> {
		self
			.encode_receipts
			.as_ref()
			.map(|receipts| {
				receipts
					.lock()
					.unwrap_or_else(|poisoned| poisoned.into_inner())
					.clone()
			})
			.unwrap_or_default()
	}

	fn verify_encode_receipts(&self) -> Result<()> {
		if let Some(receipts) = &self.encode_receipts {
			let receipts = receipts
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner());
			for receipt in receipts.iter() {
				receipt.verify()?;
			}
		}
		Ok(())
	}

	fn verify_decode_statuses(&self) -> Result<()> {
		if let Some(statuses) = &self.decode_statuses {
			let mut statuses = statuses
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner());
			for status in statuses.iter() {
				status.verify()?;
			}
			statuses.clear();
		}
		Ok(())
	}

	pub(crate) const fn epoch(&self) -> u64 {
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
	/// Returns an error when retirement or replayed decode has failed, or
	/// when Vulkan cannot query the originating timeline.
	pub fn is_complete(&self) -> Result<bool> {
		self.retirement.check_failure()?;
		let complete = self.device.is_complete(self.epoch)?;
		self.retirement.check_failure()?;
		if complete {
			self.verify_decode_statuses()?;
			self.verify_encode_receipts()?;
		}
		Ok(complete)
	}

	/// Wait until this exact submission point completes and its submitted command
	/// buffer has been retired on the host.
	///
	/// # Errors
	///
	/// Returns an error when the exact timeline wait, submission retirement,
	/// or any replayed video decode fails.
	pub fn wait(&self) -> Result<()> {
		// The worker performs the exact device wait and retains every resource.
		// Observe its outcome instead of issuing a second Vulkan wait, including
		// after the execution domain has entered terminal failure.
		self.retirement.wait()?;
		self.verify_decode_statuses()?;
		self.verify_encode_receipts()
	}

	/// Materialize every encoded packet captured by this event's plan.
	///
	/// The event waits for completion first, so returned packets are
	/// guaranteed complete feedback plus a readable bitstream. Plans without
	/// recorded encode nodes return an empty list.
	///
	/// # Errors
	///
	/// Returns an error when the wait fails, feedback did not complete, or a
	/// packet cannot be read back from its slot.
	pub fn encoded_packets(&self) -> Result<Vec<crate::video::EncodedVideoPacket>> {
		self.wait()?;
		self
			.encode_receipts()
			.iter()
			.map(|receipt| receipt.packet())
			.collect()
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
		Ok(Some(self.timing_start.as_ref().map_or_else(
			|| timing.duration(),
			|start| start.duration_until(timing),
		)?))
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
		self
			.timing_start
			.as_ref()
			.map_or_else(|| timing.duration(), |start| start.duration_until(timing))
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

#[cfg(test)]
#[path = "../../../../test/rs/runtime/event_unit.rs"]
mod tests;
