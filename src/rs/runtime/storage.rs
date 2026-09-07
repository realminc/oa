use crate::{Event, Result};

use super::vk;

/// Safe runtime storage boundary above a backend allocation.
pub(crate) struct Storage {
	buffer: Option<vk::Buffer>,
	byte_len: usize,
	device: vk::Device,
	pending_event: Option<Event>,
}

impl Storage {
	pub(super) fn from_bytes(device: &vk::Device, bytes: &[u8]) -> Result<Self> {
		if bytes.is_empty() {
			return Ok(Self {
				buffer: None,
				byte_len: 0,
				device: device.clone(),
				pending_event: None,
			});
		}

		let buffer = vk::Buffer::host_visible_storage(device, bytes.len())?;
		buffer.write(0, bytes)?;
		Ok(Self {
			buffer: Some(buffer),
			byte_len: bytes.len(),
			device: device.clone(),
			pending_event: None,
		})
	}

	pub(crate) fn read(&self, output: &mut [u8]) -> Result<()> {
		if output.len() != self.byte_len {
			return Err(crate::Error::invalid_argument(format!(
				"readback length {} does not match storage length {}",
				output.len(),
				self.byte_len
			)));
		}
		match &self.buffer {
			Some(buffer) => buffer.read(0, output),
			None => Ok(()),
		}
	}

	pub(super) fn buffer(&self) -> Option<&vk::Buffer> {
		self.buffer.as_ref()
	}

	pub(super) fn belongs_to(&self, device: &vk::Device) -> bool {
		self.device.same_as(device)
	}

	pub(crate) fn ensure_ready(&self) -> Result<()> {
		if let Some(event) = &self.pending_event
			&& !event.is_complete()?
		{
			return Err(crate::Error::not_ready("matrix storage is not ready"));
		}
		Ok(())
	}

	pub(crate) fn wait_ready(&self) -> Result<()> {
		if let Some(event) = &self.pending_event {
			event.wait()?;
		}
		Ok(())
	}

	pub(crate) fn mark_pending(&mut self, event: Event) {
		self.pending_event = Some(event);
	}
}
