use std::{cell::RefCell, rc::Rc};

use crate::{Error, Event, Result};

use super::vk;

/// Safe runtime storage boundary above a backend allocation.
#[derive(Clone)]
pub(crate) struct Storage {
	inner: Rc<StorageInner>,
}

struct StorageInner {
	buffer: Option<vk::Buffer>,
	byte_len: usize,
	device: vk::Device,
	readiness: RefCell<ReadinessSnapshot>,
}

#[derive(Clone)]
pub(super) enum ReadinessSnapshot {
	Ready,
	Recorded,
	Captured,
	Submitted(Event),
	Failed,
}

impl Storage {
	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.inner, &other.inner)
	}

	pub(super) fn owner_count(&self) -> usize {
		Rc::strong_count(&self.inner)
	}

	pub(super) fn from_bytes(device: &vk::Device, bytes: &[u8]) -> Result<Self> {
		if bytes.is_empty() {
			return Ok(Self {
				inner: Rc::new(StorageInner {
					buffer: None,
					byte_len: 0,
					device: device.clone(),
					readiness: RefCell::new(ReadinessSnapshot::Ready),
				}),
			});
		}

		let buffer = vk::Buffer::host_visible_storage(device, bytes.len())?;
		buffer.write(0, bytes)?;
		Ok(Self {
			inner: Rc::new(StorageInner {
				buffer: Some(buffer),
				byte_len: bytes.len(),
				device: device.clone(),
				readiness: RefCell::new(ReadinessSnapshot::Ready),
			}),
		})
	}

	pub(crate) fn read_prefix(&self, output: &mut [u8]) -> Result<()> {
		// Byte-oriented matrices may have a padded physical allocation. Typed
		// readback observes only the exact logical prefix.
		if output.len() > self.inner.byte_len {
			return Err(crate::Error::invalid_argument(format!(
				"readback length {} exceeds storage length {}",
				output.len(),
				self.inner.byte_len
			)));
		}
		match &self.inner.buffer {
			Some(buffer) => buffer.read(0, output),
			None => Ok(()),
		}
	}

	#[cfg(test)]
	pub(crate) fn read(&self, output: &mut [u8]) -> Result<()> {
		if output.len() != self.inner.byte_len {
			return Err(crate::Error::invalid_argument(format!(
				"readback length {} does not match storage length {}",
				output.len(),
				self.inner.byte_len
			)));
		}
		self.read_prefix(output)
	}

	pub(crate) fn write(&self, input: &[u8]) -> Result<()> {
		if input.len() != self.inner.byte_len {
			return Err(crate::Error::invalid_argument(format!(
				"upload length {} does not match storage length {}",
				input.len(),
				self.inner.byte_len
			)));
		}
		match &self.inner.buffer {
			Some(buffer) => buffer.write(0, input),
			None => Ok(()),
		}
	}

	pub(super) fn buffer(&self) -> Option<&vk::Buffer> {
		self.inner.buffer.as_ref()
	}

	pub(super) fn byte_len(&self) -> usize {
		self.inner.byte_len
	}

	pub(super) fn prepare_reuse(&self, bytes: &[u8]) -> Result<()> {
		if bytes.len() != self.inner.byte_len {
			return Err(Error::invalid_argument(
				"stable storage reuse requires an equal byte length",
			));
		}
		self.ensure_ready()?;
		self.write(bytes)?;
		*self.inner.readiness.borrow_mut() = ReadinessSnapshot::Ready;
		Ok(())
	}

	pub(super) fn belongs_to(&self, device: &vk::Device) -> bool {
		self.inner.device.same_as(device)
	}

	pub(crate) fn needs_flush(&self) -> bool {
		matches!(*self.inner.readiness.borrow(), ReadinessSnapshot::Recorded)
	}

	pub(crate) fn ensure_ready(&self) -> Result<()> {
		match self.inner.readiness.borrow().clone() {
			ReadinessSnapshot::Ready => Ok(()),
			ReadinessSnapshot::Recorded => Err(Error::not_ready(
				"matrix storage is recorded but has not been submitted",
			)),
			ReadinessSnapshot::Captured => Err(Error::not_ready(
				"matrix storage belongs to a plan that has not been submitted",
			)),
			ReadinessSnapshot::Submitted(event) if event.is_complete()? => Ok(()),
			ReadinessSnapshot::Submitted(_) => Err(Error::not_ready("matrix storage is not ready")),
			ReadinessSnapshot::Failed => Err(production_failed()),
		}
	}

	pub(crate) fn wait_ready(&self) -> Result<()> {
		match self.inner.readiness.borrow().clone() {
			ReadinessSnapshot::Ready => Ok(()),
			ReadinessSnapshot::Recorded => Err(Error::failed_precondition(
				"matrix storage was not submitted before blocking observation",
			)),
			ReadinessSnapshot::Captured => Err(Error::failed_precondition(
				"matrix storage cannot be observed before its execution plan is submitted",
			)),
			ReadinessSnapshot::Submitted(event) => event.wait(),
			ReadinessSnapshot::Failed => Err(production_failed()),
		}
	}

	pub(super) fn mark_recorded(&self) -> ReadinessSnapshot {
		std::mem::replace(
			&mut *self.inner.readiness.borrow_mut(),
			ReadinessSnapshot::Recorded,
		)
	}

	pub(super) fn restore_readiness(&self, readiness: ReadinessSnapshot) {
		*self.inner.readiness.borrow_mut() = readiness;
	}

	pub(crate) fn mark_submitted(&self, event: Event) {
		*self.inner.readiness.borrow_mut() = ReadinessSnapshot::Submitted(event);
	}

	pub(crate) fn mark_captured(&self) {
		*self.inner.readiness.borrow_mut() = ReadinessSnapshot::Captured;
	}

	pub(crate) fn validate_recording_access(&self) -> Result<()> {
		match &*self.inner.readiness.borrow() {
			ReadinessSnapshot::Ready
			| ReadinessSnapshot::Recorded
			| ReadinessSnapshot::Submitted(_) => Ok(()),
			ReadinessSnapshot::Captured => Err(Error::failed_precondition(
				"captured matrix storage must be submitted before another operation can use it",
			)),
			ReadinessSnapshot::Failed => Err(production_failed()),
		}
	}

	pub(crate) fn mark_failed(&self) {
		*self.inner.readiness.borrow_mut() = ReadinessSnapshot::Failed;
	}
}

fn production_failed() -> Error {
	Error::failed_precondition("matrix storage production failed")
}
