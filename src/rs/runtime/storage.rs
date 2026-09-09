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
	readiness: RefCell<Readiness>,
}

#[derive(Clone)]
enum Readiness {
	Ready,
	Recorded,
	Captured,
	Submitted(Event),
	Failed,
}

impl Storage {
	pub(super) fn same_as(&self, other: &Self) -> bool {
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
					readiness: RefCell::new(Readiness::Ready),
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
				readiness: RefCell::new(Readiness::Ready),
			}),
		})
	}

	pub(crate) fn read(&self, output: &mut [u8]) -> Result<()> {
		if output.len() != self.inner.byte_len {
			return Err(crate::Error::invalid_argument(format!(
				"readback length {} does not match storage length {}",
				output.len(),
				self.inner.byte_len
			)));
		}
		match &self.inner.buffer {
			Some(buffer) => buffer.read(0, output),
			None => Ok(()),
		}
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
		*self.inner.readiness.borrow_mut() = Readiness::Ready;
		Ok(())
	}

	pub(super) fn belongs_to(&self, device: &vk::Device) -> bool {
		self.inner.device.same_as(device)
	}

	pub(crate) fn needs_flush(&self) -> bool {
		matches!(*self.inner.readiness.borrow(), Readiness::Recorded)
	}

	pub(crate) fn ensure_ready(&self) -> Result<()> {
		match self.inner.readiness.borrow().clone() {
			Readiness::Ready => Ok(()),
			Readiness::Recorded => Err(Error::not_ready(
				"matrix storage is recorded but has not been submitted",
			)),
			Readiness::Captured => Err(Error::not_ready(
				"matrix storage belongs to a plan that has not been submitted",
			)),
			Readiness::Submitted(event) if event.is_complete()? => Ok(()),
			Readiness::Submitted(_) => Err(Error::not_ready("matrix storage is not ready")),
			Readiness::Failed => Err(production_failed()),
		}
	}

	pub(crate) fn wait_ready(&self) -> Result<()> {
		match self.inner.readiness.borrow().clone() {
			Readiness::Ready => Ok(()),
			Readiness::Recorded => Err(Error::failed_precondition(
				"matrix storage was not submitted before blocking observation",
			)),
			Readiness::Captured => Err(Error::failed_precondition(
				"matrix storage cannot be observed before its execution plan is submitted",
			)),
			Readiness::Submitted(event) => event.wait(),
			Readiness::Failed => Err(production_failed()),
		}
	}

	pub(crate) fn mark_recorded(&self) {
		*self.inner.readiness.borrow_mut() = Readiness::Recorded;
	}

	pub(crate) fn mark_submitted(&self, event: Event) {
		*self.inner.readiness.borrow_mut() = Readiness::Submitted(event);
	}

	pub(crate) fn mark_captured(&self) {
		*self.inner.readiness.borrow_mut() = Readiness::Captured;
	}

	pub(crate) fn validate_recording_access(&self) -> Result<()> {
		match &*self.inner.readiness.borrow() {
			Readiness::Ready | Readiness::Recorded | Readiness::Submitted(_) => Ok(()),
			Readiness::Captured => Err(Error::failed_precondition(
				"captured matrix storage must be submitted before another operation can use it",
			)),
			Readiness::Failed => Err(production_failed()),
		}
	}

	pub(crate) fn mark_failed(&self) {
		*self.inner.readiness.borrow_mut() = Readiness::Failed;
	}
}

fn production_failed() -> Error {
	Error::failed_precondition("matrix storage production failed")
}
