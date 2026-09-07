use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{Error, Result};

use super::{ComputeDispatch, Event, Storage, vk};

/// Policy for selecting the local device owned by an engine.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceSelection {
	/// Select the highest-ranked compatible hardware device.
	#[default]
	Automatic,
	/// Select an exact physical-device ordinal from backend enumeration.
	Index(usize),
}

/// Configuration used to construct one OA execution engine.
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct EngineBuilder {
	device_selection: DeviceSelection,
}

impl EngineBuilder {
	/// Create a builder using automatic hardware-device selection.
	pub const fn new() -> Self {
		Self {
			device_selection: DeviceSelection::Automatic,
		}
	}

	/// Select how this engine chooses its single local device.
	pub const fn devices(mut self, selection: DeviceSelection) -> Self {
		self.device_selection = selection;
		self
	}

	/// Construct the configured engine and its execution services.
	///
	/// # Errors
	///
	/// Returns an error when the backend cannot be loaded, its required API is
	/// unavailable, the requested device does not exist or lacks OA's required
	/// capabilities, or a device service cannot be constructed.
	pub fn build(self) -> Result<Engine> {
		Engine::build(self.device_selection)
	}
}

/// Sole owner of OA's local execution services.
///
/// The current one-device recorder is thread-affine. `Engine` intentionally
/// remains neither `Send` nor `Sync` until concurrent queue scheduling and
/// submission synchronization are implemented and verified.
pub struct Engine {
	handle: EngineHandle,
	_not_send_sync: PhantomData<Rc<()>>,
}

struct EngineState {
	// Fields drop in declaration order. Retirement disconnects first and joins only
	// an already-drained host worker; an active worker retains the device and
	// transitive instance ownership until every submitted epoch completes.
	retirement: vk::RetirementService,
	device: vk::Device,
	next_epoch: u64,
}

/// Thread-affine shared access to the execution owner retained by its values.
#[derive(Clone)]
pub(crate) struct EngineHandle {
	state: Rc<RefCell<EngineState>>,
}

impl Engine {
	/// Begin explicit engine configuration.
	pub const fn builder() -> EngineBuilder {
		EngineBuilder::new()
	}

	/// Create an engine using one compute-capable Vulkan device.
	///
	/// # Errors
	///
	/// Returns an error when Vulkan cannot be loaded, Vulkan 1.3 is unavailable,
	/// no compatible hardware device exists, or logical-device creation fails.
	pub fn new() -> Result<Self> {
		Self::builder().build()
	}

	pub(super) fn build(selection: DeviceSelection) -> Result<Self> {
		let instance = vk::Instance::new()?;
		let physical_device = vk::PhysicalDevice::select(&instance, selection)?;
		let device = vk::Device::new(&instance, physical_device)?;
		let retirement = vk::RetirementService::new(&device);

		Ok(Self {
			handle: EngineHandle {
				state: Rc::new(RefCell::new(EngineState {
					retirement,
					device,
					next_epoch: 0,
				})),
			},
			_not_send_sync: PhantomData,
		})
	}

	/// Submit an explicit completion checkpoint to this engine's compute queue.
	///
	/// The returned event completes after all work previously submitted to that
	/// queue. Dropping the event does not wait.
	///
	/// # Errors
	///
	/// Returns an error when command recording, submission, or retirement fails,
	/// or when the engine exhausts its timeline epoch space.
	pub fn checkpoint(&self) -> Result<Event> {
		self.handle.checkpoint()
	}

	pub(crate) fn handle(&self) -> EngineHandle {
		self.handle.clone()
	}
}

impl EngineHandle {
	pub(crate) fn same_as(&self, other: &Self) -> bool {
		Rc::ptr_eq(&self.state, &other.state)
	}

	pub(crate) fn create_storage(&self, bytes: &[u8]) -> Result<Storage> {
		let device = self.state.borrow().device.clone();
		Storage::from_bytes(&device, bytes)
	}

	pub(crate) fn checkpoint(&self) -> Result<Event> {
		let mut state = self.state.borrow_mut();
		let epoch = state
			.next_epoch
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("Vulkan submission epoch exhausted"))?;
		state.retirement.prepare()?;
		let command = state.device.record_empty()?;
		if let Err(error) = state.device.submit(&command, epoch) {
			state.device.free(command);
			return Err(error);
		}
		state.next_epoch = epoch;
		let retirement = state.retirement.retire(command, epoch)?;
		Ok(Event::new(state.device.clone(), epoch, retirement))
	}

	pub(crate) fn submit(&self, dispatch: ComputeDispatch<'_>) -> Result<Event> {
		let mut state = self.state.borrow_mut();
		if dispatch
			.buffers
			.iter()
			.any(|binding| !binding.storage.belongs_to(&state.device))
		{
			return Err(Error::invalid_argument(format!(
				"{} buffers must belong to the submitting engine",
				dispatch.operation
			)));
		}
		let epoch = state
			.next_epoch
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("Vulkan submission epoch exhausted"))?;
		state.retirement.prepare()?;
		let command = state.device.record_compute(&dispatch)?;
		if let Err(error) = state.device.submit(&command, epoch) {
			state.device.free(command);
			return Err(error);
		}
		state.next_epoch = epoch;
		let retirement = state.retirement.retire(command, epoch)?;
		Ok(Event::new(state.device.clone(), epoch, retirement))
	}
}

#[cfg(test)]
mod tests {
	use super::{DeviceSelection, Engine, EngineBuilder};

	#[test]
	fn default_builder_uses_automatic_device_selection() {
		assert_eq!(
			EngineBuilder::default().device_selection,
			DeviceSelection::Automatic
		);
	}

	#[test]
	#[ignore = "requires a hardware Vulkan 1.3 compute device"]
	fn allocates_records_and_frees_an_empty_command_buffer() -> crate::Result<()> {
		let engine = Engine::new()?;
		let device = engine.handle.state.borrow().device.clone();
		let command = device.record_empty()?;
		device.free(command);
		Ok(())
	}
}
