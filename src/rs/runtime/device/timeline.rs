use crate::{Error, Result};

pub(super) struct Timeline {
	handle: ash::vk::Semaphore,
}

impl Timeline {
	pub(super) fn new(device: &ash::Device) -> Result<Self> {
		let mut type_info = ash::vk::SemaphoreTypeCreateInfo::default()
			.semaphore_type(ash::vk::SemaphoreType::TIMELINE)
			.initial_value(0);
		let create_info = ash::vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

		// SAFETY: timelineSemaphore was queried and enabled for this device. The create
		// chain remains live for the call and no allocation callbacks are installed.
		let handle = unsafe { device.create_semaphore(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "timeline creation", source))?;
		Ok(Self { handle })
	}

	pub(super) fn is_complete(&self, device: &ash::Device, value: u64) -> Result<bool> {
		// SAFETY: this timeline semaphore belongs to the live device and may be queried
		// concurrently with queue signals and host waits.
		let completed = unsafe { device.get_semaphore_counter_value(self.handle) }
			.map_err(|source| Error::backend_failure("Vulkan", "timeline query", source))?;
		Ok(completed >= value)
	}

	pub(super) fn wait(&self, device: &ash::Device, value: u64) -> Result<()> {
		let semaphores = [self.handle];
		let values = [value];
		let wait_info = ash::vk::SemaphoreWaitInfo::default()
			.semaphores(&semaphores)
			.values(&values);
		// SAFETY: the semaphore/value arrays have equal length and remain live for the
		// call. An infinite timeout makes this the explicit completion boundary.
		unsafe { device.wait_semaphores(&wait_info, u64::MAX) }
			.map_err(|source| Error::backend_failure("Vulkan", "timeline wait", source))
	}

	pub(super) const fn raw(&self) -> ash::vk::Semaphore {
		self.handle
	}

	pub(super) fn destroy(&mut self, device: &ash::Device) {
		// SAFETY: the owning device destroys this semaphore once after the retirement
		// worker and every event have released their device references.
		unsafe {
			device.destroy_semaphore(self.handle, None);
		}
	}
}
