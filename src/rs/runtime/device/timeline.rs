use crate::{Error, Result};

pub(super) struct Timeline {
	handle: vk::Semaphore,
}

impl Timeline {
	pub(super) fn new(device: &vk::Device) -> Result<Self> {
		let mut type_info = vk::SemaphoreTypeCreateInfo::new()
			.set_semaphore_type(vk::SemaphoreType::TIMELINE)
			.set_initial_value(0);
		let create_info = vk::SemaphoreCreateInfo::new()
			.push_next(&mut type_info)
			.map_err(Error::from)?;

		// SAFETY: timelineSemaphore was queried and enabled for this device. The create
		// chain remains live for the call and no allocation callbacks are installed.
		let handle = unsafe { device.create_semaphore(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "timeline creation", source))?;
		Ok(Self { handle })
	}

	pub(super) fn is_complete(&self, device: &vk::Device, value: u64) -> Result<bool> {
		// SAFETY: this timeline semaphore belongs to the live device and may be queried
		// concurrently with queue signals and host waits.
		let mut completed = 0;
		unsafe { device.get_semaphore_counter_value(self.handle, &mut completed) }
			.map_err(|source| Error::backend_failure("Vulkan", "timeline query", source))?;
		Ok(completed >= value)
	}

	pub(super) fn wait(&self, device: &vk::Device, value: u64) -> Result<()> {
		let semaphores = [self.handle];
		let values = [value];
		let wait_info = vk::SemaphoreWaitInfo::new()
			.set_semaphores(&semaphores, &values)
			.map_err(Error::from)?;
		// SAFETY: the semaphore/value arrays have equal length and remain live for the
		// call. An infinite timeout makes this the explicit completion boundary.
		match unsafe { device.wait_semaphores(&wait_info, u64::MAX) } {
			Ok(vk::Result::SUCCESS) => Ok(()),
			result => Err(Error::backend_dispatch_failure(
				"Vulkan",
				"timeline wait",
				result,
			)),
		}
	}

	pub(super) const fn raw(&self) -> vk::Semaphore {
		self.handle
	}

	pub(super) fn destroy(&mut self, device: &vk::Device) {
		// SAFETY: the owning device destroys this semaphore once after the retirement
		// worker and every event have released their device references.
		let _ = unsafe { device.destroy_semaphore(self.handle, None) };
	}
}
