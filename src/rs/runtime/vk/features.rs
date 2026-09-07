use super::Instance;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DeviceFeatures {
	pub(super) timeline_semaphore: bool,
	pub(super) synchronization2: bool,
	pub(super) runtime_descriptor_array: bool,
	pub(super) descriptor_binding_partially_bound: bool,
	pub(super) descriptor_binding_storage_buffer_update_after_bind: bool,
	pub(super) descriptor_binding_update_unused_while_pending: bool,
}

impl DeviceFeatures {
	pub(super) fn query(instance: &Instance, physical: ash::vk::PhysicalDevice) -> Self {
		let mut features13 = ash::vk::PhysicalDeviceVulkan13Features::default();
		let mut features12 = ash::vk::PhysicalDeviceVulkan12Features::default();
		let mut features2 = ash::vk::PhysicalDeviceFeatures2::default()
			.push_next(&mut features12)
			.push_next(&mut features13);

		// SAFETY: `physical` belongs to this live instance. The complete output chain
		// remains valid and exclusively borrowed for the duration of the query.
		unsafe {
			instance
				.raw()
				.get_physical_device_features2(physical, &mut features2);
		}

		Self {
			timeline_semaphore: features12.timeline_semaphore == ash::vk::TRUE,
			synchronization2: features13.synchronization2 == ash::vk::TRUE,
			runtime_descriptor_array: features12.runtime_descriptor_array == ash::vk::TRUE,
			descriptor_binding_partially_bound: features12.descriptor_binding_partially_bound
				== ash::vk::TRUE,
			descriptor_binding_storage_buffer_update_after_bind: features12
				.descriptor_binding_storage_buffer_update_after_bind
				== ash::vk::TRUE,
			descriptor_binding_update_unused_while_pending: features12
				.descriptor_binding_update_unused_while_pending
				== ash::vk::TRUE,
		}
	}

	pub(super) fn missing_runtime_requirements(self) -> Option<String> {
		let requirements = [
			(self.timeline_semaphore, "timelineSemaphore"),
			(self.synchronization2, "synchronization2"),
			(self.runtime_descriptor_array, "runtimeDescriptorArray"),
			(
				self.descriptor_binding_partially_bound,
				"descriptorBindingPartiallyBound",
			),
			(
				self.descriptor_binding_storage_buffer_update_after_bind,
				"descriptorBindingStorageBufferUpdateAfterBind",
			),
			(
				self.descriptor_binding_update_unused_while_pending,
				"descriptorBindingUpdateUnusedWhilePending",
			),
		];
		let missing = requirements
			.into_iter()
			.filter_map(|(available, name)| (!available).then_some(name))
			.collect::<Vec<_>>();
		(!missing.is_empty()).then(|| missing.join(", "))
	}
}

#[cfg(test)]
mod tests {
	use super::DeviceFeatures;

	#[test]
	fn complete_runtime_requirements_have_no_missing_feature() {
		assert_eq!(
			DeviceFeatures {
				timeline_semaphore: true,
				synchronization2: true,
				runtime_descriptor_array: true,
				descriptor_binding_partially_bound: true,
				descriptor_binding_storage_buffer_update_after_bind: true,
				descriptor_binding_update_unused_while_pending: true,
			}
			.missing_runtime_requirements(),
			None
		);
	}

	#[test]
	fn missing_requirements_are_reported_exactly() {
		assert_eq!(
			DeviceFeatures {
				timeline_semaphore: false,
				synchronization2: true,
				runtime_descriptor_array: false,
				descriptor_binding_partially_bound: true,
				descriptor_binding_storage_buffer_update_after_bind: false,
				descriptor_binding_update_unused_while_pending: true,
			}
			.missing_runtime_requirements(),
			Some(
				"timelineSemaphore, runtimeDescriptorArray, descriptorBindingStorageBufferUpdateAfterBind"
					.to_owned()
			)
		);
	}
}
