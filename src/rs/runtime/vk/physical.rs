use crate::{DeviceSelection, Error, Result};

use super::{Instance, features::DeviceFeatures};

const EXPERIMENTAL_STORAGE_BUFFER_CAPACITY: u32 = 1024;

#[derive(Clone, Copy)]
pub(super) struct DeviceLimits {
	pub(super) storage_buffer_descriptors: u32,
	pub(super) max_push_constants_size: u32,
	pub(super) max_compute_work_group_count: [u32; 3],
	pub(super) max_compute_work_group_size: [u32; 3],
	pub(super) max_compute_work_group_invocations: u32,
	pub(super) timestamp_period_ns: f64,
	pub(super) compute_timestamp_valid_bits: u32,
}

#[derive(Clone, Copy)]
pub(in crate::runtime) struct PhysicalDevice {
	pub(super) handle: ash::vk::PhysicalDevice,
	pub(super) compute_queue_family: u32,
	pub(super) features: DeviceFeatures,
	pub(super) limits: DeviceLimits,
}

impl PhysicalDevice {
	pub(in crate::runtime) fn select(
		instance: &Instance,
		selection: DeviceSelection,
	) -> Result<Self> {
		// SAFETY: the instance is live and enumeration only writes Vulkan-owned handles
		// into storage managed by Ash.
		let handles = unsafe { instance.raw().enumerate_physical_devices() }.map_err(|source| {
			Error::backend_failure("Vulkan", "physical-device enumeration", source)
		})?;

		if let DeviceSelection::Index(index) = selection
			&& index >= handles.len()
		{
			return Err(Error::invalid_argument(format!(
				"Vulkan device index {index} is out of range for {} enumerated devices",
				handles.len()
			)));
		}

		let mut selected = None;
		for (index, handle) in handles.into_iter().enumerate() {
			if let DeviceSelection::Index(requested) = selection
				&& index != requested
			{
				continue;
			}

			// SAFETY: `handle` was returned by this live instance. Both queries only read
			// immutable physical-device properties.
			let mut properties12 = ash::vk::PhysicalDeviceVulkan12Properties::default();
			let mut properties2 =
				ash::vk::PhysicalDeviceProperties2::default().push_next(&mut properties12);
			unsafe {
				instance
					.raw()
					.get_physical_device_properties2(handle, &mut properties2);
			}
			let properties = properties2.properties;
			let queue_families = unsafe {
				instance
					.raw()
					.get_physical_device_queue_family_properties(handle)
			};

			if properties.api_version < ash::vk::API_VERSION_1_3 {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::no_suitable_device(format!(
						"Vulkan device index {index} is not a hardware Vulkan 1.3 compute device"
					)));
				}
				continue;
			}
			let Some(rank) = device_type_rank(properties.device_type) else {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::no_suitable_device(format!(
						"Vulkan device index {index} is not a hardware Vulkan 1.3 compute device"
					)));
				}
				continue;
			};
			let Some((compute_queue_family, compute_timestamp_valid_bits)) =
				select_compute_queue_family(&queue_families)
			else {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::no_suitable_device(format!(
						"Vulkan device index {index} is not a hardware Vulkan 1.3 compute device"
					)));
				}
				continue;
			};

			let features = DeviceFeatures::query(instance, handle);
			if let Some(missing) = features.missing_runtime_requirements() {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} lacks required {missing}"
					)));
				}
				continue;
			}
			let descriptor_limit = properties12
				.max_per_stage_descriptor_update_after_bind_storage_buffers
				.min(properties12.max_descriptor_set_update_after_bind_storage_buffers)
				.min(properties12.max_update_after_bind_descriptors_in_all_pools);
			if descriptor_limit < 3 || properties.limits.max_push_constants_size < 16 {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} cannot provide three update-after-bind storage descriptors and 16 push-constant bytes"
					)));
				}
				continue;
			}
			let limits = DeviceLimits {
				storage_buffer_descriptors: descriptor_limit
					.min(EXPERIMENTAL_STORAGE_BUFFER_CAPACITY),
				max_push_constants_size: properties.limits.max_push_constants_size,
				max_compute_work_group_count: properties.limits.max_compute_work_group_count,
				max_compute_work_group_size: properties.limits.max_compute_work_group_size,
				max_compute_work_group_invocations: properties
					.limits
					.max_compute_work_group_invocations,
				timestamp_period_ns: f64::from(properties.limits.timestamp_period),
				compute_timestamp_valid_bits,
			};

			let candidate = Self {
				handle,
				compute_queue_family,
				features,
				limits,
			};
			if selected
				.as_ref()
				.is_none_or(|(selected_rank, _)| rank > *selected_rank)
			{
				selected = Some((rank, candidate));
			}
		}

		selected
			.map(|(_, physical_device)| physical_device)
			.ok_or_else(|| {
				Error::no_suitable_device(
					"no hardware Vulkan 1.3 compute device satisfies OA's runtime capabilities",
				)
			})
	}
}

fn select_compute_queue_family(
	queue_families: &[ash::vk::QueueFamilyProperties],
) -> Option<(u32, u32)> {
	let mut shared_compute = None;

	for (index, properties) in queue_families.iter().enumerate() {
		if properties.queue_count == 0
			|| !properties
				.queue_flags
				.contains(ash::vk::QueueFlags::COMPUTE)
		{
			continue;
		}

		let index = u32::try_from(index).ok()?;
		if !properties
			.queue_flags
			.contains(ash::vk::QueueFlags::GRAPHICS)
		{
			return Some((index, properties.timestamp_valid_bits));
		}
		shared_compute.get_or_insert((index, properties.timestamp_valid_bits));
	}

	shared_compute
}

fn device_type_rank(device_type: ash::vk::PhysicalDeviceType) -> Option<u8> {
	match device_type {
		ash::vk::PhysicalDeviceType::DISCRETE_GPU => Some(4),
		ash::vk::PhysicalDeviceType::INTEGRATED_GPU => Some(3),
		ash::vk::PhysicalDeviceType::VIRTUAL_GPU => Some(2),
		ash::vk::PhysicalDeviceType::OTHER => Some(1),
		ash::vk::PhysicalDeviceType::CPU => None,
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use super::{device_type_rank, select_compute_queue_family};

	fn queue(queue_flags: ash::vk::QueueFlags, queue_count: u32) -> ash::vk::QueueFamilyProperties {
		ash::vk::QueueFamilyProperties {
			queue_flags,
			queue_count,
			timestamp_valid_bits: 48,
			..Default::default()
		}
	}

	#[test]
	fn dedicated_compute_queue_precedes_shared_compute() {
		let families = [
			queue(
				ash::vk::QueueFlags::COMPUTE | ash::vk::QueueFlags::GRAPHICS,
				1,
			),
			queue(ash::vk::QueueFlags::COMPUTE, 1),
		];

		assert_eq!(select_compute_queue_family(&families), Some((1, 48)));
	}

	#[test]
	fn empty_and_non_compute_queues_are_rejected() {
		let families = [
			queue(ash::vk::QueueFlags::COMPUTE, 0),
			queue(ash::vk::QueueFlags::TRANSFER, 1),
		];

		assert_eq!(select_compute_queue_family(&families), None);
	}

	#[test]
	fn cpu_devices_are_not_gpu_fallbacks() {
		assert_eq!(device_type_rank(ash::vk::PhysicalDeviceType::CPU), None);
		assert!(device_type_rank(ash::vk::PhysicalDeviceType::INTEGRATED_GPU).is_some());
	}
}
