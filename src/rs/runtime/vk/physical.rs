use std::ffi::CStr;

use crate::{DeviceSelection, Error, Result};

use super::{Instance, features::DeviceFeatures};

const DISCRETE_STORAGE_BUFFER_CAPACITY: u32 = 1_048_576;
const INTEGRATED_STORAGE_BUFFER_CAPACITY: u32 = 262_144;

#[derive(Clone, Copy)]
pub(super) struct DeviceLimits {
	pub(super) storage_buffer_descriptors: u32,
	pub(super) max_push_constants_size: u32,
	pub(super) max_compute_work_group_count: [u32; 3],
	pub(super) max_compute_work_group_size: [u32; 3],
	pub(super) max_compute_work_group_invocations: u32,
	pub(super) subgroup_supported_stages: ash::vk::ShaderStageFlags,
	pub(super) subgroup_supported_operations: ash::vk::SubgroupFeatureFlags,
	pub(super) timestamp_period_ns: f64,
	pub(super) compute_timestamp_valid_bits: u32,
}

#[derive(Clone)]
pub(in crate::runtime) struct PhysicalDevice {
	pub(super) handle: ash::vk::PhysicalDevice,
	pub(in crate::runtime) compute_queue_family: u32,
	pub(super) video: VideoPhysicalCapabilities,
	pub(super) features: DeviceFeatures,
	pub(super) limits: DeviceLimits,
	pub(in crate::runtime) info: PhysicalDeviceInfo,
}

#[derive(Clone, Copy, Default)]
pub(super) struct VideoPhysicalCapabilities {
	pub(super) decode_queue_family: Option<u32>,
	pub(super) encode_queue_family: Option<u32>,
	pub(super) decode_result_status_queries: bool,
	pub(super) encode_result_status_queries: bool,
	pub(super) h264_decode: bool,
	pub(super) h265_decode: bool,
	pub(super) av1_decode: bool,
	pub(super) vp9_decode: bool,
	pub(super) h264_encode: bool,
	pub(super) h265_encode: bool,
	pub(super) av1_encode: bool,
}

#[derive(Clone)]
pub(in crate::runtime) struct PhysicalDeviceInfo {
	pub(in crate::runtime) name: String,
	pub(in crate::runtime) device_type: &'static str,
	pub(in crate::runtime) api_version: String,
	pub(in crate::runtime) vendor_id: u32,
	pub(in crate::runtime) device_id: u32,
	pub(in crate::runtime) driver_name: String,
	pub(in crate::runtime) driver_info: String,
	pub(in crate::runtime) driver_id: ash::vk::DriverId,
	pub(in crate::runtime) driver_version: u32,
	pub(in crate::runtime) conformance_version: ash::vk::ConformanceVersion,
	pub(in crate::runtime) local_memory_bytes: u64,
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
			let mut subgroup_properties = ash::vk::PhysicalDeviceSubgroupProperties::default();
			let mut driver_properties = ash::vk::PhysicalDeviceDriverProperties::default();
			let mut properties12 = ash::vk::PhysicalDeviceVulkan12Properties::default();
			let mut properties2 = ash::vk::PhysicalDeviceProperties2::default()
				.push_next(&mut properties12)
				.push_next(&mut subgroup_properties)
				.push_next(&mut driver_properties);
			unsafe {
				instance
					.raw()
					.get_physical_device_properties2(handle, &mut properties2);
			}
			let properties = properties2.properties;
			// SAFETY: Vulkan guarantees both queried fixed arrays are NUL-terminated
			// strings for the duration of this call; we immediately copy them.
			let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
				.to_string_lossy()
				.into_owned();
			let driver_name = unsafe { CStr::from_ptr(driver_properties.driver_name.as_ptr()) }
				.to_string_lossy()
				.into_owned();
			let driver_info = unsafe { CStr::from_ptr(driver_properties.driver_info.as_ptr()) }
				.to_string_lossy()
				.into_owned();
			// SAFETY: `handle` belongs to this instance and the query writes a complete
			// value without retaining application memory.
			let memory = unsafe { instance.raw().get_physical_device_memory_properties(handle) };
			let heap_count = memory
				.memory_heap_count
				.min(memory.memory_heaps.len() as u32) as usize;
			let local_memory_bytes = memory.memory_heaps[..heap_count]
				.iter()
				.filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
				.map(|heap| heap.size)
				.sum();
			let queue_families = unsafe {
				instance
					.raw()
					.get_physical_device_queue_family_properties(handle)
			};
			let video = query_video_capabilities(instance, handle, &queue_families)?;

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
			let storage_buffer_capacity = match properties.device_type {
				ash::vk::PhysicalDeviceType::DISCRETE_GPU => DISCRETE_STORAGE_BUFFER_CAPACITY,
				_ => INTEGRATED_STORAGE_BUFFER_CAPACITY,
			};
			let limits = DeviceLimits {
				storage_buffer_descriptors: descriptor_limit.min(storage_buffer_capacity),
				max_push_constants_size: properties.limits.max_push_constants_size,
				max_compute_work_group_count: properties.limits.max_compute_work_group_count,
				max_compute_work_group_size: properties.limits.max_compute_work_group_size,
				max_compute_work_group_invocations: properties
					.limits
					.max_compute_work_group_invocations,
				subgroup_supported_stages: subgroup_properties.supported_stages,
				subgroup_supported_operations: subgroup_properties.supported_operations,
				timestamp_period_ns: f64::from(properties.limits.timestamp_period),
				compute_timestamp_valid_bits,
			};

			let candidate = Self {
				handle,
				compute_queue_family,
				video,
				features,
				limits,
				info: PhysicalDeviceInfo {
					name,
					device_type: device_type_label(properties.device_type),
					api_version: format_api_version(properties.api_version),
					vendor_id: properties.vendor_id,
					device_id: properties.device_id,
					driver_name,
					driver_info,
					driver_id: driver_properties.driver_id,
					driver_version: properties.driver_version,
					conformance_version: driver_properties.conformance_version,
					local_memory_bytes,
				},
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

fn query_video_capabilities(
	instance: &Instance,
	handle: ash::vk::PhysicalDevice,
	queue_families: &[ash::vk::QueueFamilyProperties],
) -> Result<VideoPhysicalCapabilities> {
	// SAFETY: the physical device belongs to the live instance and enumeration
	// only writes immutable extension records owned by the caller.
	let extensions = unsafe { instance.raw().enumerate_device_extension_properties(handle) }
		.map_err(|source| {
			Error::backend_failure("Vulkan", "video device-extension enumeration", source)
		})?;
	let has_extension = |name: &[u8]| {
		extensions.iter().any(|property| {
			// SAFETY: Vulkan guarantees a NUL-terminated extension_name array.
			unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }.to_bytes() == name
		})
	};
	let video_queue = has_extension(b"VK_KHR_video_queue");
	let result_status_support = if video_queue {
		query_video_result_status_support(instance, handle, queue_families.len())
	} else {
		vec![false; queue_families.len()]
	};
	let queue_family = |flag| {
		queue_families
			.iter()
			.enumerate()
			.filter(|(_, properties)| {
				properties.queue_count > 0 && properties.queue_flags.contains(flag)
			})
			.min_by_key(|(index, _)| (!result_status_support[*index], *index))
			.and_then(|(index, _)| u32::try_from(index).ok())
	};
	let decode_queue_family = (video_queue && has_extension(b"VK_KHR_video_decode_queue"))
		.then(|| queue_family(ash::vk::QueueFlags::VIDEO_DECODE_KHR))
		.flatten();
	let encode_queue_family = (video_queue && has_extension(b"VK_KHR_video_encode_queue"))
		.then(|| queue_family(ash::vk::QueueFlags::VIDEO_ENCODE_KHR))
		.flatten();
	Ok(VideoPhysicalCapabilities {
		decode_queue_family,
		encode_queue_family,
		decode_result_status_queries: decode_queue_family
			.and_then(|family| result_status_support.get(family as usize))
			.copied()
			.unwrap_or(false),
		encode_result_status_queries: encode_queue_family
			.and_then(|family| result_status_support.get(family as usize))
			.copied()
			.unwrap_or(false),
		h264_decode: decode_queue_family.is_some() && has_extension(b"VK_KHR_video_decode_h264"),
		h265_decode: decode_queue_family.is_some() && has_extension(b"VK_KHR_video_decode_h265"),
		av1_decode: decode_queue_family.is_some() && has_extension(b"VK_KHR_video_decode_av1"),
		vp9_decode: decode_queue_family.is_some() && has_extension(b"VK_KHR_video_decode_vp9"),
		h264_encode: encode_queue_family.is_some() && has_extension(b"VK_KHR_video_encode_h264"),
		h265_encode: encode_queue_family.is_some() && has_extension(b"VK_KHR_video_encode_h265"),
		av1_encode: encode_queue_family.is_some() && has_extension(b"VK_KHR_video_encode_av1"),
	})
}

fn query_video_result_status_support(
	instance: &Instance,
	handle: ash::vk::PhysicalDevice,
	queue_family_count: usize,
) -> Vec<bool> {
	let mut status =
		vec![ash::vk::QueueFamilyQueryResultStatusPropertiesKHR::default(); queue_family_count];
	let mut properties = status
		.iter_mut()
		.map(|status| ash::vk::QueueFamilyProperties2::default().push_next(status))
		.collect::<Vec<_>>();
	// SAFETY: `handle` belongs to the live instance. Every output structure and
	// its result-status pNext record is initialized and remains live for the call.
	unsafe {
		instance
			.raw()
			.get_physical_device_queue_family_properties2(handle, &mut properties);
	}
	drop(properties);
	status
		.into_iter()
		.map(|properties| properties.query_result_status_support != ash::vk::FALSE)
		.collect()
}

fn format_api_version(version: u32) -> String {
	format!(
		"{}.{}.{}",
		ash::vk::api_version_major(version),
		ash::vk::api_version_minor(version),
		ash::vk::api_version_patch(version)
	)
}

fn device_type_label(device_type: ash::vk::PhysicalDeviceType) -> &'static str {
	match device_type {
		ash::vk::PhysicalDeviceType::DISCRETE_GPU => "discrete GPU",
		ash::vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated GPU",
		ash::vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual GPU",
		ash::vk::PhysicalDeviceType::CPU => "CPU",
		ash::vk::PhysicalDeviceType::OTHER => "other",
		_ => "unknown",
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
