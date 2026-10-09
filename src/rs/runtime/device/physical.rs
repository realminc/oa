use super::features::{DeviceFeatures, ExecutionProfile};
use std::{cmp::Reverse, ffi::CStr};

use crate::runtime::{DeviceHardwareInfo, DeviceInfo, DeviceKind, DeviceSoftwareInfo};

use crate::{DeviceSelection, EngineRequirements, Error, ErrorKind, Result};

use super::super::Instance;
use super::info::{DevicePhysicalInfo, ElectionCandidate, log_device_election};

// OA's per-engine descriptor budgets are independent of Vulkan's device type.
// The actual allocation can shrink further in DescriptorHeap::new.
const MAX_STORAGE_BUFFER_DESCRIPTORS: u32 = 262_144;
const MAX_STORAGE_IMAGE_DESCRIPTORS: u32 = 2_048;

#[derive(Clone, Copy)]
pub(in crate::runtime) struct DeviceLimits {
	pub(in crate::runtime) storage_buffer_descriptors: u32,
	/// Bindless storage image descriptor capacity (binding=1 in the shared set).
	pub(in crate::runtime) storage_image_descriptors: u32,
	pub(in crate::runtime) max_push_constants_size: u32,
	pub(in crate::runtime) max_storage_buffer_range: u32,
	pub(in crate::runtime) max_compute_work_group_count: [u32; 3],
	pub(in crate::runtime) max_compute_work_group_size: [u32; 3],
	pub(in crate::runtime) max_compute_work_group_invocations: u32,
	pub(in crate::runtime) max_compute_shared_memory_size: u32,
	pub(in crate::runtime) subgroup_size: u32,
	pub(in crate::runtime) subgroup_supported_stages: ash::vk::ShaderStageFlags,
	pub(in crate::runtime) subgroup_supported_operations: ash::vk::SubgroupFeatureFlags,
	pub(in crate::runtime) timestamp_period_ns: f64,
	pub(in crate::runtime) compute_timestamp_valid_bits: u32,
}

pub(in crate::runtime) struct DevicePhysical {
	pub(in crate::runtime) handle: ash::vk::PhysicalDevice,
	pub(in crate::runtime) api_version: u32,
	pub(in crate::runtime) profile: ExecutionProfile,
	pub(in crate::runtime) compute_queue_family: u32,
	pub(in crate::runtime) compute_queue_count: u32,
	pub(in crate::runtime) graphics_queue_family: Option<u32>,
	pub(in crate::runtime) graphics_queue_count: u32,
	pub(in crate::runtime) transfer_queue_family: Option<u32>,
	pub(in crate::runtime) transfer_queue_count: u32,
	pub(in crate::runtime) swapchain_supported: bool,
	pub(in crate::runtime) video: VideoPhysicalCapabilities,
	pub(in crate::runtime) features: DeviceFeatures,
	pub(in crate::runtime) cooperative_matrix: CooperativeMatrixCapabilities,
	pub(in crate::runtime) limits: DeviceLimits,
	pub(in crate::runtime) info: DevicePhysicalInfo,
}

#[derive(Clone, Copy, Default)]
pub(in crate::runtime) struct VideoPhysicalCapabilities {
	pub(in crate::runtime) decode_queue_family: Option<u32>,
	pub(in crate::runtime) encode_queue_family: Option<u32>,
	pub(in crate::runtime) decode_result_status_queries: bool,
	pub(in crate::runtime) encode_result_status_queries: bool,
	pub(in crate::runtime) h264_decode: bool,
	pub(in crate::runtime) h265_decode: bool,
	pub(in crate::runtime) av1_decode: bool,
	pub(in crate::runtime) vp9_decode: bool,
	pub(in crate::runtime) h264_encode: bool,
	pub(in crate::runtime) h265_encode: bool,
	pub(in crate::runtime) av1_encode: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::runtime) struct CooperativeMatrixTuple {
	pub(in crate::runtime) m: u32,
	pub(in crate::runtime) n: u32,
	pub(in crate::runtime) k: u32,
	pub(in crate::runtime) a_type: ash::vk::ComponentTypeKHR,
	pub(in crate::runtime) b_type: ash::vk::ComponentTypeKHR,
	pub(in crate::runtime) accumulator_type: ash::vk::ComponentTypeKHR,
	pub(in crate::runtime) result_type: ash::vk::ComponentTypeKHR,
	pub(in crate::runtime) scope: ash::vk::ScopeKHR,
	pub(in crate::runtime) saturating_accumulation: bool,
}

#[derive(Clone, Debug, Default)]
pub(in crate::runtime) struct CooperativeMatrixCapabilities {
	pub(in crate::runtime) supported_stages: ash::vk::ShaderStageFlags,
	pub(in crate::runtime) tuples: Vec<CooperativeMatrixTuple>,
}

#[derive(Clone, Copy)]
struct LocalMemoryFacts {
	capacity_bytes: u64,
	budget_bytes: Option<u64>,
	usage_bytes: Option<u64>,
}

struct DeviceIdentityProperties<'a> {
	physical: &'a ash::vk::PhysicalDeviceProperties,
	driver: &'a ash::vk::PhysicalDeviceDriverProperties<'a>,
	id: &'a ash::vk::PhysicalDeviceIDProperties<'a>,
}

/// Pareto comparison of queried resource limits, not a throughput estimate.
/// Incomparable devices use a deterministic capacity preference; enumeration
/// order resolves exact ties until measured policy exists.
/// Exact-index selection bypasses this policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceCapacityKey {
	shader_float16: bool,
	shader_float64: bool,
	shader_integer_dot_product: bool,
	cooperative_matrix_compute: bool,
	shared_memory_bytes: u32,
	workgroup_invocations: u32,
	local_heap_bytes: u64,
	storage_buffer_descriptors: u32,
}

impl DeviceCapacityKey {
	fn dominates(self, other: Self) -> bool {
		self.shader_float16 >= other.shader_float16
			&& self.shader_float64 >= other.shader_float64
			&& self.shader_integer_dot_product >= other.shader_integer_dot_product
			&& self.cooperative_matrix_compute >= other.cooperative_matrix_compute
			&& self.shared_memory_bytes >= other.shared_memory_bytes
			&& self.workgroup_invocations >= other.workgroup_invocations
			&& self.local_heap_bytes >= other.local_heap_bytes
			&& self.storage_buffer_descriptors >= other.storage_buffer_descriptors
			&& self != other
	}

	fn from_device(device: &DevicePhysical) -> Self {
		Self {
			shader_float16: device.features.shader_float16,
			shader_float64: device.features.shader_float64,
			shader_integer_dot_product: device.features.shader_integer_dot_product,
			cooperative_matrix_compute: device
				.cooperative_matrix
				.supported_stages
				.contains(ash::vk::ShaderStageFlags::COMPUTE)
				&& !device.cooperative_matrix.tuples.is_empty(),
			shared_memory_bytes: device.limits.max_compute_shared_memory_size,
			workgroup_invocations: device.limits.max_compute_work_group_invocations,
			local_heap_bytes: device.info.local_memory_bytes,
			storage_buffer_descriptors: device.limits.storage_buffer_descriptors,
		}
	}

	fn deterministic_preference(self) -> (bool, bool, bool, bool, u32, u32, u32, u64) {
		(
			self.cooperative_matrix_compute,
			self.shader_integer_dot_product,
			self.shader_float16,
			self.shader_float64,
			self.shared_memory_bytes,
			self.workgroup_invocations,
			self.storage_buffer_descriptors,
			self.local_heap_bytes,
		)
	}
}

fn capacity_frontier(capacities: &[DeviceCapacityKey]) -> Vec<bool> {
	capacities
		.iter()
		.enumerate()
		.map(|(candidate_index, capacity)| {
			!capacities
				.iter()
				.enumerate()
				.any(|(other_index, other)| other_index != candidate_index && other.dominates(*capacity))
		})
		.collect()
}

fn preferred_capacity_index(capacities: &[DeviceCapacityKey], ordinals: &[usize]) -> Option<usize> {
	debug_assert_eq!(capacities.len(), ordinals.len());
	let frontier = capacity_frontier(capacities);
	frontier
		.iter()
		.enumerate()
		.filter(|(_, admitted)| **admitted)
		.max_by_key(|(index, _)| {
			(
				capacities[*index].deterministic_preference(),
				Reverse(ordinals[*index]),
			)
		})
		.map(|(index, _)| index)
}

fn descriptor_budget(buffer_limit: u32, image_limit: u32, total_limit: u32) -> Option<(u32, u32)> {
	if buffer_limit < 3 || image_limit == 0 || total_limit < 4 {
		return None;
	}
	// Both bindings consume the same all-pools and per-stage budget. Keep three
	// buffers and one image even on the smallest admitted device.
	let images = image_limit
		.min(MAX_STORAGE_IMAGE_DESCRIPTORS)
		.min(total_limit - 3);
	let buffers = buffer_limit
		.min(MAX_STORAGE_BUFFER_DESCRIPTORS)
		.min(total_limit - images);
	Some((buffers, images))
}

impl DevicePhysical {
	pub(in crate::runtime) fn device_info(&self) -> DeviceInfo {
		DeviceInfo {
			hardware: DeviceHardwareInfo {
				index: self.info.index,
				name: self.info.name.clone(),
				kind: self.info.kind,
				vendor_id: self.info.vendor_id,
				device_id: self.info.device_id,
				local_memory_bytes: self.info.local_memory_bytes,
				local_memory_budget_bytes: self.info.local_memory_budget_bytes,
				local_memory_usage_bytes: self.info.local_memory_usage_bytes,
				max_compute_shared_memory_bytes: self.limits.max_compute_shared_memory_size,
				max_compute_workgroup_invocations: self.limits.max_compute_work_group_invocations,
				subgroup_size: self.limits.subgroup_size,
				storage_buffer_descriptors: self.limits.storage_buffer_descriptors,
				storage_image_descriptors: self.limits.storage_image_descriptors,
				graphics: self.graphics_queue_family.is_some(),
				swapchain: self.swapchain_supported,
			},
			software: DeviceSoftwareInfo {
				api_version: self.info.api_version.clone(),
				driver_name: self.info.driver_name.clone(),
				driver_info: self.info.driver_info.clone(),
				driver_version: self.info.driver_version,
				capability_identity: self.info.capability_identity,
				shader_float16: self.features.shader_float16,
				shader_float64: self.features.shader_float64,
				shader_integer_dot_product: self.features.shader_integer_dot_product,
				cooperative_matrix_tuple_count: self.cooperative_matrix.tuples.len(),
			},
		}
	}

	/// Query identities without allocating execution resources or filtering devices.
	pub(in crate::runtime) fn enumerate(
		instance: &Instance,
	) -> Result<Vec<crate::runtime::DeviceSummary>> {
		let handles = unsafe { instance.raw().enumerate_physical_devices() }
			.map_err(|source| Error::backend_failure("Vulkan", "physical-device enumeration", source))?;
		Ok(
			handles
				.into_iter()
				.enumerate()
				.map(|(index, handle)| {
					// SAFETY: each handle belongs to the retained instance; this query
					// writes only an initialized, caller-owned property record.
					let properties = unsafe { instance.raw().get_physical_device_properties(handle) };
					// SAFETY: Vulkan guarantees a NUL-terminated device_name.
					let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
						.to_string_lossy()
						.into_owned();
					crate::runtime::DeviceSummary {
						index,
						name,
						kind: device_kind(properties.device_type),
						api_version: format_api_version(properties.api_version),
					}
				})
				.collect(),
		)
	}

	pub(in crate::runtime) fn select(
		instance: &Instance,
		selection: DeviceSelection,
		requirements: &EngineRequirements,
	) -> Result<Self> {
		// SAFETY: the instance is live and enumeration only writes Vulkan-owned handles
		// into storage managed by Ash.
		let handles = unsafe { instance.raw().enumerate_physical_devices() }
			.map_err(|source| Error::backend_failure("Vulkan", "physical-device enumeration", source))?;
		let enumerated_count = handles.len();

		if let DeviceSelection::Index(index) = selection
			&& index >= handles.len()
		{
			return Err(Error::invalid_argument(format!(
				"Vulkan device index {index} is out of range for {} enumerated devices",
				handles.len()
			)));
		}

		let mut candidates = Vec::new();
		let mut rejections = Vec::new();
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
			let mut id_properties = ash::vk::PhysicalDeviceIDProperties::default();
			let mut properties12 = ash::vk::PhysicalDeviceVulkan12Properties::default();
			let mut properties2 = ash::vk::PhysicalDeviceProperties2::default()
				.push_next(&mut properties12)
				.push_next(&mut subgroup_properties)
				.push_next(&mut driver_properties)
				.push_next(&mut id_properties);
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
			let memory_budget_extension =
				has_device_extension(instance, handle, ash::ext::memory_budget::NAME)?;
			let local_memory = query_local_memory(instance, handle, memory_budget_extension);
			let queue_families = unsafe {
				instance
					.raw()
					.get_physical_device_queue_family_properties(handle)
			};
			let video = query_video_capabilities(instance, handle, &queue_families)?;
			let swapchain_supported = has_device_extension(instance, handle, ash::khr::swapchain::NAME)?;

			// Exact selection includes software Vulkan, with the same capability
			// admission as hardware. Automatic never silently falls back to CPU.
			if matches!(selection, DeviceSelection::Automatic)
				&& !is_hardware_device(properties.device_type)
			{
				rejections.push(format!(
					"[{index}] {name}: {} requires explicit index selection",
					device_type_label(properties.device_type)
				));
				continue;
			}
			let Some((compute_queue_family, compute_timestamp_valid_bits)) =
				select_compute_queue_family(&queue_families)
			else {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::no_suitable_device(format!(
						"Vulkan device index {index} ({name}) has no compute-capable queue family"
					)));
				}
				rejections.push(format!("[{index}] {name}: no compute-capable queue family"));
				continue;
			};
			let graphics_queue_family = select_graphics_queue_family(&queue_families);
			let transfer_queue_family = select_transfer_queue_family(&queue_families);

			let cooperative_matrix_extension =
				has_device_extension(instance, handle, ash::khr::cooperative_matrix::NAME)?;
			let synchronization2_extension =
				has_device_extension(instance, handle, ash::khr::synchronization2::NAME)?;
			let dynamic_rendering_extension =
				has_device_extension(instance, handle, ash::khr::dynamic_rendering::NAME)?;
			let integer_dot_product_extension =
				has_device_extension(instance, handle, ash::khr::shader_integer_dot_product::NAME)?;
			let features = DeviceFeatures::query(
				instance,
				handle,
				properties.api_version,
				cooperative_matrix_extension,
				synchronization2_extension,
				dynamic_rendering_extension,
				integer_dot_product_extension,
			);
			let profile = if properties.api_version >= ash::vk::API_VERSION_1_3
				&& features
					.missing_requirements(ExecutionProfile::Strict)
					.is_none()
			{
				ExecutionProfile::Strict
			} else {
				ExecutionProfile::Compatibility
			};
			if properties.api_version < ash::vk::API_VERSION_1_2 {
				rejections.push(format!("[{index}] {name}: Vulkan version below 1.2"));
				continue;
			}
			if let Some(missing) = features.missing_requirements(profile) {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} ({name}) lacks required {missing}"
					)));
				}
				rejections.push(format!("[{index}] {name}: missing required {missing}"));
				continue;
			}
			let cooperative_matrix = query_cooperative_matrix_capabilities(
				instance,
				handle,
				cooperative_matrix_extension && features.cooperative_matrix,
			)?;
			let descriptor_limit = properties12
				.max_per_stage_descriptor_update_after_bind_storage_buffers
				.min(properties12.max_descriptor_set_update_after_bind_storage_buffers);
			let storage_image_limit = properties12
				.max_per_stage_descriptor_update_after_bind_storage_images
				.min(properties12.max_descriptor_set_update_after_bind_storage_images);
			let total_descriptor_limit = properties12
				.max_update_after_bind_descriptors_in_all_pools
				.min(properties12.max_per_stage_update_after_bind_resources);
			let budget = if profile == ExecutionProfile::Strict {
				descriptor_budget(
					descriptor_limit,
					storage_image_limit,
					total_descriptor_limit,
				)
			} else {
				let available = properties
					.limits
					.max_per_stage_descriptor_storage_buffers
					.min(properties.limits.max_descriptor_set_storage_buffers)
					.min(crate::runtime::shader::MAX_BOUNDED_STORAGE_BUFFERS);
				(available >= 3).then_some((available, 0))
			};
			let Some((storage_buffer_descriptors, storage_image_descriptors)) = budget else {
				let requirement = if profile == ExecutionProfile::Compatibility {
					"three storage-buffer descriptors per stage and set"
				} else {
					"three storage-buffer descriptors and one storage-image descriptor"
				};
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} ({name}) cannot provide {requirement}"
					)));
				}
				rejections.push(format!("[{index}] {name}: cannot provide {requirement}"));
				continue;
			};
			if properties.limits.max_push_constants_size < 16 {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} cannot provide 16 push-constant bytes"
					)));
				}
				rejections.push(format!(
					"[{index}] {name}: {} push-constant bytes is below the required 16",
					properties.limits.max_push_constants_size
				));
				continue;
			}
			// DescriptorHeap::new may choose a smaller allocation under pressure.
			let limits = DeviceLimits {
				storage_buffer_descriptors,
				storage_image_descriptors,
				max_push_constants_size: properties.limits.max_push_constants_size,
				max_storage_buffer_range: properties.limits.max_storage_buffer_range,
				max_compute_work_group_count: properties.limits.max_compute_work_group_count,
				max_compute_work_group_size: properties.limits.max_compute_work_group_size,
				max_compute_work_group_invocations: properties.limits.max_compute_work_group_invocations,
				max_compute_shared_memory_size: properties.limits.max_compute_shared_memory_size,
				subgroup_size: subgroup_properties.subgroup_size,
				subgroup_supported_stages: subgroup_properties.supported_stages,
				subgroup_supported_operations: subgroup_properties.supported_operations,
				timestamp_period_ns: f64::from(properties.limits.timestamp_period),
				compute_timestamp_valid_bits,
			};
			let capability_identity = capability_identity(
				DeviceIdentityProperties {
					physical: &properties,
					driver: &driver_properties,
					id: &id_properties,
				},
				features,
				limits,
				local_memory.capacity_bytes,
				video,
				&cooperative_matrix,
			);

			let compute_queue_count = queue_families
				.get(compute_queue_family as usize)
				.map_or(1, |f| f.queue_count.max(1));
			let graphics_queue_count = graphics_queue_family
				.and_then(|f| queue_families.get(f as usize))
				.map_or(0, |f| f.queue_count);
			let transfer_queue_count = transfer_queue_family
				.and_then(|f| queue_families.get(f as usize))
				.map_or(0, |f| f.queue_count);
			let candidate = Self {
				handle,
				api_version: properties.api_version,
				profile,
				compute_queue_family,
				compute_queue_count,
				graphics_queue_family,
				graphics_queue_count,
				transfer_queue_family,
				transfer_queue_count,
				swapchain_supported,
				video,
				features,
				cooperative_matrix,
				limits,
				info: DevicePhysicalInfo {
					index,
					name,
					kind: device_kind(properties.device_type),
					device_type: device_type_label(properties.device_type),
					api_version: format_api_version(properties.api_version),
					vendor_id: properties.vendor_id,
					device_id: properties.device_id,
					driver_name,
					driver_info,
					driver_id: driver_properties.driver_id,
					driver_version: properties.driver_version,
					conformance_version: driver_properties.conformance_version,
					capability_identity,
					local_memory_bytes: local_memory.capacity_bytes,
					local_memory_budget_bytes: local_memory.budget_bytes,
					local_memory_usage_bytes: local_memory.usage_bytes,
				},
			};
			if let Some(reason) = candidate.service_rejection(instance, requirements)? {
				if matches!(selection, DeviceSelection::Index(_)) {
					return Err(Error::missing_capability(format!(
						"Vulkan device index {index} ({}) {reason}",
						candidate.info.name
					)));
				}
				rejections.push(format!("[{index}] {}: {reason}", candidate.info.name));
				continue;
			}
			let capacity = DeviceCapacityKey::from_device(&candidate);
			candidates.push((index, capacity, candidate));
		}

		if matches!(selection, DeviceSelection::Automatic)
			&& candidates
				.iter()
				.any(|(_, _, device)| device.profile == ExecutionProfile::Strict)
		{
			candidates.retain(|(_, _, device)| device.profile == ExecutionProfile::Strict);
		}
		let capacities = candidates
			.iter()
			.map(|(_, capacity, _)| *capacity)
			.collect::<Vec<_>>();
		let frontier = capacity_frontier(&capacities);
		if matches!(selection, DeviceSelection::Automatic) {
			let display: Vec<ElectionCandidate<'_>> = candidates
				.iter()
				.enumerate()
				.map(|(i, (index, capacity, candidate))| ElectionCandidate {
					index: *index,
					name: &candidate.info.name,
					local_heap_bytes: capacity.local_heap_bytes,
					cooperative_matrix_compute: capacity.cooperative_matrix_compute,
					shader_float16: capacity.shader_float16,
					shader_integer_dot_product: capacity.shader_integer_dot_product,
					on_frontier: frontier[i],
				})
				.collect();
			log_device_election(enumerated_count, &display, &rejections);
		}
		let ordinals = candidates
			.iter()
			.map(|(index, _, _)| *index)
			.collect::<Vec<_>>();
		let selected = preferred_capacity_index(&capacities, &ordinals)
			.map(|selected_index| candidates.swap_remove(selected_index));

		selected
			.map(|(_, _, physical_device)| physical_device)
			.ok_or_else(|| {
				let mut message =
					"no hardware Vulkan 1.2+ compute device satisfies OA's runtime capabilities".to_owned();
				if !rejections.is_empty() {
					message.push_str("; rejected ");
					message.push_str(&rejections.join("; "));
				}
				Error::no_suitable_device(message)
			})
	}

	fn service_rejection(
		&self,
		instance: &Instance,
		requirements: &EngineRequirements,
	) -> Result<Option<String>> {
		if requirements.graphics && self.profile == ExecutionProfile::Compatibility {
			return Ok(Some(
				"graphics is not yet qualified on the bounded Compatibility profile".to_owned(),
			));
		}
		if requirements.graphics && self.graphics_queue_family.is_none() {
			return Ok(Some("has no graphics-capable queue family".to_owned()));
		}
		if requirements.graphics && !self.features.dynamic_rendering {
			return Ok(Some(
				"lacks dynamicRendering required by OA graphics".to_owned(),
			));
		}
		if requirements.presentation && !self.swapchain_supported {
			return Ok(Some("does not advertise VK_KHR_swapchain".to_owned()));
		}
		if !requirements.video_decode.is_empty() && !self.video.decode_result_status_queries {
			return Ok(Some(
				"lacks decode result-status queries required by OA video sessions".to_owned(),
			));
		}
		for profile in &requirements.video_decode {
			if let Err(error) = super::video::query_decode_capabilities(instance, self, *profile) {
				if error.kind() != ErrorKind::MissingCapability {
					return Err(error);
				}
				return Ok(Some(format!(
					"does not support required decode profile {profile:?}: {error}"
				)));
			}
		}
		if !requirements.video_encode.is_empty() && !self.video.encode_result_status_queries {
			return Ok(Some(
				"lacks encode result-status queries required by OA video sessions".to_owned(),
			));
		}
		for profile in &requirements.video_encode {
			if let Err(error) = super::video::query_encode_capabilities(instance, self, *profile) {
				if error.kind() != ErrorKind::MissingCapability {
					return Err(error);
				}
				return Ok(Some(format!(
					"does not support required encode profile {profile:?}: {error}"
				)));
			}
		}
		Ok(None)
	}
}

fn capability_identity(
	identity: DeviceIdentityProperties<'_>,
	features: DeviceFeatures,
	limits: DeviceLimits,
	local_memory_bytes: u64,
	video: VideoPhysicalCapabilities,
	cooperative_matrix: &CooperativeMatrixCapabilities,
) -> u64 {
	let DeviceIdentityProperties {
		physical: properties,
		driver,
		id,
	} = identity;
	let mut hash = StableHash::new();
	hash.u32(properties.api_version);
	hash.u32(properties.vendor_id);
	hash.u32(properties.device_id);
	hash.bytes(&id.device_uuid);
	hash.bytes(&id.driver_uuid);
	hash.u32(driver.driver_id.as_raw() as u32);
	hash.u32(properties.driver_version);
	hash.bytes(&[
		driver.conformance_version.major,
		driver.conformance_version.minor,
		driver.conformance_version.subminor,
		driver.conformance_version.patch,
	]);
	hash.u64(local_memory_bytes);
	hash.u32(limits.max_compute_shared_memory_size);
	hash.u32(limits.max_compute_work_group_invocations);
	for value in limits.max_compute_work_group_size {
		hash.u32(value);
	}
	for value in limits.max_compute_work_group_count {
		hash.u32(value);
	}
	hash.u32(limits.storage_buffer_descriptors);
	hash.u32(limits.storage_image_descriptors);
	hash.u32(limits.max_push_constants_size);
	hash.u32(limits.subgroup_size);
	hash.u32(limits.subgroup_supported_stages.as_raw());
	hash.u32(limits.subgroup_supported_operations.as_raw());
	hash.bytes(&[
		features.shader_int64 as u8,
		features.shader_float64 as u8,
		features.shader_float16 as u8,
		features.shader_integer_dot_product as u8,
		features.cooperative_matrix as u8,
		features.vulkan_memory_model as u8,
		features.vulkan_memory_model_device_scope as u8,
		features.vulkan_memory_model_availability_visibility_chains as u8,
		features.storage_buffer_8_bit_access as u8,
		features.uniform_and_storage_buffer_8_bit_access as u8,
		features.shader_int8 as u8,
		video.decode_result_status_queries as u8,
		video.encode_result_status_queries as u8,
		video.h264_decode as u8,
		video.h265_decode as u8,
		video.av1_decode as u8,
		video.vp9_decode as u8,
		video.h264_encode as u8,
		video.h265_encode as u8,
		video.av1_encode as u8,
	]);
	hash.u32(cooperative_matrix.supported_stages.as_raw());
	hash.u64(cooperative_matrix.tuples.len() as u64);
	for tuple in &cooperative_matrix.tuples {
		hash.u32(tuple.m);
		hash.u32(tuple.n);
		hash.u32(tuple.k);
		hash.u32(tuple.a_type.as_raw() as u32);
		hash.u32(tuple.b_type.as_raw() as u32);
		hash.u32(tuple.accumulator_type.as_raw() as u32);
		hash.u32(tuple.result_type.as_raw() as u32);
		hash.u32(tuple.scope.as_raw() as u32);
		hash.u32(tuple.saturating_accumulation as u32);
	}
	hash.finish()
}

struct StableHash(u64);

impl StableHash {
	const fn new() -> Self {
		Self(0xcbf2_9ce4_8422_2325)
	}

	fn bytes(&mut self, bytes: &[u8]) {
		for byte in bytes {
			self.0 ^= u64::from(*byte);
			self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
		}
	}

	fn u32(&mut self, value: u32) {
		self.bytes(&value.to_le_bytes());
	}

	fn u64(&mut self, value: u64) {
		self.bytes(&value.to_le_bytes());
	}

	const fn finish(self) -> u64 {
		self.0
	}
}

fn has_device_extension(
	instance: &Instance,
	handle: ash::vk::PhysicalDevice,
	name: &CStr,
) -> Result<bool> {
	// SAFETY: the physical device belongs to the live instance and enumeration
	// writes only extension records owned by the caller.
	let extensions = unsafe { instance.raw().enumerate_device_extension_properties(handle) }
		.map_err(|source| Error::backend_failure("Vulkan", "device-extension enumeration", source))?;
	Ok(extensions.iter().any(|property| {
		// SAFETY: Vulkan guarantees a NUL-terminated extension_name array.
		(unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == name
	}))
}

fn query_local_memory(
	instance: &Instance,
	handle: ash::vk::PhysicalDevice,
	memory_budget_extension: bool,
) -> LocalMemoryFacts {
	let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
	let properties2 = ash::vk::PhysicalDeviceMemoryProperties2::default();
	let mut properties2 = if memory_budget_extension {
		properties2.push_next(&mut budget)
	} else {
		properties2
	};
	// SAFETY: `handle` belongs to this instance. Every output structure remains
	// exclusively borrowed and live for the duration of the query.
	unsafe {
		instance
			.raw()
			.get_physical_device_memory_properties2(handle, &mut properties2);
	}
	let memory = properties2.memory_properties;
	let heap_count = memory
		.memory_heap_count
		.min(memory.memory_heaps.len() as u32) as usize;
	let mut capacity_bytes = 0_u64;
	let mut budget_bytes = 0_u64;
	let mut usage_bytes = 0_u64;
	for (index, heap) in memory.memory_heaps[..heap_count].iter().enumerate() {
		if !heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
			continue;
		}
		capacity_bytes = capacity_bytes.saturating_add(heap.size);
		if memory_budget_extension {
			budget_bytes = budget_bytes.saturating_add(budget.heap_budget[index]);
			usage_bytes = usage_bytes.saturating_add(budget.heap_usage[index]);
		}
	}
	LocalMemoryFacts {
		capacity_bytes,
		budget_bytes: memory_budget_extension.then_some(budget_bytes),
		usage_bytes: memory_budget_extension.then_some(usage_bytes),
	}
}

fn query_cooperative_matrix_capabilities(
	instance: &Instance,
	handle: ash::vk::PhysicalDevice,
	available: bool,
) -> Result<CooperativeMatrixCapabilities> {
	if !available {
		return Ok(CooperativeMatrixCapabilities::default());
	}

	let mut properties = ash::vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
	let mut properties2 = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut properties);
	// SAFETY: the feature and extension were queried from `handle`; the output
	// chain is initialized and remains live for this call.
	unsafe {
		instance
			.raw()
			.get_physical_device_properties2(handle, &mut properties2);
	}
	let supported_stages = properties.cooperative_matrix_supported_stages;
	if !supported_stages.contains(ash::vk::ShaderStageFlags::COMPUTE) {
		return Ok(CooperativeMatrixCapabilities {
			supported_stages,
			tuples: Vec::new(),
		});
	}

	let loader = ash::khr::cooperative_matrix::Instance::new(instance.entry(), instance.raw());
	let enumerate = loader
		.fp()
		.get_physical_device_cooperative_matrix_properties_khr;
	let mut count = 0_u32;
	// SAFETY: the count pointer is valid and a null properties pointer requests
	// only the number of records.
	let first = unsafe { enumerate(handle, &mut count, std::ptr::null_mut()) };
	if first != ash::vk::Result::SUCCESS && first != ash::vk::Result::INCOMPLETE {
		return Err(Error::backend_failure(
			"Vulkan",
			"cooperative-matrix property count query",
			first,
		));
	}
	if count == 0 {
		return Ok(CooperativeMatrixCapabilities {
			supported_stages,
			tuples: Vec::new(),
		});
	}

	for _ in 0..3 {
		let mut raw = vec![ash::vk::CooperativeMatrixPropertiesKHR::default(); count as usize];
		let mut written = count;
		// SAFETY: every output record has its required structure type, `written`
		// describes the initialized allocation, and Vulkan writes at most that many.
		let result = unsafe { enumerate(handle, &mut written, raw.as_mut_ptr()) };
		if result == ash::vk::Result::SUCCESS {
			raw.truncate((written as usize).min(raw.len()));
			let mut tuples = raw
				.into_iter()
				.map(|property| CooperativeMatrixTuple {
					m: property.m_size,
					n: property.n_size,
					k: property.k_size,
					a_type: property.a_type,
					b_type: property.b_type,
					accumulator_type: property.c_type,
					result_type: property.result_type,
					scope: property.scope,
					saturating_accumulation: property.saturating_accumulation == ash::vk::TRUE,
				})
				.collect::<Vec<_>>();
			tuples.sort_unstable();
			tuples.dedup();
			return Ok(CooperativeMatrixCapabilities {
				supported_stages,
				tuples,
			});
		}
		if result != ash::vk::Result::INCOMPLETE {
			return Err(Error::backend_failure(
				"Vulkan",
				"cooperative-matrix property query",
				result,
			));
		}
		let mut required = 0_u32;
		// SAFETY: same count-only query contract as above.
		let retry = unsafe { enumerate(handle, &mut required, std::ptr::null_mut()) };
		if retry != ash::vk::Result::SUCCESS && retry != ash::vk::Result::INCOMPLETE {
			return Err(Error::backend_failure(
				"Vulkan",
				"cooperative-matrix property recount",
				retry,
			));
		}
		count = required.max(written).max(count.saturating_add(1));
	}

	Err(Error::backend_failure(
		"Vulkan",
		"cooperative-matrix property query",
		std::io::Error::other("property count changed repeatedly during enumeration"),
	))
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
			.filter(|(_, properties)| properties.queue_count > 0 && properties.queue_flags.contains(flag))
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

fn device_kind(device_type: ash::vk::PhysicalDeviceType) -> DeviceKind {
	match device_type {
		ash::vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
		ash::vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
		ash::vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
		ash::vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
		_ => DeviceKind::Other,
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

fn select_graphics_queue_family(queue_families: &[ash::vk::QueueFamilyProperties]) -> Option<u32> {
	queue_families
		.iter()
		.enumerate()
		.filter(|(_, properties)| {
			properties.queue_count > 0
				&& properties
					.queue_flags
					.contains(ash::vk::QueueFlags::GRAPHICS)
		})
		.min_by_key(|(index, properties)| {
			(
				!properties
					.queue_flags
					.contains(ash::vk::QueueFlags::COMPUTE),
				*index,
			)
		})
		.and_then(|(index, _)| u32::try_from(index).ok())
}

/// Prefer a transfer-only family (no COMPUTE/GRAPHICS) for the DMA queue; fall
/// back to any family that can transfer if no dedicated one exists.
fn select_transfer_queue_family(queue_families: &[ash::vk::QueueFamilyProperties]) -> Option<u32> {
	let dedicated = queue_families
		.iter()
		.enumerate()
		.find(|(_, properties)| {
			properties.queue_count > 0
				&& properties
					.queue_flags
					.contains(ash::vk::QueueFlags::TRANSFER)
				&& !properties
					.queue_flags
					.contains(ash::vk::QueueFlags::COMPUTE)
				&& !properties
					.queue_flags
					.contains(ash::vk::QueueFlags::GRAPHICS)
		})
		.and_then(|(index, _)| u32::try_from(index).ok());
	if dedicated.is_some() {
		return dedicated;
	}
	// Any family that advertises TRANSFER explicitly (integrated GPUs often use
	// shared families that advertise all three flags together).
	queue_families
		.iter()
		.enumerate()
		.find(|(_, properties)| {
			properties.queue_count > 0
				&& properties
					.queue_flags
					.contains(ash::vk::QueueFlags::TRANSFER)
		})
		.and_then(|(index, _)| u32::try_from(index).ok())
}

fn is_hardware_device(device_type: ash::vk::PhysicalDeviceType) -> bool {
	matches!(
		device_type,
		ash::vk::PhysicalDeviceType::DISCRETE_GPU
			| ash::vk::PhysicalDeviceType::INTEGRATED_GPU
			| ash::vk::PhysicalDeviceType::VIRTUAL_GPU
			| ash::vk::PhysicalDeviceType::OTHER
	)
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/election_unit.rs"]
mod election_tests;
