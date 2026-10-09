//! Exact logical-device creation policy and enabled-state evidence.
//!
//! Donor: oacpp/source/cpp/lib/oa/runtime/device/deviceBuilder.cpp and
//! device/features/mlFeatures.cpp (supported versus enabled feature chains).
//! Adaptation: Rust ownership/metadata; enables every reported feature in this slice.

use std::ffi::CStr;

use super::{features::DeviceFeatures, physical::DevicePhysical};
use crate::runtime::Instance;
use crate::{Error, Result, runtime::shader::ShaderRequirements};

/// Creation settings become enabled facts only after vkCreateDevice succeeds.
/// Own strings/indices, never self-referential Vulkan pNext pointers.
pub(super) struct DeviceConfiguration {
	features: DeviceFeatures,
	extensions: Vec<&'static CStr>,
	queue_families: Vec<u32>,
	api_version: u32,
}

impl DeviceConfiguration {
	pub(super) fn new(instance: &Instance, physical: &DevicePhysical) -> Self {
		let mut queue_family_indices = vec![physical.compute_queue_family];
		for family in [
			physical.graphics_queue_family,
			physical.video.decode_queue_family,
			physical.video.encode_queue_family,
		]
		.into_iter()
		.flatten()
		{
			if !queue_family_indices.contains(&family) {
				queue_family_indices.push(family);
			}
		}
		let mut extensions = Vec::<&CStr>::new();
		add_arithmetic_extensions(&mut extensions, physical.features, physical.api_version);
		let swapchain_enabled = physical.swapchain_supported && instance.surface_enabled();
		if swapchain_enabled {
			extensions.push(ash::khr::swapchain::NAME);
		}
		if physical.video.decode_queue_family.is_some() || physical.video.encode_queue_family.is_some()
		{
			extensions.push(ash::khr::video_queue::NAME);
		}
		if physical.video.decode_queue_family.is_some() {
			extensions.push(ash::khr::video_decode_queue::NAME);
		}
		if physical.video.h264_decode {
			extensions.push(ash::khr::video_decode_h264::NAME);
		}
		if physical.video.h265_decode {
			extensions.push(ash::khr::video_decode_h265::NAME);
		}
		if physical.video.av1_decode {
			extensions.push(ash::khr::video_decode_av1::NAME);
		}
		if physical.video.vp9_decode {
			extensions.push(c"VK_KHR_video_decode_vp9");
		}
		if physical.video.encode_queue_family.is_some() {
			extensions.push(ash::khr::video_encode_queue::NAME);
		}
		if physical.video.h264_encode {
			extensions.push(ash::khr::video_encode_h264::NAME);
		}
		if physical.video.h265_encode {
			extensions.push(ash::khr::video_encode_h265::NAME);
		}
		if physical.video.av1_encode {
			extensions.push(c"VK_KHR_video_encode_av1");
		}
		if physical.info.local_memory_budget_bytes.is_some() {
			extensions.push(ash::ext::memory_budget::NAME);
		}
		if physical.api_version < ash::vk::API_VERSION_1_3 {
			extensions.push(ash::khr::synchronization2::NAME);
			if physical.features.dynamic_rendering {
				extensions.push(ash::khr::dynamic_rendering::NAME);
			}
		}

		Self {
			features: physical.features,
			extensions,
			queue_families: queue_family_indices,
			api_version: physical.api_version,
		}
	}

	pub(super) fn features(&self) -> DeviceFeatures {
		self.features
	}

	pub(super) fn extension_enabled(&self, name: &CStr) -> bool {
		self.extensions.contains(&name)
	}

	pub(super) fn queue_count(&self, family: u32) -> u32 {
		u32::from(self.queue_families.contains(&family))
	}

	pub(super) fn create_device(
		&self,
		instance: &Instance,
		physical: ash::vk::PhysicalDevice,
	) -> Result<ash::Device> {
		let priorities = [1.0_f32];
		let queue_create_infos: Vec<_> = self
			.queue_families
			.iter()
			.map(|family| {
				ash::vk::DeviceQueueCreateInfo::default()
					.queue_family_index(*family)
					.queue_priorities(&priorities)
			})
			.collect();
		let extension_names: Vec<_> = self.extensions.iter().map(|name| name.as_ptr()).collect();
		let features = self.features;
		let core_features = self.core_features();
		let mut features12 = self.features12();
		let mut features13 = self.features13();
		let mut cooperative_matrix = self.cooperative_matrix_features();
		let mut integer_dot_product = self.integer_dot_product_features();
		let mut synchronization2 = ash::vk::PhysicalDeviceSynchronization2Features::default()
			.synchronization2(features.synchronization2);
		let mut dynamic_rendering = ash::vk::PhysicalDeviceDynamicRenderingFeatures::default()
			.dynamic_rendering(features.dynamic_rendering);
		let mut create_info = ash::vk::DeviceCreateInfo::default()
			.queue_create_infos(&queue_create_infos)
			.enabled_extension_names(&extension_names)
			.enabled_features(&core_features)
			.push_next(&mut features12);
		if self.api_version >= ash::vk::API_VERSION_1_3 {
			create_info = create_info.push_next(&mut features13);
		} else {
			if features.shader_integer_dot_product {
				create_info = create_info.push_next(&mut integer_dot_product);
			}
			create_info = create_info.push_next(&mut synchronization2);
			if features.dynamic_rendering {
				create_info = create_info.push_next(&mut dynamic_rendering);
			}
		}

		if features.cooperative_matrix {
			create_info = create_info.push_next(&mut cooperative_matrix);
		}

		// SAFETY: the physical device and queue families were queried from this
		// instance, queue zero exists in each selected family, priorities are in
		// [0, 1], every enabled feature was reported by the chained feature query,
		// and all create-info references remain alive for the duration of the call.
		unsafe { instance.raw().create_device(physical, &create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "logical-device creation", source))
	}

	fn core_features(&self) -> ash::vk::PhysicalDeviceFeatures {
		let features = self.features;
		ash::vk::PhysicalDeviceFeatures::default()
			.shader_storage_buffer_array_dynamic_indexing(
				features.shader_storage_buffer_array_dynamic_indexing,
			)
			.shader_int64(features.shader_int64)
			.shader_float64(features.shader_float64)
	}

	fn features12(&self) -> ash::vk::PhysicalDeviceVulkan12Features<'static> {
		let features = self.features;
		ash::vk::PhysicalDeviceVulkan12Features::default()
			.shader_float16(features.shader_float16)
			.vulkan_memory_model(features.vulkan_memory_model)
			.vulkan_memory_model_device_scope(features.vulkan_memory_model_device_scope)
			.vulkan_memory_model_availability_visibility_chains(
				features.vulkan_memory_model_availability_visibility_chains,
			)
			.timeline_semaphore(features.timeline_semaphore)
			.storage_buffer8_bit_access(features.storage_buffer_8_bit_access)
			.uniform_and_storage_buffer8_bit_access(features.uniform_and_storage_buffer_8_bit_access)
			.shader_int8(features.shader_int8)
			.runtime_descriptor_array(features.runtime_descriptor_array)
			.descriptor_binding_partially_bound(features.descriptor_binding_partially_bound)
			.descriptor_binding_storage_buffer_update_after_bind(
				features.descriptor_binding_storage_buffer_update_after_bind,
			)
			.descriptor_binding_storage_image_update_after_bind(
				features.descriptor_binding_storage_image_update_after_bind,
			)
			.descriptor_binding_update_unused_while_pending(
				features.descriptor_binding_update_unused_while_pending,
			)
	}

	fn features13(&self) -> ash::vk::PhysicalDeviceVulkan13Features<'static> {
		ash::vk::PhysicalDeviceVulkan13Features::default()
			.shader_integer_dot_product(self.features.shader_integer_dot_product)
			.synchronization2(self.features.synchronization2)
			.dynamic_rendering(self.features.dynamic_rendering)
	}

	fn cooperative_matrix_features(
		&self,
	) -> ash::vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static> {
		ash::vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
			.cooperative_matrix(self.features.cooperative_matrix)
	}

	fn integer_dot_product_features(
		&self,
	) -> ash::vk::PhysicalDeviceShaderIntegerDotProductFeatures<'static> {
		ash::vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
			.shader_integer_dot_product(self.features.shader_integer_dot_product)
	}

	pub(super) fn missing_shader_requirements(
		&self,
		subgroup_stages: ash::vk::ShaderStageFlags,
		subgroup_operations: ash::vk::SubgroupFeatureFlags,
		requirements: ShaderRequirements,
	) -> Vec<&'static str> {
		let features = self.features;
		let mut missing = Vec::new();
		if requirements.shader_int64 && !features.shader_int64 {
			missing.push("shaderInt64");
		}
		if requirements.shader_int8 && !features.shader_int8 {
			missing.push("shaderInt8");
		}
		if requirements.uniform_and_storage_buffer_8_bit_access
			&& !features.uniform_and_storage_buffer_8_bit_access
		{
			missing.push("uniformAndStorageBuffer8BitAccess");
		}
		if requirements.subgroup_basic
			&& (!subgroup_stages.contains(ash::vk::ShaderStageFlags::COMPUTE)
				|| !subgroup_operations.contains(ash::vk::SubgroupFeatureFlags::BASIC))
		{
			missing.push("compute subgroup basic operations");
		}
		if requirements.subgroup_arithmetic
			&& (!subgroup_stages.contains(ash::vk::ShaderStageFlags::COMPUTE)
				|| !subgroup_operations.contains(ash::vk::SubgroupFeatureFlags::ARITHMETIC))
		{
			missing.push("compute subgroup arithmetic operations");
		}
		missing
	}
}

fn add_arithmetic_extensions(
	extensions: &mut Vec<&'static CStr>,
	features: DeviceFeatures,
	api_version: u32,
) {
	if features.cooperative_matrix {
		extensions.push(ash::khr::cooperative_matrix::NAME);
	}
	if api_version < ash::vk::API_VERSION_1_3 && features.shader_integer_dot_product {
		extensions.push(ash::khr::shader_integer_dot_product::NAME);
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/device_configuration_unit.rs"]
mod tests;
