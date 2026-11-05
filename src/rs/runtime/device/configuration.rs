//! Exact logical-device creation policy and enabled-state evidence.
//!
//! Donor: oacpp/source/cpp/lib/oa/runtime/device/deviceBuilder.cpp and
//! device/features/mlFeatures.cpp (supported versus enabled feature chains).
//! Adaptation: Rust ownership/metadata; enables every reported feature in this slice.

use std::ffi::CStr;

use super::{features::DeviceFeatures, physical::DevicePhysical};
use crate::runtime::Instance;
use crate::{Error, Result, runtime::dna::kernel::ShaderRequirements};

/// Creation settings become enabled facts only after vkCreateDevice succeeds.
/// Own strings/indices, never self-referential Vulkan pNext pointers.
pub(in crate::runtime) struct DeviceConfiguration {
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
			extensions.push(vk::KHR_SWAPCHAIN_EXTENSION_NAME);
		}
		if physical.video.decode_queue_family.is_some() || physical.video.encode_queue_family.is_some()
		{
			extensions.push(vk::KHR_VIDEO_QUEUE_EXTENSION_NAME);
		}
		if physical.video.decode_queue_family.is_some() {
			extensions.push(vk::KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
		}
		if physical.video.h264_decode {
			extensions.push(vk::KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
		}
		if physical.video.h265_decode {
			extensions.push(vk::KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
		}
		if physical.video.av1_decode {
			extensions.push(vk::KHR_VIDEO_DECODE_AV1_EXTENSION_NAME);
		}
		if physical.video.vp9_decode {
			extensions.push(c"VK_KHR_video_decode_vp9");
		}
		if physical.video.encode_queue_family.is_some() {
			extensions.push(vk::KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
		}
		if physical.video.h264_encode {
			extensions.push(vk::KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
		}
		if physical.video.h265_encode {
			extensions.push(vk::KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
		}
		if physical.video.av1_encode {
			extensions.push(c"VK_KHR_video_encode_av1");
		}
		if physical.info.local_memory_budget_bytes.is_some() {
			extensions.push(vk::EXT_MEMORY_BUDGET_EXTENSION_NAME);
		}
		// Donor prefers the KHR spelling when both are advertised.
		if physical.calibrated_timestamps_khr {
			extensions.push(vk::KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		} else if physical.calibrated_timestamps_ext {
			extensions.push(vk::EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		}
		if physical.api_version < vk::API_VERSION_1_3 {
			extensions.push(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
			if physical.features.dynamic_rendering {
				extensions.push(vk::KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
			}
		}

		Self {
			features: physical.features,
			extensions,
			queue_families: queue_family_indices,
			api_version: physical.api_version,
		}
	}

	pub(in crate::runtime) fn features(&self) -> DeviceFeatures {
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
		physical: vk::PhysicalDevice,
	) -> Result<vk::Device> {
		let priorities = [1.0_f32];
		let queue_create_infos: Vec<_> = self
			.queue_families
			.iter()
			.map(|family| {
				vk::DeviceQueueCreateInfo::new()
					.set_queue_family_index(*family)
					.set_queue_priorities(&priorities)
			})
			.collect::<std::result::Result<Vec<_>, _>>()
			.map_err(Error::from)?;
		let extension_names: Vec<_> = self.extensions.iter().map(|name| name.as_ptr()).collect();
		let features = self.features;
		let core_features = self.core_features();
		let mut features12 = self.features12();
		let mut features13 = self.features13();
		let mut cooperative_matrix = self.cooperative_matrix_features();
		let mut integer_dot_product = self.integer_dot_product_features();
		let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::new()
			.set_synchronization2(u32::from(features.synchronization2));
		let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::new()
			.set_dynamic_rendering(u32::from(features.dynamic_rendering));
		let mut create_info = vk::DeviceCreateInfo::new()
			.set_enabled_features(&core_features)
			.push_next(&mut features12)
			.map_err(Error::from)?
			.set_queue_create_infos(&queue_create_infos)
			.map_err(|source| Error::backend_failure("Vulkan", "device create-info queues", source))?
			.set_enabled_extension_names(&extension_names)
			.map_err(|source| {
				Error::backend_failure("Vulkan", "device create-info extensions", source)
			})?;
		if self.api_version >= vk::API_VERSION_1_3 {
			create_info = create_info
				.push_next(&mut features13)
				.map_err(Error::from)?;
		} else {
			if features.shader_integer_dot_product {
				create_info = create_info
					.push_next(&mut integer_dot_product)
					.map_err(Error::from)?;
			}
			create_info = create_info
				.push_next(&mut synchronization2)
				.map_err(Error::from)?;
			if features.dynamic_rendering {
				create_info = create_info
					.push_next(&mut dynamic_rendering)
					.map_err(Error::from)?;
			}
		}

		if features.cooperative_matrix {
			create_info = create_info
				.push_next(&mut cooperative_matrix)
				.map_err(Error::from)?;
		}

		// SAFETY: the physical device and queue families were queried from this
		// instance, queue zero exists in each selected family, priorities are in
		// [0, 1], every enabled feature was reported by the chained feature query,
		// and all create-info references remain alive for the duration of the call.
		unsafe { instance.raw().create_device(physical, &create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "logical-device creation", source))
	}

	fn core_features(&self) -> vk::PhysicalDeviceFeatures {
		let features = self.features;
		vk::PhysicalDeviceFeatures::new()
			.set_shader_storage_buffer_array_dynamic_indexing(u32::from(
				features.shader_storage_buffer_array_dynamic_indexing,
			))
			.set_shader_int64(u32::from(features.shader_int64))
			.set_shader_float64(u32::from(features.shader_float64))
	}

	fn features12(&self) -> vk::PhysicalDeviceVulkan12Features<'static> {
		let features = self.features;
		vk::PhysicalDeviceVulkan12Features::new()
			.set_shader_float16(u32::from(features.shader_float16))
			.set_vulkan_memory_model(u32::from(features.vulkan_memory_model))
			.set_vulkan_memory_model_device_scope(u32::from(features.vulkan_memory_model_device_scope))
			.set_vulkan_memory_model_availability_visibility_chains(u32::from(
				features.vulkan_memory_model_availability_visibility_chains,
			))
			.set_timeline_semaphore(u32::from(features.timeline_semaphore))
			.set_storage_buffer8_bit_access(u32::from(features.storage_buffer_8_bit_access))
			.set_uniform_and_storage_buffer8_bit_access(u32::from(
				features.uniform_and_storage_buffer_8_bit_access,
			))
			.set_shader_int8(u32::from(features.shader_int8))
			.set_runtime_descriptor_array(u32::from(features.runtime_descriptor_array))
			.set_descriptor_binding_partially_bound(u32::from(
				features.descriptor_binding_partially_bound,
			))
			.set_descriptor_binding_storage_buffer_update_after_bind(u32::from(
				features.descriptor_binding_storage_buffer_update_after_bind,
			))
			.set_descriptor_binding_storage_image_update_after_bind(u32::from(
				features.descriptor_binding_storage_image_update_after_bind,
			))
			.set_descriptor_binding_update_unused_while_pending(u32::from(
				features.descriptor_binding_update_unused_while_pending,
			))
	}

	fn features13(&self) -> vk::PhysicalDeviceVulkan13Features<'static> {
		vk::PhysicalDeviceVulkan13Features::new()
			.set_shader_integer_dot_product(u32::from(self.features.shader_integer_dot_product))
			.set_synchronization2(u32::from(self.features.synchronization2))
			.set_dynamic_rendering(u32::from(self.features.dynamic_rendering))
	}

	fn cooperative_matrix_features(&self) -> vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static> {
		vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::new()
			.set_cooperative_matrix(u32::from(self.features.cooperative_matrix))
	}

	fn integer_dot_product_features(
		&self,
	) -> vk::PhysicalDeviceShaderIntegerDotProductFeatures<'static> {
		vk::PhysicalDeviceShaderIntegerDotProductFeatures::new()
			.set_shader_integer_dot_product(u32::from(self.features.shader_integer_dot_product))
	}

	pub(super) fn missing_shader_requirements(
		&self,
		subgroup_stages: vk::ShaderStageFlags,
		subgroup_operations: vk::SubgroupFeatureFlags,
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
			&& ((subgroup_stages & vk::ShaderStageFlagBits::COMPUTE_BIT)
				!= vk::ShaderStageFlagBits::COMPUTE_BIT
				|| (subgroup_operations & vk::SubgroupFeatureFlagBits::BASIC_BIT)
					!= vk::SubgroupFeatureFlagBits::BASIC_BIT)
		{
			missing.push("compute subgroup basic operations");
		}
		if requirements.subgroup_arithmetic
			&& ((subgroup_stages & vk::ShaderStageFlagBits::COMPUTE_BIT)
				!= vk::ShaderStageFlagBits::COMPUTE_BIT
				|| (subgroup_operations & vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT)
					!= vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT)
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
		extensions.push(vk::KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	}
	if api_version < vk::API_VERSION_1_3 && features.shader_integer_dot_product {
		extensions.push(vk::KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/device_configuration_unit.rs"]
mod tests;
