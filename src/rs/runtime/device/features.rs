use crate::{Error, Result};

use super::super::Instance;

/// Admission policy, separate from device rating and per-kernel eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum ExecutionProfile {
	Strict,
	Compatibility,
}

/// Feature flags for physical queries and the selected logical configuration.
/// A queried flag alone is never proof of logical-device enablement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) struct DeviceFeatures {
	/// Both descriptor profiles index buffer arrays with uniform push data.
	pub(in crate::runtime) shader_storage_buffer_array_dynamic_indexing: bool,
	pub(in crate::runtime) shader_int64: bool,
	pub(in crate::runtime) shader_float64: bool,
	pub(in crate::runtime) shader_float16: bool,
	pub(in crate::runtime) shader_integer_dot_product: bool,
	/// Exact legal tuples live in `DevicePhysical::cooperative_matrix`.
	pub(in crate::runtime) cooperative_matrix: bool,
	pub(in crate::runtime) vulkan_memory_model: bool,
	pub(in crate::runtime) vulkan_memory_model_device_scope: bool,
	pub(in crate::runtime) vulkan_memory_model_availability_visibility_chains: bool,
	pub(in crate::runtime) timeline_semaphore: bool,
	pub(in crate::runtime) synchronization2: bool,
	pub(in crate::runtime) dynamic_rendering: bool,
	pub(in crate::runtime) storage_buffer_8_bit_access: bool,
	pub(in crate::runtime) uniform_and_storage_buffer_8_bit_access: bool,
	pub(in crate::runtime) shader_int8: bool,
	pub(in crate::runtime) runtime_descriptor_array: bool,
	pub(in crate::runtime) descriptor_binding_partially_bound: bool,
	pub(in crate::runtime) descriptor_binding_storage_buffer_update_after_bind: bool,
	pub(in crate::runtime) descriptor_binding_storage_image_update_after_bind: bool,
	pub(in crate::runtime) descriptor_binding_update_unused_while_pending: bool,
}

impl DeviceFeatures {
	pub(in crate::runtime) fn query(
		instance: &Instance,
		physical: vk::PhysicalDevice,
		api_version: u32,
		cooperative_matrix_extension: bool,
		synchronization2_extension: bool,
		dynamic_rendering_extension: bool,
		integer_dot_product_extension: bool,
	) -> Result<Self> {
		let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
		let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
		let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
		let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
		let mut integer_dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
		let mut cooperative_matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
		let mut features2 = vk::PhysicalDeviceFeatures2::default()
			.push_next(&mut features12)
			.map_err(Error::from)?;
		if api_version >= vk::API_VERSION_1_3 {
			features2 = features2.push_next(&mut features13).map_err(Error::from)?;
		} else {
			if integer_dot_product_extension {
				features2 = features2
					.push_next(&mut integer_dot_product)
					.map_err(Error::from)?;
			}
			if synchronization2_extension {
				features2 = features2
					.push_next(&mut synchronization2)
					.map_err(Error::from)?;
			}
			if dynamic_rendering_extension {
				features2 = features2
					.push_next(&mut dynamic_rendering)
					.map_err(Error::from)?;
			}
		}
		let mut features2 = if cooperative_matrix_extension {
			features2
				.push_next(&mut cooperative_matrix)
				.map_err(Error::from)?
		} else {
			features2
		};

		// SAFETY: `physical` belongs to this live instance. The complete output chain
		// remains valid and exclusively borrowed for the duration of the query.
		unsafe {
			instance
				.raw()
				.get_physical_device_features2(physical, &mut features2)?;
		}

		Ok(Self {
			shader_storage_buffer_array_dynamic_indexing: features2
				.features
				.shader_storage_buffer_array_dynamic_indexing
				== vk::TRUE,
			shader_int64: features2.features.shader_int64 == vk::TRUE,
			shader_float64: features2.features.shader_float64 == vk::TRUE,
			shader_float16: features12.shader_float16 == vk::TRUE,
			shader_integer_dot_product: if api_version >= vk::API_VERSION_1_3 {
				features13.shader_integer_dot_product == vk::TRUE
			} else {
				integer_dot_product_extension && integer_dot_product.shader_integer_dot_product == vk::TRUE
			},
			cooperative_matrix: cooperative_matrix_extension
				&& cooperative_matrix.cooperative_matrix == vk::TRUE,
			vulkan_memory_model: features12.vulkan_memory_model == vk::TRUE,
			vulkan_memory_model_device_scope: features12.vulkan_memory_model_device_scope == vk::TRUE,
			vulkan_memory_model_availability_visibility_chains: features12
				.vulkan_memory_model_availability_visibility_chains
				== vk::TRUE,
			timeline_semaphore: features12.timeline_semaphore == vk::TRUE,
			synchronization2: if api_version >= vk::API_VERSION_1_3 {
				features13.synchronization2 == vk::TRUE
			} else {
				synchronization2_extension && synchronization2.synchronization2 == vk::TRUE
			},
			dynamic_rendering: if api_version >= vk::API_VERSION_1_3 {
				features13.dynamic_rendering == vk::TRUE
			} else {
				dynamic_rendering_extension && dynamic_rendering.dynamic_rendering == vk::TRUE
			},
			storage_buffer_8_bit_access: features12.storage_buffer8_bit_access == vk::TRUE,
			uniform_and_storage_buffer_8_bit_access: features12.uniform_and_storage_buffer8_bit_access
				== vk::TRUE,
			shader_int8: features12.shader_int8 == vk::TRUE,
			runtime_descriptor_array: features12.runtime_descriptor_array == vk::TRUE,
			descriptor_binding_partially_bound: features12.descriptor_binding_partially_bound == vk::TRUE,
			descriptor_binding_storage_buffer_update_after_bind: features12
				.descriptor_binding_storage_buffer_update_after_bind
				== vk::TRUE,
			descriptor_binding_storage_image_update_after_bind: features12
				.descriptor_binding_storage_image_update_after_bind
				== vk::TRUE,
			descriptor_binding_update_unused_while_pending: features12
				.descriptor_binding_update_unused_while_pending
				== vk::TRUE,
		})
	}

	pub(in crate::runtime) fn missing_requirements(
		self,
		profile: ExecutionProfile,
	) -> Option<String> {
		let mut requirements = vec![
			(
				self.shader_storage_buffer_array_dynamic_indexing,
				"shaderStorageBufferArrayDynamicIndexing",
			),
			(self.timeline_semaphore, "timelineSemaphore"),
			(self.synchronization2, "synchronization2"),
		];
		if profile == ExecutionProfile::Strict {
			requirements.extend([
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
					self.descriptor_binding_storage_image_update_after_bind,
					"descriptorBindingStorageImageUpdateAfterBind",
				),
				(
					self.descriptor_binding_update_unused_while_pending,
					"descriptorBindingUpdateUnusedWhilePending",
				),
			]);
		}
		let missing = requirements
			.into_iter()
			.filter_map(|(available, name)| (!available).then_some(name))
			.collect::<Vec<_>>();
		(!missing.is_empty()).then(|| missing.join(", "))
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/profile_unit.rs"]
mod profile_tests;
