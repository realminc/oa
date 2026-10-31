use super::{DeviceConfiguration, DeviceFeatures, ShaderRequirements, add_arithmetic_extensions};

fn configuration(features: DeviceFeatures) -> DeviceConfiguration {
	DeviceConfiguration {
		features,
		extensions: vec![vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME],
		queue_families: vec![2],
		api_version: vk::API_VERSION_1_2,
	}
}

#[test]
fn all_reported_features_enter_the_creation_chain() {
	let configuration = configuration(DeviceFeatures {
		shader_int64: true,
		shader_float64: true,
		shader_float16: true,
		shader_integer_dot_product: true,
		cooperative_matrix: true,
		vulkan_memory_model: true,
		vulkan_memory_model_device_scope: true,
		vulkan_memory_model_availability_visibility_chains: true,
		shader_int8: true,
		timeline_semaphore: true,
		synchronization2: true,
		dynamic_rendering: true,
		..DeviceFeatures::default()
	});
	assert!(configuration.features().shader_float64);
	assert!(configuration.features().shader_float16);
	assert!(configuration.features().shader_integer_dot_product);
	assert!(configuration.features().cooperative_matrix);
	assert_eq!(
		configuration
			.cooperative_matrix_features()
			.cooperative_matrix,
		vk::TRUE
	);
	assert_eq!(
		configuration
			.integer_dot_product_features()
			.shader_integer_dot_product,
		vk::TRUE
	);
	let core = configuration.core_features();
	let features12 = configuration.features12();
	let features13 = configuration.features13();
	assert_eq!(core.shader_float64, vk::TRUE);
	assert_eq!(features12.shader_float16, vk::TRUE);
	assert_eq!(features13.shader_integer_dot_product, vk::TRUE);
	assert_eq!(core.shader_int64, vk::TRUE);
	assert_eq!(features12.shader_int8, vk::TRUE);
	assert_eq!(features12.vulkan_memory_model, vk::TRUE);
	assert_eq!(features12.vulkan_memory_model_device_scope, vk::TRUE);
	assert_eq!(
		features12.vulkan_memory_model_availability_visibility_chains,
		vk::TRUE
	);
	assert_eq!(features12.timeline_semaphore, vk::TRUE);
	assert_eq!(features13.synchronization2, vk::TRUE);
	assert_eq!(features13.dynamic_rendering, vk::TRUE);
	// These returned values are owned, unchained structures. pNext is assembled
	// only while their local borrows remain alive in create_device.
	assert!(features12.p_next.is_null());
	assert!(features13.p_next.is_null());
}

#[test]
fn arithmetic_extension_selection_tracks_core_promotion_and_reported_support() {
	let features = DeviceFeatures {
		cooperative_matrix: true,
		shader_integer_dot_product: true,
		..DeviceFeatures::default()
	};
	let mut extensions = Vec::new();
	add_arithmetic_extensions(&mut extensions, features, vk::API_VERSION_1_2);
	assert_eq!(
		extensions,
		[
			vk::KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
			vk::KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME
		]
	);
	extensions.clear();
	add_arithmetic_extensions(&mut extensions, features, vk::API_VERSION_1_3);
	assert_eq!(extensions, [vk::KHR_COOPERATIVE_MATRIX_EXTENSION_NAME]);
	extensions.clear();
	add_arithmetic_extensions(
		&mut extensions,
		DeviceFeatures::default(),
		vk::API_VERSION_1_2,
	);
	assert!(extensions.is_empty());
	let unsupported = configuration(DeviceFeatures::default());
	assert_eq!(unsupported.core_features().shader_float64, vk::FALSE);
	assert_eq!(unsupported.features12().shader_float16, vk::FALSE);
	assert_eq!(
		unsupported.features13().shader_integer_dot_product,
		vk::FALSE
	);
	assert_eq!(
		unsupported.cooperative_matrix_features().cooperative_matrix,
		vk::FALSE
	);
}

#[test]
fn shader_admission_requires_each_enabled_feature() {
	let requirements = ShaderRequirements {
		shader_int64: true,
		shader_int8: true,
		uniform_and_storage_buffer_8_bit_access: true,
		..ShaderRequirements::default()
	};
	let stages = vk::ShaderStageFlagBits::COMPUTE_BIT;
	let operations = vk::SubgroupFeatureFlagBits::BASIC_BIT;
	let disabled = configuration(DeviceFeatures::default());
	assert_eq!(
		disabled.missing_shader_requirements(stages, operations, requirements),
		[
			"shaderInt64",
			"shaderInt8",
			"uniformAndStorageBuffer8BitAccess"
		]
	);
	let enabled = configuration(DeviceFeatures {
		shader_int64: true,
		shader_int8: true,
		uniform_and_storage_buffer_8_bit_access: true,
		..DeviceFeatures::default()
	});
	assert!(
		enabled
			.missing_shader_requirements(stages, operations, requirements)
			.is_empty()
	);
	assert_eq!(
		enabled.features12().uniform_and_storage_buffer8_bit_access,
		vk::TRUE
	);
}

#[test]
fn subgroup_arithmetic_requires_the_compute_stage_and_operation() {
	let configuration = configuration(DeviceFeatures::default());
	let requirements = ShaderRequirements {
		subgroup_arithmetic: true,
		..ShaderRequirements::default()
	};
	for (stages, operations) in [
		(
			vk::ShaderStageFlagBits::FRAGMENT_BIT,
			vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT,
		),
		(
			vk::ShaderStageFlagBits::COMPUTE_BIT,
			vk::SubgroupFeatureFlagBits::BASIC_BIT,
		),
	] {
		assert_eq!(
			configuration.missing_shader_requirements(stages, operations, requirements),
			["compute subgroup arithmetic operations"]
		);
	}
	assert!(
		configuration
			.missing_shader_requirements(
				vk::ShaderStageFlagBits::COMPUTE_BIT,
				vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT,
				requirements
			)
			.is_empty()
	);
}

#[test]
fn allocated_queues_and_extensions_are_not_inferred_from_reported_support() {
	let configuration = configuration(DeviceFeatures::default());
	assert_eq!(configuration.queue_count(2), 1);
	assert_eq!(configuration.queue_count(3), 0);
	assert!(configuration.extension_enabled(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME));
	assert!(!configuration.extension_enabled(vk::KHR_SWAPCHAIN_EXTENSION_NAME));
	assert!(!configuration.extension_enabled(vk::KHR_COOPERATIVE_MATRIX_EXTENSION_NAME));
}
