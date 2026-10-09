use super::{DeviceFeatures, ExecutionProfile};

#[test]
fn compatibility_does_not_require_bindless_but_strict_does() {
	let features = DeviceFeatures {
		shader_storage_buffer_array_dynamic_indexing: true,
		timeline_semaphore: true,
		synchronization2: true,
		..DeviceFeatures::default()
	};
	assert_eq!(
		features.missing_requirements(ExecutionProfile::Compatibility),
		None
	);
	assert!(
		features
			.missing_requirements(ExecutionProfile::Strict)
			.is_some_and(|missing| missing.contains("runtimeDescriptorArray"))
	);
}

#[test]
fn compatibility_requires_timeline_and_sync2() {
	let missing = DeviceFeatures::default().missing_requirements(ExecutionProfile::Compatibility);
	assert_eq!(
		missing.as_deref(),
		Some("shaderStorageBufferArrayDynamicIndexing, timelineSemaphore, synchronization2")
	);
}

#[test]
fn both_profiles_require_uniform_buffer_array_indexing() {
	let mut features = DeviceFeatures {
		timeline_semaphore: true,
		synchronization2: true,
		runtime_descriptor_array: true,
		descriptor_binding_partially_bound: true,
		descriptor_binding_storage_buffer_update_after_bind: true,
		descriptor_binding_storage_image_update_after_bind: true,
		descriptor_binding_update_unused_while_pending: true,
		..DeviceFeatures::default()
	};
	for profile in [ExecutionProfile::Compatibility, ExecutionProfile::Strict] {
		assert_eq!(
			features.missing_requirements(profile).as_deref(),
			Some("shaderStorageBufferArrayDynamicIndexing")
		);
	}
	features.shader_storage_buffer_array_dynamic_indexing = true;
	for profile in [ExecutionProfile::Compatibility, ExecutionProfile::Strict] {
		assert_eq!(features.missing_requirements(profile), None);
	}
}
