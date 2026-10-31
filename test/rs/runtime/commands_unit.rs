use super::SyncCommands;
use crate::runtime::Instance;

/// Exercise the KHR dispatch table on a real Vulkan 1.2 device without
/// claiming that the full OA bounded-binding backend is admitted yet.
#[test]
fn synchronization2_khr_records_and_submits_on_vulkan_1_2() {
	let Ok(instance) = Instance::new(&[]) else {
		return;
	};
	let Ok(physical_devices) = (unsafe { instance.raw().enumerate_physical_devices() }) else {
		return;
	};
	let selected = physical_devices.into_iter().find_map(|physical| {
		// SAFETY: `physical` is an enumerated handle owned by this instance.
		let properties = unsafe { instance.raw().get_physical_device_properties(physical) }.ok()?;
		if properties.api_version < vk::API_VERSION_1_2 || properties.api_version >= vk::API_VERSION_1_3
		{
			return None;
		}
		let extensions = unsafe {
			instance
				.raw()
				.enumerate_device_extension_properties(physical, None)
		}
		.ok()?;
		if !extensions.iter().any(|extension| {
			let name = unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) };
			name == vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME
		}) {
			return None;
		}
		let mut sync = vk::PhysicalDeviceSynchronization2Features::default();
		let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
		let mut features2 = vk::PhysicalDeviceFeatures2::default()
			.push_next(&mut features12)
			.ok()?
			.push_next(&mut sync)
			.ok()?;
		unsafe {
			instance
				.raw()
				.get_physical_device_features2(physical, &mut features2)
		}
		.ok()?;
		if sync.synchronization2 != vk::TRUE || features12.timeline_semaphore != vk::TRUE {
			return None;
		}
		let families = unsafe {
			instance
				.raw()
				.get_physical_device_queue_family_properties(physical)
		}
		.ok()?;
		let family = families.iter().position(|family| {
			family.queue_count > 0
				&& (family.queue_flags & vk::QueueFlagBits::COMPUTE_BIT) == vk::QueueFlagBits::COMPUTE_BIT
		})?;
		Some((physical, family as u32))
	});
	let Some((physical, family)) = selected else {
		return;
	};
	let priorities = [1.0_f32];
	let Ok(queue_info) = vk::DeviceQueueCreateInfo::default()
		.queue_family_index(family)
		.queue_priorities(&priorities)
	else {
		return;
	};
	let queues = [queue_info];
	let extensions = [vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME.as_ptr()];
	let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(vk::TRUE);
	let mut sync = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(vk::TRUE);
	let Ok(create_info) = vk::DeviceCreateInfo::default()
		.queue_create_infos(&queues)
		.and_then(|info| info.enabled_extension_names(&extensions))
		.and_then(|info| info.push_next(&mut features12))
		.and_then(|info| info.push_next(&mut sync))
	else {
		return;
	};
	// SAFETY: `physical` and `family` were queried from this instance, the
	// advertised extensions/features were confirmed above, and every referenced
	// create-info input outlives the call.
	let device = unsafe { instance.raw().create_device(physical, &create_info, None) }
		.expect("Vulkan 1.2 device with advertised synchronization2 must be creatable");
	let commands = SyncCommands::new(&instance, &device, vk::API_VERSION_1_2);
	let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
	let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("command pool");
	let allocation = vk::CommandBufferAllocateInfo::default()
		.command_pool(pool)
		.level(vk::CommandBufferLevel::PRIMARY)
		.command_buffer_count(1);
	let command = unsafe { device.allocate_command_buffers(&allocation) }.expect("command buffer")[0];
	unsafe {
		device
			.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
			.expect("begin command buffer");
		let barrier = vk::MemoryBarrier2::default()
			.src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
			.src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
			.dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
			.dst_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT);
		let dependency = vk::DependencyInfo::default()
			.memory_barriers(std::slice::from_ref(&barrier))
			.expect("dependency info");
		commands
			.pipeline_barrier(&device, command, &dependency)
			.expect("KHR pipeline barrier");
		device
			.end_command_buffer(command)
			.expect("end command buffer");
		let command_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
		let submits = [vk::SubmitInfo2::default()
			.command_buffer_infos(&command_infos)
			.expect("submit info")];
		let queue = device.get_device_queue(family, 0).expect("device queue");
		commands
			.submit(&device, queue, &submits, vk::Fence::default())
			.expect("KHR queue submit");
		device.queue_wait_idle(queue).expect("KHR queue completion");
		let _ = device.destroy_command_pool(pool, None);
		device.destroy(None);
	}
}
