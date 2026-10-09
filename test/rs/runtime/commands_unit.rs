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
		let properties = unsafe { instance.raw().get_physical_device_properties(physical) };
		if properties.api_version < ash::vk::API_VERSION_1_2
			|| properties.api_version >= ash::vk::API_VERSION_1_3
		{
			return None;
		}
		let extensions = unsafe {
			instance
				.raw()
				.enumerate_device_extension_properties(physical)
		}
		.ok()?;
		if !extensions.iter().any(|extension| {
			let name = unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) };
			name == ash::khr::synchronization2::NAME
		}) {
			return None;
		}
		let mut sync = ash::vk::PhysicalDeviceSynchronization2Features::default();
		let mut features12 = ash::vk::PhysicalDeviceVulkan12Features::default();
		let mut features2 = ash::vk::PhysicalDeviceFeatures2::default()
			.push_next(&mut features12)
			.push_next(&mut sync);
		unsafe {
			instance
				.raw()
				.get_physical_device_features2(physical, &mut features2)
		};
		if sync.synchronization2 != ash::vk::TRUE || features12.timeline_semaphore != ash::vk::TRUE {
			return None;
		}
		let families = unsafe {
			instance
				.raw()
				.get_physical_device_queue_family_properties(physical)
		};
		let family = families.iter().position(|family| {
			family.queue_count > 0 && family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
		})?;
		Some((physical, family as u32))
	});
	let Some((physical, family)) = selected else {
		return;
	};
	let priorities = [1.0_f32];
	let queues = [ash::vk::DeviceQueueCreateInfo::default()
		.queue_family_index(family)
		.queue_priorities(&priorities)];
	let extensions = [ash::khr::synchronization2::NAME.as_ptr()];
	let mut features12 = ash::vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
	let mut sync = ash::vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
	let create_info = ash::vk::DeviceCreateInfo::default()
		.queue_create_infos(&queues)
		.enabled_extension_names(&extensions)
		.push_next(&mut features12)
		.push_next(&mut sync);
	let device = unsafe { instance.raw().create_device(physical, &create_info, None) }
		.expect("Vulkan 1.2 device with advertised synchronization2 must be creatable");
	let commands = SyncCommands::new(&instance, &device, ash::vk::API_VERSION_1_2);
	let pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(family);
	let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("command pool");
	let allocation = ash::vk::CommandBufferAllocateInfo::default()
		.command_pool(pool)
		.level(ash::vk::CommandBufferLevel::PRIMARY)
		.command_buffer_count(1);
	let command = unsafe { device.allocate_command_buffers(&allocation) }.expect("command buffer")[0];
	unsafe {
		device
			.begin_command_buffer(command, &ash::vk::CommandBufferBeginInfo::default())
			.expect("begin command buffer");
		let barrier = ash::vk::MemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
			.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
			.dst_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_READ);
		commands.pipeline_barrier(
			&device,
			command,
			&ash::vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
		);
		device
			.end_command_buffer(command)
			.expect("end command buffer");
		let command_infos = [ash::vk::CommandBufferSubmitInfo::default().command_buffer(command)];
		let submits = [ash::vk::SubmitInfo2::default().command_buffer_infos(&command_infos)];
		let queue = device.get_device_queue(family, 0);
		commands
			.submit(&device, queue, &submits, ash::vk::Fence::null())
			.expect("KHR queue submit");
		device.queue_wait_idle(queue).expect("KHR queue completion");
		device.destroy_command_pool(pool, None);
		device.destroy_device(None);
	}
}
