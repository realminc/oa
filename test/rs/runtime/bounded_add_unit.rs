use super::Instance;
use ash::vk;

const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matrix_add_f32_bounded.spv"));

/// A direct Vulkan 1.2 execution proof for the bounded descriptor ABI,
/// independent of Engine admission and graph recording.
#[test]
fn bounded_matrix_add_executes_on_vulkan_1_2() {
	let Ok(instance) = Instance::new(&[]) else {
		return;
	};
	let Ok(physicals) = (unsafe { instance.raw().enumerate_physical_devices() }) else {
		return;
	};
	let selected = physicals.into_iter().find_map(|physical| {
		let properties = unsafe { instance.raw().get_physical_device_properties(physical) };
		if properties.api_version < vk::API_VERSION_1_2 || properties.api_version >= vk::API_VERSION_1_3
		{
			return None;
		}
		let families = unsafe {
			instance
				.raw()
				.get_physical_device_queue_family_properties(physical)
		};
		let family = families.iter().position(|family| {
			family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
		})?;
		Some((physical, family as u32))
	});
	let Some((physical, family)) = selected else {
		return;
	};
	let words = ash::util::read_spv(&mut std::io::Cursor::new(SHADER)).expect("SPIR-V words");
	assert!(words[1] <= 0x0001_0500, "probe must target SPIR-V 1.5");
	let mut offset = 5;
	while offset < words.len() {
		let instruction = words[offset];
		let count = (instruction >> 16) as usize;
		assert!(count > 0 && offset + count <= words.len());
		if instruction as u16 == 17 {
			assert_ne!(
				words[offset + 1],
				5302,
				"RuntimeDescriptorArray is forbidden"
			);
		}
		offset += count;
	}

	let priorities = [1.0_f32];
	let queues = [vk::DeviceQueueCreateInfo::default()
		.queue_family_index(family)
		.queue_priorities(&priorities)];
	let create = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
	let device = unsafe { instance.raw().create_device(physical, &create, None) }
		.expect("Vulkan 1.2 logical device");
	let values = 17_usize;
	let byte_count = (values * std::mem::size_of::<f32>()) as vk::DeviceSize;
	let memory = unsafe {
		instance
			.raw()
			.get_physical_device_memory_properties(physical)
	};
	let mut buffers = Vec::new();
	for _ in 0..3 {
		let info = vk::BufferCreateInfo::default()
			.size(byte_count)
			.usage(vk::BufferUsageFlags::STORAGE_BUFFER)
			.sharing_mode(vk::SharingMode::EXCLUSIVE);
		let buffer = unsafe { device.create_buffer(&info, None) }.expect("storage buffer");
		let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
		let memory_type = (0..memory.memory_type_count)
			.find(|&index| {
				requirements.memory_type_bits & (1 << index) != 0
					&& memory.memory_types[index as usize].property_flags.contains(
						vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
					)
			})
			.expect("host-coherent storage memory");
		let allocate = vk::MemoryAllocateInfo::default()
			.allocation_size(requirements.size)
			.memory_type_index(memory_type);
		let allocation = unsafe { device.allocate_memory(&allocate, None) }.expect("buffer memory");
		unsafe { device.bind_buffer_memory(buffer, allocation, 0) }.expect("bind storage memory");
		buffers.push((buffer, allocation));
	}
	let left: Vec<f32> = (0..values).map(|i| i as f32 * 0.5 - 3.0).collect();
	let right: Vec<f32> = (0..values).map(|i| (i as f32).sin()).collect();
	for (index, source) in [(0, &left), (1, &right)] {
		let mapped =
			unsafe { device.map_memory(buffers[index].1, 0, byte_count, vk::MemoryMapFlags::empty()) }
				.expect("map input");
		unsafe {
			std::ptr::copy_nonoverlapping(source.as_ptr(), mapped.cast::<f32>(), values);
			device.unmap_memory(buffers[index].1);
		}
	}
	let binding = vk::DescriptorSetLayoutBinding::default()
		.binding(0)
		.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
		.descriptor_count(crate::runtime::shader::KernelId::MatrixAddF32.bounded_buffer_count())
		.stage_flags(vk::ShaderStageFlags::COMPUTE);
	let bindings = [binding];
	let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
	let descriptor_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
		.expect("bounded descriptor layout");
	let pool_size = vk::DescriptorPoolSize::default()
		.ty(vk::DescriptorType::STORAGE_BUFFER)
		.descriptor_count(crate::runtime::shader::KernelId::MatrixAddF32.bounded_buffer_count());
	let pool_sizes = [pool_size];
	let pool_info = vk::DescriptorPoolCreateInfo::default()
		.max_sets(1)
		.pool_sizes(&pool_sizes);
	let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.expect("descriptor pool");
	let layouts = [descriptor_layout];
	let allocate = vk::DescriptorSetAllocateInfo::default()
		.descriptor_pool(pool)
		.set_layouts(&layouts);
	let set = unsafe { device.allocate_descriptor_sets(&allocate) }.expect("bounded set")[0];
	let infos: Vec<_> = (0..crate::runtime::shader::KernelId::MatrixAddF32.bounded_buffer_count()
		as usize)
		.map(|i| {
			vk::DescriptorBufferInfo::default()
				.buffer(buffers[i.min(2)].0)
				.range(byte_count)
		})
		.collect();
	let write = vk::WriteDescriptorSet::default()
		.dst_set(set)
		.dst_binding(0)
		.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
		.buffer_info(&infos);
	unsafe { device.update_descriptor_sets(&[write], &[]) };

	let push_range = vk::PushConstantRange::default()
		.stage_flags(vk::ShaderStageFlags::COMPUTE)
		.size(16);
	let push_ranges = [push_range];
	let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
		.set_layouts(&layouts)
		.push_constant_ranges(&push_ranges);
	let pipeline_layout =
		unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.expect("pipeline layout");
	let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
	let shader = unsafe { device.create_shader_module(&shader_info, None) }.expect("shader module");
	let stage = vk::PipelineShaderStageCreateInfo::default()
		.stage(vk::ShaderStageFlags::COMPUTE)
		.module(shader)
		.name(c"main");
	let pipeline_info = vk::ComputePipelineCreateInfo::default()
		.stage(stage)
		.layout(pipeline_layout);
	let pipeline =
		unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }
			.expect("bounded compute pipeline")[0];
	let command_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
	let command_pool =
		unsafe { device.create_command_pool(&command_pool_info, None) }.expect("command pool");
	let command_allocate = vk::CommandBufferAllocateInfo::default()
		.command_pool(command_pool)
		.level(vk::CommandBufferLevel::PRIMARY)
		.command_buffer_count(1);
	let command = unsafe { device.allocate_command_buffers(&command_allocate) }.expect("command")[0];
	unsafe {
		device
			.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
			.expect("begin command");
		device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
		device.cmd_bind_descriptor_sets(
			command,
			vk::PipelineBindPoint::COMPUTE,
			pipeline_layout,
			0,
			&[set],
			&[],
		);
		let push = [0_u32, 1, 2, values as u32];
		let bytes = std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 16);
		device.cmd_push_constants(
			command,
			pipeline_layout,
			vk::ShaderStageFlags::COMPUTE,
			0,
			bytes,
		);
		device.cmd_dispatch(command, 1, 1, 1);
		device.end_command_buffer(command).expect("end command");
		let commands = [command];
		let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
		let queue = device.get_device_queue(family, 0);
		device
			.queue_submit(queue, &submit, vk::Fence::null())
			.expect("submit bounded add");
		device
			.queue_wait_idle(queue)
			.expect("bounded add completion");
		let mapped = device
			.map_memory(buffers[2].1, 0, byte_count, vk::MemoryMapFlags::empty())
			.expect("map output");
		let output = std::slice::from_raw_parts(mapped.cast::<f32>(), values);
		for i in 0..values {
			assert_eq!(output[i], left[i] + right[i], "element {i}");
		}
		device.unmap_memory(buffers[2].1);
		device.destroy_command_pool(command_pool, None);
		device.destroy_pipeline(pipeline, None);
		device.destroy_shader_module(shader, None);
		device.destroy_pipeline_layout(pipeline_layout, None);
		device.destroy_descriptor_pool(pool, None);
		device.destroy_descriptor_set_layout(descriptor_layout, None);
		for (buffer, allocation) in buffers {
			device.destroy_buffer(buffer, None);
			device.free_memory(allocation, None);
		}
		device.destroy_device(None);
	}
}
