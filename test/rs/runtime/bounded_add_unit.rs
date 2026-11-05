use super::Instance;

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
		let properties = unsafe { instance.raw().get_physical_device_properties(physical) }.ok()?;
		if properties.api_version < vk::API_VERSION_1_2 || properties.api_version >= vk::API_VERSION_1_3
		{
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
	assert_eq!(SHADER.len() % 4, 0, "SPIR-V byte length");
	#[allow(
		clippy::chunks_exact_to_as_chunks,
		reason = "slice::as_chunks is unstable on the pinned toolchain"
	)]
	let words: Vec<u32> = SHADER
		.chunks_exact(4)
		.map(|bytes| u32::from_le_bytes(bytes.try_into().expect("word")))
		.collect();
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
	let Ok(queue_info) = vk::DeviceQueueCreateInfo::new()
		.set_queue_family_index(family)
		.set_queue_priorities(&priorities)
	else {
		return;
	};
	let queues = [queue_info];
	let Ok(create) = vk::DeviceCreateInfo::new().set_queue_create_infos(&queues) else {
		return;
	};
	let device = unsafe { instance.raw().create_device(physical, &create, None) }
		.expect("Vulkan 1.2 logical device");
	let values = 17_usize;
	let byte_count = (values * std::mem::size_of::<f32>()) as vk::DeviceSize;
	let memory = unsafe {
		instance
			.raw()
			.get_physical_device_memory_properties(physical)
	}
	.expect("memory properties");
	let mut buffers = Vec::new();
	for _ in 0..3 {
		let info = vk::BufferCreateInfo::new()
			.set_size(byte_count)
			.set_usage(vk::BufferUsageFlagBits::STORAGE_BUFFER_BIT)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE);
		let buffer = unsafe { device.create_buffer(&info, None) }.expect("storage buffer");
		let requirements =
			unsafe { device.get_buffer_memory_requirements(buffer) }.expect("requirements");
		let wanted =
			vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT | vk::MemoryPropertyFlagBits::HOST_COHERENT_BIT;
		let memory_type = (0..memory.memory_type_count)
			.find(|&index| {
				requirements.memory_type_bits & (1 << index) != 0
					&& (memory.memory_types[index as usize].property_flags & wanted) == wanted
			})
			.expect("host-coherent storage memory");
		let allocate = vk::MemoryAllocateInfo::new()
			.set_allocation_size(requirements.size)
			.set_memory_type_index(memory_type);
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
			device.unmap_memory(buffers[index].1).expect("unmap input");
		}
	}
	let binding = vk::DescriptorSetLayoutBinding::new()
		.set_binding(0)
		.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
		.set_descriptor_count(
			crate::runtime::dna::kernel::KernelId::MatrixAddF32.bounded_buffer_count(),
		)
		.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT);
	let bindings = [binding];
	let layout_info = vk::DescriptorSetLayoutCreateInfo::new()
		.set_bindings(&bindings)
		.expect("layout bindings");
	let descriptor_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
		.expect("bounded descriptor layout");
	let pool_size = vk::DescriptorPoolSize::new()
		.set_type(vk::DescriptorType::STORAGE_BUFFER)
		.set_descriptor_count(
			crate::runtime::dna::kernel::KernelId::MatrixAddF32.bounded_buffer_count(),
		);
	let pool_sizes = [pool_size];
	let pool_info = vk::DescriptorPoolCreateInfo::new()
		.set_max_sets(1)
		.set_pool_sizes(&pool_sizes)
		.expect("pool sizes");
	let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.expect("descriptor pool");
	let layouts = [descriptor_layout];
	let allocate = vk::DescriptorSetAllocateInfo::new()
		.set_descriptor_pool(pool)
		.set_set_layouts(&layouts)
		.expect("set layouts");
	let mut sets = [vk::DescriptorSet::default()];
	unsafe { device.allocate_descriptor_sets(&allocate, &mut sets) }.expect("bounded set");
	let set = sets[0];
	let infos: Vec<_> = (0
		..crate::runtime::dna::kernel::KernelId::MatrixAddF32.bounded_buffer_count() as usize)
		.map(|i| {
			vk::DescriptorBufferInfo::new()
				.set_buffer(buffers[i.min(2)].0)
				.set_range(byte_count)
		})
		.collect();
	let write = vk::WriteDescriptorSet::new()
		.set_dst_set(set)
		.set_dst_binding(0)
		.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
		.set_buffer_info(&infos)
		.expect("buffer infos");
	unsafe { device.update_descriptor_sets(&[write], &[]) }.expect("descriptor write");

	let push_range = vk::PushConstantRange::new()
		.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT)
		.set_size(16);
	let push_ranges = [push_range];
	let pipeline_layout_info = vk::PipelineLayoutCreateInfo::new()
		.set_set_layouts(&layouts)
		.and_then(|info| info.set_push_constant_ranges(&push_ranges))
		.expect("pipeline layout info");
	let pipeline_layout =
		unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.expect("pipeline layout");
	let shader_info = vk::ShaderModuleCreateInfo::new().set_code(&words);
	let shader = unsafe { device.create_shader_module(&shader_info, None) }.expect("shader module");
	let stage = vk::PipelineShaderStageCreateInfo::new()
		.set_stage(vk::ShaderStageFlagBits::COMPUTE_BIT)
		.set_module(shader)
		.set_name(c"main");
	let pipeline_info = vk::ComputePipelineCreateInfo::new()
		.set_stage(stage)
		.set_layout(pipeline_layout);
	let mut pipelines = [vk::Pipeline::NULL];
	let result = unsafe {
		device.create_compute_pipelines(
			vk::PipelineCache::NULL,
			&[pipeline_info],
			None,
			&mut pipelines,
		)
	}
	.expect("bounded compute pipeline");
	assert_eq!(result, vk::Result::SUCCESS, "pipeline creation status");
	let pipeline = pipelines[0];
	let command_pool_info = vk::CommandPoolCreateInfo::new().set_queue_family_index(family);
	let command_pool =
		unsafe { device.create_command_pool(&command_pool_info, None) }.expect("command pool");
	let command_allocate = vk::CommandBufferAllocateInfo::new()
		.set_command_pool(command_pool)
		.set_level(vk::CommandBufferLevel::PRIMARY)
		.set_command_buffer_count(1);
	let command = unsafe { device.allocate_command_buffers(&command_allocate) }.expect("command")[0];
	unsafe {
		device
			.begin_command_buffer(command, &vk::CommandBufferBeginInfo::new())
			.expect("begin command");
		device
			.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline)
			.expect("bind pipeline");
		device
			.cmd_bind_descriptor_sets(
				command,
				vk::PipelineBindPoint::COMPUTE,
				pipeline_layout,
				0,
				&[set],
				&[],
			)
			.expect("bind descriptors");
		let push = [0_u32, 1, 2, values as u32];
		let bytes = std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 16);
		device
			.cmd_push_constants(
				command,
				pipeline_layout,
				vk::ShaderStageFlagBits::COMPUTE_BIT,
				0,
				bytes,
			)
			.expect("push constants");
		device.cmd_dispatch(command, 1, 1, 1).expect("dispatch");
		device.end_command_buffer(command).expect("end command");
		let commands = [command];
		let submit = [vk::SubmitInfo::new()
			.set_command_buffers(&commands)
			.expect("submit commands")];
		let queue = device.get_device_queue(family, 0).expect("device queue");
		device
			.queue_submit(queue, &submit, vk::Fence::NULL)
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
		device.unmap_memory(buffers[2].1).expect("unmap output");
		let _ = device.destroy_command_pool(command_pool, None);
		let _ = device.destroy_pipeline(pipeline, None);
		let _ = device.destroy_shader_module(shader, None);
		let _ = device.destroy_pipeline_layout(pipeline_layout, None);
		let _ = device.destroy_descriptor_pool(pool, None);
		let _ = device.destroy_descriptor_set_layout(descriptor_layout, None);
		for (buffer, allocation) in buffers {
			let _ = device.destroy_buffer(buffer, None);
			let _ = device.free_memory(allocation, None);
		}
		device.destroy(None);
	}
}
