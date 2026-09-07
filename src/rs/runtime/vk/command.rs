use crate::{
	Error, Result,
	runtime::{BufferAccess, ComputeDispatch, PushConstant},
};

use super::{Buffer, pipeline::ComputePipeline};

pub(super) struct CommandPool {
	handle: ash::vk::CommandPool,
}

pub(in crate::runtime) struct RecordedCommandBuffer {
	handle: ash::vk::CommandBuffer,
	_resources: Vec<Buffer>,
	_accesses: Vec<BufferAccess>,
}

impl CommandPool {
	pub(super) fn new(device: &ash::Device, queue_family: u32) -> Result<Self> {
		let create_info = ash::vk::CommandPoolCreateInfo::default()
			.flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
			.queue_family_index(queue_family);

		// SAFETY: the queue family was queried from the physical device used to create
		// this live logical device. No allocation callbacks are installed.
		let handle = unsafe { device.create_command_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "command-pool creation", source))?;

		Ok(Self { handle })
	}

	pub(super) fn record_empty(&mut self, device: &ash::Device) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device)?;
		self.finish(device, command_buffer, Vec::new(), Vec::new())
	}

	pub(super) fn record_compute(
		&mut self,
		device: &ash::Device,
		pipeline: &ComputePipeline,
		descriptor_set: ash::vk::DescriptorSet,
		buffers: &[Buffer],
		dispatch: &ComputeDispatch<'_>,
	) -> Result<RecordedCommandBuffer> {
		if buffers.len() != dispatch.buffers.len() {
			return Err(Error::invalid_argument(format!(
				"{} resolved buffer count differs from its binding count",
				dispatch.operation
			)));
		}
		if dispatch
			.workgroups
			.into_iter()
			.zip(pipeline.max_dispatch_group_count())
			.any(|(requested, limit)| requested == 0 || requested > limit)
		{
			return Err(Error::invalid_argument(format!(
				"{} workgroup count {:?} is zero or exceeds the selected device limit {:?}",
				dispatch.operation,
				dispatch.workgroups,
				pipeline.max_dispatch_group_count()
			)));
		}
		let push = encode_push_constants(dispatch.push_constants, buffers, dispatch.operation)?;
		if push.len() != pipeline.push_constant_size() as usize {
			return Err(Error::invalid_argument(format!(
				"{} push data contains {} bytes but its reflected kernel ABI requires {}",
				dispatch.operation,
				push.len(),
				pipeline.push_constant_size()
			)));
		}
		let command_buffer = self.begin(device)?;

		// SAFETY: the command buffer is recording; pipeline, layout, shared descriptor set,
		// and every descriptor-backed buffer remain live through the retained
		// resources. Push data exactly matches the reflected ABI, and every workgroup
		// count is nonzero and within the queried device limits.
		unsafe {
			device.cmd_bind_pipeline(
				command_buffer,
				ash::vk::PipelineBindPoint::COMPUTE,
				pipeline.raw(),
			);
			device.cmd_bind_descriptor_sets(
				command_buffer,
				ash::vk::PipelineBindPoint::COMPUTE,
				pipeline.layout(),
				0,
				&[descriptor_set],
				&[],
			);
			device.cmd_push_constants(
				command_buffer,
				pipeline.layout(),
				ash::vk::ShaderStageFlags::COMPUTE,
				0,
				&push,
			);
			device.cmd_dispatch(
				command_buffer,
				dispatch.workgroups[0],
				dispatch.workgroups[1],
				dispatch.workgroups[2],
			);
		}

		let accesses = dispatch
			.buffers
			.iter()
			.map(|binding| binding.access)
			.collect();
		self.finish(device, command_buffer, buffers.to_vec(), accesses)
	}

	fn begin(&mut self, device: &ash::Device) -> Result<ash::vk::CommandBuffer> {
		let allocate_info = ash::vk::CommandBufferAllocateInfo::default()
			.command_pool(self.handle)
			.level(ash::vk::CommandBufferLevel::PRIMARY)
			.command_buffer_count(1);
		// SAFETY: this pool belongs to `device`; the request count is one and the pool
		// remains live until the returned command buffer is explicitly retired.
		let command_buffers =
			unsafe { device.allocate_command_buffers(&allocate_info) }.map_err(|source| {
				Error::backend_failure("Vulkan", "command-buffer allocation", source)
			})?;
		let Some(command_buffer) = command_buffers.first().copied() else {
			return Err(Error::backend_failure(
				"Vulkan",
				"command-buffer allocation",
				std::io::Error::other(
					"Vulkan returned no command buffer for a successful allocation",
				),
			));
		};

		let begin_info = ash::vk::CommandBufferBeginInfo::default()
			.flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
		// SAFETY: the primary command buffer is in its initial state and exclusively
		// owned by this function.
		if let Err(source) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
			// SAFETY: allocation succeeded, recording did not begin, and the buffer is not
			// pending execution.
			unsafe {
				device.free_command_buffers(self.handle, &[command_buffer]);
			}
			return Err(Error::backend_failure(
				"Vulkan",
				"command-buffer begin",
				source,
			));
		}

		Ok(command_buffer)
	}

	fn finish(
		&mut self,
		device: &ash::Device,
		command_buffer: ash::vk::CommandBuffer,
		resources: Vec<Buffer>,
		accesses: Vec<BufferAccess>,
	) -> Result<RecordedCommandBuffer> {
		// SAFETY: the command buffer is recording a complete legal command sequence
		// and is not concurrently accessed.
		if let Err(source) = unsafe { device.end_command_buffer(command_buffer) } {
			// SAFETY: the buffer is not pending execution. `vkEndCommandBuffer` failure
			// leaves it invalid and legal to free.
			unsafe {
				device.free_command_buffers(self.handle, &[command_buffer]);
			}
			return Err(Error::backend_failure(
				"Vulkan",
				"command-buffer end",
				source,
			));
		}

		Ok(RecordedCommandBuffer {
			handle: command_buffer,
			_resources: resources,
			_accesses: accesses,
		})
	}

	pub(super) fn free(&mut self, device: &ash::Device, command: RecordedCommandBuffer) {
		// SAFETY: the retirement service transfers unique ownership of a completed or
		// never-submitted command buffer back to its originating pool exactly once.
		unsafe {
			device.free_command_buffers(self.handle, &[command.handle]);
		}
	}

	pub(super) fn destroy(&mut self, device: &ash::Device) {
		// SAFETY: `DeviceInner` owns this pool, destroys it once after every allocated
		// command buffer has been freed, and uses the same allocation callbacks.
		unsafe {
			device.destroy_command_pool(self.handle, None);
		}
	}
}

fn encode_push_constants(
	constants: &[PushConstant],
	buffers: &[Buffer],
	operation: &'static str,
) -> Result<Vec<u8>> {
	let byte_len = constants
		.len()
		.checked_mul(size_of::<u32>())
		.ok_or_else(|| Error::invalid_argument(format!("{operation} push data size overflows")))?;
	let mut bytes = Vec::with_capacity(byte_len);
	for constant in constants {
		let value = match *constant {
			PushConstant::StorageBuffer(binding) => buffers
				.get(binding)
				.ok_or_else(|| {
					Error::invalid_argument(format!(
						"{operation} push data references missing buffer binding {binding}"
					))
				})?
				.descriptor_index(),
			PushConstant::U32(value) => value,
			PushConstant::F32(value) => value.to_bits(),
		};
		bytes.extend_from_slice(&value.to_ne_bytes());
	}
	Ok(bytes)
}

impl RecordedCommandBuffer {
	pub(super) const fn raw(&self) -> ash::vk::CommandBuffer {
		self.handle
	}
}

#[cfg(test)]
mod tests {
	use super::encode_push_constants;
	use crate::{ErrorKind, runtime::PushConstant};

	#[test]
	fn encodes_typed_u32_push_constants() -> crate::Result<()> {
		let encoded = encode_push_constants(
			&[
				PushConstant::U32(7),
				PushConstant::U32(u32::MAX),
				PushConstant::F32(-1.5),
			],
			&[],
			"test.compute",
		)?;
		let mut expected = Vec::new();
		expected.extend_from_slice(&7_u32.to_ne_bytes());
		expected.extend_from_slice(&u32::MAX.to_ne_bytes());
		expected.extend_from_slice(&(-1.5_f32).to_ne_bytes());
		assert_eq!(encoded, expected);
		Ok(())
	}

	#[test]
	fn rejects_a_missing_push_buffer_binding() {
		let error = encode_push_constants(&[PushConstant::StorageBuffer(0)], &[], "test.compute")
			.expect_err("missing buffer binding was accepted");
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		assert!(error.message().contains("binding 0"));
	}
}
