use crate::{
	Error, Result,
	runtime::{
		BufferAccess, PushConstant,
		executable_graph::{BufferHazard, ComputeNode, ExecutableGraph},
	},
};

use std::sync::Arc;

use super::{Buffer, Device, TimestampPair, pipeline::ComputePipeline};

pub(super) struct CommandPool {
	handle: ash::vk::CommandPool,
}

pub(in crate::runtime) struct RecordedCommandBuffer {
	allocation: CommandBufferAllocation,
	timing: Option<TimestampPair>,
}

enum CommandBufferAllocation {
	Owned {
		handle: ash::vk::CommandBuffer,
		_resources: Vec<Buffer>,
		_accesses: Vec<BufferAccess>,
	},
	Reusable(ReusableCommandBuffer),
}

/// Shared ownership of one repeatedly submittable recorded command buffer.
#[derive(Clone)]
pub(in crate::runtime) struct ReusableCommandBuffer {
	inner: Arc<ReusableCommandBufferInner>,
}

struct ReusableCommandBufferInner {
	handle: ash::vk::CommandBuffer,
	_resources: Vec<Buffer>,
	_accesses: Vec<BufferAccess>,
	device: Device,
}

struct PreparedDispatch<'a> {
	node: &'a ComputeNode,
	pipeline: &'a ComputePipeline,
	push: Vec<u8>,
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
		let command_buffer =
			self.begin(device, ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
		self.finish(device, command_buffer, Vec::new(), Vec::new(), None, None)
	}

	pub(super) fn record_compute_graph(
		&mut self,
		device: &ash::Device,
		pipelines: &[ComputePipeline],
		descriptor_set: ash::vk::DescriptorSet,
		graph: &ExecutableGraph,
		timing: Option<TimestampPair>,
		reusable_device: Option<Device>,
	) -> Result<RecordedCommandBuffer> {
		let prepared = graph
			.nodes()
			.iter()
			.map(|node| prepare_dispatch(pipelines, node))
			.collect::<Result<Vec<_>>>()?;
		let usage = if reusable_device.is_some() {
			ash::vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
		} else {
			ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
		};
		let command_buffer = self.begin(device, usage)?;
		let mut resources = Vec::new();
		let mut accesses = Vec::new();
		if let Some(timing) = &timing {
			timing.record_begin(command_buffer);
		}

		for (node_index, dispatch) in prepared.iter().enumerate() {
			record_barriers(device, command_buffer, graph.barriers_before(node_index));
			record_dispatch(device, command_buffer, descriptor_set, dispatch);
			resources.extend(
				dispatch
					.node
					.buffers
					.iter()
					.map(|usage| usage.buffer.clone()),
			);
			accesses.extend(dispatch.node.buffers.iter().map(|usage| usage.access));
		}
		if let Some(timing) = &timing {
			timing.record_end(command_buffer);
		}

		self.finish(
			device,
			command_buffer,
			resources,
			accesses,
			timing,
			reusable_device,
		)
	}

	fn begin(
		&mut self,
		device: &ash::Device,
		usage: ash::vk::CommandBufferUsageFlags,
	) -> Result<ash::vk::CommandBuffer> {
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

		let begin_info = ash::vk::CommandBufferBeginInfo::default().flags(usage);
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
		timing: Option<TimestampPair>,
		reusable_device: Option<Device>,
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

		let allocation = match reusable_device {
			Some(device) => CommandBufferAllocation::Reusable(ReusableCommandBuffer {
				inner: Arc::new(ReusableCommandBufferInner {
					handle: command_buffer,
					_resources: resources,
					_accesses: accesses,
					device,
				}),
			}),
			None => CommandBufferAllocation::Owned {
				handle: command_buffer,
				_resources: resources,
				_accesses: accesses,
			},
		};
		Ok(RecordedCommandBuffer { allocation, timing })
	}

	pub(super) fn free_handle(&mut self, device: &ash::Device, handle: ash::vk::CommandBuffer) {
		// SAFETY: the retirement service or final reusable owner transfers a completed
		// or never-submitted command buffer back to its originating pool exactly once.
		unsafe {
			device.free_command_buffers(self.handle, &[handle]);
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

fn prepare_dispatch<'a>(
	pipelines: &'a [ComputePipeline],
	node: &'a ComputeNode,
) -> Result<PreparedDispatch<'a>> {
	let pipeline = pipelines.get(node.kernel.index()).ok_or_else(|| {
		Error::backend_failure(
			"Vulkan",
			"compute-pipeline resolution",
			std::io::Error::other(format!(
				"{} has no initialized compute pipeline",
				node.operation
			)),
		)
	})?;
	if node
		.workgroups
		.into_iter()
		.zip(pipeline.max_dispatch_group_count())
		.any(|(requested, limit)| requested == 0 || requested > limit)
	{
		return Err(Error::invalid_argument(format!(
			"{} workgroup count {:?} is zero or exceeds the selected device limit {:?}",
			node.operation,
			node.workgroups,
			pipeline.max_dispatch_group_count()
		)));
	}
	let descriptor_indices = node
		.buffers
		.iter()
		.map(|usage| usage.buffer.descriptor_index())
		.collect::<Vec<_>>();
	let push = encode_push_constants(&node.push_constants, &descriptor_indices, node.operation)?;
	if push.len() != pipeline.push_constant_size() as usize {
		return Err(Error::invalid_argument(format!(
			"{} push data contains {} bytes but its reflected kernel ABI requires {}",
			node.operation,
			push.len(),
			pipeline.push_constant_size()
		)));
	}
	Ok(PreparedDispatch {
		node,
		pipeline,
		push,
	})
}

fn record_barriers(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	hazards: &[BufferHazard],
) {
	if hazards.is_empty() {
		return;
	}
	let barriers = hazards
		.iter()
		.map(|hazard| {
			ash::vk::BufferMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
				.src_access_mask(hazard.source.vk_access())
				.dst_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
				.dst_access_mask(hazard.destination.vk_access())
				.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.buffer(hazard.buffer.raw())
				.offset(0)
				.size(hazard.buffer.size())
		})
		.collect::<Vec<_>>();
	let dependency = ash::vk::DependencyInfo::default().buffer_memory_barriers(&barriers);

	// SAFETY: every barrier names a live retained buffer. The producer and consumer
	// are storage-buffer accesses in compute dispatches recorded on this same queue
	// family, so COMPUTE_SHADER with SHADER_STORAGE_READ/WRITE exactly describes
	// both scopes. The full logical buffer range is valid, no image layout exists,
	// and QUEUE_FAMILY_IGNORED declares no ownership transfer.
	unsafe {
		device.cmd_pipeline_barrier2(command_buffer, &dependency);
	}
}

fn record_dispatch(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	descriptor_set: ash::vk::DescriptorSet,
	dispatch: &PreparedDispatch<'_>,
) {
	// SAFETY: the command buffer is recording; pipeline, layout, shared descriptor
	// set, and every descriptor-backed buffer remain live through retained graph
	// resources. Preflight proved exact push ABI and legal nonzero workgroups.
	unsafe {
		device.cmd_bind_pipeline(
			command_buffer,
			ash::vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.raw(),
		);
		device.cmd_bind_descriptor_sets(
			command_buffer,
			ash::vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.layout(),
			0,
			&[descriptor_set],
			&[],
		);
		device.cmd_push_constants(
			command_buffer,
			dispatch.pipeline.layout(),
			ash::vk::ShaderStageFlags::COMPUTE,
			0,
			&dispatch.push,
		);
		device.cmd_dispatch(
			command_buffer,
			dispatch.node.workgroups[0],
			dispatch.node.workgroups[1],
			dispatch.node.workgroups[2],
		);
	}
}

fn encode_push_constants(
	constants: &[PushConstant],
	descriptor_indices: &[u32],
	operation: &'static str,
) -> Result<Vec<u8>> {
	let word_count = descriptor_indices
		.len()
		.checked_add(constants.len())
		.ok_or_else(|| Error::invalid_argument(format!("{operation} push word count overflows")))?;
	let byte_len = word_count
		.checked_mul(size_of::<u32>())
		.ok_or_else(|| Error::invalid_argument(format!("{operation} push data size overflows")))?;
	let mut bytes = Vec::with_capacity(byte_len);
	for index in descriptor_indices {
		bytes.extend_from_slice(&index.to_ne_bytes());
	}
	for constant in constants {
		let value = match *constant {
			PushConstant::U32(value) => value,
			PushConstant::F32(value) => value.to_bits(),
		};
		bytes.extend_from_slice(&value.to_ne_bytes());
	}
	Ok(bytes)
}

impl RecordedCommandBuffer {
	pub(super) fn raw(&self) -> ash::vk::CommandBuffer {
		match &self.allocation {
			CommandBufferAllocation::Owned { handle, .. } => *handle,
			CommandBufferAllocation::Reusable(command) => command.inner.handle,
		}
	}

	pub(in crate::runtime) fn timing(&self) -> Option<TimestampPair> {
		self.timing.clone()
	}

	pub(super) fn reusable(&self) -> Option<ReusableCommandBuffer> {
		match &self.allocation {
			CommandBufferAllocation::Owned { .. } => None,
			CommandBufferAllocation::Reusable(command) => Some(command.clone()),
		}
	}

	pub(super) fn into_owned_handle(self) -> Option<ash::vk::CommandBuffer> {
		match self.allocation {
			CommandBufferAllocation::Owned { handle, .. } => Some(handle),
			CommandBufferAllocation::Reusable(_) => None,
		}
	}
}

impl ReusableCommandBuffer {
	pub(in crate::runtime) fn submission(&self) -> RecordedCommandBuffer {
		RecordedCommandBuffer {
			allocation: CommandBufferAllocation::Reusable(self.clone()),
			timing: None,
		}
	}
}

impl Drop for ReusableCommandBufferInner {
	fn drop(&mut self) {
		self.device.free_command_buffer_handle(self.handle);
	}
}

#[cfg(test)]
mod tests {
	use super::encode_push_constants;
	use crate::runtime::PushConstant;

	#[test]
	fn prepends_bindless_indices_before_typed_push_payload() -> crate::Result<()> {
		let encoded = encode_push_constants(
			&[
				PushConstant::U32(7),
				PushConstant::U32(u32::MAX),
				PushConstant::F32(-1.5),
			],
			&[11, 29],
			"test.compute",
		)?;
		let mut expected = Vec::new();
		expected.extend_from_slice(&11_u32.to_ne_bytes());
		expected.extend_from_slice(&29_u32.to_ne_bytes());
		expected.extend_from_slice(&7_u32.to_ne_bytes());
		expected.extend_from_slice(&u32::MAX.to_ne_bytes());
		expected.extend_from_slice(&(-1.5_f32).to_ne_bytes());
		assert_eq!(encoded, expected);
		Ok(())
	}
}
