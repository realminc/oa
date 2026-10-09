use crate::{
	Error, Result,
	runtime::{
		BufferAccess, PushConstant,
		executable_graph::{BufferHazard, ComputeNode, ExecutableGraph},
		shader::PhysicalWriteContract,
	},
};

use std::sync::{Arc, OnceLock};

use super::{
	commands::SyncCommands, descriptor::BoundedSets, features::ExecutionProfile,
	pipeline::ComputePipeline,
};
use crate::runtime::{Buffer, Device, TimestampPair};

pub(super) struct CommandPool {
	handle: ash::vk::CommandPool,
	kind: CommandPoolKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CommandPoolKind {
	Compute,
	Graphics,
	VideoDecode,
}

pub(in crate::runtime) struct RecordedCommandBuffer {
	allocation: CommandBufferAllocation,
	timing: Option<TimestampPair>,
	secret_erasures: Vec<crate::runtime::device::secret_buffer::SecretBuffer>,
	retained_secret_erasures: Vec<Arc<crate::runtime::device::secret_buffer::SecretBuffer>>,
	_secret_uses: Vec<Arc<crate::runtime::device::secret_buffer::SecretBuffer>>,
}

enum CommandBufferAllocation {
	Owned {
		handle: ash::vk::CommandBuffer,
		pool: CommandPoolKind,
		_resources: Vec<Buffer>,
		_accesses: Vec<BufferAccess>,
		_bounded_sets: Option<BoundedSets>,
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
	_bounded_sets: Option<BoundedSets>,
	device: Arc<Device>,
}

struct PreparedDispatch<'a> {
	node: &'a ComputeNode,
	pipeline: &'a ComputePipeline,
	push: Vec<u8>,
	physical_write: Option<PhysicalWriteContract>,
}

#[derive(Default)]
struct CommandRetention {
	resources: Vec<Buffer>,
	accesses: Vec<BufferAccess>,
	bounded_sets: Option<BoundedSets>,
}

impl CommandPool {
	pub(super) fn new(
		device: &ash::Device,
		queue_family: u32,
		kind: CommandPoolKind,
	) -> Result<Self> {
		let create_info = ash::vk::CommandPoolCreateInfo::default()
			.flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
			.queue_family_index(queue_family);

		// SAFETY: the queue family was queried from the physical device used to create
		// this live logical device. No allocation callbacks are installed.
		let handle = unsafe { device.create_command_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "command-pool creation", source))?;

		Ok(Self { handle, kind })
	}

	pub(super) fn record_empty(&mut self, device: &ash::Device) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device, ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
		self.finish(
			device,
			command_buffer,
			CommandRetention::default(),
			None,
			None,
		)
	}

	pub(super) fn record_custom(
		&mut self,
		device: &ash::Device,
		record: impl FnOnce(ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device, ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
		if let Err(error) = record(command_buffer) {
			// SAFETY: recording failed before submission and this pool exclusively owns
			// the command buffer, which may be freed from the recording state.
			unsafe {
				device.free_command_buffers(self.handle, &[command_buffer]);
			}
			return Err(error);
		}
		self.finish(
			device,
			command_buffer,
			CommandRetention::default(),
			None,
			None,
		)
	}

	pub(super) fn record_compute_graph(
		&mut self,
		context: &Arc<Device>,
		pipelines: &[Option<OnceLock<ComputePipeline>>],
		descriptor_set: ash::vk::DescriptorSet,
		graph: &ExecutableGraph,
		timing: Option<TimestampPair>,
		reusable_device: Option<Arc<Device>>,
	) -> Result<RecordedCommandBuffer> {
		let device = context.raw();
		let sync_commands = context.sync_commands();
		let prepared = graph
			.nodes()
			.iter()
			.map(|node| prepare_dispatch(pipelines, node, context.physical().profile))
			.collect::<Result<Vec<_>>>()?;
		if prepared.is_empty() {
			return Err(Error::invalid_argument("compute graph has no dispatches"));
		}
		let bounded_sets = if context.physical().profile == ExecutionProfile::Compatibility {
			let layouts = prepared
				.iter()
				.map(|dispatch| {
					context
						.bounded_descriptor_layout(dispatch.node.kernel.bounded_buffer_count())
						.ok_or_else(|| Error::missing_capability("bounded descriptor layout unavailable"))
				})
				.collect::<Result<Vec<_>>>()?;
			let buffers = prepared
				.iter()
				.map(|dispatch| {
					let first = dispatch
						.node
						.buffers
						.first()
						.expect("bounded dispatch has operands");
					(0..dispatch.node.kernel.bounded_buffer_count() as usize)
						.map(|slot| {
							let buffer = dispatch.node.buffers.get(slot).unwrap_or(first);
							ash::vk::DescriptorBufferInfo::default()
								.buffer(buffer.buffer.raw())
								.offset(0)
								.range(buffer.buffer.size())
						})
						.collect::<Vec<_>>()
				})
				.collect::<Vec<_>>();
			Some(BoundedSets::new(context, &layouts, &buffers)?)
		} else {
			None
		};
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
			record_barriers(
				device,
				sync_commands,
				command_buffer,
				graph.barriers_before(node_index),
			);
			let set = bounded_sets
				.as_ref()
				.map_or(descriptor_set, |sets| sets.set(node_index));
			record_dispatch(device, command_buffer, set, dispatch);
			resources.extend(
				dispatch
					.node
					.buffers
					.iter()
					.map(|usage| usage.buffer.clone()),
			);
			accesses.extend(dispatch.node.buffers.iter().map(|usage| usage.access));
		}
		record_host_visibility(device, sync_commands, command_buffer, &accesses);
		if let Some(timing) = &timing {
			timing.record_end(command_buffer);
		}

		self.finish(
			device,
			command_buffer,
			CommandRetention {
				resources,
				accesses,
				bounded_sets,
			},
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
		let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
			.map_err(|source| Error::backend_failure("Vulkan", "command-buffer allocation", source))?;
		let Some(command_buffer) = command_buffers.first().copied() else {
			return Err(Error::backend_failure(
				"Vulkan",
				"command-buffer allocation",
				std::io::Error::other("Vulkan returned no command buffer for a successful allocation"),
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
		retention: CommandRetention,
		timing: Option<TimestampPair>,
		reusable_device: Option<Arc<Device>>,
	) -> Result<RecordedCommandBuffer> {
		let CommandRetention {
			resources,
			accesses,
			bounded_sets,
		} = retention;
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
					_bounded_sets: bounded_sets,
					device,
				}),
			}),
			None => CommandBufferAllocation::Owned {
				handle: command_buffer,
				pool: self.kind,
				_resources: resources,
				_accesses: accesses,
				_bounded_sets: bounded_sets,
			},
		};
		Ok(RecordedCommandBuffer {
			allocation,
			timing,
			secret_erasures: Vec::new(),
			retained_secret_erasures: Vec::new(),
			_secret_uses: Vec::new(),
		})
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
	pipelines: &'a [Option<OnceLock<ComputePipeline>>],
	node: &'a ComputeNode,
	profile: ExecutionProfile,
) -> Result<PreparedDispatch<'a>> {
	if node.kernel.requires_secret_storage() {
		return Err(Error::failed_precondition(
			"secret kernels cannot use ordinary compute graph storage",
		));
	}
	let pipeline = pipelines
		.get(node.kernel.index())
		.and_then(Option::as_ref)
		.and_then(OnceLock::get)
		.ok_or_else(|| {
			Error::missing_capability(format!(
				"{} has no admitted compute pipeline on the selected Vulkan device",
				node.operation
			))
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
	let descriptor_indices = if profile == ExecutionProfile::Compatibility {
		if node.buffers.is_empty() || node.buffers.len() > node.kernel.bounded_buffer_count() as usize {
			return Err(Error::missing_capability(format!(
				"{} needs {} storage-buffer descriptors; Compatibility admits at most {} per dispatch",
				node.operation,
				node.buffers.len(),
				node.kernel.bounded_buffer_count()
			)));
		}
		(0..node.buffers.len() as u32).collect::<Vec<_>>()
	} else {
		node
			.buffers
			.iter()
			.map(|usage| usage.buffer.descriptor_index())
			.collect::<Vec<_>>()
	};
	let push = encode_push_constants(&node.push_constants, &descriptor_indices, node.operation)?;
	if push.len() != pipeline.push_constant_size() as usize {
		return Err(Error::invalid_argument(format!(
			"{} push data contains {} bytes but its reflected kernel ABI requires {}",
			node.operation,
			push.len(),
			pipeline.push_constant_size()
		)));
	}
	let physical_write = node.kernel.artifact().physical_write;
	if let Some(contract) = physical_write {
		let accesses = node
			.buffers
			.iter()
			.map(|usage| usage.access)
			.collect::<Vec<_>>();
		validate_physical_write_accesses(node.operation, &accesses, contract)?;
	}
	Ok(PreparedDispatch {
		node,
		pipeline,
		push,
		physical_write,
	})
}

fn validate_physical_write_accesses(
	operation: &'static str,
	accesses: &[BufferAccess],
	contract: PhysicalWriteContract,
) -> Result<()> {
	let mut declared = vec![false; accesses.len()];
	for write in contract.writes {
		let binding = usize::from(write.binding);
		let Some(access) = accesses.get(binding).copied() else {
			return Err(Error::internal(format!(
				"{operation} physical-write contract names missing binding {binding}"
			)));
		};
		if declared[binding] {
			return Err(Error::internal(format!(
				"{operation} physical-write contract repeats binding {binding}"
			)));
		}
		if access == BufferAccess::Read {
			return Err(Error::internal(format!(
				"{operation} physical-write binding {binding} is bound read-only"
			)));
		}
		declared[binding] = true;
	}
	for (binding, access) in accesses.iter().copied().enumerate() {
		if access != BufferAccess::Read && !declared[binding] {
			return Err(Error::internal(format!(
				"{operation} writable binding {binding} has no physical-write contract"
			)));
		}
	}
	Ok(())
}

fn record_barriers(
	device: &ash::Device,
	sync_commands: &SyncCommands,
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
		sync_commands.pipeline_barrier(device, command_buffer, &dependency);
	}
}

/// Current compute Buffer storage is host-visible. Completion and mapping
/// invalidation do not perform the device-to-host memory domain operation.
/// OA donor: source/cpp/lib/oa/runtime/executableGraph.cpp recordFinalBarrier.
/// Adaptation: mechanical synchronization2 dispatch and checked Rust ownership.
/// Preserve the donor's single memory-domain edge at the batch boundary, so
/// exact Event completion permits blocking and nonblocking host observation.
fn record_host_visibility(
	device: &ash::Device,
	sync_commands: &crate::runtime::device::commands::SyncCommands,
	command_buffer: ash::vk::CommandBuffer,
	accesses: &[BufferAccess],
) {
	if !accesses.iter().any(|access| *access != BufferAccess::Read) {
		return;
	}
	let barrier = ash::vk::MemoryBarrier2::default()
		.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
		.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
		.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
		.dst_access_mask(ash::vk::AccessFlags2::HOST_READ);
	let dependency =
		ash::vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
	// SAFETY: the recording command retains this batch's live host-visible
	// compute buffers. HOST_READ performs the required memory-domain
	// operation; CPU access still requires exact completion and invalidation.
	unsafe { sync_commands.pipeline_barrier(device, command_buffer, &dependency) };
}

fn record_dispatch(
	device: &ash::Device,
	command_buffer: ash::vk::CommandBuffer,
	descriptor_set: ash::vk::DescriptorSet,
	dispatch: &PreparedDispatch<'_>,
) {
	// SAFETY: the command buffer is recording; pipeline, layout, shared descriptor
	// set, and every descriptor-backed buffer remain live through retained graph
	// resources. Preflight proved exact push ABI, legal nonzero workgroups, and
	// agreement between classified writable bindings and the selected artifact's
	// physical-write contract.
	unsafe {
		debug_assert_eq!(
			dispatch.physical_write,
			dispatch.node.kernel.artifact().physical_write
		);
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
	pub(super) fn retain_secret_public_output(
		&mut self,
		public: Buffer,
		sets: Option<BoundedSets>,
	) -> Result<()> {
		self.retain_secret_public_resources(std::iter::once(public), sets)
	}

	pub(super) fn retain_secret_public_resources(
		&mut self,
		public: impl IntoIterator<Item = Buffer>,
		sets: Option<BoundedSets>,
	) -> Result<()> {
		match &mut self.allocation {
			CommandBufferAllocation::Owned {
				_resources,
				_bounded_sets,
				..
			} => {
				if _bounded_sets.is_some() {
					return Err(Error::internal(
						"secret dispatch descriptor retention already set",
					));
				}
				_resources.extend(public);
				*_bounded_sets = sets;
				Ok(())
			}
			CommandBufferAllocation::Reusable(_) => Err(Error::failed_precondition(
				"secret commands cannot be replayed",
			)),
		}
	}

	pub(in crate::runtime) fn retain_secret_erasure(
		&mut self,
		secret: crate::runtime::device::secret_buffer::SecretBuffer,
	) {
		self.secret_erasures.push(secret);
	}

	/// Shared retention is private command lifetime, never a clonable secret value.
	pub(super) fn retain_secret_use(
		&mut self,
		secret: Arc<crate::runtime::device::secret_buffer::SecretBuffer>,
	) {
		self._secret_uses.push(secret);
	}

	pub(super) fn retain_shared_secret_erasure(
		&mut self,
		secret: Arc<crate::runtime::device::secret_buffer::SecretBuffer>,
	) {
		self.retained_secret_erasures.push(secret);
	}

	// Only the retirement worker calls this after a successful timeline wait.
	pub(super) fn confirm_secret_erasures(&self) {
		for secret in self
			.secret_erasures
			.iter()
			.chain(self.retained_secret_erasures.iter().map(Arc::as_ref))
		{
			secret.confirm_erasure();
		}
	}

	pub(super) fn fail_secret_erasures(&self) {
		for secret in self
			.secret_erasures
			.iter()
			.chain(self.retained_secret_erasures.iter().map(Arc::as_ref))
		{
			secret.fail_erasure();
		}
	}

	pub(super) fn raw(&self) -> ash::vk::CommandBuffer {
		match &self.allocation {
			CommandBufferAllocation::Owned { handle, .. } => *handle,
			CommandBufferAllocation::Reusable(command) => command.inner.handle,
		}
	}

	pub(super) fn pool_kind(&self) -> CommandPoolKind {
		match &self.allocation {
			CommandBufferAllocation::Owned { pool, .. } => *pool,
			CommandBufferAllocation::Reusable(_) => CommandPoolKind::Compute,
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

	pub(super) fn into_owned_handle(
		self,
	) -> Option<(ash::vk::CommandBuffer, CommandPoolKind, Option<BoundedSets>)> {
		match self.allocation {
			CommandBufferAllocation::Owned {
				handle,
				pool,
				_bounded_sets,
				..
			} => Some((handle, pool, _bounded_sets)),
			CommandBufferAllocation::Reusable(_) => None,
		}
	}
}

impl ReusableCommandBuffer {
	pub(in crate::runtime) fn submission(&self) -> RecordedCommandBuffer {
		RecordedCommandBuffer {
			allocation: CommandBufferAllocation::Reusable(self.clone()),
			timing: None,
			secret_erasures: Vec::new(),
			retained_secret_erasures: Vec::new(),
			_secret_uses: Vec::new(),
		}
	}
}

impl Drop for ReusableCommandBufferInner {
	fn drop(&mut self) {
		self.device.free_command_buffer_handle(self.handle);
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/secret_dispatch_unit.rs"]
mod secret_dispatch_unit;
