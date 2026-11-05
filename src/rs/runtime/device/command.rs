use crate::{
	Error, Result,
	runtime::{
		BufferAccess, PushConstant,
		dna::kernel::PhysicalWriteContract,
		executable_graph::{BufferHazard, ComputeNode, ExecutableGraph},
	},
};

use std::sync::{Arc, OnceLock};

use super::{
	commands::SyncCommands, descriptor::BoundedSets, features::ExecutionProfile,
	pipeline::ComputePipeline,
};
use crate::runtime::{Buffer, Device, TimestampPair};

pub(super) struct CommandPool {
	handle: vk::CommandPool,
	kind: CommandPoolKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CommandPoolKind {
	Compute,
	Graphics,
	VideoDecode,
	VideoEncode,
}

pub(in crate::runtime) struct RecordedCommandBuffer {
	allocation: CommandBufferAllocation,
	timing: Option<TimestampPair>,
	/// A caller-owned timer pair stays retained through command retirement but
	/// is not copied onto the submission event: the timer owns the reading, and
	/// reusing the pair must not report newer samples through an older event.
	event_timing_suppressed: bool,
	brackets: Option<Box<[RecordedCommandBuffer; 2]>>,
	secret_erasures: Vec<crate::runtime::device::secret_buffer::SecretBuffer>,
	retained_secret_erasures: Vec<Arc<crate::runtime::device::secret_buffer::SecretBuffer>>,
	_secret_uses: Vec<Arc<crate::runtime::device::secret_buffer::SecretBuffer>>,
}

enum CommandBufferAllocation {
	Owned {
		handle: vk::CommandBuffer,
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
	handle: vk::CommandBuffer,
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
	pub(super) fn new(device: &vk::Device, queue_family: u32, kind: CommandPoolKind) -> Result<Self> {
		let create_info = vk::CommandPoolCreateInfo::new()
			.set_flags(vk::CommandPoolCreateFlagBits::TRANSIENT_BIT)
			.set_queue_family_index(queue_family);

		// SAFETY: the queue family was queried from the physical device used to create
		// this live logical device. No allocation callbacks are installed.
		let handle = unsafe { device.create_command_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "command-pool creation", source))?;

		Ok(Self { handle, kind })
	}

	pub(super) fn record_empty(&mut self, device: &vk::Device) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device, vk::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT)?;
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
		device: &vk::Device,
		record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device, vk::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT)?;
		if let Err(error) = record(command_buffer) {
			// SAFETY: recording failed before submission and this pool exclusively owns
			// the command buffer, which may be freed from the recording state.
			unsafe {
				device.free_command_buffers(self.handle, &[command_buffer])?;
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
		descriptor_set: vk::DescriptorSet,
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
							vk::DescriptorBufferInfo::new()
								.set_buffer(buffer.buffer.raw())
								.set_offset(0)
								.set_range(buffer.buffer.size())
						})
						.collect::<Vec<_>>()
				})
				.collect::<Vec<_>>();
			Some(BoundedSets::new(context, &layouts, &buffers)?)
		} else {
			None
		};
		let usage = if reusable_device.is_some() {
			vk::CommandBufferUsageFlagBits::SIMULTANEOUS_USE_BIT
		} else {
			vk::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT
		};
		let command_buffer = self.begin(device, usage)?;
		let mut resources = Vec::new();
		let mut accesses = Vec::new();
		if let Some(timing) = &timing {
			timing.record_begin(command_buffer)?;
		}

		for (node_index, dispatch) in prepared.iter().enumerate() {
			record_barriers(
				device,
				sync_commands,
				command_buffer,
				graph.barriers_before(node_index),
			)?;
			let set = bounded_sets
				.as_ref()
				.map_or(descriptor_set, |sets| sets.set(node_index));
			record_dispatch(device, command_buffer, set, dispatch)?;
			resources.extend(
				dispatch
					.node
					.buffers
					.iter()
					.map(|usage| usage.buffer.clone()),
			);
			accesses.extend(dispatch.node.buffers.iter().map(|usage| usage.access));
			if let Some(indirect) = &dispatch.node.indirect {
				resources.push(indirect.buffer.clone());
				accesses.push(BufferAccess::Read);
			}
		}
		record_host_visibility(device, sync_commands, command_buffer, &accesses)?;
		if let Some(timing) = &timing {
			timing.record_end(command_buffer)?;
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
		device: &vk::Device,
		usage: vk::CommandBufferUsageFlags,
	) -> Result<vk::CommandBuffer> {
		let allocate_info = vk::CommandBufferAllocateInfo::new()
			.set_command_pool(self.handle)
			.set_level(vk::CommandBufferLevel::PRIMARY)
			.set_command_buffer_count(1);
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

		let begin_info = vk::CommandBufferBeginInfo::new().set_flags(usage);
		// SAFETY: the primary command buffer is in its initial state and exclusively
		// owned by this function.
		if let Err(source) = unsafe { device.begin_command_buffer(command_buffer, &begin_info) } {
			// SAFETY: allocation succeeded, recording did not begin, and the buffer is not
			// pending execution.
			unsafe {
				device.free_command_buffers(self.handle, &[command_buffer])?;
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
		device: &vk::Device,
		command_buffer: vk::CommandBuffer,
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
				device.free_command_buffers(self.handle, &[command_buffer])?;
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
			event_timing_suppressed: false,
			brackets: None,
			secret_erasures: Vec::new(),
			retained_secret_erasures: Vec::new(),
			_secret_uses: Vec::new(),
		})
	}

	pub(super) fn free_handle(&mut self, device: &vk::Device, handle: vk::CommandBuffer) {
		// SAFETY: the retirement service or final reusable owner transfers a completed
		// or never-submitted command buffer back to its originating pool exactly once.
		unsafe {
			let _ = device.free_command_buffers(self.handle, &[handle]);
		}
	}

	pub(super) fn destroy(&mut self, device: &vk::Device) {
		// SAFETY: `DeviceInner` owns this pool, destroys it once after every allocated
		// command buffer has been freed, and uses the same allocation callbacks.
		unsafe {
			let _ = device.destroy_command_pool(self.handle, None);
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
	if node.indirect.is_none()
		&& node
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
		if access != BufferAccess::Read && !declared[binding] && contract.uniform_store.is_none() {
			return Err(Error::internal(format!(
				"{operation} writable binding {binding} has no physical-write contract"
			)));
		}
	}
	Ok(())
}

fn record_barriers(
	device: &vk::Device,
	sync_commands: &SyncCommands,
	command_buffer: vk::CommandBuffer,
	hazards: &[BufferHazard],
) -> Result<()> {
	if hazards.is_empty() {
		return Ok(());
	}
	let barriers = hazards
		.iter()
		.map(|hazard| {
			let stage = |state: crate::runtime::executable_graph::AccessState| {
				let mut stage = vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT;
				if state.uses_indirect_commands() {
					stage = stage | vk::PipelineStageFlagBits2::DRAW_INDIRECT_BIT;
				}
				stage
			};
			vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(stage(hazard.source))
				.set_src_access_mask(hazard.source.vk_access())
				.set_dst_stage_mask(stage(hazard.destination))
				.set_dst_access_mask(hazard.destination.vk_access())
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(hazard.buffer.raw())
				.set_offset(0)
				.set_size(hazard.buffer.size())
		})
		.collect::<Vec<_>>();
	let dependency = vk::DependencyInfo::new()
		.set_buffer_memory_barriers(&barriers)
		.map_err(Error::from)?;

	// SAFETY: every barrier names a live retained buffer. The producer and consumer
	// are storage-buffer or indirect-command accesses in compute dispatches
	// recorded on this same queue family, so COMPUTE_SHADER/DRAW_INDIRECT with
	// SHADER_STORAGE_READ/WRITE and INDIRECT_COMMAND_READ exactly describe both
	// scopes. The full logical buffer range is valid, no image layout exists,
	// and QUEUE_FAMILY_IGNORED declares no ownership transfer.
	unsafe {
		sync_commands.pipeline_barrier(device, command_buffer, &dependency)?;
	}
	Ok(())
}

/// Current compute Buffer storage is host-visible. Completion and mapping
/// invalidation do not perform the device-to-host memory domain operation.
/// OA donor: source/cpp/lib/oa/runtime/executableGraph.cpp recordFinalBarrier.
/// Adaptation: mechanical synchronization2 dispatch and checked Rust ownership.
/// Preserve the donor's single memory-domain edge at the batch boundary, so
/// exact Event completion permits blocking and nonblocking host observation.
fn record_host_visibility(
	device: &vk::Device,
	sync_commands: &crate::runtime::device::commands::SyncCommands,
	command_buffer: vk::CommandBuffer,
	accesses: &[BufferAccess],
) -> Result<()> {
	if !accesses.iter().any(|access| *access != BufferAccess::Read) {
		return Ok(());
	}
	let barrier = vk::MemoryBarrier2::new()
		.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
		.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
		.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT);
	let dependency = vk::DependencyInfo::new()
		.set_memory_barriers(std::slice::from_ref(&barrier))
		.map_err(Error::from)?;
	// SAFETY: the recording command retains this batch's live host-visible
	// compute buffers. HOST_READ performs the required memory-domain
	// operation; CPU access still requires exact completion and invalidation.
	unsafe {
		sync_commands.pipeline_barrier(device, command_buffer, &dependency)?;
	}
	Ok(())
}

fn record_dispatch(
	device: &vk::Device,
	command_buffer: vk::CommandBuffer,
	descriptor_set: vk::DescriptorSet,
	dispatch: &PreparedDispatch<'_>,
) -> Result<()> {
	// SAFETY: the command buffer is recording; pipeline, layout, shared descriptor
	// set, and every descriptor-backed buffer remain live through retained graph
	// resources. Preflight proved exact push ABI, legal nonzero workgroups or a
	// validated GPU-authored indirect-argument range, and agreement between
	// classified writable bindings and the selected artifact's
	// physical-write contract.
	unsafe {
		debug_assert_eq!(
			dispatch.physical_write,
			dispatch.node.kernel.artifact().physical_write
		);
		device.cmd_bind_pipeline(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.raw(),
		)?;
		device.cmd_bind_descriptor_sets(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.layout(),
			0,
			&[descriptor_set],
			&[],
		)?;
		device.cmd_push_constants(
			command_buffer,
			dispatch.pipeline.layout(),
			vk::ShaderStageFlagBits::COMPUTE_BIT,
			0,
			&dispatch.push,
		)?;
		match &dispatch.node.indirect {
			Some(indirect) => {
				device.cmd_dispatch_indirect(command_buffer, indirect.buffer.raw(), indirect.offset)?
			}
			None => device.cmd_dispatch(
				command_buffer,
				dispatch.node.workgroups[0],
				dispatch.node.workgroups[1],
				dispatch.node.workgroups[2],
			)?,
		}
	}
	Ok(())
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
	pub(super) fn with_timing_brackets(
		mut self,
		before: Self,
		after: Self,
		timing: TimestampPair,
	) -> Self {
		self.brackets = Some(Box::new([before, after]));
		self.timing = Some(timing);
		self
	}
	pub(super) fn submission_handles(&self) -> Vec<vk::CommandBuffer> {
		match &self.brackets {
			Some(pair) => vec![pair[0].raw(), self.raw(), pair[1].raw()],
			None => vec![self.raw()],
		}
	}
	pub(super) fn take_brackets(&mut self) -> Option<Box<[Self; 2]>> {
		self.brackets.take()
	}
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

	pub(super) fn raw(&self) -> vk::CommandBuffer {
		match &self.allocation {
			CommandBufferAllocation::Owned { handle, .. } => *handle,
			CommandBufferAllocation::Reusable(command) => command.inner.handle,
		}
	}

	// Classification metadata for deferred secret-scope nodes; never byte data.
	pub(crate) fn has_secret_use(&self) -> bool {
		!self._secret_uses.is_empty()
	}

	pub(crate) fn has_secret_erasure(&self) -> bool {
		!self.secret_erasures.is_empty() || !self.retained_secret_erasures.is_empty()
	}

	pub(crate) fn has_public_resources(&self) -> bool {
		match &self.allocation {
			CommandBufferAllocation::Owned { _resources, .. } => !_resources.is_empty(),
			CommandBufferAllocation::Reusable(_) => false,
		}
	}

	pub(super) fn pool_kind(&self) -> CommandPoolKind {
		match &self.allocation {
			CommandBufferAllocation::Owned { pool, .. } => *pool,
			CommandBufferAllocation::Reusable(_) => CommandPoolKind::Compute,
		}
	}

	pub(in crate::runtime) fn timing(&self) -> Option<TimestampPair> {
		if self.event_timing_suppressed {
			return None;
		}
		self.timing.clone()
	}

	/// Keep a caller-owned timestamp pair retained through command retirement
	/// without exposing it on the submission event.
	pub(in crate::runtime) fn suppress_event_timing(&mut self) {
		self.event_timing_suppressed = true;
	}

	pub(super) fn reusable(&self) -> Option<ReusableCommandBuffer> {
		match &self.allocation {
			CommandBufferAllocation::Owned { .. } => None,
			CommandBufferAllocation::Reusable(command) => Some(command.clone()),
		}
	}

	pub(super) fn into_owned_handle(
		self,
	) -> Option<(vk::CommandBuffer, CommandPoolKind, Option<BoundedSets>)> {
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
			event_timing_suppressed: false,
			brackets: None,
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
