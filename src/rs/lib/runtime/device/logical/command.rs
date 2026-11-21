use crate::{
	Error, Result,
	runtime::{
		BufferAccess, PushConstant,
		dna::{kernel::PhysicalWriteContract, tuning::ObservedRoute},
		graph::executable::{BufferCopyNode, BufferHazard, ComputeNode, ExecutableGraph, GraphNode},
	},
};

use std::sync::{Arc, OnceLock};

use super::{
	descriptor::BoundedSets,
	pipeline::ComputePipeline,
	timestamp::RouteTimings,
	video::{DecodeStatus, EncodeReceipt, PlaneCopyPhase},
};
#[cfg(oa_profile = "compatibility")]
use crate::runtime::graph::executable::BufferUse;
use crate::runtime::{Buffer, Device, TimestampPair};

pub(in crate::runtime::device) struct CommandPool {
	handle: vk::CommandPool,
	kind: CommandPoolKind,
	/// Queue family every buffer allocated from this pool records for. A
	/// graph range whose resolved family differs must not be recorded here.
	queue_family: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum CommandPoolKind {
	Compute,
	Graphics,
	Transfer,
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
	retained: Option<Arc<dyn Send + Sync>>,
	pub(in crate::runtime) decode_statuses: Vec<Arc<DecodeStatus>>,
	pub(in crate::runtime) encode_receipts: Vec<Arc<EncodeReceipt>>,
	secret_erasures: Vec<crate::runtime::device::logical::secret_buffer::SecretBuffer>,
	retained_secret_erasures: Vec<Arc<crate::runtime::device::logical::secret_buffer::SecretBuffer>>,
	_secret_uses: Vec<Arc<crate::runtime::device::logical::secret_buffer::SecretBuffer>>,
}

/// One ordered batch of commands recorded before the first submission.
pub(in crate::runtime) struct RecordedBatch {
	device: Arc<Device>,
	commands: std::collections::VecDeque<RecordedCommandBuffer>,
}

impl RecordedBatch {
	pub(in crate::runtime) fn new(device: &Arc<Device>) -> Self {
		Self {
			device: device.clone(),
			commands: std::collections::VecDeque::new(),
		}
	}

	pub(in crate::runtime) fn push(&mut self, command: RecordedCommandBuffer) {
		self.commands.push_back(command);
	}

	pub(in crate::runtime) fn belongs_to(&self, device: &Arc<Device>) -> bool {
		self.device.same_as(device)
	}

	pub(in crate::runtime) fn len(&self) -> usize {
		self.commands.len()
	}

	pub(in crate::runtime) fn next(&mut self) -> Option<RecordedCommandBuffer> {
		self.commands.pop_front()
	}

	pub(in crate::runtime) fn retain(&mut self, owner: Arc<dyn Send + Sync>) {
		for command in &mut self.commands {
			command.retain(owner.clone());
		}
	}
}

impl Drop for RecordedBatch {
	fn drop(&mut self) {
		for command in self.commands.drain(..) {
			self.device.free(command);
		}
	}
}

enum CommandBufferAllocation {
	Owned {
		handle: vk::CommandBuffer,
		pool: CommandPoolKind,
		_resources: Vec<Buffer>,
		_video_owners: Vec<Arc<dyn Send + Sync>>,
		_accesses: Vec<BufferAccess>,
		_bounded_sets: Option<BoundedSets>,
		_pipelines: Vec<Arc<ComputePipeline>>,
		// Route-observation query pools retire with their commands; the jit
		// table holds a second Arc for lazy readback.
		_route_timings: Option<Arc<RouteTimings>>,
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
	pool: CommandPoolKind,
	_resources: Vec<Buffer>,
	_video_owners: Vec<Arc<dyn Send + Sync>>,
	_accesses: Vec<BufferAccess>,
	_bounded_sets: Option<BoundedSets>,
	// Owned specializations retire with their commands, before Device release.
	_pipelines: Vec<Arc<ComputePipeline>>,
	_route_timings: Option<Arc<RouteTimings>>,
	device: Arc<Device>,
}

struct PreparedDispatch<'a> {
	node: &'a ComputeNode,
	pipeline: PreparedPipeline<'a>,
	push: Vec<u8>,
	physical_write: Option<PhysicalWriteContract>,
	/// Bindings the dispatch actually consumes: `node.jit.buffers` when an
	/// admitted specialization dispatches, otherwise `node.buffers`.
	#[cfg(oa_profile = "compatibility")]
	buffers: &'a [BufferUse],
}

enum PreparedPipeline<'a> {
	Registry(&'a ComputePipeline),
	Jit(Arc<ComputePipeline>),
}

impl PreparedPipeline<'_> {
	fn raw_pipeline(&self) -> &ComputePipeline {
		match self {
			Self::Registry(pipeline) => pipeline,
			Self::Jit(pipeline) => pipeline,
		}
	}
}

#[derive(Default)]
struct CommandRetention {
	resources: Vec<Buffer>,
	video_owners: Vec<Arc<dyn Send + Sync>>,
	accesses: Vec<BufferAccess>,
	bounded_sets: Option<BoundedSets>,
	/// Admitted specialization pipelines stay live until the recorded command
	/// is retired; the registry's pipelines live with the device instead.
	pipelines: Vec<Arc<ComputePipeline>>,
	/// Per-dispatch route-observation timestamps recorded into this command.
	route_timings: Option<Arc<RouteTimings>>,
}

struct CommandRecording<'a> {
	device: &'a vk::Device,
	pool: vk::CommandPool,
	handle: Option<vk::CommandBuffer>,
}

impl Drop for CommandRecording<'_> {
	fn drop(&mut self) {
		if let Some(handle) = self.handle {
			// SAFETY: recording holds the originating pool lock, and the
			// command has never been submitted.
			let _ = unsafe { self.device.free_command_buffers(self.pool, &[handle]) };
		}
	}
}

impl CommandPool {
	pub(in crate::runtime::device) fn new(
		device: &vk::Device,
		queue_family: u32,
		kind: CommandPoolKind,
	) -> Result<Self> {
		let create_info = vk::CommandPoolCreateInfo::new()
			.set_flags(vk::CommandPoolCreateFlagBits::TRANSIENT_BIT)
			.set_queue_family_index(queue_family);

		// SAFETY: the queue family was queried from the physical device used to create
		// this live logical device. No allocation callbacks are installed.
		let handle = unsafe { device.create_command_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "command-pool creation", source))?;

		Ok(Self {
			handle,
			kind,
			queue_family,
		})
	}

	pub(in crate::runtime::device) fn record_empty(
		&mut self,
		device: &vk::Device,
	) -> Result<RecordedCommandBuffer> {
		let command_buffer = self.begin(device, vk::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT)?;
		self.finish(
			device,
			command_buffer,
			CommandRetention::default(),
			None,
			None,
		)
	}

	pub(in crate::runtime::device) fn record_custom(
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

	#[allow(clippy::too_many_arguments)]
	pub(in crate::runtime::device) fn record_compute_graph(
		&mut self,
		context: &Arc<Device>,
		pipelines: &[Option<OnceLock<ComputePipeline>>],
		descriptor_set: vk::DescriptorSet,
		graph: &ExecutableGraph,
		nodes: std::ops::Range<usize>,
		timing: Option<TimestampPair>,
		reusable_device: Option<Arc<Device>>,
	) -> Result<RecordedCommandBuffer> {
		let device = context;
		// One command buffer serves one queue family. Every node in the
		// recorded range must resolve to this pool's family — a mixed-family
		// plan is submitted as one command buffer per queue run.
		if nodes.end > graph.nodes().len()
			|| graph.node_families()[nodes.clone()]
				.iter()
				.any(|family| *family != self.queue_family)
		{
			return Err(Error::invalid_argument(format!(
				"execution graph range {nodes:?} does not resolve wholly to queue family {}",
				self.queue_family
			)));
		}
		let prepared = graph.nodes()[nodes.clone()]
			.iter()
			.map(|node| match node {
				GraphNode::Compute(dispatch) => prepare_dispatch(context, pipelines, dispatch).map(Some),
				GraphNode::BufferCopy(copy) => {
					validate_buffer_copy(copy)?;
					Ok(None)
				}
				GraphNode::VideoDecode(video) => {
					if reusable_device.is_some() {
						return Err(Error::failed_precondition(
							"video decoding requires replay-local status queries",
						));
					}
					video.validate_ownership()?;
					Ok(None)
				}
				GraphNode::VideoEncode(video) => {
					if reusable_device.is_some() {
						return Err(Error::failed_precondition(
							"video encoding requires replay-local feedback queries",
						));
					}
					video.validate_ownership()?;
					Ok(None)
				}
				GraphNode::VideoPlanes(video) => {
					video.validate()?;
					Ok(None)
				}
				GraphNode::VideoEncodeInput(video) => {
					video.validate()?;
					Ok(None)
				}
			})
			.collect::<Result<Vec<_>>>()?;
		if prepared.is_empty() {
			return Err(Error::invalid_argument("execution graph has no commands"));
		}
		#[cfg(oa_profile = "compatibility")]
		let bounded_sets = if prepared.iter().any(Option::is_some) {
			let binding_count = |dispatch: &PreparedDispatch| match &dispatch.pipeline {
				PreparedPipeline::Jit(_) => dispatch.buffers.len() as u32,
				PreparedPipeline::Registry(_) => dispatch.node.kernel.bounded_buffer_count(),
			};
			let layouts = prepared
				.iter()
				.flatten()
				.map(|dispatch| {
					context
						.bounded_descriptor_layout(binding_count(dispatch))
						.ok_or_else(|| Error::missing_capability("bounded descriptor layout unavailable"))
				})
				.collect::<Result<Vec<_>>>()?;
			let buffers = prepared
				.iter()
				.flatten()
				.map(|dispatch| {
					let first = dispatch
						.buffers
						.first()
						.expect("bounded dispatch has operands");
					(0..binding_count(dispatch) as usize)
						.map(|slot| {
							let buffer = dispatch.buffers.get(slot).unwrap_or(first);
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
		#[cfg(oa_profile = "core")]
		let bounded_sets = None;
		let usage = if reusable_device.is_some() {
			vk::CommandBufferUsageFlagBits::SIMULTANEOUS_USE_BIT
		} else {
			vk::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT
		};
		// Route observation: timestamp each fused dispatch. A reusable command
		// resubmits the same queries, so instrumentation records one-shot
		// buffers only — a lazy readback could otherwise land mid-resubmission.
		// Pool creation is best-effort: measurement never rejects a recording.
		let route_entries: Vec<(u64, ObservedRoute)> = prepared
			.iter()
			.flatten()
			.filter_map(|dispatch| {
				dispatch.node.fusion_signature.map(|signature| {
					let route = match dispatch.pipeline {
						PreparedPipeline::Jit(_) => ObservedRoute::Specialized,
						PreparedPipeline::Registry(_) => ObservedRoute::Tape,
					};
					(signature, route)
				})
			})
			.take(RouteTimings::MAX_SLOTS)
			.collect();
		let route_timings = (reusable_device.is_none() && !route_entries.is_empty())
			.then(|| context.route_timings(route_entries))
			.transpose()
			.ok()
			.flatten();
		let command_buffer = self.begin(device.raw(), usage)?;
		let mut recording = CommandRecording {
			device: device.raw(),
			pool: self.handle,
			handle: Some(command_buffer),
		};
		let mut resources = Vec::new();
		let mut video_owners: Vec<Arc<dyn Send + Sync>> = Vec::new();
		let mut decode_statuses = Vec::new();
		let mut encode_receipts = Vec::new();
		let mut accesses = Vec::new();
		if let Some(timing) = &timing {
			timing.record_begin(command_buffer)?;
		}
		if let Some(timings) = &route_timings {
			timings.record_reset(command_buffer)?;
		}

		let mut jit_pipelines = Vec::new();
		let mut observed_slot = 0_usize;
		let mut copy_writes = false;
		let mut compute_writes = false;
		#[cfg(oa_profile = "compatibility")]
		let mut dispatch_index = 0;
		for (index, (node, dispatch)) in graph.nodes()[nodes.clone()]
			.iter()
			.zip(&prepared)
			.enumerate()
		{
			let node_index = nodes.start + index;
			record_barriers(device, command_buffer, graph.barriers_before(node_index))?;
			resources.extend(node.buffers().iter().map(|usage| usage.buffer.clone()));
			accesses.extend(node.buffers().iter().map(|usage| usage.access));
			match node {
				GraphNode::BufferCopy(copy) => {
					record_buffer_copy(device, command_buffer, copy)?;
					copy_writes = true;
					continue;
				}
				GraphNode::VideoDecode(video) => {
					decode_statuses.push(Arc::new(video.record(command_buffer)?));
					video_owners.push(video.clone());
					continue;
				}
				GraphNode::VideoEncode(video) => {
					let receipt = video.record(command_buffer)?;
					video_owners.push(receipt.clone());
					encode_receipts.push(receipt);
					video_owners.push(video.clone());
					continue;
				}

				GraphNode::VideoPlanes(video) => {
					video
						.copy
						.record(device, command_buffer, video.phase, node.buffers())?;
					video_owners.push(video.copy.clone());
					copy_writes |= video.phase == PlaneCopyPhase::Copy;
					continue;
				}
				GraphNode::VideoEncodeInput(video) => {
					video.copy.record(device, command_buffer, node.buffers())?;
					video_owners.push(video.copy.clone());
					copy_writes = true;
					continue;
				}
				GraphNode::Compute(_) => {}
			}
			let dispatch = dispatch.as_ref().expect("compute dispatch was prepared");
			compute_writes |= node
				.buffers()
				.iter()
				.any(|usage| usage.access != BufferAccess::Read);
			#[cfg(oa_profile = "compatibility")]
			let set = {
				let set = bounded_sets
					.as_ref()
					.map_or(descriptor_set, |sets| sets.set(dispatch_index));
				dispatch_index += 1;
				set
			};
			#[cfg(oa_profile = "core")]
			let set = descriptor_set;
			let slot = dispatch
				.node
				.fusion_signature
				.filter(|_| route_timings.is_some() && observed_slot < RouteTimings::MAX_SLOTS)
				.map(|_| {
					observed_slot += 1;
					observed_slot - 1
				});
			if let (Some(timings), Some(slot)) = (&route_timings, slot) {
				timings.record_before(command_buffer, slot)?;
			}
			record_dispatch(device, command_buffer, set, dispatch)?;
			if let (Some(timings), Some(slot)) = (&route_timings, slot) {
				timings.record_after(command_buffer, slot)?;
			}
			if let PreparedPipeline::Jit(pipeline) = &dispatch.pipeline {
				jit_pipelines.push(pipeline.clone());
			}
			if let Some(indirect) = &dispatch.node.indirect {
				resources.push(indirect.buffer.clone());
				accesses.push(BufferAccess::Read);
			}
		}
		// Cross-family hazards consuming this run's output release buffer
		// ownership here; the consumer run records the matching acquire in
		// `barriers_before` at its destination node. The engine timeline
		// orders the two submissions.
		let releases = (nodes.end..graph.nodes().len())
			.flat_map(|index| graph.barriers_before(index).iter())
			.filter(|hazard| hazard.is_cross_queue() && nodes.contains(&hazard.source_nodes[1]))
			.cloned()
			.collect::<Vec<_>>();
		record_release_barriers(device, command_buffer, &releases)?;
		record_host_visibility(device, command_buffer, compute_writes, copy_writes)?;
		if let Some(timing) = &timing {
			timing.record_end(command_buffer)?;
		}
		if let Some(timings) = &route_timings {
			context.jit_observe_timings(timings.clone());
		}

		let mut recorded = self.finish(
			device.raw(),
			recording.handle.take().expect("command is still recording"),
			CommandRetention {
				resources,
				video_owners,
				accesses,
				bounded_sets,
				pipelines: jit_pipelines,
				route_timings,
			},
			timing,
			reusable_device,
		)?;
		recorded.decode_statuses = decode_statuses;
		recorded.encode_receipts = encode_receipts;
		Ok(recorded)
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
			video_owners,
			accesses,
			bounded_sets,
			pipelines,
			route_timings,
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
					pool: self.kind,
					_resources: resources,
					_video_owners: video_owners,
					_accesses: accesses,
					_bounded_sets: bounded_sets,
					_pipelines: pipelines,
					_route_timings: route_timings,
					device,
				}),
			}),
			None => CommandBufferAllocation::Owned {
				handle: command_buffer,
				pool: self.kind,
				_resources: resources,
				_video_owners: video_owners,
				_accesses: accesses,
				_bounded_sets: bounded_sets,
				_pipelines: pipelines,
				_route_timings: route_timings,
			},
		};
		Ok(RecordedCommandBuffer {
			allocation,
			timing,
			event_timing_suppressed: false,
			brackets: None,
			retained: None,
			decode_statuses: Vec::new(),
			encode_receipts: Vec::new(),
			secret_erasures: Vec::new(),
			retained_secret_erasures: Vec::new(),
			_secret_uses: Vec::new(),
		})
	}

	pub(in crate::runtime::device) fn free_handle(
		&mut self,
		device: &vk::Device,
		handle: vk::CommandBuffer,
	) {
		// SAFETY: the retirement service or final reusable owner transfers a completed
		// or never-submitted command buffer back to its originating pool exactly once.
		unsafe {
			let _ = device.free_command_buffers(self.handle, &[handle]);
		}
	}

	pub(in crate::runtime::device) fn destroy(&mut self, device: &vk::Device) {
		// SAFETY: `DeviceInner` owns this pool, destroys it once after every allocated
		// command buffer has been freed, and uses the same allocation callbacks.
		unsafe {
			let _ = device.destroy_command_pool(self.handle, None);
		}
	}
}

fn reject_secret_node(node: &ComputeNode) -> Result<()> {
	if node.kernel.requires_secret_storage() {
		return Err(Error::failed_precondition(
			"secret kernels cannot use ordinary compute graph storage",
		));
	}
	Ok(())
}

fn prepare_dispatch<'a>(
	context: &'a Arc<Device>,
	pipelines: &'a [Option<OnceLock<ComputePipeline>>],
	node: &'a ComputeNode,
) -> Result<PreparedDispatch<'a>> {
	reject_secret_node(node)?;
	// An admitted specialization substitutes its own pipeline, program-order
	// bindings and push words; a table miss or absent jit request keeps the
	// canonical registry pipeline and tape-shaped bindings.
	let jit = node.jit.as_ref().and_then(|jit| {
		context
			.jit_pipeline(&jit.key)
			.map(|pipeline| (jit, pipeline))
	});
	let jit_admitted = jit.is_some();
	let (pipeline, buffers, push_constants) = match jit {
		Some((jit, pipeline)) => (
			PreparedPipeline::Jit(pipeline),
			jit.buffers.as_slice(),
			jit.push_constants.as_slice(),
		),
		None => (
			PreparedPipeline::Registry(
				pipelines
					.get(node.kernel.index())
					.and_then(Option::as_ref)
					.and_then(OnceLock::get)
					.ok_or_else(|| {
						Error::missing_capability(format!(
							"{} has no admitted compute pipeline on the selected Vulkan device",
							node.operation
						))
					})?,
			),
			node.buffers.as_slice(),
			node.push_constants.as_slice(),
		),
	};
	if node.indirect.is_none()
		&& node
			.workgroups
			.into_iter()
			.zip(pipeline.raw_pipeline().max_dispatch_group_count())
			.any(|(requested, limit)| requested == 0 || requested > limit)
	{
		return Err(Error::invalid_argument(format!(
			"{} workgroup count {:?} is zero or exceeds the selected device limit {:?}",
			node.operation,
			node.workgroups,
			pipeline.raw_pipeline().max_dispatch_group_count()
		)));
	}
	#[cfg(oa_profile = "compatibility")]
	let descriptor_indices = (0..u32::try_from(buffers.len())
		.map_err(|_| Error::invalid_argument("dispatch binding count overflow"))?)
		.collect::<Vec<_>>();
	#[cfg(oa_profile = "core")]
	let descriptor_indices = buffers
		.iter()
		.map(|usage| usage.buffer.descriptor_index())
		.collect::<Vec<_>>();
	let push = encode_push_constants(push_constants, &descriptor_indices, node.operation)?;
	if push.len() != pipeline.raw_pipeline().push_constant_size() as usize {
		return Err(Error::invalid_argument(format!(
			"{} push data contains {} bytes but its reflected kernel ABI requires {}",
			node.operation,
			push.len(),
			pipeline.raw_pipeline().push_constant_size()
		)));
	}
	let physical_write = if jit_admitted {
		// The specialization's binding accesses and ABI were validated at
		// admission; the tape kernel's physical-write contract does not apply.
		None
	} else {
		node.kernel.artifact().physical_write
	};
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
		#[cfg(oa_profile = "compatibility")]
		buffers,
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
	device: &Device,
	command_buffer: vk::CommandBuffer,
	hazards: &[BufferHazard],
) -> Result<()> {
	// Same-family hazards keep the full two-sided ordering scopes. A
	// cross-family hazard emits only the acquire scope — the producer run
	// already released ownership through `record_release_barriers` and the
	// timeline edge carries the execution ordering.
	record_buffer_barriers(device, command_buffer, hazards, |hazard| {
		let cross = hazard.is_cross_queue();
		(
			if cross {
				vk::PipelineStageFlagBits2::NONE
			} else {
				hazard.source.vk_stages()
			},
			if cross {
				vk::AccessFlagBits2::NONE
			} else {
				hazard.source.vk_access()
			},
			hazard.destination.vk_stages(),
			hazard.destination.vk_access(),
		)
	})
}

fn record_release_barriers(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	hazards: &[BufferHazard],
) -> Result<()> {
	// The release half of a queue-family ownership transfer keeps the
	// producer's stage/access scope and names no destination scope — the
	// consumer's acquire supplies that on its own queue.
	record_buffer_barriers(device, command_buffer, hazards, |hazard| {
		(
			hazard.source.vk_stages(),
			hazard.source.vk_access(),
			vk::PipelineStageFlagBits2::NONE,
			vk::AccessFlagBits2::NONE,
		)
	})
}

#[allow(clippy::too_many_arguments)]
fn record_buffer_barriers(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	hazards: &[BufferHazard],
	scopes: impl Fn(
		&BufferHazard,
	) -> (
		vk::PipelineStageFlagBits2,
		vk::AccessFlagBits2,
		vk::PipelineStageFlagBits2,
		vk::AccessFlagBits2,
	),
) -> Result<()> {
	if hazards.is_empty() {
		return Ok(());
	}
	let barriers = hazards
		.iter()
		.map(|hazard| {
			let cross = hazard.is_cross_queue();
			let (src_stage, src_access, dst_stage, dst_access) = scopes(hazard);
			vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(src_stage)
				.set_src_access_mask(src_access)
				.set_dst_stage_mask(dst_stage)
				.set_dst_access_mask(dst_access)
				.set_src_queue_family_index(if cross {
					hazard.source_family
				} else {
					vk::QUEUE_FAMILY_IGNORED
				})
				.set_dst_queue_family_index(if cross {
					hazard.destination_family
				} else {
					vk::QUEUE_FAMILY_IGNORED
				})
				.set_buffer(hazard.buffer.raw())
				.set_offset(0)
				.set_size(hazard.buffer.size())
		})
		.collect::<Vec<_>>();
	let dependency = vk::DependencyInfo::new()
		.set_buffer_memory_barriers(&barriers)
		.map_err(Error::from)?;

	// SAFETY: every barrier names a live retained buffer. Same-family
	// hazards carry both ordering scopes with QUEUE_FAMILY_IGNORED; a
	// cross-family hazard carries the scopes for its side of the ownership
	// transfer and names both resolved families. The engine timeline
	// submission ordering pairs each release with the consumer's acquire.
	// The full logical buffer range is valid and no image layout exists.
	unsafe {
		device.pipeline_barrier(command_buffer, &dependency)?;
	}
	Ok(())
}

fn validate_buffer_copy(copy: &BufferCopyNode) -> Result<()> {
	let [source, destination] = &copy.buffers;
	if source.buffer.same_as(&destination.buffer)
		|| copy.bytes == 0
		|| copy.bytes > source.buffer.size()
		|| copy.bytes > destination.buffer.size()
	{
		return Err(Error::invalid_argument(
			"buffer copy requires a nonempty range in disjoint buffers",
		));
	}
	Ok(())
}

fn record_buffer_copy(
	device: &Device,
	command_buffer: vk::CommandBuffer,
	copy: &BufferCopyNode,
) -> Result<()> {
	let [source, destination] = &copy.buffers;
	// SAFETY: both retained storage buffers have the required transfer usage.
	// Preflight checked disjointness and the entire range. The graph recorded
	// the dependencies leading into the copy on this queue.
	unsafe {
		device.cmd_copy_buffer(
			command_buffer,
			source.buffer.raw(),
			destination.buffer.raw(),
			&[vk::BufferCopy::new().set_size(copy.bytes)],
		)?;
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
	device: &Device,
	command_buffer: vk::CommandBuffer,
	compute_writes: bool,
	copy_writes: bool,
) -> Result<()> {
	if !compute_writes && !copy_writes {
		return Ok(());
	}
	let mut stages = vk::PipelineStageFlags2::empty();
	let mut accesses = vk::AccessFlags2::empty();
	if compute_writes {
		stages = stages | vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT;
		accesses = accesses | vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT;
	}
	if copy_writes {
		stages = stages | vk::PipelineStageFlagBits2::COPY_BIT;
		accesses = accesses | vk::AccessFlagBits2::TRANSFER_WRITE_BIT;
	}
	let barrier = vk::MemoryBarrier2::new()
		.set_src_stage_mask(stages)
		.set_src_access_mask(accesses)
		.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
		.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT);
	let dependency = vk::DependencyInfo::new()
		.set_memory_barriers(std::slice::from_ref(&barrier))
		.map_err(Error::from)?;
	// SAFETY: the recording command retains this batch's live host-visible
	// buffers. HOST_READ performs the required memory-domain
	// operation; CPU access still requires exact completion and invalidation.
	unsafe {
		device.pipeline_barrier(command_buffer, &dependency)?;
	}
	Ok(())
}

fn record_dispatch(
	device: &Device,
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
		debug_assert!(
			dispatch.node.jit.is_some()
				|| dispatch.physical_write == dispatch.node.kernel.artifact().physical_write
		);
		device.cmd_bind_pipeline(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.raw_pipeline().raw(),
		)?;
		device.cmd_bind_descriptor_sets(
			command_buffer,
			vk::PipelineBindPoint::COMPUTE,
			dispatch.pipeline.raw_pipeline().layout(),
			0,
			&[descriptor_set],
			&[],
		)?;
		device.cmd_push_constants(
			command_buffer,
			dispatch.pipeline.raw_pipeline().layout(),
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
	pub(in crate::runtime) fn retain(&mut self, owner: Arc<dyn Send + Sync>) {
		self.retained = Some(owner);
	}

	pub(in crate::runtime::device) fn with_timing_brackets(
		mut self,
		before: Self,
		after: Self,
		timing: TimestampPair,
	) -> Self {
		self.brackets = Some(Box::new([before, after]));
		self.timing = Some(timing);
		self
	}
	pub(in crate::runtime::device) fn submission_handles(&self) -> Vec<vk::CommandBuffer> {
		match &self.brackets {
			Some(pair) => vec![pair[0].raw(), self.raw(), pair[1].raw()],
			None => vec![self.raw()],
		}
	}
	pub(in crate::runtime::device) fn take_brackets(&mut self) -> Option<Box<[Self; 2]>> {
		self.brackets.take()
	}
	pub(in crate::runtime::device) fn retain_secret_public_output(
		&mut self,
		public: Buffer,
		sets: Option<BoundedSets>,
	) -> Result<()> {
		self.retain_secret_public_resources(std::iter::once(public), sets)
	}

	pub(in crate::runtime::device) fn retain_secret_public_resources(
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
		secret: crate::runtime::device::logical::secret_buffer::SecretBuffer,
	) {
		self.secret_erasures.push(secret);
	}

	/// Shared retention is private command lifetime, never a clonable secret value.
	pub(in crate::runtime::device) fn retain_secret_use(
		&mut self,
		secret: Arc<crate::runtime::device::logical::secret_buffer::SecretBuffer>,
	) {
		self._secret_uses.push(secret);
	}

	pub(in crate::runtime::device) fn retain_shared_secret_erasure(
		&mut self,
		secret: Arc<crate::runtime::device::logical::secret_buffer::SecretBuffer>,
	) {
		self.retained_secret_erasures.push(secret);
	}

	// Only the retirement worker calls this after a successful timeline wait.
	pub(in crate::runtime::device) fn confirm_secret_erasures(&self) {
		for secret in self
			.secret_erasures
			.iter()
			.chain(self.retained_secret_erasures.iter().map(Arc::as_ref))
		{
			secret.confirm_erasure();
		}
	}

	pub(in crate::runtime::device) fn fail_secret_erasures(&self) {
		for secret in self
			.secret_erasures
			.iter()
			.chain(self.retained_secret_erasures.iter().map(Arc::as_ref))
		{
			secret.fail_erasure();
		}
	}

	pub(in crate::runtime::device) fn raw(&self) -> vk::CommandBuffer {
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

	pub(in crate::runtime::device) fn pool_kind(&self) -> CommandPoolKind {
		match &self.allocation {
			CommandBufferAllocation::Owned { pool, .. } => *pool,
			CommandBufferAllocation::Reusable(command) => command.inner.pool,
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

	pub(in crate::runtime::device) fn reusable(&self) -> Option<ReusableCommandBuffer> {
		match &self.allocation {
			CommandBufferAllocation::Owned { .. } => None,
			CommandBufferAllocation::Reusable(command) => Some(command.clone()),
		}
	}

	pub(in crate::runtime::device) fn into_owned_handle(
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
			retained: None,
			decode_statuses: Vec::new(),
			encode_receipts: Vec::new(),
			secret_erasures: Vec::new(),
			retained_secret_erasures: Vec::new(),
			_secret_uses: Vec::new(),
		}
	}
}

impl Drop for ReusableCommandBufferInner {
	fn drop(&mut self) {
		self
			.device
			.free_command_buffer_handle(self.handle, self.pool);
	}
}

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/secret_dispatch_unit.rs"]
mod secret_dispatch_unit;

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/jit_dispatch_unit.rs"]
pub(in crate::runtime) mod jit_lab;
