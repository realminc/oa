use std::{
	mem::ManuallyDrop,
	sync::{Arc, Mutex, OnceLock},
};

use crate::{Error, Result, runtime::dna::kernel::KernelId};

pub(in crate::runtime) mod buffer;
pub(in crate::runtime) mod command;
pub(in crate::runtime) mod commands;
pub(in crate::runtime) mod descriptor;
pub(in crate::runtime) mod dispatch;
pub(in crate::runtime) mod image;
pub(crate) mod pipeline;
pub(in crate::runtime) mod pqc;
pub(in crate::runtime) mod retirement;
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private PQC storage substrate; secret algorithms are not admitted"
	)
)]
pub(in crate::runtime) mod secret_buffer;
pub(in crate::runtime) mod specialization;
pub(in crate::runtime) mod swapchain;
pub(in crate::runtime) mod timeline;
pub(in crate::runtime) mod timestamp;
pub(in crate::runtime) mod video;
pub(in crate::runtime) mod vkcache;

use buffer::RecycledBuffer;
use command::{CommandPool, CommandPoolKind};
use commands::{RenderingCommands, SyncCommands};
use descriptor::DescriptorHeap;
use pipeline::ComputePipeline;
use timeline::Timeline;

use super::{
	configuration::DeviceConfiguration, features::ExecutionProfile, physical::DevicePhysical,
};
use crate::runtime::Instance;

pub(super) const MAX_RECYCLED_STORAGE_BUFFERS: usize = 256;
pub(super) const MAX_RECYCLED_STORAGE_BYTES: usize = 256 * 1024 * 1024;

/// Per-draw parameters for [`Device::record_scene_frame`].
///
/// Each entry in the draw list is dispatched as one `vkCmdDrawIndexed` within
/// a single dynamic render pass. Push constants and the material descriptor
/// set (set 1) are re-bound per draw; the scene set (set 0) and vertex/index
/// buffers are bound once before the loop.
pub(crate) struct SceneDrawCmd {
	/// First index in the shared merged index buffer.
	pub index_start: u32,
	/// Index count for this draw (multiple of 3).
	pub index_count: u32,
	/// Raw push constant bytes (must be `SURFACE_PUSH_SIZE` long).
	pub push_data: Vec<u8>,
	/// Resolved material descriptor set (set 1).
	pub material_set: vk::DescriptorSet,
	/// When `true` this draw belongs to the transparent/blend pass.
	pub is_blend: bool,
}

/// Geometry source for one graphics frame record.
///
/// `Indexed` binds the shared vertex/index buffers and issues
/// `vkCmdDrawIndexed`; `Vertexless` binds nothing and issues `vkCmdDraw` —
/// the vertex shader derives positions from `SV_VertexID` (fullscreen
/// triangle and generated-geometry pipelines).
pub(in crate::runtime) enum FrameDraw {
	Indexed {
		vertex_buffer: vk::Buffer,
		index_buffer: vk::Buffer,
		index_count: u32,
	},
	Vertexless {
		vertex_count: u32,
	},
}

/// Vulkan executor resources, owned once by Device; never independently retained.
/// Explicit reduction-order selection for tile candidates:
/// `Deterministic` suppresses order-changing subgroup candidates,
/// `Throughput` offers every one the device supports, and `Default`
/// keeps each per-kernel flag in charge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::runtime) enum TileReduceOrder {
	Default,
	Deterministic,
	Throughput,
}

pub(in crate::runtime) struct DeviceLogical {
	pub(super) handle: ManuallyDrop<std::sync::Arc<vk::Device>>,
	pub(super) sync_commands: SyncCommands,
	pub(super) rendering_commands: RenderingCommands,
	pub(super) allocator: ManuallyDrop<vk::mem::SynchronizedAllocator<'static>>,
	pub(super) command_pool: Mutex<CommandPool>,
	pub(super) graphics_command_pool: Mutex<Option<CommandPool>>,
	pub(super) transfer_command_pool: Mutex<Option<CommandPool>>,
	pub(super) video_decode_command_pool: Mutex<Option<CommandPool>>,
	pub(super) video_encode_command_pool: Mutex<Option<CommandPool>>,
	pub(super) timeline: Timeline,
	pub(super) descriptors: DescriptorHeap,
	pub(super) storage_pool: Mutex<StoragePool>,
	// None means unsupported; an empty OnceLock means admitted but not compiled.
	pub(super) pipelines: Vec<Option<OnceLock<ComputePipeline>>>,
	pub(super) pipeline_creation: Mutex<PipelineCreation>,
	/// Opt-in `OA_JIT_FUSION` specialization state; empty until the first fused
	/// node requests admission.
	pub(super) jit: Mutex<specialization::JitFusion>,
	/// Extension-kernel pipelines admitted through the public `oa::compute`
	/// surface, keyed by caller-derived content hash. Consulted before the jit
	/// table at record time; extension admissions never touch fusion census.
	pub(super) extensions: Mutex<std::collections::HashMap<Vec<u8>, Arc<pipeline::ComputePipeline>>>,
	/// Device-lifetime fusion decision counters; never affect routing.
	pub(super) fusion_census: crate::runtime::dna::fusion::census::FusionCensus,
	/// Tape-vs-per-op element bound for elementwise fusion, seeded per
	/// device at creation. Routing policy, not a capability bound.
	pub(super) tape_bound_elements: u32,
	/// Single-head SDPA merge admission, seeded per device at creation.
	/// Every current class seeds `false` — the route changed the canonical
	/// Byte-MoE trajectory on Xe and was net-negative on Haswell.
	pub(super) single_head_sdpa_route: bool,
	/// Per-submission estimated-cost budget in nanoseconds, seeded per
	/// device at creation. Plans over the estimate split at node
	/// boundaries; a single over-budget node declines before submission.
	pub(super) submission_budget_ns: u64,
	pub(super) compute_queue: vk::Queue,
	pub(super) graphics_queue: Option<vk::Queue>,
	pub(super) transfer_queue: Option<vk::Queue>,
	pub(super) video_decode_queue: Option<vk::Queue>,
	pub(super) video_encode_queue: Option<vk::Queue>,
	pub(super) timestamp_period_ns: f64,
	pub(super) compute_timestamp_valid_bits: u32,
}

#[derive(Default)]
pub(super) struct StoragePool {
	pub(super) free: Vec<RecycledBuffer>,
	pub(super) bytes: usize,
}

/// Serializes compute-pipeline creation and every host access to the primary
/// pipeline cache it owns.
///
/// Vulkan requires external synchronization for host access to one
/// `VkPipelineCache`: lazy creation in `require_kernel`, the primary-cache
/// snapshot, and the final worker merge all hold this lock, while parallel
/// preload workers compile against their own per-worker caches.
pub(super) struct PipelineCreation {
	pub(super) cache: Option<pipeline::PipelineCache>,
}

/// Exclusive host access to the primary pipeline cache.
///
/// Vulkan requires external synchronization for every host access to one
/// `VkPipelineCache`; holding this guard across a `vkCreate*Pipelines` call
/// satisfies that requirement. The handle is `NULL` when the driver cache is
/// absent, which requests uncached pipeline creation.
pub(in crate::runtime) struct PipelineCacheGuard<'a> {
	pub(in crate::runtime::device) _guard: std::sync::MutexGuard<'a, PipelineCreation>,
	pub(in crate::runtime::device) raw: vk::PipelineCache,
}

impl PipelineCacheGuard<'_> {
	pub(in crate::runtime) const fn raw(&self) -> vk::PipelineCache {
		self.raw
	}
}

impl DeviceLogical {
	pub(super) fn new(
		instance: &Instance,
		physical: &DevicePhysical,
		configuration: &DeviceConfiguration,
	) -> Result<Self> {
		let shader_requirements = KernelId::ALL
			.into_iter()
			.map(|kernel| kernel.artifact().requirements())
			.collect::<Result<Vec<_>>>()?;
		let handle = std::sync::Arc::new(configuration.create_device(instance, physical.handle)?);

		let sync_commands = SyncCommands::new(instance, &handle, physical.api_version);
		let rendering_commands = RenderingCommands::new(instance, &handle, physical.api_version);

		// SAFETY: logical-device creation requested queue zero from this family, and
		// the logical device remains alive for the returned queue's full lifetime.
		let compute_queue =
			unsafe { handle.get_device_queue(physical.compute_queue_family, 0) }.map_err(Error::from)?;
		let graphics_queue = physical
			.graphics_queue_family
			.map(|family| {
				// SAFETY: logical-device creation requested queue zero from every unique
				// graphics family retained by physical-device selection.
				unsafe { handle.get_device_queue(family, 0) }.map_err(Error::from)
			})
			.transpose()?;
		let video_decode_queue = physical
			.video
			.decode_queue_family
			.map(|family| {
				// SAFETY: logical-device creation requested queue zero from every unique
				// selected video queue family and the device owns the returned handle.
				unsafe { handle.get_device_queue(family, 0) }.map_err(Error::from)
			})
			.transpose()?;
		let video_encode_queue = physical
			.video
			.encode_queue_family
			.map(|family| {
				// SAFETY: same queue-creation proof as the decode queue above.
				unsafe { handle.get_device_queue(family, 0) }.map_err(Error::from)
			})
			.transpose()?;
		// A dedicated transfer queue exists only when the transfer family is
		// distinct from compute — a same-family transfer resolves to the
		// compute queue instead of a second handle on one family.
		let transfer_queue = physical
			.transfer_queue_family
			.filter(|family| *family != physical.compute_queue_family)
			.map(|family| {
				// SAFETY: same queue-creation proof as the graphics queue above.
				unsafe { handle.get_device_queue(family, 0) }.map_err(Error::from)
			})
			.transpose()?;
		let mut command_pool = match CommandPool::new(
			&handle,
			physical.compute_queue_family,
			CommandPoolKind::Compute,
		) {
			Ok(command_pool) => command_pool,
			Err(error) => {
				// SAFETY: command-pool creation failed, so the device has no live child
				// resources. Creation used no allocation callbacks.
				if let Some(device) = std::sync::Arc::into_inner(handle) {
					unsafe {
						device.destroy(None);
					}
				}
				return Err(error);
			}
		};
		let mut timeline = match Timeline::new(&handle) {
			Ok(timeline) => timeline,
			Err(error) => {
				command_pool.destroy(&handle);
				// SAFETY: timeline creation failed and the command pool was destroyed, so the
				// device has no live children. No callbacks were installed.
				if let Some(device) = std::sync::Arc::into_inner(handle) {
					unsafe {
						device.destroy(None);
					}
				}
				return Err(error);
			}
		};

		// SAFETY: physical.handle belongs to this instance and created this device;
		// the allocator owns its own Arc<Device> clone. On success, Device retains
		// it until after allocator/device destruction.
		let allocator = match unsafe {
			vk::mem::Allocator::new_retained_from_physical_device(
				instance.raw(),
				physical.handle,
				handle.clone(),
			)
		} {
			Ok(allocator) => allocator,
			Err(source) => {
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: allocator creation failed and both existing device children were
				// destroyed; new_retained consumed its Arc clone, so this Arc is sole.
				// No callbacks were installed.
				if let Some(device) = std::sync::Arc::into_inner(handle) {
					unsafe {
						device.destroy(None);
					}
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"memory-allocator creation",
					source,
				));
			}
		};
		// SAFETY: the retained allocator carries no outstanding caller borrow, so
		// safe owned adoption applies and OA gets the synchronized surface.
		let allocator = vk::mem::SynchronizedAllocator::new_owned(allocator);
		let descriptors = match DescriptorHeap::new(&handle, physical.limits) {
			Ok(descriptors) => descriptors,
			Err(error) => {
				unsafe {
					let _ = allocator.with_allocator(|a| a.close());
				}
				drop(allocator);
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: every successfully created device child was destroyed above;
				// the closed allocator's Arc clone is released, so this Arc is sole.
				if let Some(device) = std::sync::Arc::into_inner(handle) {
					unsafe {
						device.destroy(None);
					}
				}
				return Err(error);
			}
		};
		// DescriptorHeap owns its effective capacity, including pool-allocation
		// reductions. Do not overwrite the physical admission budget with it.
		let timestamp_period_ns = physical.limits.timestamp_period_ns;
		let compute_timestamp_valid_bits = physical.limits.compute_timestamp_valid_bits;
		// The persisted pipeline cache is best-effort: an unreadable or
		// driver-rejected blob degrades to uncached creation. Loading it only
		// after all fallible device children exist keeps the error paths above
		// free of an extra retained device reference.
		let pipeline_cache = pipeline::PipelineCache::open(&handle, physical);
		let mut logical = Self {
			handle: ManuallyDrop::new(handle),
			sync_commands,
			rendering_commands,
			allocator: ManuallyDrop::new(allocator),
			command_pool: Mutex::new(command_pool),
			graphics_command_pool: Mutex::new(None),
			transfer_command_pool: Mutex::new(None),
			video_decode_command_pool: Mutex::new(None),
			video_encode_command_pool: Mutex::new(None),
			timeline,
			descriptors,
			storage_pool: Mutex::new(StoragePool::default()),
			pipelines: Vec::with_capacity(KernelId::ALL.len()),
			pipeline_creation: Mutex::new(PipelineCreation {
				cache: pipeline_cache,
			}),
			jit: Mutex::new(specialization::JitFusion::new()),
			extensions: Mutex::new(std::collections::HashMap::new()),
			fusion_census: crate::runtime::dna::fusion::census::FusionCensus::new(),
			tape_bound_elements: crate::runtime::dna::fusion::tape_bound_seed(&physical.device_info()),
			single_head_sdpa_route: crate::runtime::dna::fusion::single_head_sdpa_seed(
				&physical.device_info(),
			),
			submission_budget_ns: crate::runtime::graph::executable::submission_budget_seed(
				&physical.device_info(),
			),
			compute_queue,
			graphics_queue,
			transfer_queue,
			video_decode_queue,
			video_encode_queue,
			timestamp_period_ns,
			compute_timestamp_valid_bits,
		};
		// From this point, Rust unwinds every created child through one owner,
		// including the UI layout if later compute-pipeline creation fails.
		for (kernel, requirements) in KernelId::ALL.into_iter().zip(shader_requirements) {
			if !configuration
				.missing_shader_requirements(
					physical.limits.subgroup_supported_stages,
					physical.limits.subgroup_supported_operations,
					requirements,
				)
				.is_empty()
			{
				logical.pipelines.push(None);
				continue;
			}
			let artifact = kernel.artifact();
			// Pipeline creation is illegal above the queried shared-memory budget,
			// independently of Core/Compatibility or hardware/software identity.
			if artifact.workgroup_memory_size()? > physical.limits.max_compute_shared_memory_size {
				logical.pipelines.push(None);
				continue;
			}
			if physical.profile == ExecutionProfile::Compatibility
				&& logical
					.descriptors
					.bounded_layout(kernel.bounded_buffer_count())
					.is_none()
			{
				logical.pipelines.push(None);
				continue;
			}
			match pipeline::validate_artifact_limits(
				artifact.workgroup_size,
				artifact.push_constant_size()?,
				physical.limits,
			) {
				Ok(()) => {}
				Err(error) if error.kind() == crate::ErrorKind::MissingCapability => {
					logical.pipelines.push(None);
					continue;
				}
				Err(error) => return Err(error),
			}
			// Admission is metadata-only. Compile exactly the requested artifact
			// through require_kernel before recording its first dispatch. Optional
			// domains must not enter a driver compiler during Runtime construction.
			logical.pipelines.push(Some(OnceLock::new()));
		}

		Ok(logical)
	}

	/// Serialize the shared pipeline cache for the persistence coordinator.
	///
	/// Best-effort: a missing cache, poisoned lock or rejected query yields
	/// `None`, which the coordinator treats as "keep prior state".
	pub(super) fn pipeline_cache_data(&self) -> Option<Vec<u8>> {
		self
			.pipeline_creation
			.lock()
			.ok()?
			.cache
			.as_ref()?
			.data()
			.ok()
	}
}

impl Drop for DeviceLogical {
	fn drop(&mut self) {
		let storage_pool = match self.storage_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		for mut buffer in storage_pool.free.drain(..) {
			// SAFETY: final device ownership proves no live buffer can reference these
			// pooled allocations, which were created by this allocator.
			unsafe {
				let _ = self
					.allocator
					.with_allocator(|allocator| allocator.free(&mut buffer.allocation));
				let _ = self.handle.destroy_buffer(buffer.handle, None);
			}
		}
		// Command retirement has released every external pipeline/query owner.
		// Drop the jit table before descriptors and the native-device ownership
		// check: its pipelines and unread query pools retain native-device Arcs.
		let jit = self
			.jit
			.get_mut()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		drop(std::mem::replace(jit, specialization::JitFusion::new()));
		for slot in self.pipelines.iter_mut().flatten() {
			// Release each pipeline's logical-device Arc before into_inner below.
			if let Some(mut pipeline) = slot.take() {
				pipeline.destroy(&self.handle);
			}
		}
		// The merged driver blob was already handed to the Runtime persistence
		// coordinator while the device was live; here the cache handle is just
		// destroyed before the device.
		let creation = match self.pipeline_creation.get_mut() {
			Ok(creation) => creation,
			Err(poisoned) => poisoned.into_inner(),
		};
		creation.cache.take();
		self.descriptors.destroy(&self.handle);
		// SAFETY: this executor is unpublished during construction or owned by
		// the final Device, so no buffer owner remains. The allocator was initialized
		// exactly once and is dropped exactly once before its device.
		unsafe {
			let _ = self.allocator.with_allocator(|allocator| allocator.close());
			ManuallyDrop::drop(&mut self.allocator);
		}
		self.timeline.destroy(&self.handle);
		let command_pool = match self.command_pool.get_mut() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.destroy(&self.handle);
		let video_decode_pool = match self.video_decode_command_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		if let Some(pool) = video_decode_pool.as_mut() {
			pool.destroy(&self.handle);
		}
		let video_encode_pool = match self.video_encode_command_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		if let Some(pool) = video_encode_pool.as_mut() {
			pool.destroy(&self.handle);
		}
		let graphics_pool = match self.graphics_command_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		if let Some(pool) = graphics_pool.as_mut() {
			pool.destroy(&self.handle);
		}
		// SAFETY: the allocator, timeline, and command pools have been destroyed, and no
		// other logical-device children remain at this checkpoint.
		// SAFETY: all children are destroyed; the allocator's clone was released by
		// close, so this Arc is the sole device owner. No callbacks were installed.
		if let Some(device) =
			std::sync::Arc::into_inner(unsafe { ManuallyDrop::take(&mut self.handle) })
		{
			unsafe {
				device.destroy(None);
			}
		}
	}
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/device_logical_unit.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/jit_device_unit.rs"]
mod jit_lab;
