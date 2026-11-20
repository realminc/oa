use std::{
	mem::ManuallyDrop,
	sync::{Arc, Mutex, OnceLock},
};

use crate::{
	Error, Result,
	runtime::{
		dna::{ir::elementwise::Program, kernel::KernelId, tuning::ObservedRoute},
		graph::executable::ExecutableGraph,
	},
};

use super::{
	Device, NativeFlagBits,
	buffer::{MeshBuffer, RecycledBuffer},
	command::{CommandPool, CommandPoolKind, RecordedCommandBuffer, ReusableCommandBuffer},
	commands::{RenderingCommands, SyncCommands},
	configuration::DeviceConfiguration,
	descriptor::DescriptorHeap,
	features::ExecutionProfile,
	image::RenderTarget,
	info::log_compute_device,
	physical::DevicePhysical,
	pipeline::{self, ComputePipeline},
	timeline::Timeline,
	timestamp::{RouteTimings, TimestampPair},
};
use crate::runtime::{Buffer, Instance};

const MAX_RECYCLED_STORAGE_BUFFERS: usize = 256;
const MAX_RECYCLED_STORAGE_BYTES: usize = 256 * 1024 * 1024;

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
	handle: ManuallyDrop<std::sync::Arc<vk::Device>>,
	sync_commands: SyncCommands,
	rendering_commands: RenderingCommands,
	allocator: ManuallyDrop<vk::mem::SynchronizedAllocator<'static>>,
	command_pool: Mutex<CommandPool>,
	graphics_command_pool: Mutex<Option<CommandPool>>,
	transfer_command_pool: Mutex<Option<CommandPool>>,
	video_decode_command_pool: Mutex<Option<CommandPool>>,
	video_encode_command_pool: Mutex<Option<CommandPool>>,
	timeline: Timeline,
	descriptors: DescriptorHeap,
	storage_pool: Mutex<StoragePool>,
	// None means unsupported; an empty OnceLock means admitted but not compiled.
	pipelines: Vec<Option<OnceLock<ComputePipeline>>>,
	pipeline_creation: Mutex<PipelineCreation>,
	/// Opt-in `OA_JIT_FUSION` specialization state; empty until the first fused
	/// node requests admission.
	jit: Mutex<super::specialization::JitFusion>,
	/// Extension-kernel pipelines admitted through the public `oa::compute`
	/// surface, keyed by caller-derived content hash. Consulted before the jit
	/// table at record time; extension admissions never touch fusion census.
	extensions: Mutex<std::collections::HashMap<Vec<u8>, Arc<super::pipeline::ComputePipeline>>>,
	/// Device-lifetime fusion decision counters; never affect routing.
	fusion_census: crate::runtime::dna::fusion::census::FusionCensus,
	/// Tape-vs-per-op element bound for elementwise fusion, seeded per
	/// device at creation. Routing policy, not a capability bound.
	tape_bound_elements: u32,
	/// Single-head SDPA merge admission, seeded per device at creation.
	/// Every current class seeds `false` — the route changed the canonical
	/// Byte-MoE trajectory on Xe and was net-negative on Haswell.
	single_head_sdpa_route: bool,
	/// Per-submission estimated-cost budget in nanoseconds, seeded per
	/// device at creation. Plans over the estimate split at node
	/// boundaries; a single over-budget node declines before submission.
	submission_budget_ns: u64,
	compute_queue: vk::Queue,
	graphics_queue: Option<vk::Queue>,
	transfer_queue: Option<vk::Queue>,
	video_decode_queue: Option<vk::Queue>,
	video_encode_queue: Option<vk::Queue>,
	timestamp_period_ns: f64,
	compute_timestamp_valid_bits: u32,
}

#[derive(Default)]
struct StoragePool {
	free: Vec<RecycledBuffer>,
	bytes: usize,
}

/// Serializes compute-pipeline creation and every host access to the primary
/// pipeline cache it owns.
///
/// Vulkan requires external synchronization for host access to one
/// `VkPipelineCache`: lazy creation in `require_kernel`, the primary-cache
/// snapshot, and the final worker merge all hold this lock, while parallel
/// preload workers compile against their own per-worker caches.
struct PipelineCreation {
	cache: Option<pipeline::PipelineCache>,
}

/// Exclusive host access to the primary pipeline cache.
///
/// Vulkan requires external synchronization for every host access to one
/// `VkPipelineCache`; holding this guard across a `vkCreate*Pipelines` call
/// satisfies that requirement. The handle is `NULL` when the driver cache is
/// absent, which requests uncached pipeline creation.
pub(in crate::runtime) struct PipelineCacheGuard<'a> {
	_guard: std::sync::MutexGuard<'a, PipelineCreation>,
	raw: vk::PipelineCache,
}

impl PipelineCacheGuard<'_> {
	pub(in crate::runtime) const fn raw(&self) -> vk::PipelineCache {
		self.raw
	}
}

impl Device {
	pub(in crate::runtime) fn device_info(&self) -> crate::DeviceInfo {
		let mut info = self.physical.device_info();
		// Public descriptor capacities describe the actual heap, not the larger
		// admission budget. Keep the queried physical snapshot immutable.
		let (buffers, images) = self.descriptor_capacities();
		info.hardware.storage_buffer_descriptors = buffers;
		info.hardware.storage_image_descriptors = images;
		info
	}

	pub(in crate::runtime) fn configuration(&self) -> &DeviceConfiguration {
		&self.configuration
	}

	pub(super) fn descriptor_capacities(&self) -> (u32, u32) {
		self.logical.descriptors.capacities()
	}

	pub(in crate::runtime) fn execution_profile_name(&self) -> &'static str {
		match self.physical.profile {
			ExecutionProfile::Core => "Core (Vulkan 1.3+)",
			ExecutionProfile::Compatibility => "Compatibility (Vulkan 1.2)",
		}
	}

	pub(in crate::runtime) fn log_identity(&self) {
		// Currently always 1 device; the outer block is structured to
		// accept multiple devices (`,`-separated) when multi-device is added.
		crate::log_info!(
			crate::LogComponent::ENGINE,
			"oa::Engine · v{} · Vulkan · 1 device {{",
			env!("CARGO_PKG_VERSION")
		);
		log_compute_device(self, "  ", /* is_last */ true);
		crate::log_info!(crate::LogComponent::RUNTIME, "}}");
	}

	pub(in crate::runtime) fn video_device_capabilities(
		&self,
	) -> Result<crate::video::VideoDeviceCapabilities> {
		let video = self.physical.video;
		let decoder_available = self.logical.video_decode_queue.is_some()
			&& video.decode_result_status_queries
			&& (video.h264_decode || video.h265_decode || video.av1_decode || video.vp9_decode);
		let encoder_available =
			self.logical.video_encode_queue.is_some() && (video.h264_encode || video.h265_encode);

		Ok(crate::video::VideoDeviceCapabilities {
			decode_queue_family: video.decode_queue_family,
			encode_queue_family: video.encode_queue_family,
			decode_result_status_queries: video.decode_result_status_queries,
			encode_result_status_queries: video.encode_result_status_queries,
			h264_decode: video.h264_decode,
			h265_decode: video.h265_decode,
			av1_decode: video.av1_decode,
			vp9_decode: video.vp9_decode,
			h264_encode: video.h264_encode,
			h265_encode: video.h265_encode,
			av1_encode: video.av1_encode,
			video_queues_enabled: self.logical.video_decode_queue.is_some()
				|| self.logical.video_encode_queue.is_some(),
			decoder_sessions_available: decoder_available,
			encoder_sessions_available: encoder_available,
		})
	}

	pub(in crate::runtime) fn has_graphics_queue(&self) -> bool {
		self.logical.graphics_queue.is_some()
			&& self.configuration.features().dynamic_rendering
			&& self.configuration.features().shader_draw_parameters
	}

	pub(in crate::runtime) fn require_ui_bindings(&self) -> Result<()> {
		if !self.optimal_format_supports(
			vk::Format::R8G8B8A8_UNORM,
			vk::FormatFeatureFlagBits::COLOR_ATTACHMENT_BIT
				| vk::FormatFeatureFlagBits::TRANSFER_SRC_BIT
				| vk::FormatFeatureFlagBits::TRANSFER_DST_BIT,
		) {
			return Err(Error::missing_capability(
				"UI requires RGBA8 color-attachment and transfer image support",
			));
		}
		if !self.has_graphics_queue() {
			return Err(Error::missing_capability(
				"UI requires a graphics queue with dynamic rendering",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn supports_swapchain(&self) -> bool {
		self
			.configuration
			.extension_enabled(vk::KHR_SWAPCHAIN_EXTENSION_NAME)
	}

	pub(in crate::runtime) fn optimal_format_supports(
		&self,
		format: vk::Format,
		required: vk::FormatFeatureFlags,
	) -> bool {
		// SAFETY: the physical device and its owning instance are retained by this
		// Device. This query only reads immutable format capabilities.
		let mut properties = vk::FormatProperties::new();
		// SAFETY: the physical device and its owning instance are retained by this
		// Device; `properties` is live and writable for the call.
		let result = unsafe {
			self.instance.raw().get_physical_device_format_properties(
				self.physical.handle,
				format,
				&mut properties,
			)
		};
		matches!(result, Ok(())) && properties.optimal_tiling_features.contains(required)
	}

	/// Lock the pipeline-creation state and return the primary cache handle.
	///
	/// The returned guard holds the creation lock for its lifetime; pass
	/// [`PipelineCacheGuard::raw`] to `vkCreate*Pipelines` while the guard is
	/// alive. A missing cache yields `NULL` for uncached creation.
	pub(in crate::runtime) fn pipeline_cache(&self) -> Result<PipelineCacheGuard<'_>> {
		let guard = self
			.logical
			.pipeline_creation
			.lock()
			.map_err(|_| Error::failed_precondition("Vulkan pipeline creation lock is poisoned"))?;
		let raw = guard
			.cache
			.as_ref()
			.map_or(vk::PipelineCache::NULL, pipeline::PipelineCache::raw);
		Ok(PipelineCacheGuard { _guard: guard, raw })
	}

	/// Create the vertex-color lit graphics pipeline for this device.
	pub(in crate::runtime) fn create_graphics_pipeline(
		&self,
	) -> Result<super::pipeline::GraphicsPipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::GraphicsPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the flat-color material pipeline for this device.
	pub(in crate::runtime) fn create_flat_color_pipeline(
		&self,
	) -> Result<super::pipeline::FlatColorPipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::FlatColorPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the procedural canvas-grid pipeline for this device.
	pub(in crate::runtime) fn create_canvas_grid_pipeline(
		&self,
	) -> Result<super::pipeline::CanvasGridPipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::CanvasGridPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create a compute pipeline for an ad-hoc embedded artifact on the shared
	/// bindless descriptor layout. Reserved for non-registry kernels such as
	/// the video encoder's RGBA→NV12 conversion.
	#[cfg(oa_profile = "core")]
	pub(in crate::runtime) fn create_compute_pipeline(
		&self,
		artifact: &'static crate::runtime::dna::kernel::ShaderArtifact,
	) -> Result<super::pipeline::ComputePipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::ComputePipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.descriptor_layout(),
			artifact,
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the owned descriptor set layouts for the unlit pipeline.
	pub(in crate::runtime) fn create_unlit_descriptor_layouts(
		&self,
	) -> Result<super::pipeline::UnlitDescriptorLayouts> {
		super::pipeline::UnlitDescriptorLayouts::new(std::sync::Arc::clone(&self.logical.handle))
	}

	pub(in crate::runtime) fn create_unlit_descriptor_arena(
		&self,
		layout: vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<super::pipeline::UnlitDescriptorArena> {
		super::pipeline::UnlitDescriptorArena::new(
			std::sync::Arc::clone(&self.logical.handle),
			layout,
			set_count,
		)
	}

	/// Create the owned descriptor set layouts for the Surface pipeline.
	pub(in crate::runtime) fn create_surface_descriptor_layouts(
		&self,
	) -> Result<super::pipeline::SurfaceDescriptorLayouts> {
		super::pipeline::SurfaceDescriptorLayouts::new(std::sync::Arc::clone(&self.logical.handle))
	}

	pub(in crate::runtime) fn create_surface_descriptor_arena(
		&self,
		scene_layout: vk::DescriptorSetLayout,
		material_layout: vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<super::pipeline::SurfaceDescriptorArena> {
		super::pipeline::SurfaceDescriptorArena::new(
			std::sync::Arc::clone(&self.logical.handle),
			scene_layout,
			material_layout,
			set_count,
			max_materials_per_slot,
		)
	}

	/// Create the unlit textured material pipeline for this device.
	///
	/// `sampler_layout` is the descriptor-set layout for set 1 (one combined
	/// image sampler at binding 0, the base-color texture).
	pub(in crate::runtime) fn create_unlit_pipeline(
		&self,
		sampler_layout: vk::DescriptorSetLayout,
	) -> Result<super::pipeline::UnlitPipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::UnlitPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			sampler_layout,
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the Surface PBR pipeline for this device.
	///
	/// `scene_layout` is set 0 (scene lights UBO).
	/// `material_layout` is set 1 (textures + material params UBO).
	pub(in crate::runtime) fn create_surface_pipeline(
		&self,
		scene_layout: vk::DescriptorSetLayout,
		material_layout: vk::DescriptorSetLayout,
	) -> Result<super::pipeline::SurfacePipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::SurfacePipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			scene_layout,
			material_layout,
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the Surface transparent-blend pipeline for this device.
	///
	/// Shares the same pipeline layout (and descriptor set layouts) as the
	/// opaque variant; only the depth-write and blend state differ.
	pub(in crate::runtime) fn create_surface_blend_pipeline(
		&self,
		layout: vk::PipelineLayout,
	) -> Result<super::pipeline::SurfaceBlendPipeline> {
		let cache = self.pipeline_cache()?;
		super::pipeline::SurfaceBlendPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			layout,
			cache.raw(),
		)
	}

	/// Allocate a host-visible persistently-mapped buffer for mesh vertex or index data.
	pub(in crate::runtime) fn allocate_mesh_buffer(
		self: &Arc<Self>,
		size: usize,
		usage: vk::BufferUsageFlags,
	) -> Result<MeshBuffer> {
		MeshBuffer::new(self, size, usage)
	}

	pub(in crate::runtime) fn video_decode_capabilities(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeCapabilities> {
		super::video::query_decode_capabilities(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_decode_formats(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeFormats> {
		super::video::query_decode_formats(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_encode_capabilities(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeCapabilities> {
		super::video::query_encode_capabilities(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_encode_formats(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeFormats> {
		super::video::query_encode_formats(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn physical(&self) -> &DevicePhysical {
		&self.physical
	}

	/// Merged driver cache bytes for the Engine persistence coordinator, or
	/// `None` when no cache exists or the query fails — the coordinator keeps
	/// the previously committed section in that case.
	pub(in crate::runtime) fn pipeline_cache_data(&self) -> Option<Vec<u8>> {
		self.logical.pipeline_cache_data()
	}

	/// Device-lifetime fusion decision counters.
	pub(in crate::runtime) fn fusion_census(
		&self,
	) -> &crate::runtime::dna::fusion::census::FusionCensus {
		&self.logical.fusion_census
	}

	/// Per-device tape-vs-per-op element bound for elementwise fusion.
	/// Below or equal, a fused segment may take the tape route; above, the
	/// tape route declines and the segment splits to per-op kernels unless
	/// JIT admits a specialization.
	pub(in crate::runtime) fn tape_bound_elements(&self) -> u32 {
		self.logical.tape_bound_elements
	}

	/// Per-device single-head SDPA merge admission. Routing policy, not a
	/// capability bound — the qualified trajectory record, not kernel
	/// presence, is the gate.
	pub(in crate::runtime) fn single_head_sdpa_route(&self) -> bool {
		self.logical.single_head_sdpa_route
	}

	/// Per-device estimated submission-cost budget in nanoseconds. Plans
	/// whose summed node estimates exceed it split at legal node
	/// boundaries; a single node over budget declines before submission
	/// with a typed error. Estimate, not a hang guarantee.
	pub(in crate::runtime) fn submission_budget_ns(&self) -> u64 {
		self.logical.submission_budget_ns
	}

	/// Whether the opt-in `OA_JIT_FUSION` specialization route is enabled.
	pub(in crate::runtime) fn jit_fusion_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_JIT_FUSION")
	}

	/// Opt-in `OA_JIT_FUSION` admission: compile or restore the program's
	/// specialization and ensure a dispatchable pipeline exists. The artifact
	/// key is returned for the fused node on success; every failure mode —
	/// gate off, compiler or `spirv-val` absent, artifact or pipeline
	/// rejected — returns `None` so the canonical tape route executes.
	/// `tape_eligible` is `false` above the tape element bound; there the
	/// specialized route is the only fused route and no tape probe runs.
	pub(in crate::runtime) fn jit_admit(
		&self,
		program: &Program,
		tape_eligible: bool,
	) -> Option<Vec<u8>> {
		if !self.jit_fusion_enabled() {
			self
				.fusion_census()
				.jit(crate::runtime::dna::fusion::census::JitOutcome::Disabled);
			return None;
		}
		self
			.logical
			.jit
			.lock()
			.ok()?
			.admit(self, program, tape_eligible)
	}

	/// Whether the opt-in `OA_TILE_GEMM` tile-template route is enabled.
	pub(in crate::runtime) fn tile_gemm_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GEMM")
	}

	/// Ungated tile-template admission: synchronously compile and publish
	/// the block program's pipeline, returning its artifact key for the
	/// jit override on success. Callers own the route's opt-in gate;
	/// every failure mode returns `None` so the canonical registry
	/// kernel executes.
	/// Route-observation signature for a tile block program — one problem
	/// class per (template, geometry, descriptor profile).
	pub(in crate::runtime) fn tile_signature(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> u64 {
		super::specialization::block_signature(program)
	}

	pub(in crate::runtime) fn jit_admit_tile(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		self.logical.jit.lock().ok()?.admit_block(self, program)
	}

	/// CC-12 candidate evidence for `tile_signature`: `(specialized
	/// median ns, contributing samples, admission attempts)` recorded
	/// under the signature — live observations first, restored
	/// `.vkcache` medians while a window is still empty.
	pub(in crate::runtime) fn tile_route_stats(&self, signature: u64) -> (Option<u64>, usize, u64) {
		self
			.logical
			.jit
			.lock()
			.map(|mut jit| jit.route_stats(self, signature))
			.unwrap_or((None, 0, 0))
	}

	/// Opt-in `OA_TILE_GEMM` admission: synchronously compile and publish
	/// the block program's pipeline, returning its artifact key for the
	/// jit override on success. Every failure mode returns `None` so the
	/// canonical registry kernel executes.
	pub(in crate::runtime) fn jit_admit_tile_gemm(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_gemm_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Whether the opt-in `OA_TILE_SOFTMAX` tile-template route is enabled.
	pub(in crate::runtime) fn tile_softmax_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_SOFTMAX")
	}

	/// Whether the opt-in `OA_TILE_VIDEO` route is enabled.
	pub(in crate::runtime) fn tile_video_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_VIDEO")
	}

	/// Opt-in `OA_TILE_VIDEO` admission on the block pipeline table —
	/// the `to_matrix` typed plane-load programs.
	pub(in crate::runtime) fn jit_admit_tile_video(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_video_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Whether the opt-in `OA_TILE_IMAGE` route is enabled.
	pub(in crate::runtime) fn tile_image_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_IMAGE")
	}

	/// Opt-in `OA_TILE_IMAGE` admission on the block pipeline table —
	/// the image gather programs.
	pub(in crate::runtime) fn jit_admit_tile_image(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_image_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Whether the opt-in `OA_TILE_AUDIO` route is enabled.
	pub(in crate::runtime) fn tile_audio_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_AUDIO")
	}

	/// Opt-in `OA_TILE_AUDIO` admission on the block pipeline table —
	/// the audio `channel_normalize` programs.
	pub(in crate::runtime) fn jit_admit_tile_audio(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_audio_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Opt-in `OA_TILE_SOFTMAX` admission on the same block pipeline
	/// table as `jit_admit_tile_gemm`.
	pub(in crate::runtime) fn jit_admit_tile_softmax(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_softmax_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Whether the opt-in `OA_TILE_GEMV` route is enabled.
	pub(in crate::runtime) fn tile_gemv_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GEMV")
	}

	/// Opt-in `OA_TILE_GEMV` admission on the block pipeline table.
	pub(in crate::runtime) fn jit_admit_tile_gemv(
		&self,
		program: &crate::runtime::dna::ir::tile::BlockProgram,
	) -> Option<Vec<u8>> {
		if !self.tile_gemv_enabled() {
			return None;
		}
		self.jit_admit_tile(program)
	}

	/// Whether the opt-in `OA_TILE_LOG_SOFTMAX` route is enabled.
	pub(in crate::runtime) fn tile_log_softmax_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LOG_SOFTMAX")
	}

	/// Whether the opt-in `OA_TILE_RMS_NORM` route is enabled.
	pub(in crate::runtime) fn tile_rms_norm_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_RMS_NORM")
	}

	/// Whether the opt-in `OA_TILE_LAYER_NORM` route is enabled.
	pub(in crate::runtime) fn tile_layer_norm_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LAYER_NORM")
	}

	/// Whether the opt-in `OA_TILE_SOFTMAX_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_softmax_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_SOFTMAX_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_LOG_SOFTMAX_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_log_softmax_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LOG_SOFTMAX_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_LAYER_NORM_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_layer_norm_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LAYER_NORM_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_RMS_NORM_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_rms_norm_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_RMS_NORM_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_LINEAR` route is enabled.
	pub(in crate::runtime) fn tile_linear_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LINEAR")
	}

	/// Whether the opt-in `OA_TILE_LINEAR_ACT` route is enabled.
	pub(in crate::runtime) fn tile_linear_act_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LINEAR_ACT")
	}

	/// Whether the opt-in `OA_TILE_SUM` route is enabled.
	pub(in crate::runtime) fn tile_sum_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_SUM")
	}

	/// Whether the opt-in `OA_TILE_SUM_AXIS` route is enabled.
	pub(in crate::runtime) fn tile_sum_axis_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_SUM_AXIS")
	}

	/// Whether the opt-in `OA_TILE_LINEAR_TRIPLE` route is enabled.
	pub(in crate::runtime) fn tile_linear_triple_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LINEAR_TRIPLE")
	}

	pub(in crate::runtime) fn tile_linear_concat_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LINEAR_CONCAT")
	}

	/// Whether the opt-in `OA_TILE_RESIDUAL_NORM` route is enabled.
	pub(in crate::runtime) fn tile_residual_norm_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_RESIDUAL_NORM")
	}

	/// Whether the opt-in `OA_TILE_QKV` route is enabled.
	pub(in crate::runtime) fn tile_qkv_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_QKV")
	}

	/// Whether the opt-in `OA_TILE_GATE_UP` route is enabled.
	pub(in crate::runtime) fn tile_gate_up_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GATE_UP")
	}

	/// Whether the opt-in `OA_TILE_GROUPED_LINEAR_ACT` route is enabled.
	pub(in crate::runtime) fn tile_grouped_linear_act_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GROUPED_LINEAR_ACT")
	}

	/// Whether the opt-in `OA_TILE_GROUPED_LINEAR` route is enabled.
	pub(in crate::runtime) fn tile_grouped_linear_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GROUPED_LINEAR")
	}

	/// Whether the opt-in `OA_TILE_GROUPED_GEMM` route is enabled.
	pub(in crate::runtime) fn tile_grouped_gemm_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GROUPED_GEMM")
	}

	/// Whether the opt-in `OA_TILE_GROUPED_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_grouped_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GROUPED_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_MOE` route is enabled.
	pub(in crate::runtime) fn tile_moe_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_MOE")
	}

	/// Whether the opt-in `OA_TILE_CONV` route is enabled.
	pub(in crate::runtime) fn tile_conv_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_CONV")
	}

	/// Whether the opt-in `OA_TILE_POOL` route is enabled.
	pub(in crate::runtime) fn tile_pool_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_POOL")
	}

	/// Whether the opt-in `OA_TILE_CROSS_ENTROPY` route is enabled.
	pub(in crate::runtime) fn tile_cross_entropy_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_CROSS_ENTROPY")
	}

	/// Whether the opt-in `OA_TILE_EMBEDDING` route is enabled.
	pub(in crate::runtime) fn tile_embedding_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_EMBEDDING")
	}

	/// Whether the opt-in `OA_TILE_MAX` route is enabled.
	pub(in crate::runtime) fn tile_max_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_MAX")
	}

	/// Whether the opt-in `OA_TILE_MAX_WAVE` subgroup route is enabled.
	pub(in crate::runtime) fn tile_max_wave_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_MAX_WAVE")
	}

	/// Whether the opt-in `OA_TILE_ATTENTION` route is enabled.
	pub(in crate::runtime) fn tile_attention_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_ATTENTION")
	}

	/// Whether the opt-in `OA_TILE_LINEAR_BACKWARD` route is enabled.
	pub(in crate::runtime) fn tile_linear_backward_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_LINEAR_BACKWARD")
	}

	/// Whether the opt-in `OA_TILE_SUM_WAVE` subgroup route is enabled.
	/// The wave candidate changes sum's reduction order versus the
	/// shared tree, so callers qualify it against a tolerance oracle.
	pub(in crate::runtime) fn tile_sum_wave_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_SUM_WAVE")
	}

	/// Explicit reduction-order mode: `OA_TILE_REDUCE_ORDER=throughput`
	/// offers every subgroup-order candidate the device supports, and
	/// `deterministic` suppresses order-changing candidates even when
	/// their per-kernel flag is set. Unset keeps the per-kernel flags.
	pub(in crate::runtime) fn tile_reduce_order(&self) -> TileReduceOrder {
		match crate::EnvFlag::get_string("OA_TILE_REDUCE_ORDER", "").as_str() {
			"throughput" => TileReduceOrder::Throughput,
			"deterministic" => TileReduceOrder::Deterministic,
			_ => TileReduceOrder::Default,
		}
	}

	/// Whether an order-changing subgroup reduction candidate is offered
	/// under the order mode, the per-kernel flag, and the queried
	/// subgroup arithmetic support.
	pub(in crate::runtime) fn tile_reduce_wave_offered(&self, per_kernel_flag: bool) -> bool {
		let offered = match self.tile_reduce_order() {
			TileReduceOrder::Deterministic => false,
			TileReduceOrder::Throughput => true,
			TileReduceOrder::Default => per_kernel_flag,
		};
		offered
			&& (self.physical().limits.subgroup_supported_stages & vk::ShaderStageFlagBits::COMPUTE_BIT)
				== vk::ShaderStageFlagBits::COMPUTE_BIT
			&& (self.physical().limits.subgroup_supported_operations
				& vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT)
				== vk::SubgroupFeatureFlagBits::ARITHMETIC_BIT
	}

	/// Whether the opt-in `OA_TILE_GEMM_BIAS` route for the
	/// `mat_mul_nt_bias*` epilogue donors is enabled.
	pub(in crate::runtime) fn tile_gemm_bias_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GEMM_BIAS")
	}

	/// Whether the opt-in `OA_TILE_GEMM_SEARCH` geometry search is
	/// enabled: the seed tile stays the `OA_TILE_GEMM` candidate while
	/// the flag admits the bounded 32/64 row-column set as measured
	/// CC-12 candidates.
	pub(in crate::runtime) fn tile_gemm_search_enabled(&self) -> bool {
		crate::EnvFlag::is_set("OA_TILE_GEMM_SEARCH")
	}

	/// Record-time pipeline lookup for a key `jit_admit` prepared. Extension
	/// pipelines admitted through `admit_extension_pipeline` resolve first;
	/// the fusion jit table is consulted only on a miss.
	pub(in crate::runtime) fn jit_pipeline(
		&self,
		key: &[u8],
	) -> Option<std::sync::Arc<pipeline::ComputePipeline>> {
		if let Ok(extensions) = self.logical.extensions.lock()
			&& let Some(pipeline) = extensions.get(key)
		{
			return Some(pipeline.clone());
		}
		self.logical.jit.lock().ok()?.pipeline(self, key)
	}

	/// Admit a caller-owned SPIR-V stream as an extension-kernel pipeline under
	/// `key`. Identical keys reuse the cached pipeline; validation (SPIR-V
	/// structure, push range, workgroup limits) happens inside
	/// [`pipeline::ComputePipeline::from_words`].
	pub(crate) fn admit_extension_pipeline(
		&self,
		key: Vec<u8>,
		words: &[u32],
		workgroup_size: [u32; 3],
	) -> Result<Arc<pipeline::ComputePipeline>> {
		let mut extensions = self
			.logical
			.extensions
			.lock()
			.map_err(|_| Error::internal("extension pipeline table lock poisoned"))?;
		if let Some(pipeline) = extensions.get(&key) {
			return Ok(pipeline.clone());
		}
		let pipeline = pipeline::ComputePipeline::from_words(
			self.raw_arc(),
			self.descriptor_layout(),
			words,
			workgroup_size,
			self.physical().limits,
			self.pipeline_cache()?.raw(),
		)?;
		let pipeline = Arc::new(pipeline);
		extensions.insert(key, pipeline.clone());
		Ok(pipeline)
	}

	/// `.vkcache` section updates covering every admitted JIT artifact.
	pub(in crate::runtime) fn jit_section_updates(&self) -> Vec<super::vkcache::SectionUpdate> {
		let census = self.fusion_census();
		self
			.logical
			.jit
			.lock()
			.map(|mut fusion| fusion.section_updates(census))
			.unwrap_or_default()
	}

	/// Create a per-dispatch route-timing pool for one recorded graph.
	pub(in crate::runtime) fn route_timings(
		self: &Arc<Self>,
		entries: Vec<(u64, ObservedRoute)>,
	) -> Result<Arc<RouteTimings>> {
		RouteTimings::new(
			self,
			entries,
			self.logical.timestamp_period_ns,
			self.logical.compute_timestamp_valid_bits,
		)
	}

	/// Queue a recorded route-timing pool for lazy observation readback.
	pub(in crate::runtime) fn jit_observe_timings(&self, timings: Arc<RouteTimings>) {
		if let Ok(mut fusion) = self.logical.jit.lock() {
			fusion.observe(timings);
		}
	}

	pub(in crate::runtime) fn instance(&self) -> &Instance {
		&self.instance
	}

	pub(in crate::runtime) fn graphics_queue_handle(&self) -> Result<vk::Queue> {
		self
			.logical
			.graphics_queue
			.as_ref()
			.copied()
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))
	}

	pub(in crate::runtime) fn create_video_decode_session(
		self: &Arc<Self>,
		profile: crate::video::VideoDecodeProfile,
		coded_extent: crate::video::VideoExtent,
		max_dpb_slots: u32,
		max_active_references: u32,
	) -> Result<super::video::DecodeSession> {
		super::video::create_decode_session(
			self,
			profile,
			coded_extent,
			max_dpb_slots,
			max_active_references,
		)
	}

	pub(in crate::runtime) fn create_video_encode_session(
		self: &Arc<Self>,
		settings: &crate::video::VideoEncodeConfig,
		plan: crate::video::VideoEncodePlan,
	) -> Result<super::video::EncodeSession> {
		super::video::create_encode_session(self, settings, plan)
	}

	pub(in crate::runtime) fn allocator(&self) -> &vk::mem::SynchronizedAllocator<'static> {
		&self.logical.allocator
	}

	pub(in crate::runtime) fn bind_storage_buffer(
		&self,
		buffer: vk::Buffer,
		size: vk::DeviceSize,
	) -> Result<u32> {
		self
			.logical
			.descriptors
			.bind_storage_buffer(&self.logical.handle, buffer, size)
	}

	pub(in crate::runtime) fn release_storage_buffer(&self, index: u32) {
		self.logical.descriptors.release_storage_buffer(index);
	}

	/// Return the total descriptor writes issued against the device heap.
	///
	/// Writes amortize at buffer/image binding, not per dispatch — the
	/// per-plan census compares a capture's before/after snapshots.
	pub(in crate::runtime) fn descriptor_write_count(&self) -> u64 {
		self.logical.descriptors.write_count()
	}

	/// Return the shared descriptor set for command recording.
	pub(in crate::runtime) fn descriptor_set(&self) -> vk::DescriptorSet {
		self.logical.descriptors.set()
	}

	/// Return the shared bindless descriptor set layout for pipeline creation.
	pub(in crate::runtime) fn descriptor_layout(&self) -> vk::DescriptorSetLayout {
		self.logical.descriptors.layout()
	}

	pub(in crate::runtime) fn bounded_descriptor_layout(
		&self,
		count: u32,
	) -> Option<vk::DescriptorSetLayout> {
		self.logical.descriptors.bounded_layout(count)
	}

	/// Allocate one STORAGE_IMAGE descriptor slot (binding=1) and write `view`.
	pub(in crate::runtime) fn bind_storage_image(&self, view: vk::ImageView) -> Result<u32> {
		self
			.logical
			.descriptors
			.bind_storage_image(&self.logical.handle, view)
	}

	pub(in crate::runtime) fn release_storage_image(&self, index: u32) {
		self.logical.descriptors.release_storage_image(index);
	}

	pub(super) fn take_recycled_storage_buffer(&self, size: usize) -> Option<RecycledBuffer> {
		let mut pool = match self.logical.storage_pool.lock() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let index = pool.free.iter().rposition(|buffer| buffer.size == size)?;
		let buffer = pool.free.swap_remove(index);
		pool.bytes = pool.bytes.saturating_sub(buffer.size);
		Some(buffer)
	}

	pub(super) fn recycle_storage_buffer(&self, buffer: RecycledBuffer) -> Option<RecycledBuffer> {
		let mut pool = match self.logical.storage_pool.lock() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let Some(bytes) = pool.bytes.checked_add(buffer.size) else {
			return Some(buffer);
		};
		if pool.free.len() >= MAX_RECYCLED_STORAGE_BUFFERS || bytes > MAX_RECYCLED_STORAGE_BYTES {
			return Some(buffer);
		}
		pool.bytes = bytes;
		pool.free.push(buffer);
		None
	}

	pub(in crate::runtime) fn discard_one_recycled_storage_buffer(&self) -> bool {
		let buffer = {
			let mut pool = match self.logical.storage_pool.lock() {
				Ok(pool) => pool,
				Err(poisoned) => poisoned.into_inner(),
			};
			let Some(buffer) = pool.free.pop() else {
				return false;
			};
			pool.bytes = pool.bytes.saturating_sub(buffer.size);
			buffer
		};
		self.destroy_recycled_storage_buffer(buffer);
		true
	}

	fn destroy_recycled_storage_buffer(&self, mut buffer: RecycledBuffer) {
		self.release_storage_buffer(buffer.descriptor_index);
		// SAFETY: the pool exclusively owns this completed buffer/allocation pair,
		// and this device owns the allocator that created it.
		unsafe {
			let _ = self
				.allocator()
				.destroy_buffer(buffer.handle, &mut buffer.allocation);
		}
	}

	pub(in crate::runtime) fn same_as(&self, other: &Self) -> bool {
		std::ptr::eq(self, other)
	}

	pub(in crate::runtime) fn prepare_mlkem_keygen<'a>(
		self: &'a Arc<Self>,
		seed: super::secret_buffer::SecretBinding<'a>,
		private: super::secret_buffer::SecretBinding<'a>,
		public: &Buffer,
		k: u32,
	) -> Result<super::pqc::PreparedSecretKeygen<'a>> {
		let operands = super::pqc::SecretKeygenOperands::mlkem(self, seed, private, public, k)?;
		self.prepare_secret_keygen(operands)
	}

	pub(in crate::runtime) fn prepare_mldsa_keygen<'a>(
		self: &'a Arc<Self>,
		seed: super::secret_buffer::SecretBinding<'a>,
		private: super::secret_buffer::SecretBinding<'a>,
		public: &Buffer,
		parameter: u32,
	) -> Result<super::pqc::PreparedSecretKeygen<'a>> {
		let operands = super::pqc::SecretKeygenOperands::mldsa(self, seed, private, public, parameter)?;
		self.prepare_secret_keygen(operands)
	}

	fn prepare_secret_keygen<'a>(
		self: &'a Arc<Self>,
		operands: super::pqc::SecretKeygenOperands<'a>,
	) -> Result<super::pqc::PreparedSecretKeygen<'a>> {
		let kernel = operands.kernel();
		self.require_kernel(kernel)?;
		let pipeline = self
			.logical
			.pipelines
			.get(kernel.index())
			.and_then(Option::as_ref)
			.and_then(OnceLock::get)
			.ok_or_else(|| Error::missing_capability("PQC keygen pipeline unavailable"))?;
		operands.prepare(pipeline)
	}

	pub(super) fn prepare_secret_consumer<'a>(
		self: &'a Arc<Self>,
		operands: super::pqc::SecretConsumerOperands<'a>,
	) -> Result<super::pqc::PreparedSecretConsumer<'a>> {
		if !operands.belongs_to(self) {
			return Err(Error::invalid_argument(
				"PQC operands require their originating Device",
			));
		}
		let kernel = operands.kernel();
		self.require_kernel(kernel)?;
		let pipeline = self
			.logical
			.pipelines
			.get(kernel.index())
			.and_then(Option::as_ref)
			.and_then(OnceLock::get)
			.ok_or_else(|| Error::missing_capability("PQC consumer pipeline unavailable"))?;
		operands.prepare(pipeline)
	}

	pub(in crate::runtime) fn require_kernel(&self, kernel: KernelId) -> Result<()> {
		let pipeline = self.logical.pipelines.get(kernel.index()).ok_or_else(|| {
			Error::internal(format!(
				"{} has no generated pipeline-table slot",
				kernel.report_name()
			))
		})?;
		if let Some(slot) = pipeline {
			if slot.get().is_none() {
				// Serialize fallible first-use creation without publishing partial state.
				// The OnceLock keeps immutable pipeline borrows valid through recording;
				// Device remains the sole retained owner and destruction boundary.
				let mut creation = self
					.logical
					.pipeline_creation
					.lock()
					.map_err(|_| Error::failed_precondition("Vulkan pipeline creation lock is poisoned"))?;
				if slot.get().is_none() {
					let artifact = kernel.artifact();
					let layout = if self.physical.profile == ExecutionProfile::Compatibility {
						self
							.logical
							.descriptors
							.bounded_layout(kernel.bounded_buffer_count())
							.ok_or_else(|| {
								Error::missing_capability("bounded pipeline descriptor layout is unavailable")
							})?
					} else {
						self.logical.descriptors.layout()
					};
					let cache = creation
						.cache
						.as_ref()
						.map_or(vk::PipelineCache::NULL, pipeline::PipelineCache::raw);
					let created = match ComputePipeline::new(
						std::sync::Arc::clone(&self.logical.handle),
						layout,
						artifact,
						self.physical.limits,
						cache,
					) {
						Ok(pipeline) => pipeline,
						Err(first_error) => {
							if creation.cache.is_none() {
								return Err(first_error);
							}
							// A corrupt or incompatible persisted blob must not make
							// the shader unloadable; drop the cache and its persisted
							// driver section only, then retry creation uncached.
							let scope = super::vkcache::VkCacheScope::new(&self.physical);
							let scope_path = scope.path();
							crate::log_warn!(
								crate::LogComponent::COMPUTE,
								"pipeline cache: creation failed for {}; dropping {}",
								kernel.report_name(),
								scope_path.display()
							);
							creation.cache = None;
							let _ = super::vkcache::update(
								&scope_path,
								&scope,
								&[super::vkcache::SectionUpdate::remove(
									super::vkcache::SECTION_DRIVER_PIPELINE_CACHE,
								)],
							);
							ComputePipeline::new(
								std::sync::Arc::clone(&self.logical.handle),
								layout,
								artifact,
								self.physical.limits,
								vk::PipelineCache::NULL,
							)?
						}
					};
					if let Err(mut duplicate) = slot.set(created) {
						// A second publication would violate the serialized owning boundary.
						// Destroy only our unpublished pipeline, preserving the retained one.
						duplicate.destroy(&self.logical.handle);
						return Err(Error::internal(
							"Vulkan pipeline was published outside its creation lock",
						));
					}
				}
			}
			return Ok(());
		}
		if self.physical.profile == ExecutionProfile::Compatibility {
			return Err(Error::missing_capability(format!(
				"{} has no device-admitted bounded SPIR-V 1.5 pipeline on the Compatibility profile",
				kernel.report_name()
			)));
		}
		let requirements = kernel.artifact().requirements()?;
		let missing = self.configuration.missing_shader_requirements(
			self.physical.limits.subgroup_supported_stages,
			self.physical.limits.subgroup_supported_operations,
			requirements,
		);
		if missing.is_empty() {
			return Err(Error::missing_capability(format!(
				"{} has no qualified Vulkan pipeline",
				kernel.report_name()
			)));
		}
		Err(Error::missing_capability(format!(
			"{} requires Vulkan capabilities not enabled on this device: {}",
			kernel.report_name(),
			missing.join(", ")
		)))
	}

	/// Compile every admitted-but-unbuilt kernel pipeline, spreading work
	/// across `OA_SHADER_LOAD_THREADS` workers.
	///
	/// Vulkan requires external synchronization for one pipeline cache, so
	/// each worker receives its own cache seeded with the primary cache's
	/// snapshot; worker caches merge back into the primary cache only after
	/// all workers join. A single worker (or a warm persisted cache) reuses
	/// the serialized `require_kernel` path.
	///
	/// # Errors
	/// Returns the last pipeline-creation failure after all jobs finish; one
	/// failing shader does not stop the remaining pipelines from compiling.
	pub(in crate::runtime) fn preload_pipelines(&self) -> Result<()> {
		let mut jobs = Vec::new();
		for kernel in KernelId::ALL {
			let Some(Some(slot)) = self.logical.pipelines.get(kernel.index()) else {
				continue;
			};
			if slot.get().is_some() {
				continue;
			}
			let artifact = kernel.artifact();
			let layout = if self.physical.profile == ExecutionProfile::Compatibility {
				self
					.logical
					.descriptors
					.bounded_layout(kernel.bounded_buffer_count())
					.ok_or_else(|| {
						Error::missing_capability("bounded pipeline descriptor layout is unavailable")
					})?
			} else {
				self.logical.descriptors.layout()
			};
			jobs.push((kernel, artifact, layout));
		}
		if jobs.is_empty() {
			return Ok(());
		}

		// Snapshot the primary cache once before workers start; the immutable
		// seed gives every worker the warm data without sharing one cache.
		// "warm" is the persisted-seed size, not the queried blob size — even a
		// cold cache serializes a non-empty driver header.
		let (initial, warm) = {
			let creation = self
				.logical
				.pipeline_creation
				.lock()
				.map_err(|_| Error::failed_precondition("Vulkan pipeline creation lock is poisoned"))?;
			let initial = creation
				.cache
				.as_ref()
				.and_then(|cache| cache.data().ok())
				.unwrap_or_default();
			let warm = creation
				.cache
				.as_ref()
				.is_some_and(|cache| cache.initial_data_bytes() != 0);
			(initial, warm)
		};
		let configured = crate::EnvFlag::get_int("OA_SHADER_LOAD_THREADS", 0);
		let hardware = std::thread::available_parallelism().map_or(1, |count| count.get());
		let workers = if configured > 0 {
			usize::try_from(configured).unwrap_or(64).min(64)
		} else if warm {
			1
		} else {
			(hardware / 2).clamp(1, 8)
		}
		.clamp(1, jobs.len());

		crate::log_info!(
			crate::LogComponent::COMPUTE,
			"Loading {} shader pipeline{} on demand ({} thread{}, {} cache)",
			jobs.len(),
			if jobs.len() == 1 { "" } else { "s" },
			workers,
			if workers == 1 { "" } else { "s" },
			if warm { "warm" } else { "cold" }
		);
		let load_begin = std::time::Instant::now();

		let mut failures: Vec<(KernelId, Error)> = Vec::new();
		if workers == 1 {
			for (kernel, _, _) in &jobs {
				if let Err(error) = self.require_kernel(*kernel) {
					failures.push((*kernel, error));
				}
			}
		} else {
			let device = Arc::clone(&self.logical.handle);
			let worker_caches = (0..workers)
				.map(|_| pipeline::PipelineCache::new(Arc::clone(&device), &initial).ok())
				.collect::<Vec<_>>();
			let next = std::sync::atomic::AtomicUsize::new(0);
			let worker_failures = Mutex::new(Vec::<(KernelId, Error)>::new());
			let next_ref = &next;
			let failures_ref = &worker_failures;
			let jobs = &jobs;
			let pipelines = &self.logical.pipelines;
			let limits = self.physical.limits;
			let device = &device;
			std::thread::scope(|scope| {
				for worker_cache in &worker_caches {
					scope.spawn(move || {
						let cache = worker_cache
							.as_ref()
							.map_or(vk::PipelineCache::NULL, pipeline::PipelineCache::raw);
						loop {
							let index = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
							let Some((kernel, artifact, layout)) = jobs.get(index) else {
								break;
							};
							let Some(Some(slot)) = pipelines.get(kernel.index()) else {
								continue;
							};
							if slot.get().is_some() {
								continue;
							}
							// A corrupt or incompatible worker cache must not make
							// the shader unloadable; retry this pipeline uncached
							// without mutating the shared primary cache.
							let created =
								ComputePipeline::new(Arc::clone(device), *layout, artifact, limits, cache).or_else(
									|_| {
										ComputePipeline::new(
											Arc::clone(device),
											*layout,
											artifact,
											limits,
											vk::PipelineCache::NULL,
										)
									},
								);
							match created {
								Ok(pipeline) => {
									if let Err(mut duplicate) = slot.set(pipeline) {
										duplicate.destroy(device);
									}
								}
								Err(error) => {
									let _ = failures_ref
										.lock()
										.map(|mut failures| failures.push((*kernel, error)));
								}
							}
						}
					});
				}
			});
			// Merge only after every worker has joined; sources and destination
			// are then externally synchronized for vkMergePipelineCaches.
			let creation = self
				.logical
				.pipeline_creation
				.lock()
				.map_err(|_| Error::failed_precondition("Vulkan pipeline creation lock is poisoned"))?;
			if let Some(cache) = creation.cache.as_ref() {
				let sources = worker_caches
					.iter()
					.flatten()
					.map(pipeline::PipelineCache::raw)
					.collect::<Vec<_>>();
				if let Err(error) = cache.merge(&sources) {
					crate::log_warn!(
						crate::LogComponent::COMPUTE,
						"pipeline cache: worker merge failed: {error}"
					);
				}
			}
			failures.append(&mut worker_failures.into_inner().unwrap_or_default());
		}
		let load_ms = load_begin.elapsed().as_secs_f64() * 1000.0;
		for (kernel, error) in &failures {
			crate::log_warn!(
				crate::LogComponent::COMPUTE,
				"Failed to load shader '{}' on demand: {}",
				kernel.report_name(),
				error
			);
		}
		crate::log_info!(
			crate::LogComponent::COMPUTE,
			"Loaded {}/{} shader pipeline{} on demand (failed={}, threads={}, {:.2} ms)",
			jobs.len() - failures.len(),
			jobs.len(),
			if jobs.len() == 1 { "" } else { "s" },
			failures.len(),
			workers,
			load_ms
		);
		match failures.pop() {
			Some((_, error)) => Err(error),
			None => Ok(()),
		}
	}

	pub(in crate::runtime) fn raw(&self) -> &vk::Device {
		&self.logical().handle
	}

	/// Retained shared ownership of the native device. Consumers that keep
	/// `vk::Device` references across Engine borrows clone this Arc; teardown
	/// order remains owned by DeviceLogical's ManuallyDrop.
	pub(in crate::runtime) fn raw_arc(&self) -> Arc<vk::Device> {
		(*self.logical().handle).clone()
	}

	pub(in crate::runtime) fn sync_commands(&self) -> &SyncCommands {
		&self.logical.sync_commands
	}

	/// Copy the dynamic-rendering dispatch table for cross-module recording.
	pub(in crate::runtime) fn rendering_commands(&self) -> RenderingCommands {
		self.logical.rendering_commands.clone()
	}

	/// # Safety
	/// `command` must be recording on this device and the dependency must name
	/// valid retained resources with legal synchronization scopes.
	pub(in crate::runtime) unsafe fn pipeline_barrier(
		&self,
		command: vk::CommandBuffer,
		dependency: &vk::DependencyInfo<'_>,
	) -> Result<()> {
		unsafe {
			self
				.logical
				.sync_commands
				.pipeline_barrier(&self.logical.handle, command, dependency)
				.map_err(Error::from)
		}
	}

	pub(in crate::runtime) fn record_empty(&self) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_empty(&self.logical.handle)
	}

	pub(in crate::runtime) fn record_present_clear(
		&self,
		image: vk::Image,
		color: [f32; 4],
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.graphics_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let mut slot = self
			.logical
			.graphics_command_pool
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::Graphics,
			)?);
		}
		slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |command| {
				let range = vk::ImageSubresourceRange::new()
					.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
					.set_level_count(1)
					.set_layer_count(1);
				let barrier = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
					.set_old_layout(vk::ImageLayout::UNDEFINED)
					.set_new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.set_image(image)
					.set_subresource_range(range);
				let barriers = [barrier];
				let dependency = vk::DependencyInfo::new().set_image_memory_barriers(&barriers);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						command,
						&(dependency).map_err(Error::from)?,
					)
				}?;
				let clear = vk::ClearColorValue { float32: color };
				unsafe {
					self.logical.handle.cmd_clear_color_image(
						command,
						image,
						vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						&clear,
						&[range],
					)
				}?;
				let barrier = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::NONE)
					.set_old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.set_new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
					.set_image(image)
					.set_subresource_range(range);
				let barriers = [barrier];
				let dependency = vk::DependencyInfo::new().set_image_memory_barriers(&barriers);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						command,
						&(dependency).map_err(Error::from)?,
					)
				}?;
				Ok(())
			})
	}

	pub(in crate::runtime) fn record_present_blit(
		&self,
		source: vk::Image,
		source_extent: [u32; 2],
		target: vk::Image,
		target_extent: [u32; 2],
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.graphics_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let mut slot = self
			.logical
			.graphics_command_pool
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::Graphics,
			)?);
		}
		slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |command| {
				let range = vk::ImageSubresourceRange::new()
					.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
					.set_level_count(1)
					.set_layer_count(1);
				let barriers = [
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
						.set_src_access_mask(vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
						.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
						.set_old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
						.set_new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.set_image(source)
						.set_subresource_range(range),
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlagBits2::NONE)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
						.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
						.set_old_layout(vk::ImageLayout::UNDEFINED)
						.set_new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
						.set_image(target)
						.set_subresource_range(range),
				];
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							command,
							&(vk::DependencyInfo::new().set_image_memory_barriers(&barriers))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				let region = vk::ImageBlit::new()
					.set_src_subresource(
						vk::ImageSubresourceLayers::new()
							.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
							.set_layer_count(1),
					)
					.set_src_offsets([
						vk::Offset3D::new(),
						vk::Offset3D {
							x: source_extent[0] as i32,
							y: source_extent[1] as i32,
							z: 1,
						},
					])
					.set_dst_subresource(
						vk::ImageSubresourceLayers::new()
							.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
							.set_layer_count(1),
					)
					.set_dst_offsets([
						vk::Offset3D::new(),
						vk::Offset3D {
							x: target_extent[0] as i32,
							y: target_extent[1] as i32,
							z: 1,
						},
					]);
				unsafe {
					self.logical.handle.cmd_blit_image(
						command,
						source,
						vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						target,
						vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						&[region],
						vk::Filter::LINEAR,
					)
				}?;
				let barriers = [
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
						.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
						.set_dst_access_mask(vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT)
						.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.set_new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
						.set_image(source)
						.set_subresource_range(range),
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlagBits2::BLIT_BIT)
						.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::NONE)
						.set_old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
						.set_new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
						.set_image(target)
						.set_subresource_range(range),
				];
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							command,
							&(vk::DependencyInfo::new().set_image_memory_barriers(&barriers))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				Ok(())
			})
	}

	pub(in crate::runtime) fn submit_present(
		&self,
		command: &RecordedCommandBuffer,
		wait: vk::Semaphore,
		signal: vk::Semaphore,
		fence: vk::Fence,
	) -> Result<()> {
		let queue = *self
			.logical
			.graphics_queue
			.as_ref()
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let waits = [vk::SemaphoreSubmitInfo::new()
			.set_semaphore(wait)
			.set_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)];
		let commands = [vk::CommandBufferSubmitInfo::new().set_command_buffer(command.raw())];
		let signals = [vk::SemaphoreSubmitInfo::new()
			.set_semaphore(signal)
			.set_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)];
		let submits = [vk::SubmitInfo2::new()
			.set_wait_semaphore_infos(&waits)
			.map_err(Error::from)?
			.set_command_buffer_infos(&commands)
			.map_err(Error::from)?
			.set_signal_semaphore_infos(&signals)
			.map_err(Error::from)?];
		unsafe {
			self
				.logical
				.sync_commands
				.submit(&self.logical.handle, queue, &submits, fence)
				.map_err(|source| Error::backend_failure("Vulkan", "presentation queue submission", source))
		}
	}

	/// Record one graphics frame: barrier → begin rendering → draw → end rendering
	/// → image-to-buffer copy → barrier back to SHADER_READ_ONLY.
	///
	/// `push_data` is the raw bytes of the push constant block. `push_stages`
	/// selects which shader stages the push constants are visible to.
	#[allow(clippy::too_many_arguments)]
	pub(in crate::runtime) fn record_mesh_frame(
		&self,
		target: &mut RenderTarget,
		vertex_buffer: vk::Buffer,
		index_buffer: vk::Buffer,
		index_count: u32,
		clear_color: [f32; 4],
		push_data: &[u8],
		push_stages: vk::ShaderStageFlags,
		pipeline: vk::Pipeline,
		pipeline_layout: vk::PipelineLayout,
		descriptor_sets: &[vk::DescriptorSet],
	) -> Result<RecordedCommandBuffer> {
		self.record_draw_frame(
			target,
			FrameDraw::Indexed {
				vertex_buffer,
				index_buffer,
				index_count,
			},
			clear_color,
			push_data,
			push_stages,
			pipeline,
			pipeline_layout,
			descriptor_sets,
		)
	}

	/// Record one vertexless graphics frame — the vertex shader derives
	/// geometry from SV_VertexID, so no vertex or index buffers are bound.
	#[allow(clippy::too_many_arguments)]
	pub(in crate::runtime) fn record_vertexless_frame(
		&self,
		target: &mut RenderTarget,
		vertex_count: u32,
		clear_color: [f32; 4],
		push_data: &[u8],
		push_stages: vk::ShaderStageFlags,
		pipeline: vk::Pipeline,
		pipeline_layout: vk::PipelineLayout,
		descriptor_sets: &[vk::DescriptorSet],
	) -> Result<RecordedCommandBuffer> {
		self.record_draw_frame(
			target,
			FrameDraw::Vertexless { vertex_count },
			clear_color,
			push_data,
			push_stages,
			pipeline,
			pipeline_layout,
			descriptor_sets,
		)
	}

	#[allow(clippy::too_many_arguments)]
	fn record_draw_frame(
		&self,
		target: &mut RenderTarget,
		draw: FrameDraw,
		clear_color: [f32; 4],
		push_data: &[u8],
		push_stages: vk::ShaderStageFlags,
		pipeline: vk::Pipeline,
		pipeline_layout: vk::PipelineLayout,
		descriptor_sets: &[vk::DescriptorSet],
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.graphics_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let mut slot = self
			.logical
			.graphics_command_pool
			.lock()
			.unwrap_or_else(|p| p.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::Graphics,
			)?);
		}
		let color_image = target.color_image;
		let depth_image = target.depth_image;
		let color_view = target.color_view;
		let depth_view = target.depth_view;
		let readback_buf = target.readback_buffer();
		let width = target.width;
		let height = target.height;
		let old_color_layout = target.color_layout;
		let color_range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let depth_range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::DEPTH_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let result = slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |cmd| {
				// 1. barrier: transition color attachment and depth to write layouts.
				let to_attachment = [
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(
							if old_color_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT
							} else {
								vk::PipelineStageFlags2::empty()
							},
						)
						.set_src_access_mask(
							if old_color_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT
							} else {
								vk::AccessFlags2::empty()
							},
						)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::COLOR_ATTACHMENT_OUTPUT_BIT)
						.set_dst_access_mask(vk::AccessFlagBits2::COLOR_ATTACHMENT_WRITE_BIT)
						.set_old_layout(old_color_layout)
						.set_new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.set_image(color_image)
						.set_subresource_range(color_range),
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlags2::empty())
						.set_dst_stage_mask(
							vk::PipelineStageFlagBits2::EARLY_FRAGMENT_TESTS_BIT
								| vk::PipelineStageFlagBits2::LATE_FRAGMENT_TESTS_BIT,
						)
						.set_dst_access_mask(
							vk::AccessFlagBits2::DEPTH_STENCIL_ATTACHMENT_READ_BIT
								| vk::AccessFlagBits2::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						)
						.set_old_layout(vk::ImageLayout::UNDEFINED)
						.set_new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.set_image(depth_image)
						.set_subresource_range(depth_range),
				];
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new()
								.set_image_memory_barriers(&to_attachment)
								.map_err(Error::from)?),
						)
						.map_err(Error::from)?
				};
				// 2. begin dynamic rendering
				let clear_cv = vk::ClearColorValue {
					float32: clear_color,
				};
				let clear_dv = vk::ClearDepthStencilValue {
					depth: 1.0,
					stencil: 0,
				};
				let color_att = vk::RenderingAttachmentInfo::new()
					.set_image_view(color_view)
					.set_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.set_load_op(vk::AttachmentLoadOp::CLEAR)
					.set_store_op(vk::AttachmentStoreOp::STORE)
					.set_clear_value(vk::ClearValue { color: clear_cv });
				let depth_att = vk::RenderingAttachmentInfo::new()
					.set_image_view(depth_view)
					.set_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.set_load_op(vk::AttachmentLoadOp::CLEAR)
					.set_store_op(vk::AttachmentStoreOp::DONT_CARE)
					.set_clear_value(vk::ClearValue {
						depth_stencil: clear_dv,
					});
				let rendering_info = vk::RenderingInfo::new()
					.set_render_area(vk::Rect2D {
						offset: vk::Offset2D::new(),
						extent: vk::Extent2D { width, height },
					})
					.set_layer_count(1)
					.set_color_attachments(std::slice::from_ref(&color_att))
					.map_err(Error::from)?
					.set_depth_attachment(&depth_att);
				unsafe {
					self
						.logical
						.rendering_commands
						.begin(&self.logical.handle, cmd, &rendering_info)
				}?;
				// 3. set dynamic viewport and scissor
				let viewport = vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: width as f32,
					height: height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				};
				let scissor = vk::Rect2D {
					offset: vk::Offset2D::new(),
					extent: vk::Extent2D { width, height },
				};
				unsafe {
					self.logical.handle.cmd_set_viewport(cmd, 0, &[viewport])?;
					self.logical.handle.cmd_set_scissor(cmd, 0, &[scissor])?;
				};
				// 4. bind pipeline and push constants
				unsafe {
					self
						.logical
						.handle
						.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline)?;
					self.logical.handle.cmd_push_constants(
						cmd,
						pipeline_layout,
						push_stages,
						0,
						push_data,
					)?;
					if !descriptor_sets.is_empty() {
						self.logical.handle.cmd_bind_descriptor_sets(
							cmd,
							vk::PipelineBindPoint::GRAPHICS,
							pipeline_layout,
							0,
							descriptor_sets,
							&[],
						)?;
					}
				};
				// 5. bind vertex/index buffers (indexed) and draw
				unsafe {
					match draw {
						FrameDraw::Indexed {
							vertex_buffer,
							index_buffer,
							index_count,
						} => {
							self
								.logical
								.handle
								.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0])?;
							self.logical.handle.cmd_bind_index_buffer(
								cmd,
								index_buffer,
								0,
								vk::IndexType::UINT32,
							)?;
							self
								.logical
								.handle
								.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0)?;
						}
						FrameDraw::Vertexless { vertex_count } => {
							self.logical.handle.cmd_draw(cmd, vertex_count, 1, 0, 0)?;
						}
					}
				};
				// 6. end rendering
				unsafe {
					self
						.logical
						.rendering_commands
						.end(&self.logical.handle, cmd)
				}?;
				// 7. barrier: color attachment → TRANSFER_SRC for readback
				let to_transfer = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COLOR_ATTACHMENT_OUTPUT_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::COLOR_ATTACHMENT_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
					.set_old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.set_new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.set_image(color_image)
					.set_subresource_range(color_range);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new())
								.set_image_memory_barriers(std::slice::from_ref(&to_transfer))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				// 8. copy color image to host-visible readback buffer
				let copy = vk::BufferImageCopy::new()
					.set_image_subresource(
						vk::ImageSubresourceLayers::new()
							.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
							.set_layer_count(1),
					)
					.set_image_extent(vk::Extent3D {
						width,
						height,
						depth: 1,
					});
				unsafe {
					self.logical.handle.cmd_copy_image_to_buffer(
						cmd,
						color_image,
						vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						readback_buf,
						&[copy],
					)
				}?;
				// 9. barrier: TRANSFER_SRC → SHADER_READ_ONLY for future Presenter blit
				let to_sampled = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT)
					.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.set_new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
					.set_image(color_image)
					.set_subresource_range(color_range);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new())
								.set_image_memory_barriers(std::slice::from_ref(&to_sampled))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				Ok(())
			});
		if result.is_ok() {
			target.color_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		}
		result
	}

	/// Record one render pass with opaque draws first then optional blend draws.
	///
	/// `draws` carries all draw commands; entries with `is_blend == false` are
	/// emitted first (front-to-back order, depth write on via `pipeline`), then
	/// entries with `is_blend == true` are emitted with `blend_pipeline` (depth
	/// test on, depth write off, premultiplied alpha blend).
	///
	/// `push_stages` must include at least `VERTEX` (128-byte push for MVP +
	/// model matrix).
	#[allow(clippy::too_many_arguments)]
	pub(in crate::runtime) fn record_scene_frame(
		&self,
		target: &mut RenderTarget,
		vertex_buffer: vk::Buffer,
		index_buffer: vk::Buffer,
		clear_color: [f32; 4],
		push_stages: vk::ShaderStageFlags,
		pipeline: vk::Pipeline,
		blend_pipeline: vk::Pipeline,
		pipeline_layout: vk::PipelineLayout,
		scene_set: vk::DescriptorSet,
		draws: &[SceneDrawCmd],
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.graphics_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let mut slot = self
			.logical
			.graphics_command_pool
			.lock()
			.unwrap_or_else(|p| p.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::Graphics,
			)?);
		}
		let color_image = target.color_image;
		let depth_image = target.depth_image;
		let color_view = target.color_view;
		let depth_view = target.depth_view;
		let readback_buf = target.readback_buffer();
		let width = target.width;
		let height = target.height;
		let old_color_layout = target.color_layout;
		let color_range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let depth_range = vk::ImageSubresourceRange::new()
			.set_aspect_mask(vk::ImageAspectFlagBits::DEPTH_BIT)
			.set_level_count(1)
			.set_layer_count(1);
		let result = slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |cmd| {
				// 1. barrier: transition color attachment and depth to write layouts.
				let to_attachment = [
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(
							if old_color_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT
							} else {
								vk::PipelineStageFlags2::empty()
							},
						)
						.set_src_access_mask(
							if old_color_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT
							} else {
								vk::AccessFlags2::empty()
							},
						)
						.set_dst_stage_mask(vk::PipelineStageFlagBits2::COLOR_ATTACHMENT_OUTPUT_BIT)
						.set_dst_access_mask(vk::AccessFlagBits2::COLOR_ATTACHMENT_WRITE_BIT)
						.set_old_layout(old_color_layout)
						.set_new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.set_image(color_image)
						.set_subresource_range(color_range),
					vk::ImageMemoryBarrier2::new()
						.set_src_stage_mask(vk::PipelineStageFlags2::empty())
						.set_dst_stage_mask(
							vk::PipelineStageFlagBits2::EARLY_FRAGMENT_TESTS_BIT
								| vk::PipelineStageFlagBits2::LATE_FRAGMENT_TESTS_BIT,
						)
						.set_dst_access_mask(
							vk::AccessFlagBits2::DEPTH_STENCIL_ATTACHMENT_READ_BIT
								| vk::AccessFlagBits2::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						)
						.set_old_layout(vk::ImageLayout::UNDEFINED)
						.set_new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.set_image(depth_image)
						.set_subresource_range(depth_range),
				];
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new()
								.set_image_memory_barriers(&to_attachment)
								.map_err(Error::from)?),
						)
						.map_err(Error::from)?
				};
				// 2. begin dynamic rendering
				let clear_cv = vk::ClearColorValue {
					float32: clear_color,
				};
				let clear_dv = vk::ClearDepthStencilValue {
					depth: 1.0,
					stencil: 0,
				};
				let color_att = vk::RenderingAttachmentInfo::new()
					.set_image_view(color_view)
					.set_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.set_load_op(vk::AttachmentLoadOp::CLEAR)
					.set_store_op(vk::AttachmentStoreOp::STORE)
					.set_clear_value(vk::ClearValue { color: clear_cv });
				let depth_att = vk::RenderingAttachmentInfo::new()
					.set_image_view(depth_view)
					.set_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.set_load_op(vk::AttachmentLoadOp::CLEAR)
					.set_store_op(vk::AttachmentStoreOp::DONT_CARE)
					.set_clear_value(vk::ClearValue {
						depth_stencil: clear_dv,
					});
				let rendering_info = vk::RenderingInfo::new()
					.set_render_area(vk::Rect2D {
						offset: vk::Offset2D::new(),
						extent: vk::Extent2D { width, height },
					})
					.set_layer_count(1)
					.set_color_attachments(std::slice::from_ref(&color_att))
					.map_err(Error::from)?
					.set_depth_attachment(&depth_att);
				unsafe {
					self
						.logical
						.rendering_commands
						.begin(&self.logical.handle, cmd, &rendering_info)
				}?;
				// 3. set dynamic viewport and scissor
				let viewport = vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: width as f32,
					height: height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				};
				let scissor = vk::Rect2D {
					offset: vk::Offset2D::new(),
					extent: vk::Extent2D { width, height },
				};
				unsafe {
					self.logical.handle.cmd_set_viewport(cmd, 0, &[viewport])?;
					self.logical.handle.cmd_set_scissor(cmd, 0, &[scissor])?;
				};
				// 4. bind vertex/index buffers and scene set (set 0) once
				unsafe {
					self
						.logical
						.handle
						.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0])?;
					self
						.logical
						.handle
						.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32)?;
					self.logical.handle.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::GRAPHICS,
						pipeline_layout,
						0,
						&[scene_set],
						&[],
					)?;
				};

				// 5a. opaque pass — bind opaque pipeline, emit non-blend draws
				unsafe {
					self
						.logical
						.handle
						.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline)?;
				}
				for draw in draws.iter().filter(|d| !d.is_blend) {
					unsafe {
						self.logical.handle.cmd_push_constants(
							cmd,
							pipeline_layout,
							push_stages,
							0,
							&draw.push_data,
						)?;
						self.logical.handle.cmd_bind_descriptor_sets(
							cmd,
							vk::PipelineBindPoint::GRAPHICS,
							pipeline_layout,
							1,
							&[draw.material_set],
							&[],
						)?;
						self.logical.handle.cmd_draw_indexed(
							cmd,
							draw.index_count,
							1,
							draw.index_start,
							0,
							0,
						)?;
					}
				}

				// 5b. transparent pass — switch to blend pipeline, emit blend draws
				let has_blend = draws.iter().any(|d| d.is_blend);
				if has_blend {
					unsafe {
						self.logical.handle.cmd_bind_pipeline(
							cmd,
							vk::PipelineBindPoint::GRAPHICS,
							blend_pipeline,
						)?;
					}
					for draw in draws.iter().filter(|d| d.is_blend) {
						unsafe {
							self.logical.handle.cmd_push_constants(
								cmd,
								pipeline_layout,
								push_stages,
								0,
								&draw.push_data,
							)?;
							self.logical.handle.cmd_bind_descriptor_sets(
								cmd,
								vk::PipelineBindPoint::GRAPHICS,
								pipeline_layout,
								1,
								&[draw.material_set],
								&[],
							)?;
							self.logical.handle.cmd_draw_indexed(
								cmd,
								draw.index_count,
								1,
								draw.index_start,
								0,
								0,
							)?;
						}
					}
				}
				// 6. end rendering
				unsafe {
					self
						.logical
						.rendering_commands
						.end(&self.logical.handle, cmd)
				}?;
				// 7. barrier: color attachment → TRANSFER_SRC for readback
				let to_transfer = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COLOR_ATTACHMENT_OUTPUT_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::COLOR_ATTACHMENT_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
					.set_old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.set_new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.set_image(color_image)
					.set_subresource_range(color_range);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new())
								.set_image_memory_barriers(std::slice::from_ref(&to_transfer))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				// 8. copy color image to host-visible readback buffer
				let copy = vk::BufferImageCopy::new()
					.set_image_subresource(
						vk::ImageSubresourceLayers::new()
							.set_aspect_mask(vk::ImageAspectFlagBits::COLOR_BIT)
							.set_layer_count(1),
					)
					.set_image_extent(vk::Extent3D {
						width,
						height,
						depth: 1,
					});
				unsafe {
					self.logical.handle.cmd_copy_image_to_buffer(
						cmd,
						color_image,
						vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						readback_buf,
						&[copy],
					)
				}?;
				// 9. barrier: TRANSFER_SRC → SHADER_READ_ONLY for future Presenter blit
				let to_sampled = vk::ImageMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::SHADER_SAMPLED_READ_BIT)
					.set_old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.set_new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
					.set_image(color_image)
					.set_subresource_range(color_range);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(
							&self.logical.handle,
							cmd,
							&(vk::DependencyInfo::new())
								.set_image_memory_barriers(std::slice::from_ref(&to_sampled))
								.map_err(Error::from)?,
						)
						.map_err(Error::from)?
				};
				Ok(())
			});
		if result.is_ok() {
			target.color_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		}
		result
	}

	/// Record compositor compute work in the family used for presentation submission.
	pub(in crate::runtime) fn record_presentation_commands(
		&self,
		record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.graphics_queue_family
			.ok_or_else(|| Error::missing_capability("no graphics queue is enabled"))?;
		let mut slot = self
			.logical
			.graphics_command_pool
			.lock()
			.unwrap_or_else(|p| p.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::Graphics,
			)?);
		}
		slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, record)
	}

	pub(in crate::runtime) fn record_compute_commands(
		&self,
		record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_custom(&self.logical.handle, record)
	}

	pub(in crate::runtime) fn record_video_decode(
		&self,
		record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let mut slot = match self.logical.video_decode_command_pool.lock() {
			Ok(slot) => slot,
			Err(poisoned) => poisoned.into_inner(),
		};
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::VideoDecode,
			)?);
		}
		slot
			.as_mut()
			.expect("video decode command pool was initialized")
			.record_custom(&self.logical.handle, record)
	}

	pub(in crate::runtime) fn record_video_encode(
		&self,
		record: impl FnOnce(vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.physical
			.video
			.encode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video encode queue is enabled"))?;
		let mut slot = match self.logical.video_encode_command_pool.lock() {
			Ok(slot) => slot,
			Err(poisoned) => poisoned.into_inner(),
		};
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.logical.handle,
				family,
				CommandPoolKind::VideoEncode,
			)?);
		}
		slot
			.as_mut()
			.expect("video encode command pool was initialized")
			.record_custom(&self.logical.handle, record)
	}

	/// Queue-family index a command pool kind records for; the resolved
	/// family drives both recording and queue selection at submission.
	fn family_for_pool_kind(&self, kind: CommandPoolKind) -> Result<u32> {
		let physical = &self.physical;
		match kind {
			CommandPoolKind::Compute => Ok(physical.compute_queue_family),
			CommandPoolKind::Graphics => physical
				.graphics_queue_family
				.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled")),
			CommandPoolKind::Transfer => physical
				.transfer_queue_family
				.filter(|family| *family != physical.compute_queue_family)
				.ok_or_else(|| Error::missing_capability("no dedicated Vulkan transfer queue is enabled")),
			CommandPoolKind::VideoDecode => physical
				.video
				.decode_queue_family
				.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled")),
			CommandPoolKind::VideoEncode => physical
				.video
				.encode_queue_family
				.ok_or_else(|| Error::missing_capability("no Vulkan Video encode queue is enabled")),
		}
	}

	/// Run `record` against the command pool serving `family`, lazily
	/// creating the pool for enabled non-compute families on first use.
	fn with_pool_for_family<R>(
		&self,
		family: u32,
		record: impl FnOnce(&mut CommandPool) -> Result<R>,
	) -> Result<R> {
		let physical = &self.physical;
		if family == physical.compute_queue_family {
			let mut pool = self
				.logical
				.command_pool
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner());
			return record(&mut pool);
		}
		let (slot, kind) = if physical.transfer_queue_family == Some(family) {
			(
				&self.logical.transfer_command_pool,
				CommandPoolKind::Transfer,
			)
		} else if physical.graphics_queue_family == Some(family) {
			(
				&self.logical.graphics_command_pool,
				CommandPoolKind::Graphics,
			)
		} else if physical.video.decode_queue_family == Some(family) {
			(
				&self.logical.video_decode_command_pool,
				CommandPoolKind::VideoDecode,
			)
		} else if physical.video.encode_queue_family == Some(family) {
			(
				&self.logical.video_encode_command_pool,
				CommandPoolKind::VideoEncode,
			)
		} else {
			return Err(Error::missing_capability(format!(
				"no queue or command pool is enabled for queue family {family}"
			)));
		};
		let mut slot = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		if slot.is_none() {
			*slot = Some(CommandPool::new(&self.logical.handle, family, kind)?);
		}
		record(slot.as_mut().expect("command pool was initialized"))
	}

	/// Record `graph`'s node range `nodes` on the queue family the range
	/// resolves to. Every node in the range must share one resolved family;
	/// queue-run splitting is the caller's contract.
	fn record_graph_range(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
		nodes: std::ops::Range<usize>,
		timing: Option<TimestampPair>,
		reusable_device: Option<Arc<Device>>,
	) -> Result<RecordedCommandBuffer> {
		let family = graph
			.node_families()
			.get(nodes.start)
			.copied()
			.ok_or_else(|| Error::invalid_argument("execution graph range is empty"))?;
		self.with_pool_for_family(family, |pool| {
			pool.record_compute_graph(
				self,
				&self.logical.pipelines,
				self.logical.descriptors.set(),
				graph,
				nodes.clone(),
				timing.clone(),
				reusable_device.clone(),
			)
		})
	}

	pub(in crate::runtime) fn record_fresh_graph_range(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
		nodes: std::ops::Range<usize>,
		timed: bool,
	) -> Result<RecordedCommandBuffer> {
		let timing = if timed {
			let family = graph
				.node_families()
				.get(nodes.start)
				.copied()
				.ok_or_else(|| Error::invalid_argument("execution graph range is empty"))?;
			Some(self.timestamp_pair_for(family)?)
		} else {
			None
		};
		self.record_graph_range(graph, nodes, timing, None)
	}

	pub(in crate::runtime) fn record_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		self.record_graph_range(graph, 0..graph.nodes().len(), None, None)
	}

	pub(in crate::runtime) fn record_reusable_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<ReusableCommandBuffer> {
		let command =
			self.record_graph_range(graph, 0..graph.nodes().len(), None, Some(self.clone()))?;
		command.reusable().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"reusable command-buffer recording",
				std::io::Error::other("recording did not retain reusable ownership"),
			)
		})
	}

	/// Record one queue run of a mixed-family graph as a reusable command.
	/// The range's resolved family selects the pool and, at submission,
	/// the queue.
	pub(in crate::runtime) fn record_reusable_graph_range(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
		nodes: std::ops::Range<usize>,
	) -> Result<ReusableCommandBuffer> {
		let command = self.record_graph_range(graph, nodes, None, Some(self.clone()))?;
		command.reusable().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"reusable command-buffer recording",
				std::io::Error::other("recording did not retain reusable ownership"),
			)
		})
	}

	/// Create a fresh timestamp pair against this device's compute queue
	/// timestamp capability.
	pub(in crate::runtime) fn new_timestamp_pair(self: &Arc<Self>) -> Result<TimestampPair> {
		self.timestamp_pair_for(self.physical.compute_queue_family)
	}

	/// Create a timestamp pair for a specific queue family. Families
	/// advertising zero timestamp-valid bits cannot bracket timing.
	pub(in crate::runtime) fn timestamp_pair_for(
		self: &Arc<Self>,
		family: u32,
	) -> Result<TimestampPair> {
		let valid_bits = self.physical.timestamp_valid_bits(family);
		if valid_bits == 0 {
			return Err(Error::missing_capability(format!(
				"queue family {family} reports no timestamp support"
			)));
		}
		TimestampPair::new(self, self.logical.timestamp_period_ns, valid_bits)
	}

	pub(in crate::runtime) fn record_timed_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		let timing = self.new_timestamp_pair()?;
		self.record_timed_compute_graph_with(graph, timing)
	}

	/// Record a timed compute graph using a caller-owned timestamp pair. A
	/// reusable named device timer shares one pair across submissions.
	pub(in crate::runtime) fn record_timed_compute_graph_with(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
		timing: TimestampPair,
	) -> Result<RecordedCommandBuffer> {
		self.record_graph_range(graph, 0..graph.nodes().len(), Some(timing), None)
	}

	pub(in crate::runtime) fn time_reusable_command(
		self: &Arc<Self>,
		command: RecordedCommandBuffer,
	) -> Result<RecordedCommandBuffer> {
		let timing = self.timestamp_pair_for(self.family_for_pool_kind(command.pool_kind())?)?;
		self.time_reusable_command_with(command, timing)
	}

	/// Wrap a reusable command in begin/end timestamp brackets using a
	/// caller-owned pair. The bracket commands record in the same queue
	/// family as the command they wrap — a submission's bracket buffers
	/// must share its queue.
	pub(in crate::runtime) fn time_reusable_command_with(
		self: &Arc<Self>,
		command: RecordedCommandBuffer,
		timing: TimestampPair,
	) -> Result<RecordedCommandBuffer> {
		let family = self.family_for_pool_kind(command.pool_kind())?;
		self.with_pool_for_family(family, |pool| {
			let before = pool.record_custom(self.raw(), |buffer| timing.record_begin(buffer))?;
			let after = match pool.record_custom(self.raw(), |buffer| timing.record_end(buffer)) {
				Ok(after) => after,
				Err(error) => {
					pool.free_handle(self.raw(), before.raw());
					return Err(error);
				}
			};
			Ok(command.with_timing_brackets(before, after, timing))
		})
	}

	pub(in crate::runtime) fn submit(
		&self,
		command: &RecordedCommandBuffer,
		epoch: u64,
	) -> Result<()> {
		let command_infos = command
			.submission_handles()
			.into_iter()
			.map(|handle| vk::CommandBufferSubmitInfo::new().set_command_buffer(handle))
			.collect::<Vec<_>>();
		let signal_infos = [vk::SemaphoreSubmitInfo::new()
			.set_semaphore(self.logical.timeline.raw())
			.set_value(epoch)
			.set_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)];
		let wait_info = vk::SemaphoreSubmitInfo::new()
			.set_semaphore(self.logical.timeline.raw())
			.set_value(epoch.saturating_sub(1))
			.set_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT);
		let wait_infos = if epoch > 1 {
			std::slice::from_ref(&wait_info)
		} else {
			&[]
		};
		let submit_infos = [vk::SubmitInfo2::new()
			.set_wait_semaphore_infos(wait_infos)
			.map_err(Error::from)?
			.set_command_buffer_infos(&command_infos)
			.map_err(Error::from)?
			.set_signal_semaphore_infos(&signal_infos)
			.map_err(Error::from)?];
		let queue = match command.pool_kind() {
			CommandPoolKind::Compute => self.logical.compute_queue,
			CommandPoolKind::Graphics => *self
				.logical
				.graphics_queue
				.as_ref()
				.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?,
			CommandPoolKind::Transfer => *self.logical.transfer_queue.as_ref().ok_or_else(|| {
				Error::missing_capability("no dedicated Vulkan transfer queue is enabled")
			})?,
			CommandPoolKind::VideoDecode => *self
				.logical
				.video_decode_queue
				.as_ref()
				.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?,
			CommandPoolKind::VideoEncode => *self
				.logical
				.video_encode_queue
				.as_ref()
				.ok_or_else(|| Error::missing_capability("no Vulkan Video encode queue is enabled"))?,
		};

		// SAFETY: synchronization2 and timeline semaphores were enabled; the recorded
		// primary command buffer belongs to the selected queue family and is not pending.
		// The previous signal's first scope covers every earlier engine submission;
		// this submission waits on that exact timeline value at ALL_COMMANDS,
		// making writes to any Matrix input buffer available and visible before its
		// consumer dispatch. Buffer/image ownership transfers remain the recording
		// path's responsibility when queue families differ. Recorded commands retain
		// their resources until the signaled epoch is retired. The new signal uses the
		// next strictly increasing value.
		unsafe {
			self.logical.sync_commands.submit(
				&self.logical.handle,
				queue,
				&submit_infos,
				vk::Fence::default(),
			)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "queue submission", source))
	}

	pub(in crate::runtime) fn free(&self, mut command: RecordedCommandBuffer) {
		if let Some(pair) = command.take_brackets() {
			for bracket in *pair {
				self.free(bracket);
			}
		}
		// Drop reusable ownership before taking the pool lock. Its final owner frees
		// the shared handle through this same mutex.
		let Some((handle, pool, _bounded_sets)) = command.into_owned_handle() else {
			return;
		};
		self.free_command_buffer_handle(handle, pool);
	}

	pub(super) fn free_command_buffer_handle(
		&self,
		handle: vk::CommandBuffer,
		pool: CommandPoolKind,
	) {
		match pool {
			CommandPoolKind::Compute => {
				let mut command_pool = match self.logical.command_pool.lock() {
					Ok(command_pool) => command_pool,
					Err(poisoned) => poisoned.into_inner(),
				};
				command_pool.free_handle(&self.logical.handle, handle);
			}
			CommandPoolKind::Graphics => {
				let mut slot = match self.logical.graphics_command_pool.lock() {
					Ok(pool) => pool,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.logical.handle, handle);
				}
			}
			CommandPoolKind::Transfer => {
				let mut slot = match self.logical.transfer_command_pool.lock() {
					Ok(pool) => pool,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.logical.handle, handle);
				}
			}
			CommandPoolKind::VideoDecode => {
				let mut slot = match self.logical.video_decode_command_pool.lock() {
					Ok(slot) => slot,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.logical.handle, handle);
				}
			}
			CommandPoolKind::VideoEncode => {
				let mut slot = match self.logical.video_encode_command_pool.lock() {
					Ok(slot) => slot,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.logical.handle, handle);
				}
			}
		}
	}

	pub(in crate::runtime) fn is_complete(&self, epoch: u64) -> Result<bool> {
		self
			.logical
			.timeline
			.is_complete(&self.logical.handle, epoch)
	}

	pub(in crate::runtime) fn wait(&self, epoch: u64) -> Result<()> {
		self.logical.timeline.wait(&self.logical.handle, epoch)
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
			jit: Mutex::new(super::specialization::JitFusion::new()),
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
			match super::pipeline::validate_artifact_limits(
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
			// domains must not enter a driver compiler during Engine construction.
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
		drop(std::mem::replace(
			jit,
			super::specialization::JitFusion::new(),
		));
		for slot in self.pipelines.iter_mut().flatten() {
			// Release each pipeline's logical-device Arc before into_inner below.
			if let Some(mut pipeline) = slot.take() {
				pipeline.destroy(&self.handle);
			}
		}
		// The merged driver blob was already handed to the Engine persistence
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
