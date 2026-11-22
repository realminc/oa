//! One retained device domain and its queried information.

pub(in crate::runtime) mod features;
pub(in crate::runtime) mod physical;

mod configuration;
pub(super) mod info;
pub(super) mod logical;

/// Bitwise membership test for generated `vk::*FlagBits*` newtypes.
///
/// The generated bindings expose `BitAnd`/`BitOr` but no named membership
/// helper; this restores the conventional `flags.contains(bit)` spelling for
/// private native code.
pub(in crate::runtime) trait NativeFlagBits:
	Copy + PartialEq + std::ops::BitAnd<Output = Self>
{
	fn contains(self, member: Self) -> bool {
		(self & member) == member
	}
}

macro_rules! impl_native_flag_bits {
	($($ty:ty),* $(,)?) => {
		$(impl NativeFlagBits for $ty {})*
	};
}

impl_native_flag_bits! {
	vk::FormatFeatureFlagBits,
	vk::ImageCreateFlagBits,
	vk::ImageUsageFlagBits,
	vk::QueueFlagBits,
	vk::ShaderStageFlagBits,
	vk::CompositeAlphaFlagBitsKHR,
	vk::SampleCountFlagBits,
	vk::VideoCapabilityFlagBitsKHR,
	vk::VideoChromaSubsamplingFlagBitsKHR,
	vk::VideoCodingControlFlagBitsKHR,
	vk::VideoComponentBitDepthFlagBitsKHR,
	vk::VideoDecodeCapabilityFlagBitsKHR,
	vk::VideoDecodeH264PictureLayoutFlagBitsKHR,
	vk::VideoEncodeFeedbackFlagBitsKHR,
	vk::VideoEncodeH265CtbSizeFlagBitsKHR,
	vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR,
	vk::VideoEncodeRateControlModeFlagBitsKHR,
	vk::AccessFlagBits2,
	vk::PipelineStageFlagBits2,
	vk::BufferUsageFlagBits,
	vk::QueryResultFlagBits,
}

use std::sync::{Arc, Mutex, OnceLock};

use crate::{
	Error, Result,
	runtime::{
		Buffer, Instance,
		dna::{ir::elementwise::Program, kernel::KernelId, tuning::ObservedRoute},
		graph::executable::ExecutableGraph,
	},
};

use configuration::DeviceConfiguration;
use features::ExecutionProfile;
use info::log_compute_device;
use logical::{
	DeviceLogical, FrameDraw, MAX_RECYCLED_STORAGE_BUFFERS, MAX_RECYCLED_STORAGE_BYTES,
	PipelineCacheGuard, SceneDrawCmd, TileReduceOrder,
	buffer::{MeshBuffer, RecycledBuffer},
	command::{CommandPool, CommandPoolKind, RecordedCommandBuffer, ReusableCommandBuffer},
	commands::{RenderingCommands, SyncCommands},
	image::RenderTarget,
	pipeline::{self, ComputePipeline},
	pqc, secret_buffer, specialization,
	timestamp::{RouteTimings, TimestampPair},
	video, vkcache,
};
pub(in crate::runtime) use physical::DevicePhysical;

/// Engine-private execution domain, retained by resources through `Arc<Device>`.
/// Queried facts and executor resources each have one authoritative owner.
/// Owned executor fields never retain this Arc, preventing ownership cycles.
pub(crate) struct Device {
	// Drop executor resources before releasing their owning Vulkan instance.
	logical: DeviceLogical,
	physical: DevicePhysical,
	configuration: DeviceConfiguration,
	instance: super::instance::Instance,
}

impl Device {
	// ====== composition and queried facts ======

	pub(in crate::runtime) fn new(
		instance: &super::instance::Instance,
		physical: DevicePhysical,
	) -> crate::Result<Arc<Self>> {
		let configuration = DeviceConfiguration::new(instance, &physical);
		let logical = DeviceLogical::new(instance, &physical, &configuration)?;
		Ok(Arc::new(Self {
			logical,
			physical,
			configuration,
			instance: instance.clone(),
		}))
	}

	pub(in crate::runtime) fn logical(&self) -> &DeviceLogical {
		&self.logical
	}

	pub(in crate::runtime) fn compute_shared_memory_bytes(&self) -> u32 {
		self.physical.limits.max_compute_shared_memory_size
	}

	/// Queried per-dimension compute dispatch group limits.
	pub(in crate::runtime) const fn max_compute_work_group_count(&self) -> [u32; 3] {
		self.physical.limits.max_compute_work_group_count
	}

	/// Flatten linear work across two queried dispatch dimensions.
	pub(in crate::runtime) fn linear_dispatch_grid(
		&self,
		count: u32,
		group_size: u32,
	) -> crate::Result<[u32; 3]> {
		let limits = self.physical.limits;
		if group_size == 0
			|| group_size > limits.max_compute_work_group_size[0]
			|| group_size > limits.max_compute_work_group_invocations
		{
			return Err(crate::Error::invalid_argument(
				"linear dispatch group size exceeds device limits",
			));
		}
		let groups = count.div_ceil(group_size).max(1);
		let x = groups.min(limits.max_compute_work_group_count[0]);
		if x == 0 || limits.max_compute_work_group_count[1] == 0 {
			return Err(crate::Error::missing_capability(
				"device has no usable compute dispatch grid",
			));
		}
		let y = groups.div_ceil(x);
		if y > limits.max_compute_work_group_count[1] {
			return Err(crate::Error::invalid_argument(
				"linear work exceeds queried compute dispatch grid",
			));
		}
		Ok([x, y, 1])
	}

	/// Return whether the selected device can calibrate its timestamp domain
	/// against a POSIX monotonic host clock.
	pub(in crate::runtime) fn supports_clock_calibration(&self) -> bool {
		self.physical.limits.compute_timestamp_valid_bits != 0
			&& (self
				.configuration
				.extension_enabled(vk::KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
				|| self
					.configuration
					.extension_enabled(vk::EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME))
	}

	/// Sample paired device/host timestamps and retain the lowest-deviation
	/// pair, matching the donor `oa::Engine::calibrateClock`.
	pub(in crate::runtime) fn calibrate_clock(
		&self,
		sample_count: u32,
	) -> crate::Result<crate::ClockCalibration> {
		if !(1..=1024).contains(&sample_count) {
			return Err(crate::Error::invalid_argument(
				"clock calibration sample count must be in [1, 1024]",
			));
		}
		if !self.supports_clock_calibration() {
			return Err(crate::Error::unavailable(
				"selected device does not expose calibrated timestamps",
			));
		}
		let use_khr = self
			.configuration
			.extension_enabled(vk::KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		let instance = self.instance.raw();
		let query_domains = |count: *mut u32, domains: *mut vk::TimeDomainKHR| unsafe {
			if use_khr {
				instance.get_physical_device_calibrateable_time_domains_khr(
					self.physical.handle,
					count,
					domains,
				)
			} else {
				instance.get_physical_device_calibrateable_time_domains_ext(
					self.physical.handle,
					count,
					domains,
				)
			}
		};
		let mut domain_count = 0_u32;
		let count_result = query_domains(&mut domain_count, std::ptr::null_mut());
		match count_result {
			Ok(vk::Result::SUCCESS) if domain_count > 0 => {}
			Err(vk::CallError::MissingCommand(_)) => {
				return Err(crate::Error::unavailable(
					"calibrated timestamp entry points are unavailable",
				));
			}
			_ => {
				return Err(crate::Error::backend_dispatch_failure(
					"Vulkan",
					"calibrated timestamp domain query",
					count_result,
				));
			}
		}
		let mut domains = vec![vk::TimeDomainKHR::DEVICE_KHR; domain_count as usize];
		let list_result = query_domains(&mut domain_count, domains.as_mut_ptr());
		if let Err(vk::CallError::MissingCommand(_)) = list_result {
			return Err(crate::Error::unavailable(
				"calibrated timestamp entry points are unavailable",
			));
		}
		if list_result != Ok(vk::Result::SUCCESS) {
			return Err(crate::Error::backend_dispatch_failure(
				"Vulkan",
				"calibrated timestamp domain enumeration",
				list_result,
			));
		}
		let has_device = domains.contains(&vk::TimeDomainKHR::DEVICE_KHR);
		let has_monotonic = domains.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC_KHR);
		let has_monotonic_raw = domains.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW_KHR);
		if !has_device || !(has_monotonic || has_monotonic_raw) {
			return Err(crate::Error::unavailable(
				"device and POSIX monotonic clock domains cannot be calibrated together",
			));
		}

		let host_domain = if has_monotonic_raw {
			vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW_KHR
		} else {
			vk::TimeDomainKHR::CLOCK_MONOTONIC_KHR
		};
		let infos = [
			vk::CalibratedTimestampInfoKHR::new().set_time_domain(vk::TimeDomainKHR::DEVICE_KHR),
			vk::CalibratedTimestampInfoKHR::new().set_time_domain(host_domain),
		];
		let device = self.raw();
		let mut best = crate::ClockCalibration::default();
		let mut have_best = false;
		for _ in 0..sample_count {
			let mut timestamps = [0_u64; 2];
			let mut maximum_deviation = 0_u64;
			// SAFETY: infos describes two domains; timestamps has room for two
			// results and maximum_deviation is a live u64. The extension is
			// enabled on this device.
			let sample_result = unsafe {
				if use_khr {
					device.get_calibrated_timestamps_khr(
						2,
						infos.as_ptr(),
						timestamps.as_mut_ptr(),
						&mut maximum_deviation,
					)
				} else {
					device.get_calibrated_timestamps_ext(
						2,
						infos.as_ptr(),
						timestamps.as_mut_ptr(),
						&mut maximum_deviation,
					)
				}
			};
			if let Err(vk::CallError::MissingCommand(_)) = sample_result {
				return Err(crate::Error::unavailable(
					"calibrated timestamp entry points are unavailable",
				));
			}
			if sample_result != Ok(vk::Result::SUCCESS) {
				return Err(crate::Error::backend_dispatch_failure(
					"Vulkan",
					"calibrated timestamp sample",
					sample_result,
				));
			}
			if have_best && maximum_deviation >= best.maximum_deviation_nanoseconds {
				continue;
			}
			let valid_bits = self.physical.limits.compute_timestamp_valid_bits;
			if valid_bits < 64 {
				timestamps[0] &= (1_u64 << valid_bits) - 1;
			}
			best = crate::ClockCalibration {
				device_timestamp_ticks: timestamps[0],
				host_timestamp_nanoseconds: timestamps[1],
				maximum_deviation_nanoseconds: maximum_deviation,
				device_nanoseconds_per_tick: self.physical.limits.timestamp_period_ns,
				device_timestamp_valid_bits: valid_bits,
				host_clock_domain: if has_monotonic_raw {
					crate::HostClockDomain::MonotonicRaw
				} else {
					crate::HostClockDomain::Monotonic
				},
			};
			have_best = true;
		}
		Ok(best)
	}

	// ====== logical execution ======

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
	pub(in crate::runtime) fn create_graphics_pipeline(&self) -> Result<pipeline::GraphicsPipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::GraphicsPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the flat-color material pipeline for this device.
	pub(in crate::runtime) fn create_flat_color_pipeline(
		&self,
	) -> Result<pipeline::FlatColorPipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::FlatColorPipeline::new(
			std::sync::Arc::clone(&self.logical.handle),
			self.physical.limits,
			cache.raw(),
		)
	}

	/// Create the procedural canvas-grid pipeline for this device.
	pub(in crate::runtime) fn create_canvas_grid_pipeline(
		&self,
	) -> Result<pipeline::CanvasGridPipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::CanvasGridPipeline::new(
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
	) -> Result<pipeline::ComputePipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::ComputePipeline::new(
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
	) -> Result<pipeline::UnlitDescriptorLayouts> {
		pipeline::UnlitDescriptorLayouts::new(std::sync::Arc::clone(&self.logical.handle))
	}

	pub(in crate::runtime) fn create_unlit_descriptor_arena(
		&self,
		layout: vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<pipeline::UnlitDescriptorArena> {
		pipeline::UnlitDescriptorArena::new(
			std::sync::Arc::clone(&self.logical.handle),
			layout,
			set_count,
		)
	}

	/// Create the owned descriptor set layouts for the Surface pipeline.
	pub(in crate::runtime) fn create_surface_descriptor_layouts(
		&self,
	) -> Result<pipeline::SurfaceDescriptorLayouts> {
		pipeline::SurfaceDescriptorLayouts::new(std::sync::Arc::clone(&self.logical.handle))
	}

	pub(in crate::runtime) fn create_surface_descriptor_arena(
		&self,
		scene_layout: vk::DescriptorSetLayout,
		material_layout: vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<pipeline::SurfaceDescriptorArena> {
		pipeline::SurfaceDescriptorArena::new(
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
	) -> Result<pipeline::UnlitPipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::UnlitPipeline::new(
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
	) -> Result<pipeline::SurfacePipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::SurfacePipeline::new(
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
	) -> Result<pipeline::SurfaceBlendPipeline> {
		let cache = self.pipeline_cache()?;
		pipeline::SurfaceBlendPipeline::new(
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
		video::query_decode_capabilities(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_decode_formats(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeFormats> {
		video::query_decode_formats(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_encode_capabilities(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeCapabilities> {
		video::query_encode_capabilities(&self.instance, &self.physical, profile)
	}

	pub(in crate::runtime) fn video_encode_formats(
		&self,
		profile: crate::video::VideoEncodeProfile,
	) -> Result<crate::video::VideoEncodeFormats> {
		video::query_encode_formats(&self.instance, &self.physical, profile)
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
		specialization::block_signature(program)
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
	pub(in crate::runtime) fn jit_section_updates(&self) -> Vec<vkcache::SectionUpdate> {
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
	) -> Result<video::DecodeSession> {
		video::create_decode_session(
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
	) -> Result<video::EncodeSession> {
		video::create_encode_session(self, settings, plan)
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
		seed: secret_buffer::SecretBinding<'a>,
		private: secret_buffer::SecretBinding<'a>,
		public: &Buffer,
		k: u32,
	) -> Result<pqc::PreparedSecretKeygen<'a>> {
		let operands = pqc::SecretKeygenOperands::mlkem(self, seed, private, public, k)?;
		self.prepare_secret_keygen(operands)
	}

	pub(in crate::runtime) fn prepare_mldsa_keygen<'a>(
		self: &'a Arc<Self>,
		seed: secret_buffer::SecretBinding<'a>,
		private: secret_buffer::SecretBinding<'a>,
		public: &Buffer,
		parameter: u32,
	) -> Result<pqc::PreparedSecretKeygen<'a>> {
		let operands = pqc::SecretKeygenOperands::mldsa(self, seed, private, public, parameter)?;
		self.prepare_secret_keygen(operands)
	}

	fn prepare_secret_keygen<'a>(
		self: &'a Arc<Self>,
		operands: pqc::SecretKeygenOperands<'a>,
	) -> Result<pqc::PreparedSecretKeygen<'a>> {
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
		operands: pqc::SecretConsumerOperands<'a>,
	) -> Result<pqc::PreparedSecretConsumer<'a>> {
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
							let scope = vkcache::VkCacheScope::new(&self.physical);
							let scope_path = scope.path();
							crate::log_warn!(
								crate::LogComponent::COMPUTE,
								"pipeline cache: creation failed for {}; dropping {}",
								kernel.report_name(),
								scope_path.display()
							);
							creation.cache = None;
							let _ = vkcache::update(
								&scope_path,
								&scope,
								&[vkcache::SectionUpdate::remove(
									vkcache::SECTION_DRIVER_PIPELINE_CACHE,
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

	/// Queue submission through the enabled synchronization spelling —
	/// `vkQueueSubmit2` on Core, `vkQueueSubmit2KHR` where the Compatibility
	/// profile only exposes the KHR entry point.
	///
	/// # Safety
	/// The queue, command buffers, semaphores and fence must be valid for this
	/// device and obey the Vulkan submission synchronization rules.
	pub(in crate::runtime) unsafe fn queue_submit2(
		&self,
		queue: vk::Queue,
		submits: &[vk::SubmitInfo2<'_>],
		fence: vk::Fence,
	) -> Result<()> {
		unsafe {
			self
				.logical
				.sync_commands
				.submit(&self.logical.handle, queue, submits, fence)
				.map_err(Error::from)
		}
	}

	/// Timestamp write through the enabled synchronization spelling —
	/// `vkCmdWriteTimestamp2` on Core, `vkCmdWriteTimestamp2KHR` on
	/// Compatibility.
	///
	/// # Safety
	/// The query pool and index must be valid and the command buffer recording.
	pub(in crate::runtime) unsafe fn write_timestamp(
		&self,
		command: vk::CommandBuffer,
		stage: vk::PipelineStageFlags2,
		pool: vk::QueryPool,
		query: u32,
	) -> Result<()> {
		unsafe {
			self
				.logical
				.sync_commands
				.write_timestamp(&self.logical.handle, command, stage, pool, query)
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
				unsafe { self.pipeline_barrier(command, &(dependency).map_err(Error::from)?) }?;
				let clear = vk::ClearColorValue { float32: color };
				unsafe {
					self.cmd_clear_color_image(
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
				unsafe { self.pipeline_barrier(command, &(dependency).map_err(Error::from)?) }?;
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
					self.cmd_blit_image(
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
					self.cmd_set_viewport(cmd, 0, &[viewport])?;
					self.cmd_set_scissor(cmd, 0, &[scissor])?;
				};
				// 4. bind pipeline and push constants
				unsafe {
					self.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline)?;
					self.cmd_push_constants(cmd, pipeline_layout, push_stages, 0, push_data)?;
					if !descriptor_sets.is_empty() {
						self.cmd_bind_descriptor_sets(
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
							self.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0])?;
							self.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32)?;
							self.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0)?;
						}
						FrameDraw::Vertexless { vertex_count } => {
							self.cmd_draw(cmd, vertex_count, 1, 0, 0)?;
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
					self.cmd_copy_image_to_buffer(
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
					self.cmd_set_viewport(cmd, 0, &[viewport])?;
					self.cmd_set_scissor(cmd, 0, &[scissor])?;
				};
				// 4. bind vertex/index buffers and scene set (set 0) once
				unsafe {
					self.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0])?;
					self.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32)?;
					self.cmd_bind_descriptor_sets(
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
					self.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline)?;
				}
				for draw in draws.iter().filter(|d| !d.is_blend) {
					unsafe {
						self.cmd_push_constants(cmd, pipeline_layout, push_stages, 0, &draw.push_data)?;
						self.cmd_bind_descriptor_sets(
							cmd,
							vk::PipelineBindPoint::GRAPHICS,
							pipeline_layout,
							1,
							&[draw.material_set],
							&[],
						)?;
						self.cmd_draw_indexed(cmd, draw.index_count, 1, draw.index_start, 0, 0)?;
					}
				}

				// 5b. transparent pass — switch to blend pipeline, emit blend draws
				let has_blend = draws.iter().any(|d| d.is_blend);
				if has_blend {
					unsafe {
						self.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, blend_pipeline)?;
					}
					for draw in draws.iter().filter(|d| d.is_blend) {
						unsafe {
							self.cmd_push_constants(cmd, pipeline_layout, push_stages, 0, &draw.push_data)?;
							self.cmd_bind_descriptor_sets(
								cmd,
								vk::PipelineBindPoint::GRAPHICS,
								pipeline_layout,
								1,
								&[draw.material_set],
								&[],
							)?;
							self.cmd_draw_indexed(cmd, draw.index_count, 1, draw.index_start, 0, 0)?;
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
					self.cmd_copy_image_to_buffer(
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
		unsafe { self.queue_submit2(queue, &submit_infos, vk::Fence::default()) }
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

	// ====== vk native commands ======

	/// `vkCmdDispatch` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_dispatch`.
	#[inline]
	pub(crate) unsafe fn cmd_dispatch(
		&self,
		command_buffer: vk::CommandBuffer,
		group_count_x: u32,
		group_count_y: u32,
		group_count_z: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z)
		}
	}

	/// `vkCmdDispatchIndirect` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_dispatch_indirect`.
	#[inline]
	pub(crate) unsafe fn cmd_dispatch_indirect(
		&self,
		command_buffer: vk::CommandBuffer,
		buffer: vk::Buffer,
		offset: vk::DeviceSize,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_dispatch_indirect(command_buffer, buffer, offset)
		}
	}

	/// `vkCmdBindPipeline` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_bind_pipeline`.
	#[inline]
	pub(crate) unsafe fn cmd_bind_pipeline(
		&self,
		command_buffer: vk::CommandBuffer,
		pipeline_bind_point: vk::PipelineBindPoint,
		pipeline: vk::Pipeline,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_bind_pipeline(command_buffer, pipeline_bind_point, pipeline)
		}
	}

	/// `vkCmdBindDescriptorSets` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_bind_descriptor_sets`.
	#[inline]
	pub(crate) unsafe fn cmd_bind_descriptor_sets(
		&self,
		command_buffer: vk::CommandBuffer,
		pipeline_bind_point: vk::PipelineBindPoint,
		layout: vk::PipelineLayout,
		first_set: u32,
		descriptor_sets: &[vk::DescriptorSet],
		dynamic_offsets: &[u32],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_bind_descriptor_sets(
				command_buffer,
				pipeline_bind_point,
				layout,
				first_set,
				descriptor_sets,
				dynamic_offsets,
			)
		}
	}

	/// `vkCmdPushConstants` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_push_constants`.
	#[inline]
	pub(crate) unsafe fn cmd_push_constants(
		&self,
		command_buffer: vk::CommandBuffer,
		layout: vk::PipelineLayout,
		stage_flags: vk::ShaderStageFlags,
		offset: u32,
		values: &[u8],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_push_constants(command_buffer, layout, stage_flags, offset, values)
		}
	}

	/// `vkCmdCopyBuffer` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_copy_buffer`.
	#[inline]
	pub(crate) unsafe fn cmd_copy_buffer(
		&self,
		command_buffer: vk::CommandBuffer,
		src_buffer: vk::Buffer,
		dst_buffer: vk::Buffer,
		regions: &[vk::BufferCopy],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, regions)
		}
	}

	/// `vkCmdCopyBufferToImage` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_copy_buffer_to_image`.
	#[inline]
	pub(crate) unsafe fn cmd_copy_buffer_to_image(
		&self,
		command_buffer: vk::CommandBuffer,
		src_buffer: vk::Buffer,
		dst_image: vk::Image,
		dst_image_layout: vk::ImageLayout,
		regions: &[vk::BufferImageCopy],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_copy_buffer_to_image(
				command_buffer,
				src_buffer,
				dst_image,
				dst_image_layout,
				regions,
			)
		}
	}

	/// `vkCmdCopyImageToBuffer` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_copy_image_to_buffer`.
	#[inline]
	pub(crate) unsafe fn cmd_copy_image_to_buffer(
		&self,
		command_buffer: vk::CommandBuffer,
		src_image: vk::Image,
		src_image_layout: vk::ImageLayout,
		dst_buffer: vk::Buffer,
		regions: &[vk::BufferImageCopy],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_copy_image_to_buffer(
				command_buffer,
				src_image,
				src_image_layout,
				dst_buffer,
				regions,
			)
		}
	}

	/// `vkCmdBlitImage` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_blit_image`.
	#[inline]
	#[allow(clippy::too_many_arguments)]
	pub(crate) unsafe fn cmd_blit_image(
		&self,
		command_buffer: vk::CommandBuffer,
		src_image: vk::Image,
		src_image_layout: vk::ImageLayout,
		dst_image: vk::Image,
		dst_image_layout: vk::ImageLayout,
		regions: &[vk::ImageBlit],
		filter: vk::Filter,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_blit_image(
				command_buffer,
				src_image,
				src_image_layout,
				dst_image,
				dst_image_layout,
				regions,
				filter,
			)
		}
	}

	/// `vkCmdFillBuffer` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_fill_buffer`.
	#[inline]
	pub(crate) unsafe fn cmd_fill_buffer(
		&self,
		command_buffer: vk::CommandBuffer,
		buffer: vk::Buffer,
		offset: vk::DeviceSize,
		size: vk::DeviceSize,
		data: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_fill_buffer(command_buffer, buffer, offset, size, data)
		}
	}

	/// `vkCmdClearColorImage` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_clear_color_image`.
	#[inline]
	pub(crate) unsafe fn cmd_clear_color_image(
		&self,
		command_buffer: vk::CommandBuffer,
		image: vk::Image,
		image_layout: vk::ImageLayout,
		color: &vk::ClearColorValue,
		ranges: &[vk::ImageSubresourceRange],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_clear_color_image(command_buffer, image, image_layout, color, ranges)
		}
	}

	/// `vkCmdPipelineBarrier2` on this device's native dispatch table.
	///
	/// Profile-agnostic callers should prefer [`Device::pipeline_barrier`],
	/// which resolves the Core/KHR spelling from the enabled command set.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_pipeline_barrier2`.
	#[inline]
	pub(crate) unsafe fn cmd_pipeline_barrier2(
		&self,
		command_buffer: vk::CommandBuffer,
		dependency_info: &vk::DependencyInfo<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_pipeline_barrier2(command_buffer, dependency_info)
		}
	}

	/// `vkCmdResetQueryPool` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_reset_query_pool`.
	#[inline]
	pub(crate) unsafe fn cmd_reset_query_pool(
		&self,
		command_buffer: vk::CommandBuffer,
		query_pool: vk::QueryPool,
		first_query: u32,
		query_count: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_reset_query_pool(command_buffer, query_pool, first_query, query_count)
		}
	}

	/// `vkCmdBeginQuery` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_begin_query`.
	#[inline]
	pub(crate) unsafe fn cmd_begin_query(
		&self,
		command_buffer: vk::CommandBuffer,
		query_pool: vk::QueryPool,
		query: u32,
		flags: vk::QueryControlFlags,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_begin_query(command_buffer, query_pool, query, flags)
		}
	}

	/// `vkCmdEndQuery` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_end_query`.
	#[inline]
	pub(crate) unsafe fn cmd_end_query(
		&self,
		command_buffer: vk::CommandBuffer,
		query_pool: vk::QueryPool,
		query: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().cmd_end_query(command_buffer, query_pool, query) }
	}

	/// `vkCmdSetViewport` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_set_viewport`.
	#[inline]
	pub(crate) unsafe fn cmd_set_viewport(
		&self,
		command_buffer: vk::CommandBuffer,
		first_viewport: u32,
		viewports: &[vk::Viewport],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_set_viewport(command_buffer, first_viewport, viewports)
		}
	}

	/// `vkCmdSetScissor` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_set_scissor`.
	#[inline]
	pub(crate) unsafe fn cmd_set_scissor(
		&self,
		command_buffer: vk::CommandBuffer,
		first_scissor: u32,
		scissors: &[vk::Rect2D],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_set_scissor(command_buffer, first_scissor, scissors)
		}
	}

	/// `vkCmdBindVertexBuffers` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_bind_vertex_buffers`.
	#[inline]
	pub(crate) unsafe fn cmd_bind_vertex_buffers(
		&self,
		command_buffer: vk::CommandBuffer,
		first_binding: u32,
		buffers: &[vk::Buffer],
		offsets: &[vk::DeviceSize],
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_bind_vertex_buffers(command_buffer, first_binding, buffers, offsets)
		}
	}

	/// `vkCmdBindIndexBuffer` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_bind_index_buffer`.
	#[inline]
	pub(crate) unsafe fn cmd_bind_index_buffer(
		&self,
		command_buffer: vk::CommandBuffer,
		buffer: vk::Buffer,
		offset: vk::DeviceSize,
		index_type: vk::IndexType,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_bind_index_buffer(command_buffer, buffer, offset, index_type)
		}
	}

	/// `vkCmdDraw` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_draw`.
	#[inline]
	pub(crate) unsafe fn cmd_draw(
		&self,
		command_buffer: vk::CommandBuffer,
		vertex_count: u32,
		instance_count: u32,
		first_vertex: u32,
		first_instance: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_draw(
				command_buffer,
				vertex_count,
				instance_count,
				first_vertex,
				first_instance,
			)
		}
	}

	/// `vkCmdDrawIndexed` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_draw_indexed`.
	#[inline]
	pub(crate) unsafe fn cmd_draw_indexed(
		&self,
		command_buffer: vk::CommandBuffer,
		index_count: u32,
		instance_count: u32,
		first_index: u32,
		vertex_offset: i32,
		first_instance: u32,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self.raw().cmd_draw_indexed(
				command_buffer,
				index_count,
				instance_count,
				first_index,
				vertex_offset,
				first_instance,
			)
		}
	}

	/// `vkCmdBeginVideoCodingKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_begin_video_coding_khr`.
	#[inline]
	pub(crate) unsafe fn cmd_begin_video_coding_khr(
		&self,
		command_buffer: vk::CommandBuffer,
		p_begin_info: *const vk::VideoBeginCodingInfoKHR<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_begin_video_coding_khr(command_buffer, p_begin_info)
		}
	}

	/// `vkCmdDecodeVideoKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_decode_video_khr`.
	#[inline]
	pub(crate) unsafe fn cmd_decode_video_khr(
		&self,
		command_buffer: vk::CommandBuffer,
		p_decode_info: *const vk::VideoDecodeInfoKHR<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_decode_video_khr(command_buffer, p_decode_info)
		}
	}

	/// `vkCmdEncodeVideoKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_encode_video_khr`.
	#[inline]
	pub(crate) unsafe fn cmd_encode_video_khr(
		&self,
		command_buffer: vk::CommandBuffer,
		p_encode_info: *const vk::VideoEncodeInfoKHR<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_encode_video_khr(command_buffer, p_encode_info)
		}
	}

	/// `vkCmdControlVideoCodingKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_control_video_coding_khr`.
	#[inline]
	pub(crate) unsafe fn cmd_control_video_coding_khr(
		&self,
		command_buffer: vk::CommandBuffer,
		p_coding_control_info: *const vk::VideoCodingControlInfoKHR<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_control_video_coding_khr(command_buffer, p_coding_control_info)
		}
	}

	/// `vkCmdEndVideoCodingKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::cmd_end_video_coding_khr`.
	#[inline]
	pub(crate) unsafe fn cmd_end_video_coding_khr(
		&self,
		command_buffer: vk::CommandBuffer,
		p_end_coding_info: *const vk::VideoEndCodingInfoKHR<'_>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.cmd_end_video_coding_khr(command_buffer, p_end_coding_info)
		}
	}

	/// `vkQueuePresentKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::queue_present_khr`.
	#[inline]
	pub(crate) unsafe fn queue_present_khr(
		&self,
		queue: vk::Queue,
		p_present_info: *const vk::PresentInfoKHR<'_>,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe { self.raw().queue_present_khr(queue, p_present_info) }
	}

	/// `vkAcquireNextImageKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::acquire_next_image_khr`.
	#[inline]
	pub(crate) unsafe fn acquire_next_image_khr(
		&self,
		swapchain: vk::SwapchainKHR,
		timeout: u64,
		semaphore: vk::Semaphore,
		fence: vk::Fence,
		p_image_index: *mut u32,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self
				.raw()
				.acquire_next_image_khr(swapchain, timeout, semaphore, fence, p_image_index)
		}
	}

	/// `vkWaitForFences` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::wait_for_fences`.
	#[inline]
	pub(crate) unsafe fn wait_for_fences(
		&self,
		fences: &[vk::Fence],
		wait_all: bool,
		timeout_ns: u64,
	) -> std::result::Result<bool, vk::CallError> {
		unsafe { self.raw().wait_for_fences(fences, wait_all, timeout_ns) }
	}

	/// `vkResetFences` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::reset_fences`.
	#[inline]
	pub(crate) unsafe fn reset_fences(
		&self,
		fences: &[vk::Fence],
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().reset_fences(fences) }
	}

	/// `vkDeviceWaitIdle` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::device_wait_idle`.
	#[inline]
	pub(crate) unsafe fn device_wait_idle(&self) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().device_wait_idle() }
	}

	/// `vkGetQueryPoolResults` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::get_query_pool_results`.
	#[inline]
	pub(crate) unsafe fn get_query_pool_results(
		&self,
		query_pool: vk::QueryPool,
		first_query: u32,
		query_count: u32,
		data: &mut [u8],
		stride: vk::DeviceSize,
		flags: vk::QueryResultFlags,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self
				.raw()
				.get_query_pool_results(query_pool, first_query, query_count, data, stride, flags)
		}
	}

	/// `vkGetVideoSessionMemoryRequirementsKHR` on this device's native
	/// dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to
	/// `vk::Device::get_video_session_memory_requirements_khr`.
	#[inline]
	pub(crate) unsafe fn get_video_session_memory_requirements_khr(
		&self,
		video_session: vk::VideoSessionKHR,
		p_memory_requirements_count: *mut u32,
		p_memory_requirements: *mut vk::VideoSessionMemoryRequirementsKHR<'_>,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self.raw().get_video_session_memory_requirements_khr(
				video_session,
				p_memory_requirements_count,
				p_memory_requirements,
			)
		}
	}

	/// `vkGetEncodedVideoSessionParametersKHR` on this device's native
	/// dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to
	/// `vk::Device::get_encoded_video_session_parameters_khr`.
	#[inline]
	pub(crate) unsafe fn get_encoded_video_session_parameters_khr(
		&self,
		p_video_session_parameters_info: *const vk::VideoEncodeSessionParametersGetInfoKHR<'_>,
		p_feedback_info: *mut vk::VideoEncodeSessionParametersFeedbackInfoKHR<'_>,
		p_data_size: *mut usize,
		p_data: *mut std::ffi::c_void,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self.raw().get_encoded_video_session_parameters_khr(
				p_video_session_parameters_info,
				p_feedback_info,
				p_data_size,
				p_data,
			)
		}
	}

	/// `vkCreateImageView` on this device's native dispatch table. Returned
	/// handles are owned and destroyed by the caller's existing OA wrapper.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_image_view`.
	#[inline]
	pub(crate) unsafe fn create_image_view(
		&self,
		create_info: &vk::ImageViewCreateInfo<'_>,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<vk::ImageView, vk::CallError> {
		unsafe { self.raw().create_image_view(create_info, allocator) }
	}

	/// `vkDestroyImageView` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_image_view`;
	/// the caller owns the destroyed handle's lifetime.
	#[inline]
	pub(crate) unsafe fn destroy_image_view(
		&self,
		image_view: vk::ImageView,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().destroy_image_view(image_view, allocator) }
	}

	/// `vkCreateSemaphore` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_semaphore`.
	#[inline]
	pub(crate) unsafe fn create_semaphore(
		&self,
		create_info: &vk::SemaphoreCreateInfo<'_>,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<vk::Semaphore, vk::CallError> {
		unsafe { self.raw().create_semaphore(create_info, allocator) }
	}

	/// `vkDestroySemaphore` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_semaphore`.
	#[inline]
	pub(crate) unsafe fn destroy_semaphore(
		&self,
		semaphore: vk::Semaphore,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().destroy_semaphore(semaphore, allocator) }
	}

	/// `vkCreateFence` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_fence`.
	#[inline]
	pub(crate) unsafe fn create_fence(
		&self,
		create_info: &vk::FenceCreateInfo,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<vk::Fence, vk::CallError> {
		unsafe { self.raw().create_fence(create_info, allocator) }
	}

	/// `vkDestroyFence` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_fence`.
	#[inline]
	pub(crate) unsafe fn destroy_fence(
		&self,
		fence: vk::Fence,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().destroy_fence(fence, allocator) }
	}

	/// `vkCreateQueryPool` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_query_pool`.
	#[inline]
	pub(crate) unsafe fn create_query_pool(
		&self,
		create_info: &vk::QueryPoolCreateInfo<'_>,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<vk::QueryPool, vk::CallError> {
		unsafe { self.raw().create_query_pool(create_info, allocator) }
	}

	/// `vkDestroyQueryPool` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_query_pool`.
	#[inline]
	pub(crate) unsafe fn destroy_query_pool(
		&self,
		query_pool: vk::QueryPool,
		allocator: Option<&vk::AllocationCallbacks>,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().destroy_query_pool(query_pool, allocator) }
	}

	/// `vkCreateSwapchainKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_swapchain_khr`.
	#[inline]
	pub(crate) unsafe fn create_swapchain_khr(
		&self,
		p_create_info: *const vk::SwapchainCreateInfoKHR<'_>,
		p_allocator: *const vk::AllocationCallbacks,
		p_swapchain: *mut vk::SwapchainKHR,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self
				.raw()
				.create_swapchain_khr(p_create_info, p_allocator, p_swapchain)
		}
	}

	/// `vkDestroySwapchainKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_swapchain_khr`.
	#[inline]
	pub(crate) unsafe fn destroy_swapchain_khr(
		&self,
		swapchain: vk::SwapchainKHR,
		p_allocator: *const vk::AllocationCallbacks,
	) -> std::result::Result<(), vk::CallError> {
		unsafe { self.raw().destroy_swapchain_khr(swapchain, p_allocator) }
	}

	/// `vkCreateVideoSessionKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::create_video_session_khr`.
	#[inline]
	pub(crate) unsafe fn create_video_session_khr(
		&self,
		p_create_info: *const vk::VideoSessionCreateInfoKHR<'_>,
		p_allocator: *const vk::AllocationCallbacks,
		p_video_session: *mut vk::VideoSessionKHR,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self
				.raw()
				.create_video_session_khr(p_create_info, p_allocator, p_video_session)
		}
	}

	/// `vkDestroyVideoSessionKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to `vk::Device::destroy_video_session_khr`.
	#[inline]
	pub(crate) unsafe fn destroy_video_session_khr(
		&self,
		video_session: vk::VideoSessionKHR,
		p_allocator: *const vk::AllocationCallbacks,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.destroy_video_session_khr(video_session, p_allocator)
		}
	}

	/// `vkCreateVideoSessionParametersKHR` on this device's native dispatch
	/// table.
	///
	/// # Safety
	/// Identical valid-usage obligations to
	/// `vk::Device::create_video_session_parameters_khr`.
	#[inline]
	pub(crate) unsafe fn create_video_session_parameters_khr(
		&self,
		p_create_info: *const vk::VideoSessionParametersCreateInfoKHR<'_>,
		p_allocator: *const vk::AllocationCallbacks,
		p_video_session_parameters: *mut vk::VideoSessionParametersKHR,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self.raw().create_video_session_parameters_khr(
				p_create_info,
				p_allocator,
				p_video_session_parameters,
			)
		}
	}

	/// `vkDestroyVideoSessionParametersKHR` on this device's native dispatch
	/// table.
	///
	/// # Safety
	/// Identical valid-usage obligations to
	/// `vk::Device::destroy_video_session_parameters_khr`.
	#[inline]
	pub(crate) unsafe fn destroy_video_session_parameters_khr(
		&self,
		video_session_parameters: vk::VideoSessionParametersKHR,
		p_allocator: *const vk::AllocationCallbacks,
	) -> std::result::Result<(), vk::CallError> {
		unsafe {
			self
				.raw()
				.destroy_video_session_parameters_khr(video_session_parameters, p_allocator)
		}
	}

	/// `vkBindVideoSessionMemoryKHR` on this device's native dispatch table.
	///
	/// # Safety
	/// Identical valid-usage obligations to
	/// `vk::Device::bind_video_session_memory_khr`.
	#[inline]
	pub(crate) unsafe fn bind_video_session_memory_khr(
		&self,
		video_session: vk::VideoSessionKHR,
		bind_session_memory_info_count: u32,
		p_bind_session_memory_infos: *const vk::BindVideoSessionMemoryInfoKHR<'_>,
	) -> std::result::Result<vk::Result, vk::CallError> {
		unsafe {
			self.raw().bind_video_session_memory_khr(
				video_session,
				bind_session_memory_info_count,
				p_bind_session_memory_infos,
			)
		}
	}
}

pub use info::{DeviceKind, DeviceSummary};
