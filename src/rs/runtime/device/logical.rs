use std::{
	mem::ManuallyDrop,
	sync::{Arc, Mutex, OnceLock},
};

use crate::{
	Error, Result,
	runtime::{executable_graph::ExecutableGraph, shader::KernelId},
};

use super::{
	Device,
	buffer::{MeshBuffer, RecycledBuffer},
	command::{CommandPool, CommandPoolKind, RecordedCommandBuffer, ReusableCommandBuffer},
	commands::{RenderingCommands, SyncCommands},
	configuration::DeviceConfiguration,
	descriptor::DescriptorHeap,
	features::ExecutionProfile,
	image::RenderTarget,
	info::log_compute_device,
	physical::DevicePhysical,
	pipeline::ComputePipeline,
	timeline::Timeline,
	timestamp::TimestampPair,
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
	/// Raw push constant bytes (must be `STANDARD_SURFACE_PUSH_SIZE` long).
	pub push_data: Vec<u8>,
	/// Resolved material descriptor set (set 1).
	pub material_set: ash::vk::DescriptorSet,
	/// When `true` this draw belongs to the transparent/blend pass.
	pub is_blend: bool,
}

/// Vulkan executor resources, owned once by Device; never independently retained.
pub(in crate::runtime) struct DeviceLogical {
	handle: ash::Device,
	sync_commands: SyncCommands,
	rendering_commands: RenderingCommands,
	allocator: ManuallyDrop<vk_mem::Allocator>,
	command_pool: Mutex<CommandPool>,
	graphics_command_pool: Mutex<Option<CommandPool>>,
	video_decode_command_pool: Mutex<Option<CommandPool>>,
	timeline: Timeline,
	descriptors: DescriptorHeap,
	/// Shared pipeline layout for UI compute pipelines (128-byte push range).
	ui_pipeline_layout: ash::vk::PipelineLayout,
	storage_pool: Mutex<StoragePool>,
	// None means unsupported; an empty OnceLock means admitted but not compiled.
	pipelines: Vec<Option<OnceLock<ComputePipeline>>>,
	pipeline_creation: Mutex<()>,
	compute_queue: ash::vk::Queue,
	graphics_queue: Option<ash::vk::Queue>,
	video_decode_queue: Option<ash::vk::Queue>,
	video_encode_queue: Option<ash::vk::Queue>,
	timestamp_period_ns: f64,
	compute_timestamp_valid_bits: u32,
}

#[derive(Default)]
struct StoragePool {
	free: Vec<RecycledBuffer>,
	bytes: usize,
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

	pub(super) fn configuration(&self) -> &DeviceConfiguration {
		&self.configuration
	}

	pub(super) fn descriptor_capacities(&self) -> (u32, u32) {
		self.logical.descriptors.capacities()
	}

	pub(in crate::runtime) fn execution_profile_name(&self) -> &'static str {
		match self.physical.profile {
			ExecutionProfile::Strict => "Strict (Vulkan 1.3+)",
			ExecutionProfile::Compatibility => "Compatibility (Vulkan 1.2)",
		}
	}

	pub(in crate::runtime) fn log_identity(&self) {
		// Currently always 1 compute device; the outer block is structured to
		// accept multiple devices (`,`-separated) when multi-device is added.
		crate::log_info!(
			crate::LogComponent::ENGINE,
			"oa::Engine · v{} · Vulkan · 1 compute device {{",
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
			encoder_sessions_available: false,
		})
	}

	pub(in crate::runtime) fn has_graphics_queue(&self) -> bool {
		self.physical.profile == ExecutionProfile::Strict
			&& self.logical.graphics_queue.is_some()
			&& self.configuration.features().dynamic_rendering
	}

	pub(in crate::runtime) fn require_ui_bindings(&self) -> Result<()> {
		if self.physical.profile == ExecutionProfile::Compatibility {
			return Err(Error::missing_capability(
				"UI compositor has no validated bounded shader and storage-image binding path",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn supports_swapchain(&self) -> bool {
		self
			.configuration
			.extension_enabled(ash::khr::swapchain::NAME)
	}

	pub(in crate::runtime) fn optimal_format_supports(
		&self,
		format: ash::vk::Format,
		required: ash::vk::FormatFeatureFlags,
	) -> bool {
		// SAFETY: the physical device and its owning instance are retained by this
		// Device. This query only reads immutable format capabilities.
		let properties = unsafe {
			self
				.instance
				.raw()
				.get_physical_device_format_properties(self.physical.handle, format)
		};
		properties.optimal_tiling_features.contains(required)
	}

	/// Create the vertex-color lit graphics pipeline for this device.
	pub(in crate::runtime) fn create_graphics_pipeline(
		&self,
	) -> Result<super::pipeline::GraphicsPipeline> {
		super::pipeline::GraphicsPipeline::new(&self.logical.handle, self.physical.limits)
	}

	/// Create the flat-color material pipeline for this device.
	pub(in crate::runtime) fn create_flat_color_pipeline(
		&self,
	) -> Result<super::pipeline::FlatColorPipeline> {
		super::pipeline::FlatColorPipeline::new(&self.logical.handle, self.physical.limits)
	}

	/// Create the owned descriptor set layouts for the unlit pipeline.
	pub(in crate::runtime) fn create_unlit_descriptor_layouts(
		&self,
	) -> Result<super::pipeline::UnlitDescriptorLayouts> {
		super::pipeline::UnlitDescriptorLayouts::new(&self.logical.handle)
	}

	pub(in crate::runtime) fn create_unlit_descriptor_arena(
		&self,
		layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<super::pipeline::UnlitDescriptorArena> {
		super::pipeline::UnlitDescriptorArena::new(&self.logical.handle, layout, set_count)
	}

	/// Create the owned descriptor set layouts for the Standard Surface pipeline.
	pub(in crate::runtime) fn create_standard_surface_descriptor_layouts(
		&self,
	) -> Result<super::pipeline::StandardSurfaceDescriptorLayouts> {
		super::pipeline::StandardSurfaceDescriptorLayouts::new(&self.logical.handle)
	}

	pub(in crate::runtime) fn create_standard_surface_descriptor_arena(
		&self,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<super::pipeline::StandardSurfaceDescriptorArena> {
		super::pipeline::StandardSurfaceDescriptorArena::new(
			&self.logical.handle,
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
		sampler_layout: ash::vk::DescriptorSetLayout,
	) -> Result<super::pipeline::UnlitPipeline> {
		super::pipeline::UnlitPipeline::new(&self.logical.handle, sampler_layout, self.physical.limits)
	}

	/// Create the Standard Surface PBR pipeline for this device.
	///
	/// `scene_layout` is set 0 (scene lights UBO).
	/// `material_layout` is set 1 (textures + material params UBO).
	pub(in crate::runtime) fn create_standard_surface_pipeline(
		&self,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
	) -> Result<super::pipeline::StandardSurfacePipeline> {
		super::pipeline::StandardSurfacePipeline::new(
			&self.logical.handle,
			scene_layout,
			material_layout,
			self.physical.limits,
		)
	}

	/// Create the Standard Surface transparent-blend pipeline for this device.
	///
	/// Shares the same pipeline layout (and descriptor set layouts) as the
	/// opaque variant; only the depth-write and blend state differ.
	pub(in crate::runtime) fn create_standard_surface_blend_pipeline(
		&self,
		layout: ash::vk::PipelineLayout,
	) -> Result<super::pipeline::StandardSurfaceBlendPipeline> {
		super::pipeline::StandardSurfaceBlendPipeline::new(&self.logical.handle, layout)
	}

	/// Allocate a host-visible persistently-mapped buffer for mesh vertex or index data.
	pub(in crate::runtime) fn allocate_mesh_buffer(
		self: &Arc<Self>,
		size: usize,
		usage: ash::vk::BufferUsageFlags,
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

	pub(in crate::runtime) fn instance(&self) -> &Instance {
		&self.instance
	}

	pub(in crate::runtime) fn graphics_queue_handle(&self) -> Result<ash::vk::Queue> {
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

	pub(in crate::runtime) fn allocator(&self) -> &vk_mem::Allocator {
		&self.logical.allocator
	}

	pub(in crate::runtime) fn bind_storage_buffer(
		&self,
		buffer: ash::vk::Buffer,
		size: ash::vk::DeviceSize,
	) -> Result<u32> {
		self
			.logical
			.descriptors
			.bind_storage_buffer(&self.logical.handle, buffer, size)
	}

	pub(in crate::runtime) fn release_storage_buffer(&self, index: u32) {
		self.logical.descriptors.release_storage_buffer(index);
	}

	/// Return the shared descriptor set for command recording.
	pub(in crate::runtime) fn descriptor_set(&self) -> ash::vk::DescriptorSet {
		self.logical.descriptors.set()
	}

	pub(in crate::runtime) fn bounded_descriptor_layout(
		&self,
		count: u32,
	) -> Option<ash::vk::DescriptorSetLayout> {
		self.logical.descriptors.bounded_layout(count)
	}

	/// Return the shared pipeline layout for UI compute pipelines.
	pub(in crate::runtime) fn ui_pipeline_layout(&self) -> ash::vk::PipelineLayout {
		self.logical.ui_pipeline_layout
	}

	/// Allocate one STORAGE_IMAGE descriptor slot (binding=1) and write `view`.
	pub(in crate::runtime) fn bind_storage_image(&self, view: ash::vk::ImageView) -> Result<u32> {
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
			self
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
				let _creation = self
					.logical
					.pipeline_creation
					.lock()
					.map_err(|_| Error::failed_precondition("Vulkan pipeline creation lock is poisoned"))?;
				if slot.get().is_none() {
					let artifact = if self.physical.profile == ExecutionProfile::Compatibility {
						kernel.bounded_artifact()
					} else {
						kernel.artifact()
					};
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
					let created =
						ComputePipeline::new(&self.logical.handle, layout, artifact, self.physical.limits)?;
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

	pub(in crate::runtime) fn raw(&self) -> &ash::Device {
		&self.logical().handle
	}

	pub(super) fn sync_commands(&self) -> &SyncCommands {
		&self.logical.sync_commands
	}

	/// # Safety
	/// `command` must be recording on this device and the dependency must name
	/// valid retained resources with legal synchronization scopes.
	pub(in crate::runtime) unsafe fn pipeline_barrier(
		&self,
		command: ash::vk::CommandBuffer,
		dependency: &ash::vk::DependencyInfo<'_>,
	) {
		unsafe {
			self
				.logical
				.sync_commands
				.pipeline_barrier(&self.logical.handle, command, dependency);
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
		image: ash::vk::Image,
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
				let range = ash::vk::ImageSubresourceRange::default()
					.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(1);
				let barrier = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::CLEAR)
					.old_layout(ash::vk::ImageLayout::UNDEFINED)
					.new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.image(image)
					.subresource_range(range);
				let barriers = [barrier];
				let dependency = ash::vk::DependencyInfo::default().image_memory_barriers(&barriers);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(&self.logical.handle, command, &dependency)
				};
				let clear = ash::vk::ClearColorValue { float32: color };
				unsafe {
					self.logical.handle.cmd_clear_color_image(
						command,
						image,
						ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						&clear,
						&[range],
					)
				};
				let barrier = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::CLEAR)
					.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::NONE)
					.old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
					.new_layout(ash::vk::ImageLayout::PRESENT_SRC_KHR)
					.image(image)
					.subresource_range(range);
				let barriers = [barrier];
				let dependency = ash::vk::DependencyInfo::default().image_memory_barriers(&barriers);
				unsafe {
					self
						.logical
						.sync_commands
						.pipeline_barrier(&self.logical.handle, command, &dependency)
				};
				Ok(())
			})
	}

	pub(in crate::runtime) fn record_present_blit(
		&self,
		source: ash::vk::Image,
		source_extent: [u32; 2],
		target: ash::vk::Image,
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
				let range = ash::vk::ImageSubresourceRange::default()
					.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(1);
				let barriers = [
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
						.src_access_mask(ash::vk::AccessFlags2::SHADER_SAMPLED_READ)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
						.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
						.old_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
						.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.image(source)
						.subresource_range(range),
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
						.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
						.old_layout(ash::vk::ImageLayout::UNDEFINED)
						.new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
						.image(target)
						.subresource_range(range),
				];
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						command,
						&ash::vk::DependencyInfo::default().image_memory_barriers(&barriers),
					)
				};
				let region = ash::vk::ImageBlit::default()
					.src_subresource(
						ash::vk::ImageSubresourceLayers::default()
							.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
							.layer_count(1),
					)
					.src_offsets([
						ash::vk::Offset3D::default(),
						ash::vk::Offset3D {
							x: source_extent[0] as i32,
							y: source_extent[1] as i32,
							z: 1,
						},
					])
					.dst_subresource(
						ash::vk::ImageSubresourceLayers::default()
							.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
							.layer_count(1),
					)
					.dst_offsets([
						ash::vk::Offset3D::default(),
						ash::vk::Offset3D {
							x: target_extent[0] as i32,
							y: target_extent[1] as i32,
							z: 1,
						},
					]);
				unsafe {
					self.logical.handle.cmd_blit_image(
						command,
						source,
						ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						target,
						ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						&[region],
						ash::vk::Filter::LINEAR,
					)
				};
				let barriers = [
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
						.src_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
						.dst_access_mask(ash::vk::AccessFlags2::SHADER_SAMPLED_READ)
						.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
						.image(source)
						.subresource_range(range),
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::BLIT)
						.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::NONE)
						.old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
						.new_layout(ash::vk::ImageLayout::PRESENT_SRC_KHR)
						.image(target)
						.subresource_range(range),
				];
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						command,
						&ash::vk::DependencyInfo::default().image_memory_barriers(&barriers),
					)
				};
				Ok(())
			})
	}

	pub(in crate::runtime) fn submit_present(
		&self,
		command: &RecordedCommandBuffer,
		wait: ash::vk::Semaphore,
		signal: ash::vk::Semaphore,
		fence: ash::vk::Fence,
	) -> Result<()> {
		let queue = *self
			.logical
			.graphics_queue
			.as_ref()
			.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?;
		let waits = [ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(wait)
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)];
		let commands = [ash::vk::CommandBufferSubmitInfo::default().command_buffer(command.raw())];
		let signals = [ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(signal)
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)];
		let submits = [ash::vk::SubmitInfo2::default()
			.wait_semaphore_infos(&waits)
			.command_buffer_infos(&commands)
			.signal_semaphore_infos(&signals)];
		unsafe {
			self
				.logical
				.sync_commands
				.submit(&self.logical.handle, queue, &submits, fence)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "presentation queue submission", source))
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
		vertex_buffer: ash::vk::Buffer,
		index_buffer: ash::vk::Buffer,
		index_count: u32,
		clear_color: [f32; 4],
		push_data: &[u8],
		push_stages: ash::vk::ShaderStageFlags,
		pipeline: ash::vk::Pipeline,
		pipeline_layout: ash::vk::PipelineLayout,
		descriptor_sets: &[ash::vk::DescriptorSet],
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
		let color_range = ash::vk::ImageSubresourceRange::default()
			.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
			.level_count(1)
			.layer_count(1);
		let depth_range = ash::vk::ImageSubresourceRange::default()
			.aspect_mask(ash::vk::ImageAspectFlags::DEPTH)
			.level_count(1)
			.layer_count(1);
		let result = slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |cmd| {
				// 1. barrier: transition color attachment and depth to write layouts.
				let to_attachment = [
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(
							if old_color_layout == ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								ash::vk::PipelineStageFlags2::ALL_COMMANDS
							} else {
								ash::vk::PipelineStageFlags2::empty()
							},
						)
						.src_access_mask(
							if old_color_layout == ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								ash::vk::AccessFlags2::SHADER_SAMPLED_READ
							} else {
								ash::vk::AccessFlags2::empty()
							},
						)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
						.dst_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
						.old_layout(old_color_layout)
						.new_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.image(color_image)
						.subresource_range(color_range),
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::empty())
						.dst_stage_mask(
							ash::vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
								| ash::vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
						)
						.dst_access_mask(
							ash::vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
								| ash::vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
						)
						.old_layout(ash::vk::ImageLayout::UNDEFINED)
						.new_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.image(depth_image)
						.subresource_range(depth_range),
				];
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
					)
				};
				// 2. begin dynamic rendering
				let clear_cv = ash::vk::ClearColorValue {
					float32: clear_color,
				};
				let clear_dv = ash::vk::ClearDepthStencilValue {
					depth: 1.0,
					stencil: 0,
				};
				let color_att = ash::vk::RenderingAttachmentInfo::default()
					.image_view(color_view)
					.image_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(ash::vk::AttachmentLoadOp::CLEAR)
					.store_op(ash::vk::AttachmentStoreOp::STORE)
					.clear_value(ash::vk::ClearValue { color: clear_cv });
				let depth_att = ash::vk::RenderingAttachmentInfo::default()
					.image_view(depth_view)
					.image_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.load_op(ash::vk::AttachmentLoadOp::CLEAR)
					.store_op(ash::vk::AttachmentStoreOp::DONT_CARE)
					.clear_value(ash::vk::ClearValue {
						depth_stencil: clear_dv,
					});
				let rendering_info = ash::vk::RenderingInfo::default()
					.render_area(ash::vk::Rect2D {
						offset: ash::vk::Offset2D::default(),
						extent: ash::vk::Extent2D { width, height },
					})
					.layer_count(1)
					.color_attachments(std::slice::from_ref(&color_att))
					.depth_attachment(&depth_att);
				unsafe {
					self
						.logical
						.rendering_commands
						.begin(&self.logical.handle, cmd, &rendering_info)
				};
				// 3. set dynamic viewport and scissor
				let viewport = ash::vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: width as f32,
					height: height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				};
				let scissor = ash::vk::Rect2D {
					offset: ash::vk::Offset2D::default(),
					extent: ash::vk::Extent2D { width, height },
				};
				unsafe {
					self.logical.handle.cmd_set_viewport(cmd, 0, &[viewport]);
					self.logical.handle.cmd_set_scissor(cmd, 0, &[scissor]);
				};
				// 4. bind pipeline and push constants
				unsafe {
					self.logical.handle.cmd_bind_pipeline(
						cmd,
						ash::vk::PipelineBindPoint::GRAPHICS,
						pipeline,
					);
					self
						.logical
						.handle
						.cmd_push_constants(cmd, pipeline_layout, push_stages, 0, push_data);
					if !descriptor_sets.is_empty() {
						self.logical.handle.cmd_bind_descriptor_sets(
							cmd,
							ash::vk::PipelineBindPoint::GRAPHICS,
							pipeline_layout,
							0,
							descriptor_sets,
							&[],
						);
					}
				};
				// 5. bind vertex/index buffers and draw
				unsafe {
					self
						.logical
						.handle
						.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
					self.logical.handle.cmd_bind_index_buffer(
						cmd,
						index_buffer,
						0,
						ash::vk::IndexType::UINT32,
					);
					self
						.logical
						.handle
						.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
				};
				// 6. end rendering
				unsafe {
					self
						.logical
						.rendering_commands
						.end(&self.logical.handle, cmd)
				};
				// 7. barrier: color attachment → TRANSFER_SRC for readback
				let to_transfer = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
					.src_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
					.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
					.old_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.image(color_image)
					.subresource_range(color_range);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default()
							.image_memory_barriers(std::slice::from_ref(&to_transfer)),
					)
				};
				// 8. copy color image to host-visible readback buffer
				let copy = ash::vk::BufferImageCopy::default()
					.image_subresource(
						ash::vk::ImageSubresourceLayers::default()
							.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
							.layer_count(1),
					)
					.image_extent(ash::vk::Extent3D {
						width,
						height,
						depth: 1,
					});
				unsafe {
					self.logical.handle.cmd_copy_image_to_buffer(
						cmd,
						color_image,
						ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						readback_buf,
						&[copy],
					)
				};
				// 9. barrier: TRANSFER_SRC → SHADER_READ_ONLY for future Presenter blit
				let to_sampled = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
					.src_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
					.dst_access_mask(ash::vk::AccessFlags2::SHADER_SAMPLED_READ)
					.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
					.image(color_image)
					.subresource_range(color_range);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default()
							.image_memory_barriers(std::slice::from_ref(&to_sampled)),
					)
				};
				Ok(())
			});
		if result.is_ok() {
			target.color_layout = ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
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
		vertex_buffer: ash::vk::Buffer,
		index_buffer: ash::vk::Buffer,
		clear_color: [f32; 4],
		push_stages: ash::vk::ShaderStageFlags,
		pipeline: ash::vk::Pipeline,
		blend_pipeline: ash::vk::Pipeline,
		pipeline_layout: ash::vk::PipelineLayout,
		scene_set: ash::vk::DescriptorSet,
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
		let color_range = ash::vk::ImageSubresourceRange::default()
			.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
			.level_count(1)
			.layer_count(1);
		let depth_range = ash::vk::ImageSubresourceRange::default()
			.aspect_mask(ash::vk::ImageAspectFlags::DEPTH)
			.level_count(1)
			.layer_count(1);
		let result = slot
			.as_mut()
			.expect("graphics command pool initialized")
			.record_custom(&self.logical.handle, |cmd| {
				// 1. barrier: transition color attachment and depth to write layouts.
				let to_attachment = [
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(
							if old_color_layout == ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								ash::vk::PipelineStageFlags2::ALL_COMMANDS
							} else {
								ash::vk::PipelineStageFlags2::empty()
							},
						)
						.src_access_mask(
							if old_color_layout == ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
								ash::vk::AccessFlags2::SHADER_SAMPLED_READ
							} else {
								ash::vk::AccessFlags2::empty()
							},
						)
						.dst_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
						.dst_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
						.old_layout(old_color_layout)
						.new_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.image(color_image)
						.subresource_range(color_range),
					ash::vk::ImageMemoryBarrier2::default()
						.src_stage_mask(ash::vk::PipelineStageFlags2::empty())
						.dst_stage_mask(
							ash::vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
								| ash::vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
						)
						.dst_access_mask(
							ash::vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
								| ash::vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
						)
						.old_layout(ash::vk::ImageLayout::UNDEFINED)
						.new_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.image(depth_image)
						.subresource_range(depth_range),
				];
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
					)
				};
				// 2. begin dynamic rendering
				let clear_cv = ash::vk::ClearColorValue {
					float32: clear_color,
				};
				let clear_dv = ash::vk::ClearDepthStencilValue {
					depth: 1.0,
					stencil: 0,
				};
				let color_att = ash::vk::RenderingAttachmentInfo::default()
					.image_view(color_view)
					.image_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(ash::vk::AttachmentLoadOp::CLEAR)
					.store_op(ash::vk::AttachmentStoreOp::STORE)
					.clear_value(ash::vk::ClearValue { color: clear_cv });
				let depth_att = ash::vk::RenderingAttachmentInfo::default()
					.image_view(depth_view)
					.image_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.load_op(ash::vk::AttachmentLoadOp::CLEAR)
					.store_op(ash::vk::AttachmentStoreOp::DONT_CARE)
					.clear_value(ash::vk::ClearValue {
						depth_stencil: clear_dv,
					});
				let rendering_info = ash::vk::RenderingInfo::default()
					.render_area(ash::vk::Rect2D {
						offset: ash::vk::Offset2D::default(),
						extent: ash::vk::Extent2D { width, height },
					})
					.layer_count(1)
					.color_attachments(std::slice::from_ref(&color_att))
					.depth_attachment(&depth_att);
				unsafe {
					self
						.logical
						.rendering_commands
						.begin(&self.logical.handle, cmd, &rendering_info)
				};
				// 3. set dynamic viewport and scissor
				let viewport = ash::vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: width as f32,
					height: height as f32,
					min_depth: 0.0,
					max_depth: 1.0,
				};
				let scissor = ash::vk::Rect2D {
					offset: ash::vk::Offset2D::default(),
					extent: ash::vk::Extent2D { width, height },
				};
				unsafe {
					self.logical.handle.cmd_set_viewport(cmd, 0, &[viewport]);
					self.logical.handle.cmd_set_scissor(cmd, 0, &[scissor]);
				};
				// 4. bind vertex/index buffers and scene set (set 0) once
				unsafe {
					self
						.logical
						.handle
						.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
					self.logical.handle.cmd_bind_index_buffer(
						cmd,
						index_buffer,
						0,
						ash::vk::IndexType::UINT32,
					);
					self.logical.handle.cmd_bind_descriptor_sets(
						cmd,
						ash::vk::PipelineBindPoint::GRAPHICS,
						pipeline_layout,
						0,
						&[scene_set],
						&[],
					);
				};

				// 5a. opaque pass — bind opaque pipeline, emit non-blend draws
				unsafe {
					self.logical.handle.cmd_bind_pipeline(
						cmd,
						ash::vk::PipelineBindPoint::GRAPHICS,
						pipeline,
					);
				}
				for draw in draws.iter().filter(|d| !d.is_blend) {
					unsafe {
						self.logical.handle.cmd_push_constants(
							cmd,
							pipeline_layout,
							push_stages,
							0,
							&draw.push_data,
						);
						self.logical.handle.cmd_bind_descriptor_sets(
							cmd,
							ash::vk::PipelineBindPoint::GRAPHICS,
							pipeline_layout,
							1,
							&[draw.material_set],
							&[],
						);
						self
							.logical
							.handle
							.cmd_draw_indexed(cmd, draw.index_count, 1, draw.index_start, 0, 0);
					}
				}

				// 5b. transparent pass — switch to blend pipeline, emit blend draws
				let has_blend = draws.iter().any(|d| d.is_blend);
				if has_blend {
					unsafe {
						self.logical.handle.cmd_bind_pipeline(
							cmd,
							ash::vk::PipelineBindPoint::GRAPHICS,
							blend_pipeline,
						);
					}
					for draw in draws.iter().filter(|d| d.is_blend) {
						unsafe {
							self.logical.handle.cmd_push_constants(
								cmd,
								pipeline_layout,
								push_stages,
								0,
								&draw.push_data,
							);
							self.logical.handle.cmd_bind_descriptor_sets(
								cmd,
								ash::vk::PipelineBindPoint::GRAPHICS,
								pipeline_layout,
								1,
								&[draw.material_set],
								&[],
							);
							self.logical.handle.cmd_draw_indexed(
								cmd,
								draw.index_count,
								1,
								draw.index_start,
								0,
								0,
							);
						}
					}
				}
				// 6. end rendering
				unsafe {
					self
						.logical
						.rendering_commands
						.end(&self.logical.handle, cmd)
				};
				// 7. barrier: color attachment → TRANSFER_SRC for readback
				let to_transfer = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
					.src_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
					.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
					.old_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.image(color_image)
					.subresource_range(color_range);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default()
							.image_memory_barriers(std::slice::from_ref(&to_transfer)),
					)
				};
				// 8. copy color image to host-visible readback buffer
				let copy = ash::vk::BufferImageCopy::default()
					.image_subresource(
						ash::vk::ImageSubresourceLayers::default()
							.aspect_mask(ash::vk::ImageAspectFlags::COLOR)
							.layer_count(1),
					)
					.image_extent(ash::vk::Extent3D {
						width,
						height,
						depth: 1,
					});
				unsafe {
					self.logical.handle.cmd_copy_image_to_buffer(
						cmd,
						color_image,
						ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
						readback_buf,
						&[copy],
					)
				};
				// 9. barrier: TRANSFER_SRC → SHADER_READ_ONLY for future Presenter blit
				let to_sampled = ash::vk::ImageMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
					.src_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
					.dst_access_mask(ash::vk::AccessFlags2::SHADER_SAMPLED_READ)
					.old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
					.new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
					.image(color_image)
					.subresource_range(color_range);
				unsafe {
					self.logical.sync_commands.pipeline_barrier(
						&self.logical.handle,
						cmd,
						&ash::vk::DependencyInfo::default()
							.image_memory_barriers(std::slice::from_ref(&to_sampled)),
					)
				};
				Ok(())
			});
		if result.is_ok() {
			target.color_layout = ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		}
		result
	}

	pub(in crate::runtime) fn record_compute_commands(
		&self,
		record: impl FnOnce(ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_custom(&self.logical.handle, record)
	}

	pub(in crate::runtime) fn record_video_decode(
		&self,
		record: impl FnOnce(ash::vk::CommandBuffer) -> Result<()>,
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

	pub(in crate::runtime) fn record_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_compute_graph(
			self,
			&self.logical.pipelines,
			self.logical.descriptors.set(),
			graph,
			None,
			None,
		)
	}

	pub(in crate::runtime) fn record_reusable_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<ReusableCommandBuffer> {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let command = command_pool.record_compute_graph(
			self,
			&self.logical.pipelines,
			self.logical.descriptors.set(),
			graph,
			None,
			Some(self.clone()),
		)?;
		command.reusable().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"reusable command-buffer recording",
				std::io::Error::other("recording did not retain reusable ownership"),
			)
		})
	}

	pub(in crate::runtime) fn record_timed_compute_graph(
		self: &Arc<Self>,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		let timing = TimestampPair::new(
			self,
			self.logical.timestamp_period_ns,
			self.logical.compute_timestamp_valid_bits,
		)?;
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_compute_graph(
			self,
			&self.logical.pipelines,
			self.logical.descriptors.set(),
			graph,
			Some(timing),
			None,
		)
	}

	pub(in crate::runtime) fn submit(
		&self,
		command: &RecordedCommandBuffer,
		epoch: u64,
	) -> Result<()> {
		let command_infos = [ash::vk::CommandBufferSubmitInfo::default().command_buffer(command.raw())];
		let signal_infos = [ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(self.logical.timeline.raw())
			.value(epoch)
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)];
		let wait_info = ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(self.logical.timeline.raw())
			.value(epoch.saturating_sub(1))
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS);
		let wait_infos = if epoch > 1 {
			std::slice::from_ref(&wait_info)
		} else {
			&[]
		};
		let submit_infos = [ash::vk::SubmitInfo2::default()
			.wait_semaphore_infos(wait_infos)
			.command_buffer_infos(&command_infos)
			.signal_semaphore_infos(&signal_infos)];
		let queue = match command.pool_kind() {
			CommandPoolKind::Compute => self.logical.compute_queue,
			CommandPoolKind::Graphics => *self
				.logical
				.graphics_queue
				.as_ref()
				.ok_or_else(|| Error::missing_capability("no Vulkan graphics queue is enabled"))?,
			CommandPoolKind::VideoDecode => *self
				.logical
				.video_decode_queue
				.as_ref()
				.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?,
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
				ash::vk::Fence::null(),
			)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "queue submission", source))
	}

	pub(in crate::runtime) fn free(&self, command: RecordedCommandBuffer) {
		// Drop reusable ownership before taking the pool lock. Its final owner frees
		// the shared handle through this same mutex.
		let Some((handle, pool, _bounded_sets)) = command.into_owned_handle() else {
			return;
		};
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
			CommandPoolKind::VideoDecode => {
				let mut slot = match self.logical.video_decode_command_pool.lock() {
					Ok(slot) => slot,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.logical.handle, handle);
				}
			}
		}
	}

	pub(in crate::runtime) fn free_command_buffer_handle(&self, handle: ash::vk::CommandBuffer) {
		let mut command_pool = match self.logical.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.free_handle(&self.logical.handle, handle);
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
			.map(|kernel| {
				if physical.profile == ExecutionProfile::Compatibility {
					kernel.bounded_artifact().requirements()
				} else {
					kernel.artifact().requirements()
				}
			})
			.collect::<Result<Vec<_>>>()?;
		let handle = configuration.create_device(instance, physical.handle)?;

		let sync_commands = SyncCommands::new(instance, &handle, physical.api_version);
		let rendering_commands = RenderingCommands::new(instance, &handle, physical.api_version);

		// SAFETY: logical-device creation requested queue zero from this family, and
		// the logical device remains alive for the returned queue's full lifetime.
		let compute_queue = unsafe { handle.get_device_queue(physical.compute_queue_family, 0) };
		let graphics_queue = physical.graphics_queue_family.map(|family| {
			// SAFETY: logical-device creation requested queue zero from every unique
			// graphics family retained by physical-device selection.
			unsafe { handle.get_device_queue(family, 0) }
		});
		let video_decode_queue = physical.video.decode_queue_family.map(|family| {
			// SAFETY: logical-device creation requested queue zero from every unique
			// selected video queue family and the device owns the returned handle.
			unsafe { handle.get_device_queue(family, 0) }
		});
		let video_encode_queue = physical.video.encode_queue_family.map(|family| {
			// SAFETY: same queue-creation proof as the decode queue above.
			unsafe { handle.get_device_queue(family, 0) }
		});
		let mut command_pool = match CommandPool::new(
			&handle,
			physical.compute_queue_family,
			CommandPoolKind::Compute,
		) {
			Ok(command_pool) => command_pool,
			Err(error) => {
				// SAFETY: command-pool creation failed, so the device has no live child
				// resources. Creation used no allocation callbacks.
				unsafe {
					handle.destroy_device(None);
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
				unsafe {
					handle.destroy_device(None);
				}
				return Err(error);
			}
		};

		let mut allocator_create_info =
			vk_mem::AllocatorCreateInfo::new(instance.raw(), &handle, physical.handle);
		allocator_create_info.vulkan_api_version = physical.api_version.min(ash::vk::API_VERSION_1_3);
		if configuration.extension_enabled(ash::ext::memory_budget::NAME) {
			allocator_create_info.flags |= vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;
		}

		// SAFETY: the borrowed Instance remains live throughout construction.
		// On success, Device retains it until after allocator/device destruction.
		let allocator = match unsafe { vk_mem::Allocator::new(allocator_create_info) } {
			Ok(allocator) => allocator,
			Err(source) => {
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: allocator creation failed and both existing device children were
				// destroyed. No callbacks were installed.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"memory-allocator creation",
					source,
				));
			}
		};
		let mut descriptors = match DescriptorHeap::new(&handle, physical.limits, physical.profile) {
			Ok(descriptors) => descriptors,
			Err(error) => {
				drop(allocator);
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: every successfully created device child was destroyed above.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(error);
			}
		};
		// DescriptorHeap owns its effective capacity, including pool-allocation
		// reductions. Do not overwrite the physical admission budget with it.
		// Create the shared UI pipeline layout (128-byte push, same descriptor layout).
		let ui_pipeline_layout = {
			let desc_layout = descriptors.layout();
			let descriptor_layouts = [desc_layout];
			let push_range = ash::vk::PushConstantRange::default()
				.stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
				.offset(0)
				.size(128);
			let push_ranges = [push_range];
			let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
				.set_layouts(&descriptor_layouts)
				.push_constant_ranges(&push_ranges);
			match unsafe { handle.create_pipeline_layout(&layout_info, None) } {
				Ok(layout) => layout,
				Err(source) => {
					descriptors.destroy(&handle);
					drop(allocator);
					timeline.destroy(&handle);
					command_pool.destroy(&handle);
					unsafe { handle.destroy_device(None) };
					return Err(Error::backend_failure(
						"Vulkan",
						"UI pipeline layout creation",
						source,
					));
				}
			}
		};
		let timestamp_period_ns = physical.limits.timestamp_period_ns;
		let compute_timestamp_valid_bits = physical.limits.compute_timestamp_valid_bits;
		let mut logical = Self {
			handle,
			sync_commands,
			rendering_commands,
			allocator: ManuallyDrop::new(allocator),
			command_pool: Mutex::new(command_pool),
			graphics_command_pool: Mutex::new(None),
			video_decode_command_pool: Mutex::new(None),
			timeline,
			descriptors,
			ui_pipeline_layout,
			storage_pool: Mutex::new(StoragePool::default()),
			pipelines: Vec::with_capacity(KernelId::ALL.len()),
			pipeline_creation: Mutex::new(()),
			compute_queue,
			graphics_queue,
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
			let artifact = if physical.profile == ExecutionProfile::Compatibility {
				kernel.bounded_artifact()
			} else {
				kernel.artifact()
			};
			// Pipeline creation is illegal above the queried shared-memory budget,
			// independently of Strict/Compatibility or hardware/software identity.
			if artifact.workgroup_memory_size()? > physical.limits.max_compute_shared_memory_size {
				logical.pipelines.push(None);
				continue;
			}
			let descriptor_layout = if physical.profile == ExecutionProfile::Compatibility {
				let Some(layout) = logical
					.descriptors
					.bounded_layout(kernel.bounded_buffer_count())
				else {
					logical.pipelines.push(None);
					continue;
				};
				layout
			} else {
				logical.descriptors.layout()
			};
			// This composed optional kernel has substantial driver compile cost.
			// Admit it normally, then compile on its first preflighted dispatch;
			// unrelated applications must not compile PQC during Engine construction.
			if kernel.requires_secret_storage()
				|| matches!(
					kernel,
					KernelId::CryptographyMlDsa44VerifyU8
						| KernelId::CryptographyMlDsa65VerifyU8
						| KernelId::CryptographyMlDsa87VerifyU8
						| KernelId::CryptographyMlDsa44VerifyPrehashedU8
						| KernelId::CryptographyMlDsa65VerifyPrehashedU8
						| KernelId::CryptographyMlDsa87VerifyPrehashedU8
						| KernelId::CryptographyMlDsa44VerifyHashMessageU8
						| KernelId::CryptographyMlDsa65VerifyHashMessageU8
						| KernelId::CryptographyMlDsa87VerifyHashMessageU8
				) {
				logical.pipelines.push(Some(OnceLock::new()));
				continue;
			}
			match ComputePipeline::new(
				&logical.handle,
				descriptor_layout,
				artifact,
				physical.limits,
			) {
				Ok(pipeline) => logical.pipelines.push(Some(OnceLock::from(pipeline))),
				Err(error)
					if physical.profile == ExecutionProfile::Compatibility
						&& error.kind() == crate::ErrorKind::MissingCapability =>
				{
					logical.pipelines.push(None);
				}
				Err(error) => return Err(error),
			}
		}

		Ok(logical)
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
				self
					.allocator
					.destroy_buffer(buffer.handle, &mut buffer.allocation);
			}
		}
		for slot in self.pipelines.iter_mut().flatten() {
			if let Some(pipeline) = slot.get_mut() {
				pipeline.destroy(&self.handle);
			}
		}
		// SAFETY: no UI pipeline still holds a reference to this layout.
		unsafe {
			self
				.handle
				.destroy_pipeline_layout(self.ui_pipeline_layout, None);
		}
		self.descriptors.destroy(&self.handle);
		// SAFETY: this executor is unpublished during construction or owned by
		// the final Device, so no buffer owner remains. The allocator was initialized
		// exactly once and is dropped exactly once before its device.
		unsafe {
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
		let graphics_pool = match self.graphics_command_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		if let Some(pool) = graphics_pool.as_mut() {
			pool.destroy(&self.handle);
		}
		// SAFETY: the allocator, timeline, and command pools have been destroyed, and no
		// other logical-device children remain at this checkpoint.
		unsafe {
			self.handle.destroy_device(None);
		}
	}
}
