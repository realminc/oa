use crate::{Error, Result, runtime::dna::kernel::ShaderArtifact};

use super::super::physical::DeviceLimits;

/// Custom pipeline families — render passes in [`render`], UI raster
/// families in [`ui`]. The generic descriptor-driven builder lives in this
/// module; each family only declares what differs.
pub(crate) mod render;
pub(crate) mod ui;

// ── Graphics pipeline descriptors ────────────────────────────────────────────

/// Push-constant block contract for an owned pipeline layout.
#[derive(Clone, Copy)]
pub(crate) struct GraphicsPush {
	/// Block size in bytes; validated against the device limit when the
	/// caller supplies one.
	pub(crate) size: u32,
	/// Stages that read the block.
	pub(crate) stages: vk::ShaderStageFlagBits,
}

/// Where a graphics pipeline's layout comes from.
#[derive(Clone)]
pub(crate) enum GraphicsLayoutSource {
	/// Create and own a layout from descriptor set layouts and a push block.
	Owned {
		/// Descriptor set layouts in set order (empty for push-only layouts).
		set_layouts: Vec<vk::DescriptorSetLayout>,
		/// Push-constant block contract.
		push: GraphicsPush,
	},
	/// Borrow a caller-owned layout; the pipeline never destroys it. Used by
	/// the Surface transparent pass (shares the opaque layout) and by every
	/// UI raster family (shares the compositor's per-heap layouts).
	External(vk::PipelineLayout),
}

/// Vertex-fetch configuration for a graphics pipeline.
#[derive(Clone, Copy)]
pub(crate) enum GraphicsVertexInput {
	/// MeshVertex, all four attributes: position(0,f3,off0), normal(1,f3,off12),
	/// uv(2,f2,off24), color(3,f4,off32) — stride 48.
	MeshVertex,
	/// Vertex-color lit variant: position(0,f3,off0), normal(1,f3,off12),
	/// color(2,f4,off32) — stride 48. The lit shader skips the uv attribute
	/// and reads vertex color at location 2.
	MeshVertexLit,
	/// No vertex buffers — the vertex shader derives geometry from
	/// SV_VertexID (fullscreen triangle / instanced quads).
	Vertexless,
}

/// Depth-attachment behavior for a graphics pipeline.
#[derive(Clone, Copy)]
pub(crate) enum GraphicsDepth {
	/// No depth test or write.
	Disabled,
	/// Depth test and write with LESS compare (opaque 3D pass).
	WriteAndTest,
	/// Depth test with LESS compare, no write (back-to-front transparent pass).
	TestOnly,
}

/// Fixed-function blend state for the single color attachment.
#[derive(Clone, Copy)]
pub(crate) enum GraphicsBlend {
	/// Blend disabled — the fragment value replaces the attachment, matching
	/// a compute path that stored into the compose image directly.
	Opaque,
	/// `src + dst * (1 - src.a)` — fragments return premultiplied alpha,
	/// reproducing the compute `uiSourceOver` result exactly.
	PremultipliedAlpha,
	/// `dst * src` — coverage masks multiply into the attachment.
	Modulate,
	/// `src + dst` — contributions accumulate into the attachment.
	Additive,
}

/// One graphics pipeline described declaratively.
///
/// Every render or UI pass reduces to this record: which shaders, where the
/// layout comes from, vertex fetch, depth behavior, blend mode, and
/// attachment formats. [`GraphicsPipeline::create`] owns all create-info
/// assembly, shader-module lifetime, and failure cleanup — passes differ only
/// in the fields of this record.
pub(crate) struct GraphicsPipelineDesc {
	/// Diagnostic name used in creation errors.
	pub(crate) name: &'static str,
	/// SPIR-V bytes for the vertex stage.
	pub(crate) vertex_spv: &'static [u8],
	/// SPIR-V bytes for the fragment stage.
	pub(crate) fragment_spv: &'static [u8],
	/// Owned or borrowed pipeline layout.
	pub(crate) layout: GraphicsLayoutSource,
	/// Vertex-fetch state.
	pub(crate) vertex: GraphicsVertexInput,
	/// Depth test/write behavior.
	pub(crate) depth: GraphicsDepth,
	/// Color-attachment blend state.
	pub(crate) blend: GraphicsBlend,
	/// Single color attachment format.
	pub(crate) color_format: vk::Format,
	/// Depth attachment format; `UNDEFINED` when the pass has no depth target.
	pub(crate) depth_format: vk::Format,
}

// ── Graphics pipeline ─────────────────────────────────────────────────────────

/// Owned graphics pipeline created from a [`GraphicsPipelineDesc`].
///
/// Holds the pipeline handle and its layout. A borrowed layout
/// ([`GraphicsLayoutSource::External`]) is retained for binding but never
/// destroyed with the pipeline; an owned layout retires in `Drop`.
///
/// Uses Vulkan 1.3 dynamic rendering (no render-pass object), dynamic
/// viewport and scissor, triangle-list assembly, fill rasterization with no
/// culling, single-sampled output, and one color attachment.
pub(crate) struct GraphicsPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	owns_layout: bool,
	device: std::sync::Arc<vk::Device>,
}

impl GraphicsPipeline {
	/// Create a graphics pipeline described by `desc`.
	///
	/// `max_push_constants_size`, when supplied, is validated against the push
	/// block of an owned layout; borrowed layouts carry their owner's contract.
	///
	/// # Errors
	///
	/// Returns an error when the device push-constant limit is too small for
	/// an owned layout, when the SPIR-V blobs are malformed, or when layout,
	/// shader-module, or pipeline creation fails.
	pub(crate) fn create(
		device: &std::sync::Arc<vk::Device>,
		max_push_constants_size: Option<u32>,
		cache: vk::PipelineCache,
		desc: &GraphicsPipelineDesc,
	) -> Result<Self> {
		let (layout, owns_layout) = match &desc.layout {
			GraphicsLayoutSource::External(layout) => (*layout, false),
			GraphicsLayoutSource::Owned { set_layouts, push } => {
				if max_push_constants_size.is_some_and(|limit| push.size > limit) {
					return Err(Error::missing_capability(format!(
						"{} pipeline requires {} bytes of push constants",
						desc.name, push.size
					)));
				}
				let push_range = vk::PushConstantRange::new()
					.set_stage_flags(push.stages)
					.set_offset(0)
					.set_size(push.size);
				let layout_info = vk::PipelineLayoutCreateInfo::new()
					.set_set_layouts(set_layouts)
					.map_err(Error::from)?
					.set_push_constant_ranges(std::slice::from_ref(&push_range));
				let layout =
					unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
						.map_err(|s| {
							Error::backend_failure("Vulkan", "graphics pipeline-layout creation", s)
						})?;
				(layout, true)
			}
		};
		let release_layout = |device: &vk::Device| {
			if owns_layout {
				destroy_pipeline_layout(device, layout);
			}
		};

		let vert_words = decode_spirv_bytes(desc.vertex_spv).inspect_err(|_| release_layout(device))?;
		let frag_words =
			decode_spirv_bytes(desc.fragment_spv).inspect_err(|_| release_layout(device))?;
		let vert_module =
			create_shader_module(device, &vert_words).inspect_err(|_| release_layout(device))?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				release_layout(device);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_handle(
			device,
			layout,
			vert_module,
			frag_module,
			desc,
			cache,
		) {
			Ok(h) => h,
			Err(e) => {
				release_layout(device);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			owns_layout,
			device: std::sync::Arc::clone(device),
		})
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}
}

impl Drop for GraphicsPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		if self.owns_layout {
			destroy_pipeline_layout(&self.device, self.layout);
		}
	}
}

/// Assemble the create-info chain and create one graphics pipeline, then
/// destroy both shader modules regardless of outcome.
fn create_graphics_pipeline_handle(
	device: &vk::Device,
	layout: vk::PipelineLayout,
	vert_module: vk::ShaderModule,
	frag_module: vk::ShaderModule,
	desc: &GraphicsPipelineDesc,
	cache: vk::PipelineCache,
) -> Result<vk::Pipeline> {
	let stages = [
		vk::PipelineShaderStageCreateInfo::new()
			.set_stage(vk::ShaderStageFlagBits::VERTEX_BIT)
			.set_module(vert_module)
			.set_name(c"main"),
		vk::PipelineShaderStageCreateInfo::new()
			.set_stage(vk::ShaderStageFlagBits::FRAGMENT_BIT)
			.set_module(frag_module)
			.set_name(c"main"),
	];

	let attribute = |location: u32, format: vk::Format, offset: u32| {
		vk::VertexInputAttributeDescription::new()
			.set_location(location)
			.set_binding(0)
			.set_format(format)
			.set_offset(offset)
	};
	let binding = vk::VertexInputBindingDescription::new()
		.set_binding(0)
		.set_stride(48)
		.set_input_rate(vk::VertexInputRate::VERTEX);
	let attributes: Vec<vk::VertexInputAttributeDescription> = match desc.vertex {
		GraphicsVertexInput::Vertexless => Vec::new(),
		// position, normal, color at location 2 — the lit shader consumes no uv.
		GraphicsVertexInput::MeshVertexLit => vec![
			attribute(0, vk::Format::R32G32B32_SFLOAT, 0),
			attribute(1, vk::Format::R32G32B32_SFLOAT, 12),
			attribute(2, vk::Format::R32G32B32A32_SFLOAT, 32),
		],
		// position, normal, uv, color.
		GraphicsVertexInput::MeshVertex => vec![
			attribute(0, vk::Format::R32G32B32_SFLOAT, 0),
			attribute(1, vk::Format::R32G32B32_SFLOAT, 12),
			attribute(2, vk::Format::R32G32_SFLOAT, 24),
			attribute(3, vk::Format::R32G32B32A32_SFLOAT, 32),
		],
	};
	let vertex_input = if attributes.is_empty() {
		vk::PipelineVertexInputStateCreateInfo::new()
	} else {
		vk::PipelineVertexInputStateCreateInfo::new()
			.set_vertex_binding_descriptions(std::slice::from_ref(&binding))
			.map_err(Error::from)?
			.set_vertex_attribute_descriptions(&attributes)
			.map_err(Error::from)?
	};

	let assembly = vk::PipelineInputAssemblyStateCreateInfo::new()
		.set_topology(vk::PrimitiveTopology::TRIANGLE_LIST);
	let viewport_state = vk::PipelineViewportStateCreateInfo::new()
		.set_viewport_count(1)
		.set_scissor_count(1);
	let raster = vk::PipelineRasterizationStateCreateInfo::new()
		.set_polygon_mode(vk::PolygonMode::FILL)
		.set_cull_mode(vk::CullModeFlagBits::NONE)
		.set_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.set_line_width(1.0);
	let multisample = vk::PipelineMultisampleStateCreateInfo::new()
		.set_rasterization_samples(vk::SampleCountFlagBits::_1_BIT);

	let depth_stencil = match desc.depth {
		GraphicsDepth::Disabled => vk::PipelineDepthStencilStateCreateInfo::new(),
		GraphicsDepth::WriteAndTest => vk::PipelineDepthStencilStateCreateInfo::new()
			.set_depth_test_enable(u32::from(true))
			.set_depth_write_enable(u32::from(true))
			.set_depth_compare_op(vk::CompareOp::LESS)
			.set_min_depth_bounds(0.0)
			.set_max_depth_bounds(1.0),
		GraphicsDepth::TestOnly => vk::PipelineDepthStencilStateCreateInfo::new()
			.set_depth_test_enable(u32::from(true))
			.set_depth_write_enable(u32::from(false))
			.set_depth_compare_op(vk::CompareOp::LESS)
			.set_min_depth_bounds(0.0)
			.set_max_depth_bounds(1.0),
	};

	let write_mask = vk::ColorComponentFlagBits::R_BIT
		| vk::ColorComponentFlagBits::G_BIT
		| vk::ColorComponentFlagBits::B_BIT
		| vk::ColorComponentFlagBits::A_BIT;
	let blend_attachment = match desc.blend {
		GraphicsBlend::Opaque => {
			vk::PipelineColorBlendAttachmentState::new().set_color_write_mask(write_mask)
		}
		GraphicsBlend::PremultipliedAlpha => vk::PipelineColorBlendAttachmentState::new()
			.set_blend_enable(u32::from(true))
			.set_src_color_blend_factor(vk::BlendFactor::ONE)
			.set_dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.set_color_blend_op(vk::BlendOp::ADD)
			.set_src_alpha_blend_factor(vk::BlendFactor::ONE)
			.set_dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.set_alpha_blend_op(vk::BlendOp::ADD)
			.set_color_write_mask(write_mask),
		GraphicsBlend::Modulate => vk::PipelineColorBlendAttachmentState::new()
			.set_blend_enable(u32::from(true))
			.set_src_color_blend_factor(vk::BlendFactor::ZERO)
			.set_dst_color_blend_factor(vk::BlendFactor::SRC_COLOR)
			.set_color_blend_op(vk::BlendOp::ADD)
			.set_src_alpha_blend_factor(vk::BlendFactor::ZERO)
			.set_dst_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.set_alpha_blend_op(vk::BlendOp::ADD)
			.set_color_write_mask(write_mask),
		GraphicsBlend::Additive => vk::PipelineColorBlendAttachmentState::new()
			.set_blend_enable(u32::from(true))
			.set_src_color_blend_factor(vk::BlendFactor::ONE)
			.set_dst_color_blend_factor(vk::BlendFactor::ONE)
			.set_color_blend_op(vk::BlendOp::ADD)
			.set_src_alpha_blend_factor(vk::BlendFactor::ONE)
			.set_dst_alpha_blend_factor(vk::BlendFactor::ONE)
			.set_alpha_blend_op(vk::BlendOp::ADD)
			.set_color_write_mask(write_mask),
	};
	let color_blend = vk::PipelineColorBlendStateCreateInfo::new()
		.set_attachments(std::slice::from_ref(&blend_attachment))
		.map_err(Error::from)?;

	let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
	let dynamic = vk::PipelineDynamicStateCreateInfo::new()
		.set_dynamic_states(&dynamic_states)
		.map_err(Error::from)?;
	let color_formats = [desc.color_format];
	let mut rendering = vk::PipelineRenderingCreateInfo::new()
		.set_color_attachment_formats(&color_formats)
		.map_err(Error::from)?
		.set_depth_attachment_format(desc.depth_format);
	let pipeline_info = vk::GraphicsPipelineCreateInfo::new()
		.push_next(&mut rendering)
		.map_err(Error::from)?
		.set_stages(&stages)
		.map_err(Error::from)?
		.set_vertex_input_state(&vertex_input)
		.set_input_assembly_state(&assembly)
		.set_viewport_state(&viewport_state)
		.set_rasterization_state(&raster)
		.set_multisample_state(&multisample)
		.set_depth_stencil_state(&depth_stencil)
		.set_color_blend_state(&color_blend)
		.set_dynamic_state(&dynamic)
		.set_layout(layout);

	let mut pipelines = [vk::Pipeline::default(); 1];
	// SAFETY: the create-info chain and output buffer remain live and writable
	// for the call; partial output is destroyed on failure.
	let result = unsafe {
		device.create_graphics_pipelines(
			cache,
			std::slice::from_ref(&pipeline_info),
			None,
			&mut pipelines,
		)
	};
	unsafe {
		let _ = device.destroy_shader_module(vert_module, None);
		let _ = device.destroy_shader_module(frag_module, None);
	}
	match result {
		Ok(vk::Result::SUCCESS) => Ok(pipelines[0]),
		result => {
			for handle in pipelines
				.into_iter()
				.filter(|handle| *handle != vk::Pipeline::default())
			{
				destroy_pipeline(device, handle);
			}
			Err(Error::backend_dispatch_failure(
				"Vulkan",
				"graphics-pipeline creation",
				result,
			))
		}
	}
}

pub(crate) fn decode_spirv_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
	if bytes.len() < 4 || !bytes.len().is_multiple_of(4) {
		return Err(Error::backend_failure(
			"Vulkan",
			"graphics shader SPIR-V byte length",
			std::io::Error::other("not a multiple of 4"),
		));
	}
	Ok(
		bytes
			.chunks(4)
			.map(|c| {
				let mut word = [0u8; 4];
				word.copy_from_slice(c);
				u32::from_le_bytes(word)
			})
			.collect(),
	)
}

fn create_shader_module(device: &vk::Device, words: &[u32]) -> Result<vk::ShaderModule> {
	let info = vk::ShaderModuleCreateInfo::new().set_code(words);
	unsafe { device.create_shader_module(&info, None) }
		.map_err(|s| Error::backend_failure("Vulkan", "graphics shader-module creation", s))
}

// ── Pipeline cache persistence ────────────────────────────────────────────────

/// File name of the legacy raw driver blob below `var/vk`, retained only as a
/// migration source for the scoped `.vkcache` envelope.
const LEGACY_PIPELINE_CACHE_FILE: &str = "pipeline.vcache";

/// Path of the legacy raw pipeline cache — read once for import, never written.
pub(in crate::runtime) fn legacy_pipeline_cache_path() -> crate::Path {
	crate::Path::var()
		.join("vk")
		.join(LEGACY_PIPELINE_CACHE_FILE)
}

/// Owned `VkPipelineCache` optionally seeded with persisted driver data.
///
/// Vulkan requires external synchronization for every host access to one
/// pipeline cache. Serialized single-pipeline creation shares this cache under
/// the device creation lock; parallel preload gives each worker its own cache
/// and merges worker caches back only after every worker has joined.
pub(in crate::runtime::device) struct PipelineCache {
	handle: vk::PipelineCache,
	device: std::sync::Arc<vk::Device>,
	initial_data_bytes: usize,
}

impl PipelineCache {
	/// Create a pipeline cache seeded with `initial_data` (empty means cold).
	///
	/// # Errors
	/// Returns an error when `vkCreatePipelineCache` rejects the seed data or
	/// fails on the device.
	pub(in crate::runtime::device) fn new(
		device: std::sync::Arc<vk::Device>,
		initial_data: &[u8],
	) -> Result<Self> {
		let info = vk::PipelineCacheCreateInfo::new()
			.set_initial_data(initial_data)
			.map_err(Error::from)?;
		// SAFETY: the seed slice outlives the call, and the retained device
		// outlives the returned cache.
		let handle = unsafe { device.create_pipeline_cache(&info, None) }
			.map_err(|s| Error::backend_failure("Vulkan", "pipeline-cache creation", s))?;
		Ok(Self {
			handle,
			device,
			initial_data_bytes: initial_data.len(),
		})
	}

	/// Open a pipeline cache seeded with persisted driver data.
	///
	/// The authoritative source is the scope's `.vkcache` driver section. A
	/// legacy `pipeline.vcache` is imported only when no envelope exists for
	/// this scope, and only after its standard Vulkan header validates against
	/// the selected device. An absent, mismatched or corrupt seed still
	/// produces a live cold cache; `None` is returned only when the driver
	/// cannot create any cache, in which case callers run uncached.
	pub(in crate::runtime::device) fn open(
		device: &std::sync::Arc<vk::Device>,
		physical: &super::super::physical::DevicePhysical,
	) -> Option<Self> {
		let scope = super::vkcache::VkCacheScope::new(physical);
		let path = scope.path();
		let (initial_data, source) = if path.is_file() {
			// A present envelope is authoritative: no legacy import, even when
			// the driver section is absent or fails validation.
			let data = super::vkcache::VkCacheFile::open(&path, &scope)
				.ok()
				.and_then(|file| {
					file
						.section(super::vkcache::SECTION_DRIVER_PIPELINE_CACHE)
						.filter(|section| section.version == 1)
						.map(|section| section.bytes.clone())
				})
				.filter(|bytes| super::vkcache::driver_blob_matches_scope(bytes, &scope))
				.unwrap_or_default();
			(data, path.display().to_string())
		} else {
			let legacy = legacy_pipeline_cache_path();
			let data = crate::Filesystem::read_binary(&legacy)
				.ok()
				.filter(|bytes| super::vkcache::driver_blob_matches_scope(bytes, &scope))
				.unwrap_or_default();
			(data, legacy.display().to_string())
		};
		match Self::new(std::sync::Arc::clone(device), &initial_data) {
			Ok(cache) => {
				if !initial_data.is_empty() {
					crate::log_info!(
						crate::LogComponent::COMPUTE,
						"pipeline cache: loaded {} bytes from {}",
						initial_data.len(),
						source
					);
				}
				Some(cache)
			}
			Err(_) if !initial_data.is_empty() => {
				crate::log_warn!(
					crate::LogComponent::COMPUTE,
					"pipeline cache: driver rejected persisted data; starting cold"
				);
				Self::new(std::sync::Arc::clone(device), &[]).ok()
			}
			Err(_) => None,
		}
	}

	/// Serialize the driver-owned cache blob.
	pub(in crate::runtime::device) fn data(&self) -> Result<Vec<u8>> {
		// SAFETY: `handle` is a live cache created on this device.
		unsafe { self.device.get_pipeline_cache_data(self.handle) }
			.map_err(|s| Error::backend_failure("Vulkan", "pipeline-cache data query", s))
	}

	/// Merge externally synchronized worker caches into this cache.
	///
	/// Callers must have joined every thread that produced `sources`; no host
	/// may touch source or destination concurrently with the merge.
	pub(in crate::runtime::device) fn merge(&self, sources: &[vk::PipelineCache]) -> Result<()> {
		if sources.is_empty() {
			return Ok(());
		}
		// SAFETY: the caller joins all producer threads before merging, so
		// sources and destination are externally synchronized.
		unsafe { self.device.merge_pipeline_caches(self.handle, sources) }
			.map_err(|s| Error::backend_failure("Vulkan", "pipeline-cache merge", s))
	}

	pub(in crate::runtime::device) const fn raw(&self) -> vk::PipelineCache {
		self.handle
	}

	pub(in crate::runtime::device) const fn initial_data_bytes(&self) -> usize {
		self.initial_data_bytes
	}
}

impl Drop for PipelineCache {
	fn drop(&mut self) {
		// SAFETY: the handle belongs to `device` and is destroyed at most once.
		let _ = unsafe { self.device.destroy_pipeline_cache(self.handle, None) };
	}
}

// ── Compute pipeline ──────────────────────────────────────────────────────────

pub(crate) struct ComputePipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	max_dispatch_group_count: [u32; 3],
	push_constant_size: u32,
	device: std::sync::Arc<vk::Device>,
}

impl ComputePipeline {
	pub(in crate::runtime::device) fn new(
		device: std::sync::Arc<vk::Device>,
		descriptor_layout: vk::DescriptorSetLayout,
		artifact: &'static ShaderArtifact,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		Self::from_words(
			device,
			descriptor_layout,
			&artifact.spirv_words()?,
			artifact.workgroup_size,
			limits,
			cache,
		)
	}

	/// Consume validated SPIR-V synchronously; the module does not borrow its words.
	/// Callers own target, descriptor, capability and validation admission.
	pub(in crate::runtime::device) fn from_words(
		device: std::sync::Arc<vk::Device>,
		descriptor_layout: vk::DescriptorSetLayout,
		words: &[u32],
		workgroup_size: [u32; 3],
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		let push_constant_size = crate::runtime::dna::kernel::reflect_push_constant_size(words)?;
		validate_artifact_limits(workgroup_size, push_constant_size, limits)?;

		let descriptor_layouts = [descriptor_layout];
		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT)
			.set_offset(0)
			.set_size(push_constant_size);
		let push_ranges = [push_range];
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_set_layouts(&descriptor_layouts)
			.map_err(Error::from)?
			.set_push_constant_ranges(&push_ranges);
		// SAFETY: the shared descriptor layout is live and the reflected push range
		// fits the queried device limit and Vulkan's four-byte alignment requirements.
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|source| {
					Error::backend_failure("Vulkan", "compute pipeline-layout creation", source)
				})?;

		let module_info = vk::ShaderModuleCreateInfo::new().set_code(words);
		// SAFETY: the caller's spirv-val and reflection admission proved a complete
		// SPIR-V word stream; the device remains live through pipeline creation.
		let module = match unsafe { device.create_shader_module(&module_info, None) } {
			Ok(module) => module,
			Err(source) => {
				destroy_pipeline_layout(&device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute shader-module creation",
					source,
				));
			}
		};

		let stage = vk::PipelineShaderStageCreateInfo::new()
			.set_stage(vk::ShaderStageFlagBits::COMPUTE_BIT)
			.set_module(module)
			.set_name(c"main");
		let pipeline_info = vk::ComputePipelineCreateInfo::new()
			.set_stage(stage)
			.set_layout(layout);
		// SAFETY: the module contains the reflected compute entry point, and the
		// pipeline layout exactly matches its validated descriptor and push ABI.
		let mut pipelines = [vk::Pipeline::default(); 1];
		let pipeline_result =
			unsafe { device.create_compute_pipelines(cache, &[pipeline_info], None, &mut pipelines) };
		// The module is no longer needed after pipeline creation, successful or not.
		unsafe {
			let _ = device.destroy_shader_module(module, None);
		}
		let handle = match pipeline_result {
			Ok(vk::Result::SUCCESS) => pipelines[0],
			result => {
				for pipeline in pipelines
					.into_iter()
					.filter(|handle| *handle != vk::Pipeline::default())
				{
					destroy_pipeline(&device, pipeline);
				}
				destroy_pipeline_layout(&device, layout);
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"compute-pipeline creation",
					result,
				));
			}
		};

		Ok(Self {
			handle,
			layout,
			max_dispatch_group_count: limits.max_compute_work_group_count,
			push_constant_size,
			device,
		})
	}

	pub(in crate::runtime::device) fn destroy(&mut self, device: &vk::Device) {
		destroy_pipeline(device, self.handle);
		destroy_pipeline_layout(device, self.layout);
		self.handle = vk::Pipeline::NULL;
		self.layout = vk::PipelineLayout::NULL;
	}

	pub(in crate::runtime::device) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(in crate::runtime::device) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}

	pub(crate) const fn max_dispatch_group_count(&self) -> [u32; 3] {
		self.max_dispatch_group_count
	}

	pub(crate) const fn push_constant_size(&self) -> u32 {
		self.push_constant_size
	}
}

impl Drop for ComputePipeline {
	fn drop(&mut self) {
		// Registry teardown explicitly destroys pipelines before descriptor layouts;
		// owned specializations instead retire with the last recorded command.
		if self.handle != vk::Pipeline::NULL {
			destroy_pipeline(&self.device, self.handle);
		}
		if self.layout != vk::PipelineLayout::NULL {
			destroy_pipeline_layout(&self.device, self.layout);
		}
	}
}

pub(in crate::runtime::device) fn validate_artifact_limits(
	workgroup_size: [u32; 3],
	push_constant_size: u32,
	limits: DeviceLimits,
) -> Result<()> {
	let invocation_count = workgroup_size
		.into_iter()
		.try_fold(1_u32, u32::checked_mul)
		.ok_or_else(|| Error::missing_capability("shader workgroup size overflows u32"))?;
	if push_constant_size > limits.max_push_constants_size
		|| invocation_count > limits.max_compute_work_group_invocations
		|| workgroup_size
			.into_iter()
			.zip(limits.max_compute_work_group_size)
			.any(|(required, available)| required > available)
	{
		return Err(Error::missing_capability(
			"selected device cannot admit the reflected compute shader ABI",
		));
	}
	Ok(())
}

fn destroy_pipeline(device: &vk::Device, pipeline: vk::Pipeline) {
	// SAFETY: the handle belongs to `device` and is destroyed at most once after use.
	let _ = unsafe { device.destroy_pipeline(pipeline, None) };
}

fn destroy_pipeline_layout(device: &vk::Device, layout: vk::PipelineLayout) {
	// SAFETY: the layout belongs to `device` and no live pipeline uses it.
	let _ = unsafe { device.destroy_pipeline_layout(layout, None) };
}
