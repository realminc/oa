//! UI raster pipelines — one graphics pipeline per compositor command family.
//!
//! Every element family draws into the compose color attachment inside one
//! dynamic-rendering pass. Per-command descriptor sets bind the family's
//! fixed storage-buffer slots (binding 0, count = the family's `heap[N]`
//! declaration) for both execution profiles — the raster compositor never
//! uses the Strict bindless heap or a storage image.

use std::sync::Arc;

use crate::{Error, Result};

// Embedded compiled SPIR-V — vertex stages shared per shape family and one
// fragment stage per element family.
static FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_fullscreen_vert.spv"
));
static CANVAS_GRID_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_canvas_grid_vert.spv"
));
static CANVAS_GRID_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_canvas_grid_frag.spv"
));
static CLEAR_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_clear_frag.spv"
));
static RECT_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_rect_frag.spv"
));
static RECT_OUTLINE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_rect_outline_frag.spv"
));
static LINE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_line_frag.spv"
));
static TRIANGLE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_triangle_frag.spv"
));
static MASK_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_mask_frag.spv"
));
static BLIT_TEXTURE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_blit_texture_frag.spv"
));
static BLIT_PLANAR_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_blit_planar_frag.spv"
));
static HEATMAP_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_heatmap_frag.spv"
));
static WAVEFORM_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_waveform_frag.spv"
));
static PLOT_LINE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_plot_line_frag.spv"
));
static GLYPHS_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_glyphs_vert.spv"
));
static GLYPHS_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_glyphs_frag.spv"
));
static RECT_OUTLINES_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_rect_outlines_vert.spv"
));
static RECT_OUTLINES_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_rect_outlines_frag.spv"
));

/// Fixed-function color-attachment blend state matching one element family.
#[derive(Clone, Copy)]
pub(super) enum UiBlend {
	/// Blend disabled — the fragment value replaces the attachment, matching
	/// a compute path that stored into the compose image directly.
	Opaque,
	/// `src + dst * (1 - src.a)` — the fragment returns premultiplied alpha,
	/// reproducing the compute `uiSourceOver` result exactly.
	SourceOver,
	/// `dst * src` — the fragment returns the rounded-mask coverage; used by
	/// the first rounded-mask draw.
	Modulate,
	/// `src + dst` — the fragment returns the premultiplied outside color
	/// contribution; used by the second rounded-mask draw.
	Additive,
}

/// All fifteen UI raster pipelines owned by one `Ui` instance.
///
/// `UiPipelines` is placed before `layouts` so every pipeline retires before
/// the layouts they were created against.
pub(super) struct UiPipelines {
	pub(super) clear: UiPipeline,
	pub(super) rect: UiPipeline,
	pub(super) rect_outline: UiPipeline,
	pub(super) line: UiPipeline,
	pub(super) triangle: UiPipeline,
	pub(super) canvas_grid: UiPipeline,
	pub(super) blit_texture: UiPipeline,
	pub(super) blit_planar: UiPipeline,
	pub(super) heatmap: UiPipeline,
	pub(super) waveform: UiPipeline,
	pub(super) plot_line: UiPipeline,
	pub(super) glyphs: UiPipeline,
	pub(super) rect_outlines: UiPipeline,
	pub(super) mask_modulate: UiPipeline,
	pub(super) mask_additive: UiPipeline,
	pub(super) layouts: UiLayouts,
}

impl UiPipelines {
	pub(super) fn new(device: &Arc<vk::Device>, cache: vk::PipelineCache) -> Result<Self> {
		let layouts = UiLayouts::new(device)?;
		let family = |count: usize, vert: &[u8], frag: &[u8], blend: UiBlend| {
			UiPipeline::new(device, layouts.pipelines[count], vert, frag, blend, cache)
		};
		Ok(Self {
			clear: family(0, FULLSCREEN_VERT_SPV, CLEAR_FRAG_SPV, UiBlend::Opaque)?,
			rect: family(0, FULLSCREEN_VERT_SPV, RECT_FRAG_SPV, UiBlend::SourceOver)?,
			rect_outline: family(
				0,
				FULLSCREEN_VERT_SPV,
				RECT_OUTLINE_FRAG_SPV,
				UiBlend::SourceOver,
			)?,
			line: family(0, FULLSCREEN_VERT_SPV, LINE_FRAG_SPV, UiBlend::SourceOver)?,
			triangle: family(
				0,
				FULLSCREEN_VERT_SPV,
				TRIANGLE_FRAG_SPV,
				UiBlend::SourceOver,
			)?,
			canvas_grid: family(
				0,
				CANVAS_GRID_VERT_SPV,
				CANVAS_GRID_FRAG_SPV,
				UiBlend::SourceOver,
			)?,
			blit_texture: UiPipeline::new(
				device,
				layouts.blit_pipeline,
				FULLSCREEN_VERT_SPV,
				BLIT_TEXTURE_FRAG_SPV,
				UiBlend::SourceOver,
				cache,
			)?,
			blit_planar: family(
				4,
				FULLSCREEN_VERT_SPV,
				BLIT_PLANAR_FRAG_SPV,
				UiBlend::Opaque,
			)?,
			heatmap: family(1, FULLSCREEN_VERT_SPV, HEATMAP_FRAG_SPV, UiBlend::Opaque)?,
			waveform: family(
				1,
				FULLSCREEN_VERT_SPV,
				WAVEFORM_FRAG_SPV,
				UiBlend::SourceOver,
			)?,
			plot_line: family(
				1,
				FULLSCREEN_VERT_SPV,
				PLOT_LINE_FRAG_SPV,
				UiBlend::SourceOver,
			)?,
			glyphs: family(2, GLYPHS_VERT_SPV, GLYPHS_FRAG_SPV, UiBlend::SourceOver)?,
			rect_outlines: family(
				1,
				RECT_OUTLINES_VERT_SPV,
				RECT_OUTLINES_FRAG_SPV,
				UiBlend::Opaque,
			)?,
			mask_modulate: family(0, FULLSCREEN_VERT_SPV, MASK_FRAG_SPV, UiBlend::Modulate)?,
			mask_additive: family(0, FULLSCREEN_VERT_SPV, MASK_FRAG_SPV, UiBlend::Additive)?,
			layouts,
		})
	}
}

/// One vertexless or instanced graphics pipeline with its family layout.
pub(super) struct UiPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: Arc<vk::Device>,
}

impl UiPipeline {
	fn new(
		device: &Arc<vk::Device>,
		layout: vk::PipelineLayout,
		vert_spv: &[u8],
		frag_spv: &[u8],
		blend: UiBlend,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		let vert_words = decode_spirv(vert_spv)?;
		let frag_words = decode_spirv(frag_spv)?;
		let vert_module = create_shader_module(device, &vert_words)?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(module) => module,
			Err(error) => {
				let _ = unsafe { device.destroy_shader_module(vert_module, None) };
				return Err(error);
			}
		};
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
		let vertex_input = vk::PipelineVertexInputStateCreateInfo::new();
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
		let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::new();
		let write_mask = vk::ColorComponentFlagBits::R_BIT
			| vk::ColorComponentFlagBits::G_BIT
			| vk::ColorComponentFlagBits::B_BIT
			| vk::ColorComponentFlagBits::A_BIT;
		let blend_attachment = match blend {
			UiBlend::Opaque => {
				vk::PipelineColorBlendAttachmentState::new().set_color_write_mask(write_mask)
			}
			UiBlend::SourceOver => vk::PipelineColorBlendAttachmentState::new()
				.set_blend_enable(u32::from(true))
				.set_src_color_blend_factor(vk::BlendFactor::ONE)
				.set_dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
				.set_color_blend_op(vk::BlendOp::ADD)
				.set_src_alpha_blend_factor(vk::BlendFactor::ONE)
				.set_dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
				.set_alpha_blend_op(vk::BlendOp::ADD)
				.set_color_write_mask(write_mask),
			UiBlend::Modulate => vk::PipelineColorBlendAttachmentState::new()
				.set_blend_enable(u32::from(true))
				.set_src_color_blend_factor(vk::BlendFactor::ZERO)
				.set_dst_color_blend_factor(vk::BlendFactor::SRC_COLOR)
				.set_color_blend_op(vk::BlendOp::ADD)
				.set_src_alpha_blend_factor(vk::BlendFactor::ZERO)
				.set_dst_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
				.set_alpha_blend_op(vk::BlendOp::ADD)
				.set_color_write_mask(write_mask),
			UiBlend::Additive => vk::PipelineColorBlendAttachmentState::new()
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
		let color_formats = [vk::Format::R8G8B8A8_UNORM];
		let mut rendering = vk::PipelineRenderingCreateInfo::new()
			.set_color_attachment_formats(&color_formats)
			.map_err(Error::from)?
			.set_depth_attachment_format(vk::Format::UNDEFINED);
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
		// SAFETY: the create-info chain and output buffer remain live and
		// writable for the call; partial output is destroyed on failure.
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
		let handle = match result {
			Ok(vk::Result::SUCCESS) => pipelines[0],
			result => {
				for pipeline in pipelines
					.into_iter()
					.filter(|handle| *handle != vk::Pipeline::default())
				{
					let _ = unsafe { device.destroy_pipeline(pipeline, None) };
				}
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"UI raster pipeline creation",
					result,
				));
			}
		};
		Ok(Self {
			handle,
			layout,
			device: device.clone(),
		})
	}

	pub(super) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}

	pub(super) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}
}

impl Drop for UiPipeline {
	fn drop(&mut self) {
		// SAFETY: pipeline belongs to the device; layout is external and must be kept alive.
		let _ = unsafe { self.device.destroy_pipeline(self.handle, None) };
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn decode_spirv(bytes: &[u8]) -> Result<Vec<u32>> {
	if !bytes.len().is_multiple_of(4) {
		return Err(Error::backend_failure(
			"SPIR-V",
			"UI shader byte alignment",
			std::io::Error::other("SPIR-V byte count is not a multiple of 4"),
		));
	}
	Ok(
		bytes
			.as_chunks::<4>()
			.0
			.iter()
			.map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
			.collect(),
	)
}

fn create_shader_module(device: &vk::Device, words: &[u32]) -> Result<vk::ShaderModule> {
	let info = vk::ShaderModuleCreateInfo::new().set_code(words);
	unsafe { device.create_shader_module(&info, None) }
		.map_err(|source| Error::backend_failure("Vulkan", "UI shader module creation", source))
}

/// Fixed descriptor layouts for zero through four storage-buffer slots plus
/// the sampled-image blit layout.
///
/// Every raster shader declares `heap[N]` at binding 0; a family pipeline
/// layout binds the descriptor layout whose count matches its `N`. The image
/// blit declares `Sampler2D` at binding 0, so it owns a dedicated layout and
/// pipeline layout. The push range covers the largest family push block
/// (100 bytes) and is visible to both stages.
pub(super) struct UiLayouts {
	device: Arc<vk::Device>,
	pub(super) descriptors: Vec<vk::DescriptorSetLayout>,
	pub(super) pipelines: Vec<vk::PipelineLayout>,
	pub(super) blit_descriptors: vk::DescriptorSetLayout,
	pub(super) blit_pipeline: vk::PipelineLayout,
	/// `[nearest, linear]` — bound per blit command by destination scale.
	pub(super) samplers: [vk::Sampler; 2],
}

impl UiLayouts {
	fn new(device: &Arc<vk::Device>) -> Result<Self> {
		let stages = vk::ShaderStageFlagBits::VERTEX_BIT | vk::ShaderStageFlagBits::FRAGMENT_BIT;
		let mut result = Self {
			device: device.clone(),
			descriptors: Vec::new(),
			pipelines: Vec::new(),
			blit_descriptors: vk::DescriptorSetLayout::default(),
			blit_pipeline: vk::PipelineLayout::default(),
			samplers: [vk::Sampler::default(); 2],
		};
		for count in 0..=4 {
			let bindings = if count == 0 {
				Vec::new()
			} else {
				vec![
					vk::DescriptorSetLayoutBinding::new()
						.set_binding(0)
						.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.set_descriptor_count(count)
						.set_stage_flags(stages),
				]
			};
			// SAFETY: fixed counts match the shaders' literal heap[N] arrays.
			let descriptor = unsafe {
				device.create_descriptor_set_layout(
					&(vk::DescriptorSetLayoutCreateInfo::new()
						.set_bindings(&bindings)
						.map_err(Error::from)?),
					None,
				)
			}
			.map_err(|e| Error::backend_failure("Vulkan", "UI raster descriptor layout", e))?;
			result.descriptors.push(descriptor);
			let descriptors = [descriptor];
			let push = [vk::PushConstantRange::new()
				.set_size(128)
				.set_stage_flags(stages)];
			// SAFETY: the descriptor layout is retained by result and the push
			// range fits every family's push block.
			let pipeline = unsafe {
				device.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::new()
						.set_set_layouts(&descriptors)
						.map_err(Error::from)?
						.set_push_constant_ranges(&push)
						.map_err(Error::from)?,
					None,
				)
			}
			.map_err(|e| Error::backend_failure("Vulkan", "UI raster pipeline layout", e))?;
			result.pipelines.push(pipeline);
		}
		let fragment = vk::ShaderStageFlagBits::FRAGMENT_BIT;
		let blit_binding = [vk::DescriptorSetLayoutBinding::new()
			.set_binding(0)
			.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.set_descriptor_count(1)
			.set_stage_flags(fragment)];
		result.blit_descriptors = unsafe {
			device.create_descriptor_set_layout(
				&(vk::DescriptorSetLayoutCreateInfo::new()
					.set_bindings(&blit_binding)
					.map_err(Error::from)?),
				None,
			)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "UI blit descriptor layout", e))?;
		let blit_sets = [result.blit_descriptors];
		let push = [vk::PushConstantRange::new()
			.set_size(128)
			.set_stage_flags(stages)];
		result.blit_pipeline = unsafe {
			device.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::new()
					.set_set_layouts(&blit_sets)
					.map_err(Error::from)?
					.set_push_constant_ranges(&push)
					.map_err(Error::from)?,
				None,
			)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "UI blit pipeline layout", e))?;
		for (index, filter) in [vk::Filter::NEAREST, vk::Filter::LINEAR]
			.into_iter()
			.enumerate()
		{
			let info = vk::SamplerCreateInfo::new()
				.set_mag_filter(filter)
				.set_min_filter(filter)
				.set_mipmap_mode(vk::SamplerMipmapMode::NEAREST)
				.set_address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.set_address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.set_address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.set_min_lod(0.0)
				.set_max_lod(0.0);
			result.samplers[index] = unsafe { device.create_sampler(&info, None) }
				.map_err(|e| Error::backend_failure("Vulkan", "UI blit sampler creation", e))?;
		}
		Ok(result)
	}
}

impl Drop for UiLayouts {
	fn drop(&mut self) {
		// SAFETY: pipelines and submitted descriptor owners have retired first.
		unsafe {
			for sampler in &self.samplers {
				if !sampler.is_null() {
					let _ = self.device.destroy_sampler(*sampler, None);
				}
			}
			if !self.blit_pipeline.is_null() {
				let _ = self
					.device
					.destroy_pipeline_layout(self.blit_pipeline, None);
			}
			if !self.blit_descriptors.is_null() {
				let _ = self
					.device
					.destroy_descriptor_set_layout(self.blit_descriptors, None);
			}
			for layout in &self.pipelines {
				let _ = self.device.destroy_pipeline_layout(*layout, None);
			}
			for layout in &self.descriptors {
				let _ = self.device.destroy_descriptor_set_layout(*layout, None);
			}
		}
	}
}

/// Immutable per-command descriptor sets for one composition.
///
/// Each command's set binds its source buffers in literal `heap[k]` order —
/// the raster shaders index slots directly, so no command remapping is
/// required. The presentation slot or exact readback completion retains this
/// allocation.
pub(super) struct UiFrameBindings {
	device: Arc<vk::Device>,
	_engine: crate::runtime::EngineHandle,
	pool: vk::DescriptorPool,
	pub(super) sets: Vec<vk::DescriptorSet>,
	pub(super) layouts: Vec<vk::PipelineLayout>,
}

/// One command's descriptor payload — literal `heap[N]` storage buffers or
/// the blit's single combined image sampler.
enum CmdBinding {
	Buffers(Vec<vk::DescriptorBufferInfo>),
	Image {
		view: vk::ImageView,
		sampler: vk::Sampler,
	},
}

impl UiFrameBindings {
	pub(super) fn new(
		engine: &crate::runtime::EngineHandle,
		layouts: &UiLayouts,
		commands: &[super::command::UiCmd],
		resolve: impl Fn(u32) -> Result<vk::DescriptorBufferInfo>,
	) -> Result<Self> {
		use super::command::UiCmd;
		let device = engine.raw_device();
		let mut bindings = Vec::with_capacity(commands.len());
		for command in commands {
			let indices: Vec<u32> = match command {
				UiCmd::BlitRgba { view, linear, .. } => {
					bindings.push(CmdBinding::Image {
						view: *view,
						sampler: layouts.samplers[usize::from(*linear)],
					});
					continue;
				}
				UiCmd::BlitPlanar {
					r_idx,
					g_idx,
					b_idx,
					a_idx,
					..
				} => {
					// The fragment reads slots 0..4 literally; absent channels
					// bind the R plane and are never read.
					let resolve_slot = |index: u32| if index == u32::MAX { *r_idx } else { index };
					vec![
						resolve_slot(*r_idx),
						resolve_slot(*g_idx),
						resolve_slot(*b_idx),
						resolve_slot(*a_idx),
					]
				}
				UiCmd::PlotLine { values_idx, .. }
				| UiCmd::Waveform { values_idx, .. }
				| UiCmd::Heatmap { values_idx, .. } => vec![*values_idx],
				UiCmd::Glyphs {
					glyph_idx,
					atlas_idx,
					..
				} => vec![*glyph_idx, *atlas_idx],
				UiCmd::RectOutlines { rect_idx, .. } => vec![*rect_idx],
				_ => Vec::new(),
			};
			bindings.push(CmdBinding::Buffers(
				indices
					.into_iter()
					.map(&resolve)
					.collect::<Result<Vec<_>>>()?,
			));
		}
		let count = u32::try_from(commands.len())
			.map_err(|_| Error::resource_exhausted("too many UI commands"))?;
		let mut buffer_count = 0_u32;
		let mut image_count = 0_u32;
		for binding in &bindings {
			match binding {
				CmdBinding::Buffers(buffers) => {
					buffer_count = buffer_count
						.checked_add(buffers.len() as u32)
						.ok_or_else(|| Error::resource_exhausted("UI descriptor count overflow"))?;
				}
				CmdBinding::Image { .. } => {
					image_count = image_count
						.checked_add(1)
						.ok_or_else(|| Error::resource_exhausted("UI descriptor count overflow"))?;
				}
			}
		}
		let mut sizes = Vec::new();
		if buffer_count != 0 {
			sizes.push(
				vk::DescriptorPoolSize::new()
					.set_type(vk::DescriptorType::STORAGE_BUFFER)
					.set_descriptor_count(buffer_count),
			);
		}
		if image_count != 0 {
			sizes.push(
				vk::DescriptorPoolSize::new()
					.set_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.set_descriptor_count(image_count),
			);
		}
		// SAFETY: sizes cover all exact per-command layouts; sets are immutable.
		let pool = unsafe {
			device.create_descriptor_pool(
				&(vk::DescriptorPoolCreateInfo::new())
					.set_max_sets(count.max(1))
					.set_pool_sizes(&sizes)
					.map_err(Error::from)?,
				None,
			)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "UI raster descriptor pool", e))?;
		let mut result = Self {
			device: device.clone(),
			_engine: engine.clone(),
			pool,
			sets: Vec::new(),
			layouts: bindings
				.iter()
				.map(|binding| match binding {
					CmdBinding::Buffers(buffers) => layouts.pipelines[buffers.len()],
					CmdBinding::Image { .. } => layouts.blit_pipeline,
				})
				.collect(),
		};
		if count == 0 {
			return Ok(result);
		}
		let descriptors: Vec<_> = bindings
			.iter()
			.map(|binding| match binding {
				CmdBinding::Buffers(buffers) => layouts.descriptors[buffers.len()],
				CmdBinding::Image { .. } => layouts.blit_descriptors,
			})
			.collect();
		let mut sets = vec![vk::DescriptorSet::default(); descriptors.len()];
		// SAFETY: layouts belong to device and the new pool has exact capacity.
		unsafe {
			device.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::new()
					.set_descriptor_pool(pool)
					.set_set_layouts(&descriptors)
					.map_err(Error::from)?,
				&mut sets,
			)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "UI raster descriptor allocation", e))?;
		result.sets = sets;
		for (&set, binding) in result.sets.iter().zip(&bindings) {
			match binding {
				CmdBinding::Buffers(buffers) => {
					if buffers.is_empty() {
						continue;
					}
					let writes = [vk::WriteDescriptorSet::new()
						.set_dst_set(set)
						.set_dst_binding(0)
						.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.set_buffer_info(buffers)
						.map_err(Error::from)?];
					// SAFETY: fresh unsubmitted sets, live retained sources,
					// exact ranges.
					unsafe {
						device.update_descriptor_sets(&writes, &[])?;
					}
				}
				CmdBinding::Image { view, sampler } => {
					let image = vk::DescriptorImageInfo::new()
						.set_sampler(*sampler)
						.set_image_view(*view)
						.set_image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
					let writes = [vk::WriteDescriptorSet::new()
						.set_dst_set(set)
						.set_dst_binding(0)
						.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
						.set_image_info(std::slice::from_ref(&image))
						.map_err(Error::from)?];
					// SAFETY: fresh unsubmitted set; the source Texture is
					// retained in frame_textures through submission.
					unsafe {
						device.update_descriptor_sets(&writes, &[])?;
					}
				}
			}
		}
		Ok(result)
	}
}

impl Drop for UiFrameBindings {
	fn drop(&mut self) {
		// SAFETY: exact presentation/readback completion precedes final owner release.
		let _ = unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
	}
}
