//! UI raster pipelines — one graphics pipeline per compositor command family.
//!
//! Every element family draws into the compose color attachment inside one
//! dynamic-rendering pass. Per-command descriptor sets bind the family's
//! fixed storage-buffer slots (binding 0, count = the family's `heap[N]`
//! declaration) for both execution profiles — the raster compositor never
//! uses the Strict bindless heap or a storage image.

use std::sync::Arc;

use crate::{Error, Result};

use super::render::{CANVAS_GRID_FRAG_SPV, CANVAS_GRID_VERT_SPV};
use super::{
	GraphicsBlend, GraphicsDepth, GraphicsLayoutSource, GraphicsPipeline, GraphicsPipelineDesc,
	GraphicsVertexInput,
};

// Embedded compiled SPIR-V — vertex stages shared per shape family and one
// fragment stage per element family.
static FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_fullscreen_vert.spv"
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
static BLIT_YUV_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_raster_blit_yuv_frag.spv"
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

/// All fifteen UI raster pipelines owned by one `Ui` instance.
///
/// `UiPipelines` is placed before `layouts` so every pipeline retires before
/// the layouts they were created against.
pub(crate) struct UiPipelines {
	pub(crate) clear: GraphicsPipeline,
	pub(crate) rect: GraphicsPipeline,
	pub(crate) rect_outline: GraphicsPipeline,
	pub(crate) line: GraphicsPipeline,
	pub(crate) triangle: GraphicsPipeline,
	pub(crate) canvas_grid: GraphicsPipeline,
	pub(crate) blit_texture: GraphicsPipeline,
	pub(crate) blit_planar: GraphicsPipeline,
	pub(crate) blit_yuv: GraphicsPipeline,
	pub(crate) heatmap: GraphicsPipeline,
	pub(crate) waveform: GraphicsPipeline,
	pub(crate) plot_line: GraphicsPipeline,
	pub(crate) glyphs: GraphicsPipeline,
	pub(crate) rect_outlines: GraphicsPipeline,
	pub(crate) mask_modulate: GraphicsPipeline,
	pub(crate) mask_additive: GraphicsPipeline,
	pub(crate) layouts: UiLayouts,
}

impl UiPipelines {
	pub(crate) fn new(device: &Arc<vk::Device>, cache: vk::PipelineCache) -> Result<Self> {
		let layouts = UiLayouts::new(device)?;
		let family = |count: usize, vert: &'static [u8], frag: &'static [u8], blend: GraphicsBlend| {
			ui_raster_pipeline(device, layouts.pipelines[count], vert, frag, blend, cache)
		};
		Ok(Self {
			clear: family(
				0,
				FULLSCREEN_VERT_SPV,
				CLEAR_FRAG_SPV,
				GraphicsBlend::Opaque,
			)?,
			rect: family(
				0,
				FULLSCREEN_VERT_SPV,
				RECT_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			rect_outline: family(
				0,
				FULLSCREEN_VERT_SPV,
				RECT_OUTLINE_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			line: family(
				0,
				FULLSCREEN_VERT_SPV,
				LINE_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			triangle: family(
				0,
				FULLSCREEN_VERT_SPV,
				TRIANGLE_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			canvas_grid: family(
				0,
				CANVAS_GRID_VERT_SPV,
				CANVAS_GRID_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			blit_texture: ui_raster_pipeline(
				device,
				layouts.blit_pipeline,
				FULLSCREEN_VERT_SPV,
				BLIT_TEXTURE_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
				cache,
			)?,
			blit_planar: family(
				4,
				FULLSCREEN_VERT_SPV,
				BLIT_PLANAR_FRAG_SPV,
				GraphicsBlend::Opaque,
			)?,
			blit_yuv: family(
				1,
				FULLSCREEN_VERT_SPV,
				BLIT_YUV_FRAG_SPV,
				GraphicsBlend::Opaque,
			)?,
			heatmap: family(
				1,
				FULLSCREEN_VERT_SPV,
				HEATMAP_FRAG_SPV,
				GraphicsBlend::Opaque,
			)?,
			waveform: family(
				1,
				FULLSCREEN_VERT_SPV,
				WAVEFORM_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			plot_line: family(
				1,
				FULLSCREEN_VERT_SPV,
				PLOT_LINE_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			glyphs: family(
				2,
				GLYPHS_VERT_SPV,
				GLYPHS_FRAG_SPV,
				GraphicsBlend::PremultipliedAlpha,
			)?,
			rect_outlines: family(
				1,
				RECT_OUTLINES_VERT_SPV,
				RECT_OUTLINES_FRAG_SPV,
				GraphicsBlend::Opaque,
			)?,
			mask_modulate: family(
				0,
				FULLSCREEN_VERT_SPV,
				MASK_FRAG_SPV,
				GraphicsBlend::Modulate,
			)?,
			mask_additive: family(
				0,
				FULLSCREEN_VERT_SPV,
				MASK_FRAG_SPV,
				GraphicsBlend::Additive,
			)?,
			layouts,
		})
	}
}

/// Create one UI raster pass: a vertexless/instanced pipeline over a shared
/// compositor layout into the color-only compose attachment (no depth).
///
/// The layout is owned by [`UiLayouts`] and shared by every family in its
/// `heap[N]` class — the pipeline borrows it and never destroys it.
fn ui_raster_pipeline(
	device: &Arc<vk::Device>,
	layout: vk::PipelineLayout,
	vert_spv: &'static [u8],
	frag_spv: &'static [u8],
	blend: GraphicsBlend,
	cache: vk::PipelineCache,
) -> Result<GraphicsPipeline> {
	GraphicsPipeline::create(
		device,
		None,
		cache,
		&GraphicsPipelineDesc {
			name: "ui raster",
			vertex_spv: vert_spv,
			fragment_spv: frag_spv,
			layout: GraphicsLayoutSource::External(layout),
			vertex: GraphicsVertexInput::Vertexless,
			depth: GraphicsDepth::Disabled,
			blend,
			color_format: vk::Format::R8G8B8A8_UNORM,
			depth_format: vk::Format::UNDEFINED,
		},
	)
}
/// Fixed descriptor layouts for zero through four storage-buffer slots plus
/// the sampled-image blit layout.
///
/// Every raster shader declares `heap[N]` at binding 0; a family pipeline
/// layout binds the descriptor layout whose count matches its `N`. The image
/// blit declares `Sampler2D` at binding 0, so it owns a dedicated layout and
/// pipeline layout. The push range covers the largest family push block
/// (100 bytes) and is visible to both stages.
pub(crate) struct UiLayouts {
	device: Arc<vk::Device>,
	pub(crate) descriptors: Vec<vk::DescriptorSetLayout>,
	pub(crate) pipelines: Vec<vk::PipelineLayout>,
	pub(crate) blit_descriptors: vk::DescriptorSetLayout,
	pub(crate) blit_pipeline: vk::PipelineLayout,
	/// `[nearest, linear]` — bound per blit command by destination scale.
	pub(crate) samplers: [vk::Sampler; 2],
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
