//! UI compute pipelines — one per compositor command family.
//!
//! Each pipeline uses the engine's shared bindless layout so the same
//! descriptor set (heap[] + images[]) is bound once per command buffer.
//!
//! Donor reference: `oa::Ui::initBlit` in `ui.cpp`

use crate::{Error, Result};

// Embedded compiled SPIR-V — one per UI shader.
static CLEAR_COMPOSE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_clear_compose.spv"));
static BLIT_RGBA_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_blit_rgba.spv"));
static DRAW_RECT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect.spv"));
static DRAW_RECT_OUTLINE_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect_outline.spv"));
static DRAW_LINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_line.spv"));
static DRAW_PLOT_LINE_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_plot_line.spv"));
static DRAW_GLYPHS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_glyphs.spv"));

/// All six UI compute pipelines owned by one `Ui` instance.
pub(super) struct UiPipelines {
	pub(super) clear_compose: UiPipeline,
	pub(super) blit_rgba: UiPipeline,
	pub(super) draw_rect: UiPipeline,
	pub(super) draw_rect_outline: UiPipeline,
	pub(super) draw_line: UiPipeline,
	pub(super) draw_plot_line: UiPipeline,
	pub(super) draw_glyphs: UiPipeline,
}

impl UiPipelines {
	pub(super) fn new(device: &ash::Device, layout: ash::vk::PipelineLayout) -> Result<Self> {
		Ok(Self {
			clear_compose: UiPipeline::new(device, layout, CLEAR_COMPOSE_SPV)?,
			blit_rgba: UiPipeline::new(device, layout, BLIT_RGBA_SPV)?,
			draw_rect: UiPipeline::new(device, layout, DRAW_RECT_SPV)?,
			draw_rect_outline: UiPipeline::new(device, layout, DRAW_RECT_OUTLINE_SPV)?,
			draw_line: UiPipeline::new(device, layout, DRAW_LINE_SPV)?,
			draw_plot_line: UiPipeline::new(device, layout, DRAW_PLOT_LINE_SPV)?,
			draw_glyphs: UiPipeline::new(device, layout, DRAW_GLYPHS_SPV)?,
		})
	}
}

/// One compute pipeline with a shared layout.
pub(super) struct UiPipeline {
	handle: ash::vk::Pipeline,
	device: ash::Device,
}

impl UiPipeline {
	fn new(device: &ash::Device, layout: ash::vk::PipelineLayout, spv: &[u8]) -> Result<Self> {
		let words = decode_spirv(spv)?;
		let module = create_shader_module(device, &words)?;
		let entry = std::ffi::CString::new("main").unwrap();
		let stage = ash::vk::PipelineShaderStageCreateInfo::default()
			.stage(ash::vk::ShaderStageFlags::COMPUTE)
			.module(module)
			.name(&entry);
		let create_info = ash::vk::ComputePipelineCreateInfo::default()
			.stage(stage)
			.layout(layout);
		let result = unsafe {
			device.create_compute_pipelines(
				ash::vk::PipelineCache::null(),
				std::slice::from_ref(&create_info),
				None,
			)
		};
		unsafe { device.destroy_shader_module(module, None) };
		let pipelines = result.map_err(|(_, source)| {
			Error::backend_failure("Vulkan", "UI compute pipeline creation", source)
		})?;
		let handle = pipelines[0];
		Ok(Self {
			handle,
			device: device.clone(),
		})
	}

	pub(super) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}
}

impl Drop for UiPipeline {
	fn drop(&mut self) {
		// SAFETY: pipeline belongs to the device; layout is external and must be kept alive.
		unsafe { self.device.destroy_pipeline(self.handle, None) };
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

fn create_shader_module(device: &ash::Device, words: &[u32]) -> Result<ash::vk::ShaderModule> {
	let info = ash::vk::ShaderModuleCreateInfo::default().code(words);
	unsafe { device.create_shader_module(&info, None) }
		.map_err(|source| Error::backend_failure("Vulkan", "UI shader module creation", source))
}
