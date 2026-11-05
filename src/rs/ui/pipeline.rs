//! UI compute pipelines — one per compositor command family.
//!
//! Strict uses the shared Engine heap. Compatibility uses immutable per-command
//! sets with one storage image and the exact zero, one or two source buffers.
//!
//! Donor reference: `oa::Ui::initBlit` in `ui.cpp`

use std::sync::Arc;

use super::command::UiCmd;
use crate::{Error, Result};

// Embedded compiled SPIR-V — one per UI shader.
static CLEAR_COMPOSE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_clear_compose.spv"));
static BLIT_RGBA_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_blit_rgba.spv"));
static BLIT_PLANAR_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_blit_planar.spv"));
static DRAW_RECT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect.spv"));
static DRAW_RECT_OUTLINE_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect_outline.spv"));
static DRAW_LINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_line.spv"));
static DRAW_PLOT_LINE_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_plot_line.spv"));
static DRAW_GLYPHS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_glyphs.spv"));
static DRAW_RECT_OUTLINES_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect_outlines.spv"));

static CLEAR_COMPOSE_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_clear_compose_bounded.spv"));

static BLIT_RGBA_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_blit_rgba_bounded.spv"));

static BLIT_PLANAR_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_blit_planar_bounded.spv"));

static DRAW_RECT_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_rect_bounded.spv"));

static DRAW_RECT_OUTLINE_BOUNDED_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/ui_draw_rect_outline_bounded.spv"
));

static DRAW_LINE_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_line_bounded.spv"));

static DRAW_PLOT_LINE_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_plot_line_bounded.spv"));

static DRAW_GLYPHS_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_glyphs_bounded.spv"));
static DRAW_RECT_OUTLINES_BOUNDED_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/ui_draw_rect_outlines_bounded.spv"
));

static DRAW_CANVAS_GRID_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_canvas_grid.spv"));
static DRAW_CANVAS_GRID_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_canvas_grid_bounded.spv"));

static DRAW_TRIANGLE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_triangle.spv"));
static DRAW_TRIANGLE_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_triangle_bounded.spv"));

static MASK_ROUNDED_VIEWPORT_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_mask_rounded_viewport.spv"));
static MASK_ROUNDED_VIEWPORT_BOUNDED_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/ui_mask_rounded_viewport_bounded.spv"
));

static DRAW_WAVEFORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_waveform.spv"));
static DRAW_WAVEFORM_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_waveform_bounded.spv"));
static DRAW_HEATMAP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_heatmap.spv"));
static DRAW_HEATMAP_BOUNDED_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/ui_draw_heatmap_bounded.spv"));

/// All fourteen UI compute pipelines owned by one `Ui` instance.
pub(super) struct UiPipelines {
	pub(super) clear_compose: UiPipeline,
	pub(super) blit_rgba: UiPipeline,
	pub(super) blit_planar: UiPipeline,
	pub(super) draw_rect: UiPipeline,
	pub(super) draw_rect_outline: UiPipeline,
	pub(super) draw_line: UiPipeline,
	pub(super) draw_plot_line: UiPipeline,
	pub(super) draw_glyphs: UiPipeline,
	pub(super) draw_rect_outlines: UiPipeline,
	pub(super) draw_canvas_grid: UiPipeline,
	pub(super) draw_triangle: UiPipeline,
	pub(super) mask_rounded_viewport: UiPipeline,
	pub(super) draw_waveform: UiPipeline,
	pub(super) draw_heatmap: UiPipeline,
	pub(super) bounded: Option<UiLayouts>,
}

impl UiPipelines {
	pub(super) fn new(
		device: &Arc<vk::Device>,
		layout: vk::PipelineLayout,
		bounded: bool,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		let layouts = if bounded {
			Some(UiLayouts::new(device)?)
		} else {
			None
		};
		let select = |count: usize| {
			layouts
				.as_ref()
				.map_or(layout, |layouts| layouts.pipelines[count])
		};
		Ok(Self {
			clear_compose: UiPipeline::new(
				device,
				select(0),
				if bounded {
					CLEAR_COMPOSE_BOUNDED_SPV
				} else {
					CLEAR_COMPOSE_SPV
				},
				cache,
			)?,
			blit_rgba: UiPipeline::new(
				device,
				select(1),
				if bounded {
					BLIT_RGBA_BOUNDED_SPV
				} else {
					BLIT_RGBA_SPV
				},
				cache,
			)?,
			blit_planar: UiPipeline::new(
				device,
				select(4),
				if bounded {
					BLIT_PLANAR_BOUNDED_SPV
				} else {
					BLIT_PLANAR_SPV
				},
				cache,
			)?,
			draw_rect: UiPipeline::new(
				device,
				select(0),
				if bounded {
					DRAW_RECT_BOUNDED_SPV
				} else {
					DRAW_RECT_SPV
				},
				cache,
			)?,
			draw_rect_outline: UiPipeline::new(
				device,
				select(0),
				if bounded {
					DRAW_RECT_OUTLINE_BOUNDED_SPV
				} else {
					DRAW_RECT_OUTLINE_SPV
				},
				cache,
			)?,
			draw_line: UiPipeline::new(
				device,
				select(0),
				if bounded {
					DRAW_LINE_BOUNDED_SPV
				} else {
					DRAW_LINE_SPV
				},
				cache,
			)?,
			draw_plot_line: UiPipeline::new(
				device,
				select(1),
				if bounded {
					DRAW_PLOT_LINE_BOUNDED_SPV
				} else {
					DRAW_PLOT_LINE_SPV
				},
				cache,
			)?,
			draw_glyphs: UiPipeline::new(
				device,
				select(2),
				if bounded {
					DRAW_GLYPHS_BOUNDED_SPV
				} else {
					DRAW_GLYPHS_SPV
				},
				cache,
			)?,
			draw_rect_outlines: UiPipeline::new(
				device,
				select(1),
				if bounded {
					DRAW_RECT_OUTLINES_BOUNDED_SPV
				} else {
					DRAW_RECT_OUTLINES_SPV
				},
				cache,
			)?,
			draw_canvas_grid: UiPipeline::new(
				device,
				select(0),
				if bounded {
					DRAW_CANVAS_GRID_BOUNDED_SPV
				} else {
					DRAW_CANVAS_GRID_SPV
				},
				cache,
			)?,
			draw_triangle: UiPipeline::new(
				device,
				select(0),
				if bounded {
					DRAW_TRIANGLE_BOUNDED_SPV
				} else {
					DRAW_TRIANGLE_SPV
				},
				cache,
			)?,
			mask_rounded_viewport: UiPipeline::new(
				device,
				select(0),
				if bounded {
					MASK_ROUNDED_VIEWPORT_BOUNDED_SPV
				} else {
					MASK_ROUNDED_VIEWPORT_SPV
				},
				cache,
			)?,
			draw_waveform: UiPipeline::new(
				device,
				select(1),
				if bounded {
					DRAW_WAVEFORM_BOUNDED_SPV
				} else {
					DRAW_WAVEFORM_SPV
				},
				cache,
			)?,
			draw_heatmap: UiPipeline::new(
				device,
				select(1),
				if bounded {
					DRAW_HEATMAP_BOUNDED_SPV
				} else {
					DRAW_HEATMAP_SPV
				},
				cache,
			)?,
			bounded: layouts,
		})
	}
}

/// One compute pipeline with a shared layout.
pub(super) struct UiPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: Arc<vk::Device>,
}

impl UiPipeline {
	fn new(
		device: &Arc<vk::Device>,
		layout: vk::PipelineLayout,
		spv: &[u8],
		cache: vk::PipelineCache,
	) -> Result<Self> {
		let words = decode_spirv(spv)?;
		let module = create_shader_module(device, &words)?;
		let entry = c"main";
		let stage = vk::PipelineShaderStageCreateInfo::new()
			.set_stage(vk::ShaderStageFlagBits::COMPUTE_BIT)
			.set_module(module)
			.set_name(entry);
		let create_info = vk::ComputePipelineCreateInfo::new()
			.set_stage(stage)
			.set_layout(layout);
		let mut pipelines = [vk::Pipeline::default(); 1];
		// SAFETY: the create-info and output slot remain live and writable for the
		// call; partial output is destroyed on failure below.
		let result = unsafe {
			device.create_compute_pipelines(
				cache,
				std::slice::from_ref(&create_info),
				None,
				&mut pipelines,
			)
		};
		let _ = unsafe { device.destroy_shader_module(module, None) };
		let handle = match result {
			Ok(vk::Result::SUCCESS) => pipelines[0],
			result => {
				// SAFETY: partial results were never recorded or submitted.
				for pipeline in pipelines
					.into_iter()
					.filter(|handle| *handle != vk::Pipeline::default())
				{
					let _ = unsafe { device.destroy_pipeline(pipeline, None) };
				}
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"UI compute pipeline creation",
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

/// Exact fixed descriptor layouts: one image and zero, one or two buffers.
/// Field order in UiPipelines retires pipelines before their layouts.
pub(super) struct UiLayouts {
	device: Arc<vk::Device>,
	descriptors: Vec<vk::DescriptorSetLayout>,
	pipelines: Vec<vk::PipelineLayout>,
}

impl UiLayouts {
	fn new(device: &Arc<vk::Device>) -> Result<Self> {
		let mut result = Self {
			device: device.clone(),
			descriptors: Vec::new(),
			pipelines: Vec::new(),
		};
		for count in 0..=4 {
			let mut bindings = vec![
				vk::DescriptorSetLayoutBinding::new()
					.set_binding(1)
					.set_descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
					.set_descriptor_count(1)
					.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT),
			];
			if count != 0 {
				bindings.push(
					vk::DescriptorSetLayoutBinding::new()
						.set_binding(0)
						.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.set_descriptor_count(count)
						.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT),
				);
			}
			// SAFETY: fixed counts match the bounded artifacts; no indexing flags.
			let descriptor = unsafe {
				device.create_descriptor_set_layout(
					&(vk::DescriptorSetLayoutCreateInfo::new()
						.set_bindings(&bindings)
						.map_err(Error::from)?),
					None,
				)
			}
			.map_err(|e| Error::backend_failure("Vulkan", "bounded UI descriptor layout", e))?;
			result.descriptors.push(descriptor);
			let descriptors = [descriptor];
			let push = [vk::PushConstantRange::new()
				.set_size(128)
				.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT)];
			// SAFETY: the descriptor layout is retained by result and push range is legal.
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
			.map_err(|e| Error::backend_failure("Vulkan", "bounded UI pipeline layout", e))?;
			result.pipelines.push(pipeline);
		}
		Ok(result)
	}
}

impl Drop for UiLayouts {
	fn drop(&mut self) {
		// SAFETY: pipelines and submitted descriptor owners have retired first.
		unsafe {
			for layout in &self.pipelines {
				let _ = self.device.destroy_pipeline_layout(*layout, None);
			}
			for layout in &self.descriptors {
				let _ = self.device.destroy_descriptor_set_layout(*layout, None);
			}
		}
	}
}

/// Immutable descriptor pool and locally remapped commands for one composition.
/// The presentation slot or exact readback completion retains this allocation.
pub(super) struct UiFrameBindings {
	device: Arc<vk::Device>,
	_engine: crate::runtime::EngineHandle,
	pool: vk::DescriptorPool,
	pub(super) sets: Vec<vk::DescriptorSet>,
	pub(super) commands: Vec<UiCmd>,
	pub(super) layouts: Vec<vk::PipelineLayout>,
}

impl UiFrameBindings {
	pub(super) fn new(
		engine: &crate::runtime::EngineHandle,
		layouts: &UiLayouts,
		image: vk::ImageView,
		commands: &[UiCmd],
		resolve: impl Fn(u32) -> Result<vk::DescriptorBufferInfo>,
	) -> Result<Self> {
		let device = engine.raw_device();
		let mut buffers = Vec::with_capacity(commands.len());
		let mut remapped = commands.to_vec();
		for command in &mut remapped {
			let indices = match command {
				UiCmd::BlitRgba { src_idx, .. } => {
					let index = *src_idx;
					*src_idx = 0;
					vec![index]
				}
				UiCmd::BlitPlanar {
					r_idx,
					g_idx,
					b_idx,
					a_idx,
					..
				} => {
					// The bounded artifact declares four buffer slots; absent
					// channels bind the R plane and are never read.
					let source = *r_idx;
					let resolve = |index: u32| if index == u32::MAX { source } else { index };
					let indices = vec![
						resolve(*r_idx),
						resolve(*g_idx),
						resolve(*b_idx),
						resolve(*a_idx),
					];
					*r_idx = 0;
					*g_idx = 1;
					*b_idx = 2;
					*a_idx = 3;
					indices
				}
				UiCmd::PlotLine { values_idx, .. }
				| UiCmd::Waveform { values_idx, .. }
				| UiCmd::Heatmap { values_idx, .. } => {
					let index = *values_idx;
					*values_idx = 0;
					vec![index]
				}
				UiCmd::Glyphs {
					glyph_idx,
					atlas_idx,
					..
				} => {
					let indices = vec![*glyph_idx, *atlas_idx];
					*glyph_idx = 0;
					*atlas_idx = 1;
					indices
				}
				UiCmd::RectOutlines { rect_idx, .. } => {
					let index = *rect_idx;
					*rect_idx = 0;
					vec![index]
				}
				_ => Vec::new(),
			};
			buffers.push(
				indices
					.into_iter()
					.map(&resolve)
					.collect::<Result<Vec<_>>>()?,
			);
		}
		let count = u32::try_from(commands.len())
			.map_err(|_| Error::resource_exhausted("too many UI commands"))?;
		let buffer_count = buffers.iter().try_fold(0_u32, |total, buffer| {
			total
				.checked_add(buffer.len() as u32)
				.ok_or_else(|| Error::resource_exhausted("UI descriptor count overflow"))
		})?;
		let mut sizes = vec![
			vk::DescriptorPoolSize::new()
				.set_type(vk::DescriptorType::STORAGE_IMAGE)
				.set_descriptor_count(count.max(1)),
		];
		if buffer_count != 0 {
			sizes.push(
				vk::DescriptorPoolSize::new()
					.set_type(vk::DescriptorType::STORAGE_BUFFER)
					.set_descriptor_count(buffer_count),
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
		.map_err(|e| Error::backend_failure("Vulkan", "bounded UI descriptor pool", e))?;
		let mut result = Self {
			device: device.clone(),
			_engine: engine.clone(),
			pool,
			sets: Vec::new(),
			commands: remapped,
			layouts: buffers.iter().map(|b| layouts.pipelines[b.len()]).collect(),
		};
		if count == 0 {
			return Ok(result);
		}
		let descriptors: Vec<_> = buffers
			.iter()
			.map(|b| layouts.descriptors[b.len()])
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
		.map_err(|e| Error::backend_failure("Vulkan", "bounded UI descriptor allocation", e))?;
		result.sets = sets;
		let images = [vk::DescriptorImageInfo::new()
			.set_image_view(image)
			.set_image_layout(vk::ImageLayout::GENERAL)];
		for (&set, buffers) in result.sets.iter().zip(&buffers) {
			let mut writes = vec![
				vk::WriteDescriptorSet::new()
					.set_dst_set(set)
					.set_dst_binding(1)
					.set_descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
					.set_image_info(&images)
					.map_err(Error::from)?,
			];
			if !buffers.is_empty() {
				writes.push(
					vk::WriteDescriptorSet::new()
						.set_dst_set(set)
						.set_dst_binding(0)
						.set_descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
						.set_buffer_info(buffers)
						.map_err(Error::from)?,
				);
			}
			// SAFETY: fresh unsubmitted sets, live retained sources, exact ranges.
			unsafe {
				device.update_descriptor_sets(&writes, &[])?;
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
