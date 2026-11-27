//! Per-command descriptor bindings for the raster compositor.
//!
//! Each command's set binds its source buffers in literal `heap[k]` order —
//! the raster shaders index slots directly, so no command remapping is
//! required. The presentation slot or exact readback completion retains the
//! allocation.

use std::sync::Arc;

use crate::runtime::UiLayouts;
use crate::{Error, Result};

/// Immutable per-command descriptor sets for one composition.
///
/// Each command's set binds its source buffers in literal `heap[k]` order —
/// the raster shaders index slots directly, so no command remapping is
/// required. The presentation slot or exact readback completion retains this
/// allocation.
pub(crate) struct UiFrameBindings {
	device: Arc<vk::Device>,
	_runtime: crate::runtime::RuntimeHandle,
	pool: vk::DescriptorPool,
	pub(crate) sets: Vec<vk::DescriptorSet>,
	pub(crate) layouts: Vec<vk::PipelineLayout>,
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
	pub(crate) fn new(
		runtime: &crate::runtime::RuntimeHandle,
		layouts: &UiLayouts,
		commands: &[super::command::UiCmd],
		resolve: impl Fn(u32) -> Result<vk::DescriptorBufferInfo>,
	) -> Result<Self> {
		use super::command::UiCmd;
		let device = runtime.raw_device();
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
				UiCmd::BlitYuv { planes_idx, .. } => vec![*planes_idx],
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
			_runtime: runtime.clone(),
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
