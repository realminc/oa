//! Render-domain graphics pipelines and their descriptor layouts.
//!
//! Every render pass is a small [`GraphicsPipelineDesc`] returned by the
//! `*_pipeline_desc` functions below; [`crate::runtime::GraphicsPipeline`]
//! owns create-info assembly, shader-module lifetime, and failure cleanup.
//! Descriptor set layouts and per-frame arenas live beside the passes that
//! consume them.

use crate::{Error, Result};

use super::{
	GraphicsBlend, GraphicsDepth, GraphicsLayoutSource, GraphicsPipelineDesc, GraphicsPush,
	GraphicsVertexInput,
};

// Push-constant layout for the vertex-color lit graphics pipeline.
// Donor: oacpp/source/cpp/lib/oa/render/renderer3d.cpp `RenderPushConstants`.
// 64-byte row-major Mat4 + 16-byte Vec4 = 80 bytes total.
pub(crate) const GRAPHICS_PUSH_CONSTANT_SIZE: u32 = 80;

// Push size for flat-color: 64 (MVP) + 16 (material_color) + 16 (opacity+pad) = 96 bytes.
pub(crate) const FLAT_COLOR_PUSH_SIZE: u32 = 96;
// Push size for unlit: 64 (MVP) + 16 (tint) + 16 (uv_scale+uv_offset) = 96 bytes.
pub(crate) const UNLIT_PUSH_SIZE: u32 = 96;
// Push size for surface: 64 (MVP) + 64 (model matrix) = 128 bytes.
pub(crate) const SURFACE_PUSH_SIZE: u32 = 128;
/// Canvas-grid push size (100 bytes): grid parameters, fragment stage only.
pub(crate) const CANVAS_GRID_PUSH_SIZE: u32 = 100;

/// Embedded SPIR-V for the vertex-color lit vertex shader.
static RENDER_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_vertex_color_lit_vert.spv"
));
/// Embedded SPIR-V for the vertex-color lit fragment shader.
static RENDER_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_vertex_color_lit_frag.spv"
));

/// Embedded SPIR-V for the flat-color vertex shader.
static FLAT_COLOR_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_flat_color_vert.spv"
));
/// Embedded SPIR-V for the flat-color fragment shader.
static FLAT_COLOR_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_flat_color_frag.spv"
));

/// Embedded SPIR-V for the unlit vertex shader.
static UNLIT_VERT_SPV: &[u8] =
	include_bytes!(concat!(env!("OA_SHADER_DIR"), "/render_unlit_vert.spv"));
/// Embedded SPIR-V for the unlit fragment shader.
static UNLIT_FRAG_SPV: &[u8] =
	include_bytes!(concat!(env!("OA_SHADER_DIR"), "/render_unlit_frag.spv"));

/// Embedded SPIR-V for the surface vertex shader.
static SURFACE_VERT_SPV: &[u8] =
	include_bytes!(concat!(env!("OA_SHADER_DIR"), "/render_surface_vert.spv"));
/// Embedded SPIR-V for the surface fragment shader.
static SURFACE_FRAG_SPV: &[u8] =
	include_bytes!(concat!(env!("OA_SHADER_DIR"), "/render_surface_frag.spv"));

/// Embedded SPIR-V for the UI canvas-grid vertex shader.
pub(super) static CANVAS_GRID_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_canvas_grid_vert.spv"
));
/// Embedded SPIR-V for the UI canvas-grid fragment shader.
pub(super) static CANVAS_GRID_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OA_SHADER_DIR"),
	"/render_ui_canvas_grid_frag.spv"
));

/// R8G8B8A8_UNORM color attachment format used by all render targets.
pub(in crate::runtime) const RENDER_COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// D32_SFLOAT depth attachment format used by all render targets.
pub(in crate::runtime) const RENDER_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Vertex-color lit 3D renderer pipeline: 80-byte vertex push block
/// (row-major Mat4 view-projection + Vec4 light/ambient), no descriptor
/// sets, depth write+test, opaque.
pub(crate) fn lit_pipeline_desc() -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "vertex-color lit",
		vertex_spv: RENDER_VERT_SPV,
		fragment_spv: RENDER_FRAG_SPV,
		layout: GraphicsLayoutSource::Owned {
			set_layouts: Vec::new(),
			push: GraphicsPush {
				size: GRAPHICS_PUSH_CONSTANT_SIZE,
				stages: vk::ShaderStageFlagBits::VERTEX_BIT,
			},
		},
		vertex: GraphicsVertexInput::MeshVertexLit,
		depth: GraphicsDepth::WriteAndTest,
		blend: GraphicsBlend::Opaque,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}

/// Flat-color material pipeline (Tier 1 — UI and debug): 96-byte push block
/// (MVP + material_color + opacity + pad), no descriptor sets, no depth,
/// premultiplied alpha.
pub(crate) fn flat_color_pipeline_desc() -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "flat-color",
		vertex_spv: FLAT_COLOR_VERT_SPV,
		fragment_spv: FLAT_COLOR_FRAG_SPV,
		layout: GraphicsLayoutSource::Owned {
			set_layouts: Vec::new(),
			push: GraphicsPush {
				size: FLAT_COLOR_PUSH_SIZE,
				stages: vk::ShaderStageFlagBits::VERTEX_BIT | vk::ShaderStageFlagBits::FRAGMENT_BIT,
			},
		},
		vertex: GraphicsVertexInput::MeshVertex,
		depth: GraphicsDepth::Disabled,
		blend: GraphicsBlend::PremultipliedAlpha,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}

/// Unlit textured material pipeline (Tier 2 — image/video): 96-byte push
/// block (MVP + tint + uv_scale + uv_offset), descriptor set 0 base-color
/// sampler, no depth, premultiplied alpha.
pub(crate) fn unlit_pipeline_desc(sampler_layout: vk::DescriptorSetLayout) -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "unlit",
		vertex_spv: UNLIT_VERT_SPV,
		fragment_spv: UNLIT_FRAG_SPV,
		layout: GraphicsLayoutSource::Owned {
			set_layouts: vec![sampler_layout],
			push: GraphicsPush {
				size: UNLIT_PUSH_SIZE,
				stages: vk::ShaderStageFlagBits::VERTEX_BIT | vk::ShaderStageFlagBits::FRAGMENT_BIT,
			},
		},
		vertex: GraphicsVertexInput::MeshVertex,
		depth: GraphicsDepth::Disabled,
		blend: GraphicsBlend::PremultipliedAlpha,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}

/// Canvas-grid pipeline (raster compositor): vertexless fullscreen triangle,
/// 100-byte fragment push block carries all grid parameters, premultiplied
/// source-over blend.
pub(crate) fn canvas_grid_pipeline_desc() -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "canvas-grid",
		vertex_spv: CANVAS_GRID_VERT_SPV,
		fragment_spv: CANVAS_GRID_FRAG_SPV,
		layout: GraphicsLayoutSource::Owned {
			set_layouts: Vec::new(),
			push: GraphicsPush {
				size: CANVAS_GRID_PUSH_SIZE,
				stages: vk::ShaderStageFlagBits::FRAGMENT_BIT,
			},
		},
		vertex: GraphicsVertexInput::Vertexless,
		depth: GraphicsDepth::Disabled,
		blend: GraphicsBlend::PremultipliedAlpha,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}

/// Surface PBR opaque pipeline (Tier 3): 128-byte vertex push block
/// (MVP + model), set 0 scene-lights UBO, set 1 base-color/ORM/normal
/// samplers + material-params UBO, depth test/write, opaque.
pub(crate) fn surface_pipeline_desc(
	scene_layout: vk::DescriptorSetLayout,
	material_layout: vk::DescriptorSetLayout,
) -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "surface",
		vertex_spv: SURFACE_VERT_SPV,
		fragment_spv: SURFACE_FRAG_SPV,
		layout: GraphicsLayoutSource::Owned {
			set_layouts: vec![scene_layout, material_layout],
			push: GraphicsPush {
				size: SURFACE_PUSH_SIZE,
				stages: vk::ShaderStageFlagBits::VERTEX_BIT,
			},
		},
		vertex: GraphicsVertexInput::MeshVertex,
		depth: GraphicsDepth::WriteAndTest,
		blend: GraphicsBlend::Opaque,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}

/// Surface transparent pipeline: identical shaders and descriptor layout as
/// [`surface_pipeline_desc`] — borrows the opaque pipeline's layout — but
/// tests depth without writing it and blends premultiplied alpha for the
/// back-to-front transparent pass.
pub(crate) fn surface_blend_pipeline_desc(layout: vk::PipelineLayout) -> GraphicsPipelineDesc {
	GraphicsPipelineDesc {
		name: "surface-blend",
		vertex_spv: SURFACE_VERT_SPV,
		fragment_spv: SURFACE_FRAG_SPV,
		layout: GraphicsLayoutSource::External(layout),
		vertex: GraphicsVertexInput::MeshVertex,
		depth: GraphicsDepth::TestOnly,
		blend: GraphicsBlend::PremultipliedAlpha,
		color_format: RENDER_COLOR_FORMAT,
		depth_format: RENDER_DEPTH_FORMAT,
	}
}
// ── Descriptor set layouts ────────────────────────────────────────────────────

/// Owned descriptor set layout for the unlit textured pipeline (set 0).
///
/// Binding 0: combined-image-sampler (base-color texture), fragment stage.
pub(crate) struct UnlitDescriptorLayouts {
	/// Set 0 layout (base-color sampler).
	pub(crate) sampler_layout: vk::DescriptorSetLayout,
	device: std::sync::Arc<vk::Device>,
}

impl UnlitDescriptorLayouts {
	pub(in crate::runtime) fn new(device: std::sync::Arc<vk::Device>) -> Result<Self> {
		let sampler_binding = vk::DescriptorSetLayoutBinding::new()
			.set_binding(0)
			.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.set_descriptor_count(1)
			.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT);
		let bindings = [sampler_binding];
		let info = vk::DescriptorSetLayoutCreateInfo::new().set_bindings(&bindings);
		let layout =
			unsafe { device.create_descriptor_set_layout(&(info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "unlit descriptor-set-layout creation", s))?;
		Ok(Self {
			sampler_layout: layout,
			device,
		})
	}
}

impl Drop for UnlitDescriptorLayouts {
	fn drop(&mut self) {
		let _ = unsafe {
			self
				.device
				.destroy_descriptor_set_layout(self.sampler_layout, None)
		};
	}
}

/// Per-frame sampled-image descriptors for the unlit pipeline.
///
/// One descriptor set is allocated for each renderer target slot. A set is
/// updated only after that slot's producer has completed, so descriptors are
/// never mutated while a submitted draw can still consume them.
pub(crate) struct UnlitDescriptorArena {
	device: std::sync::Arc<vk::Device>,
	pool: vk::DescriptorPool,
	sampler: vk::Sampler,
	sets: Vec<vk::DescriptorSet>,
}

impl UnlitDescriptorArena {
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		layout: vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<Self> {
		let sampler_info = vk::SamplerCreateInfo::new()
			.set_mag_filter(vk::Filter::LINEAR)
			.set_min_filter(vk::Filter::LINEAR)
			.set_mipmap_mode(vk::SamplerMipmapMode::NEAREST)
			.set_address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.set_address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.set_address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.set_min_lod(0.0)
			.set_max_lod(0.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "unlit sampler creation", source))?;
		let pool_size = vk::DescriptorPoolSize::new()
			.set_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.set_descriptor_count(set_count);
		let pool_info = vk::DescriptorPoolCreateInfo::new()
			.set_max_sets(set_count)
			.set_pool_sizes(std::slice::from_ref(&pool_size));
		let pool =
			match unsafe { device.create_descriptor_pool(&(pool_info).map_err(Error::from)?, None) } {
				Ok(pool) => pool,
				Err(source) => {
					unsafe { device.destroy_sampler(sampler, None) }?;
					return Err(Error::backend_failure(
						"Vulkan",
						"unlit descriptor-pool creation",
						source,
					));
				}
			};
		let layouts = vec![layout; set_count as usize];
		let allocate_info = vk::DescriptorSetAllocateInfo::new()
			.set_descriptor_pool(pool)
			.set_set_layouts(&layouts);
		let allocate_info = allocate_info.map_err(Error::from)?;
		let mut sets = vec![vk::DescriptorSet::default(); layouts.len()];
		let sets = match unsafe { device.allocate_descriptor_sets(&allocate_info, &mut sets) } {
			Ok(()) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None)?;
					device.destroy_sampler(sampler, None)?;
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"unlit descriptor-set allocation",
					source,
				));
			}
		};
		Ok(Self {
			device,
			pool,
			sampler,
			sets,
		})
	}

	pub(crate) fn write_sampled_image(
		&self,
		set_index: usize,
		view: vk::ImageView,
	) -> Result<vk::DescriptorSet> {
		let set = *self
			.sets
			.get(set_index)
			.ok_or_else(|| Error::internal("unlit descriptor-set index is out of range"))?;
		let image = vk::DescriptorImageInfo::new()
			.set_sampler(self.sampler)
			.set_image_view(view)
			.set_image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
		let write = vk::WriteDescriptorSet::new()
			.set_dst_set(set)
			.set_dst_binding(0)
			.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.set_image_info(std::slice::from_ref(&image));
		unsafe {
			self
				.device
				.update_descriptor_sets(&[(write).map_err(Error::from)?], &[])
		}?;
		Ok(set)
	}
}

impl Drop for UnlitDescriptorArena {
	fn drop(&mut self) {
		unsafe {
			let _ = self.device.destroy_descriptor_pool(self.pool, None);
			let _ = self.device.destroy_sampler(self.sampler, None);
		}
	}
}

/// Owned descriptor set layouts for the Surface PBR pipeline.
///
/// Set 0, binding 0: scene lights UBO (`SceneLights`), fragment stage.
/// Set 1, binding 0: base-color combined-image-sampler, fragment stage.
/// Set 1, binding 1: ORM combined-image-sampler, fragment stage.
/// Set 1, binding 2: normal combined-image-sampler, fragment stage.
/// Set 1, binding 3: material params UBO (`MaterialParams`), fragment stage.
pub(crate) struct SurfaceDescriptorLayouts {
	/// Set 0 layout (scene lights UBO).
	pub(crate) scene_layout: vk::DescriptorSetLayout,
	/// Set 1 layout (3 samplers + material params UBO).
	pub(crate) material_layout: vk::DescriptorSetLayout,
	device: std::sync::Arc<vk::Device>,
}

impl SurfaceDescriptorLayouts {
	pub(in crate::runtime) fn new(device: std::sync::Arc<vk::Device>) -> Result<Self> {
		// Set 0: scene lights UBO.
		let scene_binding = vk::DescriptorSetLayoutBinding::new()
			.set_binding(0)
			.set_descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
			.set_descriptor_count(1)
			.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT);
		let scene_bindings = [scene_binding];
		let scene_info = vk::DescriptorSetLayoutCreateInfo::new().set_bindings(&scene_bindings);
		let scene_layout =
			unsafe { device.create_descriptor_set_layout(&(scene_info).map_err(Error::from)?, None) }
				.map_err(|s| {
					Error::backend_failure("Vulkan", "surface scene descriptor-set-layout creation", s)
				})?;

		// Set 1: base-color / ORM / normal samplers + material params UBO.
		let mat_bindings = [
			vk::DescriptorSetLayoutBinding::new()
				.set_binding(0)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_descriptor_count(1)
				.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT),
			vk::DescriptorSetLayoutBinding::new()
				.set_binding(1)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_descriptor_count(1)
				.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT),
			vk::DescriptorSetLayoutBinding::new()
				.set_binding(2)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_descriptor_count(1)
				.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT),
			vk::DescriptorSetLayoutBinding::new()
				.set_binding(3)
				.set_descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.set_descriptor_count(1)
				.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT),
		];
		let mat_info = vk::DescriptorSetLayoutCreateInfo::new().set_bindings(&mat_bindings);
		let material_layout =
			match unsafe { device.create_descriptor_set_layout(&(mat_info).map_err(Error::from)?, None) }
			{
				Ok(l) => l,
				Err(s) => {
					unsafe { device.destroy_descriptor_set_layout(scene_layout, None) }?;
					return Err(Error::backend_failure(
						"Vulkan",
						"surface material descriptor-set-layout creation",
						s,
					));
				}
			};

		Ok(Self {
			scene_layout,
			material_layout,
			device,
		})
	}
}

impl Drop for SurfaceDescriptorLayouts {
	fn drop(&mut self) {
		unsafe {
			let _ = self
				.device
				.destroy_descriptor_set_layout(self.material_layout, None);
			let _ = self
				.device
				.destroy_descriptor_set_layout(self.scene_layout, None);
		}
	}
}

/// Per-frame uniform + sampled-image descriptors for the Surface pipeline.
///
/// Allocates `slot_count` scene sets (set 0) and
/// `slot_count × max_materials_per_slot` material sets (set 1). Owns a shared
/// linear sampler used for all texture bindings; callers supply a white-texture
/// fallback view when a material has no texture assigned.
pub(crate) struct SurfaceDescriptorArena {
	device: std::sync::Arc<vk::Device>,
	pool: vk::DescriptorPool,
	sampler: vk::Sampler,
	scene_sets: Vec<vk::DescriptorSet>,
	/// Flat `[slot_count × max_materials_per_slot]` material sets.
	material_sets: Vec<vk::DescriptorSet>,
	/// Number of material set slots per render slot.
	max_materials: usize,
}

impl SurfaceDescriptorArena {
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		scene_layout: vk::DescriptorSetLayout,
		material_layout: vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<Self> {
		let max_materials_per_slot = max_materials_per_slot.max(1);
		let sampler_info = vk::SamplerCreateInfo::new()
			.set_mag_filter(vk::Filter::LINEAR)
			.set_min_filter(vk::Filter::LINEAR)
			.set_mipmap_mode(vk::SamplerMipmapMode::NEAREST)
			.set_address_mode_u(vk::SamplerAddressMode::REPEAT)
			.set_address_mode_v(vk::SamplerAddressMode::REPEAT)
			.set_address_mode_w(vk::SamplerAddressMode::REPEAT)
			.set_min_lod(0.0)
			.set_max_lod(0.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "surface sampler creation", source))?;
		let mat_set_total = set_count.saturating_mul(max_materials_per_slot);
		let pool_sizes = [
			vk::DescriptorPoolSize::new()
				.set_type(vk::DescriptorType::UNIFORM_BUFFER)
				// 1 scene UBO per slot + 1 material UBO per material set
				.set_descriptor_count(set_count.saturating_add(mat_set_total)),
			vk::DescriptorPoolSize::new()
				.set_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				// 3 texture bindings per material set
				.set_descriptor_count(mat_set_total.saturating_mul(3)),
		];
		let pool_info = vk::DescriptorPoolCreateInfo::new()
			.set_max_sets(set_count.saturating_add(mat_set_total))
			.set_pool_sizes(&pool_sizes);
		let pool =
			match unsafe { device.create_descriptor_pool(&(pool_info).map_err(Error::from)?, None) } {
				Ok(p) => p,
				Err(source) => {
					unsafe { device.destroy_sampler(sampler, None) }?;
					return Err(Error::backend_failure(
						"Vulkan",
						"surface descriptor-pool creation",
						source,
					));
				}
			};
		let allocate_n = |layout: vk::DescriptorSetLayout, n: u32| -> Result<Vec<vk::DescriptorSet>> {
			let layouts = vec![layout; n as usize];
			let info = vk::DescriptorSetAllocateInfo::new()
				.set_descriptor_pool(pool)
				.set_set_layouts(&layouts)
				.map_err(Error::from)?;
			let mut sets = vec![vk::DescriptorSet::default(); layouts.len()];
			unsafe { device.allocate_descriptor_sets(&info, &mut sets) }.map_err(Error::from)?;
			Ok(sets)
		};
		let scene_sets = match allocate_n(scene_layout, set_count) {
			Ok(sets) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None)?;
					device.destroy_sampler(sampler, None)?;
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"surface scene descriptor allocation",
					source,
				));
			}
		};
		let material_sets = match allocate_n(material_layout, mat_set_total) {
			Ok(sets) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None)?;
					device.destroy_sampler(sampler, None)?;
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"surface material descriptor allocation",
					source,
				));
			}
		};
		Ok(Self {
			device,
			pool,
			sampler,
			scene_sets,
			material_sets,
			max_materials: max_materials_per_slot as usize,
		})
	}

	/// Write the scene-lights UBO into set 0 for `slot_index`.
	pub(crate) fn write_scene_slot(
		&self,
		slot_index: usize,
		scene_buffer: vk::Buffer,
		scene_size: u64,
	) -> Result<vk::DescriptorSet> {
		let scene_set = *self
			.scene_sets
			.get(slot_index)
			.ok_or_else(|| Error::internal("surface scene set index is out of range"))?;
		let scene_info = vk::DescriptorBufferInfo::new()
			.set_buffer(scene_buffer)
			.set_range(scene_size);
		let write = vk::WriteDescriptorSet::new()
			.set_dst_set(scene_set)
			.set_dst_binding(0)
			.set_descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
			.set_buffer_info(std::slice::from_ref(&scene_info));
		unsafe {
			self
				.device
				.update_descriptor_sets(&[(write).map_err(Error::from)?], &[])
		}?;
		Ok(scene_set)
	}

	/// Write one material set (set 1) at `(slot_index, material_index)`.
	///
	/// Bindings:
	/// - 0: base-color sampler (`base_view`, or `white_view` when `None`)
	/// - 1: ORM sampler (`orm_view`, or `white_view` when `None`)
	/// - 2: normal sampler (`normal_view`, or `white_view` when `None`)
	/// - 3: material params UBO
	///
	/// The caller is responsible for ensuring that `material_index` is within
	/// `[0, max_materials_per_slot)`.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn write_material_slot(
		&self,
		slot_index: usize,
		material_index: usize,
		material_buffer: vk::Buffer,
		material_size: u64,
		base_view: Option<vk::ImageView>,
		orm_view: Option<vk::ImageView>,
		normal_view: Option<vk::ImageView>,
		white_view: vk::ImageView,
	) -> Result<vk::DescriptorSet> {
		let flat = slot_index * self.max_materials + material_index;
		let mat_set = *self
			.material_sets
			.get(flat)
			.ok_or_else(|| Error::internal("surface material set index is out of range"))?;
		let sampler_image = |view: Option<vk::ImageView>| {
			vk::DescriptorImageInfo::new()
				.set_sampler(self.sampler)
				.set_image_view(view.unwrap_or(white_view))
				.set_image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		};
		let base_img = sampler_image(base_view);
		let orm_img = sampler_image(orm_view);
		let normal_img = sampler_image(normal_view);
		let mat_buf_info = vk::DescriptorBufferInfo::new()
			.set_buffer(material_buffer)
			.set_range(material_size);
		let writes = [
			vk::WriteDescriptorSet::new()
				.set_dst_set(mat_set)
				.set_dst_binding(0)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_image_info(std::slice::from_ref(&base_img)),
			vk::WriteDescriptorSet::new()
				.set_dst_set(mat_set)
				.set_dst_binding(1)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_image_info(std::slice::from_ref(&orm_img)),
			vk::WriteDescriptorSet::new()
				.set_dst_set(mat_set)
				.set_dst_binding(2)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_image_info(std::slice::from_ref(&normal_img)),
			vk::WriteDescriptorSet::new()
				.set_dst_set(mat_set)
				.set_dst_binding(3)
				.set_descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.set_buffer_info(std::slice::from_ref(&mat_buf_info)),
		]
		.into_iter()
		.collect::<std::result::Result<Vec<_>, _>>()
		.map_err(Error::from)?;
		unsafe { self.device.update_descriptor_sets(&writes, &[]) }?;
		Ok(mat_set)
	}

	/// Maximum number of distinct materials per render slot.
	#[allow(dead_code)]
	pub(crate) const fn max_materials(&self) -> usize {
		self.max_materials
	}
}

impl Drop for SurfaceDescriptorArena {
	fn drop(&mut self) {
		unsafe {
			let _ = self.device.destroy_descriptor_pool(self.pool, None);
			let _ = self.device.destroy_sampler(self.sampler, None);
		}
	}
}
