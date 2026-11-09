use crate::{Error, Result, runtime::dna::kernel::ShaderArtifact};

use super::physical::DeviceLimits;

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
	env!("OUT_DIR"),
	"/render_vertex_color_lit_vert.spv"
));
/// Embedded SPIR-V for the vertex-color lit fragment shader.
static RENDER_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/render_vertex_color_lit_frag.spv"
));

/// Embedded SPIR-V for the flat-color vertex shader.
static FLAT_COLOR_VERT_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_flat_color_vert.spv"));
/// Embedded SPIR-V for the flat-color fragment shader.
static FLAT_COLOR_FRAG_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_flat_color_frag.spv"));

/// Embedded SPIR-V for the unlit vertex shader.
static UNLIT_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/render_unlit_vert.spv"));
/// Embedded SPIR-V for the unlit fragment shader.
static UNLIT_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/render_unlit_frag.spv"));

/// Embedded SPIR-V for the surface vertex shader.
static SURFACE_VERT_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_surface_vert.spv"));
/// Embedded SPIR-V for the surface fragment shader.
static SURFACE_FRAG_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_surface_frag.spv"));

/// Embedded SPIR-V for the UI canvas-grid vertex shader.
static CANVAS_GRID_VERT_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_ui_canvas_grid_vert.spv"));
/// Embedded SPIR-V for the UI canvas-grid fragment shader.
static CANVAS_GRID_FRAG_SPV: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/render_ui_canvas_grid_frag.spv"));

/// R8G8B8A8_UNORM color attachment format used by all render targets.
pub(in crate::runtime) const RENDER_COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// D32_SFLOAT depth attachment format used by all render targets.
pub(in crate::runtime) const RENDER_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Graphics pipeline for the vertex-color lit 3D renderer.
///
/// Uses Vulkan 1.3 dynamic rendering (no render-pass object), dynamic viewport
/// and scissor, depth test/write with LESS comparison, and an 80-byte push
/// constant block (row-major Mat4 view-projection + Vec4 light/ambient).
pub(crate) struct GraphicsPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: std::sync::Arc<vk::Device>,
}

impl GraphicsPipeline {
	/// Create the vertex-color lit graphics pipeline for one logical device.
	///
	/// # Errors
	///
	/// Returns an error when shader-module creation, layout creation, or
	/// pipeline creation fails, or when the device push-constant limit is
	/// below 80 bytes.
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		if limits.max_push_constants_size < GRAPHICS_PUSH_CONSTANT_SIZE {
			return Err(Error::missing_capability(
				"graphics pipeline requires 80 bytes of push constants",
			));
		}

		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::VERTEX_BIT)
			.set_offset(0)
			.set_size(GRAPHICS_PUSH_CONSTANT_SIZE);
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_push_constant_ranges(std::slice::from_ref(&push_range));
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "graphics pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(RENDER_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_words = decode_spirv_bytes(RENDER_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;

		let vert_module = create_shader_module(&device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				destroy_pipeline_layout(&device, layout);
				return Err(e);
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

		// MeshVertex layout: position (vec3, offset 0), normal (vec3, offset 12),
		// uv (vec2, offset 24), color (vec4, offset 32) — stride 48 bytes.
		// The graphics pipeline only binds position (loc 0), normal (loc 1), and
		// color (loc 2); uv is present in the struct but not consumed by this shader.
		let binding = vk::VertexInputBindingDescription::new()
			.set_binding(0)
			.set_stride(48)
			.set_input_rate(vk::VertexInputRate::VERTEX);
		let attributes = [
			vk::VertexInputAttributeDescription::new()
				.set_location(0)
				.set_binding(0)
				.set_format(vk::Format::R32G32B32_SFLOAT)
				.set_offset(0), // position
			vk::VertexInputAttributeDescription::new()
				.set_location(1)
				.set_binding(0)
				.set_format(vk::Format::R32G32B32_SFLOAT)
				.set_offset(12), // normal
			vk::VertexInputAttributeDescription::new()
				.set_location(2)
				.set_binding(0)
				.set_format(vk::Format::R32G32B32A32_SFLOAT)
				.set_offset(32), // color
		];
		let vertex_input = vk::PipelineVertexInputStateCreateInfo::new()
			.set_vertex_binding_descriptions(std::slice::from_ref(&binding))
			.map_err(Error::from)?
			.set_vertex_attribute_descriptions(&attributes)
			.map_err(Error::from)?;
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
		let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::new()
			.set_depth_test_enable(u32::from(true))
			.set_depth_write_enable(u32::from(true))
			.set_depth_compare_op(vk::CompareOp::LESS)
			.set_min_depth_bounds(0.0)
			.set_max_depth_bounds(1.0);
		let blend_attachment = vk::PipelineColorBlendAttachmentState::new().set_color_write_mask(
			vk::ColorComponentFlagBits::R_BIT
				| vk::ColorComponentFlagBits::G_BIT
				| vk::ColorComponentFlagBits::B_BIT
				| vk::ColorComponentFlagBits::A_BIT,
		);
		let blend = vk::PipelineColorBlendStateCreateInfo::new()
			.set_attachments(std::slice::from_ref(&blend_attachment))
			.map_err(Error::from)?;
		let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
		let dynamic = vk::PipelineDynamicStateCreateInfo::new()
			.set_dynamic_states(&dynamic_states)
			.map_err(Error::from)?;
		let color_formats = [RENDER_COLOR_FORMAT];
		let mut rendering = vk::PipelineRenderingCreateInfo::new()
			.set_color_attachment_formats(&color_formats)
			.map_err(Error::from)?
			.set_depth_attachment_format(RENDER_DEPTH_FORMAT);
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
			.set_color_blend_state(&blend)
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
				for handle in pipelines
					.into_iter()
					.filter(|handle| *handle != vk::Pipeline::default())
				{
					destroy_pipeline(&device, handle);
				}
				destroy_pipeline_layout(&device, layout);
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"graphics-pipeline creation",
					result,
				));
			}
		};
		Ok(Self {
			handle,
			layout,
			device,
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
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── FlatColorPipeline ─────────────────────────────────────────────────────────

/// Graphics pipeline for the flat-color material (Tier 1 — UI and debug).
///
/// Push constants: 96 bytes (MVP Mat4 + material_color Vec4 + opacity float + pad).
/// No descriptor sets. Depth test disabled; premultiplied alpha blend enabled.
pub(crate) struct FlatColorPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: std::sync::Arc<vk::Device>,
}

impl FlatColorPipeline {
	/// Create the flat-color pipeline for one logical device.
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		if limits.max_push_constants_size < FLAT_COLOR_PUSH_SIZE {
			return Err(Error::missing_capability(
				"flat-color pipeline requires 96 bytes of push constants",
			));
		}
		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::VERTEX_BIT | vk::ShaderStageFlagBits::FRAGMENT_BIT)
			.set_offset(0)
			.set_size(FLAT_COLOR_PUSH_SIZE);
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_push_constant_ranges(std::slice::from_ref(&push_range));
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "flat-color pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(FLAT_COLOR_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_words = decode_spirv_bytes(FLAT_COLOR_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let vert_module = create_shader_module(&device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			&device,
			layout,
			vert_module,
			frag_module,
			PipelineVertexInput::MeshVertex,
			PipelineDepth::Disabled,
			PipelineBlend::PremultipliedAlpha,
			cache,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device,
		})
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}
}

impl Drop for FlatColorPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── CanvasGridPipeline ────────────────────────────────────────────────────────

/// Graphics pipeline for the procedural canvas grid (raster compositor).
///
/// Vertexless fullscreen triangle (SV_VertexID); 100-byte fragment push block
/// carries all grid parameters; premultiplied source-over blend composes the
/// grid layer over the color attachment exactly like the compute path's
/// `uiSourceOver` fold.
pub(crate) struct CanvasGridPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: std::sync::Arc<vk::Device>,
}

impl CanvasGridPipeline {
	/// Create the canvas-grid pipeline for one logical device.
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		if limits.max_push_constants_size < CANVAS_GRID_PUSH_SIZE {
			return Err(Error::missing_capability(
				"canvas-grid pipeline requires 100 bytes of push constants",
			));
		}
		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::FRAGMENT_BIT)
			.set_offset(0)
			.set_size(CANVAS_GRID_PUSH_SIZE);
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_push_constant_ranges(std::slice::from_ref(&push_range));
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "canvas-grid pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(CANVAS_GRID_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_words = decode_spirv_bytes(CANVAS_GRID_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let vert_module = create_shader_module(&device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			&device,
			layout,
			vert_module,
			frag_module,
			PipelineVertexInput::Vertexless,
			PipelineDepth::Disabled,
			PipelineBlend::PremultipliedAlpha,
			cache,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device,
		})
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}
}

impl Drop for CanvasGridPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── UnlitPipeline ─────────────────────────────────────────────────────────────

/// Graphics pipeline for the unlit textured material (Tier 2 — image/video).
///
/// Push constants: 96 bytes (MVP + tint + uv_scale + uv_offset).
/// Descriptor set 0, binding 0: base-color sampler.
/// Depth test optional (disabled here for fullscreen quads); alpha blend enabled.
pub(crate) struct UnlitPipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: std::sync::Arc<vk::Device>,
}

impl UnlitPipeline {
	/// Create the unlit pipeline for one logical device.
	///
	/// `sampler_layout` is the descriptor-set layout for set 0 (one combined
	/// image sampler at binding 0).
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		sampler_layout: vk::DescriptorSetLayout,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		if limits.max_push_constants_size < UNLIT_PUSH_SIZE {
			return Err(Error::missing_capability(
				"unlit pipeline requires 96 bytes of push constants",
			));
		}
		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::VERTEX_BIT | vk::ShaderStageFlagBits::FRAGMENT_BIT)
			.set_offset(0)
			.set_size(UNLIT_PUSH_SIZE);
		let set_layouts = [sampler_layout];
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_set_layouts(&set_layouts)
			.map_err(Error::from)?
			.set_push_constant_ranges(std::slice::from_ref(&push_range));
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "unlit pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(UNLIT_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_words = decode_spirv_bytes(UNLIT_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let vert_module = create_shader_module(&device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			&device,
			layout,
			vert_module,
			frag_module,
			PipelineVertexInput::MeshVertex,
			PipelineDepth::Disabled,
			PipelineBlend::PremultipliedAlpha,
			cache,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device,
		})
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}
}

impl Drop for UnlitPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── SurfacePipeline ───────────────────────────────────────────────────

/// Graphics pipeline for the Surface PBR material (Tier 3).
///
/// Push constants: 128 bytes (MVP Mat4 + model Mat4).
/// Descriptor set 0: scene-lights UBO.
/// Descriptor set 1: base-color, ORM, normal, material-params bindings.
/// Depth test/write enabled (LESS). Opaque blend (no alpha).
pub(crate) struct SurfacePipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	device: std::sync::Arc<vk::Device>,
}

impl SurfacePipeline {
	/// Create the surface opaque pipeline.
	///
	/// `scene_layout` is set 0 (scene lights UBO).
	/// `material_layout` is set 1 (textures + material params).
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		scene_layout: vk::DescriptorSetLayout,
		material_layout: vk::DescriptorSetLayout,
		limits: DeviceLimits,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		if limits.max_push_constants_size < SURFACE_PUSH_SIZE {
			return Err(Error::missing_capability(
				"surface pipeline requires 128 bytes of push constants",
			));
		}
		let push_range = vk::PushConstantRange::new()
			.set_stage_flags(vk::ShaderStageFlagBits::VERTEX_BIT)
			.set_offset(0)
			.set_size(SURFACE_PUSH_SIZE);
		let set_layouts = [scene_layout, material_layout];
		let layout_info = vk::PipelineLayoutCreateInfo::new()
			.set_set_layouts(&set_layouts)
			.map_err(Error::from)?
			.set_push_constant_ranges(std::slice::from_ref(&push_range));
		let layout =
			unsafe { device.create_pipeline_layout(&(layout_info).map_err(Error::from)?, None) }
				.map_err(|s| Error::backend_failure("Vulkan", "surface pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(SURFACE_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_words = decode_spirv_bytes(SURFACE_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let vert_module = create_shader_module(&device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(&device, layout))?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			&device,
			layout,
			vert_module,
			frag_module,
			PipelineVertexInput::MeshVertex,
			PipelineDepth::WriteAndTest,
			PipelineBlend::Opaque,
			cache,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(&device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device,
		})
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}
}

impl Drop for SurfacePipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── SurfaceBlendPipeline ─────────────────────────────────────────────

/// Transparent variant of the Surface pipeline.
///
/// Identical shader modules and descriptor layout as [`SurfacePipeline`]
/// but uses `PipelineDepth::TestOnly` (depth test on, write off) and
/// `PipelineBlend::PremultipliedAlpha`. Used for the back-to-front transparent
/// pass in scene rendering.
pub(crate) struct SurfaceBlendPipeline {
	handle: vk::Pipeline,
	device: std::sync::Arc<vk::Device>,
}

impl SurfaceBlendPipeline {
	/// Create the surface alpha-blend pipeline.
	///
	/// Shares the same `layout` (and therefore the same push range and
	/// descriptor set layouts) as the opaque [`SurfacePipeline`].
	pub(in crate::runtime) fn new(
		device: std::sync::Arc<vk::Device>,
		layout: vk::PipelineLayout,
		cache: vk::PipelineCache,
	) -> Result<Self> {
		let vert_words = decode_spirv_bytes(SURFACE_VERT_SPV)?;
		let frag_words = decode_spirv_bytes(SURFACE_FRAG_SPV)?;
		let vert_module = create_shader_module(&device, &vert_words)?;
		let frag_module = match create_shader_module(&device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) }?;
				return Err(e);
			}
		};
		let handle = create_graphics_pipeline_inner(
			&device,
			layout,
			vert_module,
			frag_module,
			PipelineVertexInput::MeshVertex,
			PipelineDepth::TestOnly,
			PipelineBlend::PremultipliedAlpha,
			cache,
		)?;
		Ok(Self { handle, device })
	}

	pub(crate) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}
}

impl Drop for SurfaceBlendPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
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

// ── Shared pipeline creation helper ──────────────────────────────────────────

enum PipelineDepth {
	Disabled,
	WriteAndTest,
	/// Depth test enabled, depth write disabled — used for transparent passes.
	TestOnly,
}

enum PipelineBlend {
	Opaque,
	PremultipliedAlpha,
}

/// Vertex-fetch configuration for a graphics pipeline.
enum PipelineVertexInput {
	/// MeshVertex layout: position(0,f3,off0), normal(1,f3,off12),
	/// uv(2,f2,off24), color(3,f4,off32) — stride 48.
	MeshVertex,
	/// No vertex buffers — vertex shader derives geometry from SV_VertexID.
	Vertexless,
}

/// Create a graphics pipeline and immediately destroy its shader modules.
/// Vertex input layout matches MeshVertex (stride 48, locations 0–3).
#[allow(clippy::too_many_arguments)]
fn create_graphics_pipeline_inner(
	device: &vk::Device,
	layout: vk::PipelineLayout,
	vert_module: vk::ShaderModule,
	frag_module: vk::ShaderModule,
	vertex: PipelineVertexInput,
	depth: PipelineDepth,
	blend: PipelineBlend,
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

	// MeshVertex layout: position(0,f3,off0), normal(1,f3,off12),
	// uv(2,f2,off24), color(3,f4,off32) — stride 48.
	let binding = vk::VertexInputBindingDescription::new()
		.set_binding(0)
		.set_stride(48)
		.set_input_rate(vk::VertexInputRate::VERTEX);
	let attributes = [
		vk::VertexInputAttributeDescription::new()
			.set_location(0)
			.set_binding(0)
			.set_format(vk::Format::R32G32B32_SFLOAT)
			.set_offset(0),
		vk::VertexInputAttributeDescription::new()
			.set_location(1)
			.set_binding(0)
			.set_format(vk::Format::R32G32B32_SFLOAT)
			.set_offset(12),
		vk::VertexInputAttributeDescription::new()
			.set_location(2)
			.set_binding(0)
			.set_format(vk::Format::R32G32_SFLOAT)
			.set_offset(24),
		vk::VertexInputAttributeDescription::new()
			.set_location(3)
			.set_binding(0)
			.set_format(vk::Format::R32G32B32A32_SFLOAT)
			.set_offset(32),
	];
	let vertex_input = match vertex {
		PipelineVertexInput::MeshVertex => vk::PipelineVertexInputStateCreateInfo::new()
			.set_vertex_binding_descriptions(std::slice::from_ref(&binding))
			.map_err(Error::from)?
			.set_vertex_attribute_descriptions(&attributes)
			.map_err(Error::from)?,
		PipelineVertexInput::Vertexless => vk::PipelineVertexInputStateCreateInfo::new(),
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

	let depth_stencil = match depth {
		PipelineDepth::Disabled => vk::PipelineDepthStencilStateCreateInfo::new(),
		PipelineDepth::WriteAndTest => vk::PipelineDepthStencilStateCreateInfo::new()
			.set_depth_test_enable(u32::from(true))
			.set_depth_write_enable(u32::from(true))
			.set_depth_compare_op(vk::CompareOp::LESS)
			.set_min_depth_bounds(0.0)
			.set_max_depth_bounds(1.0),
		PipelineDepth::TestOnly => vk::PipelineDepthStencilStateCreateInfo::new()
			.set_depth_test_enable(u32::from(true))
			.set_depth_write_enable(u32::from(false))
			.set_depth_compare_op(vk::CompareOp::LESS)
			.set_min_depth_bounds(0.0)
			.set_max_depth_bounds(1.0),
	};

	let blend_attachment = match blend {
		PipelineBlend::Opaque => vk::PipelineColorBlendAttachmentState::new().set_color_write_mask(
			vk::ColorComponentFlagBits::R_BIT
				| vk::ColorComponentFlagBits::G_BIT
				| vk::ColorComponentFlagBits::B_BIT
				| vk::ColorComponentFlagBits::A_BIT,
		),
		PipelineBlend::PremultipliedAlpha => vk::PipelineColorBlendAttachmentState::new()
			.set_blend_enable(u32::from(true))
			.set_src_color_blend_factor(vk::BlendFactor::ONE)
			.set_dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.set_color_blend_op(vk::BlendOp::ADD)
			.set_src_alpha_blend_factor(vk::BlendFactor::ONE)
			.set_dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.set_alpha_blend_op(vk::BlendOp::ADD)
			.set_color_write_mask(
				vk::ColorComponentFlagBits::R_BIT
					| vk::ColorComponentFlagBits::G_BIT
					| vk::ColorComponentFlagBits::B_BIT
					| vk::ColorComponentFlagBits::A_BIT,
			),
	};
	let color_blend = vk::PipelineColorBlendStateCreateInfo::new()
		.set_attachments(std::slice::from_ref(&blend_attachment))
		.map_err(Error::from)?;

	let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
	let dynamic = vk::PipelineDynamicStateCreateInfo::new()
		.set_dynamic_states(&dynamic_states)
		.map_err(Error::from)?;
	let color_formats = [RENDER_COLOR_FORMAT];
	let mut rendering = vk::PipelineRenderingCreateInfo::new()
		.set_color_attachment_formats(&color_formats)
		.map_err(Error::from)?
		.set_depth_attachment_format(RENDER_DEPTH_FORMAT);
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

fn decode_spirv_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
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
pub(super) struct PipelineCache {
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
	pub(super) fn new(device: std::sync::Arc<vk::Device>, initial_data: &[u8]) -> Result<Self> {
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
	pub(super) fn open(
		device: &std::sync::Arc<vk::Device>,
		physical: &super::physical::DevicePhysical,
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
	pub(super) fn data(&self) -> Result<Vec<u8>> {
		// SAFETY: `handle` is a live cache created on this device.
		unsafe { self.device.get_pipeline_cache_data(self.handle) }
			.map_err(|s| Error::backend_failure("Vulkan", "pipeline-cache data query", s))
	}

	/// Merge externally synchronized worker caches into this cache.
	///
	/// Callers must have joined every thread that produced `sources`; no host
	/// may touch source or destination concurrently with the merge.
	pub(super) fn merge(&self, sources: &[vk::PipelineCache]) -> Result<()> {
		if sources.is_empty() {
			return Ok(());
		}
		// SAFETY: the caller joins all producer threads before merging, so
		// sources and destination are externally synchronized.
		unsafe { self.device.merge_pipeline_caches(self.handle, sources) }
			.map_err(|s| Error::backend_failure("Vulkan", "pipeline-cache merge", s))
	}

	pub(super) const fn raw(&self) -> vk::PipelineCache {
		self.handle
	}

	pub(super) const fn initial_data_bytes(&self) -> usize {
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

pub(in crate::runtime) struct ComputePipeline {
	handle: vk::Pipeline,
	layout: vk::PipelineLayout,
	max_dispatch_group_count: [u32; 3],
	push_constant_size: u32,
	device: std::sync::Arc<vk::Device>,
}

impl ComputePipeline {
	pub(super) fn new(
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
	pub(super) fn from_words(
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

	pub(super) fn destroy(&mut self, device: &vk::Device) {
		destroy_pipeline(device, self.handle);
		destroy_pipeline_layout(device, self.layout);
		self.handle = vk::Pipeline::NULL;
		self.layout = vk::PipelineLayout::NULL;
	}

	pub(super) const fn raw(&self) -> vk::Pipeline {
		self.handle
	}

	pub(super) const fn layout(&self) -> vk::PipelineLayout {
		self.layout
	}

	pub(super) const fn max_dispatch_group_count(&self) -> [u32; 3] {
		self.max_dispatch_group_count
	}

	pub(super) const fn push_constant_size(&self) -> u32 {
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

pub(super) fn validate_artifact_limits(
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
