use crate::{Error, Result, runtime::shader::ShaderArtifact};

use super::physical::DeviceLimits;

// Push-constant layout for the vertex-color lit graphics pipeline.
// Donor: oacpp/source/cpp/lib/oa/render/renderer3d.cpp `RenderPushConstants`.
// 64-byte row-major Mat4 + 16-byte Vec4 = 80 bytes total.
pub(crate) const GRAPHICS_PUSH_CONSTANT_SIZE: u32 = 80;

// Push size for flat-color: 64 (MVP) + 16 (material_color) + 16 (opacity+pad) = 96 bytes.
pub(crate) const FLAT_COLOR_PUSH_SIZE: u32 = 96;
// Push size for unlit: 64 (MVP) + 16 (tint) + 16 (uv_scale+uv_offset) = 96 bytes.
pub(crate) const UNLIT_PUSH_SIZE: u32 = 96;
// Push size for standard surface: 64 (MVP) + 64 (model matrix) = 128 bytes.
pub(crate) const STANDARD_SURFACE_PUSH_SIZE: u32 = 128;

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

/// Embedded SPIR-V for the standard-surface vertex shader.
static STANDARD_SURFACE_VERT_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/render_standard_surface_vert.spv"
));
/// Embedded SPIR-V for the standard-surface fragment shader.
static STANDARD_SURFACE_FRAG_SPV: &[u8] = include_bytes!(concat!(
	env!("OUT_DIR"),
	"/render_standard_surface_frag.spv"
));

/// R8G8B8A8_UNORM color attachment format used by all render targets.
pub(in crate::runtime) const RENDER_COLOR_FORMAT: ash::vk::Format = ash::vk::Format::R8G8B8A8_UNORM;
/// D32_SFLOAT depth attachment format used by all render targets.
pub(in crate::runtime) const RENDER_DEPTH_FORMAT: ash::vk::Format = ash::vk::Format::D32_SFLOAT;

/// Graphics pipeline for the vertex-color lit 3D renderer.
///
/// Uses Vulkan 1.3 dynamic rendering (no render-pass object), dynamic viewport
/// and scissor, depth test/write with LESS comparison, and an 80-byte push
/// constant block (row-major Mat4 view-projection + Vec4 light/ambient).
pub(crate) struct GraphicsPipeline {
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	device: ash::Device,
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
	pub(in crate::runtime) fn new(device: &ash::Device, limits: DeviceLimits) -> Result<Self> {
		if limits.max_push_constants_size < GRAPHICS_PUSH_CONSTANT_SIZE {
			return Err(Error::missing_capability(
				"graphics pipeline requires 80 bytes of push constants",
			));
		}

		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::VERTEX)
			.offset(0)
			.size(GRAPHICS_PUSH_CONSTANT_SIZE);
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.push_constant_ranges(std::slice::from_ref(&push_range));
		let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
			.map_err(|s| Error::backend_failure("Vulkan", "graphics pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(RENDER_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_words = decode_spirv_bytes(RENDER_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;

		let vert_module = create_shader_module(device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) };
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};

		let stages = [
			ash::vk::PipelineShaderStageCreateInfo::default()
				.stage(ash::vk::ShaderStageFlags::VERTEX)
				.module(vert_module)
				.name(c"main"),
			ash::vk::PipelineShaderStageCreateInfo::default()
				.stage(ash::vk::ShaderStageFlags::FRAGMENT)
				.module(frag_module)
				.name(c"main"),
		];

		// MeshVertex layout: position (vec3, offset 0), normal (vec3, offset 12),
		// uv (vec2, offset 24), color (vec4, offset 32) — stride 48 bytes.
		// The graphics pipeline only binds position (loc 0), normal (loc 1), and
		// color (loc 2); uv is present in the struct but not consumed by this shader.
		let binding = ash::vk::VertexInputBindingDescription::default()
			.binding(0)
			.stride(48)
			.input_rate(ash::vk::VertexInputRate::VERTEX);
		let attributes = [
			ash::vk::VertexInputAttributeDescription::default()
				.location(0)
				.binding(0)
				.format(ash::vk::Format::R32G32B32_SFLOAT)
				.offset(0), // position
			ash::vk::VertexInputAttributeDescription::default()
				.location(1)
				.binding(0)
				.format(ash::vk::Format::R32G32B32_SFLOAT)
				.offset(12), // normal
			ash::vk::VertexInputAttributeDescription::default()
				.location(2)
				.binding(0)
				.format(ash::vk::Format::R32G32B32A32_SFLOAT)
				.offset(32), // color
		];
		let vertex_input = ash::vk::PipelineVertexInputStateCreateInfo::default()
			.vertex_binding_descriptions(std::slice::from_ref(&binding))
			.vertex_attribute_descriptions(&attributes);
		let assembly = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
			.topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);
		let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default()
			.viewport_count(1)
			.scissor_count(1);
		let raster = ash::vk::PipelineRasterizationStateCreateInfo::default()
			.polygon_mode(ash::vk::PolygonMode::FILL)
			.cull_mode(ash::vk::CullModeFlags::NONE)
			.front_face(ash::vk::FrontFace::COUNTER_CLOCKWISE)
			.line_width(1.0);
		let multisample = ash::vk::PipelineMultisampleStateCreateInfo::default()
			.rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);
		let depth_stencil = ash::vk::PipelineDepthStencilStateCreateInfo::default()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(ash::vk::CompareOp::LESS)
			.min_depth_bounds(0.0)
			.max_depth_bounds(1.0);
		let blend_attachment = ash::vk::PipelineColorBlendAttachmentState::default()
			.color_write_mask(ash::vk::ColorComponentFlags::RGBA);
		let blend = ash::vk::PipelineColorBlendStateCreateInfo::default()
			.attachments(std::slice::from_ref(&blend_attachment));
		let dynamic_states = [
			ash::vk::DynamicState::VIEWPORT,
			ash::vk::DynamicState::SCISSOR,
		];
		let dynamic =
			ash::vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
		let color_formats = [RENDER_COLOR_FORMAT];
		let mut rendering = ash::vk::PipelineRenderingCreateInfo::default()
			.color_attachment_formats(&color_formats)
			.depth_attachment_format(RENDER_DEPTH_FORMAT);
		let pipeline_info = ash::vk::GraphicsPipelineCreateInfo::default()
			.push_next(&mut rendering)
			.stages(&stages)
			.vertex_input_state(&vertex_input)
			.input_assembly_state(&assembly)
			.viewport_state(&viewport_state)
			.rasterization_state(&raster)
			.multisample_state(&multisample)
			.depth_stencil_state(&depth_stencil)
			.color_blend_state(&blend)
			.dynamic_state(&dynamic)
			.layout(layout);

		let result = unsafe {
			device.create_graphics_pipelines(
				ash::vk::PipelineCache::null(),
				std::slice::from_ref(&pipeline_info),
				None,
			)
		};
		unsafe {
			device.destroy_shader_module(vert_module, None);
			device.destroy_shader_module(frag_module, None);
		}
		let pipelines = match result {
			Ok(pipelines) => pipelines,
			Err((partial, source)) => {
				for p in partial {
					destroy_pipeline(device, p);
				}
				destroy_pipeline_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"graphics-pipeline creation",
					source,
				));
			}
		};
		let Some(handle) = pipelines.first().copied() else {
			destroy_pipeline_layout(device, layout);
			return Err(Error::backend_failure(
				"Vulkan",
				"graphics-pipeline creation",
				std::io::Error::other("Vulkan returned no pipeline after successful creation"),
			));
		};
		Ok(Self {
			handle,
			layout,
			device: device.clone(),
		})
	}

	pub(crate) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> ash::vk::PipelineLayout {
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
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	device: ash::Device,
}

impl FlatColorPipeline {
	/// Create the flat-color pipeline for one logical device.
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(device: &ash::Device, limits: DeviceLimits) -> Result<Self> {
		if limits.max_push_constants_size < FLAT_COLOR_PUSH_SIZE {
			return Err(Error::missing_capability(
				"flat-color pipeline requires 96 bytes of push constants",
			));
		}
		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT)
			.offset(0)
			.size(FLAT_COLOR_PUSH_SIZE);
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.push_constant_ranges(std::slice::from_ref(&push_range));
		let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
			.map_err(|s| Error::backend_failure("Vulkan", "flat-color pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(FLAT_COLOR_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_words = decode_spirv_bytes(FLAT_COLOR_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let vert_module = create_shader_module(device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) };
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			device,
			layout,
			vert_module,
			frag_module,
			PipelineDepth::Disabled,
			PipelineBlend::PremultipliedAlpha,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device: device.clone(),
		})
	}

	pub(crate) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> ash::vk::PipelineLayout {
		self.layout
	}
}

impl Drop for FlatColorPipeline {
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
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	device: ash::Device,
}

impl UnlitPipeline {
	/// Create the unlit pipeline for one logical device.
	///
	/// `sampler_layout` is the descriptor-set layout for set 0 (one combined
	/// image sampler at binding 0).
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: &ash::Device,
		sampler_layout: ash::vk::DescriptorSetLayout,
		limits: DeviceLimits,
	) -> Result<Self> {
		if limits.max_push_constants_size < UNLIT_PUSH_SIZE {
			return Err(Error::missing_capability(
				"unlit pipeline requires 96 bytes of push constants",
			));
		}
		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT)
			.offset(0)
			.size(UNLIT_PUSH_SIZE);
		let set_layouts = [sampler_layout];
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.set_layouts(&set_layouts)
			.push_constant_ranges(std::slice::from_ref(&push_range));
		let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
			.map_err(|s| Error::backend_failure("Vulkan", "unlit pipeline-layout creation", s))?;

		let vert_words = decode_spirv_bytes(UNLIT_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_words = decode_spirv_bytes(UNLIT_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let vert_module = create_shader_module(device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) };
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			device,
			layout,
			vert_module,
			frag_module,
			PipelineDepth::Disabled,
			PipelineBlend::PremultipliedAlpha,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device: device.clone(),
		})
	}

	pub(crate) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> ash::vk::PipelineLayout {
		self.layout
	}
}

impl Drop for UnlitPipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── StandardSurfacePipeline ───────────────────────────────────────────────────

/// Graphics pipeline for the Standard Surface PBR material (Tier 3).
///
/// Push constants: 128 bytes (MVP Mat4 + model Mat4).
/// Descriptor set 0: scene-lights UBO.
/// Descriptor set 1: base-color, ORM, normal, material-params bindings.
/// Depth test/write enabled (LESS). Opaque blend (no alpha).
pub(crate) struct StandardSurfacePipeline {
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	device: ash::Device,
}

impl StandardSurfacePipeline {
	/// Create the standard-surface opaque pipeline.
	///
	/// `scene_layout` is set 0 (scene lights UBO).
	/// `material_layout` is set 1 (textures + material params).
	#[allow(private_interfaces)]
	pub(in crate::runtime) fn new(
		device: &ash::Device,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
		limits: DeviceLimits,
	) -> Result<Self> {
		if limits.max_push_constants_size < STANDARD_SURFACE_PUSH_SIZE {
			return Err(Error::missing_capability(
				"standard-surface pipeline requires 128 bytes of push constants",
			));
		}
		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::VERTEX)
			.offset(0)
			.size(STANDARD_SURFACE_PUSH_SIZE);
		let set_layouts = [scene_layout, material_layout];
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.set_layouts(&set_layouts)
			.push_constant_ranges(std::slice::from_ref(&push_range));
		let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(|s| {
			Error::backend_failure("Vulkan", "standard-surface pipeline-layout creation", s)
		})?;

		let vert_words = decode_spirv_bytes(STANDARD_SURFACE_VERT_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_words = decode_spirv_bytes(STANDARD_SURFACE_FRAG_SPV)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let vert_module = create_shader_module(device, &vert_words)
			.inspect_err(|_| destroy_pipeline_layout(device, layout))?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) };
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};

		let handle = match create_graphics_pipeline_inner(
			device,
			layout,
			vert_module,
			frag_module,
			PipelineDepth::WriteAndTest,
			PipelineBlend::Opaque,
		) {
			Ok(h) => h,
			Err(e) => {
				destroy_pipeline_layout(device, layout);
				return Err(e);
			}
		};
		Ok(Self {
			handle,
			layout,
			device: device.clone(),
		})
	}

	pub(crate) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(crate) const fn layout(&self) -> ash::vk::PipelineLayout {
		self.layout
	}
}

impl Drop for StandardSurfacePipeline {
	fn drop(&mut self) {
		destroy_pipeline(&self.device, self.handle);
		destroy_pipeline_layout(&self.device, self.layout);
	}
}

// ── StandardSurfaceBlendPipeline ─────────────────────────────────────────────

/// Transparent variant of the Standard Surface pipeline.
///
/// Identical shader modules and descriptor layout as [`StandardSurfacePipeline`]
/// but uses `PipelineDepth::TestOnly` (depth test on, write off) and
/// `PipelineBlend::PremultipliedAlpha`. Used for the back-to-front transparent
/// pass in scene rendering.
pub(crate) struct StandardSurfaceBlendPipeline {
	handle: ash::vk::Pipeline,
	device: ash::Device,
}

impl StandardSurfaceBlendPipeline {
	/// Create the standard-surface alpha-blend pipeline.
	///
	/// Shares the same `layout` (and therefore the same push range and
	/// descriptor set layouts) as the opaque [`StandardSurfacePipeline`].
	pub(in crate::runtime) fn new(
		device: &ash::Device,
		layout: ash::vk::PipelineLayout,
	) -> Result<Self> {
		let vert_words = decode_spirv_bytes(STANDARD_SURFACE_VERT_SPV)?;
		let frag_words = decode_spirv_bytes(STANDARD_SURFACE_FRAG_SPV)?;
		let vert_module = create_shader_module(device, &vert_words)?;
		let frag_module = match create_shader_module(device, &frag_words) {
			Ok(m) => m,
			Err(e) => {
				unsafe { device.destroy_shader_module(vert_module, None) };
				return Err(e);
			}
		};
		let handle = create_graphics_pipeline_inner(
			device,
			layout,
			vert_module,
			frag_module,
			PipelineDepth::TestOnly,
			PipelineBlend::PremultipliedAlpha,
		)?;
		Ok(Self {
			handle,
			device: device.clone(),
		})
	}

	pub(crate) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}
}

impl Drop for StandardSurfaceBlendPipeline {
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
	pub(crate) sampler_layout: ash::vk::DescriptorSetLayout,
	device: ash::Device,
}

impl UnlitDescriptorLayouts {
	pub(in crate::runtime) fn new(device: &ash::Device) -> Result<Self> {
		let sampler_binding = ash::vk::DescriptorSetLayoutBinding::default()
			.binding(0)
			.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.descriptor_count(1)
			.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT);
		let bindings = [sampler_binding];
		let info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
		let layout = unsafe { device.create_descriptor_set_layout(&info, None) }
			.map_err(|s| Error::backend_failure("Vulkan", "unlit descriptor-set-layout creation", s))?;
		Ok(Self {
			sampler_layout: layout,
			device: device.clone(),
		})
	}
}

impl Drop for UnlitDescriptorLayouts {
	fn drop(&mut self) {
		unsafe {
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
	device: ash::Device,
	pool: ash::vk::DescriptorPool,
	sampler: ash::vk::Sampler,
	sets: Vec<ash::vk::DescriptorSet>,
}

impl UnlitDescriptorArena {
	pub(in crate::runtime) fn new(
		device: &ash::Device,
		layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
	) -> Result<Self> {
		let sampler_info = ash::vk::SamplerCreateInfo::default()
			.mag_filter(ash::vk::Filter::LINEAR)
			.min_filter(ash::vk::Filter::LINEAR)
			.mipmap_mode(ash::vk::SamplerMipmapMode::NEAREST)
			.address_mode_u(ash::vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_v(ash::vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.address_mode_w(ash::vk::SamplerAddressMode::CLAMP_TO_EDGE)
			.min_lod(0.0)
			.max_lod(0.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "unlit sampler creation", source))?;
		let pool_size = ash::vk::DescriptorPoolSize::default()
			.ty(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.descriptor_count(set_count);
		let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
			.max_sets(set_count)
			.pool_sizes(std::slice::from_ref(&pool_size));
		let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
			Ok(pool) => pool,
			Err(source) => {
				unsafe { device.destroy_sampler(sampler, None) };
				return Err(Error::backend_failure(
					"Vulkan",
					"unlit descriptor-pool creation",
					source,
				));
			}
		};
		let layouts = vec![layout; set_count as usize];
		let allocate_info = ash::vk::DescriptorSetAllocateInfo::default()
			.descriptor_pool(pool)
			.set_layouts(&layouts);
		let sets = match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
			Ok(sets) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None);
					device.destroy_sampler(sampler, None);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"unlit descriptor-set allocation",
					source,
				));
			}
		};
		Ok(Self {
			device: device.clone(),
			pool,
			sampler,
			sets,
		})
	}

	pub(crate) fn write_sampled_image(
		&self,
		set_index: usize,
		view: ash::vk::ImageView,
	) -> Result<ash::vk::DescriptorSet> {
		let set = *self
			.sets
			.get(set_index)
			.ok_or_else(|| Error::internal("unlit descriptor-set index is out of range"))?;
		let image = ash::vk::DescriptorImageInfo::default()
			.sampler(self.sampler)
			.image_view(view)
			.image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
		let write = ash::vk::WriteDescriptorSet::default()
			.dst_set(set)
			.dst_binding(0)
			.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.image_info(std::slice::from_ref(&image));
		unsafe { self.device.update_descriptor_sets(&[write], &[]) };
		Ok(set)
	}
}

impl Drop for UnlitDescriptorArena {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_descriptor_pool(self.pool, None);
			self.device.destroy_sampler(self.sampler, None);
		}
	}
}

/// Owned descriptor set layouts for the Standard Surface PBR pipeline.
///
/// Set 0, binding 0: scene lights UBO (`SceneLights`), fragment stage.
/// Set 1, binding 0: base-color combined-image-sampler, fragment stage.
/// Set 1, binding 1: ORM combined-image-sampler, fragment stage.
/// Set 1, binding 2: normal combined-image-sampler, fragment stage.
/// Set 1, binding 3: material params UBO (`MaterialParams`), fragment stage.
pub(crate) struct StandardSurfaceDescriptorLayouts {
	/// Set 0 layout (scene lights UBO).
	pub(crate) scene_layout: ash::vk::DescriptorSetLayout,
	/// Set 1 layout (3 samplers + material params UBO).
	pub(crate) material_layout: ash::vk::DescriptorSetLayout,
	device: ash::Device,
}

impl StandardSurfaceDescriptorLayouts {
	pub(in crate::runtime) fn new(device: &ash::Device) -> Result<Self> {
		// Set 0: scene lights UBO.
		let scene_binding = ash::vk::DescriptorSetLayoutBinding::default()
			.binding(0)
			.descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
			.descriptor_count(1)
			.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT);
		let scene_bindings = [scene_binding];
		let scene_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&scene_bindings);
		let scene_layout =
			unsafe { device.create_descriptor_set_layout(&scene_info, None) }.map_err(|s| {
				Error::backend_failure(
					"Vulkan",
					"standard-surface scene descriptor-set-layout creation",
					s,
				)
			})?;

		// Set 1: base-color / ORM / normal samplers + material params UBO.
		let mat_bindings = [
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(1)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(2)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(1)
				.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
			ash::vk::DescriptorSetLayoutBinding::default()
				.binding(3)
				.descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
				.descriptor_count(1)
				.stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
		];
		let mat_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&mat_bindings);
		let material_layout = match unsafe { device.create_descriptor_set_layout(&mat_info, None) } {
			Ok(l) => l,
			Err(s) => {
				unsafe { device.destroy_descriptor_set_layout(scene_layout, None) };
				return Err(Error::backend_failure(
					"Vulkan",
					"standard-surface material descriptor-set-layout creation",
					s,
				));
			}
		};

		Ok(Self {
			scene_layout,
			material_layout,
			device: device.clone(),
		})
	}
}

impl Drop for StandardSurfaceDescriptorLayouts {
	fn drop(&mut self) {
		unsafe {
			self
				.device
				.destroy_descriptor_set_layout(self.material_layout, None);
			self
				.device
				.destroy_descriptor_set_layout(self.scene_layout, None);
		}
	}
}

/// Per-frame uniform + sampled-image descriptors for the Standard Surface pipeline.
///
/// Allocates `slot_count` scene sets (set 0) and
/// `slot_count × max_materials_per_slot` material sets (set 1). Owns a shared
/// linear sampler used for all texture bindings; callers supply a white-texture
/// fallback view when a material has no texture assigned.
pub(crate) struct StandardSurfaceDescriptorArena {
	device: ash::Device,
	pool: ash::vk::DescriptorPool,
	sampler: ash::vk::Sampler,
	scene_sets: Vec<ash::vk::DescriptorSet>,
	/// Flat `[slot_count × max_materials_per_slot]` material sets.
	material_sets: Vec<ash::vk::DescriptorSet>,
	/// Number of material set slots per render slot.
	max_materials: usize,
}

impl StandardSurfaceDescriptorArena {
	pub(in crate::runtime) fn new(
		device: &ash::Device,
		scene_layout: ash::vk::DescriptorSetLayout,
		material_layout: ash::vk::DescriptorSetLayout,
		set_count: u32,
		max_materials_per_slot: u32,
	) -> Result<Self> {
		let max_materials_per_slot = max_materials_per_slot.max(1);
		let sampler_info = ash::vk::SamplerCreateInfo::default()
			.mag_filter(ash::vk::Filter::LINEAR)
			.min_filter(ash::vk::Filter::LINEAR)
			.mipmap_mode(ash::vk::SamplerMipmapMode::NEAREST)
			.address_mode_u(ash::vk::SamplerAddressMode::REPEAT)
			.address_mode_v(ash::vk::SamplerAddressMode::REPEAT)
			.address_mode_w(ash::vk::SamplerAddressMode::REPEAT)
			.min_lod(0.0)
			.max_lod(0.0);
		let sampler = unsafe { device.create_sampler(&sampler_info, None) }.map_err(|source| {
			Error::backend_failure("Vulkan", "standard-surface sampler creation", source)
		})?;
		let mat_set_total = set_count.saturating_mul(max_materials_per_slot);
		let pool_sizes = [
			ash::vk::DescriptorPoolSize::default()
				.ty(ash::vk::DescriptorType::UNIFORM_BUFFER)
				// 1 scene UBO per slot + 1 material UBO per material set
				.descriptor_count(set_count.saturating_add(mat_set_total)),
			ash::vk::DescriptorPoolSize::default()
				.ty(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				// 3 texture bindings per material set
				.descriptor_count(mat_set_total.saturating_mul(3)),
		];
		let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
			.max_sets(set_count.saturating_add(mat_set_total))
			.pool_sizes(&pool_sizes);
		let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
			Ok(p) => p,
			Err(source) => {
				unsafe { device.destroy_sampler(sampler, None) };
				return Err(Error::backend_failure(
					"Vulkan",
					"standard-surface descriptor-pool creation",
					source,
				));
			}
		};
		let allocate_n = |layout: ash::vk::DescriptorSetLayout, n: u32| {
			let layouts = vec![layout; n as usize];
			let info = ash::vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(pool)
				.set_layouts(&layouts);
			unsafe { device.allocate_descriptor_sets(&info) }
		};
		let scene_sets = match allocate_n(scene_layout, set_count) {
			Ok(sets) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None);
					device.destroy_sampler(sampler, None);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"standard-surface scene descriptor allocation",
					source,
				));
			}
		};
		let material_sets = match allocate_n(material_layout, mat_set_total) {
			Ok(sets) => sets,
			Err(source) => {
				unsafe {
					device.destroy_descriptor_pool(pool, None);
					device.destroy_sampler(sampler, None);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"standard-surface material descriptor allocation",
					source,
				));
			}
		};
		Ok(Self {
			device: device.clone(),
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
		scene_buffer: ash::vk::Buffer,
		scene_size: u64,
	) -> Result<ash::vk::DescriptorSet> {
		let scene_set = *self
			.scene_sets
			.get(slot_index)
			.ok_or_else(|| Error::internal("standard-surface scene set index is out of range"))?;
		let scene_info = ash::vk::DescriptorBufferInfo::default()
			.buffer(scene_buffer)
			.range(scene_size);
		let write = ash::vk::WriteDescriptorSet::default()
			.dst_set(scene_set)
			.dst_binding(0)
			.descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
			.buffer_info(std::slice::from_ref(&scene_info));
		unsafe { self.device.update_descriptor_sets(&[write], &[]) };
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
		material_buffer: ash::vk::Buffer,
		material_size: u64,
		base_view: Option<ash::vk::ImageView>,
		orm_view: Option<ash::vk::ImageView>,
		normal_view: Option<ash::vk::ImageView>,
		white_view: ash::vk::ImageView,
	) -> Result<ash::vk::DescriptorSet> {
		let flat = slot_index * self.max_materials + material_index;
		let mat_set = *self
			.material_sets
			.get(flat)
			.ok_or_else(|| Error::internal("standard-surface material set index is out of range"))?;
		let sampler_image = |view: Option<ash::vk::ImageView>| {
			ash::vk::DescriptorImageInfo::default()
				.sampler(self.sampler)
				.image_view(view.unwrap_or(white_view))
				.image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		};
		let base_img = sampler_image(base_view);
		let orm_img = sampler_image(orm_view);
		let normal_img = sampler_image(normal_view);
		let mat_buf_info = ash::vk::DescriptorBufferInfo::default()
			.buffer(material_buffer)
			.range(material_size);
		let writes = [
			ash::vk::WriteDescriptorSet::default()
				.dst_set(mat_set)
				.dst_binding(0)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.image_info(std::slice::from_ref(&base_img)),
			ash::vk::WriteDescriptorSet::default()
				.dst_set(mat_set)
				.dst_binding(1)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.image_info(std::slice::from_ref(&orm_img)),
			ash::vk::WriteDescriptorSet::default()
				.dst_set(mat_set)
				.dst_binding(2)
				.descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.image_info(std::slice::from_ref(&normal_img)),
			ash::vk::WriteDescriptorSet::default()
				.dst_set(mat_set)
				.dst_binding(3)
				.descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
				.buffer_info(std::slice::from_ref(&mat_buf_info)),
		];
		unsafe { self.device.update_descriptor_sets(&writes, &[]) };
		Ok(mat_set)
	}

	/// Maximum number of distinct materials per render slot.
	#[allow(dead_code)]
	pub(crate) const fn max_materials(&self) -> usize {
		self.max_materials
	}
}

impl Drop for StandardSurfaceDescriptorArena {
	fn drop(&mut self) {
		unsafe {
			self.device.destroy_descriptor_pool(self.pool, None);
			self.device.destroy_sampler(self.sampler, None);
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

/// Create a graphics pipeline and immediately destroy its shader modules.
/// Vertex input layout matches MeshVertex (stride 48, locations 0–3).
fn create_graphics_pipeline_inner(
	device: &ash::Device,
	layout: ash::vk::PipelineLayout,
	vert_module: ash::vk::ShaderModule,
	frag_module: ash::vk::ShaderModule,
	depth: PipelineDepth,
	blend: PipelineBlend,
) -> Result<ash::vk::Pipeline> {
	let stages = [
		ash::vk::PipelineShaderStageCreateInfo::default()
			.stage(ash::vk::ShaderStageFlags::VERTEX)
			.module(vert_module)
			.name(c"main"),
		ash::vk::PipelineShaderStageCreateInfo::default()
			.stage(ash::vk::ShaderStageFlags::FRAGMENT)
			.module(frag_module)
			.name(c"main"),
	];

	// MeshVertex layout: position(0,f3,off0), normal(1,f3,off12),
	// uv(2,f2,off24), color(3,f4,off32) — stride 48.
	let binding = ash::vk::VertexInputBindingDescription::default()
		.binding(0)
		.stride(48)
		.input_rate(ash::vk::VertexInputRate::VERTEX);
	let attributes = [
		ash::vk::VertexInputAttributeDescription::default()
			.location(0)
			.binding(0)
			.format(ash::vk::Format::R32G32B32_SFLOAT)
			.offset(0),
		ash::vk::VertexInputAttributeDescription::default()
			.location(1)
			.binding(0)
			.format(ash::vk::Format::R32G32B32_SFLOAT)
			.offset(12),
		ash::vk::VertexInputAttributeDescription::default()
			.location(2)
			.binding(0)
			.format(ash::vk::Format::R32G32_SFLOAT)
			.offset(24),
		ash::vk::VertexInputAttributeDescription::default()
			.location(3)
			.binding(0)
			.format(ash::vk::Format::R32G32B32A32_SFLOAT)
			.offset(32),
	];
	let vertex_input = ash::vk::PipelineVertexInputStateCreateInfo::default()
		.vertex_binding_descriptions(std::slice::from_ref(&binding))
		.vertex_attribute_descriptions(&attributes);

	let assembly = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
		.topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);
	let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default()
		.viewport_count(1)
		.scissor_count(1);
	let raster = ash::vk::PipelineRasterizationStateCreateInfo::default()
		.polygon_mode(ash::vk::PolygonMode::FILL)
		.cull_mode(ash::vk::CullModeFlags::NONE)
		.front_face(ash::vk::FrontFace::COUNTER_CLOCKWISE)
		.line_width(1.0);
	let multisample = ash::vk::PipelineMultisampleStateCreateInfo::default()
		.rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);

	let depth_stencil = match depth {
		PipelineDepth::Disabled => ash::vk::PipelineDepthStencilStateCreateInfo::default(),
		PipelineDepth::WriteAndTest => ash::vk::PipelineDepthStencilStateCreateInfo::default()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(ash::vk::CompareOp::LESS)
			.min_depth_bounds(0.0)
			.max_depth_bounds(1.0),
		PipelineDepth::TestOnly => ash::vk::PipelineDepthStencilStateCreateInfo::default()
			.depth_test_enable(true)
			.depth_write_enable(false)
			.depth_compare_op(ash::vk::CompareOp::LESS)
			.min_depth_bounds(0.0)
			.max_depth_bounds(1.0),
	};

	let blend_attachment = match blend {
		PipelineBlend::Opaque => ash::vk::PipelineColorBlendAttachmentState::default()
			.color_write_mask(ash::vk::ColorComponentFlags::RGBA),
		PipelineBlend::PremultipliedAlpha => ash::vk::PipelineColorBlendAttachmentState::default()
			.blend_enable(true)
			.src_color_blend_factor(ash::vk::BlendFactor::ONE)
			.dst_color_blend_factor(ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(ash::vk::BlendOp::ADD)
			.src_alpha_blend_factor(ash::vk::BlendFactor::ONE)
			.dst_alpha_blend_factor(ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.alpha_blend_op(ash::vk::BlendOp::ADD)
			.color_write_mask(ash::vk::ColorComponentFlags::RGBA),
	};
	let color_blend = ash::vk::PipelineColorBlendStateCreateInfo::default()
		.attachments(std::slice::from_ref(&blend_attachment));

	let dynamic_states = [
		ash::vk::DynamicState::VIEWPORT,
		ash::vk::DynamicState::SCISSOR,
	];
	let dynamic = ash::vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
	let color_formats = [RENDER_COLOR_FORMAT];
	let mut rendering = ash::vk::PipelineRenderingCreateInfo::default()
		.color_attachment_formats(&color_formats)
		.depth_attachment_format(RENDER_DEPTH_FORMAT);
	let pipeline_info = ash::vk::GraphicsPipelineCreateInfo::default()
		.push_next(&mut rendering)
		.stages(&stages)
		.vertex_input_state(&vertex_input)
		.input_assembly_state(&assembly)
		.viewport_state(&viewport_state)
		.rasterization_state(&raster)
		.multisample_state(&multisample)
		.depth_stencil_state(&depth_stencil)
		.color_blend_state(&color_blend)
		.dynamic_state(&dynamic)
		.layout(layout);

	let result = unsafe {
		device.create_graphics_pipelines(
			ash::vk::PipelineCache::null(),
			std::slice::from_ref(&pipeline_info),
			None,
		)
	};
	unsafe {
		device.destroy_shader_module(vert_module, None);
		device.destroy_shader_module(frag_module, None);
	}
	match result {
		Ok(pipelines) => pipelines.first().copied().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"graphics-pipeline creation",
				std::io::Error::other("Vulkan returned no pipeline after successful creation"),
			)
		}),
		Err((partial, source)) => {
			for p in partial {
				destroy_pipeline(device, p);
			}
			Err(Error::backend_failure(
				"Vulkan",
				"graphics-pipeline creation",
				source,
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

fn create_shader_module(device: &ash::Device, words: &[u32]) -> Result<ash::vk::ShaderModule> {
	let info = ash::vk::ShaderModuleCreateInfo::default().code(words);
	unsafe { device.create_shader_module(&info, None) }
		.map_err(|s| Error::backend_failure("Vulkan", "graphics shader-module creation", s))
}

pub(super) struct ComputePipeline {
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	max_dispatch_group_count: [u32; 3],
	push_constant_size: u32,
}

impl ComputePipeline {
	pub(super) fn new(
		device: &ash::Device,
		descriptor_layout: ash::vk::DescriptorSetLayout,
		artifact: &'static ShaderArtifact,
		limits: DeviceLimits,
	) -> Result<Self> {
		let push_constant_size = artifact.push_constant_size()?;
		validate_artifact_limits(artifact.workgroup_size, push_constant_size, limits)?;

		let descriptor_layouts = [descriptor_layout];
		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(push_constant_size);
		let push_ranges = [push_range];
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.set_layouts(&descriptor_layouts)
			.push_constant_ranges(&push_ranges);
		// SAFETY: the shared descriptor layout is live and the reflected push range
		// fits the queried device limit and Vulkan's four-byte alignment requirements.
		let layout =
			unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(|source| {
				Error::backend_failure("Vulkan", "compute pipeline-layout creation", source)
			})?;

		let words = match artifact.spirv_words() {
			Ok(words) => words,
			Err(error) => {
				destroy_pipeline_layout(device, layout);
				return Err(error);
			}
		};
		let module_info = ash::vk::ShaderModuleCreateInfo::default().code(&words);
		// SAFETY: build-time spirv-val and runtime word validation proved a complete
		// SPIR-V word stream; the device remains live through pipeline creation.
		let module = match unsafe { device.create_shader_module(&module_info, None) } {
			Ok(module) => module,
			Err(source) => {
				destroy_pipeline_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute shader-module creation",
					source,
				));
			}
		};

		let stage = ash::vk::PipelineShaderStageCreateInfo::default()
			.stage(ash::vk::ShaderStageFlags::COMPUTE)
			.module(module)
			.name(c"main");
		let pipeline_info = ash::vk::ComputePipelineCreateInfo::default()
			.stage(stage)
			.layout(layout);
		// SAFETY: the module contains the reflected compute entry point, and the
		// pipeline layout exactly matches its validated descriptor and push ABI.
		let pipeline_result = unsafe {
			device.create_compute_pipelines(ash::vk::PipelineCache::null(), &[pipeline_info], None)
		};
		// The module is no longer needed after pipeline creation, successful or not.
		unsafe {
			device.destroy_shader_module(module, None);
		}
		let pipelines = match pipeline_result {
			Ok(pipelines) => pipelines,
			Err((partial, source)) => {
				for pipeline in partial {
					destroy_pipeline(device, pipeline);
				}
				destroy_pipeline_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute-pipeline creation",
					source,
				));
			}
		};
		let Some(handle) = pipelines.first().copied() else {
			destroy_pipeline_layout(device, layout);
			return Err(Error::backend_failure(
				"Vulkan",
				"compute-pipeline creation",
				std::io::Error::other("Vulkan returned no pipeline after successful creation"),
			));
		};

		Ok(Self {
			handle,
			layout,
			max_dispatch_group_count: limits.max_compute_work_group_count,
			push_constant_size,
		})
	}

	pub(super) fn destroy(&mut self, device: &ash::Device) {
		destroy_pipeline(device, self.handle);
		destroy_pipeline_layout(device, self.layout);
	}

	pub(super) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(super) const fn layout(&self) -> ash::vk::PipelineLayout {
		self.layout
	}

	pub(super) const fn max_dispatch_group_count(&self) -> [u32; 3] {
		self.max_dispatch_group_count
	}

	pub(super) const fn push_constant_size(&self) -> u32 {
		self.push_constant_size
	}
}

fn validate_artifact_limits(
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

fn destroy_pipeline(device: &ash::Device, pipeline: ash::vk::Pipeline) {
	// SAFETY: the handle belongs to `device` and is destroyed at most once after use.
	unsafe { device.destroy_pipeline(pipeline, None) }
}

fn destroy_pipeline_layout(device: &ash::Device, layout: ash::vk::PipelineLayout) {
	// SAFETY: the layout belongs to `device` and no live pipeline uses it.
	unsafe { device.destroy_pipeline_layout(layout, None) }
}
