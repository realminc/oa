//! Headless render-to-texture session with an explicit bounded target ring.
//!
//! One `Renderer` admits four frame sources:
//! - `begin_frame(texture)` — deep-copies a packed Texture (always available).
//! - `begin_frame_mesh(mesh, camera)` — vertex-color lit draw call.
//! - `begin_frame_flat_color(mesh, camera, material)` — solid-color draw call.
//! - `begin_frame_unlit(mesh, camera, material)` — textured unlit draw call.
//! - `begin_frame_scene(scene, camera)` — material-dispatched scene draw
//!   (Standard Surface PBR; requires graphics path and non-empty scene).
//!
//! `submit_frame` returns a [`RenderFrame`] carrying the color `Texture` and
//! exact producer `Event` for every path.
//!
//! Donor: `oa::Renderer` (`oacpp/source/cpp/include/oa/render/renderer.h`).

use std::{
	cell::RefCell,
	collections::HashMap,
	rc::{Rc, Weak},
};

use crate::{
	Color, Engine, Error, Event, Result,
	render::material::{FlatColorMaterial, StandardSurfaceMaterial, TextureHandle, UnlitMaterial},
	runtime::{
		FlatColorPipeline, GraphicsPipeline, MeshBuffer, NativeRgbaImage, RenderTarget, SceneDrawCmd,
		StandardSurfaceBlendPipeline, StandardSurfaceDescriptorArena, StandardSurfaceDescriptorLayouts,
		StandardSurfacePipeline, UnlitDescriptorArena, UnlitDescriptorLayouts, UnlitPipeline,
	},
};

use super::{
	Camera, MeshData, Scene, Texture, blit,
	scene::compile_scene_packets,
	texture::{RenderTargetLease, texture_from_render_target, texture_from_rgba8},
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for one headless [`Renderer`] target ring.
#[derive(Clone, Debug, PartialEq)]
pub struct RendererConfig {
	/// Target width in pixels. Must be non-zero.
	pub width: u32,
	/// Target height in pixels. Must be non-zero.
	pub height: u32,
	/// Number of simultaneously retained target slots (1–4).
	pub target_slot_count: u32,
	/// Maximum vertex count per mesh frame. Zero disables the graphics path.
	pub max_vertex_count: u32,
	/// Maximum index count per mesh frame. Zero disables the graphics path.
	pub max_index_count: u32,
	/// Background clear color (linear RGBA, each component 0.0–1.0).
	pub clear_color: Color,
	/// Normalized light direction (XYZ) pointing from the surface toward the
	/// light; used by the vertex-color-lit and standard-surface paths.
	pub light_direction: [f32; 3],
	/// Ambient light intensity (0.0–1.0) for the vertex-color-lit path.
	pub ambient: f32,
	/// Maximum number of distinct materials per `begin_frame_scene` call.
	///
	/// Must be in `[1, 64]`. Each material slot allocates one UBO and one
	/// descriptor set per render target slot. Default is `8`.
	pub max_materials_per_scene: u32,
}

impl Default for RendererConfig {
	fn default() -> Self {
		Self {
			width: 256,
			height: 192,
			target_slot_count: 3,
			max_vertex_count: 4096,
			max_index_count: 12288,
			clear_color: Color::new(0.02, 0.03, 0.05, 1.0),
			light_direction: [0.35, 0.82, 0.45],
			ambient: 0.24,
			max_materials_per_scene: 8,
		}
	}
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Engine-borrowing headless render-to-texture session.
///
/// Admits multiple frame sources through the same slot-ring lifecycle.
/// `submit_frame` returns a [`RenderFrame`] carrying the color `Texture` and
/// exact producer `Event`.
pub struct Renderer<'engine> {
	engine: &'engine Engine,
	config: RendererConfig,
	state: Rc<RefCell<RendererState>>,
}

/// Generation-safe lease for one submitted headless target.
#[must_use]
pub struct RenderFrame {
	slot: usize,
	generation: u64,
	texture: Texture,
	producer: Event,
	/// True when the slot has a graphics render target (not a buffer blit).
	is_graphics: bool,
	state: Weak<RefCell<RendererState>>,
	released: bool,
}

// ── Internal state ────────────────────────────────────────────────────────────

pub(super) struct RendererState {
	/// Vertex-color lit pipeline (original mesh path).
	pipeline: Option<GraphicsPipeline>,
	/// Flat-color material pipeline.
	flat_color_pipeline: Option<FlatColorPipeline>,
	/// Unlit textured material pipeline.
	unlit_pipeline: Option<UnlitPipeline>,
	/// One sampled-image descriptor set per target-ring slot.
	unlit_descriptors: Option<UnlitDescriptorArena>,
	/// Owned descriptor set layouts for the unlit pipeline (set 1).
	/// Must be declared after `unlit_pipeline` so it drops after the pipeline.
	_unlit_layouts: Option<UnlitDescriptorLayouts>,
	/// Standard Surface PBR material pipeline (opaque pass).
	standard_surface_pipeline: Option<StandardSurfacePipeline>,
	/// Standard Surface alpha-blend pipeline (transparent pass).
	/// Must be declared before `_standard_surface_layouts` so it drops before
	/// the layout — but after `standard_surface_pipeline` so destroy order is
	/// opaque then blend then layouts.
	standard_surface_blend_pipeline: Option<StandardSurfaceBlendPipeline>,
	standard_surface_descriptors: Option<StandardSurfaceDescriptorArena>,
	/// Owned descriptor set layouts for the Standard Surface pipeline (sets 0+1).
	/// Must be declared after `standard_surface_pipeline` so it drops after.
	_standard_surface_layouts: Option<StandardSurfaceDescriptorLayouts>,
	textures: HashMap<TextureHandle, Texture>,
	white_texture: Option<Texture>,
	slots: Vec<Slot>,
	building: Option<Building>,
	closed: bool,
}

struct Slot {
	generation: u64,
	state: SlotState,
	/// Present when the graphics path is enabled.
	graphics: Option<GraphicsSlot>,
	sampled_image: Option<Rc<NativeRgbaImage>>,
	texture_lease: Option<Weak<crate::render::texture::TextureSemantic>>,
}

struct GraphicsSlot {
	target: RenderTarget,
	vertex_buf: MeshBuffer,
	index_buf: MeshBuffer,
	scene_uniform: MeshBuffer,
	/// One UBO slot per material-descriptor index (`max_materials_per_scene`).
	material_uniforms: Vec<MeshBuffer>,
}

struct Building {
	slot: usize,
	generation: u64,
	source: BuildingSource,
}

/// Per-draw packet carried in `BuildingSource::StandardSurface`.
struct PendingDraw {
	index_start: u32,
	index_count: u32,
	/// 128-byte push block (MVP + model matrix).
	push_data: [u8; 128],
	/// Index into `material_uniforms` / descriptor arena for this draw.
	mat_index: usize,
	/// Serialized material params to upload before submit.
	material_data: Box<[u8; MATERIAL_PARAMS_SIZE]>,
	/// Optional registered texture handles for the three sampler bindings.
	base_texture: Option<TextureHandle>,
	orm_texture: Option<TextureHandle>,
	normal_texture: Option<TextureHandle>,
	/// True when this draw belongs to the transparent/blend pass.
	is_blend: bool,
}

enum BuildingSource {
	Texture(Texture),
	/// Vertex-color lit (original mesh path) — 80-byte push.
	Mesh {
		index_count: u32,
		push_data: [u8; 80],
	},
	/// Flat-color material — 96-byte push; VERT+FRAG stages.
	FlatColor {
		index_count: u32,
		push_data: [u8; 96],
	},
	/// Unlit textured material — 96-byte push; VERT+FRAG stages.
	Unlit {
		index_count: u32,
		push_data: [u8; 96],
		texture: Rc<NativeRgbaImage>,
	},
	/// Standard Surface PBR — per-draw packets sharing one scene UBO.
	StandardSurface {
		scene_data: Box<[u8; SCENE_LIGHTS_SIZE]>,
		draws: Vec<PendingDraw>,
	},
}

enum SlotState {
	Free,
	Building,
	Submitted(Event),
}

impl RendererState {
	pub(super) fn mark_texture_consumed(
		&mut self,
		slot_index: usize,
		generation: u64,
		event: &Event,
	) -> Result<()> {
		let slot = self
			.slots
			.get_mut(slot_index)
			.ok_or_else(|| Error::failed_precondition("render target slot is no longer available"))?;
		if slot.generation != generation {
			return Err(Error::failed_precondition(
				"render target slot belongs to a later frame",
			));
		}
		let SlotState::Submitted(current) = &mut slot.state else {
			return Err(Error::failed_precondition(
				"render target has no submitted producer",
			));
		};
		if !event.follows_or_equals(current) {
			return Err(Error::invalid_argument(
				"render target consumer event precedes its producer or another consumer",
			));
		}
		*current = event.clone();
		Ok(())
	}
}

impl Drop for RendererState {
	fn drop(&mut self) {
		let has_unretired_submission = self.slots.iter().any(|slot| match &slot.state {
			SlotState::Submitted(event) => !matches!(event.is_complete(), Ok(true)),
			SlotState::Free | SlotState::Building => false,
		});
		if !has_unretired_submission {
			return;
		}

		// Renderer Drop is not a completion boundary. If the caller omits `close`,
		// retain every Vulkan object reachable by a pending graphics command rather
		// than destroying live targets, buffers, descriptors, layouts, or pipelines.
		std::mem::forget(std::mem::take(&mut self.slots));
		std::mem::forget(std::mem::take(&mut self.textures));
		std::mem::forget(self.white_texture.take());
		std::mem::forget(self.pipeline.take());
		std::mem::forget(self.flat_color_pipeline.take());
		std::mem::forget(self.unlit_pipeline.take());
		std::mem::forget(self.unlit_descriptors.take());
		std::mem::forget(self._unlit_layouts.take());
		std::mem::forget(self.standard_surface_pipeline.take());
		std::mem::forget(self.standard_surface_descriptors.take());
		std::mem::forget(self._standard_surface_layouts.take());
	}
}

// ── Return-type aliases ───────────────────────────────────────────────────────

/// Resolved draw parameters from [`Renderer::resolve_graphics_source`].
type DrawParams = (
	u32,
	Vec<u8>,
	ash::vk::ShaderStageFlags,
	ash::vk::Pipeline,
	ash::vk::PipelineLayout,
	Vec<ash::vk::DescriptorSet>,
);

// ── Push-constant helpers ─────────────────────────────────────────────────────

const VERTEX_STRIDE: usize = 48;
const SCENE_LIGHTS_SIZE: usize = 416;
const MATERIAL_PARAMS_SIZE: usize = 80;

/// Build the 80-byte vertex-color-lit push block (MVP + light/ambient).
fn build_vcl_push(vp: &crate::vlm::Mat4, light: [f32; 3], ambient: f32) -> [u8; 80] {
	let mut data = [0u8; 80];
	for (i, row) in vp.m.iter().enumerate() {
		for (j, val) in row.iter().enumerate() {
			let off = (i * 4 + j) * 4;
			data[off..off + 4].copy_from_slice(&val.to_le_bytes());
		}
	}
	data[64..68].copy_from_slice(&light[0].to_le_bytes());
	data[68..72].copy_from_slice(&light[1].to_le_bytes());
	data[72..76].copy_from_slice(&light[2].to_le_bytes());
	data[76..80].copy_from_slice(&ambient.to_le_bytes());
	data
}

/// Build the 96-byte flat-color push block (MVP + material_color + opacity + pad).
fn build_flat_color_push(mvp: &crate::vlm::Mat4, mat: &FlatColorMaterial) -> [u8; 96] {
	let mut data = [0u8; 96];
	for (i, row) in mvp.m.iter().enumerate() {
		for (j, val) in row.iter().enumerate() {
			let off = (i * 4 + j) * 4;
			data[off..off + 4].copy_from_slice(&val.to_le_bytes());
		}
	}
	// offset 64: material_color (float4)
	for (k, c) in mat.color.to_array().iter().enumerate() {
		data[64 + k * 4..64 + k * 4 + 4].copy_from_slice(&c.to_le_bytes());
	}
	// offset 80: opacity (float) + 12 bytes padding
	data[80..84].copy_from_slice(&mat.opacity.to_le_bytes());
	data
}

/// Build the 96-byte unlit push block (MVP + tint + uv_scale + uv_offset).
fn build_unlit_push(mvp: &crate::vlm::Mat4, mat: &UnlitMaterial) -> [u8; 96] {
	let mut data = [0u8; 96];
	for (i, row) in mvp.m.iter().enumerate() {
		for (j, val) in row.iter().enumerate() {
			let off = (i * 4 + j) * 4;
			data[off..off + 4].copy_from_slice(&val.to_le_bytes());
		}
	}
	// offset 64: tint (float4)
	for (k, c) in mat.tint.to_array().iter().enumerate() {
		data[64 + k * 4..64 + k * 4 + 4].copy_from_slice(&c.to_le_bytes());
	}
	// offset 80: uv_scale (float2)
	data[80..84].copy_from_slice(&mat.uv_scale[0].to_le_bytes());
	data[84..88].copy_from_slice(&mat.uv_scale[1].to_le_bytes());
	// offset 88: uv_offset (float2)
	data[88..92].copy_from_slice(&mat.uv_offset[0].to_le_bytes());
	data[92..96].copy_from_slice(&mat.uv_offset[1].to_le_bytes());
	data
}

/// Build the 128-byte standard-surface push block (MVP + model matrix).
fn build_standard_surface_push(mvp: &crate::vlm::Mat4, model: &crate::vlm::Mat4) -> [u8; 128] {
	let mut data = [0u8; 128];
	for (i, row) in mvp.m.iter().enumerate() {
		for (j, val) in row.iter().enumerate() {
			let off = (i * 4 + j) * 4;
			data[off..off + 4].copy_from_slice(&val.to_le_bytes());
		}
	}
	for (i, row) in model.m.iter().enumerate() {
		for (j, val) in row.iter().enumerate() {
			let off = 64 + (i * 4 + j) * 4;
			data[off..off + 4].copy_from_slice(&val.to_le_bytes());
		}
	}
	data
}

fn build_scene_lights(config: &RendererConfig) -> [u8; SCENE_LIGHTS_SIZE] {
	let mut data = [0_u8; SCENE_LIGHTS_SIZE];
	for (index, value) in config.light_direction.iter().enumerate() {
		data[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
	}
	data[12..16].copy_from_slice(&1.0_f32.to_le_bytes());
	for offset in [16, 20, 24] {
		data[offset..offset + 4].copy_from_slice(&1.0_f32.to_le_bytes());
	}
	data[384..388].copy_from_slice(&1_i32.to_le_bytes());
	data[388..392].copy_from_slice(&0_i32.to_le_bytes());
	data[392..396].copy_from_slice(&1.0_f32.to_le_bytes());
	data[396..400].copy_from_slice(&config.ambient.to_le_bytes());
	for offset in [400, 404, 408] {
		data[offset..offset + 4].copy_from_slice(&1.0_f32.to_le_bytes());
	}
	data[412..416].copy_from_slice(&0_i32.to_le_bytes());
	data
}

fn build_material_params(material: &StandardSurfaceMaterial) -> [u8; MATERIAL_PARAMS_SIZE] {
	let mut data = [0_u8; MATERIAL_PARAMS_SIZE];
	let mut write = |offset: usize, value: f32| {
		data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
	};
	for (index, value) in material.base_color.iter().enumerate() {
		write(index * 4, *value);
	}
	write(12, material.metalness);
	write(16, material.roughness);
	write(20, material.normal_scale);
	write(24, material.specular_ior);
	write(28, material.emission_scale);
	for (index, value) in material.specular_tint.iter().enumerate() {
		write(32 + index * 4, *value);
	}
	write(44, material.opacity);
	for (index, value) in material.emission_color.iter().enumerate() {
		write(48 + index * 4, *value);
	}
	write(64, material.uv_scale[0]);
	write(68, material.uv_scale[1]);
	write(72, material.uv_offset[0]);
	write(76, material.uv_offset[1]);
	data
}

fn mesh_vertex_bytes(mesh: &MeshData) -> Vec<u8> {
	let mut out = Vec::with_capacity(mesh.vertices.len() * VERTEX_STRIDE);
	for v in &mesh.vertices {
		out.extend_from_slice(&v.position.x.to_le_bytes());
		out.extend_from_slice(&v.position.y.to_le_bytes());
		out.extend_from_slice(&v.position.z.to_le_bytes());
		out.extend_from_slice(&v.normal.x.to_le_bytes());
		out.extend_from_slice(&v.normal.y.to_le_bytes());
		out.extend_from_slice(&v.normal.z.to_le_bytes());
		out.extend_from_slice(&v.uv.x.to_le_bytes());
		out.extend_from_slice(&v.uv.y.to_le_bytes());
		out.extend_from_slice(&v.color.x.to_le_bytes());
		out.extend_from_slice(&v.color.y.to_le_bytes());
		out.extend_from_slice(&v.color.z.to_le_bytes());
		out.extend_from_slice(&v.color.w.to_le_bytes());
	}
	out
}

// ── impl Renderer ─────────────────────────────────────────────────────────────

impl<'engine> Renderer<'engine> {
	pub(crate) fn owns_frame(&self, frame: &RenderFrame) -> bool {
		frame
			.state
			.upgrade()
			.is_some_and(|state| Rc::ptr_eq(&state, &self.state))
			&& frame.texture.engine_handle().same_as(&self.engine.handle())
	}

	/// Create a bounded headless target ring borrowing `engine`.
	///
	/// When `max_vertex_count` / `max_index_count` are non-zero and the Engine
	/// has a graphics queue, all graphics pipelines are created and per-slot
	/// render targets and mesh buffers are allocated.
	///
	/// # Errors
	///
	/// Returns an error for zero extent, zero or >4 target slots, invalid
	/// light/color config, a missing graphics queue when mesh capacity is
	/// requested, or device allocation failure.
	pub fn new(engine: &'engine Engine, config: RendererConfig) -> Result<Self> {
		validate_config(&config)?;
		let mesh_enabled =
			config.max_vertex_count > 0 && config.max_index_count > 0 && engine.has_graphics();
		let handle = engine.handle();

		let (
			pipeline,
			flat_color_pipeline,
			unlit_pipeline,
			unlit_descriptors,
			unlit_layouts,
			standard_surface_pipeline,
			standard_surface_blend_pipeline,
			standard_surface_descriptors,
			standard_surface_layouts,
		) = if mesh_enabled {
			let vcl = handle.create_graphics_pipeline()?;
			let fc = handle.create_flat_color_pipeline()?;
			let unlit_layouts = handle.create_unlit_descriptor_layouts()?;
			let unlit_descriptors = handle
				.create_unlit_descriptor_arena(unlit_layouts.sampler_layout, config.target_slot_count)?;
			let unlit = handle.create_unlit_pipeline(unlit_layouts.sampler_layout)?;
			let ss_layouts = handle.create_standard_surface_descriptor_layouts()?;
			let ss_descriptors = handle.create_standard_surface_descriptor_arena(
				ss_layouts.scene_layout,
				ss_layouts.material_layout,
				config.target_slot_count,
				config.max_materials_per_scene,
			)?;
			let ss = handle
				.create_standard_surface_pipeline(ss_layouts.scene_layout, ss_layouts.material_layout)?;
			// The blend pipeline reuses the opaque pipeline's layout.
			let ss_blend = handle.create_standard_surface_blend_pipeline(ss.layout())?;
			(
				Some(vcl),
				Some(fc),
				Some(unlit),
				Some(unlit_descriptors),
				Some(unlit_layouts),
				Some(ss),
				Some(ss_blend),
				Some(ss_descriptors),
				Some(ss_layouts),
			)
		} else {
			(None, None, None, None, None, None, None, None, None)
		};
		let white_texture = mesh_enabled
			.then(|| texture_from_rgba8(engine, &[255, 255, 255, 255], 1, 1))
			.transpose()?;

		let vertex_cap = config.max_vertex_count as usize * VERTEX_STRIDE;
		let index_cap = config.max_index_count as usize * size_of::<u32>();
		let mut slots = Vec::with_capacity(config.target_slot_count as usize);
		for _ in 0..config.target_slot_count {
			let graphics = if mesh_enabled {
				let target = handle.allocate_render_target(config.width, config.height)?;
				let vertex_buf =
					handle.allocate_mesh_buffer(vertex_cap, ash::vk::BufferUsageFlags::VERTEX_BUFFER)?;
				let index_buf =
					handle.allocate_mesh_buffer(index_cap, ash::vk::BufferUsageFlags::INDEX_BUFFER)?;
				let scene_uniform = handle
					.allocate_mesh_buffer(SCENE_LIGHTS_SIZE, ash::vk::BufferUsageFlags::UNIFORM_BUFFER)?;
				let mut material_uniforms = Vec::with_capacity(config.max_materials_per_scene as usize);
				for _ in 0..config.max_materials_per_scene {
					material_uniforms.push(handle.allocate_mesh_buffer(
						MATERIAL_PARAMS_SIZE,
						ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
					)?);
				}
				Some(GraphicsSlot {
					target,
					vertex_buf,
					index_buf,
					scene_uniform,
					material_uniforms,
				})
			} else {
				None
			};
			slots.push(Slot {
				generation: 0,
				state: SlotState::Free,
				graphics,
				sampled_image: None,
				texture_lease: None,
			});
		}
		Ok(Self {
			engine,
			config,
			state: Rc::new(RefCell::new(RendererState {
				pipeline,
				flat_color_pipeline,
				unlit_pipeline,
				unlit_descriptors,
				_unlit_layouts: unlit_layouts,
				standard_surface_pipeline,
				standard_surface_blend_pipeline,
				standard_surface_descriptors,
				_standard_surface_layouts: standard_surface_layouts,
				textures: HashMap::new(),
				white_texture,
				slots,
				building: None,
				closed: false,
			})),
		})
	}

	/// Return this session's fixed configuration.
	pub fn config(&self) -> &RendererConfig {
		&self.config
	}

	/// Return whether the graphics draw paths are available.
	pub fn has_mesh_path(&self) -> bool {
		self.state.borrow().pipeline.is_some()
	}

	/// Associate a semantic material texture handle with one native Texture.
	///
	/// Replacing an existing handle affects future frames only. Submitted frames
	/// retain the exact native image they referenced until their producer event
	/// completes.
	pub fn register_texture(&self, handle: TextureHandle, texture: &Texture) -> Result<()> {
		if handle.0 == 0 {
			return Err(Error::invalid_argument(
				"render::Renderer texture handles must be non-zero",
			));
		}
		if !texture.engine_handle().same_as(&self.engine.handle()) {
			return Err(Error::invalid_argument(
				"render::Renderer registered Texture belongs to a different Engine",
			));
		}
		if texture.native().is_none() {
			return Err(Error::failed_precondition(
				"render::Renderer sampled textures require native image backing",
			));
		}
		let mut state = self.state.borrow_mut();
		if state.closed {
			return Err(Error::failed_precondition("render::Renderer is closed"));
		}
		state.textures.insert(handle, texture.clone());
		Ok(())
	}

	// ── Frame sources ─────────────────────────────────────────────────────────

	/// Begin one frame from a completed packed source `Texture` (blit path).
	pub fn begin_frame(&self, source: &Texture) -> Result<()> {
		if !source.engine_handle().same_as(&self.engine.handle()) {
			return Err(Error::invalid_argument(
				"render::Renderer source Texture belongs to a different Engine",
			));
		}
		if source.width() != self.config.width as usize
			|| source.height() != self.config.height as usize
		{
			return Err(Error::invalid_argument(format!(
				"render::Renderer source extent {}x{} does not match configured {}x{}",
				source.width(),
				source.height(),
				self.config.width,
				self.config.height,
			)));
		}
		let (slot, generation) = self.acquire_free_slot()?;
		match blit(source) {
			Ok(texture) => {
				self.state.borrow_mut().building = Some(Building {
					slot,
					generation,
					source: BuildingSource::Texture(texture),
				});
				Ok(())
			}
			Err(error) => {
				self.release_slot(slot, generation);
				Err(error)
			}
		}
	}

	/// Begin one frame using the vertex-color-lit pipeline.
	///
	/// Requires `has_mesh_path()`.
	pub fn begin_frame_mesh(&self, mesh: &MeshData, camera: &Camera) -> Result<()> {
		self.require_graphics("mesh")?;
		camera.validate()?;
		self.validate_mesh_capacity(mesh)?;
		let (slot_idx, generation) = self.acquire_free_slot()?;
		let vp = camera.get_view_projection_matrix();
		let push_data = build_vcl_push(&vp, self.config.light_direction, self.config.ambient);
		let index_count = u32::try_from(mesh.indices.len())
			.map_err(|_| Error::out_of_range("mesh index count overflows u32"))?;
		if let Err(e) = self.write_mesh_to_slot(slot_idx, mesh) {
			self.release_slot(slot_idx, generation);
			return Err(e);
		}
		self.state.borrow_mut().building = Some(Building {
			slot: slot_idx,
			generation,
			source: BuildingSource::Mesh {
				index_count,
				push_data,
			},
		});
		Ok(())
	}

	/// Begin one frame using the flat-color material pipeline.
	///
	/// Requires `has_mesh_path()`. `material` parameters are baked into the
	/// push constant block at record time — no GPU buffer is retained.
	pub fn begin_frame_flat_color(
		&self,
		mesh: &MeshData,
		camera: &Camera,
		material: &FlatColorMaterial,
	) -> Result<()> {
		self.require_graphics("flat_color")?;
		material.validate()?;
		camera.validate()?;
		self.validate_mesh_capacity(mesh)?;
		let (slot_idx, generation) = self.acquire_free_slot()?;
		let mvp = camera.get_view_projection_matrix();
		let push_data = build_flat_color_push(&mvp, material);
		let index_count = u32::try_from(mesh.indices.len())
			.map_err(|_| Error::out_of_range("mesh index count overflows u32"))?;
		if let Err(e) = self.write_mesh_to_slot(slot_idx, mesh) {
			self.release_slot(slot_idx, generation);
			return Err(e);
		}
		self.state.borrow_mut().building = Some(Building {
			slot: slot_idx,
			generation,
			source: BuildingSource::FlatColor {
				index_count,
				push_data,
			},
		});
		Ok(())
	}

	/// Begin one frame using the unlit textured material pipeline.
	///
	/// Requires `has_mesh_path()`. Material tint and UV transform are baked
	/// into the push block. `base_color_texture` resolves through
	/// [`Renderer::register_texture`]; `None` uses the renderer's retained white
	/// texture so tint-only unlit materials remain a valid draw path.
	pub fn begin_frame_unlit(
		&self,
		mesh: &MeshData,
		camera: &Camera,
		material: &UnlitMaterial,
	) -> Result<()> {
		self.require_graphics("unlit")?;
		material.validate()?;
		camera.validate()?;
		self.validate_mesh_capacity(mesh)?;
		let texture = {
			let state = self.state.borrow();
			let texture = if let Some(handle) = material.base_color_texture {
				state.textures.get(&handle).ok_or_else(|| {
					Error::invalid_argument(format!(
						"render::Renderer has no Texture registered for handle {}",
						handle.0
					))
				})?
			} else {
				state
					.white_texture
					.as_ref()
					.ok_or_else(|| Error::failed_precondition("unlit default texture unavailable"))?
			};
			texture
				.native()
				.ok_or_else(|| Error::failed_precondition("unlit Texture is not image-backed"))?
		};
		let (slot_idx, generation) = self.acquire_free_slot()?;
		let mvp = camera.get_view_projection_matrix();
		let push_data = build_unlit_push(&mvp, material);
		let index_count = u32::try_from(mesh.indices.len())
			.map_err(|_| Error::out_of_range("mesh index count overflows u32"))?;
		if let Err(e) = self.write_mesh_to_slot(slot_idx, mesh) {
			self.release_slot(slot_idx, generation);
			return Err(e);
		}
		self.state.borrow_mut().building = Some(Building {
			slot: slot_idx,
			generation,
			source: BuildingSource::Unlit {
				index_count,
				push_data,
				texture,
			},
		});
		Ok(())
	}

	/// Begin one frame by compiling and drawing a semantic `Scene`.
	///
	/// Requires `has_mesh_path()`. The scene is validated and compiled into a
	/// merged world-space mesh. Each visible mesh node with geometry produces
	/// one draw call; nodes with different materials bind different material
	/// UBOs and descriptor sets within the same render pass.
	///
	/// Material dispatch:
	/// - `Material::Standard` → Standard Surface PBR pipeline.
	/// - `None` (no assignment) → `StandardSurfaceMaterial::default()`.
	/// - Non-Standard materials and scenes exceeding `max_materials_per_scene`
	///   are rejected.
	///
	/// # Errors
	///
	/// Returns an error when the graphics path is unavailable, the scene fails
	/// validation or produces an empty mesh, the compiled vertex/index count
	/// exceeds configured capacity, more than `max_materials_per_scene`
	/// distinct materials are referenced, any assigned material is not
	/// `Material::Standard`, or buffer write fails.
	pub fn begin_frame_scene(&self, scene: &Scene, camera: &Camera) -> Result<()> {
		self.require_graphics("scene")?;
		camera.validate()?;
		let (compiled, packets) = compile_scene_packets(scene)?;
		if compiled.vertices.is_empty() || compiled.indices.is_empty() {
			return Err(Error::invalid_argument(
				"render::Renderer begin_frame_scene: compiled scene contains no visible geometry",
			));
		}
		self.validate_mesh_capacity(&compiled)?;

		// ── Build material UBO map ─────────────────────────────────────────
		// Map MaterialId (or None) → material UBO slot index (0-based).
		let max_mat = self.config.max_materials_per_scene as usize;
		let mut mat_slot_map: std::collections::HashMap<Option<crate::render::MaterialId>, usize> =
			std::collections::HashMap::new();
		for packet in &packets {
			let key = packet.material_id;
			if !mat_slot_map.contains_key(&key) {
				let next = mat_slot_map.len();
				if next >= max_mat {
					return Err(Error::resource_exhausted(
						"render::Renderer scene exceeds max_materials_per_scene; increase \
						 RendererConfig::max_materials_per_scene",
					));
				}
				mat_slot_map.insert(key, next);
			}
		}

		// ── Validate all materials are Standard ────────────────────────────
		let mvp = camera.get_view_projection_matrix();
		let model = crate::vlm::Mat4::identity();
		let push_data = build_standard_surface_push(&mvp, &model);
		let scene_data = Box::new(build_scene_lights(&self.config));

		let mut draws: Vec<PendingDraw> = Vec::with_capacity(packets.len());
		for packet in &packets {
			let mat_index = mat_slot_map[&packet.material_id];
			let material = packet
				.material_id
				.and_then(|id| scene.materials.iter().find(|m| m.id == id))
				.map(|entry| &entry.data);
			let standard = match material {
				Some(super::Material::Standard(m)) => m.clone(),
				Some(_) => {
					return Err(Error::invalid_argument(
						"render::Renderer scene path requires Standard materials; \
						 FlatColor and Unlit are not supported in begin_frame_scene",
					));
				}
				None => StandardSurfaceMaterial::default(),
			};
			draws.push(PendingDraw {
				index_start: packet.index_start,
				index_count: packet.index_count,
				push_data,
				mat_index,
				material_data: Box::new(build_material_params(&standard)),
				base_texture: standard.base_color_texture,
				orm_texture: standard.orm_texture,
				normal_texture: standard.normal_texture,
				is_blend: standard.alpha_mode == crate::render::material::AlphaMode::Blend,
			});
		}

		let (slot_idx, generation) = self.acquire_free_slot()?;
		if let Err(e) = self.write_mesh_to_slot(slot_idx, &compiled) {
			self.release_slot(slot_idx, generation);
			return Err(e);
		}
		self.state.borrow_mut().building = Some(Building {
			slot: slot_idx,
			generation,
			source: BuildingSource::StandardSurface { scene_data, draws },
		});
		Ok(())
	}

	/// Submit the in-progress frame and return its target lease and producer.
	pub fn submit_frame(&self) -> Result<RenderFrame> {
		let building = self
			.state
			.borrow_mut()
			.building
			.take()
			.ok_or_else(|| Error::failed_precondition("render::Renderer has no frame in progress"))?;

		match building.source {
			BuildingSource::Texture(texture) => {
				let producer = match self.engine.checkpoint() {
					Ok(event) => event,
					Err(error) => {
						self.release_slot(building.slot, building.generation);
						return Err(error);
					}
				};
				let mut state = self.state.borrow_mut();
				let slot = &mut state.slots[building.slot];
				if slot.generation != building.generation || !matches!(slot.state, SlotState::Building) {
					return Err(Error::failed_precondition(
						"render::Renderer frame lease was invalidated",
					));
				}
				slot.state = SlotState::Submitted(producer.clone());
				Ok(RenderFrame {
					slot: building.slot,
					generation: building.generation,
					texture,
					producer,
					is_graphics: false,
					state: Rc::downgrade(&self.state),
					released: false,
				})
			}

			// ── Standard Surface — multi-draw scene path ─────────────────────
			BuildingSource::StandardSurface { scene_data, draws } => {
				let handle = self.engine.handle();
				let producer = {
					let mut state = self.state.borrow_mut();
					// Resolve pipeline and arena handles.
					let p = state
						.standard_surface_pipeline
						.as_ref()
						.ok_or_else(|| Error::failed_precondition("standard-surface pipeline unavailable"))?
						.raw();
					let pl = state.standard_surface_pipeline.as_ref().unwrap().layout();
					// Write scene UBO and build per-draw SceneDrawCmds.
					let slot_ref = &mut state.slots[building.slot];
					let gfx = slot_ref
						.graphics
						.as_mut()
						.ok_or_else(|| Error::failed_precondition("standard-surface graphics slot missing"))?;
					gfx.scene_uniform.write_and_flush(scene_data.as_ref())?;
					let scene_raw = gfx.scene_uniform.raw();
					// Resolve the white-texture view for sampler fallback.
					let white_view = state
						.white_texture
						.as_ref()
						.and_then(|t| t.native())
						.map(|n| n.view())
						.ok_or_else(|| {
							Error::failed_precondition("standard-surface white fallback texture unavailable")
						})?;
					// Write scene descriptor set.
					let scene_set = state
						.standard_surface_descriptors
						.as_ref()
						.ok_or_else(|| {
							Error::failed_precondition("standard-surface descriptor arena unavailable")
						})?
						.write_scene_slot(building.slot, scene_raw, SCENE_LIGHTS_SIZE as u64)?;
					// Per-draw: write material UBO + descriptor set.
					let mut draw_cmds: Vec<SceneDrawCmd> = Vec::with_capacity(draws.len());
					for draw in &draws {
						let mat_uniform = state.slots[building.slot]
							.graphics
							.as_mut()
							.ok_or_else(|| {
								Error::failed_precondition("standard-surface graphics slot missing (material)")
							})?
							.material_uniforms
							.get(draw.mat_index)
							.ok_or_else(|| {
								Error::internal("standard-surface material UBO index exceeds allocated count")
							})?;
						mat_uniform.write_and_flush(draw.material_data.as_ref())?;
						let mat_raw = mat_uniform.raw();
						let base_view = draw
							.base_texture
							.and_then(|h| state.textures.get(&h))
							.and_then(|t| t.native())
							.map(|n| n.view());
						let orm_view = draw
							.orm_texture
							.and_then(|h| state.textures.get(&h))
							.and_then(|t| t.native())
							.map(|n| n.view());
						let normal_view = draw
							.normal_texture
							.and_then(|h| state.textures.get(&h))
							.and_then(|t| t.native())
							.map(|n| n.view());
						let mat_set = state
							.standard_surface_descriptors
							.as_ref()
							.unwrap()
							.write_material_slot(
								building.slot,
								draw.mat_index,
								mat_raw,
								MATERIAL_PARAMS_SIZE as u64,
								base_view,
								orm_view,
								normal_view,
								white_view,
							)?;
						draw_cmds.push(SceneDrawCmd {
							index_start: draw.index_start,
							index_count: draw.index_count,
							push_data: draw.push_data.to_vec(),
							material_set: mat_set,
							is_blend: draw.is_blend,
						});
					}
					// Depth-sort blend draws back-to-front by index_start (proxy for
					// submission order; a full depth sort requires world-space centroids
					// which are not available here without transform data).
					// Opaque draws come first in the slice; blend draws follow in reverse
					// index order (back = higher index_start).
					draw_cmds.sort_by(|a, b| match (a.is_blend, b.is_blend) {
						(false, true) => std::cmp::Ordering::Less,
						(true, false) => std::cmp::Ordering::Greater,
						(true, true) => b.index_start.cmp(&a.index_start),
						(false, false) => std::cmp::Ordering::Equal,
					});
					// Record and submit.
					let blend_p = state
						.standard_surface_blend_pipeline
						.as_ref()
						.ok_or_else(|| {
							Error::failed_precondition("standard-surface blend pipeline unavailable")
						})?
						.raw();
					let slot_mut = &mut state.slots[building.slot];
					let gfx_mut = slot_mut.graphics.as_mut().expect("graphics slot populated");
					let result = handle.record_and_submit_scene_frame(
						&mut gfx_mut.target,
						&gfx_mut.vertex_buf,
						&gfx_mut.index_buf,
						self.config.clear_color.to_array(),
						ash::vk::ShaderStageFlags::VERTEX,
						p,
						blend_p,
						pl,
						scene_set,
						&draw_cmds,
					);
					match result {
						Ok(event) => event,
						Err(e) => {
							slot_mut.state = SlotState::Free;
							slot_mut.sampled_image = None;
							return Err(e);
						}
					}
				};
				let texture = {
					let state = self.state.borrow();
					let gfx = state.slots[building.slot]
						.graphics
						.as_ref()
						.expect("graphics slot populated");
					texture_from_render_target(
						handle,
						self.state.clone(),
						RenderTargetLease {
							slot: building.slot,
							generation: building.generation,
						},
						gfx.target.color_image,
						gfx.target.color_view,
						gfx.target.width,
						gfx.target.height,
					)
				};
				{
					let mut state = self.state.borrow_mut();
					state.slots[building.slot].state = SlotState::Submitted(producer.clone());
					state.slots[building.slot].sampled_image = None;
					state.slots[building.slot].texture_lease = Some(texture.lease_weak());
				}
				Ok(RenderFrame {
					slot: building.slot,
					generation: building.generation,
					texture,
					producer,
					is_graphics: true,
					state: Rc::downgrade(&self.state),
					released: false,
				})
			}

			// ── Single-draw graphics sources ──────────────────────────────────
			source => {
				let (index_count, push_bytes, push_stages, pipeline_vk, pipeline_layout, descriptor_sets) =
					{
						let state = self.state.borrow();
						self.resolve_graphics_source(&state, building.slot, &source)
					}?;
				let sampled_image = match &source {
					BuildingSource::Unlit { texture, .. } => Some(texture.clone()),
					_ => None,
				};
				let handle = self.engine.handle();
				let producer = {
					let mut state = self.state.borrow_mut();
					let slot = &mut state.slots[building.slot];
					let gfx = slot.graphics.as_mut().expect("graphics slot populated");
					let result = handle.record_and_submit_mesh_frame(
						&mut gfx.target,
						&gfx.vertex_buf,
						&gfx.index_buf,
						index_count,
						self.config.clear_color.to_array(),
						&push_bytes,
						push_stages,
						pipeline_vk,
						pipeline_layout,
						&descriptor_sets,
					);
					match result {
						Ok(event) => event,
						Err(e) => {
							slot.state = SlotState::Free;
							slot.sampled_image = None;
							return Err(e);
						}
					}
				};
				let texture = {
					let state = self.state.borrow();
					let gfx = state.slots[building.slot]
						.graphics
						.as_ref()
						.expect("graphics slot populated");
					texture_from_render_target(
						handle,
						self.state.clone(),
						RenderTargetLease {
							slot: building.slot,
							generation: building.generation,
						},
						gfx.target.color_image,
						gfx.target.color_view,
						gfx.target.width,
						gfx.target.height,
					)
				};
				{
					let mut state = self.state.borrow_mut();
					state.slots[building.slot].state = SlotState::Submitted(producer.clone());
					state.slots[building.slot].sampled_image = sampled_image;
					state.slots[building.slot].texture_lease = Some(texture.lease_weak());
				}
				Ok(RenderFrame {
					slot: building.slot,
					generation: building.generation,
					texture,
					producer,
					is_graphics: true,
					state: Rc::downgrade(&self.state),
					released: false,
				})
			}
		}
	}

	/// Cancel the in-progress frame without submitting or waiting.
	pub fn cancel_frame(&self) -> Result<()> {
		let building = self
			.state
			.borrow_mut()
			.building
			.take()
			.ok_or_else(|| Error::failed_precondition("render::Renderer has no frame in progress"))?;
		self.release_slot(building.slot, building.generation);
		Ok(())
	}

	/// Recycle submitted slots whose exact producer event has completed.
	pub fn collect(&self) -> Result<usize> {
		let mut state = self.state.borrow_mut();
		let mut released = 0;
		for slot in &mut state.slots {
			if let SlotState::Submitted(event) = &slot.state
				&& event.is_complete()?
				&& slot
					.texture_lease
					.as_ref()
					.is_none_or(|lease| lease.strong_count() == 0)
			{
				slot.state = SlotState::Free;
				slot.sampled_image = None;
				slot.texture_lease = None;
				released += 1;
			}
		}
		Ok(released)
	}

	/// Wait for outstanding submitted frames and permanently close the session.
	pub fn close(&self) -> Result<()> {
		let events = {
			let mut state = self.state.borrow_mut();
			if state.closed {
				return Ok(());
			}
			if state.building.is_some() {
				return Err(Error::failed_precondition(
					"render::Renderer cannot close with a frame in progress; submit or cancel it",
				));
			}
			state.closed = true;
			state
				.slots
				.iter()
				.filter_map(|slot| match &slot.state {
					SlotState::Submitted(event) => Some(event.clone()),
					_ => None,
				})
				.collect::<Vec<_>>()
		};
		for event in events {
			event.wait()?;
		}
		self.collect()?;
		Ok(())
	}

	// ── Private helpers ───────────────────────────────────────────────────────

	fn require_graphics(&self, path: &str) -> Result<()> {
		if self.state.borrow().pipeline.is_none() {
			Err(Error::failed_precondition(format!(
				"render::Renderer {path} path requires a graphics-capable Engine and non-zero \
				 max_vertex_count/max_index_count in RendererConfig",
			)))
		} else {
			Ok(())
		}
	}

	fn validate_mesh_capacity(&self, mesh: &MeshData) -> Result<()> {
		if mesh.vertices.is_empty() || mesh.indices.is_empty() || !mesh.indices.len().is_multiple_of(3)
		{
			return Err(Error::invalid_argument(
				"render::Renderer requires a non-empty indexed triangle-list mesh",
			));
		}
		if mesh.vertices.len() > self.config.max_vertex_count as usize {
			return Err(Error::resource_exhausted(
				"render::Renderer mesh vertex count exceeds max_vertex_count",
			));
		}
		if mesh.indices.len() > self.config.max_index_count as usize {
			return Err(Error::resource_exhausted(
				"render::Renderer mesh index count exceeds max_index_count",
			));
		}
		Ok(())
	}

	fn write_mesh_to_slot(&self, slot_idx: usize, mesh: &MeshData) -> Result<()> {
		let state = self.state.borrow();
		let gfx = state.slots[slot_idx]
			.graphics
			.as_ref()
			.expect("mesh slot has graphics");
		let vbytes = mesh_vertex_bytes(mesh);
		let ibytes: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
		gfx
			.vertex_buf
			.write_and_flush(&vbytes)
			.and_then(|()| gfx.index_buf.write_and_flush(&ibytes))
	}

	/// Resolve a graphics `BuildingSource` to the raw Vulkan handles and push
	/// bytes needed by `record_and_submit_mesh_frame`.
	///
	/// Returns owned push bytes so the `RendererState` borrow can be released
	/// before the bytes are passed to `record_and_submit_mesh_frame`.
	fn resolve_graphics_source(
		&self,
		state: &RendererState,
		slot: usize,
		source: &BuildingSource,
	) -> Result<DrawParams> {
		match source {
			BuildingSource::Mesh {
				index_count,
				push_data,
			} => {
				let p = state.pipeline.as_ref().expect("vcl pipeline live");
				Ok((
					*index_count,
					push_data.to_vec(),
					ash::vk::ShaderStageFlags::VERTEX,
					p.raw(),
					p.layout(),
					Vec::new(),
				))
			}
			BuildingSource::FlatColor {
				index_count,
				push_data,
			} => {
				let p = state
					.flat_color_pipeline
					.as_ref()
					.ok_or_else(|| Error::failed_precondition("flat-color pipeline unavailable"))?;
				Ok((
					*index_count,
					push_data.to_vec(),
					ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT,
					p.raw(),
					p.layout(),
					Vec::new(),
				))
			}
			BuildingSource::Unlit {
				index_count,
				push_data,
				texture,
			} => {
				let p = state
					.unlit_pipeline
					.as_ref()
					.ok_or_else(|| Error::failed_precondition("unlit pipeline unavailable"))?;
				let descriptors = state
					.unlit_descriptors
					.as_ref()
					.ok_or_else(|| Error::failed_precondition("unlit descriptor arena unavailable"))?;
				let set = descriptors.write_sampled_image(slot, texture.view())?;
				Ok((
					*index_count,
					push_data.to_vec(),
					ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT,
					p.raw(),
					p.layout(),
					vec![set],
				))
			}
			BuildingSource::StandardSurface { .. } => {
				unreachable!("StandardSurface source handled in the StandardSurface match arm")
			}
			BuildingSource::Texture(_) => {
				unreachable!("Texture source handled before resolve_graphics_source")
			}
		}
	}

	fn acquire_free_slot(&self) -> Result<(usize, u64)> {
		let mut state = self.state.borrow_mut();
		if state.closed {
			return Err(Error::failed_precondition("render::Renderer is closed"));
		}
		if state.building.is_some() {
			return Err(Error::failed_precondition(
				"render::Renderer already has a frame in progress",
			));
		}
		let Some((idx, entry)) = state
			.slots
			.iter_mut()
			.enumerate()
			.find(|(_, s)| matches!(s.state, SlotState::Free))
		else {
			return Err(Error::resource_exhausted(
				"render::Renderer target ring has no reusable slot; consume or collect a frame",
			));
		};
		entry.generation = entry.generation.checked_add(1).ok_or_else(|| {
			Error::resource_exhausted("render::Renderer target generation space is exhausted")
		})?;
		entry.sampled_image = None;
		entry.texture_lease = None;
		entry.state = SlotState::Building;
		Ok((idx, entry.generation))
	}

	fn release_slot(&self, slot: usize, generation: u64) {
		let mut state = self.state.borrow_mut();
		if let Some(entry) = state.slots.get_mut(slot)
			&& entry.generation == generation
		{
			entry.state = SlotState::Free;
			entry.sampled_image = None;
			entry.texture_lease = None;
		}
	}
}

// ── impl RenderFrame ──────────────────────────────────────────────────────────

impl RenderFrame {
	/// Borrow the color target texture for this frame.
	pub const fn color(&self) -> &Texture {
		&self.texture
	}

	/// Return the exact producer completion event for this target.
	pub const fn producer(&self) -> &Event {
		&self.producer
	}

	/// Return the target width in pixels.
	pub fn width(&self) -> usize {
		self.texture.width()
	}

	/// Return the target height in pixels.
	pub fn height(&self) -> usize {
		self.texture.height()
	}

	/// Explicitly wait and read back RGBA8 pixels, releasing the ring slot.
	pub fn consume_readback(mut self) -> Result<Vec<u8>> {
		let result = if self.is_graphics {
			self.producer.wait()?;
			if let Some(state) = self.state.upgrade() {
				let state = state.borrow();
				state.slots[self.slot]
					.graphics
					.as_ref()
					.ok_or_else(|| Error::failed_precondition("render::RenderFrame graphics slot missing"))?
					.target
					.read_color_rgba8()
			} else {
				Err(Error::failed_precondition(
					"render::RenderFrame renderer has been dropped",
				))
			}
		} else {
			self.texture.read_rgba8()
		};
		if result.is_ok() {
			self.release_completed();
		} else {
			self.do_release();
		}
		self.released = true;
		result
	}

	/// Explicitly abandon this target without waiting.
	pub fn abandon(mut self) {
		self.do_release();
		self.released = true;
	}

	fn do_release(&self) {
		if let Some(state) = self.state.upgrade() {
			let mut state = state.borrow_mut();
			if let Some(slot) = state.slots.get_mut(self.slot)
				&& slot.generation == self.generation
				&& !matches!(&slot.state, SlotState::Submitted(event) if event.follows_or_equals(&self.producer))
			{
				slot.state = SlotState::Submitted(self.producer.clone());
			}
		}
	}

	fn release_completed(&self) {
		if let Some(state) = self.state.upgrade() {
			let mut state = state.borrow_mut();
			if let Some(slot) = state.slots.get_mut(self.slot)
				&& slot.generation == self.generation
				&& self.texture.lease_strong_count() == 1
				&& matches!(&slot.state, SlotState::Submitted(event) if matches!(event.is_complete(), Ok(true)))
			{
				slot.state = SlotState::Free;
				slot.sampled_image = None;
				slot.texture_lease = None;
			}
		}
	}
}

impl Drop for RenderFrame {
	fn drop(&mut self) {
		if !self.released {
			self.do_release();
		}
	}
}

// ── Config validation ─────────────────────────────────────────────────────────

fn validate_config(config: &RendererConfig) -> Result<()> {
	if config.width == 0
		|| config.height == 0
		|| config.target_slot_count == 0
		|| config.target_slot_count > 4
	{
		return Err(Error::invalid_argument(
			"render::RendererConfig requires non-zero extent and 1–4 target slots",
		));
	}
	if !config
		.clear_color
		.to_array()
		.iter()
		.all(|v| v.is_finite() && (0.0..=1.0).contains(v))
	{
		return Err(Error::invalid_argument(
			"render::RendererConfig clear_color must be finite values in [0, 1]",
		));
	}
	if !config.light_direction.iter().all(|v| v.is_finite())
		|| !config.ambient.is_finite()
		|| !(0.0..=1.0).contains(&config.ambient)
	{
		return Err(Error::invalid_argument(
			"render::RendererConfig light_direction must be finite; ambient must be in [0, 1]",
		));
	}
	if config.max_materials_per_scene == 0 || config.max_materials_per_scene > 64 {
		return Err(Error::invalid_argument(
			"render::RendererConfig max_materials_per_scene must be in [1, 64]",
		));
	}
	Ok(())
}
