//! Semantic material values — CPU-side only, no GPU handles.
//!
//! Three tiers cover the full fidelity range without shared
//! over-parameterization:
//!
//! - [`FlatColorMaterial`] — solid-color UI and debug geometry; zero textures,
//!   no lighting.
//! - [`UnlitMaterial`] — textured image/video display; one texture, no
//!   lighting.
//! - [`StandardSurfaceMaterial`] — a Standard Surface/OpenPBR-oriented semantic
//!   parameter value. The current renderer lowers only its scalar base GGX,
//!   direct-light, ambient, and emission subset.
//!
//! All color and intensity values are **linear** at this API boundary. sRGB
//! conversion is handled by the Vulkan image-view format, not in shaders.
//!
//! Shading follows the OA material and lighting contracts.

use crate::{Color, Error, Result};

// ── Ids and handles ───────────────────────────────────────────────────────────

/// Opaque index into the owning [`Scene`]'s texture registry.
///
/// A handle is valid only within the `Scene` that issued it. Zero is reserved
/// and will be rejected by `validate_scene`.
///
/// [`Scene`]: super::scene::Scene
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u64);

/// Stable identity of one material in a [`Scene`]. Zero is invalid.
///
/// [`Scene`]: super::scene::Scene
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u64);

// ── AlphaMode ─────────────────────────────────────────────────────────────────

/// Alpha blending mode for [`UnlitMaterial`] and [`StandardSurfaceMaterial`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaMode {
	/// Alpha is ignored; surface is drawn in the opaque pass with depth write.
	#[default]
	Opaque,
	/// Fragment is discarded when `alpha < alpha_cutoff`; drawn in the opaque
	/// pass (avoids blending cost).
	AlphaTest,
	/// Premultiplied alpha blend over the existing color buffer; drawn in a
	/// back-to-front transparent pass with depth write disabled.
	Blend,
}

// ── FlatColorMaterial ─────────────────────────────────────────────────────────

/// Solid-color material for UI elements and debug geometry.
///
/// Zero texture bindings; lighting is disabled. Matches the `flat_color`
/// Slang shader variant.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatColorMaterial {
	/// Linear RGBA color, each component in `[0, 1]`.
	pub color: Color,
	/// Final alpha multiplier applied after `color.a`, in `[0, 1]`.
	pub opacity: f32,
}

impl Default for FlatColorMaterial {
	/// Opaque white.
	fn default() -> Self {
		Self {
			color: Color::new(1.0, 1.0, 1.0, 1.0),
			opacity: 1.0,
		}
	}
}

impl FlatColorMaterial {
	/// Validate that all components are finite and in `[0, 1]`.
	pub fn validate(&self) -> Result<()> {
		if self
			.color
			.to_array()
			.iter()
			.any(|c| !c.is_finite() || *c < 0.0 || *c > 1.0)
			|| !self.opacity.is_finite()
			|| self.opacity < 0.0
			|| self.opacity > 1.0
		{
			return Err(Error::invalid_argument(
				"render::FlatColorMaterial color and opacity must be finite and in [0, 1]",
			));
		}
		Ok(())
	}
}

// ── UnlitMaterial ─────────────────────────────────────────────────────────────

/// Textured unlit material for image and video display.
///
/// One optional base-color texture; no lighting computation. Matches the
/// `unlit` Slang shader variant.
#[derive(Clone, Debug, PartialEq)]
pub struct UnlitMaterial {
	/// Optional base-color texture (sRGB RGBA8). When `None`, `tint` is used
	/// as the solid color.
	pub base_color_texture: Option<TextureHandle>,
	/// Linear RGBA tint multiplied with the texture sample (or used directly
	/// when no texture is bound). Default `[1, 1, 1, 1]`.
	pub tint: Color,
	/// UV scale applied before sampling. Default `[1, 1]`.
	pub uv_scale: [f32; 2],
	/// UV offset applied after scaling. Default `[0, 0]`.
	pub uv_offset: [f32; 2],
	/// Alpha mode for this surface.
	pub alpha_mode: AlphaMode,
}

impl Default for UnlitMaterial {
	/// Opaque white, no texture, identity UV transform.
	fn default() -> Self {
		Self {
			base_color_texture: None,
			tint: Color::new(1.0, 1.0, 1.0, 1.0),
			uv_scale: [1.0, 1.0],
			uv_offset: [0.0, 0.0],
			alpha_mode: AlphaMode::Opaque,
		}
	}
}

impl UnlitMaterial {
	/// Validate that all scalar components are finite.
	pub fn validate(&self) -> Result<()> {
		if self.base_color_texture.is_some_and(|handle| handle.0 == 0)
			|| self.tint.to_array().iter().any(|c| !c.is_finite())
			|| self.uv_scale.iter().any(|c| !c.is_finite())
			|| self.uv_offset.iter().any(|c| !c.is_finite())
		{
			return Err(Error::invalid_argument(
				"render::UnlitMaterial tint, uv_scale, and uv_offset must be finite",
			));
		}
		Ok(())
	}
}

// ── StandardSurfaceMaterial ───────────────────────────────────────────────────

/// Full Standard Surface PBR material.
///
/// Parameterized on Autodesk Standard Surface / OpenPBR. Maps losslessly to
/// glTF 2.0 `pbrMetallicRoughness` + `KHR_materials_ior`, `KHR_materials_clearcoat`,
/// `KHR_materials_sheen`, and `KHR_materials_subsurface_scattering`. Arnold
/// and Cycles `standard_surface` import is a natural correspondence.
///
/// **Texture channel packing (ORM convention):**
/// - `orm_texture` R = ambient occlusion, G = roughness, B = metalness.
/// - `normal_texture` stores tangent-space XY as BC5 SNORM; Z is reconstructed
///   as `sqrt(saturate(1 - dot(n.xy, n.xy)))`.
///
/// All color values are **linear** at this API boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardSurfaceMaterial {
	// ── Base lobe ─────────────────────────────────────────────────────────────
	/// Linear RGB base color. Default `[0.8, 0.8, 0.8]`.
	pub base_color: [f32; 3],
	/// Optional base-color texture. Format admission belongs to the renderer.
	pub base_color_texture: Option<TextureHandle>,

	/// Metalness in `[0, 1]`. Default `0.0` (dielectric).
	pub metalness: f32,
	/// Perceptual roughness in `[0, 1]`. Default `0.5`. Squared to α in the
	/// shader to match Unreal/Filament behaviour.
	pub roughness: f32,
	/// Optional ORM texture (R=occlusion, G=roughness, B=metalness).
	pub orm_texture: Option<TextureHandle>,

	// ── Normal ────────────────────────────────────────────────────────────────
	/// Optional tangent-space normal texture (BC5 SNORM XY).
	pub normal_texture: Option<TextureHandle>,
	/// Normal map strength multiplier. Default `1.0`.
	pub normal_scale: f32,

	// ── Specular ──────────────────────────────────────────────────────────────
	/// Index of refraction for dielectric F0 derivation. Default `1.5`
	/// (glass/plastic → F0 ≈ 0.04).
	pub specular_ior: f32,
	/// Linear RGB tint applied to the specular F0. Default `[1, 1, 1]`.
	pub specular_tint: [f32; 3],

	// ── Emission ──────────────────────────────────────────────────────────────
	/// Linear RGB emissive color. Default `[0, 0, 0]`.
	pub emission_color: [f32; 3],
	/// HDR intensity multiplier for emission. Default `0.0`.
	pub emission_scale: f32,
	/// Optional emissive texture (BC6H UFLOAT or RGBA16F).
	pub emission_texture: Option<TextureHandle>,

	// ── Clear coat ────────────────────────────────────────────────────────────
	/// Clear coat weight in `[0, 1]`. Default `0.0`.
	pub coat_weight: f32,
	/// Clear coat perceptual roughness in `[0, 1]`. Default `0.03`.
	pub coat_roughness: f32,
	/// Clear coat IOR. Default `1.5`.
	pub coat_ior: f32,
	/// Optional separate tangent-space normal for the coat layer.
	pub coat_normal_texture: Option<TextureHandle>,

	// ── Sheen (fabric/cloth) ──────────────────────────────────────────────────
	/// Sheen weight in `[0, 1]`. Default `0.0`.
	pub sheen_weight: f32,
	/// Linear RGB sheen color. Default `[1, 1, 1]`.
	pub sheen_color: [f32; 3],
	/// Sheen perceptual roughness in `[0, 1]`. Default `0.3`.
	pub sheen_roughness: f32,

	// ── Subsurface scattering ─────────────────────────────────────────────────
	/// Subsurface weight in `[0, 1]`. Default `0.0`. When `> 0.5` the diffuse
	/// lobe is replaced by the pre-integrated normalized diffusion approximation.
	pub subsurface_weight: f32,
	/// Linear RGB subsurface color. Default `[1, 1, 1]`.
	pub subsurface_color: [f32; 3],
	/// Per-channel scattering distance in world units (RGB). Default `[1, 1, 1]`.
	pub subsurface_radius: [f32; 3],

	// ── Alpha ─────────────────────────────────────────────────────────────────
	/// Global opacity multiplier in `[0, 1]`. Default `1.0`.
	pub opacity: f32,
	/// Optional single-channel (R) opacity texture.
	pub opacity_texture: Option<TextureHandle>,
	/// Alpha mode for this surface.
	pub alpha_mode: AlphaMode,
	/// Alpha cutoff threshold for `AlphaMode::AlphaTest`. Default `0.5`.
	pub alpha_cutoff: f32,

	// ── UV transform (shared across all maps) ─────────────────────────────────
	/// UV scale applied before sampling all maps. Default `[1, 1]`.
	pub uv_scale: [f32; 2],
	/// UV offset applied after scaling. Default `[0, 0]`.
	pub uv_offset: [f32; 2],
}

impl Default for StandardSurfaceMaterial {
	/// Neutral gray dielectric: `base_color = [0.8, 0.8, 0.8]`, `roughness = 0.5`,
	/// `metalness = 0.0`, all optional textures `None`, all optional lobes at
	/// weight `0.0`.
	fn default() -> Self {
		Self {
			base_color: [0.8, 0.8, 0.8],
			base_color_texture: None,
			metalness: 0.0,
			roughness: 0.5,
			orm_texture: None,
			normal_texture: None,
			normal_scale: 1.0,
			specular_ior: 1.5,
			specular_tint: [1.0, 1.0, 1.0],
			emission_color: [0.0, 0.0, 0.0],
			emission_scale: 0.0,
			emission_texture: None,
			coat_weight: 0.0,
			coat_roughness: 0.03,
			coat_ior: 1.5,
			coat_normal_texture: None,
			sheen_weight: 0.0,
			sheen_color: [1.0, 1.0, 1.0],
			sheen_roughness: 0.3,
			subsurface_weight: 0.0,
			subsurface_color: [1.0, 1.0, 1.0],
			subsurface_radius: [1.0, 1.0, 1.0],
			opacity: 1.0,
			opacity_texture: None,
			alpha_mode: AlphaMode::Opaque,
			alpha_cutoff: 0.5,
			uv_scale: [1.0, 1.0],
			uv_offset: [0.0, 0.0],
		}
	}
}

impl StandardSurfaceMaterial {
	/// Validate that all scalar parameters are finite and in-range.
	pub fn validate(&self) -> Result<()> {
		let in_unit = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
		let finite3 = |c: [f32; 3]| c.iter().all(|v| v.is_finite());
		let finite2 = |c: [f32; 2]| c.iter().all(|v| v.is_finite());
		let texture_handles = [
			self.base_color_texture,
			self.orm_texture,
			self.normal_texture,
			self.emission_texture,
			self.coat_normal_texture,
			self.opacity_texture,
		];
		if texture_handles
			.into_iter()
			.flatten()
			.any(|handle| handle.0 == 0)
			|| !finite3(self.base_color)
			|| !in_unit(self.metalness)
			|| !in_unit(self.roughness)
			|| !self.normal_scale.is_finite()
			|| !self.specular_ior.is_finite()
			|| self.specular_ior <= 0.0
			|| !finite3(self.specular_tint)
			|| !finite3(self.emission_color)
			|| !self.emission_scale.is_finite()
			|| self.emission_scale < 0.0
			|| !in_unit(self.coat_weight)
			|| !in_unit(self.coat_roughness)
			|| !self.coat_ior.is_finite()
			|| self.coat_ior <= 0.0
			|| !in_unit(self.sheen_weight)
			|| !finite3(self.sheen_color)
			|| !in_unit(self.sheen_roughness)
			|| !in_unit(self.subsurface_weight)
			|| !finite3(self.subsurface_color)
			|| !finite3(self.subsurface_radius)
			|| self.subsurface_radius.iter().any(|v| *v < 0.0)
			|| !in_unit(self.opacity)
			|| !in_unit(self.alpha_cutoff)
			|| !finite2(self.uv_scale)
			|| !finite2(self.uv_offset)
		{
			return Err(Error::invalid_argument(
				"render::StandardSurfaceMaterial parameters must be finite and in-range",
			));
		}
		Ok(())
	}
}

// ── Material ──────────────────────────────────────────────────────────────────

/// Semantic material value — CPU-side only, no GPU handles.
///
/// Three tiers correspond to distinct shader pipelines:
/// - [`Material::FlatColor`] — `flat_color` shader, UI and debug.
/// - [`Material::Unlit`] — `unlit` shader, image/video display.
/// - [`Material::Standard`] — `standard_surface` shader, full PBR.
#[derive(Clone, Debug, PartialEq)]
pub enum Material {
	FlatColor(FlatColorMaterial),
	Unlit(UnlitMaterial),
	Standard(StandardSurfaceMaterial),
}

impl Default for Material {
	/// Neutral gray opaque `Standard` material.
	fn default() -> Self {
		Self::Standard(StandardSurfaceMaterial::default())
	}
}

impl Material {
	/// Validate the enclosed material's scalar parameters.
	pub fn validate(&self) -> Result<()> {
		match self {
			Self::FlatColor(m) => m.validate(),
			Self::Unlit(m) => m.validate(),
			Self::Standard(m) => m.validate(),
		}
	}
}
