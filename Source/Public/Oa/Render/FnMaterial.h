// OaFnMaterial — Functional material operations.
//
// Stateless material math and parameter packing. Operates on OaMaterialData POD.
// For the OOP wrapper, see <Oa/Render/Material.h>.
//
// Usage:
//   OaMaterialData mat = OaFnMaterial::CreateUnlit();
//   OaFnMaterial::SetAlbedoTexture(mat, texture);
//   OaFnMaterial::SetBlendMode(mat, OaBlendMode::AlphaBlend);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Vlm.h>

// Blend mode
enum class OaBlendMode : OaU8 {
	Opaque,      // No blending
	AlphaBlend,  // Standard alpha blending
	Additive,    // Additive blending
	Multiply,    // Multiply blending
};

// Culling mode
enum class OaCullMode : OaU8 {
	None,          // No culling
	Back,          // Back-face culling (default)
	Front,         // Front-face culling
	FrontAndBack,  // Both sides
};

// Material type
enum class OaMaterialType : OaU8 {
	Unlit,     // Texture-only, no lighting (image/video viewer, UI)
	Lit,       // Basic lit with ambient + diffuse
	Pbr,       // Full PBR (albedo, roughness, metallic, normal, AO)
};

// CPU-side material data (POD, GPU upload is separate)
struct OaMaterialData {
	// Type
	OaMaterialType Type = OaMaterialType::Unlit;

	// Scalar parameters
	VlmVec4 AlbedoColor       = {1.0f, 1.0f, 1.0f, 1.0f};
	OaF32  Roughness         = 0.5f;
	OaF32  Metallic          = 0.0f;
	OaF32  Ao                = 1.0f;
	VlmVec4 EmissiveColor     = {0.0f, 0.0f, 0.0f, 1.0f};
	OaF32  EmissiveIntensity = 0.0f;

	// Render state
	OaBlendMode BlendMode    = OaBlendMode::Opaque;
	OaCullMode  CullMode     = OaCullMode::Back;
	bool        DoubleSided  = false;
	bool        DepthTest    = true;
	bool        DepthWrite   = true;

	// Texture bindless indices (-1 = none)
	// These are indices into the bindless descriptor array.
	OaI32 AlbedoTextureIdx   = -1;
	OaI32 NormalTextureIdx  = -1;
	OaI32 RoughnessTextureIdx = -1;
	OaI32 MetallicTextureIdx  = -1;
	OaI32 AoTextureIdx        = -1;
	OaI32 EmissiveTextureIdx  = -1;
};

namespace OaFnMaterial {

// ─── Factory ─────────────────────────────────────────────────────────────

// Unlit material for image/video/2D rendering (texture-only)
[[nodiscard]] OaMaterialData CreateUnlit();

// Basic lit material (diffuse + ambient, no textures)
[[nodiscard]] OaMaterialData CreateLit(const VlmVec4& InAlbedo = {1.0f, 1.0f, 1.0f, 1.0f});

// PBR material with default scalars
[[nodiscard]] OaMaterialData CreatePbr(
	OaF32 InRoughness = 0.5f,
	OaF32 InMetallic  = 0.0f
);

// ─── Setters ─────────────────────────────────────────────────────────────

void SetAlbedoColor(OaMaterialData& InMat, const VlmVec4& InColor);
void SetRoughness(OaMaterialData& InMat, OaF32 InRoughness);
void SetMetallic(OaMaterialData& InMat, OaF32 InMetallic);
void SetAo(OaMaterialData& InMat, OaF32 InAo);
void SetEmissiveColor(OaMaterialData& InMat, const VlmVec4& InColor);
void SetEmissiveIntensity(OaMaterialData& InMat, OaF32 InIntensity);

void SetBlendMode(OaMaterialData& InMat, OaBlendMode InMode);
void SetCullMode(OaMaterialData& InMat, OaCullMode InMode);
void SetDoubleSided(OaMaterialData& InMat, bool InDoubleSided);
void SetDepthTest(OaMaterialData& InMat, bool InEnabled);
void SetDepthWrite(OaMaterialData& InMat, bool InEnabled);

// Texture bindless indices (-1 to unbind)
void SetAlbedoTextureIdx(OaMaterialData& InMat, OaI32 InIdx);
void SetNormalTextureIdx(OaMaterialData& InMat, OaI32 InIdx);
void SetRoughnessTextureIdx(OaMaterialData& InMat, OaI32 InIdx);
void SetMetallicTextureIdx(OaMaterialData& InMat, OaI32 InIdx);
void SetAoTextureIdx(OaMaterialData& InMat, OaI32 InIdx);
void SetEmissiveTextureIdx(OaMaterialData& InMat, OaI32 InIdx);

// ─── Queries ─────────────────────────────────────────────────────────────

[[nodiscard]] bool IsTransparent(const OaMaterialData& InMat) noexcept;
[[nodiscard]] bool HasTexture(const OaMaterialData& InMat) noexcept;

} // namespace OaFnMaterial
