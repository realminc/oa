// OaFnMaterial implementation

#include <Oa/Render/FnMaterial.h>

namespace OaFnMaterial {

// ─── Factory ─────────────────────────────────────────────────────────────

OaMaterialData CreateUnlit() {
	OaMaterialData mat;
	mat.Type = OaMaterialType::Unlit;
	mat.BlendMode = OaBlendMode::Opaque;
	mat.CullMode = OaCullMode::None;
	mat.DepthTest = false;
	mat.DepthWrite = false;
	return mat;
}

OaMaterialData CreateLit(const VlmVec4& InAlbedo) {
	OaMaterialData mat;
	mat.Type = OaMaterialType::Lit;
	mat.AlbedoColor = InAlbedo;
	mat.BlendMode = OaBlendMode::Opaque;
	mat.CullMode = OaCullMode::Back;
	mat.DepthTest = true;
	mat.DepthWrite = true;
	return mat;
}

OaMaterialData CreatePbr(OaF32 InRoughness, OaF32 InMetallic) {
	OaMaterialData mat;
	mat.Type = OaMaterialType::Pbr;
	mat.Roughness = InRoughness;
	mat.Metallic = InMetallic;
	mat.BlendMode = OaBlendMode::Opaque;
	mat.CullMode = OaCullMode::Back;
	mat.DepthTest = true;
	mat.DepthWrite = true;
	return mat;
}

// ─── Setters ─────────────────────────────────────────────────────────────

void SetAlbedoColor(OaMaterialData& InMat, const VlmVec4& InColor) {
	InMat.AlbedoColor = InColor;
}

void SetRoughness(OaMaterialData& InMat, OaF32 InRoughness) {
	InMat.Roughness = InRoughness;
}

void SetMetallic(OaMaterialData& InMat, OaF32 InMetallic) {
	InMat.Metallic = InMetallic;
}

void SetAo(OaMaterialData& InMat, OaF32 InAo) {
	InMat.Ao = InAo;
}

void SetEmissiveColor(OaMaterialData& InMat, const VlmVec4& InColor) {
	InMat.EmissiveColor = InColor;
}

void SetEmissiveIntensity(OaMaterialData& InMat, OaF32 InIntensity) {
	InMat.EmissiveIntensity = InIntensity;
}

void SetBlendMode(OaMaterialData& InMat, OaBlendMode InMode) {
	InMat.BlendMode = InMode;
}

void SetCullMode(OaMaterialData& InMat, OaCullMode InMode) {
	InMat.CullMode = InMode;
}

void SetDoubleSided(OaMaterialData& InMat, bool InDoubleSided) {
	InMat.DoubleSided = InDoubleSided;
	if (InDoubleSided) {
		InMat.CullMode = OaCullMode::None;
	}
}

void SetDepthTest(OaMaterialData& InMat, bool InEnabled) {
	InMat.DepthTest = InEnabled;
}

void SetDepthWrite(OaMaterialData& InMat, bool InEnabled) {
	InMat.DepthWrite = InEnabled;
}

void SetAlbedoTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.AlbedoTextureIdx = InIdx;
}

void SetNormalTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.NormalTextureIdx = InIdx;
}

void SetRoughnessTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.RoughnessTextureIdx = InIdx;
}

void SetMetallicTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.MetallicTextureIdx = InIdx;
}

void SetAoTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.AoTextureIdx = InIdx;
}

void SetEmissiveTextureIdx(OaMaterialData& InMat, OaI32 InIdx) {
	InMat.EmissiveTextureIdx = InIdx;
}

// ─── Queries ─────────────────────────────────────────────────────────────

bool IsTransparent(const OaMaterialData& InMat) noexcept {
	return InMat.BlendMode != OaBlendMode::Opaque || InMat.AlbedoColor.W < 1.0f;
}

bool HasTexture(const OaMaterialData& InMat) noexcept {
	return InMat.AlbedoTextureIdx >= 0 ||
	       InMat.NormalTextureIdx >= 0 ||
	       InMat.RoughnessTextureIdx >= 0 ||
	       InMat.MetallicTextureIdx >= 0 ||
	       InMat.AoTextureIdx >= 0 ||
	       InMat.EmissiveTextureIdx >= 0;
}

} // namespace OaFnMaterial
