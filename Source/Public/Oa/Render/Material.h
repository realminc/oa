// OaMaterial — OOP wrapper for material data.
//
// Wraps OaFnMaterial functional API. All methods delegate to OaFnMaterial.
// For the functional namespace, see <Oa/Render/FnMaterial.h>.
//
// Usage (image/video viewer):
//   OaMaterial mat = OaMaterial::CreateUnlit();
//   mat.SetAlbedoTextureIdx(textureBindlessIdx);
//   mat.SetBlendMode(OaBlendMode::Opaque);
//
// Usage (3D PBR scene):
//   OaMaterial mat = OaMaterial::CreatePbr(0.5f, 0.0f);
//   mat.SetAlbedoColor({1.0f, 0.0f, 0.0f, 1.0f});

#pragma once

#include <Oa/Render/FnMaterial.h>

class OaMaterial {
public:
	OaMaterial() = default;
	~OaMaterial() = default;

	// Factory methods
	[[nodiscard]] static OaMaterial CreateUnlit() {
		OaMaterial m;
		m.Data_ = OaFnMaterial::CreateUnlit();
		return m;
	}

	[[nodiscard]] static OaMaterial CreateLit(const VlmVec4& InAlbedo = {1.0f, 1.0f, 1.0f, 1.0f}) {
		OaMaterial m;
		m.Data_ = OaFnMaterial::CreateLit(InAlbedo);
		return m;
	}

	[[nodiscard]] static OaMaterial CreatePbr(OaF32 InRoughness = 0.5f, OaF32 InMetallic = 0.0f) {
		OaMaterial m;
		m.Data_ = OaFnMaterial::CreatePbr(InRoughness, InMetallic);
		return m;
	}

	// Scalar parameters
	void SetAlbedoColor(const VlmVec4& InColor) { OaFnMaterial::SetAlbedoColor(Data_, InColor); }
	void SetRoughness(OaF32 InRoughness) { OaFnMaterial::SetRoughness(Data_, InRoughness); }
	void SetMetallic(OaF32 InMetallic) { OaFnMaterial::SetMetallic(Data_, InMetallic); }
	void SetAo(OaF32 InAo) { OaFnMaterial::SetAo(Data_, InAo); }
	void SetEmissiveColor(const VlmVec4& InColor) { OaFnMaterial::SetEmissiveColor(Data_, InColor); }
	void SetEmissiveIntensity(OaF32 InIntensity) { OaFnMaterial::SetEmissiveIntensity(Data_, InIntensity); }

	[[nodiscard]] const VlmVec4& GetAlbedoColor() const noexcept { return Data_.AlbedoColor; }
	[[nodiscard]] OaF32 GetRoughness() const noexcept { return Data_.Roughness; }
	[[nodiscard]] OaF32 GetMetallic() const noexcept { return Data_.Metallic; }
	[[nodiscard]] OaF32 GetAo() const noexcept { return Data_.Ao; }
	[[nodiscard]] const VlmVec4& GetEmissiveColor() const noexcept { return Data_.EmissiveColor; }
	[[nodiscard]] OaF32 GetEmissiveIntensity() const noexcept { return Data_.EmissiveIntensity; }

	// Render state
	void SetBlendMode(OaBlendMode InMode) { OaFnMaterial::SetBlendMode(Data_, InMode); }
	void SetCullMode(OaCullMode InMode) { OaFnMaterial::SetCullMode(Data_, InMode); }
	void SetDoubleSided(bool InDoubleSided) { OaFnMaterial::SetDoubleSided(Data_, InDoubleSided); }
	void SetDepthTest(bool InEnabled) { OaFnMaterial::SetDepthTest(Data_, InEnabled); }
	void SetDepthWrite(bool InEnabled) { OaFnMaterial::SetDepthWrite(Data_, InEnabled); }

	[[nodiscard]] OaBlendMode GetBlendMode() const noexcept { return Data_.BlendMode; }
	[[nodiscard]] OaCullMode GetCullMode() const noexcept { return Data_.CullMode; }
	[[nodiscard]] bool IsDoubleSided() const noexcept { return Data_.DoubleSided; }
	[[nodiscard]] bool IsDepthTestEnabled() const noexcept { return Data_.DepthTest; }
	[[nodiscard]] bool IsDepthWriteEnabled() const noexcept { return Data_.DepthWrite; }

	// Texture bindless indices
	void SetAlbedoTextureIdx(OaI32 InIdx) { OaFnMaterial::SetAlbedoTextureIdx(Data_, InIdx); }
	void SetNormalTextureIdx(OaI32 InIdx) { OaFnMaterial::SetNormalTextureIdx(Data_, InIdx); }
	void SetRoughnessTextureIdx(OaI32 InIdx) { OaFnMaterial::SetRoughnessTextureIdx(Data_, InIdx); }
	void SetMetallicTextureIdx(OaI32 InIdx) { OaFnMaterial::SetMetallicTextureIdx(Data_, InIdx); }
	void SetAoTextureIdx(OaI32 InIdx) { OaFnMaterial::SetAoTextureIdx(Data_, InIdx); }
	void SetEmissiveTextureIdx(OaI32 InIdx) { OaFnMaterial::SetEmissiveTextureIdx(Data_, InIdx); }

	[[nodiscard]] OaI32 GetAlbedoTextureIdx() const noexcept { return Data_.AlbedoTextureIdx; }
	[[nodiscard]] OaI32 GetNormalTextureIdx() const noexcept { return Data_.NormalTextureIdx; }
	[[nodiscard]] OaI32 GetRoughnessTextureIdx() const noexcept { return Data_.RoughnessTextureIdx; }
	[[nodiscard]] OaI32 GetMetallicTextureIdx() const noexcept { return Data_.MetallicTextureIdx; }
	[[nodiscard]] OaI32 GetAoTextureIdx() const noexcept { return Data_.AoTextureIdx; }
	[[nodiscard]] OaI32 GetEmissiveTextureIdx() const noexcept { return Data_.EmissiveTextureIdx; }

	// Queries
	[[nodiscard]] bool IsTransparent() const noexcept { return OaFnMaterial::IsTransparent(Data_); }
	[[nodiscard]] bool HasTexture() const noexcept { return OaFnMaterial::HasTexture(Data_); }

	// Direct data access
	[[nodiscard]] OaMaterialData& GetData() noexcept { return Data_; }
	[[nodiscard]] const OaMaterialData& GetData() const noexcept { return Data_; }

private:
	OaMaterialData Data_;
};
