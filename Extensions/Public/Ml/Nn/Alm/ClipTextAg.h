#pragma once

// OaClipTextAg — exact frozen CLIP text-with-projection tower used by OaAlm.
//
// Heavy tensor work is OA/Vulkan. Tokenization is a deterministic CPU boundary;
// ForwardTokens accepts already-tokenized fixed-context IDs plus the flat EOS row
// selected by that tokenizer, avoiding a GPU->CPU argmax/readback in production.

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

#include <type_traits>

struct OaClipTextConfig {
	OaI32 VocabSize = 49408;
	OaI32 ContextLength = 77;
	OaI32 HiddenSize = 768;
	OaI32 IntermediateSize = 3072;
	OaI32 NumHeads = 12;
	OaI32 NumLayers = 12;
	OaI32 ProjectionDim = 768;
	OaF32 LayerNormEps = 1e-5F;
	OaF32 QuickGeluAlpha = 1.702F;
	OaI32 BosToken = 49406;
	OaI32 EosToken = 49407;
	OaI32 PadToken = 49407;

	[[nodiscard]] OaStatus Validate() const;
	[[nodiscard]] static OaClipTextConfig ViTL14();
};
static_assert(std::is_trivially_copyable_v<OaClipTextConfig>);
static_assert(sizeof(OaClipTextConfig) == 48,
	"OaClipTextConfig is serialized in the v1 architecture payload");

class OaClipTextAg : public OaModule {
public:
	explicit OaClipTextAg(const OaClipTextConfig& InConfig = OaClipTextConfig::ViTL14());

	// Compatibility path: derives EOS rows with a synchronized argmax. Production
	// prompt inference should call ForwardTokens with tokenizer-provided EOS rows.
	OaMatrix Forward(const OaMatrix& InTokenIds) override;

	// InTokenIds: [B, ContextLength] UInt32/Int32.
	// InFlatEosRows: [B] UInt32, each value b*ContextLength+eos_position.
	[[nodiscard]] OaMatrix ForwardTokens(
		const OaMatrix& InTokenIds, const OaMatrix& InFlatEosRows);
	[[nodiscard]] static OaResult<OaSharedPtr<OaClipTextAg>> LoadOam(const OaString& InPath);

	void Freeze();
	[[nodiscard]] const OaClipTextConfig& Config() const noexcept { return Config_; }

private:
	OaClipTextConfig Config_;
	OaSharedPtr<OaEmbedding> TokenEmbedding_;
	OaSharedPtr<OaEmbedding> PositionEmbedding_;
	OaVec<OaSharedPtr<OaModule>> Layers_;
	OaSharedPtr<OaLayerNorm> FinalLayerNorm_;
	OaSharedPtr<OaLinear> TextProjection_;
	OaMatrix PositionIds_;
};
