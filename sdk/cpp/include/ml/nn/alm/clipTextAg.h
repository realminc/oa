#pragma once

// ClipTextAg — exact frozen CLIP text-with-projection tower used by oa::Alm.
//
// Heavy tensor work is OA/vulkan. Tokenization is a deterministic CPU boundary;
// forwardTokens accepts already-tokenized fixed-context IDs plus the flat EOS row
// selected by that tokenizer, avoiding a GPU->cPU argmax/readback in production.

#include <oa/ml/module.h>
#include <oa/ml/nn.h>

#include <type_traits>

namespace oa { class Engine; }

namespace oa {

struct ClipTextConfig {
	oa::I32 vocabSize = 49408;
	oa::I32 contextLength = 77;
	oa::I32 hiddenSize = 768;
	oa::I32 intermediateSize = 3072;
	oa::I32 numHeads = 12;
	oa::I32 numLayers = 12;
	oa::I32 projectionDim = 768;
	oa::F32 layerNormEps = 1e-5F;
	oa::F32 quickGeluAlpha = 1.702F;
	oa::I32 bosToken = 49406;
	oa::I32 eosToken = 49407;
	oa::I32 padToken = 49407;

	[[nodiscard]] oa::Status validate() const;
	[[nodiscard]] static ClipTextConfig viTL14();
};
static_assert(std::is_trivially_copyable_v<ClipTextConfig>);
static_assert(sizeof(ClipTextConfig) == 48,
	"ClipTextConfig is serialized in the v1 architecture payload");

class ClipTextAg : public oa::Module {
public:
	explicit ClipTextAg(const ClipTextConfig& inConfig = ClipTextConfig::viTL14());

	// Compatibility path: derives EOS rows with a synchronized argmax. Production
	// prompt inference should call forwardTokens with tokenizer-provided EOS rows.
	oa::Matrix forward(const oa::Matrix& inTokenIds) override;

	// inTokenIds: [B, contextLength] UInt32/Int32.
	// inFlatEosRows: [B] UInt32, each value b*contextLength+eos_position.
	[[nodiscard]] oa::Matrix forwardTokens(
		const oa::Matrix& inTokenIds, const oa::Matrix& inFlatEosRows);
	[[nodiscard]] static oa::Result<oa::SharedPtr<ClipTextAg>> loadArchive(
		oa::Engine& inEngine, const oa::String& inPath);

	void freeze();
	[[nodiscard]] const ClipTextConfig& config() const noexcept { return config_; }

private:
	ClipTextConfig config_;
	oa::SharedPtr<oa::Embedding> tokenEmbedding_;
	oa::SharedPtr<oa::Embedding> positionEmbedding_;
	oa::Vec<oa::SharedPtr<oa::Module>> layers_;
	oa::SharedPtr<oa::LayerNorm> finalLayerNorm_;
	oa::SharedPtr<oa::Linear> textProjection_;
	oa::Matrix positionIds_;
};

} // namespace oa
