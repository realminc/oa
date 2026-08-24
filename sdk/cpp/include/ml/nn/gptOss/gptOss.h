#pragma once

// exact OpenAI gpt-oss architecture contract. This is intentionally not a
// generic configurable GPT class: published presets and scaled parity fixtures
// share the same operations, while checkpoint import validates every field.

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

struct GptOssConfig {
	oa::I32 vocabSize = 201088;
	oa::I32 numLayers = 24;
	oa::I32 hiddenSize = 2880;
	oa::I32 intermediateSize = 2880;
	oa::I32 numAttentionHeads = 64;
	oa::I32 numKvHeads = 8;
	oa::I32 headDim = 64;
	oa::I32 numExperts = 32;
	oa::I32 expertsPerToken = 4;
	oa::I32 slidingWindow = 128;
	oa::I32 originalContextLength = 4096;
	oa::I32 maxPositionEmbeddings = 131072;
	oa::F32 ropeTheta = 150000.0F;
	oa::F32 ropeScalingFactor = 32.0F;
	oa::F32 ropeNtkAlpha = 1.0F;   // Hugging face beta_slow
	oa::F32 ropeNtkBeta = 32.0F;   // Hugging face beta_fast
	oa::F32 rmsNormEps = 1e-5F;
	oa::F32 swiGluAlpha = 1.702F;
	oa::F32 swiGluLimit = 7.0F;
	oa::I32 padToken = 199999;
	oa::I32 eosToken = 200002;
	bool attentionBias = true;
	bool tieWordEmbeddings = false;
	bool mxFp4Experts = true;

	[[nodiscard]] static GptOssConfig preset20B();
	[[nodiscard]] static GptOssConfig preset120B();
	[[nodiscard]] static oa::Result<GptOssConfig> fromJson(const oa::String& inPath);

	// topology validation permits scaled parity fixtures. FromJson additionally
	// enforces the source model_type, alternating layer_types, and MXFP4 method.
	[[nodiscard]] oa::Status validate() const;
	[[nodiscard]] bool layerUsesSlidingAttention(oa::I32 inLayer) const {
		return (inLayer & 1) == 0;
	}
	[[nodiscard]] oa::I32 queryWidth() const { return numAttentionHeads * headDim; }
	[[nodiscard]] oa::I32 kvWidth() const { return numKvHeads * headDim; }
	[[nodiscard]] oa::I32 expectedLogicalWeightCount() const { return 3 + numLayers * 19; }
	[[nodiscard]] bool isPublished20B() const;
	[[nodiscard]] bool isPublished120B() const;
};

} // namespace oa
