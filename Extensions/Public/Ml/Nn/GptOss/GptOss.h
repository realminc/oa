#pragma once

// Exact OpenAI gpt-oss architecture contract. This is intentionally not a
// generic configurable GPT class: published presets and scaled parity fixtures
// share the same operations, while checkpoint import validates every field.

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

struct OaGptOssConfig {
	OaI32 VocabSize = 201088;
	OaI32 NumLayers = 24;
	OaI32 HiddenSize = 2880;
	OaI32 IntermediateSize = 2880;
	OaI32 NumAttentionHeads = 64;
	OaI32 NumKvHeads = 8;
	OaI32 HeadDim = 64;
	OaI32 NumExperts = 32;
	OaI32 ExpertsPerToken = 4;
	OaI32 SlidingWindow = 128;
	OaI32 OriginalContextLength = 4096;
	OaI32 MaxPositionEmbeddings = 131072;
	OaF32 RopeTheta = 150000.0F;
	OaF32 RopeScalingFactor = 32.0F;
	OaF32 RopeNtkAlpha = 1.0F;   // Hugging Face beta_slow
	OaF32 RopeNtkBeta = 32.0F;   // Hugging Face beta_fast
	OaF32 RmsNormEps = 1e-5F;
	OaF32 SwiGluAlpha = 1.702F;
	OaF32 SwiGluLimit = 7.0F;
	OaI32 PadToken = 199999;
	OaI32 EosToken = 200002;
	bool AttentionBias = true;
	bool TieWordEmbeddings = false;
	bool MxFp4Experts = true;

	[[nodiscard]] static OaGptOssConfig Preset20B();
	[[nodiscard]] static OaGptOssConfig Preset120B();
	[[nodiscard]] static OaResult<OaGptOssConfig> FromJson(const OaString& InPath);

	// Topology validation permits scaled parity fixtures. FromJson additionally
	// enforces the source model_type, alternating layer_types, and MXFP4 method.
	[[nodiscard]] OaStatus Validate() const;
	[[nodiscard]] bool LayerUsesSlidingAttention(OaI32 InLayer) const {
		return (InLayer & 1) == 0;
	}
	[[nodiscard]] OaI32 QueryWidth() const { return NumAttentionHeads * HeadDim; }
	[[nodiscard]] OaI32 KvWidth() const { return NumKvHeads * HeadDim; }
	[[nodiscard]] OaI32 ExpectedLogicalWeightCount() const { return 3 + NumLayers * 19; }
	[[nodiscard]] bool IsPublished20B() const;
	[[nodiscard]] bool IsPublished120B() const;
};
