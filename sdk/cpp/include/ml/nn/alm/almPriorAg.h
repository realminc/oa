#pragma once

// AlmPriorAg — stage 2 autoregressive language model for motion generation.
//
// A decoder-only LM that generates motion token sequences. Trained on tokenized
// motion data from stage 1 (VQ-VAE frozen).
//
// backbone: causal Transformer. The attention path is permanent; each block's
// FFN can be dense, MoE, or selected by a hybrid cadence without changing the
// oa::Alm training/generation pipeline.
//
// architecture:
//   - Optional projected frozen-text prefix + motion-token embedding + learned positions
//   - N decoder layers: pre-norm causal attention → dense or MoE FFN
//   - output head (logits over vocab)
//
// training:
//   - Next-token prediction; CrossEntropy on token predictions
//   - sequence format: [SOM] code0 code1 ... codeN [EOM] [PAD...]

#include <ml/nn/alm/almConfig.h>
#include <oa/ml.h>
#include <oa/ml/nn.h>

namespace oa {

// motion language model
class AlmPriorAg : public oa::Module {
public:
	explicit AlmPriorAg(const AlmPriorConfig& inConfig);

	// forward pass: token ids [B, T] → logits [B, T, vocabSize]
	oa::Matrix forward(const oa::Matrix& inTokenIds) override;
	// Frozen caption features [B,textFeatureDim] become one prefix token. The
	// returned logits still correspond only to the T motion-token positions.
	oa::Matrix forwardConditioned(
		const oa::Matrix& inTokenIds, const oa::Matrix& inTextFeatures);

	// generate token sequence autoregressively (feeds the growing prefix, samples
	// the last position). inUseCache is reserved for the Transformer KV-cache path.
	// Returns: [B, MaxLen] token ids (padded with padToken).
	oa::Matrix generate(
		oa::I32 inBatchSize,
		oa::F32 inTemperature = 1.0F,
		oa::I32 inTopK = 0,
		oa::F32 inTopP = 0.9F,
		oa::I32 inMaxLen = 256,
		bool inUseCache = true
	);
	oa::Matrix generateConditioned(
		const oa::Matrix& inTextFeatures,
		oa::F32 inTemperature = 1.0F,
		oa::I32 inTopK = 0,
		oa::F32 inTopP = 0.9F,
		oa::I32 inMaxLen = 256,
		bool inUseCache = true
	);

	// Decode tokens to motion using tokenizer
	oa::Matrix decodeToMotion(
		const oa::Matrix& inTokenIds,
		class AlmTokenizerAg& inTokenizer
	);

	[[nodiscard]] const AlmPriorConfig& config() const noexcept { return config_; }

	// MoE training contract. auxLoss() is valid after forward and must be added
	// to the task loss before backward. updateMoeRoutingBias() runs after the
	// synchronized optimizer step. Dense priors return an empty auxLoss and no-op.
	[[nodiscard]] oa::Matrix moeAuxLoss() const;
	void updateMoeRoutingBias();
	[[nodiscard]] oa::Vector<oa::MoeRouteStats> moeRouteStats() const;

private:
	oa::Matrix forwardImpl(const oa::Matrix& inTokenIds, const oa::Matrix* inTextFeatures);
	oa::Matrix generateImpl(const oa::Matrix* inTextFeatures, oa::I32 inBatchSize,
		oa::F32 inTemperature, oa::I32 inTopK, oa::F32 inTopP, oa::I32 inMaxLen,
		bool inUseCache);

	AlmPriorConfig config_;
	oa::I32 maxSeqLen_ = 512;

	// architecture
	oa::SharedPtr<oa::Embedding> tokenEmbed_;
	oa::SharedPtr<oa::Embedding> posEmbed_;
	oa::SharedPtr<oa::Linear> textProjection_;

	// Cached [B,T] position-index buffer — positions are [0..t-1] per row, so
	// rebuild + H2D only when (B,T) changes instead of every forward.
	oa::Matrix posIdxCache_;
	oa::I32    cachedPosB_ = -1;
	oa::I32    cachedPosT_ = -1;

	oa::Vector<oa::SharedPtr<oa::TransformerBlock>> layers_;
	oa::SharedPtr<oa::RmsNorm> finalNorm_;
	oa::SharedPtr<oa::Linear> outputHead_;

};

} // namespace oa
