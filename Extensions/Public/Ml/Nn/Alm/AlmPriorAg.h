#pragma once

// OaAlmPriorAg — Stage 2 autoregressive language model for motion generation.
//
// A decoder-only LM that generates motion token sequences. Trained on tokenized
// motion data from Stage 1 (VQ-VAE frozen).
//
// Backbone: causal Transformer. The attention path is permanent; each block's
// FFN can be dense, MoE, or selected by a hybrid cadence without changing the
// OaAlm training/generation pipeline.
//
// Architecture:
//   - Optional projected frozen-text prefix + motion-token embedding + learned positions
//   - N decoder layers: pre-norm causal attention → dense or MoE FFN
//   - Output head (logits over vocab)
//
// Training:
//   - Next-token prediction; CrossEntropy on token predictions
//   - Sequence format: [SOM] code0 code1 ... codeN [EOM] [PAD...]

#include <Ml/Nn/Alm/AlmConfig.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Nn.h>

// Motion language model
class OaAlmPriorAg : public OaModule {
public:
	explicit OaAlmPriorAg(const OaAlmPriorConfig& InConfig);

	// Forward pass: token ids [B, T] → logits [B, T, VocabSize]
	OaMatrix Forward(const OaMatrix& InTokenIds) override;
	// Frozen caption features [B,TextFeatureDim] become one prefix token. The
	// returned logits still correspond only to the T motion-token positions.
	OaMatrix ForwardConditioned(
		const OaMatrix& InTokenIds, const OaMatrix& InTextFeatures);

	// Generate token sequence autoregressively (feeds the growing prefix, samples
	// the last position). InUseCache is reserved for the Transformer KV-cache path.
	// Returns: [B, MaxLen] token ids (padded with PadToken).
	OaMatrix Generate(
		OaI32 InBatchSize,
		OaF32 InTemperature = 1.0F,
		OaI32 InTopK = 0,
		OaF32 InTopP = 0.9F,
		OaI32 InMaxLen = 256,
		bool InUseCache = true
	);
	OaMatrix GenerateConditioned(
		const OaMatrix& InTextFeatures,
		OaF32 InTemperature = 1.0F,
		OaI32 InTopK = 0,
		OaF32 InTopP = 0.9F,
		OaI32 InMaxLen = 256,
		bool InUseCache = true
	);

	// Decode tokens to motion using tokenizer
	OaMatrix DecodeToMotion(
		const OaMatrix& InTokenIds,
		class OaAlmTokenizerAg& InTokenizer
	);

	const OaAlmPriorConfig& GetConfig() const { return Config_; }

	// MoE training contract. AuxLoss() is valid after Forward and must be added
	// to the task loss before Backward. UpdateMoeRoutingBias() runs after the
	// synchronized optimizer step. Dense priors return an empty AuxLoss and no-op.
	[[nodiscard]] OaMatrix MoeAuxLoss() const;
	void UpdateMoeRoutingBias();
	[[nodiscard]] OaVec<OaMoeRouteStats> MoeRouteStats() const;

private:
	OaMatrix ForwardImpl(const OaMatrix& InTokenIds, const OaMatrix* InTextFeatures);
	OaMatrix GenerateImpl(const OaMatrix* InTextFeatures, OaI32 InBatchSize,
		OaF32 InTemperature, OaI32 InTopK, OaF32 InTopP, OaI32 InMaxLen,
		bool InUseCache);

	OaAlmPriorConfig Config_;
	OaI32 MaxSeqLen_ = 512;

	// Architecture
	OaSharedPtr<OaEmbedding> TokenEmbed_;
	OaSharedPtr<OaEmbedding> PosEmbed_;
	OaSharedPtr<OaLinear> TextProjection_;

	// Cached [B,T] position-index buffer — positions are [0..T-1] per row, so
	// rebuild + H2D only when (B,T) changes instead of every Forward.
	OaMatrix PosIdxCache_;
	OaI32    CachedPosB_ = -1;
	OaI32    CachedPosT_ = -1;

	OaVec<OaSharedPtr<OaTransformerBlock>> Layers_;
	OaSharedPtr<OaRmsNorm> FinalNorm_;
	OaSharedPtr<OaLinear> OutputHead_;

};
