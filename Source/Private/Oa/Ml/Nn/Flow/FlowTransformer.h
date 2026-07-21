#pragma once

#include <Oa/Ml/Module.h>

#include "../Transformer/Transformer.h"

/// Configuration for a bidirectional denoising Transformer. NumExperts == 0
/// selects an ordinary dense FFN; a positive value selects the shared dropless
/// MoE implementation without changing the model's input/output contract.
struct OaFlowTransformerConfig {
	OaI32 DModel = 0;
	OaI32 HiddenDim = 0;
	OaI32 SequenceLength = 0;
	OaI32 NumLayers = 1;
	OaI32 NumHeads = 1;
	OaI32 NumExperts = 0;
	OaI32 ExpertsPerToken = 0;
	OaF32 Epsilon = 1.0e-5F;
	bool AdaptiveConditioning = true;
};

/// Reusable bidirectional Transformer backbone for flow/diffusion denoisers.
/// Input is conditioned token state [B,S,D] or flattened [B*S,D]; output keeps
/// the same shape. This is a sibling model family that composes generic
/// Transformer blocks; it is not part of the generic Transformer itself.
class OaFlowTransformer final : public OaModule {
public:
	explicit OaFlowTransformer(const OaFlowTransformerConfig& InConfig);

	OaMatrix Forward(const OaMatrix& InTokens) override;
	OaMatrix ForwardMasked(
		const OaMatrix& InTokens,
		const OaMatrix& InTokenMask);
	OaMatrix ForwardConditioned(
		const OaMatrix& InTokens,
		const OaMatrix& InCondition,
		const OaMatrix& InTokenMask = {});
	void SetSequenceLength(OaI32 InSequenceLength);

	[[nodiscard]] const OaFlowTransformerConfig& Config() const noexcept {
		return Config_;
	}
	[[nodiscard]] bool IsMoe() const noexcept { return Config_.NumExperts > 0; }
	[[nodiscard]] OaI32 NumLayers() const noexcept { return Config_.NumLayers; }
	[[nodiscard]] OaTransformerBlock& Block(OaI32 InIndex);
	[[nodiscard]] const OaTransformerBlock& Block(OaI32 InIndex) const;

private:
	OaFlowTransformerConfig Config_;
	OaVec<OaSharedPtr<OaTransformerBlock>> Blocks_;
	OaSharedPtr<OaLayerNorm> OutputNorm_;

	OaMatrix ForwardImpl(
		const OaMatrix& InTokens,
		const OaMatrix* InTokenMask,
		const OaMatrix* InCondition);
};
