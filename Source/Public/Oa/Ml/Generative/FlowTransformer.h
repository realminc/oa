#pragma once

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

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
/// the same shape. Callers project modality data and add time/text/class
/// conditioning before Forward, keeping this backbone modality-independent.
class OaFlowTransformer final : public OaModule {
public:
	explicit OaFlowTransformer(const OaFlowTransformerConfig& InConfig);

	OaMatrix Forward(const OaMatrix& InTokens) override;
	/// Bidirectional forward with compact binary token validity [B,S] or
	/// [B,S,1]. Invalid keys are excluded in every layer; callers should also
	/// mask invalid query rows in their task loss.
	OaMatrix ForwardMasked(
		const OaMatrix& InTokens, const OaMatrix& InTokenMask);
	/// DiT-style AdaLN-Zero forward. InCondition is [B,DModel] and modulates
	/// both attention and FFN residuals in every block.
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
