#pragma once

#include <oa/ml/module.h>

#include <oa/ml/nn/transformer/transformer.h>

namespace oa {

/// Configuration for a bidirectional denoising Transformer. numExperts == 0
/// selects an ordinary dense FFN; a positive value selects the shared dropless
/// MoE implementation without changing the model's input/output contract.
struct FlowTransformerConfig {
	oa::I32 dModel = 0;
	oa::I32 hiddenDim = 0;
	oa::I32 sequenceLength = 0;
	oa::I32 numLayers = 1;
	oa::I32 numHeads = 1;
	oa::I32 numExperts = 0;
	oa::I32 expertsPerToken = 0;
	oa::F32 epsilon = 1.0e-5F;
	bool adaptiveConditioning = true;
};

/// Reusable bidirectional Transformer backbone for flow/diffusion denoisers.
/// input is conditioned token state [B,S,D] or flattened [B*S,D]; output keeps
/// the same shape. This is a sibling model family that composes generic
/// Transformer blocks; it is not part of the generic Transformer itself.
class FlowTransformer final : public oa::Module {
public:
	explicit FlowTransformer(const FlowTransformerConfig& inConfig);

	oa::Matrix forward(const oa::Matrix& inTokens) override;
	oa::Matrix forwardMasked(
		const oa::Matrix& inTokens,
		const oa::Matrix& inTokenMask);
	oa::Matrix forwardConditioned(
		const oa::Matrix& inTokens,
		const oa::Matrix& inCondition,
		const oa::Matrix& inTokenMask = {});
	void setSequenceLength(oa::I32 inSequenceLength);

	[[nodiscard]] const FlowTransformerConfig& config() const noexcept {
		return config_;
	}
	[[nodiscard]] bool isMoe() const noexcept { return config_.numExperts > 0; }
	[[nodiscard]] oa::I32 numLayers() const noexcept { return config_.numLayers; }
	[[nodiscard]] oa::TransformerBlock& block(oa::I32 inIndex);
	[[nodiscard]] const oa::TransformerBlock& block(oa::I32 inIndex) const;

private:
	FlowTransformerConfig config_;
	oa::Vector<oa::SharedPtr<oa::TransformerBlock>> blocks_;
	oa::SharedPtr<oa::LayerNorm> outputNorm_;

	oa::Matrix forwardImpl(
		const oa::Matrix& inTokens,
		const oa::Matrix* inTokenMask,
		const oa::Matrix* inCondition);
};

} // namespace oa
