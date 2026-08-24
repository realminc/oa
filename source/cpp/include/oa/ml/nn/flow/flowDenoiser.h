#pragma once

#include <oa/ml/nn/flow/flowTimeEmbedding.h>
#include <oa/ml/nn/flow/flowTransformer.h>
#include <oa/ml/nn/linear/linear.h>

namespace oa {

/// Modality-independent flow denoiser. Image patches and motion frames use the
/// same token contract; callers choose inputDim, sequence length and optional
/// condition features. numExperts selects dense or shared dropless-MoE FFNs.
struct FlowDenoiserConfig {
	oa::I32 inputDim = 0;
	oa::I32 conditionDim = 0;
	FlowTransformerConfig backbone;
	oa::F32 timeMaxPeriod = 10000.0F;
	oa::F32 timeScale = 1000.0F;
	/// Per-sample conditioning dropout used during training for classifier-free
	/// guidance. time conditioning is never dropped. range: [0,1).
	oa::F32 conditionDropoutP = 0.0F;
};

class FlowDenoiser final : public oa::Module {
public:
	explicit FlowDenoiser(const FlowDenoiserConfig& inConfig);

	oa::Matrix forward(const oa::Matrix& inSample) override;
	oa::Matrix forwardConditioned(
		const oa::Matrix& inSample,
		const oa::Matrix& inTime,
		const oa::Matrix& inCondition = {},
		const oa::Matrix& inTokenMask = {});
	oa::Matrix forwardGuided(
		const oa::Matrix& inSample,
		const oa::Matrix& inTime,
		const oa::Matrix& inCondition,
		oa::F32 inGuidanceScale,
		const oa::Matrix& inTokenMask = {});

	[[nodiscard]] const FlowDenoiserConfig& config() const noexcept {
		return config_;
	}
	[[nodiscard]] bool isMoe() const noexcept { return backbone_->isMoe(); }
	[[nodiscard]] FlowTransformer& backbone() noexcept { return *backbone_; }

private:
	FlowDenoiserConfig config_;
	oa::SharedPtr<oa::Linear> inputProjection_;
	oa::SharedPtr<FlowTimeEmbedding> timeEmbedding_;
	oa::SharedPtr<oa::Linear> conditionProjection_;
	oa::SharedPtr<FlowTransformer> backbone_;
	oa::SharedPtr<oa::Linear> outputProjection_;
	oa::Matrix position_;
};

} // namespace oa
