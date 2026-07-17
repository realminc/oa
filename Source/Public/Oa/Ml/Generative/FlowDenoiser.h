#pragma once

#include <Oa/Ml/Generative/Flow.h>
#include <Oa/Ml/Generative/FlowTransformer.h>

/// Modality-independent flow denoiser. Image patches and motion frames use the
/// same token contract; callers choose InputDim, sequence length and optional
/// condition features. NumExperts selects the dense or shared dropless-MoE FFN
/// without changing the training or sampling contract above it. Dense and MoE
/// configurations intentionally retain distinct checkpoint fingerprints.
struct OaFlowDenoiserConfig {
	OaI32 InputDim = 0;
	OaI32 ConditionDim = 0;
	OaFlowTransformerConfig Backbone;
	OaF32 TimeMaxPeriod = 10000.0F;
	OaF32 TimeScale = 1000.0F;
	/// Per-sample conditioning dropout used during training for classifier-free
	/// guidance. Time conditioning is never dropped. Range: [0,1).
	OaF32 ConditionDropoutP = 0.0F;
};

class OaFlowDenoiser final : public OaModule {
public:
	explicit OaFlowDenoiser(const OaFlowDenoiserConfig& InConfig);

	/// Unconditional endpoint convenience (t=0). Training and sampling should
	/// ordinarily call ForwardConditioned with explicit normalized time.
	OaMatrix Forward(const OaMatrix& InSample) override;
	OaMatrix ForwardConditioned(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InCondition = {},
		const OaMatrix& InTokenMask = {});
	/// Classifier-free guided prediction: uncond + scale*(cond-uncond).
	/// Runs under eval semantics so training-time condition dropout is disabled.
	OaMatrix ForwardGuided(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InCondition,
		OaF32 InGuidanceScale,
		const OaMatrix& InTokenMask = {});

	[[nodiscard]] const OaFlowDenoiserConfig& Config() const noexcept {
		return Config_;
	}
	[[nodiscard]] bool IsMoe() const noexcept { return Backbone_->IsMoe(); }
	[[nodiscard]] OaFlowTransformer& Backbone() noexcept { return *Backbone_; }

private:
	OaFlowDenoiserConfig Config_;
	OaSharedPtr<OaLinear> InputProjection_;
	OaSharedPtr<OaFlowTimeEmbedding> TimeEmbedding_;
	OaSharedPtr<OaLinear> ConditionProjection_;
	OaSharedPtr<OaFlowTransformer> Backbone_;
	OaSharedPtr<OaLinear> OutputProjection_;
	OaMatrix Position_;
};
