#pragma once

#include "FlowTimeEmbedding.h"
#include "FlowTransformer.h"
#include "../Linear/Linear.gen.h"

/// Modality-independent flow denoiser. Image patches and motion frames use the
/// same token contract; callers choose InputDim, sequence length and optional
/// condition features. NumExperts selects dense or shared dropless-MoE FFNs.
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

	OaMatrix Forward(const OaMatrix& InSample) override;
	OaMatrix ForwardConditioned(
		const OaMatrix& InSample,
		const OaMatrix& InTime,
		const OaMatrix& InCondition = {},
		const OaMatrix& InTokenMask = {});
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
