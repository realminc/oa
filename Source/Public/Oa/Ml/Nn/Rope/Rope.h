#pragma once

#include <Oa/Ml/Module.h>

/// Rotary position embedding over a flattened [tokens, heads * head_dim] input.
class OaRoPE : public OaModule {
public:
	OaRoPE(
		OaI32 InNumHeads,
		OaI32 InHeadDim,
		OaF32 InThetaBase = 10000.0F);

	OaMatrix Forward(const OaMatrix& InInput) override;

private:
	OaI32 NumHeads_;
	OaI32 HeadDim_;
	OaF32 ThetaBase_;
};
