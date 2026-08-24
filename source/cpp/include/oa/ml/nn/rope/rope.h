#pragma once

#include <oa/ml/module.h>

namespace oa {

/// Rotary position embedding over a flattened [tokens, heads * head_dim] input.
class Rope : public oa::Module {
public:
	Rope(
		oa::I32 inNumHeads,
		oa::I32 inHeadDim,
		oa::F32 inThetaBase = 10000.0F);

	oa::Matrix forward(const oa::Matrix& inInput) override;

private:
	oa::I32 numHeads_;
	oa::I32 headDim_;
	oa::F32 thetaBase_;
};

} // namespace oa
