// Ffn — SwiGLU FFN Module (level 1 API)
//
// architecture: RMSNorm → gate(D→DFF) → Up(D→DFF) → SwiGLU → down(DFF→D) → +residual
//
// level 1 API: Inherits from oa::Module, uses oa::FnMatrix operations
// autograd-compatible: forward automatically tracked, backward via oa::GradFn::backward

#pragma once

#include <oa/ml/module.h>
#include <oa/ml/nn.h>

namespace oa {

class Ffn : public oa::Module {
public:
	// Constructors
	Ffn() = default;
	Ffn(oa::I32 inDModel, oa::I32 inDFF, oa::F32 inRmsEps = 1e-5f);

	// Destructor
	~Ffn() override = default;

	// Initialization
	void init(oa::I32 inDModel, oa::I32 inDFF, oa::F32 inRmsEps = 1e-5f);

	// forward pass (autograd-tracked)
	oa::Matrix forward(const oa::Matrix& inX) override;

	// Accessors
	[[nodiscard]] oa::I32 dModel() const { return dModel_; }
	[[nodiscard]] oa::I32 dFF() const { return dFF_; }
	[[nodiscard]] oa::F32 rmsEps() const { return rmsEps_; }

private:
	// architecture components
	oa::SharedPtr<oa::RmsNorm> norm_;
	oa::SharedPtr<oa::Linear>  gate_;  // D → DFF
	oa::SharedPtr<oa::Linear>  up_;    // D → DFF
	oa::SharedPtr<oa::Linear>  down_;  // DFF → D

	// Configuration
	oa::I32 dModel_ = 0;
	oa::I32 dFF_ = 0;
	oa::F32 rmsEps_ = 1e-5f;
};

} // namespace oa
