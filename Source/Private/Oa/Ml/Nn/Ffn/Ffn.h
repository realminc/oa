// OaFfn — SwiGLU FFN Module (Level 1 API)
//
// Architecture: RMSNorm → Gate(D→DFF) → Up(D→DFF) → SwiGLU → Down(DFF→D) → +residual
//
// Level 1 API: Inherits from OaModule, uses OaFnMatrix operations
// Autograd-compatible: Forward automatically tracked, backward via OaGradFn::Backward

#pragma once

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

class OaFfn : public OaModule {
public:
	// Constructors
	OaFfn() = default;
	OaFfn(OaI32 InDModel, OaI32 InDFF, OaF32 InRmsEps = 1e-5f);

	// Destructor
	~OaFfn() override = default;

	// Initialization
	void Init(OaI32 InDModel, OaI32 InDFF, OaF32 InRmsEps = 1e-5f);

	// Forward pass (autograd-tracked)
	OaMatrix Forward(const OaMatrix& InX) override;

	// Accessors
	[[nodiscard]] OaI32 DModel() const { return DModel_; }
	[[nodiscard]] OaI32 DFF() const { return DFF_; }
	[[nodiscard]] OaF32 RmsEps() const { return RmsEps_; }

private:
	// Architecture components
	OaSharedPtr<OaRmsNorm> Norm_;
	OaSharedPtr<OaLinear> Gate_;  // D → DFF
	OaSharedPtr<OaLinear> Up_;    // D → DFF
	OaSharedPtr<OaLinear> Down_;  // DFF → D

	// Configuration
	OaI32 DModel_ = 0;
	OaI32 DFF_ = 0;
	OaF32 RmsEps_ = 1e-5f;
};
