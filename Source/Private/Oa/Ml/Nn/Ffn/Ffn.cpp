// OaFfn — SwiGLU FFN Implementation (Level 1 API)

#include "Ffn.h"
#include <Oa/Core/FnMatrix.h>

OaFfn::OaFfn(OaI32 InDModel, OaI32 InDFF, OaF32 InRmsEps) {
	Init(InDModel, InDFF, InRmsEps);
}

void OaFfn::Init(OaI32 InDModel, OaI32 InDFF, OaF32 InRmsEps) {
	DModel_ = InDModel;
	DFF_ = InDFF;
	RmsEps_ = InRmsEps;
	
	auto wd = OaFnMatrix::GetWeightDtype();
	
	// Create layers
	Norm_ = OaMakeSharedPtr<OaRmsNorm>(InDModel, InRmsEps);
	Gate_ = OaMakeSharedPtr<OaLinear>(InDModel, InDFF);
	Up_ = OaMakeSharedPtr<OaLinear>(InDModel, InDFF);
	Down_ = OaMakeSharedPtr<OaLinear>(InDFF, InDModel);
	
	// Initialize weights with autograd tracking
	// Grad is the single source of truth on each param's Data (OaParameter::Grad());
	// SetRequiresGrad allocates it. No manual snapshot to re-sync — reassigning Data
	// (e.g. RandXavier below) is automatically reflected by Grad().
	auto& gateParams = Gate_->Parameters();
	gateParams[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{InDFF, InDModel}, wd);
	gateParams[0].Data.SetRequiresGrad(true);
	gateParams[1].Data.SetRequiresGrad(true);

	auto& upParams = Up_->Parameters();
	upParams[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{InDFF, InDModel}, wd);
	upParams[0].Data.SetRequiresGrad(true);
	upParams[1].Data.SetRequiresGrad(true);

	auto& downParams = Down_->Parameters();
	downParams[0].Data = OaFnMatrix::RandXavier(OaMatrixShape{InDModel, InDFF}, wd);
	downParams[0].Data.SetRequiresGrad(true);
	downParams[1].Data.SetRequiresGrad(true);
	
	// Register as children for parameter tracking
	RegisterModule("norm", Norm_);
	RegisterModule("gate", Gate_);
	RegisterModule("up", Up_);
	RegisterModule("down", Down_);
}

OaMatrix OaFfn::Forward(const OaMatrix& InX) {
	// RMSNorm
	auto normed = Norm_->Forward(InX);
	
	// Gate and Up projections
	auto gate = Gate_->Forward(normed);
	auto up = Up_->Forward(normed);
	
	// SwiGLU: silu(gate) * up, fused into a single Swiglu dispatch.
	auto swiglu = OaFnMatrix::Swiglu(gate, up);
	
	// Down projection
	auto out = Down_->Forward(swiglu);
	
	// Residual connection
	return OaFnMatrix::Add(InX, out);
}

