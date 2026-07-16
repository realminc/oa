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
	
	// Gate and Up share the same activation. Keep the independently named child
	// modules/checkpoint weights, but execute both projections in one dispatch.
	auto& gateParams = Gate_->Parameters();
	auto& upParams = Up_->Parameters();
	const bool hasBias = gateParams.Size() > 1 and upParams.Size() > 1;
	auto packed = hasBias
		? OaFnMatrix::PackedLinear2(normed, gateParams[0].Data, upParams[0].Data,
			gateParams[1].Data, upParams[1].Data)
		: OaFnMatrix::PackedLinear2(normed, gateParams[0].Data, upParams[0].Data);
	OaI64 widths[] = {DFF_, DFF_};
	auto gateUp = OaFnMatrix::Split(packed, OaSpan<OaI64>(widths, 2), 1);
	auto gate = gateUp[0];
	auto up = gateUp[1];
	
	// SwiGLU: silu(gate) * up, fused into a single Swiglu dispatch.
	auto swiglu = OaFnMatrix::Swiglu(gate, up);
	
	// Down projection
	auto out = Down_->Forward(swiglu);
	
	// Residual connection
	return OaFnMatrix::Add(InX, out);
}
