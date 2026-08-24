// oa::Ffn — SwiGLU FFN Implementation (level 1 API)

#include <oa/ml/nn/ffn/ffn.h>
#include <oa/core/fnMatrix.h>

oa::Ffn::Ffn(oa::I32 inDModel, oa::I32 inDFF, oa::F32 inRmsEps) {
	init(inDModel, inDFF, inRmsEps);
}

void oa::Ffn::init(oa::I32 inDModel, oa::I32 inDFF, oa::F32 inRmsEps) {
	dModel_ = inDModel;
	dFF_ = inDFF;
	rmsEps_ = inRmsEps;
	
	auto wd = oa::FnMatrix::weightDtype();
	
	// Create layers
	norm_ = oa::makeShared<oa::RmsNorm>(inDModel, inRmsEps);
	gate_ = oa::makeShared<oa::Linear>(inDModel, inDFF);
	up_ = oa::makeShared<oa::Linear>(inDModel, inDFF);
	down_ = oa::makeShared<oa::Linear>(inDFF, inDModel);
	
	// initialize weights with autograd tracking
	// grad is the single source of truth on each param's Data (oa::Parameter::grad());
	// setRequiresGrad allocates it. No manual snapshot to re-sync — reassigning Data
	// (e.g. RandXavier below) is automatically reflected by grad().
	auto& gateParams = gate_->parameters();
	gateParams[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{inDFF, inDModel}, wd);
	gateParams[0].data.setRequiresGrad(true);
	gateParams[1].data.setRequiresGrad(true);

	auto& upParams = up_->parameters();
	upParams[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{inDFF, inDModel}, wd);
	upParams[0].data.setRequiresGrad(true);
	upParams[1].data.setRequiresGrad(true);

	auto& downParams = down_->parameters();
	downParams[0].data = oa::FnMatrix::randXavier(oa::MatrixShape{inDModel, inDFF}, wd);
	downParams[0].data.setRequiresGrad(true);
	downParams[1].data.setRequiresGrad(true);
	
	// register as children for parameter tracking
	registerModule("norm", norm_);
	registerModule("gate", gate_);
	registerModule("up", up_);
	registerModule("down", down_);
}

oa::Matrix oa::Ffn::forward(const oa::Matrix& inX) {
	// RMSNorm
	auto normed = norm_->forward(inX);
	
	// gate and Up share the same activation but retain independent tuned GEMM
	// routes and output storage.
	auto gate = gate_->forward(normed);
	auto up = up_->forward(normed);
	
	// SwiGLU: silu(gate) * up, fused into a single Swiglu dispatch.
	auto swiglu = oa::FnMatrix::swiglu(gate, up);
	
	// Down projection
	auto out = down_->forward(swiglu);
	
	// residual connection
	return oa::FnMatrix::add(inX, out);
}
