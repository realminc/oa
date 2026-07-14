// OaRoPE — Rotary Position Embedding (RoPE)
//
// Dispatches RopeApply/RopeApplyBwd compute shaders.

#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>
#include <cassert>
#include <cmath>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

OaRoPE::OaRoPE(OaI32 InNumHeads, OaI32 InHeadDim, OaF32 InThetaBase)
	: NumHeads_(InNumHeads), HeadDim_(InHeadDim), ThetaBase_(InThetaBase) {}

OaMatrix OaRoPE::Forward(const OaMatrix& InInput) {
	OaI64 T = InInput.Size(0);
	OaI64 D = InInput.Size(1);
	OaI64 expectedD = static_cast<OaI64>(NumHeads_) * HeadDim_;
	assert(D == expectedD && "OaRoPE: input dim must match num_heads * head_dim");
	(void)expectedD;

	auto& ctx = OaContext::GetDefault();
	OaMatrix out = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	OaU32 halfDim = static_cast<OaU32>(HeadDim_ / 2);
	OaU32 total = static_cast<OaU32>(T) * static_cast<OaU32>(NumHeads_) * halfDim;

	struct Push {
		OaU32 T;
		OaU32 num_heads;
		OaU32 head_dim;
		OaF32 theta_base;
		OaU32 pos_offset;
	} push{
		static_cast<OaU32>(T),
		static_cast<OaU32>(NumHeads_),
		static_cast<OaU32>(HeadDim_),
		ThetaBase_,
		0u
	};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("RopeApply", {&InInput, &out}, access, &push, sizeof(push), DivCeil(total, 256));

	if (OaFnAutograd::IsEnabled() and InInput.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradRoPE>();
		gradFn->Saved_ = OaVec<OaMatrix>{InInput};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InInput});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->NumHeads_ = NumHeads_;
		gradFn->HeadDim_ = HeadDim_;
		gradFn->ThetaBase_ = ThetaBase_;
		gradFn->PosOffset_ = 0;
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}
