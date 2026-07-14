// OaFnMatrix::Bmm — batched matrix multiply A[N,M,K] @ B[N,K,P] = out[N,M,P].
// GPU kernel "Bmm" (Ops/Bmm.slang) + autograd (OaGradBmm). The enabler for
// differentiable forward kinematics and any per-row small-matrix algebra.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

namespace {
OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }
}

OaMatrix OaFnMatrix::Bmm(const OaMatrix& InA, const OaMatrix& InB) {
	auto& ctx = OaContext::GetDefault();
	OA_ASSERT(InA.Rank() == 3 && InB.Rank() == 3 && "Bmm expects rank-3 [N,M,K] @ [N,K,P]");
	const OaI64 N = InA.Size(0), M = InA.Size(1), K = InA.Size(2), P = InB.Size(2);
	OA_ASSERT(InB.Size(0) == N && InB.Size(1) == K && "Bmm batch/inner dim mismatch");

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{N, M, P}, InA.GetDtype());
	struct { OaU32 Batch; OaU32 M; OaU32 K; OaU32 P; } push{
		static_cast<OaU32>(N), static_cast<OaU32>(M), static_cast<OaU32>(K), static_cast<OaU32>(P)
	};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Bmm", {&InA, &InB, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(N * M * P), 256));

	if (OaFnAutograd::IsEnabled() and (InA.RequiresGrad() or InB.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradBmm>();
		gradFn->Saved_       = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(  OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_  = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}
