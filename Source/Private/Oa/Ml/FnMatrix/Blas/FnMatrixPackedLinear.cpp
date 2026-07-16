#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Autograd/Matrix/AutogradMatrix.h>
#include <Oa/Runtime/Context.h>

namespace {
OaU32 DivCeil(OaU32 a, OaU32 b) { return (a + b - 1U) / b; }
bool ValidWeight(const OaMatrix& x, const OaMatrix& w) {
	return x.Rank() == 2 and w.Rank() == 2 and x.GetDtype() == w.GetDtype()
		and x.Size(1) == w.Size(1);
}
bool ValidBias(const OaMatrix& b, const OaMatrix& w) {
	return b.IsEmpty() or (b.Rank() == 1 and b.GetDtype() == w.GetDtype()
		and b.Size(0) == w.Size(0));
}
} // namespace

OaMatrix OaFnMatrix::PackedLinear2(
	const OaMatrix& InX, const OaMatrix& InWeight0, const OaMatrix& InWeight1,
	const OaMatrix& InBias0, const OaMatrix& InBias1) {
	if (not ValidWeight(InX, InWeight0) or not ValidWeight(InX, InWeight1)
		or not ValidBias(InBias0, InWeight0) or not ValidBias(InBias1, InWeight1)
		or InBias0.IsEmpty() != InBias1.IsEmpty()) return {};
	const OaU32 M = static_cast<OaU32>(InX.Size(0)), K = static_cast<OaU32>(InX.Size(1));
	const OaU32 N0 = static_cast<OaU32>(InWeight0.Size(0)), N1 = static_cast<OaU32>(InWeight1.Size(0));
	auto out = Empty({M, N0 + N1}, InX.GetDtype());
	const bool hasBias = not InBias0.IsEmpty();
	const OaMatrix& dummy = InWeight0;
	struct Push { OaU32 M, K, N0, N1, N2, Count, HasBias; } push{M, K, N0, N1, 0U, 2U, hasBias};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("PackedLinear", {&InX, &InWeight0, &InWeight1, &dummy,
		hasBias ? &InBias0 : &dummy, hasBias ? &InBias1 : &dummy, &dummy, &out}, access,
		&push, sizeof(push), DivCeil(M * (N0 + N1), 256U));
	if (OaFnAutograd::IsEnabled() and (InX.RequiresGrad() or InWeight0.RequiresGrad()
		or InWeight1.RequiresGrad() or (hasBias and (InBias0.RequiresGrad() or InBias1.RequiresGrad())))) {
		auto grad = OaMakeSharedPtr<OaGradPackedLinear2>();
		grad->Saved_ = {InX, InWeight0, InWeight1};
		grad->N0_ = N0; grad->N1_ = N1; grad->HasBias_ = hasBias;
		OaVec<OaMatrix> inputs{InX, InWeight0, InWeight1};
		if (hasBias) { inputs.PushBack(InBias0); inputs.PushBack(InBias1); }
		grad->SetGraphInputs(inputs); grad->SequenceNr_ = OaFnAutograd::NextSeq();
		grad->OutputShape_ = out.GetShape(); out.MutAutograd().GradFn = grad;
	}
	return out;
}

OaMatrix OaFnMatrix::PackedLinear3(
	const OaMatrix& InX, const OaMatrix& InWeight0, const OaMatrix& InWeight1,
	const OaMatrix& InWeight2, const OaMatrix& InBias0, const OaMatrix& InBias1,
	const OaMatrix& InBias2) {
	if (not ValidWeight(InX, InWeight0) or not ValidWeight(InX, InWeight1)
		or not ValidWeight(InX, InWeight2) or not ValidBias(InBias0, InWeight0)
		or not ValidBias(InBias1, InWeight1) or not ValidBias(InBias2, InWeight2)
		or InBias0.IsEmpty() != InBias1.IsEmpty() or InBias0.IsEmpty() != InBias2.IsEmpty()) return {};
	const OaU32 M = static_cast<OaU32>(InX.Size(0)), K = static_cast<OaU32>(InX.Size(1));
	const OaU32 N0 = static_cast<OaU32>(InWeight0.Size(0)), N1 = static_cast<OaU32>(InWeight1.Size(0));
	const OaU32 N2 = static_cast<OaU32>(InWeight2.Size(0));
	auto out = Empty({M, N0 + N1 + N2}, InX.GetDtype());
	const bool hasBias = not InBias0.IsEmpty();
	const OaMatrix& dummy = InWeight0;
	struct Push { OaU32 M, K, N0, N1, N2, Count, HasBias; } push{M, K, N0, N1, N2, 3U, hasBias};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("PackedLinear", {&InX, &InWeight0, &InWeight1, &InWeight2,
		hasBias ? &InBias0 : &dummy, hasBias ? &InBias1 : &dummy,
		hasBias ? &InBias2 : &dummy, &out}, access, &push, sizeof(push),
		DivCeil(M * (N0 + N1 + N2), 256U));
	if (OaFnAutograd::IsEnabled() and (InX.RequiresGrad() or InWeight0.RequiresGrad()
		or InWeight1.RequiresGrad() or InWeight2.RequiresGrad() or (hasBias and
			(InBias0.RequiresGrad() or InBias1.RequiresGrad() or InBias2.RequiresGrad())))) {
		auto grad = OaMakeSharedPtr<OaGradPackedLinear3>();
		grad->Saved_ = {InX, InWeight0, InWeight1, InWeight2};
		grad->N0_ = N0; grad->N1_ = N1; grad->N2_ = N2; grad->HasBias_ = hasBias;
		OaVec<OaMatrix> inputs{InX, InWeight0, InWeight1, InWeight2};
		if (hasBias) { inputs.PushBack(InBias0); inputs.PushBack(InBias1); inputs.PushBack(InBias2); }
		grad->SetGraphInputs(inputs); grad->SequenceNr_ = OaFnAutograd::NextSeq();
		grad->OutputShape_ = out.GetShape(); out.MutAutograd().GradFn = grad;
	}
	return out;
}
