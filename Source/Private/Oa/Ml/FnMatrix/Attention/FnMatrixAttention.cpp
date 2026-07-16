#include <Oa/Ml/FnMatrix.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Runtime/Context.h>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

class OaGradSplitHeads final : public OaGradNode {
public:
	OaI32 Batch = 0;
	OaI32 SeqLen = 0;
	OaI32 NumHeads = 0;

	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (!OutDIn.Empty()) OutDIn[0] = OaFnMatrix::MergeHeads(InDOut, Batch, SeqLen, NumHeads);
	}
};

class OaGradMergeHeads final : public OaGradNode {
public:
	OaI32 Batch = 0;
	OaI32 SeqLen = 0;
	OaI32 NumHeads = 0;
	void Backward(const OaMatrix& InDOut, OaVec<OaMatrix>& OutDIn) override {
		if (!OutDIn.Empty()) OutDIn[0] = OaFnMatrix::SplitHeads(InDOut, Batch, SeqLen, NumHeads);
	}
};

} // namespace

OaMatrix OaFnMatrix::SplitHeads(
	const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen, OaI32 InNumHeads) {
	if (InX.Rank() != 2 || InBatch <= 0 || InSeqLen <= 0 || InNumHeads <= 0 ||
		InX.Size(0) != static_cast<OaI64>(InBatch) * InSeqLen ||
		InX.Size(1) % InNumHeads != 0) return {};
	const OaU32 b = static_cast<OaU32>(InBatch);
	const OaU32 s = static_cast<OaU32>(InSeqLen);
	const OaU32 h = static_cast<OaU32>(InNumHeads);
	const OaU32 p = static_cast<OaU32>(InX.Size(1) / InNumHeads);
	// With one head, [B*S,D] and [B*H,S,D/H] == [B,S,D] have identical
	// contiguous storage. Keep this as a differentiable metadata-only view. The
	// canonical NLP tutorials use one head, so this removes three forward copies
	// for Q/K/V and their three inverse backward copies without a new kernel.
	if (InNumHeads == 1) {
		return OaFnMatrix::Reshape(InX, OaMatrixShape{b, s, p});
	}
	auto out = OaFnMatrix::Empty(OaMatrixShape{b * h, s, p}, InX.GetDtype());
	struct { OaU32 Batch, SeqLen, NumHeads, HeadDim; } push{b, s, h, p};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("SplitHeads", {&InX, &out}, access, &push, sizeof(push), DivCeil(static_cast<OaU32>(out.NumElements()), 256));

	if (OaFnAutograd::IsEnabled() && InX.RequiresGrad()) {
		auto grad = OaMakeSharedPtr<OaGradSplitHeads>();
		grad->SetGraphInputs({InX});
		grad->SequenceNr_ = OaFnAutograd::NextSeq();
		grad->OutputShape_ = out.GetShape();
		grad->Batch = InBatch;
		grad->SeqLen = InSeqLen;
		grad->NumHeads = InNumHeads;
		out.MutAutograd().GradFn = grad;
	}
	return out;
}

OaMatrix OaFnMatrix::MergeHeads(const OaMatrix& InX, OaI32 InBatch, OaI32 InSeqLen, OaI32 InNumHeads) {
	if (InX.Rank() != 3 || InBatch <= 0 || InSeqLen <= 0 || InNumHeads <= 0 ||
		InX.Size(0) != static_cast<OaI64>(InBatch) * InNumHeads ||
		InX.Size(1) != InSeqLen
	) return {};
	const OaU32 b = static_cast<OaU32>(InBatch);
	const OaU32 s = static_cast<OaU32>(InSeqLen);
	const OaU32 h = static_cast<OaU32>(InNumHeads);
	const OaU32 p = static_cast<OaU32>(InX.Size(2));
	// Exact inverse of the one-head SplitHeads view above. No permutation exists
	// when H == 1, so materializing a copy is unnecessary.
	if (InNumHeads == 1) {
		return OaFnMatrix::Reshape(InX, OaMatrixShape{b * s, p});
	}
	auto out = OaFnMatrix::Empty(OaMatrixShape{b * s, h * p}, InX.GetDtype());
	struct { OaU32 Batch, SeqLen, NumHeads, HeadDim; } push{b, s, h, p};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MergeHeads", {&InX, &out}, access, &push, sizeof(push),
		DivCeil(static_cast<OaU32>(out.NumElements()), 256));
	if (OaFnAutograd::IsEnabled() && InX.RequiresGrad()) {
		auto grad = OaMakeSharedPtr<OaGradMergeHeads>();
		grad->SetGraphInputs({InX});
		grad->SequenceNr_ = OaFnAutograd::NextSeq();
		grad->OutputShape_ = out.GetShape();
		grad->Batch = InBatch;
		grad->SeqLen = InSeqLen;
		grad->NumHeads = InNumHeads;
		out.MutAutograd().GradFn = grad;
	}
	return out;
}
