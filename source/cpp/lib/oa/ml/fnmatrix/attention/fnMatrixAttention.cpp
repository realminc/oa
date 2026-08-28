#include <oa/ml/fnMatrix.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

namespace oa {
namespace {

class GradSplitHeads final : public GradNode {
public:
	oa::I32 batch = 0;
	oa::I32 seqLen = 0;
	oa::I32 numHeads = 0;

	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (!outDIn.empty()) outDIn[0] = oa::FnMatrix::mergeHeads(inDOut, batch, seqLen, numHeads);
	}
};

class GradMergeHeads final : public GradNode {
public:
	oa::I32 batch = 0;
	oa::I32 seqLen = 0;
	oa::I32 numHeads = 0;
	void backward(const oa::Matrix& inDOut, oa::Vector<oa::Matrix>& outDIn) override {
		if (!outDIn.empty()) outDIn[0] = oa::FnMatrix::splitHeads(inDOut, batch, seqLen, numHeads);
	}
};

} // namespace
} // namespace oa

oa::Matrix oa::FnMatrix::splitHeads(
	const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen, oa::I32 inNumHeads) {
	if (inX.rank() != 2 || inBatch <= 0 || inSeqLen <= 0 || inNumHeads <= 0 ||
		inX.size(0) != static_cast<oa::I64>(inBatch) * inSeqLen ||
		inX.size(1) % inNumHeads != 0) return {};
	const oa::U32 b = static_cast<oa::U32>(inBatch);
	const oa::U32 s = static_cast<oa::U32>(inSeqLen);
	const oa::U32 h = static_cast<oa::U32>(inNumHeads);
	const oa::U32 p = static_cast<oa::U32>(inX.size(1) / inNumHeads);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out;
	// With one head, [B*S,D] and [B*H,S,D/H] == [B,S,D] have identical
	// contiguous storage. Keep this as a differentiable metadata-only view. The
	// canonical NLP tutorials use one head, so this removes three forward copies
	// for Q/K/V and their three inverse backward copies without a new kernel.
	if (inNumHeads == 1) {
		out = oa::FnMatrix::reshape(inX, oa::MatrixShape{b, s, p});
	} else {
		out = oa::FnMatrix::empty(oa::MatrixShape{b * h, s, p}, inX.getDtype());
		struct { oa::U32 Batch, seqLen, numHeads, headDim; } push{b, s, h, p};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "SplitHeads", {&inX, &out}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(out.numElements()), 256));
	}

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::splitHeads,
		{&inX}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger(
				"sequenceLength", inSeqLen),
			oa::OpAttribute::fromSignedInteger(
				"headCount", inNumHeads),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() && inX.requiresGrad() && inNumHeads != 1) {
		auto grad = oa::makeShared<oa::GradSplitHeads>();
		grad->setGraphInputs({inX});
		grad->sequenceNr_ = oa::FnAutograd::nextSeq();
		grad->outputShape_ = out.getShape();
		grad->batch = inBatch;
		grad->seqLen = inSeqLen;
		grad->numHeads = inNumHeads;
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = grad;
	}
	return out;
}

oa::Matrix oa::FnMatrix::mergeHeads(const oa::Matrix& inX, oa::I32 inBatch, oa::I32 inSeqLen, oa::I32 inNumHeads) {
	if (inX.rank() != 3 || inBatch <= 0 || inSeqLen <= 0 || inNumHeads <= 0 ||
		inX.size(0) != static_cast<oa::I64>(inBatch) * inNumHeads ||
		inX.size(1) != inSeqLen
	) return {};
	const oa::U32 b = static_cast<oa::U32>(inBatch);
	const oa::U32 s = static_cast<oa::U32>(inSeqLen);
	const oa::U32 h = static_cast<oa::U32>(inNumHeads);
	const oa::U32 p = static_cast<oa::U32>(inX.size(2));
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix out;
	// exact inverse of the one-head SplitHeads view above. No permutation exists
	// when H == 1, so materializing a copy is unnecessary.
	if (inNumHeads == 1) {
		out = oa::FnMatrix::reshape(inX, oa::MatrixShape{b * s, p});
	} else {
		out = oa::FnMatrix::empty(oa::MatrixShape{b * s, h * p}, inX.getDtype());
		struct { oa::U32 Batch, seqLen, numHeads, headDim; } push{b, s, h, p};
		oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		ctx.add( "MergeHeads", {&inX, &out}, access, &push, sizeof(push),
			divCeil(static_cast<oa::U32>(out.numElements()), 256));
	}

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::mergeHeads,
		{&inX}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger(
				"sequenceLength", inSeqLen),
			oa::OpAttribute::fromSignedInteger(
				"headCount", inNumHeads),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() && inX.requiresGrad() && inNumHeads != 1) {
		auto grad = oa::makeShared<oa::GradMergeHeads>();
		grad->setGraphInputs({inX});
		grad->sequenceNr_ = oa::FnAutograd::nextSeq();
		grad->outputShape_ = out.getShape();
		grad->batch = inBatch;
		grad->seqLen = inSeqLen;
		grad->numHeads = inNumHeads;
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = grad;
	}
	return out;
}
