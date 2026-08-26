#include <oa/ml/fnMatrix.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/assert.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executionSession.h>

namespace oa {
namespace {

constexpr oa::I64 kFlashMaxSeqLen = 1024;

void validateFlashInputs(
	const oa::Matrix& inQ, const oa::Matrix& inK, const oa::Matrix& inV) {
	OA_REQUIRE_MSG(inQ.rank() == 3 && inK.getShape() == inQ.getShape()
		&& inV.getShape() == inQ.getShape(),
		"FlashAttentionCausal expects equal Q/K/V [batchHeads,sequence,headDim]");
	OA_REQUIRE_MSG(inQ.size(0) > 0 && inQ.size(1) > 0 && inQ.size(2) > 0
		&& inQ.size(1) <= kFlashMaxSeqLen,
		"FlashAttentionCausal requires non-empty dimensions and sequence <= 1024");
	OA_REQUIRE_MSG(inK.getDtype() == inQ.getDtype()
		&& inV.getDtype() == inQ.getDtype(),
		"FlashAttentionCausal requires one Q/K/V dtype");
	OA_REQUIRE_MSG(inQ.getDtype() == oa::ScalarType::Float32,
		"FlashAttentionCausal v1 is verified for Float32 storage only");
}

class GradFlashAttentionCausal final : public GradNode {
public:
	oa::F32 scale = 1.0F;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		if (outDIn.empty()) return;
		auto gradients = oa::FnMatrix::flashAttentionCausalBwd(
			saved(0), saved(1), saved(2), saved(3), saved(4), inDOut, scale);
		if (outDIn.size() > 0) outDIn[0] = gradients.dq;
		if (outDIn.size() > 1) outDIn[1] = gradients.dk;
		if (outDIn.size() > 2) outDIn[2] = gradients.dv;
	}
};

} // namespace
} // namespace oa

oa::Matrix oa::FnMatrix::flashAttentionCausal(
	const oa::Matrix& inQ, const oa::Matrix& inK, const oa::Matrix& inV, oa::F32 inScale) {
	validateFlashInputs(inQ, inK, inV);
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto output = oa::FnMatrix::empty(inQ.getShape(), inQ.getDtype());
	// log-sum-exp is a numerical statistic, not an activation payload. Keep it
	// FP32 even when Q/K/V use BF16 storage so backward probability recomputation
	// has the same stability contract as forward.
	auto logSumExp = oa::FnMatrix::empty(
		oa::MatrixShape{inQ.size(0), inQ.size(1)}, oa::ScalarType::Float32);
	struct Push {
		oa::U32 batchHeads;
		oa::U32 seqLen;
		oa::U32 headDim;
		oa::F32 scale;
	} push{
		static_cast<oa::U32>(inQ.size(0)), static_cast<oa::U32>(inQ.size(1)),
		static_cast<oa::U32>(inQ.size(2)), inScale,
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
	};
	ctx.add( "FlashCausal", {&inQ, &inK, &inV, &output, &logSumExp},
		access, &push, sizeof(push), push.batchHeads * push.seqLen, 1, 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::flashAttentionCausal,
		{&inQ, &inK, &inV}, {&output, &logSumExp},
		{oa::OpAttribute::fromFloat("scale", inScale)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() &&
		(inQ.requiresGrad() || inK.requiresGrad() || inV.requiresGrad())) {
		auto grad = oa::makeShared<oa::GradFlashAttentionCausal>();
		grad->saveForBackward(inQ, inK, inV, output, logSumExp);
		grad->setGraphInputs({inQ, inK, inV});
		grad->sequenceNr_ = oa::FnAutograd::nextSeq();
		grad->outputShape_ = output.getShape();
		grad->scale = inScale;
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
		output.mutAutograd().gradFn = grad;
	}
	return output;
}

oa::FlashAttentionBwdResult oa::FnMatrix::flashAttentionCausalBwd(
	const oa::Matrix& inQ, const oa::Matrix& inK, const oa::Matrix& inV,
	const oa::Matrix& inOutput, const oa::Matrix& inLogSumExp,
	const oa::Matrix& inGradOutput, oa::F32 inScale) {
	validateFlashInputs(inQ, inK, inV);
	OA_REQUIRE_MSG((inOutput.getShape() == inQ.getShape()
		&& inGradOutput.getShape() == inQ.getShape()
		&& inLogSumExp.getShape() == oa::MatrixShape{inQ.size(0), inQ.size(1)}
		&& inLogSumExp.getDtype() == oa::ScalarType::Float32),
		"FlashAttentionCausalBwd received incompatible saved tensors");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	auto gradQ = oa::FnMatrix::empty(inQ.getShape(), inQ.getDtype());
	auto gradK = oa::FnMatrix::empty(inK.getShape(), inK.getDtype());
	auto gradV = oa::FnMatrix::empty(inV.getShape(), inV.getDtype());
	struct Push {
		oa::U32 batchHeads;
		oa::U32 seqLen;
		oa::U32 headDim;
		oa::F32 scale;
	} push{
		static_cast<oa::U32>(inQ.size(0)), static_cast<oa::U32>(inQ.size(1)),
		static_cast<oa::U32>(inQ.size(2)), inScale,
	};
	oa::BufferAccess qAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add( "FlashCausalBwdQ",
		{&inQ, &inK, &inV, &inOutput, &inGradOutput, &inLogSumExp, &gradQ},
		qAccess, &push, sizeof(push), push.batchHeads * push.seqLen, 1, 1);
	oa::BufferAccess kvAccess[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write,
	};
	ctx.add( "FlashCausalBwdKV",
		{&inQ, &inK, &inV, &inOutput, &inGradOutput, &inLogSumExp, &gradK, &gradV},
		kvAccess, &push, sizeof(push), push.batchHeads * push.seqLen, 1, 1);
	oa::FlashAttentionBwdResult result{
		.dq = gradQ,
		.dk = gradK,
		.dv = gradV,
	};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::flashAttentionCausalBwd,
		{&inQ, &inK, &inV, &inOutput, &inLogSumExp, &inGradOutput},
		{&result.dq, &result.dk, &result.dv},
		{oa::OpAttribute::fromFloat("scale", inScale)}).isOk())
	{
		return {};
	}
	return result;
}
