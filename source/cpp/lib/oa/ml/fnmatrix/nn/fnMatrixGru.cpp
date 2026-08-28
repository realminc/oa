// oa::FnMatrix GRU — Fused GRU cell pointwise operations.
//
// GruCellPointwise replaces ~10 elementwise dispatches (Slice×6 + Sigmoid×2 +
// Tanh + Mul×2 + Add/Sub×3) with a single per-element kernel over [B, H].

#include <oa/ml/fnMatrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/autograd/matrix/autogradBlas.h>
#include <oa/ml/autograd/matrix/autogradRecurrent.h>
#include <oa/runtime/executionSession.h>

#include "../../autograd/autogradAttach.gen.h"

namespace {

void attachLinearProjection(
	oa::Matrix& outGatesH,
	const oa::Matrix& inHidden,
	const oa::Matrix& inWeightHh,
	const oa::Matrix& inBiasHh)
{
	const bool hasBias = not inBiasHh.isEmpty();
	if (not oa::FnAutograd::isEnabled()
		or not (inHidden.requiresGrad() or inWeightHh.requiresGrad()
			or (hasBias and inBiasHh.requiresGrad())))
	{
		return;
	}
	auto grad = oa::makeShared<oa::GradLinear>();
	grad->saveForBackward(inHidden, inWeightHh);
	oa::Vector<oa::Matrix> inputs{inHidden, inWeightHh};
	if (hasBias) inputs.pushBack(inBiasHh);
	grad->setGraphInputs(oa::move(inputs));
	grad->sequenceNr_ = oa::FnAutograd::nextSeq();
	grad->outputShape_ = outGatesH.getShape();
	outGatesH.mutAutograd().gradFn = grad;
}

oa::Status attachGruCellLinear(
	oa::Matrix& out,
	oa::Matrix& inOutGatesH,
	const oa::Matrix& inGatesI,
	const oa::Matrix& inHidden,
	const oa::Matrix& inWeightHh,
	const oa::Matrix& inBiasHh,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride,
	oa::U32 inSemanticOp)
{
	attachLinearProjection(inOutGatesH, inHidden, inWeightHh, inBiasHh);
	if (not oa::FnAutograd::isEnabled()
		or not (inGatesI.requiresGrad() or inOutGatesH.requiresGrad()
			or inHidden.requiresGrad()))
	{
		return oa::Status::ok();
	}
	auto grad = oa::makeShared<oa::GradGruCellPointwise>();
	grad->saveForBackward(inGatesI, inOutGatesH, inHidden);
	grad->setGraphInputs({inGatesI, inOutGatesH, inHidden});
	grad->sequenceNr_ = oa::FnAutograd::nextSeq();
	grad->outputShape_ = out.getShape();
	grad->hiddenSize_ = inHiddenSize;
	grad->timeOffset_ = inTimeOffset;
	grad->batchStride_ = inBatchStride;
	OA_RETURN_IF_ERROR(oa::FnAutograd::attachSemantic(
		grad, inSemanticOp));
	out.mutAutograd().gradFn = grad;
	return oa::Status::ok();
}

oa::Status attachGruScan(
	oa::Matrix& out,
	const oa::Matrix& inHprev,
	const oa::Matrix& inGatesI,
	const oa::Matrix& inWeightHh,
	const oa::Matrix& inBiasHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	oa::U32 inSemanticOp)
{
	const bool hasBias = not inBiasHh.isEmpty();
	if (not oa::FnAutograd::isEnabled()
		or not (inGatesI.requiresGrad() or inWeightHh.requiresGrad()
			or (hasBias and inBiasHh.requiresGrad())))
	{
		return oa::Status::ok();
	}
	auto grad = oa::makeShared<oa::GradGruScan>();
	grad->saveForBackward(inGatesI, inWeightHh,
		hasBias ? inBiasHh : inGatesI, inHprev);
	oa::Vector<oa::Matrix> inputs{inGatesI, inWeightHh};
	if (hasBias) inputs.pushBack(inBiasHh);
	grad->setGraphInputs(oa::move(inputs));
	grad->sequenceNr_ = oa::FnAutograd::nextSeq();
	grad->outputShape_ = out.getShape();
	grad->hiddenSize_ = inHiddenSize;
	grad->seqLen_ = inSeqLen;
	grad->batch_ = inBatch;
	grad->hasBias_ = hasBias;
	OA_RETURN_IF_ERROR(oa::FnAutograd::attachSemantic(
		grad, inSemanticOp));
	out.mutAutograd().gradFn = grad;
	return oa::Status::ok();
}

} // namespace

namespace oa {

namespace FnMatrix {

oa::Matrix gruCellPointwise(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inGatesH,
	const oa::Matrix& inHidden,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 B = static_cast<oa::I32>(inHidden.size(0));
	const oa::I32 H = inHiddenSize;
	
	oa::Matrix out = empty(oa::MatrixShape{B, H}, inHidden.getDtype());
	
	struct {
		oa::U32 hidden_size;
		oa::U32 count;
		oa::U32 time_offset;
		oa::U32 batch_stride;
	} push{
		static_cast<oa::U32>(H),
		static_cast<oa::U32>(B * H),
		inTimeOffset,
		inBatchStride
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // gates_h
		oa::BufferAccess::Read,  // hidden
		oa::BufferAccess::Write  // out
	};

	ctx.add( "GruCellPointwise",
		{&inGatesI, &inGatesH, &inHidden, &out},
		access, &push, sizeof(push), oa::divCeil(static_cast<oa::U32>(B * H), 256));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::gruCellPointwise,
		{&inGatesI, &inGatesH, &inHidden}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromUnsignedInteger("timeOffset", inTimeOffset),
			oa::OpAttribute::fromUnsignedInteger("batchStride", inBatchStride),
		});
	if (not semantic.isOk()) return {};
	const auto attached =
		oa::detail::generatedAutogradAttach::FnMatrix::gruCellPointwise(
			out, inGatesI, inGatesH, inHidden, inHiddenSize, inTimeOffset,
			inBatchStride, semantic.getValue());
	if (not attached.isOk()) return {};
	return out;
}

oa::GruCellPointwiseBwdResult gruCellPointwiseBwd(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inGatesH,
	const oa::Matrix& inHidden,
	const oa::Matrix& inGradOutput,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 B = static_cast<oa::I32>(inHidden.size(0));
	const oa::I32 H = inHiddenSize;
	
	oa::Matrix dGatesI = zeros(inGatesI.getShape(), inGatesI.getDtype());
	oa::Matrix dGatesH = empty(inGatesH.getShape(), inGatesH.getDtype());
	oa::Matrix dHidden = empty(inHidden.getShape(), inHidden.getDtype());
	
	struct {
		oa::U32 hidden_size;
		oa::U32 count;
		oa::U32 time_offset;
		oa::U32 batch_stride;
	} push {
		static_cast<oa::U32>(H),
		static_cast<oa::U32>(B * H),
		inTimeOffset,
		inBatchStride
	};
	
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // gates_h
		oa::BufferAccess::Read,  // hidden
		oa::BufferAccess::Read,  // grad_output
		oa::BufferAccess::Write, // d_gates_i
		oa::BufferAccess::Write, // d_gates_h
		oa::BufferAccess::Write  // d_hidden
	};
	
	ctx.add( "GruCellPointwiseBwd",
		{&inGatesI, &inGatesH, &inHidden, &inGradOutput, &dGatesI, &dGatesH, &dHidden},
		access, &push, sizeof(push), oa::divCeil(static_cast<oa::U32>(B * H), 256));

	oa::GruCellPointwiseBwdResult result{
		.dGatesI = dGatesI,
		.dGatesH = dGatesH,
		.dHidden = dHidden,
	};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::gruCellPointwiseBwd,
		{&inGatesI, &inGatesH, &inHidden, &inGradOutput},
		{&result.dGatesI, &result.dGatesH, &result.dHidden},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromUnsignedInteger("timeOffset", inTimeOffset),
			oa::OpAttribute::fromUnsignedInteger("batchStride", inBatchStride),
		}).isOk())
	{
		return {};
	}
	return result;
}

oa::Matrix gruCellLinear(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inHidden,
	const oa::Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride,
	const oa::Matrix& inBiasHh
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 B = static_cast<oa::I32>(inHidden.size(0));
	const oa::I32 H = inHiddenSize;
	const bool hasBias = not inBiasHh.isEmpty();

	oa::Matrix out = empty(oa::MatrixShape{B, H}, inHidden.getDtype());
	// The kernel always writes gatesH because reverse mode needs the exact hidden
	// projection. It is private saved state, not a second public result.
	oa::Matrix gatesH = empty(oa::MatrixShape{B, static_cast<oa::I64>(3) * H}, inHidden.getDtype());
	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inHidden;

	struct {
		oa::U32 hidden_size;
		oa::U32 batch;
		oa::U32 time_offset;
		oa::U32 batch_stride;
		oa::U32 has_bias;
		oa::U32 save_gates_h;
	} push{};
	push.hidden_size = static_cast<oa::U32>(H);
	push.batch = static_cast<oa::U32>(B);
	push.time_offset = inTimeOffset;
	push.batch_stride = inBatchStride;
	push.has_bias = hasBias ? 1U : 0U;
	push.save_gates_h = 1U;

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // h
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Write, // out
		oa::BufferAccess::Write  // gates_h
	};

	ctx.add( "GruCellLinear",
		{&inHidden, &inWeightHh, biasHh, &inGatesI, &out, &gatesH},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::gruCellLinear,
		{&inGatesI, &inHidden, &inWeightHh, hasBias ? &inBiasHh : nullptr},
		{&out},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromUnsignedInteger("timeOffset", inTimeOffset),
			oa::OpAttribute::fromUnsignedInteger("batchStride", inBatchStride),
		});
	if (not semantic.isOk()) return {};
	if (not attachGruCellLinear(
		out, gatesH, inGatesI, inHidden, inWeightHh, inBiasHh,
		inHiddenSize, inTimeOffset, inBatchStride, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

// ─── GruScan / GruScanBwd — whole-sequence recurrent scan ────────────────────
// One workgroup per batch element; the timestep loop runs inside the kernel, so the
// whole recurrent pass is a single dispatch (forward and backward each). See the
// .slang headers for the math. weight/bias grad reuses LinearWeightBiasBwd.

oa::GruScanResult gruScan(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const oa::Matrix& inBiasHh)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 H = inHiddenSize;
	const oa::I32 S = inSeqLen;
	const oa::I32 B = inBatch;
	const bool hasBias = not inBiasHh.isEmpty();

	oa::Matrix out   = empty(oa::MatrixShape{B, S, H}, inGatesI.getDtype());
	oa::Matrix hprev = empty(oa::MatrixShape{B, S, H}, inGatesI.getDtype());
	// Placeholder bind when no bias (kernel gates has_bias=0 and ignores b_hh_idx).
	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inGatesI;

	struct {
		oa::U32 hidden_size;
		oa::U32 seq_len;
		oa::U32 batch;
		oa::U32 has_bias;
	} push{
		static_cast<oa::U32>(H),
		static_cast<oa::U32>(S),
		static_cast<oa::U32>(B),
		hasBias ? 1U : 0U};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Write, // out
		oa::BufferAccess::Write  // hprev
	};

	ctx.add( "GruScan",
		{&inGatesI, &inWeightHh, biasHh, &out, &hprev},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	oa::GruScanResult result{.out = out, .hPrev = hprev};
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::gruScan,
		{&inGatesI, &inWeightHh, hasBias ? &inBiasHh : nullptr},
		{&result.out, &result.hPrev},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
		});
	if (not semantic.isOk()) return {};
	if (not attachGruScan(
		result.out, result.hPrev, inGatesI, inWeightHh, inBiasHh,
		inHiddenSize, inSeqLen, inBatch, semantic.getValue()).isOk())
	{
		return {};
	}
	return result;
}

oa::GruScanBwdResult gruScanBwd(
	const oa::Matrix& inDOut,
	const oa::Matrix& inGatesI,
	const oa::Matrix& inHprev,
	const oa::Matrix& inWeightHh,
	oa::I32 inHiddenSize,
	oa::I32 inSeqLen,
	oa::I32 inBatch,
	const oa::Matrix& inBiasHh)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 H = inHiddenSize;
	const oa::I32 S = inSeqLen;
	const oa::I32 B = inBatch;
	const bool hasBias = not inBiasHh.isEmpty();

	oa::Matrix dGatesI = zeros(inGatesI.getShape(), inGatesI.getDtype());
	oa::Matrix dGatesH = empty(oa::MatrixShape{static_cast<oa::I64>(B) * S, static_cast<oa::I64>(3) * H}, inGatesI.getDtype());
	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inGatesI;

	struct {
		oa::U32 hidden_size;
		oa::U32 seq_len;
		oa::U32 batch;
		oa::U32 has_bias;
	} push{
		static_cast<oa::U32>(H),
		static_cast<oa::U32>(S),
		static_cast<oa::U32>(B),
		hasBias ? 1U : 0U};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // dout
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // hprev
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Write, // d_gates_i
		oa::BufferAccess::Write  // d_gates_h
	};

	ctx.add( "GruScanBwd",
		{&inDOut, &inGatesI, &inHprev, &inWeightHh, biasHh, &dGatesI, &dGatesH},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	oa::GruScanBwdResult result{.dGatesI = dGatesI, .dGatesH = dGatesH};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::gruScanBwd,
		{
			&inDOut, &inGatesI, &inHprev, &inWeightHh,
			hasBias ? &inBiasHh : nullptr,
		},
		{&result.dGatesI, &result.dGatesH},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
		}).isOk())
	{
		return {};
	}
	return result;
}

} // namespace FnMatrix

} // namespace oa
