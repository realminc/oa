// oa::FnMatrix RNN — Fused vanilla-RNN cell pointwise operations.
//
// RnnCellPointwise replaces the Add + Tanh dispatch pair with a single
// per-element kernel over [B, H]:  h_new = tanh(gates_i + gates_h).

#include <oa/ml/fnMatrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/ml/autograd/matrix/autogradBlas.h>
#include <oa/ml/autograd/matrix/autogradRecurrent.h>
#include <oa/runtime/executionSession.h>

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

oa::Status attachRnnPointwise(
	oa::Matrix& out,
	const oa::Matrix& inGatesI,
	const oa::Matrix& inGatesH,
	oa::U32 inSemanticOp)
{
	if (not oa::FnAutograd::isEnabled()
		or not (inGatesI.requiresGrad() or inGatesH.requiresGrad()))
	{
		return oa::Status::ok();
	}
	auto grad = oa::makeShared<oa::GradRnnCellPointwise>();
	grad->saveForBackward(inGatesI, inGatesH);
	grad->setGraphInputs({inGatesI, inGatesH});
	grad->sequenceNr_ = oa::FnAutograd::nextSeq();
	grad->outputShape_ = out.getShape();
	grad->hiddenSize_ = static_cast<oa::I32>(inGatesH.size(1));
	grad->timeOffset_ = 0;
	grad->batchStride_ = 1;
	OA_RETURN_IF_ERROR(oa::FnAutograd::attachSemantic(
		grad, inSemanticOp));
	out.mutAutograd().gradFn = grad;
	return oa::Status::ok();
}

oa::Status attachRnnCellLinear(
	oa::Matrix& out,
	oa::Matrix& inOutGatesH,
	const oa::Matrix& inGatesI,
	const oa::Matrix& inHidden,
	const oa::Matrix& inWeightHh,
	const oa::Matrix& inBiasHh,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride,
	oa::U32 inSemanticOp)
{
	attachLinearProjection(inOutGatesH, inHidden, inWeightHh, inBiasHh);
	if (not oa::FnAutograd::isEnabled()
		or not (inGatesI.requiresGrad() or inOutGatesH.requiresGrad()))
	{
		return oa::Status::ok();
	}
	auto grad = oa::makeShared<oa::GradRnnCellPointwise>();
	grad->saveForBackward(inGatesI, inOutGatesH);
	grad->setGraphInputs({inGatesI, inOutGatesH});
	grad->sequenceNr_ = oa::FnAutograd::nextSeq();
	grad->outputShape_ = out.getShape();
	grad->hiddenSize_ = static_cast<oa::I32>(inHidden.size(1));
	grad->timeOffset_ = inTimeOffset;
	grad->batchStride_ = inBatchStride;
	OA_RETURN_IF_ERROR(oa::FnAutograd::attachSemantic(
		grad, inSemanticOp));
	out.mutAutograd().gradFn = grad;
	return oa::Status::ok();
}

oa::Status attachRnnScan(
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
	auto grad = oa::makeShared<oa::GradRnnScan>();
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

oa::Matrix rnnCellPointwise(const oa::Matrix& inGatesI,	const oa::Matrix& inGatesH) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I64 count = inGatesI.numElements();

	oa::Matrix out = empty(inGatesI.getShape(), inGatesI.getDtype());

	// Buffer bindless indices are auto-prepended to the push block by the
	// bindless dispatch path (see oavk::Stream::RecordDispatch); the shader reads
	// them as its leading struct members. The user push therefore carries only
	// the scalar params, in the order the shader declares *after* the indices.
	struct {
		oa::U32 count;
	} push{
		static_cast<oa::U32>(count)
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // gates_h
		oa::BufferAccess::Write  // out
	};

	ctx.add( "RnnCellPointwise",
		{&inGatesI, &inGatesH, &out},
		access, &push, sizeof(push), oa::divCeil(static_cast<oa::U32>(count), 256)
	);

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::rnnCellPointwise,
		{&inGatesI, &inGatesH}, {&out});
	if (not semantic.isOk()) return {};
	if (not attachRnnPointwise(
		out, inGatesI, inGatesH, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::RnnCellPointwiseBwdResult rnnCellPointwiseBwd(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inGatesH,
	const oa::Matrix& inGradOutput,
	oa::I32 inHiddenSize,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// gates_h / grad_output are this timestep's [B, H]; gates_i is the whole-sequence
	// [B*T, H] projection. count = B*H threads (one timestep). d_gates_i carries the
	// full gi shape, zeroed, so each timestep scatters only its own rows — disjoint
	// fan-in over the shared gi buffer accumulates to the correct input-projection grad.
	const oa::I32 H = inHiddenSize;
	const oa::I32 B = static_cast<oa::I32>(inGradOutput.size(0));
	const oa::I64 count = static_cast<oa::I64>(B) * H;

	oa::Matrix dGatesI = zeros(inGatesI.getShape(), inGatesI.getDtype());
	oa::Matrix dGatesH = empty(inGatesH.getShape(), inGatesH.getDtype());

	// Buffer bindless indices are auto-prepended by the bindless dispatch path;
	// the user push carries only the scalar params (see forward for details).
	struct {
		oa::U32 hidden_size;
		oa::U32 count;
		oa::U32 time_offset;
		oa::U32 batch_stride;
	} push{
		static_cast<oa::U32>(H),
		static_cast<oa::U32>(count),
		inTimeOffset,
		inBatchStride
	};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // gates_h
		oa::BufferAccess::Read,  // grad_output
		oa::BufferAccess::Write, // d_gates_i
		oa::BufferAccess::Write  // d_gates_h
	};

	ctx.add( "RnnCellPointwiseBwd",
		{&inGatesI, &inGatesH, &inGradOutput, &dGatesI, &dGatesH},
		access, &push, sizeof(push), oa::divCeil(static_cast<oa::U32>(count), 256));

	oa::RnnCellPointwiseBwdResult result{
		.dGatesI = dGatesI,
		.dGatesH = dGatesH,
	};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::rnnCellPointwiseBwd,
		{&inGatesI, &inGatesH, &inGradOutput},
		{&result.dGatesI, &result.dGatesH},
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

oa::Matrix rnnCellLinear(
	const oa::Matrix& inGatesI,
	const oa::Matrix& inHidden,
	const oa::Matrix& inWeightHh,
	oa::U32 inTimeOffset,
	oa::U32 inBatchStride,
	const oa::Matrix& inBiasHh
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 B = static_cast<oa::I32>(inHidden.size(0));
	const oa::I32 H = static_cast<oa::I32>(inHidden.size(1));
	const bool hasBias = not inBiasHh.isEmpty();

	oa::Matrix out = empty(oa::MatrixShape{B, H}, inHidden.getDtype());
	// Reverse mode consumes the exact hidden projection, so the fused operation
	// retains it as private state rather than exposing an output parameter.
	oa::Matrix gh = empty(oa::MatrixShape{B, H}, inHidden.getDtype());
	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inHidden;

	struct {
		oa::U32 hidden_size;
		oa::U32 batch;
		oa::U32 time_offset;
		oa::U32 batch_stride;
		oa::U32 has_bias;
		oa::U32 save_gh;
	} push{};
	push.hidden_size = static_cast<oa::U32>(H);
	push.batch = static_cast<oa::U32>(B);
	push.time_offset = inTimeOffset;
	push.batch_stride = inBatchStride;
	push.has_bias = hasBias ? 1U : 0U;
	push.save_gh = 1U;

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // h
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Read,  // gi
		oa::BufferAccess::Write, // out
		oa::BufferAccess::Write  // gh
	};

	ctx.add( "RnnCellLinear",
		{&inHidden, &inWeightHh, biasHh, &inGatesI, &out, &gh},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::rnnCellLinear,
		{&inGatesI, &inHidden, &inWeightHh, hasBias ? &inBiasHh : nullptr},
		{&out},
		{
			oa::OpAttribute::fromUnsignedInteger("timeOffset", inTimeOffset),
			oa::OpAttribute::fromUnsignedInteger("batchStride", inBatchStride),
		});
	if (not semantic.isOk()) return {};
	if (not attachRnnCellLinear(
		out, gh, inGatesI, inHidden, inWeightHh, inBiasHh, inTimeOffset,
		inBatchStride, semantic.getValue()).isOk())
	{
		return {};
	}
	return out;
}

oa::RnnScanResult rnnScan(
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

	oa::Matrix out   = oa::FnMatrix::empty(oa::MatrixShape{B, S, H}, inGatesI.getDtype());
	oa::Matrix hprev = oa::FnMatrix::empty(oa::MatrixShape{B, S, H}, inGatesI.getDtype());

	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inGatesI;

	struct Push {
		oa::U32 hiddenSize;
		oa::U32 seqLen;
		oa::U32 batch;
		oa::U32 hasBias;
	} push{};
	push.hiddenSize = static_cast<oa::U32>(H);
	push.seqLen     = static_cast<oa::U32>(S);
	push.batch      = static_cast<oa::U32>(B);
	push.hasBias    = hasBias ? 1U : 0U;

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Write, // out
		oa::BufferAccess::Write  // hprev
	};

	ctx.add( "RnnScan",
		{&inGatesI, &inWeightHh, biasHh, &out, &hprev},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	oa::RnnScanResult result{.out = out, .hPrev = hprev};
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::rnnScan,
		{&inGatesI, &inWeightHh, hasBias ? &inBiasHh : nullptr},
		{&result.out, &result.hPrev},
		{
			oa::OpAttribute::fromSignedInteger("hiddenSize", inHiddenSize),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
		});
	if (not semantic.isOk()) return {};
	if (not attachRnnScan(
		result.out, result.hPrev, inGatesI, inWeightHh, inBiasHh,
		inHiddenSize, inSeqLen, inBatch, semantic.getValue()).isOk())
	{
		return {};
	}
	return result;
}

oa::RnnScanBwdResult rnnScanBwd(
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

	oa::Matrix dGatesI = oa::FnMatrix::zeros(inGatesI.getShape(), inGatesI.getDtype());
	oa::Matrix dGatesH = oa::FnMatrix::zeros(inGatesI.getShape(), inGatesI.getDtype());

	const oa::Matrix* biasHh = hasBias ? &inBiasHh : &inGatesI;

	struct Push {
		oa::U32 hiddenSize;
		oa::U32 seqLen;
		oa::U32 batch;
		oa::U32 hasBias;
	} push{};
	push.hiddenSize = static_cast<oa::U32>(H);
	push.seqLen     = static_cast<oa::U32>(S);
	push.batch      = static_cast<oa::U32>(B);
	push.hasBias    = hasBias ? 1U : 0U;

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,  // dout
		oa::BufferAccess::Read,  // gates_i
		oa::BufferAccess::Read,  // hprev
		oa::BufferAccess::Read,  // w_hh
		oa::BufferAccess::Read,  // b_hh
		oa::BufferAccess::Write, // d_gates_i
		oa::BufferAccess::Write  // d_gates_h
	};

	ctx.add( "RnnScanBwd",
		{&inDOut, &inGatesI, &inHprev, &inWeightHh, biasHh, &dGatesI, &dGatesH},
		access, &push, sizeof(push), static_cast<oa::U32>(B));

	oa::RnnScanBwdResult result{.dGatesI = dGatesI, .dGatesH = dGatesH};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::rnnScanBwd,
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
