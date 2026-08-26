// FnMatrixChannelNorm — fused channel-wise LayerNorm for [B,C,T] tensors.
// Replaces normC (Transpose + LayerNorm + Transpose = 3 dispatches) with 1.
// Supports autograd via oa::GradChannelNorm (uses ChannelNormBwd kernel).

#include <oa/ml/fnMatrix.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/ml/autograd.h>
#include <oa/ml/autograd/matrix/autogradChannelNorm.h>

#include <assert.h>

oa::Matrix oa::FnMatrix::channelNorm(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.numElements() == static_cast<oa::I64>(inBatch) * inChannels * inSeqLen);
	assert(inWeight.numElements() == inChannels);
	assert(inBias.numElements() == inChannels);

	oa::Matrix out = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	struct {
		oa::U32 batch;
		oa::U32 channels;
		oa::U32 seqLen;
		oa::F32 eps;
	} push{
		.batch = static_cast<oa::U32>(inBatch),
		.channels = static_cast<oa::U32>(inChannels),
		.seqLen = static_cast<oa::U32>(inSeqLen),
		.eps = inEps};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // weight
		oa::BufferAccess::Read,   // bias
		oa::BufferAccess::Write   // out
	};
	ctx.add( "ChannelNorm",
		{&inX, &inWeight, &inBias, &out},
		access, &push, sizeof(push),
		static_cast<oa::U32>(inBatch) * static_cast<oa::U32>(inSeqLen));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::channelNorm,
		{&inX, &inWeight, &inBias}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger("channels", inChannels),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromFloat("eps", inEps),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and
		(inX.requiresGrad() or inWeight.requiresGrad() or inBias.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradChannelNorm>();
		gradFn->batch_ = inBatch;
		gradFn->channels_ = inChannels;
		gradFn->seqLen_ = inSeqLen;
		gradFn->eps_ = inEps;
		gradFn->saveForBackward(inX, inWeight);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inX, inWeight, inBias});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
		out.setRequiresGrad(true);
	}

	return out;
}

oa::ChannelNormBwdResult oa::FnMatrix::channelNormBwd(
	const oa::Matrix& inX, const oa::Matrix& inWeight,
	const oa::Matrix& inGradOutput,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.numElements() == static_cast<oa::I64>(inBatch) * inChannels * inSeqLen);
	assert(inWeight.numElements() == inChannels);

	const oa::I64 rows = static_cast<oa::I64>(inBatch) * inSeqLen;

	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	oa::Matrix dWcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, inChannels}, inX.getDtype());
	oa::Matrix dBcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, inChannels}, inX.getDtype());

	struct {
		oa::U32 batch;
		oa::U32 channels;
		oa::U32 seqLen;
		oa::F32 eps;
	} push{
		.batch = static_cast<oa::U32>(inBatch),
		.channels = static_cast<oa::U32>(inChannels),
		.seqLen = static_cast<oa::U32>(inSeqLen),
		.eps = inEps};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // w
		oa::BufferAccess::Read,   // dy
		oa::BufferAccess::Write,  // dx
		oa::BufferAccess::Write,  // dw_contrib
		oa::BufferAccess::Write   // dbias_contrib
	};
	ctx.add( "ChannelNormBwd",
		{&inX, &inWeight, &inGradOutput, &dX, &dWcontrib, &dBcontrib},
		access, &push, sizeof(push),
		static_cast<oa::U32>(rows));

	oa::Matrix dWeight = oa::FnMatrix::sum(dWcontrib, 0).reshape(inWeight.getShape());
	oa::Matrix dBias = oa::FnMatrix::sum(dBcontrib, 0).reshape(inWeight.getShape());

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::channelNormBwd,
		{&inX, &inWeight, &inGradOutput}, {&dX, &dWeight, &dBias},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger("channels", inChannels),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromFloat("eps", inEps),
		}).isOk())
	{
		return {};
	}
	return {.dx = dX, .dWeight = dWeight, .dBias = dBias};
}

// ─── ChannelNormRelu: fused ChannelNorm + ReLU ────────────────────────────

oa::Matrix oa::FnMatrix::channelNormRelu(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.numElements() == static_cast<oa::I64>(inBatch) * inChannels * inSeqLen);
	assert(inWeight.numElements() == inChannels);
	assert(inBias.numElements() == inChannels);

	oa::Matrix out = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	struct {
		oa::U32 batch;
		oa::U32 channels;
		oa::U32 seqLen;
		oa::F32 eps;
	} push{
		.batch = static_cast<oa::U32>(inBatch),
		.channels = static_cast<oa::U32>(inChannels),
		.seqLen = static_cast<oa::U32>(inSeqLen),
		.eps = inEps};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // weight
		oa::BufferAccess::Read,   // bias
		oa::BufferAccess::Write   // out
	};
	ctx.add( "ChannelNormRelu",
		{&inX, &inWeight, &inBias, &out},
		access, &push, sizeof(push),
		static_cast<oa::U32>(inBatch) * static_cast<oa::U32>(inSeqLen));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::channelNormRelu,
		{&inX, &inWeight, &inBias}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger("channels", inChannels),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromFloat("eps", inEps),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and
		(inX.requiresGrad() or inWeight.requiresGrad() or inBias.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradChannelNormRelu>();
		gradFn->batch_ = inBatch;
		gradFn->channels_ = inChannels;
		gradFn->seqLen_ = inSeqLen;
		gradFn->eps_ = inEps;
		gradFn->saveForBackward(inX, inWeight, out);
		gradFn->setGraphInputs(oa::Vec<oa::Matrix>{inX, inWeight, inBias});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
		out.mutAutograd().gradFn = gradFn;
		out.setRequiresGrad(true);
	}

	return out;
}

oa::ChannelNormBwdResult oa::FnMatrix::channelNormReluBwd(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inFwdOut,
	const oa::Matrix& inGradOutput,
	oa::I32 inBatch, oa::I32 inChannels, oa::I32 inSeqLen, oa::F32 inEps)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.numElements() == static_cast<oa::I64>(inBatch) * inChannels * inSeqLen);
	assert(inWeight.numElements() == inChannels);

	const oa::I64 rows = static_cast<oa::I64>(inBatch) * inSeqLen;

	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	oa::Matrix dWcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, inChannels}, inX.getDtype());
	oa::Matrix dBcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, inChannels}, inX.getDtype());

	struct {
		oa::U32 batch;
		oa::U32 channels;
		oa::U32 seqLen;
		oa::F32 eps;
	} push{
		.batch = static_cast<oa::U32>(inBatch),
		.channels = static_cast<oa::U32>(inChannels),
		.seqLen = static_cast<oa::U32>(inSeqLen),
		.eps = inEps};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // x
		oa::BufferAccess::Read,   // w
		oa::BufferAccess::Read,   // fwd_out (for ReLU mask)
		oa::BufferAccess::Read,   // dy
		oa::BufferAccess::Write,  // dx
		oa::BufferAccess::Write,  // dw_contrib
		oa::BufferAccess::Write   // dbias_contrib
	};
	ctx.add( "ChannelNormReluBwd",
		{&inX, &inWeight, &inFwdOut, &inGradOutput, &dX, &dWcontrib, &dBcontrib},
		access, &push, sizeof(push),
		static_cast<oa::U32>(rows));

	oa::Matrix dWeight = oa::FnMatrix::sum(dWcontrib, 0).reshape(inWeight.getShape());
	oa::Matrix dBias = oa::FnMatrix::sum(dBcontrib, 0).reshape(inWeight.getShape());

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::channelNormReluBwd,
		{&inX, &inWeight, &inFwdOut, &inGradOutput},
		{&dX, &dWeight, &dBias},
		{
			oa::OpAttribute::fromSignedInteger("batch", inBatch),
			oa::OpAttribute::fromSignedInteger("channels", inChannels),
			oa::OpAttribute::fromSignedInteger("sequenceLength", inSeqLen),
			oa::OpAttribute::fromFloat("eps", inEps),
		}).isOk())
	{
		return {};
	}
	return {.dx = dX, .dWeight = dWeight, .dBias = dBias};
}
