// FnMatrixChannelNorm — fused channel-wise LayerNorm for [B,C,T] tensors.
// Replaces NormC (Transpose + LayerNorm + Transpose = 3 dispatches) with 1.
// Supports autograd via OaGradChannelNorm (uses ChannelNormBwd kernel).

#include <Oa/Ml/FnMatrix.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ml/Autograd.h>
#include <Ml/Autograd/Matrix/AutogradMatrix.h>

#include <cassert>

OaMatrix OaFnMatrix::ChannelNorm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps)
{
	auto& ctx = OaContext::GetDefault();
	assert(InX.NumElements() == static_cast<OaI64>(InBatch) * InChannels * InSeqLen);
	assert(InWeight.NumElements() == InChannels);
	assert(InBias.NumElements() == InChannels);

	OaMatrix out = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	struct {
		OaU32 Batch;
		OaU32 Channels;
		OaU32 SeqLen;
		OaF32 Eps;
	} push{
		.Batch = static_cast<OaU32>(InBatch),
		.Channels = static_cast<OaU32>(InChannels),
		.SeqLen = static_cast<OaU32>(InSeqLen),
		.Eps = InEps};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // weight
		OaBufferAccess::Read,   // bias
		OaBufferAccess::Write   // out
	};
	ctx.Add("ChannelNorm",
		{&InX, &InWeight, &InBias, &out},
		access, &push, sizeof(push),
		static_cast<OaU32>(InBatch) * static_cast<OaU32>(InSeqLen));

	if (OaFnAutograd::IsEnabled() and
		(InX.RequiresGrad() or InWeight.RequiresGrad() or InBias.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradChannelNorm>();
		gradFn->Batch_ = InBatch;
		gradFn->Channels_ = InChannels;
		gradFn->SeqLen_ = InSeqLen;
		gradFn->Eps_ = InEps;
		gradFn->Saved_ = OaVec<OaMatrix>{InX, InWeight};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX, InWeight, InBias});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
		out.SetRequiresGrad(true);
	}

	return out;
}

OaFnMatrix::OaChannelNormBwdResult OaFnMatrix::ChannelNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InGradOutput,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps)
{
	auto& ctx = OaContext::GetDefault();
	assert(InX.NumElements() == static_cast<OaI64>(InBatch) * InChannels * InSeqLen);
	assert(InWeight.NumElements() == InChannels);

	const OaI64 rows = static_cast<OaI64>(InBatch) * InSeqLen;

	OaMatrix dX = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	OaMatrix dWcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, InChannels}, InX.Dtype_);
	OaMatrix dBcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, InChannels}, InX.Dtype_);

	struct {
		OaU32 Batch;
		OaU32 Channels;
		OaU32 SeqLen;
		OaF32 Eps;
	} push{
		.Batch = static_cast<OaU32>(InBatch),
		.Channels = static_cast<OaU32>(InChannels),
		.SeqLen = static_cast<OaU32>(InSeqLen),
		.Eps = InEps};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // w
		OaBufferAccess::Read,   // dy
		OaBufferAccess::Write,  // dx
		OaBufferAccess::Write,  // dw_contrib
		OaBufferAccess::Write   // dbias_contrib
	};
	ctx.Add("ChannelNormBwd",
		{&InX, &InWeight, &InGradOutput, &dX, &dWcontrib, &dBcontrib},
		access, &push, sizeof(push),
		static_cast<OaU32>(rows));

	OaMatrix dWeight = OaFnMatrix::Sum(dWcontrib, 0).Reshape(InWeight.GetShape());
	OaMatrix dBias = OaFnMatrix::Sum(dBcontrib, 0).Reshape(InWeight.GetShape());

	return {.DX = dX, .DWeight = dWeight, .DBias = dBias};
}

// ─── ChannelNormRelu: fused ChannelNorm + ReLU ────────────────────────────

OaMatrix OaFnMatrix::ChannelNormRelu(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps)
{
	auto& ctx = OaContext::GetDefault();
	assert(InX.NumElements() == static_cast<OaI64>(InBatch) * InChannels * InSeqLen);
	assert(InWeight.NumElements() == InChannels);
	assert(InBias.NumElements() == InChannels);

	OaMatrix out = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	struct {
		OaU32 Batch;
		OaU32 Channels;
		OaU32 SeqLen;
		OaF32 Eps;
	} push{
		.Batch = static_cast<OaU32>(InBatch),
		.Channels = static_cast<OaU32>(InChannels),
		.SeqLen = static_cast<OaU32>(InSeqLen),
		.Eps = InEps};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // weight
		OaBufferAccess::Read,   // bias
		OaBufferAccess::Write   // out
	};
	ctx.Add("ChannelNormRelu",
		{&InX, &InWeight, &InBias, &out},
		access, &push, sizeof(push),
		static_cast<OaU32>(InBatch) * static_cast<OaU32>(InSeqLen));

	if (OaFnAutograd::IsEnabled() and
		(InX.RequiresGrad() or InWeight.RequiresGrad() or InBias.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradChannelNormRelu>();
		gradFn->Batch_ = InBatch;
		gradFn->Channels_ = InChannels;
		gradFn->SeqLen_ = InSeqLen;
		gradFn->Eps_ = InEps;
		gradFn->Saved_ = OaVec<OaMatrix>{InX, InWeight, out};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX, InWeight, InBias});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
		out.SetRequiresGrad(true);
	}

	return out;
}

OaFnMatrix::OaChannelNormBwdResult OaFnMatrix::ChannelNormReluBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InFwdOut,
	const OaMatrix& InGradOutput,
	OaI32 InBatch, OaI32 InChannels, OaI32 InSeqLen, OaF32 InEps)
{
	auto& ctx = OaContext::GetDefault();
	assert(InX.NumElements() == static_cast<OaI64>(InBatch) * InChannels * InSeqLen);
	assert(InWeight.NumElements() == InChannels);

	const OaI64 rows = static_cast<OaI64>(InBatch) * InSeqLen;

	OaMatrix dX = OaFnMatrix::Empty(InX.Shape_, InX.Dtype_);
	OaMatrix dWcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, InChannels}, InX.Dtype_);
	OaMatrix dBcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, InChannels}, InX.Dtype_);

	struct {
		OaU32 Batch;
		OaU32 Channels;
		OaU32 SeqLen;
		OaF32 Eps;
	} push{
		.Batch = static_cast<OaU32>(InBatch),
		.Channels = static_cast<OaU32>(InChannels),
		.SeqLen = static_cast<OaU32>(InSeqLen),
		.Eps = InEps};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // x
		OaBufferAccess::Read,   // w
		OaBufferAccess::Read,   // fwd_out (for ReLU mask)
		OaBufferAccess::Read,   // dy
		OaBufferAccess::Write,  // dx
		OaBufferAccess::Write,  // dw_contrib
		OaBufferAccess::Write   // dbias_contrib
	};
	ctx.Add("ChannelNormReluBwd",
		{&InX, &InWeight, &InFwdOut, &InGradOutput, &dX, &dWcontrib, &dBcontrib},
		access, &push, sizeof(push),
		static_cast<OaU32>(rows));

	OaMatrix dWeight = OaFnMatrix::Sum(dWcontrib, 0).Reshape(InWeight.GetShape());
	OaMatrix dBias = OaFnMatrix::Sum(dBcontrib, 0).Reshape(InWeight.GetShape());

	return {.DX = dX, .DWeight = dWeight, .DBias = dBias};
}
