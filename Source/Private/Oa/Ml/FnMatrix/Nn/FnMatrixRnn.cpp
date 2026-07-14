// OaFnMatrix RNN — Fused vanilla-RNN cell pointwise operations.
//
// RnnCellPointwise replaces the Add + Tanh dispatch pair with a single
// per-element kernel over [B, H]:  h_new = tanh(gates_i + gates_h).

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>

namespace OaFnMatrix {

OaMatrix RnnCellPointwise(
	const OaMatrix& InGatesI,
	const OaMatrix& InGatesH
) {
	auto& ctx = OaContext::GetDefault();
	const OaI64 count = InGatesI.NumElements();

	OaMatrix out = Empty(InGatesI.GetShape(), InGatesI.GetDtype());

	// Buffer bindless indices are auto-prepended to the push block by the
	// bindless dispatch path (see OaVkStream::RecordDispatch); the shader reads
	// them as its leading struct members. The user push therefore carries only
	// the scalar params, in the order the shader declares *after* the indices.
	struct {
		OaU32 count;
	} push{
		static_cast<OaU32>(count)
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // gates_h
		OaBufferAccess::Write  // out
	};

	ctx.Add("RnnCellPointwise",
		{&InGatesI, &InGatesH, &out},
		access, &push, sizeof(push), OaDivCeil(static_cast<OaU32>(count), 256));

	return out;
}

OaRnnCellPointwiseBwdResult RnnCellPointwiseBwd(
	const OaMatrix& InGatesI,
	const OaMatrix& InGatesH,
	const OaMatrix& InGradOutput,
	OaI32 InHiddenSize,
	OaU32 InTimeOffset,
	OaU32 InBatchStride
) {
	auto& ctx = OaContext::GetDefault();
	// gates_h / grad_output are this timestep's [B, H]; gates_i is the whole-sequence
	// [B*T, H] projection. count = B*H threads (one timestep). d_gates_i carries the
	// full gi shape, zeroed, so each timestep scatters only its own rows — disjoint
	// fan-in over the shared gi buffer accumulates to the correct input-projection grad.
	const OaI32 H = InHiddenSize;
	const OaI32 B = static_cast<OaI32>(InGradOutput.Size(0));
	const OaI64 count = static_cast<OaI64>(B) * H;

	OaMatrix dGatesI = Zeros(InGatesI.GetShape(), InGatesI.GetDtype());
	OaMatrix dGatesH = Empty(InGatesH.GetShape(), InGatesH.GetDtype());

	// Buffer bindless indices are auto-prepended by the bindless dispatch path;
	// the user push carries only the scalar params (see forward for details).
	struct {
		OaU32 hidden_size;
		OaU32 count;
		OaU32 time_offset;
		OaU32 batch_stride;
	} push{
		static_cast<OaU32>(H),
		static_cast<OaU32>(count),
		InTimeOffset,
		InBatchStride
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // gates_h
		OaBufferAccess::Read,  // grad_output
		OaBufferAccess::Write, // d_gates_i
		OaBufferAccess::Write  // d_gates_h
	};

	ctx.Add("RnnCellPointwiseBwd",
		{&InGatesI, &InGatesH, &InGradOutput, &dGatesI, &dGatesH},
		access, &push, sizeof(push), OaDivCeil(static_cast<OaU32>(count), 256));

	return {.DGatesI = dGatesI, .DGatesH = dGatesH};
}

OaMatrix RnnCellLinear(
	const OaMatrix& InGi,
	const OaMatrix& InHidden,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaU32 InTimeOffset,
	OaU32 InBatchStride,
	OaMatrix* OutGatesH
) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = static_cast<OaI32>(InHidden.Size(0));
	const OaI32 H = static_cast<OaI32>(InHidden.Size(1));

	OaMatrix out = Empty(OaMatrixShape{B, H}, InHidden.GetDtype());
	// gh must own real storage: the kernel writes the hidden projection here and the
	// backward path reads it back. Binding the caller's default-constructed (storage-
	// less) OutGatesH would route the write to a null bindless slot, so gh reads back
	// as zeros in RnnCellPointwiseBwd — silently breaking the recurrent/input grads.
	OaMatrix gh = Empty(OaMatrixShape{B, H}, InHidden.GetDtype());
	OaMatrix* ghPtr = &gh;

	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InHidden;  // placeholder when bias absent

	struct {
		OaU32 hidden_size;
		OaU32 batch;
		OaU32 time_offset;
		OaU32 batch_stride;
		OaU32 has_bias;
		OaU32 save_gh;
	} push{};
	push.hidden_size = static_cast<OaU32>(H);
	push.batch = static_cast<OaU32>(B);
	push.time_offset = InTimeOffset;
	push.batch_stride = InBatchStride;
	push.has_bias = (InBiasHh != nullptr) ? 1U : 0U;
	push.save_gh = 1U;

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // h
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Read,  // gi
		OaBufferAccess::Write, // out
		OaBufferAccess::Write  // gh
	};

	ctx.Add("RnnCellLinear",
		{&InHidden, &InWeightHh, biasHh, &InGi, &out, ghPtr},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	if (OutGatesH != nullptr) {
		*OutGatesH = gh;
	}
	return out;
}

OaRnnScanResult RnnScan(
	const OaMatrix& InGatesI,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch)
{
	auto& ctx = OaContext::GetDefault();
	const OaI32 H = InHiddenSize;
	const OaI32 S = InSeqLen;
	const OaI32 B = InBatch;

	OaMatrix out   = OaFnMatrix::Empty(OaMatrixShape{B, S, H}, InGatesI.GetDtype());
	OaMatrix hprev = OaFnMatrix::Empty(OaMatrixShape{B, S, H}, InGatesI.GetDtype());

	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InGatesI;  // placeholder when bias absent

	struct Push {
		OaU32 HiddenSize;
		OaU32 SeqLen;
		OaU32 Batch;
		OaU32 HasBias;
	} push{};
	push.HiddenSize = static_cast<OaU32>(H);
	push.SeqLen     = static_cast<OaU32>(S);
	push.Batch      = static_cast<OaU32>(B);
	push.HasBias    = (InBiasHh != nullptr) ? 1U : 0U;

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Write, // out
		OaBufferAccess::Write  // hprev
	};

	ctx.Add("RnnScan",
		{&InGatesI, &InWeightHh, biasHh, &out, &hprev},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	return {.Out = out, .Hprev = hprev};
}

OaRnnScanBwdResult RnnScanBwd(
	const OaMatrix& InDOut,
	const OaMatrix& InGatesI,
	const OaMatrix& InHprev,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaI32 InSeqLen,
	OaI32 InBatch)
{
	auto& ctx = OaContext::GetDefault();
	const OaI32 H = InHiddenSize;
	const OaI32 S = InSeqLen;
	const OaI32 B = InBatch;

	OaMatrix dGatesI = OaFnMatrix::Zeros(InGatesI.GetShape(), InGatesI.GetDtype());
	OaMatrix dGatesH = OaFnMatrix::Zeros(InGatesI.GetShape(), InGatesI.GetDtype());

	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InGatesI;

	struct Push {
		OaU32 HiddenSize;
		OaU32 SeqLen;
		OaU32 Batch;
		OaU32 HasBias;
	} push{};
	push.HiddenSize = static_cast<OaU32>(H);
	push.SeqLen     = static_cast<OaU32>(S);
	push.Batch      = static_cast<OaU32>(B);
	push.HasBias    = (InBiasHh != nullptr) ? 1U : 0U;

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // dout
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // hprev
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Write, // d_gates_i
		OaBufferAccess::Write  // d_gates_h
	};

	ctx.Add("RnnScanBwd",
		{&InDOut, &InGatesI, &InHprev, &InWeightHh, biasHh, &dGatesI, &dGatesH},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	return {.DGatesI = dGatesI, .DGatesH = dGatesH};
}

} // namespace OaFnMatrix
