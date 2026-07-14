// OaFnMatrix GRU — Fused GRU cell pointwise operations.
//
// GruCellPointwise replaces ~10 elementwise dispatches (Slice×6 + Sigmoid×2 +
// Tanh + Mul×2 + Add/Sub×3) with a single per-element kernel over [B, H].

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>

namespace OaFnMatrix {

OaMatrix GruCellPointwise(
	const OaMatrix& InGatesI,
	const OaMatrix& InGatesH,
	const OaMatrix& InHidden,
	OaI32 InHiddenSize,
	OaU32 InTimeOffset,
	OaU32 InBatchStride
) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = static_cast<OaI32>(InHidden.Size(0));
	const OaI32 H = InHiddenSize;
	
	OaMatrix out = Empty(OaMatrixShape{B, H}, InHidden.GetDtype());
	
	struct {
		OaU32 hidden_size;
		OaU32 count;
		OaU32 time_offset;
		OaU32 batch_stride;
	} push{
		static_cast<OaU32>(H),
		static_cast<OaU32>(B * H),
		InTimeOffset,
		InBatchStride
	};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // gates_h
		OaBufferAccess::Read,  // hidden
		OaBufferAccess::Write  // out
	};

	ctx.Add("GruCellPointwise",
		{&InGatesI, &InGatesH, &InHidden, &out},
		access, &push, sizeof(push), OaDivCeil(static_cast<OaU32>(B * H), 256));

	return out;
}

OaGruCellPointwiseBwdResult GruCellPointwiseBwd(
	const OaMatrix& InGatesI,
	const OaMatrix& InGatesH,
	const OaMatrix& InHidden,
	const OaMatrix& InGradOutput,
	OaI32 InHiddenSize,
	OaU32 InTimeOffset,
	OaU32 InBatchStride
) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = static_cast<OaI32>(InHidden.Size(0));
	const OaI32 H = InHiddenSize;
	
	OaMatrix dGatesI = Zeros(InGatesI.GetShape(), InGatesI.GetDtype());
	OaMatrix dGatesH = Empty(InGatesH.GetShape(), InGatesH.GetDtype());
	OaMatrix dHidden = Empty(InHidden.GetShape(), InHidden.GetDtype());
	
	struct {
		OaU32 hidden_size;
		OaU32 count;
		OaU32 time_offset;
		OaU32 batch_stride;
	} push {
		static_cast<OaU32>(H),
		static_cast<OaU32>(B * H),
		InTimeOffset,
		InBatchStride
	};
	
	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // gates_h
		OaBufferAccess::Read,  // hidden
		OaBufferAccess::Read,  // grad_output
		OaBufferAccess::Write, // d_gates_i
		OaBufferAccess::Write, // d_gates_h
		OaBufferAccess::Write  // d_hidden
	};
	
	ctx.Add("GruCellPointwiseBwd",
		{&InGatesI, &InGatesH, &InHidden, &InGradOutput, &dGatesI, &dGatesH, &dHidden},
		access, &push, sizeof(push), OaDivCeil(static_cast<OaU32>(B * H), 256));
	
	return {.DGatesI = dGatesI, .DGatesH = dGatesH, .DHidden = dHidden};
}

OaMatrix GruCellLinear(
	const OaMatrix& InGatesI,
	const OaMatrix& InHidden,
	const OaMatrix& InWeightHh,
	const OaMatrix* InBiasHh,
	OaI32 InHiddenSize,
	OaU32 InTimeOffset,
	OaU32 InBatchStride,
	OaMatrix* OutGatesH
) {
	auto& ctx = OaContext::GetDefault();
	const OaI32 B = static_cast<OaI32>(InHidden.Size(0));
	const OaI32 H = InHiddenSize;

	OaMatrix out = Empty(OaMatrixShape{B, H}, InHidden.GetDtype());
	// gatesH must own real storage: the kernel writes the [B, 3H] hidden projection
	// here and GruCellPointwiseBwd reads it back. Binding the caller's default-
	// constructed (storage-less) OutGatesH would route the write to a null bindless
	// slot, so gatesH reads back as zeros — silently corrupting the recurrent grads.
	OaMatrix gatesH = Empty(OaMatrixShape{B, static_cast<OaI64>(3) * H}, InHidden.GetDtype());
	OaMatrix* gatesHPtr = &gatesH;

	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InHidden;  // placeholder when bias absent

	struct {
		OaU32 hidden_size;
		OaU32 batch;
		OaU32 time_offset;
		OaU32 batch_stride;
		OaU32 has_bias;
		OaU32 save_gates_h;
	} push{};
	push.hidden_size = static_cast<OaU32>(H);
	push.batch = static_cast<OaU32>(B);
	push.time_offset = InTimeOffset;
	push.batch_stride = InBatchStride;
	push.has_bias = (InBiasHh != nullptr) ? 1U : 0U;
	push.save_gates_h = 1U;

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // h
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Write, // out
		OaBufferAccess::Write  // gates_h
	};

	ctx.Add("GruCellLinear",
		{&InHidden, &InWeightHh, biasHh, &InGatesI, &out, gatesHPtr},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	if (OutGatesH != nullptr) {
		*OutGatesH = gatesH;
	}
	return out;
}

// ─── GruScan / GruScanBwd — whole-sequence recurrent scan ────────────────────
// One workgroup per batch element; the timestep loop runs inside the kernel, so the
// whole recurrent pass is a single dispatch (forward and backward each). See the
// .slang headers for the math. Weight/bias grad reuses LinearWeightBiasBwd.

OaGruScanResult GruScan(
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

	OaMatrix out   = Empty(OaMatrixShape{B, S, H}, InGatesI.GetDtype());
	OaMatrix hprev = Empty(OaMatrixShape{B, S, H}, InGatesI.GetDtype());
	// Placeholder bind when no bias (kernel gates has_bias=0 and ignores b_hh_idx).
	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InGatesI;

	struct {
		OaU32 hidden_size;
		OaU32 seq_len;
		OaU32 batch;
		OaU32 has_bias;
	} push{
		static_cast<OaU32>(H),
		static_cast<OaU32>(S),
		static_cast<OaU32>(B),
		(InBiasHh != nullptr) ? 1U : 0U};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Write, // out
		OaBufferAccess::Write  // hprev
	};

	ctx.Add("GruScan",
		{&InGatesI, &InWeightHh, biasHh, &out, &hprev},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	return {.Out = out, .Hprev = hprev};
}

OaGruScanBwdResult GruScanBwd(
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

	OaMatrix dGatesI = Zeros(InGatesI.GetShape(), InGatesI.GetDtype());
	OaMatrix dGatesH = Empty(OaMatrixShape{static_cast<OaI64>(B) * S, static_cast<OaI64>(3) * H}, InGatesI.GetDtype());
	const OaMatrix* biasHh = (InBiasHh != nullptr) ? InBiasHh : &InGatesI;

	struct {
		OaU32 hidden_size;
		OaU32 seq_len;
		OaU32 batch;
		OaU32 has_bias;
	} push{
		static_cast<OaU32>(H),
		static_cast<OaU32>(S),
		static_cast<OaU32>(B),
		(InBiasHh != nullptr) ? 1U : 0U};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,  // dout
		OaBufferAccess::Read,  // gates_i
		OaBufferAccess::Read,  // hprev
		OaBufferAccess::Read,  // w_hh
		OaBufferAccess::Read,  // b_hh
		OaBufferAccess::Write, // d_gates_i
		OaBufferAccess::Write  // d_gates_h
	};

	ctx.Add("GruScanBwd",
		{&InDOut, &InGatesI, &InHprev, &InWeightHh, biasHh, &dGatesI, &dGatesH},
		access, &push, sizeof(push), static_cast<OaU32>(B));

	return {.DGatesI = dGatesI, .DGatesH = dGatesH};
}

} // namespace OaFnMatrix

