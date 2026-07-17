// OaFnMatrix — Backward pass implementations for neural network operations.
// Manual implementations for gradient computation kernels.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Validation.h>

#include <cassert>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// ─── ReluBwd ────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::ReluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("ReluBwd", {&InForwardOutput, &InGradOutput, &gradInput}, access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── TanhBwd ────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::TanhBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("TanhBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── GeluBwd ────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::GeluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("GeluBwd", {&InInput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── SiluBwd ────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::SiluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SiluBwd", {&InInput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── SoftplusBwd ─────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::SoftplusBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SoftplusBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── SoftmaxBwd ─────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::SoftmaxBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();

	// The shader does a per-row jacobian (dx_i = s_i*(d_out_i - sum(d_out*s))) and
	// needs {rows, cols} after the 3 auto-prepended buffer indices, dispatched one
	// workgroup per row. The previous host pushed a single {Count} field (cols read
	// garbage) and dispatched n/256 groups — so for any rows>1 the gradient was
	// wrong. Match Softmax forward's row/col determination exactly.
	OaU32 rows, cols;
	if (InForwardOutput.Rank() == 2) {
		rows = static_cast<OaU32>(InForwardOutput.Size(0));
		cols = static_cast<OaU32>(InForwardOutput.Size(1));
	} else {
		rows = 1;
		cols = static_cast<OaU32>(InForwardOutput.NumElements());
	}

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Rows; OaU32 Cols; } push{rows, cols};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SoftmaxBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), rows);

	return gradInput;
}

OaMatrix OaFnMatrix::LogSoftmaxBwd(
	const OaMatrix& InForwardOutput,
	const OaMatrix& InGradOutput) {
	if (InForwardOutput.GetShape() != InGradOutput.GetShape()
		|| InForwardOutput.GetDtype() != InGradOutput.GetDtype()
		|| (InForwardOutput.Rank() != 1 && InForwardOutput.Rank() != 2)) {
		return {};
	}
	auto& ctx = OaContext::GetDefault();
	const OaU32 rows = InForwardOutput.Rank() == 2
		? static_cast<OaU32>(InForwardOutput.Size(0)) : 1U;
	const OaU32 cols = InForwardOutput.Rank() == 2
		? static_cast<OaU32>(InForwardOutput.Size(1))
		: static_cast<OaU32>(InForwardOutput.NumElements());
	OaMatrix gradInput = OaFnMatrix::Empty(
		InForwardOutput.GetShape(), InForwardOutput.GetDtype());
	struct { OaU32 Rows; OaU32 Cols; } push{rows, cols};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LogSoftmaxBwd",
		{&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), rows);
	return gradInput;
}

// ─── SoftmaxScaledMasked ─────────────────────────────────────────────────────

OaMatrix OaFnMatrix::SoftmaxScaledMasked(
	const OaMatrix& InScores, const OaMatrix& InMask, OaF32 InScale) {
	auto& ctx = OaContext::GetDefault();

	OaU32 rows, cols;
	if (InScores.Rank() == 2) {
		rows = static_cast<OaU32>(InScores.Size(0));
		cols = static_cast<OaU32>(InScores.Size(1));
	} else {
		rows = 1;
		cols = static_cast<OaU32>(InScores.NumElements());
	}

	OaMatrix out = OaFnMatrix::Empty(InScores.GetShape(), InScores.GetDtype());

	struct { OaU32 Rows; OaU32 Cols; OaF32 Scale; } push{rows, cols, InScale};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write
	};
	ctx.Add("SoftmaxScaledMasked", {&InScores, &InMask, &out}, access, &push, sizeof(push), rows, 1, 1);

	if (OaFnAutograd::IsEnabled() and InScores.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSoftmaxScaledMasked>();
		gradFn->Saved_ = OaVec<OaMatrix>{out};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InScores});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		gradFn->Scale_ = InScale;
		out.MutAutograd().GradFn = gradFn;
	}

	return out;
}

OaMatrix OaFnMatrix::SoftmaxScaledMaskedBwd(
	const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput, OaF32 InScale) {
	auto& ctx = OaContext::GetDefault();

	OaU32 rows, cols;
	if (InForwardOutput.Rank() == 2) {
		rows = static_cast<OaU32>(InForwardOutput.Size(0));
		cols = static_cast<OaU32>(InForwardOutput.Size(1));
	} else {
		rows = 1;
		cols = static_cast<OaU32>(InForwardOutput.NumElements());
	}

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Rows; OaU32 Cols; OaF32 Scale; } push{rows, cols, InScale};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write
	};
	ctx.Add("SoftmaxScaledMaskedBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), rows, 1, 1);

	return gradInput;
}

// ─── SwigluBwd ─────────────────────────────────────────────────────────────

OaFnMatrix::OaSwigluBwdResult OaFnMatrix::SwigluBwd(
	const OaMatrix& InGate, const OaMatrix& InUp, const OaMatrix& InOut,
	const OaMatrix& InGradOutput
) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InGate.NumElements());

	OaMatrix gateGrad = OaFnMatrix::Empty(InGate.GetShape(), InGate.GetDtype());
	OaMatrix upGrad = OaFnMatrix::Empty(InUp.GetShape(), InUp.GetDtype());

	// Shader declares exactly 5 buffer indices {gate, up, d_out, d_gate, d_up} and
	// recomputes silu(gate) itself — it never reads the forward output. Binding InOut
	// (a 6th buffer) shifted every index: d_out read from InOut, d_gate written to
	// InGradOutput, d_up written to gateGrad, upGrad never written. (void)InOut.
	(void)InOut;
	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write
	};
	ctx.Add("SwigluBwd", {&InGate, &InUp, &InGradOutput, &gateGrad, &upGrad},
		access, &push, sizeof(push), DivCeil(n, 256));

	return {.GateGrad = gateGrad, .UpGrad = upGrad};
}

// ─── LayerNormBwd ───────────────────────────────────────────────────────────

OaFnMatrix::OaLayerNormBwdResult OaFnMatrix::LayerNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	const OaMatrix& InOut, const OaMatrix& InMean, const OaMatrix& InRstd,
	const OaMatrix& InGradOutput
) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim (must match the forward): [B,T,C] → rows=B*T, cols=C.
	// The previous rows=Size(0)/cols=Size(1) wrote dWeight/dBias with the wrong stride
	// on rank-3 input → NaN gradients.
	const OaI64 cols = InX.Rank() >= 1 ? InX.Size(InX.Rank() - 1) : InX.NumElements();
	const OaI64 rows = cols > 0 ? InX.NumElements() / cols : 1;

	OaMatrix dX = OaFnMatrix::Empty(InX.GetShape(), InX.GetDtype());
	// The shader writes dw_contrib PER ELEMENT at row*cols + i — i.e. a full
	// [rows, cols] per-row contribution buffer that the host must column-sum over
	// rows to get dWeight [cols]. Binding a [cols]-sized buffer here (as the old
	// code did) overflowed the heap by (rows-1)*cols floats and returned only
	// row 0's contribution un-reduced → wrong dWeight + memory corruption for any
	// rows>1 (batch>1 or seq>1). rows==1 happened to fit exactly, hiding the bug.
	OaMatrix dWcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, cols}, InX.GetDtype());

	struct { OaU32 Rows; OaU32 Cols; OaF32 Eps; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(cols), 1e-5f};
	// Shader expects: x, w, dy, dx, dw_contrib (5 buffers)
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write
	};
	ctx.Add("LayerNormBwd",
		{&InX, &InWeight, &InGradOutput, &dX, &dWcontrib},
		access, &push, sizeof(push), static_cast<OaU32>(rows));

	// dWeight = column-sum of the per-row contributions over rows → [cols].
	OaMatrix dWeight = OaFnMatrix::Sum(dWcontrib, 0).Reshape(InWeight.GetShape());

	// dBias = sum of gradOutput over ALL rows (batch AND sequence). Flatten to
	// [rows, cols] first: Sum(InGradOutput, 0) on a rank-3 [B,T,C] would only sum
	// over B and return [T,C], not [C].
	OaMatrix dyRows = InGradOutput.Reshape(OaMatrixShape{rows, cols});
	OaMatrix dBias = OaFnMatrix::Sum(dyRows, 0).Reshape(InBias.GetShape());

	return {.DX = dX, .DWeight = dWeight, .DBias = dBias};
}

// ─── RmsNormBwd ────────────────────────────────────────────────────────────

OaFnMatrix::OaRmsNormBwdResult OaFnMatrix::RmsNormBwd(
	const OaMatrix& InX, const OaMatrix& InWeight,
	const OaMatrix& InOut, const OaMatrix& InRstd,
	const OaMatrix& InGradOutput
) {
	auto& ctx = OaContext::GetDefault();
	// Normalize over the LAST dim (must match the forward): [B,T,C] → rows=B*T, cols=C.
	// The previous rows=Size(0)/cols=Size(1) wrote dWeight/dBias with the wrong stride
	// on rank-3 input → NaN gradients.
	const OaI64 cols = InX.Rank() >= 1 ? InX.Size(InX.Rank() - 1) : InX.NumElements();
	const OaI64 rows = cols > 0 ? InX.NumElements() / cols : 1;

	OaMatrix dX = OaFnMatrix::Empty(InX.GetShape(), InX.GetDtype());
	// Per-row dw contribution buffer [rows, cols], column-summed to dWeight [cols]
	// below. The old code bound a [cols] buffer directly (heap overflow for rows>1)
	// and returned only row 0's contribution un-reduced. See LayerNormBwd above.
	OaMatrix dWcontrib = OaFnMatrix::Empty(OaMatrixShape{rows, cols}, InX.GetDtype());

	// The RmsNormBwd shader recomputes inv_rms = rsqrt(mean(x^2) + eps), so it needs
	// eps in the push tail. The old struct omitted it. It must mirror the shader's
	// {rows, cols, eps} tail and match the forward eps (1e-5, the RmsNorm default).
	struct { OaU32 Rows; OaU32 Cols; OaF32 Eps; } push{
		static_cast<OaU32>(rows), static_cast<OaU32>(cols), 1e-5f};
	// CRITICAL: the shader declares exactly 5 buffer indices
	// (x, w, dy, dx, dw_contrib). Buffer indices are auto-prepended IN ORDER, so the
	// bound set must be exactly those 5 — the shader recomputes inv_rms from x and
	// never reads `out`/`rstd`. The old code bound 7 buffers
	// {x, w, out, rstd, dy, dx, dw_contrib}; the extra out/rstd shifted everything,
	// so the shader read dy from `out`, wrote dx into `rstd`, and wrote dw_contrib
	// into the dy input → garbage, non-deterministic gradients.
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Write, OaBufferAccess::Write
	};
	ctx.Add("RmsNormBwd",
		{&InX, &InWeight, &InGradOutput, &dX, &dWcontrib},
		access, &push, sizeof(push), static_cast<OaU32>(rows));

	OaMatrix dWeight = OaFnMatrix::Sum(dWcontrib, 0).Reshape(InWeight.GetShape());

	return {.DX = dX, .DWeight = dWeight};
}

// ─── MaxPool2dBwd ────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::MaxPool2dBwd(
	const OaMatrix& InX, const OaMatrix& InIndices, const OaMatrix& InGradOutput,
	OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding
) {
	auto& ctx = OaContext::GetDefault();
	assert(InX.Rank() == 4 && "MaxPool2dBwd requires 4D input [N, C, H, W]");
	assert(InGradOutput.Rank() == 4 && "MaxPool2dBwd requires 4D grad_output");

	OaI64 N = InX.Size(0);
	OaI64 C = InX.Size(1);
	OaI64 H = InX.Size(2);
	OaI64 W = InX.Size(3);
	OaI64 H_out = InGradOutput.Size(2);
	OaI64 W_out = InGradOutput.Size(3);

	OaMatrix gradInput = OaFnMatrix::Empty(InX.GetShape(), InX.GetDtype());

	// Must mirror the shader's full PushConstants tail: the kernel walks each input
	// position's pooling window, so it needs OutHeight/OutWidth/KernelSize/Stride/
	// Padding. The old struct supplied only the first four fields (the rest read as
	// garbage) and the function didn't even receive kernel/stride/padding.
	struct {
		OaU32 BatchSize;
		OaU32 Channels;
		OaU32 InHeight;
		OaU32 InWidth;
		OaU32 OutHeight;
		OaU32 OutWidth;
		OaU32 KernelSize;
		OaU32 Stride;
		OaU32 Padding;
	} push{
		static_cast<OaU32>(N), static_cast<OaU32>(C),
		static_cast<OaU32>(H), static_cast<OaU32>(W),
		static_cast<OaU32>(H_out), static_cast<OaU32>(W_out),
		static_cast<OaU32>(InKernelSize), static_cast<OaU32>(InStride),
		static_cast<OaU32>(InPadding)
	};

	OaU32 grid_x = DivCeil(static_cast<OaU32>(H), 16);
	OaU32 grid_y = DivCeil(static_cast<OaU32>(W), 16);
	OaU32 grid_z = static_cast<OaU32>(N * C);
	// Shader declares exactly 3 buffer indices {grad_in, grad_out, indices}. Bind
	// only those in that order. The old code bound 5 {InX, InOut, indices, grad_out,
	// grad_in}: grad_in mapped to InX (so it WROTE the gradient into the read-only
	// forward input) and grad_out mapped to InOut (so it read the upstream gradient
	// from the forward output) — gradInput was never written.
	OaBufferAccess access[] = {
		OaBufferAccess::Write, OaBufferAccess::Read, OaBufferAccess::Read
	};
	ctx.Add("MaxPool2dBwd", {&gradInput, &InGradOutput, &InIndices},
		access, &push, sizeof(push), grid_x, grid_y, grid_z);

	return gradInput;
}

// ─── AvgPool2dBwd ────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::AvgPool2dBwd(const OaMatrix& InX, const OaMatrix& InGradOutput,
                                   OaI32 InKernelSize, OaI32 InStride, OaI32 InPadding) {
	auto& ctx = OaContext::GetDefault();
	assert(InX.Rank() == 4 && "AvgPool2dBwd requires 4D input [N, C, H, W]");
	assert(InGradOutput.Rank() == 4 && "AvgPool2dBwd requires 4D grad_output");

	OaI64 N = InX.Size(0);
	OaI64 C = InX.Size(1);
	OaI64 H = InX.Size(2);
	OaI64 W = InX.Size(3);
	
	OaI64 H_out = InGradOutput.Size(2);
	OaI64 W_out = InGradOutput.Size(3);

	// Initialize to zero since we accumulate gradients
	OaMatrix gradInput = OaFnMatrix::Zeros(InX.GetShape(), InX.GetDtype());

	// Push constants match shader: buffer indices (auto-prepended) + params
	struct {
		OaU32 BatchSize;
		OaU32 Channels;
		OaU32 InHeight;
		OaU32 InWidth;
		OaU32 OutHeight;
		OaU32 OutWidth;
		OaU32 KernelSize;
		OaU32 Stride;
		OaU32 Padding;
	} push{
		static_cast<OaU32>(N), static_cast<OaU32>(C),
		static_cast<OaU32>(H), static_cast<OaU32>(W),
		static_cast<OaU32>(H_out), static_cast<OaU32>(W_out),
		static_cast<OaU32>(InKernelSize), static_cast<OaU32>(InStride), static_cast<OaU32>(InPadding)
	};

	// Grid covers input dimensions (where we write gradients)
	OaU32 grid_x = DivCeil(static_cast<OaU32>(H), 16);
	OaU32 grid_y = DivCeil(static_cast<OaU32>(W), 16);
	OaU32 grid_z = static_cast<OaU32>(N * C);
	
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write
	};
	ctx.Add("AvgPool2dBwd", {&InGradOutput, &gradInput}, access, &push, sizeof(push), grid_x, grid_y, grid_z);

	return gradInput;
}

// ─── SigmoidBwd ─────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::SigmoidBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SigmoidBwd", {&InForwardOutput, &InGradOutput, &gradInput},		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── LeakyReluBwd ────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::LeakyReluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput,
	OaF32 InAlpha) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	// Shader PushConstants tail is {count, alpha} (buffer indices auto-prepended).
	// Alpha must be threaded through or the negative-side gradient collapses to 0.
	struct { OaU32 Count; OaF32 Alpha; } push{n, InAlpha};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LeakyReluBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── EluBwd ─────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::EluBwd(const OaMatrix& InForwardOutput, const OaMatrix& InGradOutput,
	OaF32 InAlpha) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InForwardOutput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InForwardOutput.GetShape(), InForwardOutput.GetDtype());

	// Shader takes the forward OUTPUT y and alpha; for y<=0 the gradient factor is
	// (y + alpha) == alpha*exp(x). Alpha must be threaded through (push tail
	// {count, alpha}; buffer indices auto-prepended).
	struct { OaU32 Count; OaF32 Alpha; } push{n, InAlpha};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("EluBwd", {&InForwardOutput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── MishBwd ────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::MishBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MishBwd", {&InInput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── SiluMulBwd ─────────────────────────────────────────────────────────────

// Backward of SiluMul. Takes the forward INPUT (gate||up, the un-invertible
// SiLU(gate)*up output cannot recover gate/up) and the upstream grad; produces the
// input-shaped gradient. The shader uses only the first half of InGradOutput.
OaMatrix OaFnMatrix::SiluMulBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	const OaU32 intermediate = static_cast<OaU32>(InInput.Size(-1) / 2);
	struct { OaU32 Count; OaU32 IntermediateSize; } push{n, intermediate};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("SiluMulBwd", {&InInput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n / 2, 256));

	return gradInput;
}

// ─── GegluBwd ───────────────────────────────────────────────────────────────

// Backward of Geglu. Takes the forward INPUT (up||gate); up*GELU(gate) cannot be
// inverted to recover up/gate. Produces the input-shaped gradient; the shader uses
// only the first half of InGradOutput.
OaMatrix OaFnMatrix::GegluBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	const OaU32 intermediate = static_cast<OaU32>(InInput.Size(-1) / 2);
	struct { OaU32 Count; OaU32 IntermediateSize; } push{n, intermediate};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("GegluBwd", {&InInput, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n / 2, 256));

	return gradInput;
}

// ─── MaxBwd ─────────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::MaxBwd(const OaMatrix& InInput, const OaMatrix& InMaxValue,
	const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	OaU32 n = static_cast<OaU32>(InInput.NumElements());

	OaMatrix gradInput = OaFnMatrix::Empty(InInput.GetShape(), InInput.GetDtype());

	// Shader contract: buffers {input, max, grad_out, grad_in} (indices
	// auto-prepended) + push tail {count}. grad_in[i] = (X[i]==max) ? grad : 0.
	struct { OaU32 Count; } push{n};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read,
		OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("MaxBwd", {&InInput, &InMaxValue, &InGradOutput, &gradInput},
		access, &push, sizeof(push), DivCeil(n, 256));

	return gradInput;
}

// ─── CrossEntropyBwd ─────────────────────────────────────────────────────────
// REMOVED: This was a broken duplicate of OaFnLoss::CrossEntropyBwd.
// Bug: bound 4 buffers for a 3-index shader + wrong push struct (Count vs Batch/Classes/TargetDtype).
// The live autograd path uses OaFnLoss::CrossEntropyBwd (correct: 3 buffers, proper push).
// Python bindings also use OaFnLoss::CrossEntropyBwd.
// This function was never called in the codebase.

// ─── GatherBwd ──────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::GatherBwd(
	const OaMatrix& InIndices,
	const OaMatrix& InGradOutput,
	OaI32 InVocabSize,
	OaI32 InEmbedDim
) {
	auto& ctx = OaContext::GetDefault();
	OaU32 numIndices = static_cast<OaU32>(InIndices.NumElements());
	OaU32 indexDtype = (InIndices.GetDtype() == OaScalarType::UInt8) ? 0U : 1U;

	// Each workgroup owns one vocabulary row and fully writes every element, so
	// unlike an atomic scatter this deterministic implementation needs no clear.
	OaMatrix gradTable = OaFnMatrix::Empty(
		OaMatrixShape{InVocabSize, InEmbedDim}, InGradOutput.GetDtype());

	// NOTE: the framework auto-injects the bound buffers' heap indices as the leading
	// push-constant u32s (indices_idx, d_out_idx, d_table_idx) — see Gather forward,
	// whose C++ push omits them too. Declaring them here (and zeroing them) shifted every
	// subsequent field by 3 u32s, so the kernel read num_indices=0 / vocab_size=0 → its
	// scatter loop never ran → the embedding table received ZERO gradient. Only the
	// trailing scalar fields belong in the C++ push.
	struct {
		OaU32 VocabSize;
		OaU32 EmbedDim;
		OaU32 NumIndices;
		OaU32 IndexDtype;
	} push{
		.VocabSize = static_cast<OaU32>(InVocabSize),
		.EmbedDim = static_cast<OaU32>(InEmbedDim),
		.NumIndices = numIndices,
		.IndexDtype = indexDtype,
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,       // indices
		OaBufferAccess::Read,       // d_out (gathered gradient)
		OaBufferAccess::ReadWrite,  // d_table (scatter target)
	};
	// One workgroup per vocab row; 256 threads cooperate across embed_dim.
	ctx.Add("GatherBwd", {&InIndices, &InGradOutput, &gradTable},
		access, &push, sizeof(push), static_cast<OaU32>(InVocabSize));

	return gradTable;
}

// ─── LinearDataBwd ──────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::LinearDataBwd(const OaMatrix& InGradOutput, const OaMatrix& InWeight) {
	auto& ctx = OaContext::GetDefault();
	
	// d_input = d_output @ weight
	// InGradOutput: [batch, out_features]
	// InWeight: [out_features, in_features]
	// Result: [batch, in_features]
	
	OaU32 M = static_cast<OaU32>(InGradOutput.Size(0));  // batch
	OaU32 N = static_cast<OaU32>(InWeight.Size(0));      // out_features
	OaU32 K = static_cast<OaU32>(InWeight.Size(1));      // in_features
	
	OaMatrix gradInput = OaFnMatrix::Empty(OaMatrixShape{M, K}, InGradOutput.GetDtype());
	
	struct { OaU32 M; OaU32 N; OaU32 K; } push{.M = M, .N = N, .K = K};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LinearDataBwd", {&InGradOutput, &InWeight, &gradInput}, access,
		&push, sizeof(push), DivCeil(M, 32), DivCeil(K, 32), 1);
	
	return gradInput;
}

OaMatrix OaFnMatrix::LinearDataReluBwd(
	const OaMatrix& InGradOutput,
	const OaMatrix& InWeight,
	const OaMatrix& InActivation
) {
	auto& ctx = OaContext::GetDefault();

	// d_input = (d_output @ weight) * (activation > 0)
	// InGradOutput: [batch, out_features]
	// InWeight: [out_features, in_features]
	// InActivation: [batch, in_features]
	OaU32 M = static_cast<OaU32>(InGradOutput.Size(0));  // batch
	OaU32 N = static_cast<OaU32>(InWeight.Size(0));      // out_features
	OaU32 K = static_cast<OaU32>(InWeight.Size(1));      // in_features

	OaMatrix gradInput = OaFnMatrix::Empty(OaMatrixShape{M, K}, InGradOutput.GetDtype());

	struct {
		OaU32 M;
		OaU32 N;
		OaU32 K;
	} push{.M = M, .N = N, .K = K};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	ctx.Add("LinearDataReluBwd",
		{&InGradOutput, &InWeight, &InActivation, &gradInput},
		access, &push, sizeof(push), DivCeil(M * K, 256));

	return gradInput;
}

// ─── LinearReluBwdData — in-layer LinearRelu data backward ──────────────────

OaMatrix OaFnMatrix::LinearReluBwdData(
	const OaMatrix& InGradOutput,
	const OaMatrix& InWeight,
	const OaMatrix& InActivation
) {
	auto& ctx = OaContext::GetDefault();

	// For y = ReLU(x @ W^T + b):
	//   d_x[i,k] = sum_j (d_y[i,j] * (act[i,j] > 0)) * W[j,k]
	// Gate is applied INSIDE the inner sum so d_z is never materialized.
	// InGradOutput:  [batch, out_features]
	// InActivation:  [batch, out_features]  (post-ReLU forward output)
	// InWeight:      [out_features, in_features]
	OaU32 M = static_cast<OaU32>(InGradOutput.Size(0));  // batch
	OaU32 N = static_cast<OaU32>(InWeight.Size(0));      // out_features
	OaU32 K = static_cast<OaU32>(InWeight.Size(1));      // in_features

	OaMatrix gradInput = OaFnMatrix::Empty(OaMatrixShape{M, K}, InGradOutput.GetDtype());

	struct {
		OaU32 M;
		OaU32 N;
		OaU32 K;
	} push{.M = M, .N = N, .K = K};

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	ctx.Add("LinearReluBwdData",
		{&InGradOutput, &InActivation, &InWeight, &gradInput},
		access, &push, sizeof(push), DivCeil(M * K, 256));

	return gradInput;
}

// ─── LinearWeightBwd ────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::LinearWeightBwd(const OaMatrix& InInput, const OaMatrix& InGradOutput) {
	auto& ctx = OaContext::GetDefault();
	
	// d_weight = input^T @ d_output
	// InInput: [batch, in_features]
	// InGradOutput: [batch, out_features]
	// Result: [out_features, in_features]
	
	OaU32 M = static_cast<OaU32>(InInput.Size(0));       // batch
	OaU32 K = static_cast<OaU32>(InInput.Size(1));       // in_features
	OaU32 N = static_cast<OaU32>(InGradOutput.Size(1));  // out_features
	
	OaMatrix gradWeight = OaFnMatrix::Empty(OaMatrixShape{N, K}, InInput.GetDtype());
	
	struct { OaU32 M; OaU32 N; OaU32 K; } push{.M = M, .N = N, .K = K};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("LinearWeightBwd", {&InGradOutput, &InInput, &gradWeight}, access, &push, sizeof(push), DivCeil(N * K, 256));
	
	return gradWeight;
}

OaFnMatrix::OaLinearWeightBiasBwdResult OaFnMatrix::LinearWeightBiasBwd(
	const OaMatrix& InInput,
	const OaMatrix& InGradOutput
) {
	auto& ctx = OaContext::GetDefault();

	// d_weight = input^T @ d_output, d_bias = sum(d_output, dim=0)
	// InInput: [batch, in_features]
	// InGradOutput: [batch, out_features]
	// GradWeight: [out_features, in_features], GradBias: [out_features]

	OaU32 M = static_cast<OaU32>(InInput.Size(0));       // batch
	OaU32 K = static_cast<OaU32>(InInput.Size(1));       // in_features
	OaU32 N = static_cast<OaU32>(InGradOutput.Size(1));  // out_features

	OaMatrix gradWeight = OaFnMatrix::Empty(OaMatrixShape{N, K}, InInput.GetDtype());
	OaMatrix gradBias = OaFnMatrix::Empty(OaMatrixShape{N}, InGradOutput.GetDtype());

	ctx.AddLinearBwdWeightBias(InInput, InGradOutput, gradWeight, gradBias, M, N, K);

	return {.GradWeight = gradWeight, .GradBias = gradBias};
}

// ─── UpsampleBwd ────────────────────────────────────────────────────────────

OaMatrix OaFnMatrix::UpsampleBwd(
	const OaMatrix& InInput, const OaMatrix& InDOut,
	OaI32 InScaleFactor, bool InIsBilinear) {
	auto shape = InInput.GetShape();
	OaU32 B = (OaU32)shape[0];
	OaU32 C = (OaU32)shape[1];
	OaU32 H = (OaU32)shape[2];
	OaU32 W = (OaU32)shape[3];
	OaU32 outH = H * InScaleFactor;
	OaU32 outW = W * InScaleFactor;

	// Initialize to zero since we accumulate gradients
	auto output = OaFnMatrix::Zeros(shape, InInput.GetDtype());

	const char* shaderName = InIsBilinear ? "ResizeBilinearBwd" : "ResizeNearestBwd";

	// Push constants: buffer indices auto-prepended by runtime, then our params
	struct Push {
		OaU32 BatchSize;
		OaU32 Channels;
		OaU32 HIn;
		OaU32 WIn;
		OaU32 HOut;
		OaU32 WOut;
	};
	Push push = {
		B, C, H, W, outH, outW
	};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaU32 gx = InIsBilinear ? DivCeil(outW, 16) : DivCeil(W, 16);
	OaU32 gy = InIsBilinear ? DivCeil(outH, 16) : DivCeil(H, 16);
	OaU32 gz = B * C;
	OaContext::GetDefault().Add(shaderName, {&InDOut, &output}, access, &push, sizeof(push), gx, gy, gz);

	return output;
}

// ─── BatchNorm2dBwd ─────────────────────────────────────────────────────────

OaFnMatrix::OaBatchNorm2dBwdResult OaFnMatrix::BatchNorm2dBwd(
	const OaMatrix& InX, const OaMatrix& InGamma, const OaMatrix& /*InBeta*/,
	const OaMatrix& InMean, const OaMatrix& InVar, const OaMatrix& /*InOut*/,
	const OaMatrix& InDOut, OaF32 InEps, bool /*InIsTraining*/) {
	auto shape = InX.GetShape();
	OaI64 N = shape[0];
	OaI64 C = shape[1];
	OaI64 H = shape[2];
	OaI64 W = shape[3];

	// inv_std = 1 / sqrt(var + eps)
	auto var_eps = OaFnMatrix::Add(
		InVar,
		OaFnMatrix::Full(InVar.GetShape(), InEps, InVar.GetDtype()));
	auto inv_std = OaFnMatrix::Div(
		OaFnMatrix::Ones(InVar.GetShape(), InVar.GetDtype()),
		OaFnMatrix::Sqrt(var_eps));

	// x_hat = (x - mean) * inv_std  (broadcasts automatically)
	auto x_centered = OaFnMatrix::Sub(InX, InMean);
	auto x_hat = OaFnMatrix::Mul(x_centered, inv_std);

	// d_beta = sum(d_out) over N,H,W → [1, C, 1, 1]
	auto sum_dout_z = OaFnMatrix::Sum(InDOut, 3);     // [N, C, H, 1]
	auto sum_dout_y = OaFnMatrix::Sum(sum_dout_z, 2); // [N, C, 1, 1]
	auto sum_dout_x = OaFnMatrix::Sum(sum_dout_y, 0); // [1, C, 1, 1]

	// d_gamma = sum(d_out * x_hat) over N,H,W → [1, C, 1, 1]
	auto d_out_x_hat = OaFnMatrix::Mul(InDOut, x_hat);
	auto sum_dxh_z = OaFnMatrix::Sum(d_out_x_hat, 3);
	auto sum_dxh_y = OaFnMatrix::Sum(sum_dxh_z, 2);
	auto sum_dxh_x = OaFnMatrix::Sum(sum_dxh_y, 0);

	// mean_dout = sum(d_out) / (N*H*W)
	OaF32 nhw = static_cast<OaF32>(N * H * W);
	auto mean_dout = OaFnMatrix::Scale(sum_dout_x, 1.0f / nhw);

	// mean_dout_xhat = sum(d_out * x_hat) / (N*H*W)
	auto mean_dout_xhat = OaFnMatrix::Scale(sum_dxh_x, 1.0f / nhw);

	// d_x = gamma * inv_std * (d_out - mean_dout - x_hat * mean_dout_xhat)
	auto term1 = OaFnMatrix::Sub(InDOut, mean_dout);            // broadcast
	auto term2 = OaFnMatrix::Mul(x_hat, mean_dout_xhat);        // broadcast
	auto term3 = OaFnMatrix::Sub(term1, term2);
	auto d_x = OaFnMatrix::Mul(
		OaFnMatrix::Mul(InGamma, inv_std), term3);              // broadcast

	return {
		.DX = d_x,
		.DGamma = sum_dxh_x.Reshape(OaMatrixShape{C}),
		.DBias = sum_dout_x.Reshape(OaMatrixShape{C})
	};
}
