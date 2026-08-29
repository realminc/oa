// oa::FnMatrix — backward pass implementations for neural network operations.
// Manual implementations for gradient computation kernels.

#include <oa/ml/fnMatrix.h>
#include <oa/ml/autograd/matrix/autogradActivation.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dnn/graphLowering.h>
#include <oa/core/validation.h>
#include <oa/core/fnmatrix/fnMatrixAxis.h>

#include <oa/core/std/assert.h>

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

static oa::Matrix commitMatrixResult(
	oa::Matrix inResult,
	oa::OpLoweringScope& inLowering,
	const oa::OpContract& inContract,
	oa::MatrixArgs inInputs,
	oa::OpAttributeArgs inAttributes = {})
{
	const auto status = inLowering.commit(
		inContract, inInputs, {&inResult}, inAttributes);
	return status.isOk() ? inResult : oa::Matrix{};
}

// ─── ReluBwd ────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::reluBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "ReluBwd", {&inForwardOutput, &inGradOutput, &gradInput}, access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::reluBwd,
		{&inForwardOutput, &inGradOutput});
}

// ─── TanhBwd ────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::tanhBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "TanhBwd", {&inForwardOutput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::tanhBwd,
		{&inForwardOutput, &inGradOutput});
}

// ─── GeluBwd ────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::geluBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inInput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "GeluBwd", {&inInput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::geluBwd,
		{&inInput, &inGradOutput});
}

// ─── SiluBwd ────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::siluBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inInput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SiluBwd", {&inInput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::siluBwd,
		{&inInput, &inGradOutput});
}

// ─── SoftplusBwd ─────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::softplusBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SoftplusBwd", {&inForwardOutput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::softplusBwd,
		{&inForwardOutput, &inGradOutput});
}

// ─── SoftmaxScaledMasked ─────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::softmaxScaledMasked(
	const oa::Matrix& inScores, const oa::Matrix& inMask, oa::F32 inScale) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 rows, cols;
	if (inScores.rank() == 2) {
		rows = static_cast<oa::U32>(inScores.size(0));
		cols = static_cast<oa::U32>(inScores.size(1));
	} else {
		rows = 1;
		cols = static_cast<oa::U32>(inScores.numElements());
	}

	oa::Matrix out = oa::FnMatrix::empty(inScores.getShape(), inScores.getDtype());

	struct { oa::U32 rows; oa::U32 cols; oa::F32 scale; } push{rows, cols, inScale};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write
	};
	const bool useNarrow = cols > 0U and cols <= 32U
		and not oa::EnvFlag::isSet("OA_DISABLE_NARROW_ROW_KERNELS");
	ctx.add(
		useNarrow ? "SoftmaxScaledMaskedN32" : "SoftmaxScaledMasked",
		{&inScores, &inMask, &out}, access, &push, sizeof(push), rows, 1, 1);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::softmaxScaledMasked,
		{&inScores, &inMask}, {&out},
		{oa::OpAttribute::fromFloat("scale", inScale)});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and inScores.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradSoftmaxScaledMasked>();
		gradFn->saveForBackward(out);
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inScores});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->outputShape_ = out.getShape();
		gradFn->scale_ = inScale;
		out.mutAutograd().gradFn = gradFn;
		if (not oa::FnAutograd::attachSemantic(
			gradFn, semantic.getValue()).isOk())
		{
			return {};
		}
	}

	return out;
}

oa::Matrix oa::FnMatrix::softmaxScaledMaskedBwd(
	const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput, oa::F32 inScale) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::U32 rows, cols;
	if (inForwardOutput.rank() == 2) {
		rows = static_cast<oa::U32>(inForwardOutput.size(0));
		cols = static_cast<oa::U32>(inForwardOutput.size(1));
	} else {
		rows = 1;
		cols = static_cast<oa::U32>(inForwardOutput.numElements());
	}

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	struct { oa::U32 rows; oa::U32 cols; oa::F32 scale; } push{rows, cols, inScale};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write
	};
	ctx.add( "SoftmaxScaledMaskedBwd", {&inForwardOutput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), rows, 1, 1);

	return commitMatrixResult(
		gradInput, lowering,
		oa::detail::opRegistry::FnMatrix::softmaxScaledMaskedBwd,
		{&inForwardOutput, &inGradOutput},
		{oa::OpAttribute::fromFloat("scale", inScale)});
}

// ─── SwigluBwd ─────────────────────────────────────────────────────────────

oa::SwigluBwdResult oa::FnMatrix::swigluBwd(
	const oa::Matrix& inGate, const oa::Matrix& inUp, const oa::Matrix& inOut,
	const oa::Matrix& inGradOutput
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inGate.numElements());

	oa::Matrix gateGrad = oa::FnMatrix::empty(inGate.getShape(), inGate.getDtype());
	oa::Matrix upGrad = oa::FnMatrix::empty(inUp.getShape(), inUp.getDtype());

	// shader declares exactly 5 buffer indices {gate, up, d_out, d_gate, d_up} and
	// recomputes silu(gate) itself — it never reads the forward output. binding inOut
	// (a 6th buffer) shifted every index: d_out read from inOut, d_gate written to
	// inGradOutput, d_up written to gateGrad, upGrad never written. (void)inOut.
	(void)inOut;
	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write
	};
	ctx.add( "SwigluBwd", {&inGate, &inUp, &inGradOutput, &gateGrad, &upGrad},
		access, &push, sizeof(push), divCeil(n, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::swigluBwd,
		{&inGate, &inUp, &inOut, &inGradOutput},
		{&gateGrad, &upGrad}).isOk())
	{
		return {};
	}
	return {.dGate = gateGrad, .dUp = upGrad};
}

// ─── LayerNormBwd ───────────────────────────────────────────────────────────

oa::LayerNormBwdResult oa::FnMatrix::layerNormBwd(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	const oa::Matrix& inOut, const oa::Matrix& inMean, const oa::Matrix& inRstd,
	const oa::Matrix& inGradOutput, oa::F32 inEps
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// normalize over the LAST dim (must match the forward): [B,T,C] → rows=B*T, cols=C.
	// The previous rows=size(0)/cols=size(1) wrote dWeight/dBias with the wrong stride
	// on rank-3 input → NaN gradients.
	const oa::I64 cols = inX.rank() >= 1 ? inX.size(inX.rank() - 1) : inX.numElements();
	const oa::I64 rows = cols > 0 ? inX.numElements() / cols : 1;

	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	// The shader writes dw_contrib PER ELEMENT at row*cols + i — i.e. a full
	// [rows, cols] per-row contribution buffer that the host must column-sum over
	// rows to get dWeight [cols]. binding a [cols]-sized buffer here (as the old
	// code did) overflowed the heap by (rows-1)*cols floats and returned only
	// row 0's contribution un-reduced → wrong dWeight + memory corruption for any
	// rows>1 (batch>1 or seq>1). rows==1 happened to fit exactly, hiding the bug.
	oa::Matrix dWcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, cols}, inX.getDtype());

	struct { oa::U32 rows; oa::U32 cols; oa::F32 eps; } push{
		static_cast<oa::U32>(rows), static_cast<oa::U32>(cols), inEps};
	// shader expects: x, w, dy, dx, dw_contrib (5 buffers)
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write
	};
	ctx.add( "LayerNormBwd",
		{&inX, &inWeight, &inGradOutput, &dX, &dWcontrib},
		access, &push, sizeof(push), static_cast<oa::U32>(rows));

	// dWeight = column-sum of the per-row contributions over rows → [cols].
	oa::Matrix dWeight = oa::FnMatrix::sum(dWcontrib, 0).reshape(inWeight.getShape());

	// dBias = sum of gradOutput over ALL rows (batch AND sequence). Flatten to
	// [rows, cols] first: sum(inGradOutput, 0) on a rank-3 [B,T,C] would only sum
	// over B and return [T,C], not [C].
	oa::Matrix dyRows = inGradOutput.reshape(oa::MatrixShape{rows, cols});
	oa::Matrix dBias = oa::FnMatrix::sum(dyRows, 0).reshape(inBias.getShape());

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::layerNormBwd,
		{&inX, &inWeight, &inBias, &inOut, &inMean, &inRstd, &inGradOutput},
		{&dX, &dWeight, &dBias},
		{oa::OpAttribute::fromFloat("epsilon", inEps)}).isOk())
	{
		return {};
	}
	return {.dX = dX, .dWeight = dWeight, .dBias = dBias};
}

// ─── RmsNormBwd ────────────────────────────────────────────────────────────

oa::RmsNormBwdResult oa::FnMatrix::rmsNormBwd(
	const oa::Matrix& inX, const oa::Matrix& inWeight,
	const oa::Matrix& inGradOutput, oa::F32 inEps
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// normalize over the LAST dim (must match the forward): [B,T,C] → rows=B*T, cols=C.
	// The previous rows=size(0)/cols=size(1) wrote dWeight/dBias with the wrong stride
	// on rank-3 input → NaN gradients.
	const oa::I64 cols = inX.rank() >= 1 ? inX.size(inX.rank() - 1) : inX.numElements();
	const oa::I64 rows = cols > 0 ? inX.numElements() / cols : 1;

	oa::Matrix dX = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());
	// Per-row dw contribution buffer [rows, cols], column-summed to dWeight [cols]
	// below. The old code bound a [cols] buffer directly (heap overflow for rows>1)
	// and returned only row 0's contribution un-reduced. See LayerNormBwd above.
	oa::Matrix dWcontrib = oa::FnMatrix::empty(oa::MatrixShape{rows, cols}, inX.getDtype());

	// The RmsNormBwd shader recomputes inv_rms = rsqrt(mean(x^2) + eps), so it needs
	// eps in the push tail. It must mirror the shader's {rows, cols, eps} tail and
	// use the exact forward epsilon retained by the autograd node.
	struct { oa::U32 rows; oa::U32 cols; oa::F32 eps; } push{
		static_cast<oa::U32>(rows), static_cast<oa::U32>(cols), inEps};
	// CRITICAL: the shader declares exactly 5 buffer indices
	// (x, w, dy, dx, dw_contrib). Buffer indices are auto-prepended IN ORDER, so the
	// bound set must be exactly those 5 — the shader recomputes inv_rms from x and
	// never reads `out`/`rstd`. The old code bound 7 buffers
	// {x, w, out, rstd, dy, dx, dw_contrib}; the extra out/rstd shifted everything,
	// so the shader read dy from `out`, wrote dx into `rstd`, and wrote dw_contrib
	// into the dy input → garbage, non-deterministic gradients.
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Write, oa::BufferAccess::Write
	};
	ctx.add( "RmsNormBwd",
		{&inX, &inWeight, &inGradOutput, &dX, &dWcontrib},
		access, &push, sizeof(push), static_cast<oa::U32>(rows));

	oa::Matrix dWeight = oa::FnMatrix::sum(dWcontrib, 0).reshape(inWeight.getShape());

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::rmsNormBwd,
		{&inX, &inWeight, &inGradOutput},
		{&dX, &dWeight},
		{oa::OpAttribute::fromFloat("epsilon", inEps)}).isOk())
	{
		return {};
	}
	return {.dX = dX, .dWeight = dWeight};
}

// ─── MaxPool2dBwd ────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::maxPool2dBwd(
	const oa::Matrix& inX, const oa::Matrix& inIndices, const oa::Matrix& inGradOutput,
	oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	OA_REQUIRE_MSG(inX.rank() == 4, "MaxPool2dBwd requires 4D input [N, C, H, W]");
	OA_REQUIRE_MSG(inGradOutput.rank() == 4, "MaxPool2dBwd requires 4D grad_output");

	oa::I64 N = inX.size(0);
	oa::I64 C = inX.size(1);
	oa::I64 H = inX.size(2);
	oa::I64 W = inX.size(3);
	oa::I64 H_out = inGradOutput.size(2);
	oa::I64 W_out = inGradOutput.size(3);

	oa::Matrix gradInput = oa::FnMatrix::empty(inX.getShape(), inX.getDtype());

	// Must mirror the shader's full PushConstants tail: the kernel walks each input
	// position's pooling window, so it needs outHeight/outWidth/KernelSize/Stride/
	// Padding. The old struct supplied only the first four fields (the rest read as
	// garbage) and the function didn't even receive kernel/stride/padding.
	struct {
		oa::U32 batchSize;
		oa::U32 channels;
		oa::U32 inHeight;
		oa::U32 inWidth;
		oa::U32 outHeight;
		oa::U32 outWidth;
		oa::U32 KernelSize;
		oa::U32 Stride;
		oa::U32 Padding;
	} push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(C),
		static_cast<oa::U32>(H), static_cast<oa::U32>(W),
		static_cast<oa::U32>(H_out), static_cast<oa::U32>(W_out),
		static_cast<oa::U32>(inKernelSize), static_cast<oa::U32>(inStride),
		static_cast<oa::U32>(inPadding)
	};

	oa::U32 grid_x = divCeil(static_cast<oa::U32>(H), 16);
	oa::U32 grid_y = divCeil(static_cast<oa::U32>(W), 16);
	oa::U32 grid_z = static_cast<oa::U32>(N * C);
	// shader declares exactly 3 buffer indices {grad_in, grad_out, indices}. Bind
	// only those in that order. The old code bound 5 {inX, inOut, indices, grad_out,
	// grad_in}: grad_in mapped to inX (so it WROTE the gradient into the read-only
	// forward input) and grad_out mapped to inOut (so it read the upstream gradient
	// from the forward output) — gradInput was never written.
	oa::BufferAccess access[] = {
		oa::BufferAccess::Write, oa::BufferAccess::Read, oa::BufferAccess::Read
	};
	ctx.add( "MaxPool2dBwd", {&gradInput, &inGradOutput, &inIndices},
		access, &push, sizeof(push), grid_x, grid_y, grid_z);

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::maxPool2dBwd,
		{&inX, &inIndices, &inGradOutput},
		{
			oa::OpAttribute::fromSignedInteger(
				"kernelSize", inKernelSize),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
		});
}

// ─── AvgPool2dBwd ────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::avgPool2dBwd(const oa::Matrix& inX, const oa::Matrix& inGradOutput,
                                   oa::I32 inKernelSize, oa::I32 inStride, oa::I32 inPadding) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	OA_REQUIRE_MSG(inX.rank() == 4, "AvgPool2dBwd requires 4D input [N, C, H, W]");
	OA_REQUIRE_MSG(inGradOutput.rank() == 4, "AvgPool2dBwd requires 4D grad_output");

	oa::I64 N = inX.size(0);
	oa::I64 C = inX.size(1);
	oa::I64 H = inX.size(2);
	oa::I64 W = inX.size(3);
	
	oa::I64 H_out = inGradOutput.size(2);
	oa::I64 W_out = inGradOutput.size(3);

	// initialize to zero since we accumulate gradients
	oa::Matrix gradInput = oa::FnMatrix::zeros(inX.getShape(), inX.getDtype());

	// Push constants match shader: buffer indices (auto-prepended) + params
	struct {
		oa::U32 batchSize;
		oa::U32 channels;
		oa::U32 inHeight;
		oa::U32 inWidth;
		oa::U32 outHeight;
		oa::U32 outWidth;
		oa::U32 KernelSize;
		oa::U32 Stride;
		oa::U32 Padding;
	} push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(C),
		static_cast<oa::U32>(H), static_cast<oa::U32>(W),
		static_cast<oa::U32>(H_out), static_cast<oa::U32>(W_out),
		static_cast<oa::U32>(inKernelSize), static_cast<oa::U32>(inStride), static_cast<oa::U32>(inPadding)
	};

	// grid covers input dimensions (where we write gradients)
	oa::U32 grid_x = divCeil(static_cast<oa::U32>(H), 16);
	oa::U32 grid_y = divCeil(static_cast<oa::U32>(W), 16);
	oa::U32 grid_z = static_cast<oa::U32>(N * C);
	
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Write
	};
	ctx.add( "AvgPool2dBwd", {&inGradOutput, &gradInput}, access, &push, sizeof(push), grid_x, grid_y, grid_z);

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::avgPool2dBwd,
		{&inX, &inGradOutput},
		{
			oa::OpAttribute::fromSignedInteger(
				"kernelSize", inKernelSize),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
		});
}

// ─── SigmoidBwd ─────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::sigmoidBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SigmoidBwd", {&inForwardOutput, &inGradOutput, &gradInput},		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::sigmoidBwd,
		{&inForwardOutput, &inGradOutput});
}

// ─── LeakyReluBwd ────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::leakyReluBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput,
	oa::F32 inAlpha) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	// shader PushConstants tail is {count, alpha} (buffer indices auto-prepended).
	// alpha must be threaded through or the negative-side gradient collapses to 0.
	struct { oa::U32 Count; oa::F32 alpha; } push{n, inAlpha};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "LeakyReluBwd", {&inForwardOutput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::leakyReluBwd,
		{&inForwardOutput, &inGradOutput},
		{oa::OpAttribute::fromFloat("alpha", inAlpha)});
}

// ─── EluBwd ─────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::eluBwd(const oa::Matrix& inForwardOutput, const oa::Matrix& inGradOutput,
	oa::F32 inAlpha) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inForwardOutput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inForwardOutput.getShape(), inForwardOutput.getDtype());

	// shader takes the forward OUTPUT y and alpha; for y<=0 the gradient factor is
	// (y + alpha) == alpha*exp(x). alpha must be threaded through (push tail
	// {count, alpha}; buffer indices auto-prepended).
	struct { oa::U32 Count; oa::F32 alpha; } push{n, inAlpha};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "EluBwd", {&inForwardOutput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::eluBwd,
		{&inForwardOutput, &inGradOutput},
		{oa::OpAttribute::fromFloat("alpha", inAlpha)});
}

// ─── MishBwd ────────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::mishBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inInput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	struct { oa::U32 Count; } push{n};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "MishBwd", {&inInput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::mishBwd,
		{&inInput, &inGradOutput});
}

// ─── SiluMulBwd ─────────────────────────────────────────────────────────────

// backward of SiluMul. Takes the forward INPUT (gate||up, the un-invertible
// siLU(gate)*up output cannot recover gate/up) and the upstream grad; produces the
// input-shaped gradient. The shader uses only the first half of inGradOutput.
oa::Matrix oa::FnMatrix::siluMulBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inInput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	const oa::U32 intermediate = static_cast<oa::U32>(inInput.size(-1) / 2);
	struct { oa::U32 Count; oa::U32 intermediateSize; } push{n, intermediate};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "SiluMulBwd", {&inInput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n / 2, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::siluMulBwd,
		{&inInput, &inGradOutput});
}

// ─── GegluBwd ───────────────────────────────────────────────────────────────

// backward of Geglu. Takes the forward INPUT (up||gate); up*GELU(gate) cannot be
// inverted to recover up/gate. Produces the input-shaped gradient; the shader uses
// only the first half of inGradOutput.
oa::Matrix oa::FnMatrix::gegluBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::U32 n = static_cast<oa::U32>(inInput.numElements());

	oa::Matrix gradInput = oa::FnMatrix::empty(inInput.getShape(), inInput.getDtype());

	const oa::U32 intermediate = static_cast<oa::U32>(inInput.size(-1) / 2);
	struct { oa::U32 Count; oa::U32 intermediateSize; } push{n, intermediate};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "GegluBwd", {&inInput, &inGradOutput, &gradInput},
		access, &push, sizeof(push), divCeil(n / 2, 256));

	return commitMatrixResult(
		gradInput, lowering, oa::detail::opRegistry::FnMatrix::gegluBwd,
		{&inInput, &inGradOutput});
}

// ─── CrossEntropyBwd ─────────────────────────────────────────────────────────
// REMOVED: This was a broken duplicate of oa::FnLoss::CrossEntropyBwd.
// Bug: bound 4 buffers for a 3-index shader + wrong push struct (Count vs Batch/classes/targetDtype).
// The live autograd path uses oa::FnLoss::crossEntropyBwd (correct: 3 buffers, proper push).
// Python bindings also use oa::FnLoss::CrossEntropyBwd.
// This function was never called in the codebase.

// ─── LinearDataBwd ──────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::linearDataBwd(const oa::Matrix& inGradOutput, const oa::Matrix& inWeight) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// d_input = d_output @ weight
	// inGradOutput: [batch, out_features]
	// inWeight: [out_features, in_features]
	// result: [batch, in_features]
	
	oa::U32 M = static_cast<oa::U32>(inGradOutput.size(0));  // batch
	oa::U32 N = static_cast<oa::U32>(inWeight.size(0));      // out_features
	oa::U32 K = static_cast<oa::U32>(inWeight.size(1));      // in_features
	
	oa::Matrix gradInput = oa::FnMatrix::empty(oa::MatrixShape{M, K}, inGradOutput.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearDataBwd,
		{&inGradOutput, &inWeight}, {&gradInput});
	if (not semantic.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearDataBwd semantic recording failed: {}",
			semantic.getStatus().getMessage().cStr());
		return {};
	}
	
	struct { oa::U32 M; oa::U32 N; oa::U32 K; } push{.M = M, .N = N, .K = K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "LinearDataBwd", {&inGradOutput, &inWeight, &gradInput}, access,
		&push, sizeof(push), divCeil(M, 32), divCeil(K, 32), 1,
		oa::detail::opRegistry::FnMatrix::linearDataBwd.name, 0,
		oa::detail::opRegistry::FnMatrix::linearDataBwd.hash, 0, 0, semantic.getValue());
	
	return gradInput;
}

oa::Matrix oa::FnMatrix::linearDataReluBwd(
	const oa::Matrix& inGradOutput,
	const oa::Matrix& inWeight,
	const oa::Matrix& inActivation
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// d_input = (d_output @ weight) * (activation > 0)
	// inGradOutput: [batch, out_features]
	// inWeight: [out_features, in_features]
	// inActivation: [batch, in_features]
	oa::U32 M = static_cast<oa::U32>(inGradOutput.size(0));  // batch
	oa::U32 N = static_cast<oa::U32>(inWeight.size(0));      // out_features
	oa::U32 K = static_cast<oa::U32>(inWeight.size(1));      // in_features

	oa::Matrix gradInput = oa::FnMatrix::empty(oa::MatrixShape{M, K}, inGradOutput.getDtype());

	struct {
		oa::U32 M;
		oa::U32 N;
		oa::U32 K;
	} push{.M = M, .N = N, .K = K};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add( "LinearDataReluBwd",
		{&inGradOutput, &inWeight, &inActivation, &gradInput},
		access, &push, sizeof(push), divCeil(M * K, 256));

	return commitMatrixResult(
		gradInput, lowering,
		oa::detail::opRegistry::FnMatrix::linearDataReluBwd,
		{&inGradOutput, &inWeight, &inActivation});
}

// ─── LinearReluBwdData — in-layer LinearRelu data backward ──────────────────

oa::Matrix oa::FnMatrix::linearReluBwdData(
	const oa::Matrix& inGradOutput,
	const oa::Matrix& inWeight,
	const oa::Matrix& inActivation
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// For y = reLU(x @ W^T + b):
	//   d_x[i,k] = sum_j (d_y[i,j] * (act[i,j] > 0)) * W[j,k]
	// gate is applied INSIDE the inner sum so d_z is never materialized.
	// inGradOutput:  [batch, out_features]
	// inActivation:  [batch, out_features]  (post-ReLU forward output)
	// inWeight:      [out_features, in_features]
	oa::U32 M = static_cast<oa::U32>(inGradOutput.size(0));  // batch
	oa::U32 N = static_cast<oa::U32>(inWeight.size(0));      // out_features
	oa::U32 K = static_cast<oa::U32>(inWeight.size(1));      // in_features

	oa::Matrix gradInput = oa::FnMatrix::empty(oa::MatrixShape{M, K}, inGradOutput.getDtype());

	struct {
		oa::U32 M;
		oa::U32 N;
		oa::U32 K;
	} push{.M = M, .N = N, .K = K};

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	ctx.add( "LinearReluBwdData",
		{&inGradOutput, &inActivation, &inWeight, &gradInput},
		access, &push, sizeof(push), divCeil(M * K, 256));

	return commitMatrixResult(
		gradInput, lowering,
		oa::detail::opRegistry::FnMatrix::linearReluBwdData,
		{&inGradOutput, &inWeight, &inActivation});
}

// ─── LinearWeightBwd ────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::linearWeightBwd(const oa::Matrix& inInput, const oa::Matrix& inGradOutput) {
	auto& ctx = oa::ExecutionSession::getActive();
	
	// d_weight = input^T @ d_output
	// inInput: [batch, in_features]
	// inGradOutput: [batch, out_features]
	// result: [out_features, in_features]
	
	oa::U32 M = static_cast<oa::U32>(inInput.size(0));       // batch
	oa::U32 K = static_cast<oa::U32>(inInput.size(1));       // in_features
	oa::U32 N = static_cast<oa::U32>(inGradOutput.size(1));  // out_features
	
	oa::Matrix gradWeight = oa::FnMatrix::empty(oa::MatrixShape{N, K}, inInput.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearWeightBwd,
		{&inInput, &inGradOutput}, {&gradWeight});
	if (not semantic.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearWeightBwd semantic recording failed: {}",
			semantic.getStatus().getMessage().cStr());
		return {};
	}
	
	struct { oa::U32 M; oa::U32 N; oa::U32 K; } push{.M = M, .N = N, .K = K};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "LinearWeightBwd", {&inGradOutput, &inInput, &gradWeight}, access,
		&push, sizeof(push), divCeil(N * K, 256), 1, 1,
		oa::detail::opRegistry::FnMatrix::linearWeightBwd.name, 0,
		oa::detail::opRegistry::FnMatrix::linearWeightBwd.hash, 0, 0, semantic.getValue());
	
	return gradWeight;
}

oa::LinearWeightBiasBwdResult oa::FnMatrix::linearWeightBiasBwd(
	const oa::Matrix& inInput,
	const oa::Matrix& inGradOutput
) {
	auto& ctx = oa::ExecutionSession::getActive();

	// d_weight = input^T @ d_output, d_bias = sum(d_output, dim=0)
	// inInput: [batch, in_features]
	// inGradOutput: [batch, out_features]
	// gradWeight: [out_features, in_features], gradBias: [out_features]

	oa::U32 K = static_cast<oa::U32>(inInput.size(1));       // in_features
	oa::U32 N = static_cast<oa::U32>(inGradOutput.size(1));  // out_features

	oa::Matrix gradWeight = oa::FnMatrix::empty(oa::MatrixShape{N, K}, inInput.getDtype());
	oa::Matrix gradBias = oa::FnMatrix::empty(oa::MatrixShape{N}, inGradOutput.getDtype());

	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd,
		{&inInput, &inGradOutput}, {&gradWeight, &gradBias});
	if (not semantic.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearWeightBiasBwd semantic recording failed: {}",
			semantic.getStatus().getMessage().cStr());
		return {};
	}
	const auto lowering = oa::DnnGraphLowering::recordLinearWeightBiasBackward(
		ctx, {
			.input = &inInput,
			.gradOutput = &inGradOutput,
			.gradWeight = &gradWeight,
			.gradBias = &gradBias,
			.operation = oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.name,
			.opContractHash = oa::detail::opRegistry::FnMatrix::linearWeightBiasBwd.hash,
			.semanticOp = semantic.getValue(),
		});
	if (not lowering.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearWeightBiasBwd lowering failed: {}",
			lowering.getMessage().cStr());
		return {};
	}

	return {.gradWeight = gradWeight, .gradBias = gradBias};
}

// ─── UpsampleBwd ────────────────────────────────────────────────────────────

oa::Matrix oa::FnMatrix::upsampleBwd(
	const oa::Matrix& inInput, const oa::Matrix& inDOut,
	oa::I32 inScaleFactor, bool inIsBilinear) {
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto shape = inInput.getShape();
	oa::U32 B = (oa::U32)shape[0];
	oa::U32 C = (oa::U32)shape[1];
	oa::U32 H = (oa::U32)shape[2];
	oa::U32 W = (oa::U32)shape[3];
	oa::U32 outH = H * inScaleFactor;
	oa::U32 outW = W * inScaleFactor;

	// initialize to zero since we accumulate gradients
	auto output = oa::FnMatrix::zeros(shape, inInput.getDtype());

	const char* shaderName = inIsBilinear ? "ResizeBilinearBwd" : "ResizeNearestBwd";

	// Push constants: buffer indices auto-prepended by runtime, then our params
	struct Push {
		oa::U32 batchSize;
		oa::U32 channels;
		oa::U32 hIn;
		oa::U32 wIn;
		oa::U32 hOut;
		oa::U32 wOut;
	};
	Push push = {
		B, C, H, W, outH, outW
	};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::U32 gx = inIsBilinear ? divCeil(outW, 16) : divCeil(W, 16);
	oa::U32 gy = inIsBilinear ? divCeil(outH, 16) : divCeil(H, 16);
	oa::U32 gz = B * C;
	context.add( shaderName, {&inDOut, &output}, access, &push, sizeof(push), gx, gy, gz);

	return commitMatrixResult(
		output, lowering, oa::detail::opRegistry::FnMatrix::upsampleBwd,
		{&inInput, &inDOut},
		{
			oa::OpAttribute::fromSignedInteger(
				"scaleFactor", inScaleFactor),
			oa::OpAttribute::fromBoolean(
				"bilinear", inIsBilinear),
		});
}

// ─── BatchNorm2dBwd ─────────────────────────────────────────────────────────

oa::BatchNorm2dBwdResult oa::FnMatrix::batchNorm2dBwd(
	const oa::Matrix& inX, const oa::Matrix& inGamma, const oa::Matrix& inBeta,
	const oa::Matrix& inMean, const oa::Matrix& inVar, const oa::Matrix& inOut,
	const oa::Matrix& inDOut, oa::F32 inEps, bool inIsTraining) {
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto shape = inX.getShape();
	oa::I64 N = shape[0];
	oa::I64 C = shape[1];
	oa::I64 H = shape[2];
	oa::I64 W = shape[3];

	// inv_std = 1 / sqrt(var + eps)
	auto var_eps = oa::FnMatrix::add(
		inVar,
		oa::FnMatrix::full(inVar.getShape(), inEps, inVar.getDtype()));
	auto inv_std = oa::FnMatrix::div(
		oa::FnMatrix::ones(inVar.getShape(), inVar.getDtype()),
		oa::FnMatrix::sqrt(var_eps));

	// x_hat = (x - mean) * inv_std  (broadcasts automatically)
	auto x_centered = oa::FnMatrix::sub(inX, inMean);
	auto x_hat = oa::FnMatrix::mul(x_centered, inv_std);

	// d_beta = sum(d_out) over N,H,W → [1, C, 1, 1]
	auto sum_dout_z = oa::FnMatrix::sum(inDOut, 3);     // [N, C, H, 1]
	auto sum_dout_y = oa::FnMatrix::sum(sum_dout_z, 2); // [N, C, 1, 1]
	auto sum_dout_x = oa::FnMatrix::sum(sum_dout_y, 0); // [1, C, 1, 1]

	// d_gamma = sum(d_out * x_hat) over N,H,W → [1, C, 1, 1]
	auto d_out_x_hat = oa::FnMatrix::mul(inDOut, x_hat);
	auto sum_dxh_z = oa::FnMatrix::sum(d_out_x_hat, 3);
	auto sum_dxh_y = oa::FnMatrix::sum(sum_dxh_z, 2);
	auto sum_dxh_x = oa::FnMatrix::sum(sum_dxh_y, 0);

	// mean_dout = sum(d_out) / (N*H*W)
	oa::F32 nhw = static_cast<oa::F32>(N * H * W);
	auto mean_dout = oa::FnMatrix::scale(sum_dout_x, 1.0f / nhw);

	// mean_dout_xhat = sum(d_out * x_hat) / (N*H*W)
	auto mean_dout_xhat = oa::FnMatrix::scale(sum_dxh_x, 1.0f / nhw);

	// d_x = gamma * inv_std * (d_out - mean_dout - x_hat * mean_dout_xhat)
	auto term1 = oa::FnMatrix::sub(inDOut, mean_dout);            // broadcast
	auto term2 = oa::FnMatrix::mul(x_hat, mean_dout_xhat);        // broadcast
	auto term3 = oa::FnMatrix::sub(term1, term2);
	auto d_x = oa::FnMatrix::mul(
		oa::FnMatrix::mul(inGamma, inv_std), term3);              // broadcast

	oa::BatchNorm2dBwdResult result{
		.dx = d_x,
		.dGamma = sum_dxh_x.reshape(oa::MatrixShape{C}),
		.dBias = sum_dout_x.reshape(oa::MatrixShape{C})
	};
	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::batchNorm2dBwd,
		{&inX, &inGamma, &inBeta, &inMean, &inVar, &inOut, &inDOut},
		{&result.dx, &result.dGamma, &result.dBias},
		{
			oa::OpAttribute::fromFloat("epsilon", inEps),
			oa::OpAttribute::fromBoolean("training", inIsTraining),
		}).isOk())
	{
		return {};
	}
	return result;
}
