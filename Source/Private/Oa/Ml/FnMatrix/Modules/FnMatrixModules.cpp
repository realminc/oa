// OaFnMatrix — Neural Network Layer Operations
//
// BiasAdd, Conv1d, Conv2d implementations for OaMatrix.
// These are stateless functions that delegate to Vulkan compute kernels.

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>

#include <cassert>

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

// Conv1d (scalar direct convolution) retired — OaConv1d and all callers use the
// im2col + GEMM path (OaFnMatrix::Conv1dGemm). Conv1dBwdData/Conv1dBwdWeight below
// survive: they back OaConvTranspose1d.

// Conv2d — 2D Convolution
OaMatrix OaFnMatrix::Conv2d(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InGroups
) {
	auto& ctx = OaContext::GetDefault();

	// Input: [N, InC, H, W], Weight: [OutC, InC, K, K], Bias: [OutC]
	assert(InX.Rank() == 4 && "Conv2d input must be 4D [N, InC, H, W]");
	assert(InWeight.Rank() == 4 && "Conv2d weight must be 4D [OutC, InC, K, K]");
	assert(InBias.Rank() == 1 && "Conv2d bias must be 1D [OutC]");

	OaU32 N = static_cast<OaU32>(InX.Size(0));
	OaU32 InC = static_cast<OaU32>(InX.Size(1));
	OaU32 H = static_cast<OaU32>(InX.Size(2));
	OaU32 W = static_cast<OaU32>(InX.Size(3));
	OaU32 OutC = static_cast<OaU32>(InWeight.Size(0));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 S = InStride;
	OaU32 P = InPadding;
	assert(InGroups > 0 && InC % InGroups == 0 && OutC % InGroups == 0);
	assert(static_cast<OaU32>(InWeight.Size(1)) == InC / InGroups);

	// Output dimensions: (H + 2*P - K) / S + 1, (W + 2*P - K) / S + 1
	OaU32 outH = (((H + (2 * P)) - K) / S) + 1;
	OaU32 outW = (((W + (2 * P)) - K) / S) + 1;
	if (outH == 0 || outW == 0) {
		return OaMatrix();
	}

	// Allocate output: [N, OutC, outH, outW]
	auto out = Empty(OaMatrixShape{N, OutC, outH, outW}, InX.GetDtype());

	// Dispatch Conv2d kernel (Stream.cpp prepends buffer indices)
	struct {
		OaU32 N; OaU32 InC; OaU32 OutC;
		OaU32 H; OaU32 W; OaU32 K; OaU32 S; OaU32 P;
		OaU32 OutH; OaU32 OutW; OaU32 Groups;
	} push{N, InC, OutC, H, W, K, S, P, outH, outW, InGroups};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Conv2d", {&InX, &InWeight, &InBias, &out}, access, &push, sizeof(push),
		DivCeil(N * OutC * outH * outW, 256));


	return out;
}

// Conv2dBwdData — Backward pass for 2D convolution (input gradient)
OaMatrix OaFnMatrix::Conv2dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	const OaMatrixShape& InInputShape,
	OaU32 InGroups
) {
	auto& ctx = OaContext::GetDefault();

	// d_out: [N, OutC, OutH, OutW], Weight: [OutC, InC, K, K]
	// d_input: [N, InC, H, W]
	assert(InDOut.Rank() == 4 && "Conv2dBwdData d_out must be 4D [N, OutC, OutH, OutW]");
	assert(InWeight.Rank() == 4 && "Conv2dBwdData weight must be 4D [OutC, InC, K, K]");
	assert(InInputShape.Rank == 4 && "Conv2dBwdData input_shape must be 4D [N, InC, H, W]");

	OaU32 N = static_cast<OaU32>(InDOut.Size(0));
	OaU32 OutC = static_cast<OaU32>(InDOut.Size(1));
	OaU32 OutH = static_cast<OaU32>(InDOut.Size(2));
	OaU32 OutW = static_cast<OaU32>(InDOut.Size(3));
	OaU32 InC = static_cast<OaU32>(InInputShape[1]);
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 H = static_cast<OaU32>(InInputShape[2]);
	OaU32 W = static_cast<OaU32>(InInputShape[3]);
	OaU32 S = InStride;
	OaU32 P = InPadding;
	assert(InGroups > 0 && InC % InGroups == 0 && OutC % InGroups == 0);
	assert(static_cast<OaU32>(InWeight.Size(1)) == InC / InGroups);

	// Allocate input gradient: [N, InC, H, W]
	auto d_input = Empty(InInputShape, InDOut.GetDtype());

	// Dispatch Conv2dBwdData kernel
	struct {
		OaU32 N; OaU32 InC; OaU32 OutC;
		OaU32 H; OaU32 W; OaU32 K; OaU32 S; OaU32 P;
		OaU32 OutH; OaU32 OutW; OaU32 Groups;
	} push{N, InC, OutC, H, W, K, S, P, OutH, OutW, InGroups};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Conv2dBwdData", {&InDOut, &InWeight, &d_input}, access, &push, sizeof(push),
		DivCeil(N * InC * H * W, 256));

	return d_input;
}

// OaFnMatrix::OaConv2dBwdWeightResult is declared in <Oa/Ml/FnMatrix.h>.
OaFnMatrix::OaConv2dBwdWeightResult OaFnMatrix::Conv2dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InGroups
) {
	auto& ctx = OaContext::GetDefault();

	// input: [N, InC, H, W], d_out: [N, OutC, OutH, OutW]
	// d_weight: [OutC, InC, K, K], d_bias: [OutC]
	assert(InInput.Rank() == 4 && "Conv2dBwdWeight input must be 4D [N, InC, H, W]");
	assert(InDOut.Rank() == 4 && "Conv2dBwdWeight d_out must be 4D [N, OutC, OutH, OutW]");
	assert(InWeight.Rank() == 4 && "Conv2dBwdWeight weight must be 4D [OutC, InC, K, K]");

	OaU32 N = static_cast<OaU32>(InInput.Size(0));
	OaU32 InC = static_cast<OaU32>(InInput.Size(1));
	OaU32 H = static_cast<OaU32>(InInput.Size(2));
	OaU32 W = static_cast<OaU32>(InInput.Size(3));
	OaU32 OutC = static_cast<OaU32>(InDOut.Size(1));
	OaU32 OutH = static_cast<OaU32>(InDOut.Size(2));
	OaU32 OutW = static_cast<OaU32>(InDOut.Size(3));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 S = InStride;
	OaU32 P = InPadding;
	OaU32 weightInC = static_cast<OaU32>(InWeight.Size(1));
	assert(InGroups > 0 && InC % InGroups == 0 && OutC % InGroups == 0);
	assert(weightInC == InC / InGroups);

	// Allocate gradients
	auto d_weight = Empty(InWeight.GetShape(), InDOut.GetDtype());
	auto d_bias = Empty(OaMatrixShape{OutC}, InDOut.GetDtype());

	// Dispatch Conv2dBwdWeight kernel
	OaU32 weightCount = OutC * weightInC * K * K;
	OaU32 total = weightCount + OutC;

	struct {
		OaU32 N; OaU32 InC; OaU32 OutC;
		OaU32 H; OaU32 W; OaU32 K; OaU32 S; OaU32 P;
		OaU32 OutH; OaU32 OutW;
		OaU32 Total; OaU32 Groups;
	} push{N, InC, OutC, H, W, K, S, P, OutH, OutW, total, InGroups};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
	ctx.Add("Conv2dBwdWeight", {&InInput, &InDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		DivCeil(total, 256));

	return {d_weight, d_bias};
}

// ConvTranspose2d — 2D Transposed Convolution
OaMatrix OaFnMatrix::ConvTranspose2d(
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaU32 InStride,
	OaU32 InPadding
) {
	// Input: [N, InC, H, W], Weight: [InC, OutC, K, K], Bias: [OutC]
	assert(InX.Rank() == 4 && "ConvTranspose2d input must be 4D [N, InC, H, W]");
	assert(InWeight.Rank() == 4 && "ConvTranspose2d weight must be 4D [InC, OutC, K, K]");
	assert(InBias.Rank() == 1 && "ConvTranspose2d bias must be 1D [OutC]");

	OaU32 N = static_cast<OaU32>(InX.Size(0));
	OaU32 H = static_cast<OaU32>(InX.Size(2));
	OaU32 W = static_cast<OaU32>(InX.Size(3));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 OutC = static_cast<OaU32>(InWeight.Size(1));
	OaU32 S = InStride;
	OaU32 P = InPadding;

	// Output dimensions: (H - 1) * S - 2P + K
	OaU32 outH = ((H - 1) * S) - (2 * P) + K;
	OaU32 outW = ((W - 1) * S) - (2 * P) + K;
	assert(outH > 0 && outW > 0 && "ConvTranspose2d output dimensions must be positive");

	// Transposed convolution forward is the adjoint of Conv2d backward-data.
	// Reuse Conv2dBwdData: it maps [N, OutC_conv, OutH, OutW] -> [N, InC_conv, H, W]
	// with weight [OutC_conv, InC_conv, K, K]. We map:
	//   OutC_conv = InC, InC_conv = OutC, OutH=H, OutW=W, H_out=outH, W_out=outW.
	const OaMatrixShape outShape = OaMatrixShape{N, OutC, outH, outW};
	auto out = Conv2dBwdData(InX, InWeight, S, P, outShape);
	return BiasAdd(out, InBias);
}

// ConvTranspose2dBwdData — Backward pass for 2D transposed convolution (input gradient)
OaMatrix OaFnMatrix::ConvTranspose2dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	const OaMatrixShape& InInputShape
) {
	// d_out: [N, OutC, OutH, OutW], Weight: [InC, OutC, K, K]
	// d_input: [N, InC, H, W]
	// Adjoint of transposed convolution is regular convolution with the same weight.
	// Conv2d interprets weight as [OutC, InC, K, K]; our weight is [InC, OutC, K, K],
	// which matches the adjoint conv's layout (OutC_adj=InC, InC_adj=OutC).
	auto zeroBias = Zeros(OaMatrixShape{InInputShape[1]}, InDOut.GetDtype());
	return Conv2d(InDOut, InWeight, zeroBias, InStride, InPadding);
}

// OaFnMatrix::OaConvTranspose2dBwdWeightResult is declared in <Oa/Ml/FnMatrix.h>.
OaFnMatrix::OaConvTranspose2dBwdWeightResult OaFnMatrix::ConvTranspose2dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding
) {
	auto& ctx = OaContext::GetDefault();

	// input: [N, InC, H, W], d_out: [N, OutC, OutH, OutW]
	// d_weight: [InC, OutC, K, K], d_bias: [OutC]
	assert(InInput.Rank() == 4 && "ConvTranspose2dBwdWeight input must be 4D [N, InC, H, W]");
	assert(InDOut.Rank() == 4 && "ConvTranspose2dBwdWeight d_out must be 4D [N, OutC, OutH, OutW]");
	assert(InWeight.Rank() == 4 && "ConvTranspose2dBwdWeight weight must be 4D [InC, OutC, K, K]");

	OaU32 N = static_cast<OaU32>(InInput.Size(0));
	OaU32 InC = static_cast<OaU32>(InInput.Size(1));
	OaU32 H = static_cast<OaU32>(InInput.Size(2));
	OaU32 W = static_cast<OaU32>(InInput.Size(3));
	OaU32 OutC = static_cast<OaU32>(InDOut.Size(1));
	OaU32 OutH = static_cast<OaU32>(InDOut.Size(2));
	OaU32 OutW = static_cast<OaU32>(InDOut.Size(3));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 S = InStride;
	OaU32 P = InPadding;

	auto d_weight = Empty(InWeight.GetShape(), InDOut.GetDtype());
	auto d_bias = Empty(OaMatrixShape{OutC}, InDOut.GetDtype());

	OaU32 weightCount = InC * OutC * K * K;
	OaU32 total = weightCount + OutC;

	struct {
		OaU32 N; OaU32 InC; OaU32 OutC;
		OaU32 H; OaU32 W; OaU32 K; OaU32 S; OaU32 P;
		OaU32 OutH; OaU32 OutW;
		OaU32 Total;
	} push{N, InC, OutC, H, W, K, S, P, OutH, OutW, total};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
	ctx.Add("ConvTranspose2dBwdWeight", {&InInput, &InDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		DivCeil(total, 256));

	return OaConvTranspose2dBwdWeightResult{.GradWeight = d_weight, .GradBias = d_bias};
}

// Conv1dBwdData — Backward pass for 1D convolution (input gradient)
OaMatrix OaFnMatrix::Conv1dBwdData(
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InDilation,
	const OaMatrixShape& InInputShape
) {
	auto& ctx = OaContext::GetDefault();

	// d_out: [N, OutC, OutL], Weight: [OutC, InC, K]
	// d_input: [N, InC, L]
	assert(InDOut.Rank() == 3 && "Conv1dBwdData d_out must be 3D [N, OutC, OutL]");
	assert(InWeight.Rank() == 3 && "Conv1dBwdData weight must be 3D [OutC, InC, K]");
	assert(InInputShape.Rank == 3 && "Conv1dBwdData input_shape must be 3D [N, InC, L]");

	OaU32 N = static_cast<OaU32>(InDOut.Size(0));
	OaU32 OutC = static_cast<OaU32>(InDOut.Size(1));
	OaU32 OutL = static_cast<OaU32>(InDOut.Size(2));
	OaU32 InC = static_cast<OaU32>(InWeight.Size(1));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 L = static_cast<OaU32>(InInputShape[2]);
	OaU32 S = InStride;
	OaU32 P = InPadding;
	OaU32 D = InDilation;

	// Allocate input gradient: [N, InC, L]
	auto d_input = Empty(InInputShape, InDOut.GetDtype());

	// Dispatch Conv1dBwdData kernel
	struct {
		OaU32 N; OaU32 InC; OaU32 OutC; OaU32 L; OaU32 K; OaU32 S; OaU32 P; OaU32 D; OaU32 OutL;
	} push{N, InC, OutC, L, K, S, P, D, OutL};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Conv1dBwdData", {&InDOut, &InWeight, &d_input}, access, &push, sizeof(push),
		DivCeil(N * InC * L, 256));

	return d_input;
}

// OaFnMatrix::OaConv1dBwdWeightResult is declared in <Oa/Ml/FnMatrix.h>.
OaFnMatrix::OaConv1dBwdWeightResult OaFnMatrix::Conv1dBwdWeight(
	const OaMatrix& InInput,
	const OaMatrix& InDOut,
	const OaMatrix& InWeight,
	OaU32 InStride,
	OaU32 InPadding,
	OaU32 InDilation
) {
	auto& ctx = OaContext::GetDefault();

	// input: [N, InC, L], d_out: [N, OutC, OutL]
	// d_weight: [OutC, InC, K], d_bias: [OutC]
	assert(InInput.Rank() == 3 && "Conv1dBwdWeight input must be 3D [N, InC, L]");
	assert(InDOut.Rank() == 3 && "Conv1dBwdWeight d_out must be 3D [N, OutC, OutL]");
	assert(InWeight.Rank() == 3 && "Conv1dBwdWeight weight must be 3D [OutC, InC, K]");

	OaU32 N = static_cast<OaU32>(InInput.Size(0));
	OaU32 InC = static_cast<OaU32>(InInput.Size(1));
	OaU32 L = static_cast<OaU32>(InInput.Size(2));
	OaU32 OutC = static_cast<OaU32>(InDOut.Size(1));
	OaU32 OutL = static_cast<OaU32>(InDOut.Size(2));
	OaU32 K = static_cast<OaU32>(InWeight.Size(2));
	OaU32 S = InStride;
	OaU32 P = InPadding;
	OaU32 D = InDilation;

	// Allocate gradients
	auto d_weight = Empty(InWeight.GetShape(), InDOut.GetDtype());
	auto d_bias = Empty(OaMatrixShape{OutC}, InDOut.GetDtype());

	// Dispatch Conv1dBwdWeight kernel
	OaU32 weightCount = OutC * InC * K;
	OaU32 total = weightCount + OutC;

	struct {
		OaU32 N; OaU32 InC; OaU32 OutC; OaU32 L; OaU32 K; OaU32 S; OaU32 P; OaU32 D; OaU32 OutL; OaU32 Total;
	} push{N, InC, OutC, L, K, S, P, D, OutL, total};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Write};
	ctx.Add("Conv1dBwdWeight", {&InInput, &InDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		DivCeil(total, 256));

	return {d_weight, d_bias};
}
