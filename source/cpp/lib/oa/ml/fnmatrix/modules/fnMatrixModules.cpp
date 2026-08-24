// oa::FnMatrix — Neural Network layer operations
//
// BiasAdd, Conv1d, Conv2d implementations for oa::Matrix.
// These are stateless functions that delegate to vulkan compute kernels.

#include <oa/ml/fnMatrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrix.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include "../../autograd/autogradAttach.gen.h"

#include <cassert>

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

// Conv1d (scalar direct convolution) retired — oa::Conv1d and all callers use the
// im2col + GEMM path (oa::FnMatrix::conv1dGemm). Conv1dBwdData/Conv1dBwdWeight below
// survive: they back oa::ConvTranspose1d.

// Conv2d — 2D Convolution
oa::Matrix oa::FnMatrix::conv2d(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inGroups
) {
	auto& ctx = oa::ExecutionSession::getActive();

	// input: [N, inC, H, W], weight: [outC, inC, K, K], Bias: [outC]
	assert(inX.rank() == 4 && "Conv2d input must be 4D [N, inC, H, W]");
	assert(inWeight.rank() == 4 && "Conv2d weight must be 4D [outC, inC, K, K]");
	assert(inBias.rank() == 1 && "Conv2d bias must be 1D [outC]");

	oa::U32 N = static_cast<oa::U32>(inX.size(0));
	oa::U32 inC = static_cast<oa::U32>(inX.size(1));
	oa::U32 H = static_cast<oa::U32>(inX.size(2));
	oa::U32 W = static_cast<oa::U32>(inX.size(3));
	oa::U32 outC = static_cast<oa::U32>(inWeight.size(0));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 S = inStride;
	oa::U32 P = inPadding;
	assert(inGroups > 0 && inC % inGroups == 0 && outC % inGroups == 0);
	assert(static_cast<oa::U32>(inWeight.size(1)) == inC / inGroups);

	// output dimensions: (H + 2*P - K) / S + 1, (W + 2*P - K) / S + 1
	oa::U32 outH = (((H + (2 * P)) - K) / S) + 1;
	oa::U32 outW = (((W + (2 * P)) - K) / S) + 1;
	if (outH == 0 || outW == 0) {
		return oa::Matrix();
	}

	// allocate output: [N, outC, outH, outW]
	auto out = empty(oa::MatrixShape{N, outC, outH, outW}, inX.getDtype());
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::conv2d,
		{&inX, &inWeight, &inBias}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromUnsignedInteger("groups", inGroups),
		});
	if (not semantic.isOk()) return {};

	// Dispatch Conv2d kernel (stream.cpp prepends buffer indices)
	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC;
		oa::U32 H; oa::U32 W; oa::U32 K; oa::U32 S; oa::U32 P;
		oa::U32 outH; oa::U32 outW; oa::U32 Groups;
	} push{N, inC, outC, H, W, K, S, P, outH, outW, inGroups};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Conv2d", {&inX, &inWeight, &inBias, &out}, access, &push, sizeof(push),
		divCeil(N * outC * outH * outW, 256), 1, 1,
		oa::detail::opRegistry::FnMatrix::conv2d.name, 0,
		oa::detail::opRegistry::FnMatrix::conv2d.hash, 0, 0,
		semantic.getValue());

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::conv2d(
		out, inX, inWeight, inBias, inStride, inPadding, inGroups,
		semantic.getValue());
	if (not attached.isOk()) return {};
	return out;
}

// Conv2dBwdData — backward pass for 2D convolution (input gradient)
oa::Matrix oa::FnMatrix::conv2dBwdData(
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	const oa::MatrixShape& inInputShape,
	oa::U32 inGroups
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// d_out: [N, outC, outH, outW], weight: [outC, inC, K, K]
	// d_input: [N, inC, H, W]
	assert(inDOut.rank() == 4 && "Conv2dBwdData d_out must be 4D [N, outC, outH, outW]");
	assert(inWeight.rank() == 4 && "Conv2dBwdData weight must be 4D [outC, inC, K, K]");
	assert(inInputShape.rank == 4 && "Conv2dBwdData input_shape must be 4D [N, inC, H, W]");

	oa::U32 N = static_cast<oa::U32>(inDOut.size(0));
	oa::U32 outC = static_cast<oa::U32>(inDOut.size(1));
	oa::U32 outH = static_cast<oa::U32>(inDOut.size(2));
	oa::U32 outW = static_cast<oa::U32>(inDOut.size(3));
	oa::U32 inC = static_cast<oa::U32>(inInputShape[1]);
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 H = static_cast<oa::U32>(inInputShape[2]);
	oa::U32 W = static_cast<oa::U32>(inInputShape[3]);
	oa::U32 S = inStride;
	oa::U32 P = inPadding;
	assert(inGroups > 0 && inC % inGroups == 0 && outC % inGroups == 0);
	assert(static_cast<oa::U32>(inWeight.size(1)) == inC / inGroups);

	// allocate input gradient: [N, inC, H, W]
	auto d_input = empty(inInputShape, inDOut.getDtype());

	// Dispatch Conv2dBwdData kernel
	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC;
		oa::U32 H; oa::U32 W; oa::U32 K; oa::U32 S; oa::U32 P;
		oa::U32 outH; oa::U32 outW; oa::U32 Groups;
	} push{N, inC, outC, H, W, K, S, P, outH, outW, inGroups};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Conv2dBwdData", {&inDOut, &inWeight, &d_input}, access, &push, sizeof(push),
		divCeil(N * inC * H * W, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::conv2dBwdData,
		{&inDOut, &inWeight}, {&d_input},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromShape("inputShape", inInputShape),
			oa::OpAttribute::fromUnsignedInteger("groups", inGroups),
		}).isOk())
	{
		return {};
	}
	return d_input;
}

// oa::Conv2dBwdWeightResult is declared in <oa/ml/fnMatrix.h>.
oa::Conv2dBwdWeightResult oa::FnMatrix::conv2dBwdWeight(
	const oa::Matrix& inInput,
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inGroups
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// input: [N, inC, H, W], d_out: [N, outC, outH, outW]
	// d_weight: [outC, inC, K, K], d_bias: [outC]
	assert(inInput.rank() == 4 && "Conv2dBwdWeight input must be 4D [N, inC, H, W]");
	assert(inDOut.rank() == 4 && "Conv2dBwdWeight d_out must be 4D [N, outC, outH, outW]");
	assert(inWeight.rank() == 4 && "Conv2dBwdWeight weight must be 4D [outC, inC, K, K]");

	oa::U32 N = static_cast<oa::U32>(inInput.size(0));
	oa::U32 inC = static_cast<oa::U32>(inInput.size(1));
	oa::U32 H = static_cast<oa::U32>(inInput.size(2));
	oa::U32 W = static_cast<oa::U32>(inInput.size(3));
	oa::U32 outC = static_cast<oa::U32>(inDOut.size(1));
	oa::U32 outH = static_cast<oa::U32>(inDOut.size(2));
	oa::U32 outW = static_cast<oa::U32>(inDOut.size(3));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 S = inStride;
	oa::U32 P = inPadding;
	oa::U32 weightInC = static_cast<oa::U32>(inWeight.size(1));
	assert(inGroups > 0 && inC % inGroups == 0 && outC % inGroups == 0);
	assert(weightInC == inC / inGroups);

	// allocate gradients
	auto d_weight = empty(inWeight.getShape(), inDOut.getDtype());
	auto d_bias = empty(oa::MatrixShape{outC}, inDOut.getDtype());

	// Dispatch Conv2dBwdWeight kernel
	oa::U32 weightCount = outC * weightInC * K * K;
	oa::U32 total = weightCount + outC;

	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC;
		oa::U32 H; oa::U32 W; oa::U32 K; oa::U32 S; oa::U32 P;
		oa::U32 outH; oa::U32 outW;
		oa::U32 total; oa::U32 Groups;
	} push{N, inC, outC, H, W, K, S, P, outH, outW, total, inGroups};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Conv2dBwdWeight", {&inInput, &inDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		divCeil(total, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::conv2dBwdWeight,
		{&inInput, &inDOut, &inWeight}, {&d_weight, &d_bias},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromUnsignedInteger("groups", inGroups),
		}).isOk())
	{
		return {};
	}
	return {d_weight, d_bias};
}

// ConvTranspose2d — 2D Transposed Convolution
oa::Matrix oa::FnMatrix::convTranspose2d(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias,
	oa::U32 inStride,
	oa::U32 inPadding
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// input: [N, inC, H, W], weight: [inC, outC, K, K], Bias: [outC]
	assert(inX.rank() == 4 && "ConvTranspose2d input must be 4D [N, inC, H, W]");
	assert(inWeight.rank() == 4 && "ConvTranspose2d weight must be 4D [inC, outC, K, K]");
	assert(inBias.rank() == 1 && "ConvTranspose2d bias must be 1D [outC]");

	oa::U32 N = static_cast<oa::U32>(inX.size(0));
	oa::U32 H = static_cast<oa::U32>(inX.size(2));
	oa::U32 W = static_cast<oa::U32>(inX.size(3));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 outC = static_cast<oa::U32>(inWeight.size(1));
	oa::U32 S = inStride;
	oa::U32 P = inPadding;

	// output dimensions: (H - 1) * S - 2P + K
	oa::U32 outH = ((H - 1) * S) - (2 * P) + K;
	oa::U32 outW = ((W - 1) * S) - (2 * P) + K;
	assert(outH > 0 && outW > 0 && "ConvTranspose2d output dimensions must be positive");

	// Transposed convolution forward is the adjoint of Conv2d backward-data.
	// Reuse Conv2dBwdData: it maps [N, outC_conv, outH, outW] -> [N, inC_conv, H, W]
	// with weight [outC_conv, inC_conv, K, K]. We map:
	//   outC_conv = inC, inC_conv = outC, outH=H, outW=W, H_out=outH, W_out=outW.
	const oa::MatrixShape outShape = oa::MatrixShape{N, outC, outH, outW};
	auto out = conv2dBwdData(inX, inWeight, S, P, outShape);
	out = biasAdd(out, inBias);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::convTranspose2d,
		{&inX, &inWeight, &inBias}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
		});
	if (not semantic.isOk()) return {};
	const auto attached =
		oa::detail::generatedAutogradAttach::FnMatrix::convTranspose2d(
			out, inX, inWeight, inBias, inStride, inPadding,
			semantic.getValue());
	if (not attached.isOk()) return {};
	return out;
}

// ConvTranspose2dBwdData — backward pass for 2D transposed convolution (input gradient)
oa::Matrix oa::FnMatrix::convTranspose2dBwdData(
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	const oa::MatrixShape& inInputShape
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// d_out: [N, outC, outH, outW], weight: [inC, outC, K, K]
	// d_input: [N, inC, H, W]
	// Adjoint of transposed convolution is regular convolution with the same weight.
	// Conv2d interprets weight as [outC, inC, K, K]; our weight is [inC, outC, K, K],
	// which matches the adjoint conv's layout (outC_adj=inC, inC_adj=outC).
	auto zeroBias = zeros(oa::MatrixShape{inInputShape[1]}, inDOut.getDtype());
	auto out = conv2d(inDOut, inWeight, zeroBias, inStride, inPadding);
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::convTranspose2dBwdData,
		{&inDOut, &inWeight}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromShape("inputShape", inInputShape),
		});
	if (not semantic.isOk()) return {};
	const auto attached =
		oa::detail::generatedAutogradAttach::FnMatrix::convTranspose2dBwdData(
			out, inDOut, inWeight, inStride, inPadding,
			semantic.getValue());
	if (not attached.isOk()) return {};
	return out;
}

// oa::ConvTranspose2dBwdWeightResult is declared in <oa/ml/fnMatrix.h>.
oa::ConvTranspose2dBwdWeightResult oa::FnMatrix::convTranspose2dBwdWeight(
	const oa::Matrix& inInput,
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// input: [N, inC, H, W], d_out: [N, outC, outH, outW]
	// d_weight: [inC, outC, K, K], d_bias: [outC]
	assert(inInput.rank() == 4 && "ConvTranspose2dBwdWeight input must be 4D [N, inC, H, W]");
	assert(inDOut.rank() == 4 && "ConvTranspose2dBwdWeight d_out must be 4D [N, outC, outH, outW]");
	assert(inWeight.rank() == 4 && "ConvTranspose2dBwdWeight weight must be 4D [inC, outC, K, K]");

	oa::U32 N = static_cast<oa::U32>(inInput.size(0));
	oa::U32 inC = static_cast<oa::U32>(inInput.size(1));
	oa::U32 H = static_cast<oa::U32>(inInput.size(2));
	oa::U32 W = static_cast<oa::U32>(inInput.size(3));
	oa::U32 outC = static_cast<oa::U32>(inDOut.size(1));
	oa::U32 outH = static_cast<oa::U32>(inDOut.size(2));
	oa::U32 outW = static_cast<oa::U32>(inDOut.size(3));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 S = inStride;
	oa::U32 P = inPadding;

	auto d_weight = empty(inWeight.getShape(), inDOut.getDtype());
	auto d_bias = empty(oa::MatrixShape{outC}, inDOut.getDtype());

	oa::U32 weightCount = inC * outC * K * K;
	oa::U32 total = weightCount + outC;

	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC;
		oa::U32 H; oa::U32 W; oa::U32 K; oa::U32 S; oa::U32 P;
		oa::U32 outH; oa::U32 outW;
		oa::U32 total;
	} push{N, inC, outC, H, W, K, S, P, outH, outW, total};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "ConvTranspose2dBwdWeight", {&inInput, &inDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		divCeil(total, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::convTranspose2dBwdWeight,
		{&inInput, &inDOut, &inWeight}, {&d_weight, &d_bias},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
		}).isOk())
	{
		return {};
	}
	return oa::ConvTranspose2dBwdWeightResult{.gradWeight = d_weight, .gradBias = d_bias};
}

// Conv1dBwdData — backward pass for 1D convolution (input gradient)
oa::Matrix oa::FnMatrix::conv1dBwdData(
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inDilation,
	const oa::MatrixShape& inInputShape
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// d_out: [N, outC, outL], weight: [outC, inC, K]
	// d_input: [N, inC, L]
	assert(inDOut.rank() == 3 && "Conv1dBwdData d_out must be 3D [N, outC, outL]");
	assert(inWeight.rank() == 3 && "Conv1dBwdData weight must be 3D [outC, inC, K]");
	assert(inInputShape.rank == 3 && "Conv1dBwdData input_shape must be 3D [N, inC, L]");

	oa::U32 N = static_cast<oa::U32>(inDOut.size(0));
	oa::U32 outC = static_cast<oa::U32>(inDOut.size(1));
	oa::U32 outL = static_cast<oa::U32>(inDOut.size(2));
	oa::U32 inC = static_cast<oa::U32>(inWeight.size(1));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 L = static_cast<oa::U32>(inInputShape[2]);
	oa::U32 S = inStride;
	oa::U32 P = inPadding;
	oa::U32 D = inDilation;

	// allocate input gradient: [N, inC, L]
	auto d_input = empty(inInputShape, inDOut.getDtype());

	// Dispatch Conv1dBwdData kernel
	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC; oa::U32 L; oa::U32 K; oa::U32 S; oa::U32 P; oa::U32 D; oa::U32 outL;
	} push{N, inC, outC, L, K, S, P, D, outL};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Conv1dBwdData", {&inDOut, &inWeight, &d_input}, access, &push, sizeof(push),
		divCeil(N * inC * L, 256));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::conv1dBwdData,
		{&inDOut, &inWeight}, {&d_input},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromUnsignedInteger("dilation", inDilation),
			oa::OpAttribute::fromShape("inputShape", inInputShape),
		});
	if (not semantic.isOk()) return {};
	const auto attached =
		oa::detail::generatedAutogradAttach::FnMatrix::conv1dBwdData(
			d_input, inDOut, inWeight, inStride, inPadding, inDilation,
			semantic.getValue());
	if (not attached.isOk()) return {};
	return d_input;
}

// oa::Conv1dBwdWeightResult is declared in <oa/ml/fnMatrix.h>.
oa::Conv1dBwdWeightResult oa::FnMatrix::conv1dBwdWeight(
	const oa::Matrix& inInput,
	const oa::Matrix& inDOut,
	const oa::Matrix& inWeight,
	oa::U32 inStride,
	oa::U32 inPadding,
	oa::U32 inDilation
) {
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	// input: [N, inC, L], d_out: [N, outC, outL]
	// d_weight: [outC, inC, K], d_bias: [outC]
	assert(inInput.rank() == 3 && "Conv1dBwdWeight input must be 3D [N, inC, L]");
	assert(inDOut.rank() == 3 && "Conv1dBwdWeight d_out must be 3D [N, outC, outL]");
	assert(inWeight.rank() == 3 && "Conv1dBwdWeight weight must be 3D [outC, inC, K]");

	oa::U32 N = static_cast<oa::U32>(inInput.size(0));
	oa::U32 inC = static_cast<oa::U32>(inInput.size(1));
	oa::U32 L = static_cast<oa::U32>(inInput.size(2));
	oa::U32 outC = static_cast<oa::U32>(inDOut.size(1));
	oa::U32 outL = static_cast<oa::U32>(inDOut.size(2));
	oa::U32 K = static_cast<oa::U32>(inWeight.size(2));
	oa::U32 S = inStride;
	oa::U32 P = inPadding;
	oa::U32 D = inDilation;

	// allocate gradients
	auto d_weight = empty(inWeight.getShape(), inDOut.getDtype());
	auto d_bias = empty(oa::MatrixShape{outC}, inDOut.getDtype());

	// Dispatch Conv1dBwdWeight kernel
	oa::U32 weightCount = outC * inC * K;
	oa::U32 total = weightCount + outC;

	struct {
		oa::U32 N; oa::U32 inC; oa::U32 outC; oa::U32 L; oa::U32 K; oa::U32 S; oa::U32 P; oa::U32 D; oa::U32 outL; oa::U32 total;
	} push{N, inC, outC, L, K, S, P, D, outL, total};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Write};
	ctx.add( "Conv1dBwdWeight", {&inInput, &inDOut, &d_weight, &d_bias}, access, &push, sizeof(push),
		divCeil(total, 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::conv1dBwdWeight,
		{&inInput, &inDOut, &inWeight}, {&d_weight, &d_bias},
		{
			oa::OpAttribute::fromUnsignedInteger("stride", inStride),
			oa::OpAttribute::fromUnsignedInteger("padding", inPadding),
			oa::OpAttribute::fromUnsignedInteger("dilation", inDilation),
		}).isOk())
	{
		return {};
	}
	return {d_weight, d_bias};
}
