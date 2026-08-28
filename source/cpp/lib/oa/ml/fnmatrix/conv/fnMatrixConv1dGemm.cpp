// FnMatrixConv1dGemm — run a 1-D convolution as a single tensor-core matmul.
//
// The direct Conv1d kernel is a scalar nested loop (inC*K MACs per output
// element, no tensor cores). For the oa::Alm tokenizer's wide convs
// (inC=outC=512, K=3) that path is compute-bound on the scalar ALU. Reshaping
// the conv as im2col → GEMM routes the heavy work through the CmSg/CmWg
// (bf16) or gemmTiled (fp32) matmul stack, which the GEMM router picks by the
// active engine precision.
//
//   x    [N, inC, L]
//   cols = im2Col1d(x)                 [N*outL, inC*K]     (differentiable)
//   w    [outC, inC, K] -> [outC, inC*K]                   (view)
//   y    = Linear(cols, wReshaped, b)  [N*outL, outC]      (tensor-core GEMM)
//   out  = y -> [N, outL, outC] -> transpose(1,2)          [N, outC, outL]
//
// The only new differentiable primitive is im2Col1d (backward = Col2Im1d); the
// GEMM gradient comes free from Linear, and reshape/transpose are differentiable.

#include <oa/ml/fnMatrix.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/ml/autograd.h>
#include <oa/ml/autograd/matrix/autogradConv.h>

#include <assert.h>

namespace {

// outL = (L + 2P - D*(K-1) - 1) / S + 1
oa::I32 conv1dOutLen(oa::I32 L, oa::I32 K, oa::I32 S, oa::I32 P, oa::I32 D) {
	return (((L + (2 * P)) - (D * (K - 1)) - 1) / S) + 1;
}

}  // namespace

oa::Matrix oa::FnMatrix::im2Col1d(
	const oa::Matrix& inX, oa::I32 inK, oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.rank() == 3 && "Im2Col1d input must be 3D [N, inC, L]");

	const oa::I32 N = static_cast<oa::I32>(inX.size(0));
	const oa::I32 inC = static_cast<oa::I32>(inX.size(1));
	const oa::I32 L = static_cast<oa::I32>(inX.size(2));
	const oa::I32 outL = conv1dOutLen(L, inK, inStride, inPadding, inDilation);

	const oa::I64 rows = static_cast<oa::I64>(N) * outL;
	const oa::I64 innerK = static_cast<oa::I64>(inC) * inK;
	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{rows, innerK}, inX.getDtype());

	struct {
		oa::U32 N; oa::U32 inC; oa::U32 L; oa::U32 K; oa::U32 S; oa::U32 P; oa::U32 D; oa::U32 outL;
	} push{
		static_cast<oa::U32>(N), static_cast<oa::U32>(inC), static_cast<oa::U32>(L),
		static_cast<oa::U32>(inK), static_cast<oa::U32>(inStride),
		static_cast<oa::U32>(inPadding), static_cast<oa::U32>(inDilation),
		static_cast<oa::U32>(outL)};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Im2Col1d", {&inX, &out}, access, &push, sizeof(push),
		oa::divCeil(static_cast<oa::U32>(rows * innerK), 256));

	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::im2Col1d, {&inX}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("kernel", inK),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
			oa::OpAttribute::fromSignedInteger("dilation", inDilation),
		});
	if (not semantic.isOk()) return {};

	if (oa::FnAutograd::isEnabled() and inX.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradIm2Col1d>();
		gradFn->n_ = N; gradFn->inC_ = inC; gradFn->l_ = L; gradFn->k_ = inK;
		gradFn->s_ = inStride; gradFn->p_ = inPadding; gradFn->d_ = inDilation;
		gradFn->outL_ = outL;
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inX});
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

oa::Matrix oa::FnMatrix::col2Im1d(
	const oa::Matrix& inDCols, oa::I32 inN, oa::I32 inC, oa::I32 inL, oa::I32 inK,
	oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation, oa::I32 inOutL)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);

	oa::Matrix dX = oa::FnMatrix::empty(oa::MatrixShape{inN, inC, inL}, inDCols.getDtype());

	struct {
		oa::U32 N; oa::U32 inC; oa::U32 L; oa::U32 K; oa::U32 S; oa::U32 P; oa::U32 D; oa::U32 outL;
	} push{
		static_cast<oa::U32>(inN), static_cast<oa::U32>(inC), static_cast<oa::U32>(inL),
		static_cast<oa::U32>(inK), static_cast<oa::U32>(inStride),
		static_cast<oa::U32>(inPadding), static_cast<oa::U32>(inDilation),
		static_cast<oa::U32>(inOutL)};

	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Col2Im1d", {&inDCols, &dX}, access, &push, sizeof(push),
		oa::divCeil(static_cast<oa::U32>(inN) * static_cast<oa::U32>(inC) * static_cast<oa::U32>(inL), 256));

	if (not lowering.commit(
		oa::detail::opRegistry::FnMatrix::col2Im1d, {&inDCols}, {&dX},
		{
			oa::OpAttribute::fromSignedInteger("batch", inN),
			oa::OpAttribute::fromSignedInteger("channels", inC),
			oa::OpAttribute::fromSignedInteger("inputLength", inL),
			oa::OpAttribute::fromSignedInteger("kernel", inK),
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
			oa::OpAttribute::fromSignedInteger("dilation", inDilation),
			oa::OpAttribute::fromSignedInteger("outputLength", inOutL),
		}).isOk())
	{
		return {};
	}
	return dX;
}

oa::Matrix oa::FnMatrix::conv1dGemm(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	assert(inX.rank() == 3 && "Conv1dGemm input must be 3D [N, inC, L]");
	assert(inWeight.rank() == 3 && "Conv1dGemm weight must be 3D [outC, inC, K]");

	const oa::I32 N = static_cast<oa::I32>(inX.size(0));
	const oa::I32 inC = static_cast<oa::I32>(inX.size(1));
	const oa::I32 outC = static_cast<oa::I32>(inWeight.size(0));
	const oa::I32 K = static_cast<oa::I32>(inWeight.size(2));
	const oa::I32 L = static_cast<oa::I32>(inX.size(2));
	const oa::I32 outL = conv1dOutLen(L, K, inStride, inPadding, inDilation);

	// Use the differentiable primitives: matMulNt (the autograd GEMM — Linear's
	// ctx.addLinear path is forward-only for non-parameter weights) + BiasAdd,
	// and the oa::FnMatrix reshape/transpose (the oa::Matrix:: methods are view-only
	// and would sever the backward graph).
	auto cols = oa::FnMatrix::im2Col1d(inX, K, inStride, inPadding, inDilation);  // [N*outL, inC*K]
	auto wFlat = oa::FnMatrix::reshape(inWeight, oa::MatrixShape{outC, static_cast<oa::I64>(inC) * K});
	auto y = oa::FnMatrix::matMulNt(cols, wFlat);                                  // [N*outL, outC]
	if (not inBias.isEmpty()) {
		// Differentiable broadcast add ([M,outC] + [1,outC]); oa::FnMatrix::biasAdd
		// is a forward-only inference kernel and would sever the backward graph.
		y = oa::FnMatrix::add(y, oa::FnMatrix::reshape(inBias, oa::MatrixShape{1, outC}));
	}
	auto y3 = oa::FnMatrix::reshape(y, oa::MatrixShape{N, outL, outC});
	auto out = oa::FnMatrix::transpose(y3, 1, 2);                                  // [N, outC, outL]
	const oa::Matrix* bias = inBias.isEmpty() ? nullptr : &inBias;
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::conv1dGemm,
		{&inX, &inWeight, bias}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
			oa::OpAttribute::fromSignedInteger("dilation", inDilation),
		});
	if (not semantic.isOk()) return {};
	if (auto grad = out.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return out;
}

oa::Matrix oa::FnMatrix::conv1dReluGemm(
	const oa::Matrix& inX, const oa::Matrix& inWeight, const oa::Matrix& inBias,
	oa::I32 inStride, oa::I32 inPadding, oa::I32 inDilation)
{
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 N = static_cast<oa::I32>(inX.size(0));
	const oa::I32 inC = static_cast<oa::I32>(inX.size(1));
	const oa::I32 outC = static_cast<oa::I32>(inWeight.size(0));
	const oa::I32 K = static_cast<oa::I32>(inWeight.size(2));
	const oa::I32 L = static_cast<oa::I32>(inX.size(2));
	const oa::I32 outL = conv1dOutLen(L, K, inStride, inPadding, inDilation);

	auto cols = oa::FnMatrix::im2Col1d(inX, K, inStride, inPadding, inDilation);
	auto wFlat = oa::FnMatrix::reshape(inWeight, oa::MatrixShape{outC, static_cast<oa::I64>(inC) * K});
	auto y = oa::FnMatrix::matMulNt(cols, wFlat);
	if (not inBias.isEmpty()) {
		y = oa::FnMatrix::add(y, oa::FnMatrix::reshape(inBias, oa::MatrixShape{1, outC}));
	}
	y = oa::FnMatrix::relu(y);
	auto y3 = oa::FnMatrix::reshape(y, oa::MatrixShape{N, outL, outC});
	auto out = oa::FnMatrix::transpose(y3, 1, 2);
	const oa::Matrix* bias = inBias.isEmpty() ? nullptr : &inBias;
	const auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnMatrix::conv1dReluGemm,
		{&inX, &inWeight, bias}, {&out},
		{
			oa::OpAttribute::fromSignedInteger("stride", inStride),
			oa::OpAttribute::fromSignedInteger("padding", inPadding),
			oa::OpAttribute::fromSignedInteger("dilation", inDilation),
		});
	if (not semantic.isOk()) return {};
	if (auto grad = out.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return out;
}
