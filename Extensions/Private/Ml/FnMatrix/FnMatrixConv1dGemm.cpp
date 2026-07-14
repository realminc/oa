// FnMatrixConv1dGemm — run a 1-D convolution as a single tensor-core matmul.
//
// The direct Conv1d kernel is a scalar nested loop (InC*K MACs per output
// element, no tensor cores). For the OaAlm tokenizer's wide convs
// (InC=OutC=512, K=3) that path is compute-bound on the scalar ALU. Reshaping
// the conv as im2col → GEMM routes the heavy work through the CmSg/CmWg
// (bf16) or GemmTiled (fp32) matmul stack, which the GEMM router picks by the
// active engine precision.
//
//   x    [N, InC, L]
//   cols = Im2Col1d(x)                 [N*OutL, InC*K]     (differentiable)
//   w    [OutC, InC, K] -> [OutC, InC*K]                   (view)
//   y    = Linear(cols, wReshaped, b)  [N*OutL, OutC]      (tensor-core GEMM)
//   out  = y -> [N, OutL, OutC] -> transpose(1,2)          [N, OutC, OutL]
//
// The only new differentiable primitive is Im2Col1d (backward = Col2Im1d); the
// GEMM gradient comes free from Linear, and reshape/transpose are differentiable.

#include <Oa/Ml/FnMatrix.h>

#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ml/Autograd.h>
#include <Ml/Autograd/Matrix/AutogradMatrix.h>

#include <cassert>

namespace {

// outL = (L + 2P - D*(K-1) - 1) / S + 1
OaI32 Conv1dOutLen(OaI32 L, OaI32 K, OaI32 S, OaI32 P, OaI32 D) {
	return (((L + (2 * P)) - (D * (K - 1)) - 1) / S) + 1;
}

}  // namespace

OaMatrix OaFnMatrix::Im2Col1d(
	const OaMatrix& InX, OaI32 InK, OaI32 InStride, OaI32 InPadding, OaI32 InDilation)
{
	auto& ctx = OaContext::GetDefault();
	assert(InX.Rank() == 3 && "Im2Col1d input must be 3D [N, InC, L]");

	const OaI32 N = static_cast<OaI32>(InX.Size(0));
	const OaI32 InC = static_cast<OaI32>(InX.Size(1));
	const OaI32 L = static_cast<OaI32>(InX.Size(2));
	const OaI32 outL = Conv1dOutLen(L, InK, InStride, InPadding, InDilation);

	const OaI64 rows = static_cast<OaI64>(N) * outL;
	const OaI64 innerK = static_cast<OaI64>(InC) * InK;
	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{rows, innerK}, InX.GetDtype());

	struct {
		OaU32 N; OaU32 InC; OaU32 L; OaU32 K; OaU32 S; OaU32 P; OaU32 D; OaU32 OutL;
	} push{
		static_cast<OaU32>(N), static_cast<OaU32>(InC), static_cast<OaU32>(L),
		static_cast<OaU32>(InK), static_cast<OaU32>(InStride),
		static_cast<OaU32>(InPadding), static_cast<OaU32>(InDilation),
		static_cast<OaU32>(outL)};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Im2Col1d", {&InX, &out}, access, &push, sizeof(push),
		OaDivCeil(static_cast<OaU32>(rows * innerK), 256));

	if (OaFnAutograd::IsEnabled() and InX.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradIm2Col1d>();
		gradFn->N_ = N; gradFn->InC_ = InC; gradFn->L_ = L; gradFn->K_ = InK;
		gradFn->S_ = InStride; gradFn->P_ = InPadding; gradFn->D_ = InDilation;
		gradFn->OutL_ = outL;
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InX});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->OutputShape_ = out.GetShape();
		out.MutAutograd().GradFn = gradFn;
		out.SetRequiresGrad(true);
	}

	return out;
}

OaMatrix OaFnMatrix::Col2Im1d(
	const OaMatrix& InDCols, OaI32 InN, OaI32 InC, OaI32 InL, OaI32 InK,
	OaI32 InStride, OaI32 InPadding, OaI32 InDilation, OaI32 InOutL)
{
	auto& ctx = OaContext::GetDefault();

	OaMatrix dX = OaFnMatrix::Empty(OaMatrixShape{InN, InC, InL}, InDCols.GetDtype());

	struct {
		OaU32 N; OaU32 InC; OaU32 L; OaU32 K; OaU32 S; OaU32 P; OaU32 D; OaU32 OutL;
	} push{
		static_cast<OaU32>(InN), static_cast<OaU32>(InC), static_cast<OaU32>(InL),
		static_cast<OaU32>(InK), static_cast<OaU32>(InStride),
		static_cast<OaU32>(InPadding), static_cast<OaU32>(InDilation),
		static_cast<OaU32>(InOutL)};

	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Col2Im1d", {&InDCols, &dX}, access, &push, sizeof(push),
		OaDivCeil(static_cast<OaU32>(InN) * static_cast<OaU32>(InC) * static_cast<OaU32>(InL), 256));

	return dX;
}

OaMatrix OaFnMatrix::Conv1dGemm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InStride, OaI32 InPadding, OaI32 InDilation)
{
	assert(InX.Rank() == 3 && "Conv1dGemm input must be 3D [N, InC, L]");
	assert(InWeight.Rank() == 3 && "Conv1dGemm weight must be 3D [OutC, InC, K]");

	const OaI32 N = static_cast<OaI32>(InX.Size(0));
	const OaI32 InC = static_cast<OaI32>(InX.Size(1));
	const OaI32 OutC = static_cast<OaI32>(InWeight.Size(0));
	const OaI32 K = static_cast<OaI32>(InWeight.Size(2));
	const OaI32 L = static_cast<OaI32>(InX.Size(2));
	const OaI32 outL = Conv1dOutLen(L, K, InStride, InPadding, InDilation);

	// Use the differentiable primitives: MatMulNt (the autograd GEMM — Linear's
	// ctx.AddLinear path is forward-only for non-parameter weights) + BiasAdd,
	// and the OaFnMatrix reshape/transpose (the OaMatrix:: methods are view-only
	// and would sever the backward graph).
	auto cols = OaFnMatrix::Im2Col1d(InX, K, InStride, InPadding, InDilation);  // [N*outL, InC*K]
	auto wFlat = OaFnMatrix::Reshape(InWeight, OaMatrixShape{OutC, static_cast<OaI64>(InC) * K});
	auto y = OaFnMatrix::MatMulNt(cols, wFlat);                                  // [N*outL, OutC]
	if (not InBias.IsEmpty()) {
		// Differentiable broadcast add ([M,OutC] + [1,OutC]); OaFnMatrix::BiasAdd
		// is a forward-only inference kernel and would sever the backward graph.
		y = OaFnMatrix::Add(y, OaFnMatrix::Reshape(InBias, OaMatrixShape{1, OutC}));
	}
	auto y3 = OaFnMatrix::Reshape(y, OaMatrixShape{N, outL, OutC});
	return OaFnMatrix::Transpose(y3, 1, 2);                                      // [N, OutC, outL]
}

OaMatrix OaFnMatrix::Conv1dReluGemm(
	const OaMatrix& InX, const OaMatrix& InWeight, const OaMatrix& InBias,
	OaI32 InStride, OaI32 InPadding, OaI32 InDilation)
{
	const OaI32 N = static_cast<OaI32>(InX.Size(0));
	const OaI32 InC = static_cast<OaI32>(InX.Size(1));
	const OaI32 OutC = static_cast<OaI32>(InWeight.Size(0));
	const OaI32 K = static_cast<OaI32>(InWeight.Size(2));
	const OaI32 L = static_cast<OaI32>(InX.Size(2));
	const OaI32 outL = Conv1dOutLen(L, K, InStride, InPadding, InDilation);

	auto cols = OaFnMatrix::Im2Col1d(InX, K, InStride, InPadding, InDilation);
	auto wFlat = OaFnMatrix::Reshape(InWeight, OaMatrixShape{OutC, static_cast<OaI64>(InC) * K});
	auto y = OaFnMatrix::MatMulNt(cols, wFlat);
	if (not InBias.IsEmpty()) {
		y = OaFnMatrix::Add(y, OaFnMatrix::Reshape(InBias, OaMatrixShape{1, OutC}));
	}
	y = OaFnMatrix::Relu(y);
	auto y3 = OaFnMatrix::Reshape(y, OaMatrixShape{N, outL, OutC});
	return OaFnMatrix::Transpose(y3, 1, 2);
}
