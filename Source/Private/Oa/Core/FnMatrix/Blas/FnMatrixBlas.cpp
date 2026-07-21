// OaFnMatrix — BLAS operations: MatMulNt (semantic recording into OaContext).
//
// MatMulNt is a general-purpose BLAS operation that stays in Core.
// Linear and LinearRelu are ML-specific neural network layers in Ml/.

#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Ml/Autograd/Nodes.h>
#include "../../../Ml/Autograd/AutogradAttach.gen.h"
#include <Oa/Runtime/Gemm/GraphLowering.h>

#include <Oa/Core/Operation.h>

// MatMulNt
OaMatrix OaFnMatrix::MatMulNt(
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatMulPrecision InPrecision)
{
	auto inferredShape = OaInferBinaryOperationShape(
		OaOperationRegistry::MatMulNt, InA, InB);
	if (not inferredShape.IsOk()) return {};

	OaI64 K = InA.Size(InA.Rank() - 1);
	OaI64 N = InB.Size(0);

	OaI64 M2d = 1;
	for (OaI32 i = 0; i < InA.Rank() - 1; ++i) M2d *= InA.Size(i);

	OA_LOG_DEBUG(OaLogComponent::ML, "MatMulNt: A.rank=%d K=%lld B=[%lld,%lld] M2d=%lld N=%lld",
		InA.Rank(), K, InB.Size(0), InB.Size(1), M2d, N);

	OaMatrix out = OaFnMatrix::Empty(inferredShape.GetValue(), InA.Dtype_);
	const auto lowering = OaGemmGraphLowering::RecordMatMulNt(
		InA, InB, out,
		static_cast<OaU32>(M2d),
		static_cast<OaU32>(N),
		static_cast<OaU32>(K),
		InPrecision);
	if (not lowering.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core, "MatMulNt lowering failed: %s",
			lowering.GetStatus().GetMessage().c_str());
		return {};
	}

	const auto attached = OaGeneratedAutogradAttach::OaFnMatrix::MatMulNt(
		out, InA, InB, lowering.GetValue());
	if (not attached.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"MatMulNt semantic autograd attachment failed: %s",
			attached.GetMessage().c_str());
		return {};
	}

	return out;
}
