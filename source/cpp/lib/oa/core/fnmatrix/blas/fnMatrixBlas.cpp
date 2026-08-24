// oa::FnMatrix — BLAS operations: matMulNt semantic recording.
//
// MatMulNt is a general-purpose BLAS operation that stays in Core.
// Linear and LinearRelu are ML-specific neural network layers in Ml/.

#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include "../../autograd/autogradAttach.gen.h"
#include <oa/runtime/gemm/graphLowering.h>

#include <oa/core/op.h>

// MatMulNt
oa::Matrix oa::FnMatrix::matMulNt(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	oa::MatMulPrecision inPrecision)
{
	auto inferredShape = oa::inferBinaryOpShape(
		oa::detail::opRegistry::FnMatrix::matMulNt, inA, inB);
	if (not inferredShape.isOk()) return {};

	oa::I64 K = inA.size(inA.rank() - 1);
	oa::I64 N = inB.size(0);

	oa::I64 M2d = 1;
	for (oa::I32 i = 0; i < inA.rank() - 1; ++i) M2d *= inA.size(i);

	OaLogDebug(oa::LogComponent::Compute, "MatMulNt: A.rank=%d K=%lld B=[%lld,%lld] M2d=%lld N=%lld",
		inA.rank(), K, inB.size(0), inB.size(1), M2d, N);

	oa::Matrix out = oa::FnMatrix::empty(inferredShape.getValue(), inA.getDtype());
	const auto lowering = oa::GemmGraphLowering::recordMatMulNt(
		inA, inB, out,
		static_cast<oa::U32>(M2d),
		static_cast<oa::U32>(N),
		static_cast<oa::U32>(K),
		inPrecision);
	if (not lowering.isOk()) {
		OaLogError(oa::LogComponent::Compute, "MatMulNt lowering failed: %s",
			lowering.getStatus().getMessage().cStr());
		return {};
	}

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::matMulNt(
		out, inA, inB, lowering.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"MatMulNt semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}

	return out;
}
