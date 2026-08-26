// oa::FnMatrix — ML Linear layer operations (semantic recording into oa::ExecutionSession).
//
// Linear and LinearRelu are ML-specific neural network layers, moved from Core
// to Ml to separate general matrix operations from ML-specific functionality.
// MatMul remains in Core as it's a general BLAS operation.

#include <oa/ml/fnMatrix.h>
#include <oa/core/matrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/gemm/graphLowering.h>
#include "../../autograd/autogradAttach.gen.h"

#include <oa/core/validation.h>

#include <assert.h>

#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
static oa::Status validateLinear(
	const oa::Matrix& inX, const oa::Matrix& inWeight)
{
	OA_VALIDATE(inX.rank() >= 2, oa::ValidationSeverity::Error, oa::LogComponent::Ml,
		"Linear: input must be >=2D, got rank=%d", inX.rank());
	OA_VALIDATE(inWeight.rank() == 2, oa::ValidationSeverity::Error, oa::LogComponent::Ml,
		"Linear: weight must be 2D [N,K], got rank=%d", inWeight.rank());
	OA_VALIDATE_DTYPE(inX, inWeight, "Linear");
	OA_VALIDATE_SHAPE_COMPAT(inX, inWeight, "Linear");
	return oa::Status::ok();
}
#endif

namespace {

oa::Status recordLinear(
	oa::ExecutionSession& inContext,
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix* inBias,
	oa::Matrix& outY,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK,
	oa::GemmEpilogue inEpilogue,
	const oa::OpContract& inContract,
	oa::U32 inSemanticOp)
{
	auto status = oa::GemmGraphLowering::record(inContext, {
		.a = &inX,
		.b = &inWeight,
		.bias = inBias,
		.c = &outY,
		.m = inM,
		.n = inN,
		.k = inK,
		.precision = oa::MatMulPrecision::Auto,
		.epilogue = inEpilogue,
		.operation = inContract.name,
		.opContractHash = inContract.hash,
		.semanticOp = inSemanticOp,
	});
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Ml, "%.*s lowering failed: %s",
			static_cast<int>(inContract.name.size()), inContract.name.data(),
			status.getMessage().cStr());
	}
	return status;
}

oa::MatrixShape linearOutputShape(const oa::Matrix& inX, oa::I64 inN) {
	auto shape = inX.getShape();
	shape.dims[static_cast<oa::Usize>(shape.rank - 1)] = inN;
	return shape;
}

// GEMM executes a two-dimensional view while the semantic operation consumes
// and produces the public-rank values. This metadata-only execution view must
// not be recorded as a separate public view operation.
oa::Matrix executionView2d(
	const oa::Matrix& inMatrix,
	oa::I64 inRows,
	oa::I64 inColumns)
{
	auto view = inMatrix;
	oa::MatrixAccess::shape(view) = oa::MatrixShape{inRows, inColumns};
	oa::MatrixAccess::stride(view) = oa::Stride::rowMajor(view.getShape());
	oa::MatrixAccess::syncDescriptor(view);
	return view;
}

} // namespace

oa::Matrix oa::FnMatrix::linear(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateLinear(inX, inWeight).isOk());
#endif
	assert(inX.rank() >= 2 && "Linear: input must be >=2D");
	assert(inWeight.rank() == 2 && "Linear: weight must be 2D [N, K]");
	assert(inX.size(inX.rank() - 1) == inWeight.size(1)
		&& "Linear: input last dim must match weight K");
	if (not inBias.isEmpty()) {
		assert(inBias.rank() == 1 && "Linear: bias must be 1D [N]");
		assert(inBias.size(0) == inWeight.size(0)
			&& "Linear: bias size must match weight output dim");
	}

	auto& ctx = oa::ExecutionSession::getActive();

	oa::I64 K = inX.size(inX.rank() - 1);
	oa::I64 N = inWeight.size(0);

	oa::I64 M2d = 1;
	for (oa::I32 i = 0; i < inX.rank() - 1; ++i) M2d *= inX.size(i);

	oa::Matrix out = oa::FnMatrix::empty(linearOutputShape(inX, N), inX.getDtype());
	oa::Matrix executionX = executionView2d(inX, M2d, K);
	oa::Matrix executionOut = executionView2d(out, M2d, N);
	const oa::Matrix* bias = inBias.isEmpty() ? nullptr : &inBias;
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linear,
		{&inX, &inWeight, bias}, {&out});
	if (not semantic.isOk()) return {};
	const auto lowering = recordLinear(
		ctx,
		executionX,
		inWeight,
		bias,
		executionOut,
		static_cast<oa::U32>(M2d),
		static_cast<oa::U32>(N),
		static_cast<oa::U32>(K),
		bias != nullptr ? oa::GemmEpilogue::Bias : oa::GemmEpilogue::None,
		oa::detail::opRegistry::FnMatrix::linear,
		semantic.getValue());
	if (not lowering.isOk()) return {};

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::linear(
		out, inX, inWeight, inBias, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"Linear semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::linearRelu(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateLinear(inX, inWeight).isOk());
#endif
	assert(inX.rank() >= 2 && "LinearRelu: input must be >=2D");
	assert(inWeight.rank() == 2 && "LinearRelu: weight must be 2D [N, K]");
	assert(inX.size(inX.rank() - 1) == inWeight.size(1)
		&& "LinearRelu: input last dim must match weight K");
	if (inBias.isEmpty()) {
		return oa::FnMatrix::relu(oa::FnMatrix::linear(inX, inWeight));
	}
	assert(inBias.rank() == 1 && "LinearRelu: bias must be 1D [N]");
	assert(inBias.size(0) == inWeight.size(0)
		&& "LinearRelu: bias size must match weight output dim");

	auto& ctx = oa::ExecutionSession::getActive();

	oa::I64 K = inX.size(inX.rank() - 1);
	oa::I64 N = inWeight.size(0);

	oa::I64 M2d = 1;
	for (oa::I32 i = 0; i < inX.rank() - 1; ++i) M2d *= inX.size(i);

	oa::Matrix out = oa::FnMatrix::empty(linearOutputShape(inX, N), inX.getDtype());
	oa::Matrix executionX = executionView2d(inX, M2d, K);
	oa::Matrix executionOut = executionView2d(out, M2d, N);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearRelu,
		{&inX, &inWeight, &inBias}, {&out});
	if (not semantic.isOk()) return {};

	const auto lowering = recordLinear(
		ctx,
		executionX,
		inWeight,
		&inBias,
		executionOut,
		static_cast<oa::U32>(M2d),
		static_cast<oa::U32>(N),
		static_cast<oa::U32>(K),
		oa::GemmEpilogue::BiasRelu,
		oa::detail::opRegistry::FnMatrix::linearRelu,
		semantic.getValue());
	if (not lowering.isOk()) return {};

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::linearRelu(
		out, inX, inWeight, inBias, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearRelu semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::linearGelu(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateLinear(inX, inWeight).isOk());
#endif
	assert(inX.rank() >= 2 && "LinearGelu: input must be >=2D");
	assert(inWeight.rank() == 2 && "LinearGelu: weight must be 2D [N, K]");
	assert(inX.size(inX.rank() - 1) == inWeight.size(1)
		&& "LinearGelu: input last dim must match weight K");
	if (inBias.isEmpty()) {
		return oa::FnMatrix::gelu(oa::FnMatrix::linear(inX, inWeight));
	}
	assert(inBias.rank() == 1 && "LinearGelu: bias must be 1D [N]");
	assert(inBias.size(0) == inWeight.size(0)
		&& "LinearGelu: bias size must match weight output dim");

	auto& ctx = oa::ExecutionSession::getActive();

	oa::I64 K = inX.size(inX.rank() - 1);
	oa::I64 N = inWeight.size(0);

	oa::I64 M2d = 1;
	for (oa::I32 i = 0; i < inX.rank() - 1; ++i) M2d *= inX.size(i);

	oa::Matrix out = oa::FnMatrix::empty(linearOutputShape(inX, N), inX.getDtype());
	oa::Matrix executionX = executionView2d(inX, M2d, K);
	oa::Matrix executionOut = executionView2d(out, M2d, N);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearGelu,
		{&inX, &inWeight, &inBias}, {&out});
	if (not semantic.isOk()) return {};

	const auto lowering = recordLinear(
		ctx,
		executionX,
		inWeight,
		&inBias,
		executionOut,
		static_cast<oa::U32>(M2d),
		static_cast<oa::U32>(N),
		static_cast<oa::U32>(K),
		oa::GemmEpilogue::BiasGelu,
		oa::detail::opRegistry::FnMatrix::linearGelu,
		semantic.getValue());
	if (not lowering.isOk()) return {};

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::linearGelu(
		out, inX, inWeight, inBias, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearGelu semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}

oa::Matrix oa::FnMatrix::linearSilu(
	const oa::Matrix& inX,
	const oa::Matrix& inWeight,
	const oa::Matrix& inBias)
{
#if not defined(NDEBUG) or defined(OA_ENABLE_VALIDATION)
	OA_ASSERT(validateLinear(inX, inWeight).isOk());
#endif
	assert(inX.rank() >= 2 && "LinearSilu: input must be >=2D");
	assert(inWeight.rank() == 2 && "LinearSilu: weight must be 2D [N, K]");
	assert(inX.size(inX.rank() - 1) == inWeight.size(1)
		&& "LinearSilu: input last dim must match weight K");
	if (inBias.isEmpty()) {
		return oa::FnMatrix::silu(oa::FnMatrix::linear(inX, inWeight));
	}
	assert(inBias.rank() == 1 && "LinearSilu: bias must be 1D [N]");
	assert(inBias.size(0) == inWeight.size(0)
		&& "LinearSilu: bias size must match weight output dim");

	auto& ctx = oa::ExecutionSession::getActive();

	oa::I64 K = inX.size(inX.rank() - 1);
	oa::I64 N = inWeight.size(0);

	oa::I64 M2d = 1;
	for (oa::I32 i = 0; i < inX.rank() - 1; ++i) M2d *= inX.size(i);

	oa::Matrix out = oa::FnMatrix::empty(linearOutputShape(inX, N), inX.getDtype());
	oa::Matrix executionX = executionView2d(inX, M2d, K);
	oa::Matrix executionOut = executionView2d(out, M2d, N);
	const auto semantic = ctx.recordOp(
		oa::detail::opRegistry::FnMatrix::linearSilu,
		{&inX, &inWeight, &inBias}, {&out});
	if (not semantic.isOk()) return {};

	const auto lowering = recordLinear(
		ctx,
		executionX,
		inWeight,
		&inBias,
		executionOut,
		static_cast<oa::U32>(M2d),
		static_cast<oa::U32>(N),
		static_cast<oa::U32>(K),
		oa::GemmEpilogue::BiasSilu,
		oa::detail::opRegistry::FnMatrix::linearSilu,
		semantic.getValue());
	if (not lowering.isOk()) return {};

	const auto attached = oa::detail::generatedAutogradAttach::FnMatrix::linearSilu(
		out, inX, inWeight, inBias, semantic.getValue());
	if (not attached.isOk()) {
		OaLogError(oa::LogComponent::Ml,
			"LinearSilu semantic autograd attachment failed: %s",
			attached.getMessage().cStr());
		return {};
	}
	return out;
}
