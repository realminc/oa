#include <oa/runtime/gemm/graphLowering.h>

#include <oa/core/envFlag.h>
#include <oa/core/matrix.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dispatchDesc.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/gemm/dispatch.h>
#include <oa/runtime/gemm/router.h>

namespace {

oa::U32 deriveNodeDtype(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	const oa::Matrix* inBias,
	const oa::Matrix& outC)
{
	const oa::ScalarType dtypes[] = {inA.getDtype(), inB.getDtype(),
		inBias != nullptr ? inBias->getDtype() : inA.getDtype(),
		outC.getDtype()};
	for (const auto dtype : dtypes) {
		if (dtype == oa::ScalarType::BFloat16) {
			return 1U;
		}
	}
	return 0U;
}

oa::GemmPrecision toGemmPrecision(oa::MatMulPrecision inPrecision) {
	switch (inPrecision) {
		case oa::MatMulPrecision::Auto: return oa::GemmPrecision::Auto;
		case oa::MatMulPrecision::Fp32: return oa::GemmPrecision::Fp32;
		case oa::MatMulPrecision::Bf16: return oa::GemmPrecision::Bf16;
	}
	return oa::GemmPrecision::Auto;
}

} // namespace

oa::Result<oa::U32> oa::GemmGraphLowering::recordMatMulNt(
	const oa::Matrix& inA,
	const oa::Matrix& inB,
	oa::Matrix& outC,
	oa::U32 inM,
	oa::U32 inN,
	oa::U32 inK,
	oa::MatMulPrecision inPrecision)
{
	auto& context = oa::ExecutionSession::getActive();
	const auto semantic = context.recordOp(
		oa::detail::opRegistry::FnMatrix::matMulNt, {&inA, &inB}, {&outC});
	if (not semantic.isOk()) return semantic.getStatus();

	oa::Matrix reshapedA;
	const oa::Matrix* executionA = &inA;
	if (inA.rank() != 2) {
		reshapedA = inA.reshape(oa::MatrixShape{inM, inK});
		executionA = &reshapedA;
	}
	const auto recorded = record(context, {
		.a = executionA,
		.b = &inB,
		.c = &outC,
		.m = inM,
		.n = inN,
		.k = inK,
		.precision = inPrecision,
		.epilogue = oa::GemmEpilogue::None,
		.operation = oa::detail::opRegistry::FnMatrix::matMulNt.name,
		.opContractHash = oa::detail::opRegistry::FnMatrix::matMulNt.hash,
		.semanticOp = semantic.getValue(),
	});
	if (not recorded.isOk()) return recorded;
	return semantic.getValue();
}

oa::Status oa::GemmGraphLowering::record(
	oa::ExecutionSession& inContext,
	const oa::GemmGraphDesc& inDesc)
{
	auto& engine = inContext.engine();
	if (inDesc.a == nullptr or inDesc.b == nullptr or inDesc.c == nullptr
		or inDesc.m == 0U or inDesc.n == 0U or inDesc.k == 0U)
	{
		return oa::Status::error("GEMM graph lowering received an incomplete descriptor");
	}
	const bool fused = inDesc.epilogue != oa::GemmEpilogue::None;
	if (fused != (inDesc.bias != nullptr and not inDesc.bias->isEmpty())) {
		return oa::Status::error("GEMM graph epilogue and bias contract disagree");
	}
	const oa::Matrix& a = *inDesc.a;
	const oa::Matrix& b = *inDesc.b;
	oa::Matrix& c = *inDesc.c;
	if (a.getDtype() != b.getDtype() or a.getDtype() != c.getDtype()
		or (inDesc.bias != nullptr and a.getDtype() != inDesc.bias->getDtype()))
	{
		return oa::Status::error("GEMM graph lowering requires one storage dtype");
	}
	if (a.getDtype() != oa::ScalarType::Float32
		and a.getDtype() != oa::ScalarType::BFloat16)
	{
		return oa::Status::error(oa::StatusCode::Unimplemented,
			"GEMM storage must be Float32 or BFloat16; no implicit FP16/FP64 reinterpretation is allowed");
	}

	const bool isBf16Input = deriveNodeDtype(a, b, inDesc.bias, c) != 0U;
	const oa::StoragePrecision storage = isBf16Input
		? oa::StoragePrecision::Bf16 : oa::StoragePrecision::Fp32;
	auto problem = oa::GemmRouter::problemForRaw(
		inDesc.m, inDesc.n, inDesc.k, storage, storage, true);
	problem.epilogue = inDesc.epilogue;
	problem.precisionHint =
		isBf16Input and not oa::EnvFlag::isSet("OA_GEMM_FORCE_FP32")
			? oa::GemmPrecision::Bf16
			: toGemmPrecision(inDesc.precision);

	const oa::U32 scalarBytes = static_cast<oa::U32>(oa::scalarSize(a.getDtype()));
	if (a.rank() == 2 and b.rank() == 2 and scalarBytes != 0U) {
		problem.a = {
			.offset = static_cast<oa::U32>(a.byteOffset() / scalarBytes),
			.rowStride = static_cast<oa::U32>(a.getStride().stepElements(0)),
			.colStride = static_cast<oa::U32>(a.getStride().stepElements(1)),
			.batchStride = inDesc.m * inDesc.k,
		};
		problem.b = {
			.offset = static_cast<oa::U32>(b.byteOffset() / scalarBytes),
			.rowStride = static_cast<oa::U32>(b.getStride().stepElements(0)),
			.colStride = static_cast<oa::U32>(b.getStride().stepElements(1)),
			.batchStride = inDesc.n * inDesc.k,
		};
		problem.c = {
			.offset = static_cast<oa::U32>(c.byteOffset() / scalarBytes),
			.rowStride = static_cast<oa::U32>(c.getStride().stepElements(0)),
			.colStride = static_cast<oa::U32>(c.getStride().stepElements(1)),
			.batchStride = inDesc.m * inDesc.n,
		};
		problem.aContiguous = a.getStride().matchesRowMajor(a.getShape());
		problem.bContiguous = b.getStride().matchesRowMajor(b.getShape());
	}

	const auto plan = oa::GemmRouter::plan(engine, problem, inDesc.preference);
	if (not plan) {
		return oa::Status::error("GEMM graph lowering found no legal matrix plan");
	}
	if (not oa::GemmRouter::validatePlan(engine, plan, problem)) {
		return oa::Status::error("GEMM graph lowering received an invalid matrix plan");
	}
	auto described = oa::GemmDispatch::describeValidatedPlan(plan, problem);
	if (not described.isOk()) return described.getStatus();
	const auto& launch = described.getValue();

	const bool identityOrder = launch.bufferOrder[0] == 0U
		and launch.bufferOrder[1] == 1U and launch.bufferOrder[2] == 2U
		and (launch.bufferCount == 3U or launch.bufferOrder[3] == 3U);
	const bool coopVecOrder = launch.bufferOrder[0] == 1U
		and launch.bufferOrder[1] == 0U and launch.bufferOrder[2] == 2U;
	if ((launch.bufferCount != 3U and launch.bufferCount != 4U)
		or (not identityOrder and not coopVecOrder))
	{
		return oa::Status::error(
			"GEMM graph lowering received an unsupported buffer permutation");
	}

	oa::BufferAccess plainAccess[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	oa::BufferAccess fusedAccess[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	oa::ComputeDispatchDesc dispatch;
	dispatch.operation = inDesc.operation;
	if (inDesc.semanticOp != oa::invalidSemanticOpId) {
		dispatch.semanticOps = oa::Span<const oa::U32>(
			&inDesc.semanticOp, 1U);
	}
	dispatch.implementationId = plan.variant;
	dispatch.opContractHash = inDesc.opContractHash;
	dispatch.problemContractHash = plan.problemContractHash;
	dispatch.kernelContentHash = plan.shaderContentHash;
	dispatch.kernelSelection = oa::GemmRouter::classifySelection(plan, problem);
	dispatch.kernel = launch.kernelName;
	dispatch.pushData = launch.pushData;
	dispatch.pushSize = launch.pushSize;
	dispatch.groupsX = launch.grid.x;
	dispatch.groupsY = launch.grid.y;
	dispatch.groupsZ = launch.grid.z;

	if (coopVecOrder) {
		const oa::Matrix* matrices[] = {&b, &a, &c};
		dispatch.access = plainAccess;
		return inContext.record( {
			.dispatch = dispatch,
			.matrices = matrices,
		});
	}
	if (fused) {
		const oa::Matrix* matrices[] = {&a, &b, inDesc.bias, &c};
		dispatch.access = fusedAccess;
		return inContext.record( {
			.dispatch = dispatch,
			.matrices = matrices,
		});
	}
	const oa::Matrix* matrices[] = {&a, &b, &c};
	dispatch.access = plainAccess;
	return inContext.record( {
		.dispatch = dispatch,
		.matrices = matrices,
	});
}
