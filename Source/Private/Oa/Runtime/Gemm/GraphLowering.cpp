#include <Oa/Runtime/Gemm/GraphLowering.h>

#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Operation.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Gemm/Dispatch.h>
#include <Oa/Runtime/Gemm/Router.h>

namespace {

OaU32 DeriveNodeDtype(
	const OaMatrix& InA,
	const OaMatrix& InB,
	const OaMatrix* InBias,
	const OaMatrix& OutC)
{
	const OaScalarType dtypes[] = {InA.GetDtype(), InB.GetDtype(),
		InBias != nullptr ? InBias->GetDtype() : InA.GetDtype(),
		OutC.GetDtype()};
	for (const auto dtype : dtypes) {
		if (dtype == OaScalarType::BFloat16
			or dtype == OaScalarType::Float16)
		{
			return 1U;
		}
	}
	return 0U;
}

OaGemmPrecision ToGemmPrecision(OaMatMulPrecision InPrecision) {
	switch (InPrecision) {
		case OaMatMulPrecision::Auto: return OaGemmPrecision::Auto;
		case OaMatMulPrecision::Fp32: return OaGemmPrecision::Fp32;
		case OaMatMulPrecision::Bf16: return OaGemmPrecision::Bf16;
	}
	return OaGemmPrecision::Auto;
}

} // namespace

OaResult<OaSemanticOperationId> OaGemmGraphLowering::RecordMatMulNt(
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaMatMulPrecision InPrecision)
{
	auto& context = OaContext::GetDefault();
	const auto semantic = context.RecordOperation(
		OaOperationRegistry::MatMulNt, {&InA, &InB}, {&OutC});
	if (not semantic.IsOk()) return semantic.GetStatus();

	OaMatrix reshapedA;
	const OaMatrix* executionA = &InA;
	if (InA.Rank() != 2) {
		reshapedA = InA.Reshape(OaMatrixShape{InM, InK});
		executionA = &reshapedA;
	}
	const auto recorded = Record(context, {
		.A = executionA,
		.B = &InB,
		.C = &OutC,
		.M = InM,
		.N = InN,
		.K = InK,
		.Precision = InPrecision,
		.Epilogue = OaGemmEpilogue::None,
		.Operation = OaOperationRegistry::MatMulNt.Name,
		.OperationContractHash = OaOperationRegistry::MatMulNt.Hash,
		.SemanticOperation = semantic.GetValue(),
	});
	if (not recorded.IsOk()) return recorded;
	return semantic.GetValue();
}

OaStatus OaGemmGraphLowering::Record(
	OaContext& InContext,
	const OaGemmGraphDesc& InDesc)
{
	auto* engine = InContext.GetEngine();
	if (engine == nullptr) {
		return OaStatus::Error("GEMM graph lowering requires an active engine");
	}
	if (InDesc.A == nullptr or InDesc.B == nullptr or InDesc.C == nullptr
		or InDesc.M == 0U or InDesc.N == 0U or InDesc.K == 0U)
	{
		return OaStatus::Error("GEMM graph lowering received an incomplete descriptor");
	}
	const bool fused = InDesc.Epilogue != OaGemmEpilogue::None;
	if (fused != (InDesc.Bias != nullptr and not InDesc.Bias->IsEmpty())) {
		return OaStatus::Error("GEMM graph epilogue and bias contract disagree");
	}
	const OaMatrix& a = *InDesc.A;
	const OaMatrix& b = *InDesc.B;
	OaMatrix& c = *InDesc.C;
	if (a.GetDtype() != b.GetDtype() or a.GetDtype() != c.GetDtype()
		or (InDesc.Bias != nullptr and a.GetDtype() != InDesc.Bias->GetDtype()))
	{
		return OaStatus::Error("GEMM graph lowering requires one storage dtype");
	}

	const bool isBf16Input = DeriveNodeDtype(a, b, InDesc.Bias, c) != 0U;
	const OaStoragePrecision storage = isBf16Input
		? OaStoragePrecision::Bf16 : OaStoragePrecision::Fp32;
	auto problem = OaGemmRouter::ProblemForRaw(
		InDesc.M, InDesc.N, InDesc.K, storage, storage, true);
	problem.Epilogue = InDesc.Epilogue;
	problem.PrecisionHint =
		isBf16Input and not OaEnvFlag::IsSet("OA_GEMM_FORCE_FP32")
			? OaGemmPrecision::Bf16
			: ToGemmPrecision(InDesc.Precision);

	const OaU32 scalarBytes = static_cast<OaU32>(OaScalarSize(a.GetDtype()));
	if (a.Rank() == 2 and b.Rank() == 2 and scalarBytes != 0U) {
		problem.A = {
			.Offset = static_cast<OaU32>(a.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(a.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(a.GetStride().StepElements(1)),
			.BatchStride = InDesc.M * InDesc.K,
		};
		problem.B = {
			.Offset = static_cast<OaU32>(b.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(b.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(b.GetStride().StepElements(1)),
			.BatchStride = InDesc.N * InDesc.K,
		};
		problem.C = {
			.Offset = static_cast<OaU32>(c.ByteOffset() / scalarBytes),
			.RowStride = static_cast<OaU32>(c.GetStride().StepElements(0)),
			.ColStride = static_cast<OaU32>(c.GetStride().StepElements(1)),
			.BatchStride = InDesc.M * InDesc.N,
		};
		problem.AContiguous = a.GetStride().MatchesRowMajor(a.GetShape());
		problem.BContiguous = b.GetStride().MatchesRowMajor(b.GetShape());
	}

	const auto plan = OaGemmRouter::Plan(*engine, problem);
	if (not plan) {
		return OaStatus::Error("GEMM graph lowering found no legal matrix plan");
	}
	if (not OaGemmRouter::ValidatePlan(*engine, plan, problem)) {
		return OaStatus::Error("GEMM graph lowering received an invalid matrix plan");
	}
	auto described = OaGemmDispatch::DescribeValidatedPlan(plan, problem);
	if (not described.IsOk()) return described.GetStatus();
	const auto& launch = described.GetValue();

	const bool identityOrder = launch.BufferOrder[0] == 0U
		and launch.BufferOrder[1] == 1U and launch.BufferOrder[2] == 2U
		and (launch.BufferCount == 3U or launch.BufferOrder[3] == 3U);
	const bool coopVecOrder = launch.BufferOrder[0] == 1U
		and launch.BufferOrder[1] == 0U and launch.BufferOrder[2] == 2U;
	if ((launch.BufferCount != 3U and launch.BufferCount != 4U)
		or (not identityOrder and not coopVecOrder))
	{
		return OaStatus::Error(
			"GEMM graph lowering received an unsupported buffer permutation");
	}

	OaBufferAccess plainAccess[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	OaBufferAccess fusedAccess[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	OaComputeDispatchDesc dispatch;
	dispatch.Operation = InDesc.Operation;
	if (InDesc.SemanticOperation != OaInvalidSemanticOperationId) {
		dispatch.SemanticOperations = OaSpan<const OaSemanticOperationId>(
			&InDesc.SemanticOperation, 1U);
	}
	dispatch.ImplementationId = plan.Variant;
	dispatch.OperationContractHash = InDesc.OperationContractHash;
	dispatch.ProblemContractHash = plan.ProblemContractHash;
	dispatch.KernelContentHash = plan.ShaderContentHash;
	dispatch.Kernel = launch.KernelName;
	dispatch.PushData = launch.PushData;
	dispatch.PushSize = launch.PushSize;
	dispatch.GroupsX = launch.Grid.X;
	dispatch.GroupsY = launch.Grid.Y;
	dispatch.GroupsZ = launch.Grid.Z;

	if (coopVecOrder) {
		const OaMatrix* matrices[] = {&b, &a, &c};
		dispatch.Access = plainAccess;
		return InContext.Record({
			.Dispatch = dispatch,
			.Matrices = matrices,
		});
	}
	if (fused) {
		const OaMatrix* matrices[] = {&a, &b, InDesc.Bias, &c};
		dispatch.Access = fusedAccess;
		return InContext.Record({
			.Dispatch = dispatch,
			.Matrices = matrices,
		});
	}
	const OaMatrix* matrices[] = {&a, &b, &c};
	dispatch.Access = plainAccess;
	return InContext.Record({
		.Dispatch = dispatch,
		.Matrices = matrices,
	});
}
