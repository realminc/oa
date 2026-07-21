#include "GraphLowering.h"

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/DispatchDesc.h>

#include <climits>

OaStatus OaDnnGraphLowering::RecordLinearWeightBiasBackward(
	OaContext& InContext,
	const OaLinearWeightBiasBwdGraphDesc& InDesc)
{
	if (InDesc.Input == nullptr or InDesc.GradOutput == nullptr
		or InDesc.GradWeight == nullptr or InDesc.GradBias == nullptr)
	{
		return OaStatus::InvalidArgument(
			"LinearWeightBiasBwd lowering requires four matrices");
	}
	const auto& input = *InDesc.Input;
	const auto& gradOutput = *InDesc.GradOutput;
	auto& gradWeight = *InDesc.GradWeight;
	auto& gradBias = *InDesc.GradBias;
	if (input.Rank() != 2 or gradOutput.Rank() != 2
		or gradWeight.Rank() != 2 or gradBias.Rank() != 1)
	{
		return OaStatus::InvalidArgument(
			"LinearWeightBiasBwd lowering requires rank-2 inputs/weight and rank-1 bias");
	}
	const OaI64 m64 = input.Size(0);
	const OaI64 k64 = input.Size(1);
	const OaI64 n64 = gradOutput.Size(1);
	if (m64 <= 0 or n64 <= 0 or k64 <= 0
		or m64 > UINT32_MAX or n64 > UINT32_MAX or k64 > UINT32_MAX
		or gradOutput.Size(0) != m64
		or gradWeight.Size(0) != n64 or gradWeight.Size(1) != k64
		or gradBias.Size(0) != n64)
	{
		return OaStatus::InvalidArgument(
			"LinearWeightBiasBwd lowering received incompatible shapes");
	}
	if (input.GetDtype() != gradOutput.GetDtype()
		or input.GetDtype() != gradWeight.GetDtype()
		or input.GetDtype() != gradBias.GetDtype())
	{
		return OaStatus::Error(OaStatusCode::DtypeMismatch,
			"LinearWeightBiasBwd lowering requires one storage dtype");
	}

	const OaU32 m = static_cast<OaU32>(m64);
	const OaU32 n = static_cast<OaU32>(n64);
	const OaU32 k = static_cast<OaU32>(k64);
	const OaU32 tileN = OaDivCeil(n, 32U);
	const OaU32 tileK = OaDivCeil(k, 32U);
	const bool useTiled = m >= 64U and tileN * tileK >= 9U;

	OaBufferAccess access[] = {
		OaBufferAccess::Read,
		OaBufferAccess::Read,
		OaBufferAccess::Write,
		OaBufferAccess::Write,
	};
	const OaMatrix* matrices[] = {
		&gradOutput, &input, &gradWeight, &gradBias,
	};
	OaComputeDispatchDesc dispatch;
	dispatch.Operation = InDesc.Operation;
	if (InDesc.SemanticOperation != OaInvalidSemanticOperationId) {
		dispatch.SemanticOperations = OaSpan<const OaSemanticOperationId>(
			&InDesc.SemanticOperation, 1U);
	}
	dispatch.OperationContractHash = InDesc.OperationContractHash;
	dispatch.Access = access;
	if (useTiled) {
		struct Push { OaU32 M, N, K; } push{m, n, k};
		dispatch.Kernel = "LinearWeightBiasBwdTiled";
		dispatch.PushData = &push;
		dispatch.PushSize = sizeof(push);
		dispatch.GroupsX = tileN;
		dispatch.GroupsY = tileK;
		return InContext.Record({.Dispatch = dispatch, .Matrices = matrices});
	}

	struct Push { OaU32 M, N, K, Total; }
		push{m, n, k, n * k + n};
	dispatch.Kernel = "LinearWeightBiasBwd";
	dispatch.PushData = &push;
	dispatch.PushSize = sizeof(push);
	dispatch.GroupsX = OaDivCeil(push.Total, 256U);
	return InContext.Record({.Dispatch = dispatch, .Matrices = matrices});
}
