#include "graphLowering.h"

#include <oa/core/bufferAccess.h>
#include <oa/core/envFlag.h>
#include <oa/core/matrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/dispatchDesc.h>

#include <climits>

oa::Status oa::DnnGraphLowering::recordLinearWeightBiasBackward(
	oa::ExecutionSession& inContext,
	const oa::LinearWeightBiasBwdGraphDesc& inDesc)
{
	if (inDesc.input == nullptr or inDesc.gradOutput == nullptr
		or inDesc.gradWeight == nullptr or inDesc.gradBias == nullptr)
	{
		return oa::Status::invalidArgument(
			"LinearWeightBiasBwd lowering requires four matrices");
	}
	const auto& input = *inDesc.input;
	const auto& gradOutput = *inDesc.gradOutput;
	auto& gradWeight = *inDesc.gradWeight;
	auto& gradBias = *inDesc.gradBias;
	if (input.rank() != 2 or gradOutput.rank() != 2
		or gradWeight.rank() != 2 or gradBias.rank() != 1)
	{
		return oa::Status::invalidArgument(
			"LinearWeightBiasBwd lowering requires rank-2 inputs/weight and rank-1 bias");
	}
	const oa::I64 m64 = input.size(0);
	const oa::I64 k64 = input.size(1);
	const oa::I64 n64 = gradOutput.size(1);
	if (m64 <= 0 or n64 <= 0 or k64 <= 0
		or m64 > UINT32_MAX or n64 > UINT32_MAX or k64 > UINT32_MAX
		or gradOutput.size(0) != m64
		or gradWeight.size(0) != n64 or gradWeight.size(1) != k64
		or gradBias.size(0) != n64)
	{
		return oa::Status::invalidArgument(
			"LinearWeightBiasBwd lowering received incompatible shapes");
	}
	if (input.getDtype() != gradOutput.getDtype()
		or input.getDtype() != gradWeight.getDtype()
		or input.getDtype() != gradBias.getDtype())
	{
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			"LinearWeightBiasBwd lowering requires one storage dtype");
	}

	const oa::U32 m = static_cast<oa::U32>(m64);
	const oa::U32 n = static_cast<oa::U32>(n64);
	const oa::U32 k = static_cast<oa::U32>(k64);
	const oa::U32 tileN = oa::divCeil(n, 32U);
	const oa::U32 tileK = oa::divCeil(k, 32U);
	const bool useRows32 = input.getDtype() == oa::ScalarType::Float32
		and m >= 512U and n >= 16U and n <= 64U and k >= 16U and k <= 64U
		and not oa::EnvFlag::isSet("OA_DISABLE_LINEAR_PARAM_ROWS32");
	const bool useTiled = m >= 64U and tileN * tileK >= 9U;

	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
		oa::BufferAccess::Write,
	};
	const oa::Matrix* matrices[] = {
		&gradOutput, &input, &gradWeight, &gradBias,
	};
	oa::ComputeDispatchDesc dispatch;
	dispatch.operation = inDesc.operation;
	if (inDesc.semanticOp != oa::invalidSemanticOpId) {
		dispatch.semanticOps = oa::Span<const oa::U32>(
			&inDesc.semanticOp, 1U);
	}
	dispatch.opContractHash = inDesc.opContractHash;
	dispatch.access = access;
	if (useRows32) {
		struct Push { oa::U32 M, N, K; } push{m, n, k};
		dispatch.kernel = "LinearWeightBiasBwdRows32";
		dispatch.pushData = &push;
		dispatch.pushSize = sizeof(push);
		dispatch.groupsX = n;
		dispatch.groupsY = tileK;
		return inContext.record( {.dispatch = dispatch, .matrices = matrices});
	}
	if (useTiled) {
		struct Push { oa::U32 M, N, K; } push{m, n, k};
		dispatch.kernel = "LinearWeightBiasBwdTiled";
		dispatch.pushData = &push;
		dispatch.pushSize = sizeof(push);
		dispatch.groupsX = tileN;
		dispatch.groupsY = tileK;
		return inContext.record( {.dispatch = dispatch, .matrices = matrices});
	}

	struct Push { oa::U32 M, N, K, total; }
		push{m, n, k, n * k + n};
	dispatch.kernel = "LinearWeightBiasBwd";
	dispatch.pushData = &push;
	dispatch.pushSize = sizeof(push);
	dispatch.groupsX = oa::divCeil(push.total, 256U);
	return inContext.record( {.dispatch = dispatch, .matrices = matrices});
}
