#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/Engine.h>

class OaVkStream;
class OaDispatchHint;
class OaDispatchTicket;

class OaVkBatch {
public:
	OaVkStream* Stream = nullptr;
};

class OaVkDispatch {
public:
	[[nodiscard]] static OaStatus Run(
		OaComputeEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);
	[[nodiscard]] static OaStatus Run(
		OaComputeEngine& InRuntime,
		OaKernelId InKernelId,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);

	// GPU-driven dispatch: workgroup counts read from InIndirectBuffer.
	[[nodiscard]] static OaStatus RunIndirect(
		OaComputeEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer,
		OaU64 InOffset = 0
	);

	[[nodiscard]] static OaResult<OaVkBatch> BeginBatch(OaComputeEngine& InRuntime);

	[[nodiscard]] static OaStatus Record(
		OaVkBatch& InBatch,
		OaComputeEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);
	[[nodiscard]] static OaStatus Record(
		OaVkBatch& InBatch,
		OaComputeEngine& InRuntime,
		OaKernelId InKernelId,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);

	[[nodiscard]] static OaStatus Flush(OaVkBatch& InBatch, OaComputeEngine& InRuntime);

	// Hint-driven dispatch — scheduler routes to optimal device node.
	[[nodiscard]] static OaStatus Run(
		OaComputeEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ,
		const OaDispatchHint& InHint
	);

	// Explicit device node targeting.
	[[nodiscard]] static OaStatus RunOn(
		OaComputeEngine& InRuntime,
		OaU32 InNodeIndex,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);

	// Async dispatch — returns a ticket for later synchronization.
	[[nodiscard]] static OaResult<OaDispatchTicket> RunAsync(
		OaComputeEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);
};

// Look up SPIR-V by name (liboa + external providers) and create pipeline
[[nodiscard]] inline OaStatus OaVkEnsureShader(
	OaComputeEngine& InRt, const char* InName, OaPipelineSpec InSpec
) {
	auto* spirv = OaSpvFindAny(InName);
	if (spirv) {
		return InRt.EnsurePipeline(
			InName, OaSpan<const OaU8>(spirv->Data, spirv->Size), InSpec
		);
	}
	return OaStatus::NotFound("shader not in embedded registry");
}
