#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/Engine.h>

class OaVkStream;

class OaVkBatch {
public:
	OaVkStream* Stream = nullptr;
};

class OaVkDispatch {
public:
	[[nodiscard]] static OaStatus Run(
		OaEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);
	[[nodiscard]] static OaStatus Run(
		OaEngine& InRuntime,
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
		OaEngine& InRuntime,
		OaStringView InPipelineName,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer,
		OaU64 InOffset = 0
	);

	[[nodiscard]] static OaResult<OaVkBatch> BeginBatch(OaEngine& InRuntime);

	[[nodiscard]] static OaStatus Record(
		OaVkBatch& InBatch,
		OaEngine& InRuntime,
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
		OaEngine& InRuntime,
		OaKernelId InKernelId,
		OaSpan<OaVkBuffer> InBuffers,
		const void* InPushData,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1
	);

	[[nodiscard]] static OaStatus Flush(OaVkBatch& InBatch, OaEngine& InRuntime);

	// Explicit device node targeting.
	[[nodiscard]] static OaStatus RunOn(
		OaEngine& InRuntime,
		OaU32 InNodeIndex,
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
	OaEngine& InRt, const char* InName, OaPipelineSpec InSpec
) {
	auto* spirv = OaSpvFindAny(InName);
	if (spirv) {
		return InRt.EnsurePipeline(
			InName, OaSpan<const OaU8>(spirv->Data, spirv->Size), InSpec
		);
	}
	return OaStatus::NotFound("shader not in embedded registry");
}
