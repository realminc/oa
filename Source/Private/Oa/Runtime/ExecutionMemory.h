#pragma once

#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/SemanticBinding.h>

class OaContext;
class OaMatrix;

// Private execution-memory seam used by OA implementation units. Repeatable
// frame storage belongs to OaExecutionSession and is deliberately absent from
// the public OaContext compatibility contract.
class OaExecutionMemory {
public:
	static void BeginStableFrame(OaContext& InContext);
	static void EndStableFrame(OaContext& InContext) noexcept;
	static void SealStableInputs(OaContext& InContext);
	static void SealAllStableResourcesExternal(OaContext& InContext);
	[[nodiscard]] static OaBool IsStableFrameActive(
		const OaContext& InContext) noexcept;
	[[nodiscard]] static OaBool AreStableInputsSealed(
		const OaContext& InContext) noexcept;
	[[nodiscard]] static OaUsize StableExternalResourceCount(
		const OaContext& InContext) noexcept;
	[[nodiscard]] static OaUsize StableTransientResourceCount(
		const OaContext& InContext) noexcept;
	[[nodiscard]] static OaSharedPtr<OaVkBuffer> AllocateMatrixBuffer(
		OaContext& InContext,
		OaU64 InBytes,
		OaMemoryPlacement InPlacement = OaMemoryPlacement::Auto);
	[[nodiscard]] static OaSharedPtr<OaVkBuffer> AllocateMatrixBufferOnNode(
		OaContext& InContext, OaU64 InBytes, OaU32 InNodeIndex);
	[[nodiscard]] static OaStatus SnapshotSemanticBindings(
		const OaContext& InContext,
		OaSpan<const OaMatrix* const> InObservedOutputs,
		OaVec<OaSemanticStorageBinding>& OutBindings,
		OaVec<OaCapturedResourceDesc>& OutResourceDescs,
		OaVec<OaSharedPtr<OaVkBuffer>>& OutResources);
	static void ReleaseStableTransients(
		OaContext& InContext, OaSpan<void*> InRetiredHandles);
};
