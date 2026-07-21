#include "ExecutionMemory.h"

#include "Context/ContextImpl.h"

#include <Oa/Runtime/Context.h>

void OaExecutionMemory::BeginStableFrame(OaContext& InContext) {
	InContext.Impl_->Execution_.BeginStableResourceFrame();
}

void OaExecutionMemory::EndStableFrame(OaContext& InContext) noexcept {
	InContext.Impl_->Execution_.EndStableResourceFrame();
}

void OaExecutionMemory::SealStableInputs(OaContext& InContext) {
	InContext.Impl_->Execution_.SealStableResourceInputs();
}

void OaExecutionMemory::SealAllStableResourcesExternal(OaContext& InContext) {
	InContext.Impl_->Execution_.SealAllStableResourcesExternal();
}

OaBool OaExecutionMemory::IsStableFrameActive(
	const OaContext& InContext) noexcept
{
	return InContext.Impl_->Execution_.IsStableResourceFrameActive();
}

OaBool OaExecutionMemory::AreStableInputsSealed(
	const OaContext& InContext) noexcept
{
	return InContext.Impl_->Execution_.AreStableResourceInputsSealed();
}

OaUsize OaExecutionMemory::StableExternalResourceCount(
	const OaContext& InContext) noexcept
{
	return InContext.Impl_->Execution_.StableExternalResourceCount();
}

OaUsize OaExecutionMemory::StableTransientResourceCount(
	const OaContext& InContext) noexcept
{
	return InContext.Impl_->Execution_.StableTransientResourceCount();
}

OaSharedPtr<OaVkBuffer> OaExecutionMemory::AllocateMatrixBuffer(
	OaContext& InContext,
	OaU64 InBytes,
	OaMemoryPlacement InPlacement)
{
	return InContext.Impl_->Execution_.AllocateMatrixBuffer(
		InBytes, InPlacement);
}

OaSharedPtr<OaVkBuffer> OaExecutionMemory::AllocateMatrixBufferOnNode(
	OaContext& InContext, OaU64 InBytes, OaU32 InNodeIndex)
{
	return InContext.Impl_->Execution_.AllocateMatrixBufferOnNode(
		InBytes, InNodeIndex);
}

OaStatus OaExecutionMemory::SnapshotSemanticBindings(
	const OaContext& InContext,
	OaSpan<const OaMatrix* const> InObservedOutputs,
	OaVec<OaSemanticStorageBinding>& OutBindings,
	OaVec<OaCapturedResourceDesc>& OutResourceDescs,
	OaVec<OaSharedPtr<OaVkBuffer>>& OutResources)
{
	return InContext.Impl_->Execution_.SnapshotSemanticBindings(
		InObservedOutputs, OutBindings, OutResourceDescs, OutResources);
}

void OaExecutionMemory::ReleaseStableTransients(
	OaContext& InContext, OaSpan<void*> InRetiredHandles)
{
	InContext.Impl_->Execution_.ReleaseStableTransientResources(
		InRetiredHandles);
}
