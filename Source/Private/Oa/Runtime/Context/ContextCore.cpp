// OaContext implementation — Unified execution context for all compute operations

#include <Oa/Runtime/Context.h>
#include "ContextImpl.h"
#include <Oa/Runtime/SemanticGraph.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>

#include <atomic>

// ═════════════════════════════════════════════════════════════════════════════
// Thread-local default context
// ═════════════════════════════════════════════════════════════════════════════

static thread_local OaContext* sDefaultContext = nullptr;

// ═════════════════════════════════════════════════════════════════════════════
// OaContext Implementation
// ═════════════════════════════════════════════════════════════════════════════

OaContext::OaContext(OaEngine* InEngine)
	: Impl_(new OaContextImpl(InEngine))
{
	assert(InEngine and "Engine cannot be null");
	
	OA_LOG_INFO(OaLogComponent::Core, "OaContext created");
}

OaContext::~OaContext() {
	// Destruction is never an execution boundary. An active, never-submitted
	// batch is cancelled; an incomplete submitted batch transfers to engine
	// retirement with its exact event and retained graph resources.
	const OaBool failedRecording =
		not Impl_->Execution_.RecordingStatus().IsOk();
	const OaBool discarded = failedRecording
		or Impl_->Execution_.HasUnexecutedWork()
		or Impl_->Execution_.IsBatchActive();
	if (const auto status = Impl_->Execution_.Abandon(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext failed to abandon execution state: %s",
			status.GetMessage().c_str());
	}
	if (failedRecording or Impl_->Execution_.HasUnexecutedWork()) {
		Impl_->Execution_.Clear();
	}
	if (discarded) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaContext discarded an unsubmitted recording during destruction");
	}
	delete Impl_;
	Impl_ = nullptr;
	OA_LOG_INFO(OaLogComponent::Core, "OaContext destroyed");
}

OaContext* OaContext::Create(OaEngine* InEngine) {
	assert(InEngine and "OaContext::Create: null engine");
	return new OaContext(InEngine);
}

OaEngine& OaContext::Engine() const noexcept {
	assert(Impl_->Engine_ and "OaContext::Engine: engine is null");
	return *Impl_->Engine_;
}

OaEngine* OaContext::VkCompute() const noexcept { return Impl_->Engine_; }
OaEngine* OaContext::GetEngine() const noexcept { return Impl_->Engine_; }
OaComputeGraph* OaContext::Graph() const noexcept { return Impl_->Execution_.Graph(); }
OaSemanticGraph* OaContext::SemanticGraph() const noexcept {
	return Impl_->Execution_.SemanticGraph();
}
const OaContextExecutionStats& OaContext::LastExecutionStats() const noexcept {
	return Impl_->Execution_.Stats();
}
bool OaContext::HasCompute()    const noexcept { return Impl_->Engine_ != nullptr and Impl_->Engine_->HasCompute(); }

OaStatus OaContext::Record(const OaComputeDispatchDesc& InDesc) {
	return Impl_->Execution_.Record(InDesc);
}

OaStatus OaContext::Record(const OaMatrixDispatchDesc& InDesc) {
	if (not InDesc.Dispatch.Buffers.Empty()
		or not InDesc.Dispatch.BufferOwners.Empty()
		or InDesc.Dispatch.Indirect
		or InDesc.Dispatch.IndirectBuffer.Buffer
		or InDesc.Dispatch.IndirectOffset != 0)
	{
		return Impl_->Execution_.RejectRecording(OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"matrix dispatch record: raw buffer and indirect fields must be empty"));
	}

	OaVec<OaVkBuffer> buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> owners;
	buffers.Reserve(InDesc.Matrices.Size());
	owners.Reserve(InDesc.Matrices.Size());

	OaU32 dtype = 0;
	OaBool retainsIndirectArgs = false;
	for (const OaMatrix* matrix : InDesc.Matrices) {
		if (matrix) {
			const OaScalarType scalarType = matrix->GetDtype();
			if (scalarType == OaScalarType::BFloat16
				or scalarType == OaScalarType::Float16)
			{
				dtype = 1;
			}
			if (matrix == InDesc.IndirectArgs) retainsIndirectArgs = true;
		}
		if (matrix and matrix->VkBuf_) {
			buffers.PushBack(*matrix->VkBuf_);
			owners.PushBack(matrix->VkBuf_);
		} else {
			buffers.PushBack(OaVkBuffer{});
			owners.PushBack({});
		}
	}

	if (InDesc.IndirectArgs
		and (not InDesc.IndirectArgs->VkBuf_ or not retainsIndirectArgs))
	{
		return Impl_->Execution_.RejectRecording(OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"matrix dispatch record: indirect arguments must have storage and be retained in Matrices"));
	}
	if (not InDesc.IndirectArgs and InDesc.IndirectOffset != 0) {
		return Impl_->Execution_.RejectRecording(OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"matrix dispatch record: indirect offset without indirect arguments"));
	}

	OaComputeDispatchDesc dispatch = InDesc.Dispatch;
	dispatch.Buffers = buffers.Span();
	dispatch.BufferOwners = owners.Span();
	dispatch.Dtype = dtype;
	if (InDesc.IndirectArgs) {
		dispatch.IndirectBuffer = *InDesc.IndirectArgs->VkBuf_;
		dispatch.IndirectOffset = InDesc.IndirectOffset;
		dispatch.Indirect = true;
	}
	return Record(dispatch);
}

OaResult<OaSemanticOperationId> OaContext::RecordOperation(
	const OaOperationContract& InContract,
	std::initializer_list<const OaMatrix*> InInputs,
	std::initializer_list<const OaMatrix*> InOutputs,
	std::initializer_list<OaOperationAttribute> InAttributes)
{
	return Impl_->Execution_.RecordOperation(
		InContract,
		OaSpan<const OaMatrix* const>(InInputs.begin(), InInputs.size()),
		OaSpan<const OaMatrix* const>(InOutputs.begin(), InOutputs.size()),
		OaSpan<const OaOperationAttribute>(
			InAttributes.begin(), InAttributes.size()));
}

void OaContext::SetDefault(OaContext* InContext) {
	sDefaultContext = InContext;
}

OaContext* OaContext::GetDefaultPtr() noexcept {
	return sDefaultContext;
}

OaContext& OaContext::GetDefault() {
	assert(sDefaultContext and "Default context not set. Call OaContext::SetDefault() or initialize engine.");
	return *sDefaultContext;
}

// ═════════════════════════════════════════════════════════════════════════════
// OaContext dispatch recording
// ═════════════════════════════════════════════════════════════════════════════

void OaContext::Add(
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	OaStringView InOperation,
	OaU64 InImplementationId,
	OaU64 InOperationContractHash,
	OaU64 InKernelContentHash,
	OaU64 InProblemContractHash,
	OaSemanticOperationId InSemanticOperation
) {
	OaComputeDispatchDesc desc;
	desc.Operation = InOperation;
	if (InSemanticOperation != OaInvalidSemanticOperationId) {
		desc.SemanticOperations = OaSpan<const OaSemanticOperationId>(
			&InSemanticOperation, 1U);
	}
	desc.ImplementationId = InImplementationId;
	desc.OperationContractHash = InOperationContractHash;
	desc.ProblemContractHash = InProblemContractHash;
	desc.KernelContentHash = InKernelContentHash;
	desc.Kernel = InKernelName;
	desc.Buffers = InBuffers;
	desc.Access = InAccess;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.GroupsX = InGroupsX;
	desc.GroupsY = InGroupsY;
	desc.GroupsZ = InGroupsZ;
	(void)Record(desc);
}

void OaContext::Add(
	OaStringView InKernelName,
	std::initializer_list<const OaMatrix*> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	OaStringView InOperation,
	OaU64 InImplementationId,
	OaU64 InOperationContractHash,
	OaU64 InKernelContentHash,
	OaU64 InProblemContractHash,
	OaSemanticOperationId InSemanticOperation
) {
	OaMatrixDispatchDesc desc;
	desc.Dispatch.Operation = InOperation;
	if (InSemanticOperation != OaInvalidSemanticOperationId) {
		desc.Dispatch.SemanticOperations =
			OaSpan<const OaSemanticOperationId>(&InSemanticOperation, 1U);
	}
	desc.Dispatch.ImplementationId = InImplementationId;
	desc.Dispatch.OperationContractHash = InOperationContractHash;
	desc.Dispatch.ProblemContractHash = InProblemContractHash;
	desc.Dispatch.KernelContentHash = InKernelContentHash;
	desc.Dispatch.Kernel = InKernelName;
	desc.Dispatch.Access = InAccess;
	desc.Dispatch.PushData = InPush;
	desc.Dispatch.PushSize = InPushSize;
	desc.Dispatch.GroupsX = InGroupsX;
	desc.Dispatch.GroupsY = InGroupsY;
	desc.Dispatch.GroupsZ = InGroupsZ;
	desc.Matrices = OaSpan<const OaMatrix* const>(InMatrices.begin(), InMatrices.size());
	(void)Record(desc);
}
