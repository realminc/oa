#include "ExecutionSession.h"

#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>

#include <cassert>
#include <vulkan/vulkan.h>

static bool OaExecutionAccessReads(OaBufferAccess InAccess) {
	return InAccess == OaBufferAccess::Read or InAccess == OaBufferAccess::ReadWrite;
}

static bool OaExecutionAccessWrites(OaBufferAccess InAccess) {
	return InAccess == OaBufferAccess::Write or InAccess == OaBufferAccess::ReadWrite;
}

OaExecutionSession::OaExecutionSession(OaEngine* InEngine)
	: Engine_(InEngine)
	, Graph_(new OaComputeGraph())
	, Builder_(Graph_)
{
	assert(Engine_ and "Engine cannot be null");
}

OaExecutionSession::~OaExecutionSession() {
	// Destruction never submits or waits. Unsubmitted primary recording is reset;
	// submitted graphs move to engine retirement when their exact event is still
	// incomplete.
	if (const auto status = Abandon(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"execution session abandonment failed: %s",
			status.GetMessage().c_str());
	}
	for (auto* graph : DeferredGraphs_) {
		if (Engine_) graph->Reset(Engine_->Device);
		delete graph;
	}
	for (auto* graph : ReusableGraphs_) {
		if (Engine_) graph->Destroy(Engine_->Device);
		delete graph;
	}
	if (Graph_) {
		if (Engine_) Graph_->Destroy(Engine_->Device);
		delete Graph_;
	}
}

OaStatus OaExecutionSession::Record(const OaComputeDispatchDesc& InDesc) {
	if (not RecordingFailure_.IsOk()) return RecordingFailure_;
	if (not Engine_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaExecutionSession::Record '%.*s': null engine",
			static_cast<int>(InDesc.Kernel.Size()), InDesc.Kernel.Data());
		return FailRecording_(OaStatus::Error(OaStatusCode::Internal,
			"execution session record: null engine"));
	}
	const auto status = Builder_.Record(InDesc);
	if (not status.IsOk()) return FailRecording_(status);
	Executed_ = false;
	return OaStatus::Ok();
}

OaResult<OaSemanticValueId> OaExecutionSession::FindOrAddSemanticValue(
	const OaMatrix& InMatrix, OaBool InExternal)
{
	const auto& storage = InMatrix.VkBuf_;
	// Search newest-first. In-place operations create a new logical value for
	// the same storage range, and subsequent readers must observe that version.
	for (OaUsize index = SemanticValueBindings_.Size(); index > 0; --index) {
		const auto& binding = SemanticValueBindings_[index - 1];
		if (binding.Storage.Get() != storage.Get()
			or binding.ByteOffset != InMatrix.ByteOffset()
			or binding.Shape != InMatrix.GetShape()
			or binding.Dtype != InMatrix.GetDtype())
		{
			continue;
		}
		OaBool strideMatches = true;
		for (OaI32 dimension = 0; dimension < binding.Shape.Rank; ++dimension) {
			if (binding.Strides[static_cast<OaUsize>(dimension)]
				!= InMatrix.GetStride().StepElements(dimension))
			{
				strideMatches = false;
				break;
			}
		}
		if (strideMatches) return binding.Value;
	}

	OaSemanticValueDesc desc;
	desc.Kind = OaOperationValueKind::Matrix;
	desc.Shape = InMatrix.GetShape();
	desc.Dtype = InMatrix.GetDtype();
	desc.External = InExternal;
	auto added = SemanticGraph_.AddValue(desc);
	if (not added.IsOk()) return added.GetStatus();

	SemanticValueBinding binding;
	binding.Storage = storage;
	binding.ByteOffset = InMatrix.ByteOffset();
	binding.Shape = InMatrix.GetShape();
	binding.Dtype = InMatrix.GetDtype();
	binding.Value = added.GetValue();
	for (OaI32 dimension = 0; dimension < binding.Shape.Rank; ++dimension) {
		binding.Strides[static_cast<OaUsize>(dimension)] =
			InMatrix.GetStride().StepElements(dimension);
	}
	SemanticValueBindings_.PushBack(binding);
	return added.GetValue();
}

OaResult<OaSemanticValueId> OaExecutionSession::AddSemanticOutputValue(
	const OaMatrix& InMatrix)
{
	OaSemanticValueDesc desc;
	desc.Kind = OaOperationValueKind::Matrix;
	desc.Shape = InMatrix.GetShape();
	desc.Dtype = InMatrix.GetDtype();
	desc.External = false;
	auto added = SemanticGraph_.AddValue(desc);
	if (not added.IsOk()) return added.GetStatus();

	SemanticValueBinding binding;
	binding.Storage = InMatrix.VkBuf_;
	binding.ByteOffset = InMatrix.ByteOffset();
	binding.Shape = InMatrix.GetShape();
	binding.Dtype = InMatrix.GetDtype();
	binding.Value = added.GetValue();
	for (OaI32 dimension = 0; dimension < binding.Shape.Rank; ++dimension) {
		binding.Strides[static_cast<OaUsize>(dimension)] =
			InMatrix.GetStride().StepElements(dimension);
	}
	SemanticValueBindings_.PushBack(binding);
	return added.GetValue();
}

OaResult<OaSemanticOperationId> OaExecutionSession::RecordOperation(
	const OaOperationContract& InContract,
	OaSpan<const OaMatrix* const> InInputs,
	OaSpan<const OaMatrix* const> InOutputs,
	OaSpan<const OaOperationAttribute> InAttributes)
{
	if (not RecordingFailure_.IsOk()) return RecordingFailure_;
	const auto fail = [this](const OaStatus& InFailure)
		-> OaResult<OaSemanticOperationId>
	{
		return FailRecording_(InFailure);
	};
	constexpr OaU8 MaxPackedValueKinds =
		static_cast<OaU8>(sizeof(OaU32) * 2U);
	if (InInputs.Size() != InContract.InputCount
		or InOutputs.Size() != InContract.OutputCount)
	{
		return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
			"execution session semantic operation arity mismatch"));
	}
	if (InContract.InputCount > MaxPackedValueKinds
		or InContract.OutputCount > MaxPackedValueKinds)
	{
		return fail(OaStatus::Error(OaStatusCode::OutOfRange,
			"execution session semantic operation exceeds packed kind capacity"));
	}
	if (InContract.Name.Empty() or InContract.Hash == 0U) {
		return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
			"execution session semantic operation requires a named contract"));
	}
	for (OaU32 index = 0; index < InContract.InputCount; ++index) {
		const auto kind = static_cast<OaOperationValueKind>(
			(InContract.InputKinds >> (index * 4U)) & 0x0fU);
		if (kind != OaOperationValueKind::Matrix) {
			return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
				"matrix execution session received a non-matrix semantic input"));
		}
	}
	for (OaU32 index = 0; index < InContract.OutputCount; ++index) {
		const auto kind = static_cast<OaOperationValueKind>(
			(InContract.OutputKinds >> (index * 4U)) & 0x0fU);
		if (kind != OaOperationValueKind::Matrix) {
			return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
				"matrix execution session received a non-matrix semantic output"));
		}
	}
	OaVec<OaSemanticValueId> inputs;
	OaVec<OaSemanticValueId> outputs;
	inputs.Reserve(InInputs.Size());
	outputs.Reserve(InOutputs.Size());
	for (const auto* matrix : InInputs) {
		if (not matrix or not matrix->HasStorage()) {
			return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
				"execution session semantic operation has an invalid input matrix"));
		}
		auto value = FindOrAddSemanticValue(*matrix, true);
		if (not value.IsOk()) return fail(value.GetStatus());
		inputs.PushBack(value.GetValue());
	}
	for (const auto* matrix : InOutputs) {
		if (not matrix or not matrix->HasStorage()) {
			return fail(OaStatus::Error(OaStatusCode::InvalidArgument,
				"execution session semantic operation has an invalid output matrix"));
		}
		// Every semantic output is a new SSA version, including in-place writes
		// that intentionally reuse the same physical storage as an input.
		auto value = AddSemanticOutputValue(*matrix);
		if (not value.IsOk()) return fail(value.GetStatus());
		outputs.PushBack(value.GetValue());
	}
	auto operation = SemanticGraph_.AddOperation(
		InContract,
		OaSpan<const OaSemanticValueId>(inputs.Data(), inputs.Size()),
		OaSpan<const OaSemanticValueId>(outputs.Data(), outputs.Size()),
		{}, InAttributes);
	if (not operation.IsOk()) return fail(operation.GetStatus());
	Executed_ = false;
	return operation;
}

OaStatus OaExecutionSession::SnapshotSemanticBindings(
	OaSpan<const OaMatrix* const> InObservedOutputs,
	OaVec<OaSemanticStorageBinding>& OutBindings,
	OaVec<OaCapturedResourceDesc>& OutResourceDescs,
	OaVec<OaSharedPtr<OaVkBuffer>>& OutResources) const
{
	OutBindings.Clear();
	OutResourceDescs.Clear();
	OutResources.Clear();
	if (SemanticValueBindings_.Size() != SemanticGraph_.ValueCount()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"semantic storage snapshot requires one binding per value");
	}
	if (StableExternalResourceCount_ > StableResourceSlots_.Size()) {
		return OaStatus::Error(OaStatusCode::Internal,
			"semantic storage snapshot has an invalid replay-input prefix");
	}

	OutBindings.Reserve(SemanticValueBindings_.Size());
	OutResourceDescs.Reserve(
		SemanticValueBindings_.Size() + InObservedOutputs.Size());
	OutResources.Reserve(
		SemanticValueBindings_.Size() + InObservedOutputs.Size());
	const auto isStableReplayInput = [&](const OaSharedPtr<OaVkBuffer>& InStorage) {
		for (OaUsize index = 0; index < StableExternalResourceCount_; ++index) {
			if (StableResourceSlots_[index].Get() == InStorage.Get()) return true;
		}
		return false;
	};
	const auto isStableTransient = [&](const OaSharedPtr<OaVkBuffer>& InStorage) {
		const OaUsize usedStableResources = StableResourceFrameActive_
			? StableResourceCursor_ : StableResourceCount_;
		for (OaUsize index = StableExternalResourceCount_;
			index < usedStableResources; ++index)
		{
			if (StableResourceSlots_[index].Get() == InStorage.Get()) return true;
		}
		return false;
	};
	const auto findOrAddResource = [&](const OaSharedPtr<OaVkBuffer>& InStorage) {
		for (OaU32 index = 0; index < OutResources.Size(); ++index) {
			if (OutResources[index].Get() == InStorage.Get()) return index;
		}
		const auto resource = static_cast<OaSemanticResourceId>(OutResources.Size());
		OutResources.PushBack(InStorage);
		OaCapturedResourceDesc desc;
		desc.Resource = resource;
		desc.StableReplayInput = isStableReplayInput(InStorage);
		desc.StableTransient = isStableTransient(InStorage);
		desc.Placement = InStorage->Placement;
		desc.ByteSize = InStorage->Size;
		OutResourceDescs.PushBack(desc);
		return resource;
	};
	const auto fail = [&](OaStatusCode InCode, const char* InMessage) {
		OutBindings.Clear();
		OutResourceDescs.Clear();
		OutResources.Clear();
		return OaStatus::Error(InCode, InMessage);
	};
	for (const auto& binding : SemanticValueBindings_) {
		const auto* value = SemanticGraph_.FindValue(binding.Value);
		if (value == nullptr or not binding.Storage) {
			return fail(OaStatusCode::FailedPrecondition,
				"semantic storage snapshot contains an unbound value");
		}
		if (value->Shape != binding.Shape or value->Dtype != binding.Dtype) {
			return fail(OaStatusCode::FailedPrecondition,
				"semantic storage snapshot metadata diverges from the semantic graph");
		}

		const auto resource = findOrAddResource(binding.Storage);
		OutResourceDescs[resource].SemanticExternal |= value->External;
		OaSemanticStorageBinding snapshot;
		snapshot.Value = binding.Value;
		snapshot.Resource = resource;
		snapshot.ByteOffset = binding.ByteOffset;
		snapshot.Shape = binding.Shape;
		snapshot.Strides = binding.Strides;
		snapshot.Dtype = binding.Dtype;
		snapshot.SemanticExternal = value->External;
		snapshot.StableReplayInput = OutResourceDescs[resource].StableReplayInput;
		OutBindings.PushBack(snapshot);
	}

	for (const auto* output : InObservedOutputs) {
		if (output == nullptr or not output->VkBuf_) {
			return fail(OaStatusCode::InvalidArgument,
				"captured observed output must own allocated storage");
		}
		OaBool appearsInGraph = false;
		for (const auto& node : Graph_->Nodes()) {
			for (const auto& buffer : node.Buffers) {
				if (buffer.Buffer == output->VkBuf_->Buffer) {
					appearsInGraph = true;
					break;
				}
			}
			if (appearsInGraph) break;
		}
		if (not appearsInGraph) {
			return fail(OaStatusCode::InvalidArgument,
				"captured observed output is not used by the recorded graph");
		}
		const auto resource = findOrAddResource(output->VkBuf_);
		OutResourceDescs[resource].ObservedOutput = true;
		for (auto& binding : OutBindings) {
			if (binding.Resource == resource) binding.ObservedOutput = true;
		}
	}

	// Schema migration is incomplete, so append every strongly owned executable
	// resource not already reached through a semantic value or observed output.
	// First graph appearance makes the IDs deterministic; stable-transient state
	// remains the positive eligibility proof for compatibility-only resources.
	for (const auto& node : Graph_->Nodes()) {
		for (const auto& owner : node.BufferOwners) {
			if (owner) (void)findOrAddResource(owner);
		}
	}

	const auto lifetimes = Graph_->ComputeLifetimes();
	for (auto& resource : OutResourceDescs) {
		const auto& owner = OutResources[resource.Resource];
		const OaUsize usedStableResources = StableResourceFrameActive_
			? StableResourceCursor_ : StableResourceCount_;
		for (OaUsize index = 0; index < usedStableResources; ++index) {
			if (StableResourceSlots_[index].Get() == owner.Get()) {
				++resource.CaptureRetainedOwnerCount;
			}
		}
		for (const auto& binding : SemanticValueBindings_) {
			if (binding.Storage.Get() == owner.Get()) {
				++resource.CaptureRetainedOwnerCount;
			}
		}
		for (const auto& node : Graph_->Nodes()) {
			for (const auto& nodeOwner : node.BufferOwners) {
				if (nodeOwner.Get() == owner.Get()) {
					++resource.CaptureRetainedOwnerCount;
				}
			}
		}
		for (const auto& lifetime : lifetimes) {
			if (lifetime.Buffer != owner->Buffer) continue;
			resource.HasLifetime = true;
			resource.ByteSize = lifetime.Size;
			resource.FirstAccess = lifetime.FirstAccess;
			resource.LastAccess = lifetime.LastAccess;
			break;
		}
		resource.AliasCandidate = resource.StableTransient
			and resource.HasLifetime and not resource.IsExternallyLive();
	}
	return OaStatus::Ok();
}

void OaExecutionSession::ReleaseStableTransientResources(
	OaSpan<void*> InRetiredHandles)
{
	const OaUsize used = StableResourceFrameActive_
		? StableResourceCursor_ : StableResourceCount_;
	for (OaUsize index = StableExternalResourceCount_; index < used; ++index) {
		auto& slot = StableResourceSlots_[index];
		if (not slot) continue;
		for (void* handle : InRetiredHandles) {
			if (slot->Buffer == handle) {
				slot->Flags |= OA_VK_BUFFER_FLAG_TRANSIENT;
				break;
			}
		}
	}
	if (StableResourceSlots_.Size() > StableExternalResourceCount_) {
		StableResourceSlots_.Resize(StableExternalResourceCount_);
	}
	StableResourceCount_ = StableExternalResourceCount_;
	if (StableResourceFrameActive_) {
		StableResourceCursor_ = StableExternalResourceCount_;
	}
}

void OaExecutionSession::ClearSemanticRecording() noexcept {
	SemanticGraph_.Reset();
	SemanticValueBindings_.Clear();
}

void OaExecutionSession::MarkExecuted() noexcept {
	Executed_ = true;
	ClearSemanticRecording();
}

void OaExecutionSession::DiscardActiveRecording() {
	assert(Graph_ and "Graph is null");
	if (Engine_) Graph_->Reset(Engine_->Device);
	else Graph_->Reset();
	Executed_ = true;
	ClearSemanticRecording();
	RecordingFailure_ = OaStatus::Ok();
}

OaU32 OaExecutionSession::NodeCount() const noexcept {
	assert(Graph_ and "Graph is null");
	return Graph_->NodeCount();
}

OaBool OaExecutionSession::HasUnexecutedWork() const noexcept {
	return not Executed_ and (
		(Graph_ and Graph_->NodeCount() > 0)
		or SemanticGraph_.OperationCount() > 0);
}

OaStatus OaExecutionSession::ValidateLowering() const {
	if (not RecordingFailure_.IsOk()) return RecordingFailure_;
	if (not Graph_) {
		return OaStatus::Error(OaStatusCode::Internal,
			"execution session lowering validation has no executable graph");
	}
	return OaValidateSemanticLowering(SemanticGraph_, *Graph_);
}

OaStatus OaExecutionSession::RejectRecording(const OaStatus& InFailure) {
	return FailRecording_(InFailure);
}

OaStatus OaExecutionSession::ConsumeRecordingFailure() {
	if (RecordingFailure_.IsOk()) return OaStatus::Ok();
	OaStatus failure = RecordingFailure_;
	RecordingFailure_ = OaStatus::Ok();
	return failure;
}

OaStatus OaExecutionSession::FailRecording_(const OaStatus& InFailure) {
	if (InFailure.IsOk()) return OaStatus::Ok();
	if (RecordingFailure_.IsOk()) RecordingFailure_ = InFailure;

	// Recording is one transaction across its semantic and executable forms.
	// The first authoring error rolls both back immediately. If earlier graphs
	// were encoded into an unsubmitted batch, reset that primary command buffer
	// too so a later Submit/Sync cannot execute only a prefix of the workload.
	if (Graph_) {
		if (Engine_) Graph_->Reset(Engine_->Device);
		else Graph_->Reset();
	}
	Executed_ = true;
	ClearSemanticRecording();
	if (ActiveBatchStream_ != nullptr) {
		const auto cancelStatus = CancelActiveBatch_();
		if (not cancelStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"execution session failed to cancel aborted recording batch: %s",
				cancelStatus.GetMessage().c_str());
		}
	}
	return RecordingFailure_;
}

OaExecutionSession::BatchBufferState* OaExecutionSession::FindBatchBufferState(
	OaVec<BatchBufferState>& InStates, const OaVkBuffer& InBuffer)
{
	for (auto& state : InStates) {
		if (state.Buffer.SynchronizationIdentity()
			== InBuffer.SynchronizationIdentity()) return &state;
	}
	return nullptr;
}

void OaExecutionSession::MergeBatchBufferState(
	OaVec<BatchBufferState>& InStates,
	const OaVkBuffer& InBuffer,
	OaBool InRead,
	OaBool InWrite,
	OaBool InIndirectRead)
{
	if (not InBuffer.Buffer) return;
	auto* state = FindBatchBufferState(InStates, InBuffer);
	if (not state) {
		BatchBufferState value;
		value.Buffer = InBuffer;
		InStates.PushBack(value);
		state = &InStates.Back();
	}
	state->Read = state->Read or InRead;
	state->Write = state->Write or InWrite;
	state->IndirectRead = state->IndirectRead or InIndirectRead;
}

OaU32 OaExecutionSession::EmitBatchBoundaryBarriers(
	void* InPrimaryCommandBuffer,
	const OaComputeGraph& InIncoming)
{
	// Secondary command buffers do not create dependencies between themselves.
	// Carry exact buffer access state across graph boundaries so the primary
	// emits only real RAW/WAR/WAW barriers and leaves unrelated work independent.
	OaVec<BatchBufferState> incoming;
	for (const auto& node : InIncoming.Nodes()) {
		for (OaU32 i = 0; i < static_cast<OaU32>(node.Buffers.Size()); ++i) {
			MergeBatchBufferState(
				incoming, node.Buffers[i], OaExecutionAccessReads(node.Access[i]),
				OaExecutionAccessWrites(node.Access[i]), false);
		}
		if (node.Indirect) {
			MergeBatchBufferState(incoming, node.IndirectBuffer, true, false, true);
		}
	}

	OaVec<VkBufferMemoryBarrier2> barriers;
	OaVec<VkMemoryBarrier2> aliasBarriers;
	for (const auto& current : incoming) {
		auto* previous = FindBatchBufferState(BatchBufferStates_, current.Buffer);
		if (not previous) continue;
		const bool hazard =
			(previous->Write and (current.Read or current.Write))
			or (previous->Read and current.Write);
		if (not hazard) continue;

		VkPipelineStageFlags2 srcStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (previous->IndirectRead) {
			srcStages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		VkAccessFlags2 srcAccess = 0;
		if (previous->Read) srcAccess |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (previous->Write) srcAccess |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (previous->IndirectRead) srcAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		VkPipelineStageFlags2 dstStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (current.IndirectRead) {
			dstStages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		VkAccessFlags2 dstAccess = 0;
		if (current.Read) dstAccess |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (current.Write) dstAccess |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (current.IndirectRead) dstAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		if (previous->Buffer.Buffer != current.Buffer.Buffer) {
			VkMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
			barrier.srcStageMask = srcStages;
			barrier.srcAccessMask = previous->Write ? srcAccess : 0;
			barrier.dstStageMask = dstStages;
			barrier.dstAccessMask = previous->Write ? dstAccess : 0;
			aliasBarriers.PushBack(barrier);
		} else {
			VkBufferMemoryBarrier2 barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
			barrier.srcStageMask = srcStages;
			barrier.srcAccessMask = previous->Write ? srcAccess : 0;
			barrier.dstStageMask = dstStages;
			barrier.dstAccessMask = previous->Write ? dstAccess : 0;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.buffer = static_cast<VkBuffer>(current.Buffer.Buffer);
			barrier.offset = 0;
			barrier.size = VK_WHOLE_SIZE;
			barriers.PushBack(barrier);
		}
	}

	if (not barriers.Empty() or not aliasBarriers.Empty()) {
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.memoryBarrierCount = static_cast<OaU32>(aliasBarriers.Size());
		dependency.pMemoryBarriers = aliasBarriers.Data();
		dependency.bufferMemoryBarrierCount = static_cast<OaU32>(barriers.Size());
		dependency.pBufferMemoryBarriers = barriers.Data();
		vkCmdPipelineBarrier2(
			static_cast<VkCommandBuffer>(InPrimaryCommandBuffer), &dependency);
	}

	for (const auto& current : incoming) {
		auto* previous = FindBatchBufferState(BatchBufferStates_, current.Buffer);
		if (previous) {
			*previous = current;
		} else {
			BatchBufferStates_.PushBack(current);
		}
	}
	return static_cast<OaU32>(barriers.Size() + aliasBarriers.Size());
}

OaResult<OaU32> OaExecutionSession::RecordActiveGraphInBatch(
	void* InPrimaryCommandBuffer)
{
	assert(Graph_ and "Graph is null");
	const auto previousStates = BatchBufferStates_;
	const OaU32 barrierCount = EmitBatchBoundaryBarriers(
		InPrimaryCommandBuffer, *Graph_);
	const auto status = Graph_->RecordReplay(*Engine_, InPrimaryCommandBuffer);
	if (not status.IsOk()) {
		BatchBufferStates_ = previousStates;
		return status;
	}
	RotateAfterBatch();
	return barrierCount;
}

OaStatus OaExecutionSession::BeginBatch() {
	if (ActiveBatchStream_ != nullptr) return OaStatus::Ok();
	if (PendingBatchStream_ != nullptr or PendingEvent_.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution session requires Wait() before beginning another batch");
	}
	if (Engine_ == nullptr) {
		return OaStatus::Error(OaStatusCode::Internal,
			"execution session batch has no engine");
	}

	auto* stream = Engine_->AcquireStream();
	if (stream == nullptr) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"execution session failed to acquire a batch stream");
	}
	const auto beginStatus = stream->Begin(Engine_->Device);
	if (not beginStatus.IsOk()) {
		Engine_->ReleaseStream(stream);
		return beginStatus;
	}
	ActiveBatchStream_ = stream;
	return OaStatus::Ok();
}

OaResult<OaEvent> OaExecutionSession::SubmitBatch() {
	if (ActiveBatchStream_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution session has no recorded batch to submit");
	}

	auto* stream = ActiveBatchStream_;
	stream->RecordHostReadbackBarrier();
	const auto submitStatus = stream->Submit(*Engine_);
	ActiveBatchStream_ = nullptr;
	ClearBatchHazards();
	if (not submitStatus.IsOk()) {
		const auto resetStatus = stream->ResetUnsubmitted(Engine_->Device);
		// A command buffer that failed to reset is quarantined in the engine-owned
		// pool until close; returning it to the free stack would permit unsafe reuse.
		if (resetStatus.IsOk()) Engine_->ReleaseStream(stream);
		ReclaimCompletedGraphs();
		return not resetStatus.IsOk() ? resetStatus : submitStatus;
	}

	const OaEvent completion = stream->Completion(Engine_->Device);
	PendingBatchStream_ = stream;
	PendingEvent_ = completion;
	if (not completion.IsValid()) {
		return OaStatus::Error(OaStatusCode::Internal,
			"execution session submitted a batch without a valid completion event");
	}
	return completion;
}

OaStatus OaExecutionSession::CompletePendingBatch_() {
	if (PendingBatchStream_ == nullptr) {
		PendingEvent_ = {};
		return OaStatus::Ok();
	}
	OA_RETURN_IF_ERROR(PendingBatchStream_->Synchronize(Engine_->Device));
	Engine_->ReleaseStream(PendingBatchStream_);
	PendingBatchStream_ = nullptr;
	PendingEvent_ = {};
	ReclaimCompletedGraphs();
	return OaStatus::Ok();
}

OaStatus OaExecutionSession::Wait(const OaEvent& InEvent) {
	if (not InEvent.IsValid()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"execution session cannot wait on an invalid event");
	}
	if (PendingBatchStream_ == nullptr or not PendingEvent_.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"execution session has no pending batch event");
	}
	if (not PendingEvent_.IsSameCompletion(InEvent)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"event does not belong to this execution session's pending batch");
	}
	OA_RETURN_IF_ERROR(InEvent.Wait());
	return CompletePendingBatch_();
}

OaStatus OaExecutionSession::CancelActiveBatch_() {
	if (ActiveBatchStream_ == nullptr) return OaStatus::Ok();
	auto* stream = ActiveBatchStream_;
	ActiveBatchStream_ = nullptr;
	const auto resetStatus = stream->ResetUnsubmitted(Engine_->Device);
	if (resetStatus.IsOk()) Engine_->ReleaseStream(stream);
	ClearBatchHazards();
	ReclaimCompletedGraphs();
	return resetStatus;
}

void OaExecutionSession::RetirePendingBatch_() {
	if (PendingBatchStream_ == nullptr) {
		PendingEvent_ = {};
		return;
	}
	OaVec<OaUniquePtr<OaComputeGraph>> graphs;
	for (auto* graph : DeferredGraphs_) {
		graphs.PushBack(OaUniquePtr<OaComputeGraph>(graph));
	}
	DeferredGraphs_.Clear();
	Engine_->RetireContextBatch(
		PendingBatchStream_, PendingEvent_, OaStdMove(graphs));
	PendingBatchStream_ = nullptr;
	PendingEvent_ = {};
	ClearBatchHazards();
}

OaStatus OaExecutionSession::Abandon() {
	OaStatus result = CancelActiveBatch_();
	if (PendingBatchStream_ != nullptr) {
		if (PendingEvent_.IsValid() and PendingEvent_.IsComplete()) {
			// The poll above is the completion proof. Destruction must not call a
			// wait path, even one with a fast-poll optimization.
			PendingBatchStream_->Submitted = false;
			Engine_->ReleaseStream(PendingBatchStream_);
			PendingBatchStream_ = nullptr;
			PendingEvent_ = {};
			ReclaimCompletedGraphs();
		} else {
			RetirePendingBatch_();
		}
	}
	return result;
}

void OaExecutionSession::RotateAfterBatch() {
	assert(Graph_ and "Graph is null");
	DeferredGraphs_.PushBack(Graph_);
	if (not ReusableGraphs_.Empty()) {
		Graph_ = ReusableGraphs_.Back();
		ReusableGraphs_.PopBack();
	} else {
		Graph_ = new OaComputeGraph();
	}
	Builder_.Attach(Graph_);
	Executed_ = true;
	ClearSemanticRecording();
	RecordingFailure_ = OaStatus::Ok();
}

void OaExecutionSession::ReclaimCompletedGraphs() {
	if (DeferredGraphs_.Empty()) return;

	for (auto* graph : DeferredGraphs_) {
		graph->ClearNodes();
		graph->ReleaseCompletedBufferOwners();
	}
	if (Graph_->NodeCount() == 0) {
		Graph_->Destroy(Engine_->Device);
		delete Graph_;
		Graph_ = DeferredGraphs_[0];
		Builder_.Attach(Graph_);
		for (OaUsize i = DeferredGraphs_.Size(); i > 1; --i) {
			ReusableGraphs_.PushBack(DeferredGraphs_[i - 1]);
		}
	} else {
		for (OaUsize i = DeferredGraphs_.Size(); i > 0; --i) {
			ReusableGraphs_.PushBack(DeferredGraphs_[i - 1]);
		}
	}
	DeferredGraphs_.Clear();
	BatchBufferStates_.Clear();
}

void OaExecutionSession::Clear() {
	assert(Graph_ and "Graph is null");
	if (Engine_) {
		Graph_->ClearNodes();
	} else {
		Graph_->Reset();
	}
	Executed_ = false;
	ClearSemanticRecording();
	RecordingFailure_ = OaStatus::Ok();
}

void OaExecutionSession::BeginStableResourceFrame() {
	assert(not StableResourceFrameActive_
		and "stable resource frames cannot be nested");
	StableResourceCursor_ = 0;
	StableResourceFrameActive_ = true;
	StableResourceInputsSealed_ = false;
}

void OaExecutionSession::EndStableResourceFrame() noexcept {
	if (StableResourceInputsSealed_) {
		StableResourceCount_ = StableResourceCursor_;
	}
	StableResourceCursor_ = 0;
	StableResourceFrameActive_ = false;
	StableResourceInputsSealed_ = false;
}

void OaExecutionSession::SealStableResourceInputs() {
	assert(StableResourceFrameActive_
		and "stable inputs require an active resource frame");
	assert(not StableResourceInputsSealed_
		and "stable resource inputs may only be sealed once per frame");
	StableExternalResourceCount_ = StableResourceCursor_;
	StableResourceInputsSealed_ = true;
}

void OaExecutionSession::SealAllStableResourcesExternal() {
	assert(StableResourceFrameActive_
		and "stable resources require an active resource frame");
	StableExternalResourceCount_ = StableResourceCursor_;
	StableResourceInputsSealed_ = true;
}

OaSharedPtr<OaVkBuffer> OaExecutionSession::AllocateMatrixBuffer(
	OaU64 InBytes, OaMemoryPlacement InPlacement)
{
	if (not Engine_ or InBytes == 0) return {};
	static const OaBool logStableResourceMisses =
		OaEnvFlag::IsSet("OA_LOG_STABLE_RESOURCE_MISSES");

	const auto allocate = [&]() -> OaSharedPtr<OaVkBuffer> {
		auto result = Engine_->AllocBuffer(InBytes, InPlacement);
		if (not result) return {};
		return Engine_->AdoptBufferLease_(std::move(*result));
	};

	if (not StableResourceFrameActive_) return allocate();

	const OaUsize slot = StableResourceCursor_++;
	if (slot < StableResourceSlots_.Size()) {
		auto& existing = StableResourceSlots_[slot];
		// Stable frames deliberately reuse storage by allocation ordinal. A
		// retained matrix object does not change the slot's identity.
		const OaMemoryPlacement resolved = InPlacement == OaMemoryPlacement::Auto
			? Engine_->DefaultMatrixPlacement()
			: InPlacement;
		if (existing and existing->Capacity >= InBytes
			and existing->Placement == resolved)
		{
			existing->Size = InBytes;
			Engine_->UpdateBufferDescriptor(*existing);
			return existing;
		}
		if (logStableResourceMisses) {
			OA_LOG_INFO(OaLogComponent::Core,
				"Stable resource slot %zu replaced: %llu -> %llu bytes",
				slot,
				static_cast<unsigned long long>(existing ? existing->Size : 0),
				static_cast<unsigned long long>(InBytes));
		}
		existing = allocate();
		return existing;
	}

	auto buffer = allocate();
	if (logStableResourceMisses) {
		OA_LOG_INFO(OaLogComponent::Core,
			"Stable resource slot %zu created: %llu bytes", slot,
			static_cast<unsigned long long>(InBytes));
	}
	StableResourceSlots_.PushBack(buffer);
	return buffer;
}

OaSharedPtr<OaVkBuffer> OaExecutionSession::AllocateMatrixBufferOnNode(
	OaU64 InBytes, OaU32 InNodeIndex)
{
	if (not Engine_ or InBytes == 0) return {};
	auto result = Engine_->AllocBufferOnNode(InNodeIndex, InBytes);
	if (not result) return {};
	return Engine_->AdoptBufferLease_(std::move(*result));
}
