#pragma once

#include "GraphBuilder.h"

#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/SemanticBinding.h>
#include <Oa/Runtime/SemanticGraph.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;
class OaComputeGraph;
class OaMatrix;
class OaVkStream;

// Internal owner of one context's executable-graph lifecycle. Recording,
// active/deferred/reusable graph rotation, cross-graph hazards, completion
// state, phase statistics, and post-completion reclamation live here. Queue
// submission and host-boundary policy remain in OaContext until the next
// migration seam is proven.
class OaExecutionSession {
public:
	explicit OaExecutionSession(OaEngine* InEngine);
	~OaExecutionSession();

	OaExecutionSession(const OaExecutionSession&) = delete;
	OaExecutionSession& operator=(const OaExecutionSession&) = delete;
	OaExecutionSession(OaExecutionSession&&) = delete;
	OaExecutionSession& operator=(OaExecutionSession&&) = delete;

	[[nodiscard]] OaStatus Record(const OaComputeDispatchDesc& InDesc);
	[[nodiscard]] OaResult<OaSemanticOperationId> RecordOperation(
		const OaOperationContract& InContract,
		OaSpan<const OaMatrix* const> InInputs,
		OaSpan<const OaMatrix* const> InOutputs,
		OaSpan<const OaOperationAttribute> InAttributes);
	[[nodiscard]] OaComputeGraph* Graph() const noexcept { return Graph_; }
	[[nodiscard]] OaSemanticGraph* SemanticGraph() noexcept { return &SemanticGraph_; }
	[[nodiscard]] const OaSemanticGraph* SemanticGraph() const noexcept {
		return &SemanticGraph_;
	}
	[[nodiscard]] OaU32 NodeCount() const noexcept;
	[[nodiscard]] OaBool HasUnexecutedWork() const noexcept;
	[[nodiscard]] OaStatus ValidateLowering() const;
	[[nodiscard]] const OaStatus& RecordingStatus() const noexcept {
		return RecordingFailure_;
	}
	[[nodiscard]] OaStatus RejectRecording(const OaStatus& InFailure);
	[[nodiscard]] OaStatus ConsumeRecordingFailure();
	[[nodiscard]] OaResult<OaU32> RecordActiveGraphInBatch(
		void* InPrimaryCommandBuffer);
	[[nodiscard]] OaStatus BeginBatch();
	[[nodiscard]] OaResult<OaEvent> SubmitBatch();
	[[nodiscard]] OaStatus Wait(const OaEvent& InEvent);
	[[nodiscard]] OaStatus Abandon();
	[[nodiscard]] OaBool IsBatchActive() const noexcept {
		return ActiveBatchStream_ != nullptr;
	}
	[[nodiscard]] OaBool HasPendingSubmission() const noexcept {
		return PendingBatchStream_ != nullptr;
	}
	[[nodiscard]] OaVkStream* ActiveBatchStream() const noexcept {
		return ActiveBatchStream_;
	}

	[[nodiscard]] OaContextExecutionStats& Stats() noexcept { return Stats_; }
	[[nodiscard]] const OaContextExecutionStats& Stats() const noexcept { return Stats_; }
	void ResetStats() noexcept { Stats_ = OaContextExecutionStats{}; }

	[[nodiscard]] const OaEvent& PendingEvent() const noexcept { return PendingEvent_; }

	void MarkExecuted() noexcept;
	void DiscardActiveRecording();
	void ClearBatchHazards() { BatchBufferStates_.Clear(); }
	void RotateAfterBatch();
	void ReclaimCompletedGraphs();
	void Clear();

	// Repeatable execution frames reuse temporary matrix storage by allocation
	// ordinal. This is execution-session memory policy, not recorder state.
	void BeginStableResourceFrame();
	void EndStableResourceFrame() noexcept;
	// Seal the replay-input prefix. Stable slots allocated before this call are
	// externally live across executions; later slots are capture-local
	// temporaries that a future lifetime planner may consider for aliasing.
	void SealStableResourceInputs();
	// Callers without a prepare/record boundary must conservatively retain every
	// stable slot. This deliberately trades memory for correctness.
	void SealAllStableResourcesExternal();
	[[nodiscard]] OaBool IsStableResourceFrameActive() const noexcept {
		return StableResourceFrameActive_;
	}
	[[nodiscard]] OaBool AreStableResourceInputsSealed() const noexcept {
		return StableResourceInputsSealed_;
	}
	[[nodiscard]] OaUsize StableExternalResourceCount() const noexcept {
		return StableExternalResourceCount_;
	}
	[[nodiscard]] OaUsize StableTransientResourceCount() const noexcept {
		const OaUsize used = StableResourceFrameActive_
			? StableResourceCursor_ : StableResourceCount_;
		return used > StableExternalResourceCount_
			? used - StableExternalResourceCount_
			: 0;
	}
	[[nodiscard]] OaSharedPtr<OaVkBuffer> AllocateMatrixBuffer(
		OaU64 InBytes,
		OaMemoryPlacement InPlacement = OaMemoryPlacement::Auto);
	[[nodiscard]] OaSharedPtr<OaVkBuffer> AllocateMatrixBufferOnNode(
		OaU64 InBytes, OaU32 InNodeIndex);
	[[nodiscard]] OaStatus SnapshotSemanticBindings(
		OaSpan<const OaMatrix* const> InObservedOutputs,
		OaVec<OaSemanticStorageBinding>& OutBindings,
		OaVec<OaCapturedResourceDesc>& OutResourceDescs,
		OaVec<OaSharedPtr<OaVkBuffer>>& OutResources) const;
	void ReleaseStableTransientResources(OaSpan<void*> InRetiredHandles);

private:
	class BatchBufferState {
	public:
		OaVkBuffer Buffer;
		OaBool Read = false;
		OaBool Write = false;
		OaBool IndirectRead = false;
	};
	class SemanticValueBinding {
	public:
		OaSharedPtr<OaVkBuffer> Storage;
		OaU64 ByteOffset = 0;
		OaMatrixShape Shape{};
		OaArray<OaI64, OA_MAX_TENSOR_DIMS> Strides{};
		OaScalarType Dtype = OaScalarType::Float32;
		OaSemanticValueId Value = OaInvalidSemanticValueId;
	};
	[[nodiscard]] OaResult<OaSemanticValueId> FindOrAddSemanticValue(
		const OaMatrix& InMatrix, OaBool InExternal);
	[[nodiscard]] OaResult<OaSemanticValueId> AddSemanticOutputValue(
		const OaMatrix& InMatrix);
	void ClearSemanticRecording() noexcept;
	[[nodiscard]] static BatchBufferState* FindBatchBufferState(
		OaVec<BatchBufferState>& InStates, const OaVkBuffer& InBuffer);
	static void MergeBatchBufferState(
		OaVec<BatchBufferState>& InStates,
		const OaVkBuffer& InBuffer,
		OaBool InRead,
		OaBool InWrite,
		OaBool InIndirectRead);
	[[nodiscard]] OaU32 EmitBatchBoundaryBarriers(
		void* InPrimaryCommandBuffer,
		const OaComputeGraph& InIncoming);
	[[nodiscard]] OaStatus CancelActiveBatch_();
	[[nodiscard]] OaStatus CompletePendingBatch_();
	void RetirePendingBatch_();
	[[nodiscard]] OaStatus FailRecording_(const OaStatus& InFailure);

	OaEngine* Engine_ = nullptr;
	OaComputeGraph* Graph_ = nullptr;
	OaGraphBuilder Builder_;
	OaSemanticGraph SemanticGraph_;
	OaVec<SemanticValueBinding> SemanticValueBindings_;
	OaBool Executed_ = false;
	OaVec<OaComputeGraph*> DeferredGraphs_;
	OaVec<OaComputeGraph*> ReusableGraphs_;
	OaVec<BatchBufferState> BatchBufferStates_;
	OaVkStream* ActiveBatchStream_ = nullptr;
	OaVkStream* PendingBatchStream_ = nullptr;
	OaEvent PendingEvent_;
	OaStatus RecordingFailure_;
	OaContextExecutionStats Stats_;
	OaVec<OaSharedPtr<OaVkBuffer>> StableResourceSlots_;
	OaUsize StableResourceCursor_ = 0;
	OaUsize StableResourceCount_ = 0;
	OaUsize StableExternalResourceCount_ = 0;
	OaBool StableResourceFrameActive_ = false;
	OaBool StableResourceInputsSealed_ = false;
};
