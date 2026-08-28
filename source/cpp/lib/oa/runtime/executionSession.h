#pragma once

#include "executableGraphBuilder.h"

#include <oa/core/bufferAccess.h>
#include <oa/core/std/array.h>
#include <oa/core/status.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/executionStats.h>
#include <oa/runtime/semanticBinding.h>
#include <oa/runtime/semanticGraph.h>
#include <oa/runtime/sync.h>

#include <initializer_list>

namespace oavk { class Stream; }

namespace oa {

class ExecutableGraph;
class Engine;
class Matrix;
class OpLoweringScope;
class Timer;

// Fixed inline argument packs preserve concise `{&a, &b}` lowering call sites.
// std::initializer_list is the compiler-defined brace-list bridge; it does not
// justify a module-specific list/container type. These packs remain private to
// the semantic/lowering seam and borrow every referenced value.
class MatrixArgs {
public:
	MatrixArgs() = default;
	MatrixArgs(std::initializer_list<const oa::Matrix*> inValues)
		: size_(inValues.size()) {
		OA_REQUIRE(size_ <= values_.size());
		oa::Usize index = 0;
		for (const oa::Matrix* value : inValues) values_[index++] = value;
	}

	template <typename... Values>
		requires (sizeof...(Values) > 0U and sizeof...(Values) <= 32U
			and (... and not oa::IsSameV<oa::RemoveCvrefT<Values>, oa::MatrixArgs>))
	MatrixArgs(Values... inValues)
		: size_(sizeof...(Values)) {
		const oa::Matrix* incoming[] = {
			static_cast<const oa::Matrix*>(inValues)...};
		for (oa::Usize index = 0; index < size_; ++index) values_[index] = incoming[index];
	}

	[[nodiscard]] oa::Span<const oa::Matrix* const> span() const noexcept {
		return {values_.data(), size_};
	}

private:
	oa::Array<const oa::Matrix*, 32> values_{};
	oa::Usize size_ = 0;
};

class OpAttributeArgs {
public:
	OpAttributeArgs() = default;
	OpAttributeArgs(std::initializer_list<oa::OpAttribute> inValues)
		: size_(inValues.size()) {
		OA_REQUIRE(size_ <= values_.size());
		oa::Usize index = 0;
		for (const oa::OpAttribute& value : inValues) values_[index++] = value;
	}

	template <typename... Values>
		requires (sizeof...(Values) > 0U and sizeof...(Values) <= 16U
			and (... and not oa::IsSameV<oa::RemoveCvrefT<Values>, oa::OpAttributeArgs>))
	OpAttributeArgs(Values&&... inValues)
		: size_(sizeof...(Values)) {
		oa::Usize index = 0;
		((values_[index++] = oa::forward<Values>(inValues)), ...);
	}

	[[nodiscard]] oa::Span<const oa::OpAttribute> span() const noexcept {
		return {values_.data(), size_};
	}

private:
	oa::Array<oa::OpAttribute, 16> values_{};
	oa::Usize size_ = 0;
};

// Private mutable execution owner for one engine recording. It owns semantic
// authoring, executable lowering, one-shot batching, stable temporary slots,
// statistics, and exact completion state. The thread-local active pointer only
// selects a borrowed session; it never creates or owns an engine or session.
class ExecutionSession {
public:
	explicit ExecutionSession(oa::Engine* inEngine);
	~ExecutionSession();

	ExecutionSession(const ExecutionSession&) = delete;
	ExecutionSession& operator=(const ExecutionSession&) = delete;
	ExecutionSession(ExecutionSession&&) = delete;
	ExecutionSession& operator=(ExecutionSession&&) = delete;

	static void setActive(oa::ExecutionSession* inSession) noexcept;
	[[nodiscard]] static oa::ExecutionSession* getActivePtr() noexcept;
	[[nodiscard]] static oa::ExecutionSession& getActive();
	[[nodiscard]] static oa::ExecutionSession& forEngine(oa::Engine& inEngine) noexcept;
	[[nodiscard]] static const oa::ExecutionSession& forEngine(
		const oa::Engine& inEngine) noexcept;

	// Select an isolated recorder for the current thread. Scope destruction only
	// restores the previous selector; it never submits or waits.
	class RecordingScope {
	public:
		explicit RecordingScope(oa::ExecutionSession& inSession) noexcept;
		~RecordingScope();

		RecordingScope(const RecordingScope&) = delete;
		RecordingScope& operator=(const RecordingScope&) = delete;
		RecordingScope(RecordingScope&&) noexcept = delete;
		RecordingScope& operator=(RecordingScope&&) noexcept = delete;

	private:
		oa::ExecutionSession* previous_ = nullptr;
	};

	[[nodiscard]] oa::Engine& engine() const noexcept;
	[[nodiscard]] oa::ScalarType weightDtype() const noexcept;
	[[nodiscard]] oa::U32 subgroupSize() const noexcept;

	[[nodiscard]] oa::Status record(const oa::ComputeDispatchDesc& inDesc);
	[[nodiscard]] oa::Status record(const oa::MatrixDispatchDesc& inDesc);
	[[nodiscard]] oa::Result<oa::U32> recordOp(
		const oa::OpContract& inContract,
		oa::Span<const oa::Matrix* const> inInputs,
		oa::Span<const oa::Matrix* const> inOutputs,
		oa::Span<const oa::OpAttribute> inAttributes = {});
	[[nodiscard]] oa::Result<oa::U32> recordOp(
		const oa::OpContract& inContract,
		oa::MatrixArgs inInputs,
		oa::MatrixArgs inOutputs,
		oa::OpAttributeArgs inAttributes = {});
	[[nodiscard]] oa::Status recordView(
		const oa::Matrix& inSource,
		const oa::Matrix& inView);

	void add(
		oa::StringView inKernelName,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush,
		oa::U32 inPushSize,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1,
		oa::StringView inOperation = {},
		oa::U64 inImplementationId = 0,
		oa::U64 inOpContractHash = 0,
		oa::U64 inKernelContentHash = 0,
		oa::U64 inProblemContractHash = 0,
		oa::U32 inSemanticOp = oa::invalidSemanticOpId);
	void add(
		oa::StringView inKernelName,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush,
		oa::U32 inPushSize,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1,
		oa::StringView inOperation = {},
		oa::U64 inImplementationId = 0,
		oa::U64 inOpContractHash = 0,
		oa::U64 inKernelContentHash = 0,
		oa::U64 inProblemContractHash = 0,
		oa::U32 inSemanticOp = oa::invalidSemanticOpId);
	void add(
		oa::StringView inKernelName,
		oa::MatrixArgs inMatrices,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush,
		oa::U32 inPushSize,
		oa::U32 inGroupsX,
		oa::U32 inGroupsY = 1,
		oa::U32 inGroupsZ = 1,
		oa::StringView inOperation = {},
		oa::U64 inImplementationId = 0,
		oa::U64 inOpContractHash = 0,
		oa::U64 inKernelContentHash = 0,
		oa::U64 inProblemContractHash = 0,
		oa::U32 inSemanticOp = oa::invalidSemanticOpId);

	[[nodiscard]] oa::ExecutableGraph* graph() noexcept { return graph_; }
	[[nodiscard]] const oa::ExecutableGraph* graph() const noexcept { return graph_; }
	[[nodiscard]] oa::SemanticGraph* semanticGraph() noexcept { return &semanticGraph_; }
	[[nodiscard]] const oa::SemanticGraph* semanticGraph() const noexcept {
		return &semanticGraph_;
	}
	[[nodiscard]] oa::U32 nodeCount() const noexcept;
	[[nodiscard]] oa::Bool hasUnexecutedWork() const noexcept;
	[[nodiscard]] oa::Status validateCapture();
	[[nodiscard]] const oa::Status& recordingStatus() const noexcept {
		return recordingFailure_;
	}

	[[nodiscard]] oa::Result<oa::Event> submit(oa::Timer* inTimer = nullptr);
	[[nodiscard]] oa::Status wait(const oa::Event& inEvent);
	[[nodiscard]] oa::Status submitAndWait(oa::Timer* inTimer = nullptr);
	[[nodiscard]] oa::Status abandon();
	[[nodiscard]] oa::Bool isBatchActive() const noexcept {
		return activeBatchStream_ != nullptr;
	}
	[[nodiscard]] oa::Bool hasPendingSubmission() const noexcept {
		return pendingBatchStream_ != nullptr;
	}
	[[nodiscard]] oa::Bool isPendingEvent(const oa::Event& inEvent) const noexcept;
	[[nodiscard]] oavk::Stream* activeBatchStream() const noexcept {
		return activeBatchStream_;
	}

	[[nodiscard]] oa::ExecutionStats& stats() noexcept { return stats_; }
	[[nodiscard]] const oa::ExecutionStats& stats() const noexcept { return stats_; }
	[[nodiscard]] const oa::ExecutionStats& lastExecutionStats() const noexcept {
		return stats_;
	}
	void resetStats() noexcept { stats_ = oa::ExecutionStats{}; }

	[[nodiscard]] const oa::Event& pendingEvent() const noexcept { return pendingEvent_; }

	void markExecuted() noexcept;
	void discardActiveRecording();
	void clearBatchHazards() { batchBufferStates_.clear(); }
	void rotateAfterBatch();
	void reclaimCompletedGraphs();
	void clear();

	// Repeatable execution frames reuse temporary matrix storage by allocation
	// ordinal. This is execution-session memory policy, not recorder state.
	void beginStableResourceFrame();
	void endStableResourceFrame() noexcept;
	// Seal the replay-input prefix. Stable slots allocated before this call are
	// externally live across executions; later slots are capture-local
	// temporaries that a future lifetime planner may consider for aliasing.
	void sealStableResourceInputs();
	// Callers without a prepare/record boundary must conservatively retain every
	// stable slot. This deliberately trades memory for correctness.
	void sealAllStableResourcesExternal();
	[[nodiscard]] oa::Bool isStableResourceFrameActive() const noexcept {
		return stableResourceFrameActive_;
	}
	[[nodiscard]] oa::Bool areStableResourceInputsSealed() const noexcept {
		return stableResourceInputsSealed_;
	}
	[[nodiscard]] oa::Usize stableExternalResourceCount() const noexcept {
		return stableExternalResourceCount_;
	}
	[[nodiscard]] oa::Usize stableTransientResourceCount() const noexcept {
		const oa::Usize used = stableResourceFrameActive_
			? stableResourceCursor_ : stableResourceCount_;
		return used > stableExternalResourceCount_
			? used - stableExternalResourceCount_
			: 0;
	}
	[[nodiscard]] oa::SharedPtr<oavk::Buffer> allocateMatrixBuffer(
		oa::U64 inBytes,
		oa::MemoryPlacement inPlacement = oa::MemoryPlacement::Auto);
	[[nodiscard]] oa::Status snapshotSemanticBindings(
		oa::Span<const oa::Matrix* const> inObservedOutputs,
		oa::Vector<oa::SemanticStorageBinding>& outBindings,
		oa::Vector<oa::CapturedResourceDesc>& outResourceDescs,
		oa::Vector<oa::SharedPtr<oavk::Buffer>>& outResources) const;
	void releaseStableTransientResources(oa::Span<void*> inRetiredHandles);

private:
	friend class oa::OpLoweringScope;

	class BatchBufferState {
	public:
		oavk::Buffer buffer;
		oa::Bool read = false;
		oa::Bool write = false;
		oa::Bool indirectRead = false;
	};
	class SemanticValueBinding {
	public:
		oa::SharedPtr<oavk::Buffer> storage;
		oa::U64 byteOffset = 0;
		oa::OpValueKind kind = oa::OpValueKind::Matrix;
		oa::MatrixShape shape{};
		oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> strides{};
		oa::ScalarType dtype = oa::ScalarType::Float32;
		oa::U32 value = oa::invalidSemanticValueId;
	};
	[[nodiscard]] oa::Result<oa::U32> findOrAddSemanticValue(
		const oa::Matrix& inMatrix,
		oa::OpValueKind inKind,
		oa::Bool inExternal);
	[[nodiscard]] oa::Result<oa::U32> addSemanticOutputValue(
		const oa::Matrix& inMatrix,
		oa::OpValueKind inKind);
	void clearSemanticRecording() noexcept;
	[[nodiscard]] static BatchBufferState* findBatchBufferState(
		oa::Vector<BatchBufferState>& inStates, const oavk::Buffer& inBuffer);
	static void mergeBatchBufferState(
		oa::Vector<BatchBufferState>& inStates,
		const oavk::Buffer& inBuffer,
		oa::Bool inRead,
		oa::Bool inWrite,
		oa::Bool inIndirectRead);
	[[nodiscard]] oa::U32 emitBatchBoundaryBarriers(
		void* inPrimaryCommandBuffer,
		const oa::ExecutableGraph& inIncoming);
	void beginOpLowering() noexcept;
	[[nodiscard]] oa::Result<oa::U32> finishOpLowering(
		oa::U32 inFirstNode,
		const oa::OpContract& inContract,
		oa::Span<const oa::Matrix* const> inInputs,
		oa::Span<const oa::Matrix* const> inOutputs,
		oa::Span<const oa::OpAttribute> inAttributes);
	void cancelOpLowering(oa::U32 inFirstNode) noexcept;
	[[nodiscard]] oa::Status validateLowering() const;
	[[nodiscard]] oa::Status rejectRecording(const oa::Status& inFailure);
	[[nodiscard]] oa::Status consumeRecordingFailure();
	[[nodiscard]] oa::Result<oa::U32> recordActiveGraphInBatch_(
		void* inPrimaryCommandBuffer);
	[[nodiscard]] oa::Status beginBatch_();
	[[nodiscard]] oa::Result<oa::Event> submitBatch_();
	[[nodiscard]] oa::Status waitBatch_(const oa::Event& inEvent);
	[[nodiscard]] oa::Status beginSubmission_();
	[[nodiscard]] oa::Status recordActiveGraph_();
	[[nodiscard]] oa::Status recordSubmission_(oa::Timer* inTimer);
	[[nodiscard]] oa::Result<oa::Event> submitRecorded_();
	[[nodiscard]] oa::Status cancelActiveBatch_();
	[[nodiscard]] oa::Status completePendingBatch_();
	void retirePendingBatch_();
	[[nodiscard]] oa::Status failRecording_(const oa::Status& inFailure);

	oa::Engine* engine_ = nullptr;
	oa::ExecutableGraph* graph_ = nullptr;
	oa::ExecutableGraphBuilder builder_;
	oa::SemanticGraph semanticGraph_;
	oa::Vector<SemanticValueBinding> semanticValueBindings_;
	oa::Bool executed_ = false;
	oa::Vector<oa::ExecutableGraph*> deferredGraphs_;
	oa::Vector<oa::ExecutableGraph*> reusableGraphs_;
	oa::Vector<BatchBufferState> batchBufferStates_;
	oavk::Stream* activeBatchStream_ = nullptr;
	oavk::Stream* pendingBatchStream_ = nullptr;
	oa::Event pendingEvent_;
	oa::Status recordingFailure_;
	oa::ExecutionStats stats_;
	oa::Vector<oa::SharedPtr<oavk::Buffer>> stableResourceSlots_;
	oa::Usize stableResourceCursor_ = 0;
	oa::Usize stableResourceCount_ = 0;
	oa::Usize stableExternalResourceCount_ = 0;
	oa::Bool stableResourceFrameActive_ = false;
	oa::Bool stableResourceInputsSealed_ = false;
	oa::U32 opLoweringDepth_ = 0U;
};

// Transactional relation between one semantic operation and the executable
// nodes emitted by its lowering. Destruction rolls back an incomplete lowering;
// it never submits or waits.
class OpLoweringScope {
public:
	explicit OpLoweringScope(oa::ExecutionSession& inSession);
	~OpLoweringScope();

	OpLoweringScope(const OpLoweringScope&) = delete;
	OpLoweringScope& operator=(const OpLoweringScope&) = delete;
	OpLoweringScope(OpLoweringScope&&) noexcept = delete;
	OpLoweringScope& operator=(OpLoweringScope&&) noexcept = delete;

	[[nodiscard]] oa::Status commit(
		const oa::OpContract& inContract,
		oa::Span<const oa::Matrix* const> inInputs,
		oa::Span<const oa::Matrix* const> inOutputs,
		oa::Span<const oa::OpAttribute> inAttributes = {});
	[[nodiscard]] oa::Status commit(
		const oa::OpContract& inContract,
		oa::MatrixArgs inInputs,
		oa::MatrixArgs inOutputs,
		oa::OpAttributeArgs inAttributes = {});
	[[nodiscard]] oa::Result<oa::U32> commitWithId(
		const oa::OpContract& inContract,
		oa::Span<const oa::Matrix* const> inInputs,
		oa::Span<const oa::Matrix* const> inOutputs,
		oa::Span<const oa::OpAttribute> inAttributes = {});
	[[nodiscard]] oa::Result<oa::U32> commitWithId(
		const oa::OpContract& inContract,
		oa::MatrixArgs inInputs,
		oa::MatrixArgs inOutputs,
		oa::OpAttributeArgs inAttributes = {});

private:
	oa::ExecutionSession* session_ = nullptr;
	oa::U32 firstNode_ = 0U;
	oa::Bool active_ = false;
};

} // namespace oa
