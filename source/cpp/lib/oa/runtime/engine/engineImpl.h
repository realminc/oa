#pragma once

#include <oa/runtime/engine.h>
#include <oa/core/thread.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/bindless.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/device.h>
#include <oa/runtime/pipeline.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/uploadRing.h>
#include "../../core/logAccess.h"
#include "../timerRegistry.h"

#include <oa/core/std/atomic.h>
#include <oa/core/std/sync.h>

namespace oa {

struct RetiredPresenter;
struct RetiredUploadRing;

// The complete vulkan execution state is private to the runtime build. The
// installed oa::Engine header carries only one owning pointer, so adding queues,
// caches, pools, or retirement records no longer changes the public class
// layout or exports raw vulkan resource ownership to consumers.
struct RetiredImageDispatch {
	oavk::Stream* stream = nullptr;
	oa::Vec<oa::U32> storageImageSlots;
	oa::Vec<oa::U32> sampledImageSlots;
	oa::Vec<oa::U32> samplerSlots;
	oa::Vec<VkImageView> imageViews;
};

struct HostVisibleBufferCacheEntry {
	oa::U64 capacity = 0;
	oavk::Buffer buffer;
};

} // namespace oa

class oa::Engine::Impl {
public:
	Impl();
	~Impl();

	Impl(const Impl&) = delete;
	Impl& operator=(const Impl&) = delete;

	class BufferLeaseRegistry;
	class GemmState;
	using RetiredServiceCompleteFn = oa::Status (*)(void*);
	using RetiredServiceReleaseFn = void (*)(void*);

	class RetiredServiceState {
	public:
		RetiredServiceState() = default;
		RetiredServiceState(
			void* inPayload,
			RetiredServiceCompleteFn inComplete,
			RetiredServiceReleaseFn inRelease) noexcept
			: payload_(inPayload)
			, complete_(inComplete)
			, release_(inRelease)
		{}
		RetiredServiceState(const RetiredServiceState&) = delete;
		RetiredServiceState& operator=(const RetiredServiceState&) = delete;
		RetiredServiceState(RetiredServiceState&& inOther) noexcept {
			moveFrom_(oa::move(inOther));
		}
		RetiredServiceState& operator=(RetiredServiceState&& inOther) noexcept {
			if (this != &inOther) {
				releasePayload_();
				moveFrom_(oa::move(inOther));
			}
			return *this;
		}
		~RetiredServiceState() { releasePayload_(); }

		[[nodiscard]] oa::Status complete() {
			oa::Status status = payload_ and complete_
				? complete_(payload_)
				: oa::Status::ok();
			if (status.isOk()) releasePayload_();
			return status;
		}
		void detachWithoutRelease() noexcept {
			payload_ = nullptr;
			complete_ = nullptr;
			release_ = nullptr;
		}

	private:
		void moveFrom_(RetiredServiceState&& inOther) noexcept {
			payload_ = inOther.payload_;
			complete_ = inOther.complete_;
			release_ = inOther.release_;
			inOther.payload_ = nullptr;
			inOther.complete_ = nullptr;
			inOther.release_ = nullptr;
		}
		void releasePayload_() noexcept {
			if (payload_ and release_) release_(payload_);
			payload_ = nullptr;
			complete_ = nullptr;
			release_ = nullptr;
		}

		void* payload_ = nullptr;
		RetiredServiceCompleteFn complete_ = nullptr;
		RetiredServiceReleaseFn release_ = nullptr;
	};

	struct RetiredSessionBatch {
		oavk::Stream* stream = nullptr;
		oa::Event completion;
		oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> graphs;
	};
	enum class GraphicsStreamSlotState : oa::U8 {
		Free,
		Recording,
		Submitted,
		Retired,
		Quarantined,
	};
	struct GraphicsStreamSlot {
		oa::UniquePtr<oavk::Stream> stream;
		GraphicsStreamSlotState state = GraphicsStreamSlotState::Free;
		oa::U64 generation = 0;
		oa::Event completion;
	};

	oa::UniquePtr<oa::ExecutionSession> session_;
	oa::UniquePtr<oa::Log> logger_;
	oa::LogSelection previousLogSelection_;
	oa::SharedPtr<oa::TimerRegistry> timerRegistry_;
	oavk::Device device_;
	OaVma allocator_;
	oavk::BindlessHeap bindless_;
	oa::PipelineRegistry pipelines_;
	oa::UniquePtr<GemmState> gemmState_;
	oa::EngineState state_ = oa::EngineState::Empty;
	oa::SharedPtr<BufferLeaseRegistry> bufferLeaseRegistry_;
	oa::Precision precision_ = oa::Precision::FP32;
	oa::MemoryPlacement matrixPlacement_ = oa::MemoryPlacement::HostUpload;
	mutable oa::Atomic<oa::U64> gemmCapsMask_{0};

	oa::Vec<oa::UniquePtr<oavk::Stream>> streamPool_;
	oa::Vec<oa::U32> freeStack_;
	oa::Spinlock streamPoolLock_;
	oa::Vec<oa::RetiredImageDispatch> retiredImageDispatches_;
	oa::Mutex retiredImageDispatchMutex_;
	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> retiredExecutionPlans_;
	oa::Mutex retiredExecutionPlanMutex_;
	oa::Vec<RetiredSessionBatch> retiredSessionBatches_;
	oa::Mutex retiredSessionBatchMutex_;
	oa::Vec<oa::UniquePtr<oa::RetiredUploadRing>> retiredUploadRings_;
	oa::Mutex retiredUploadRingMutex_;
	oa::Vec<oa::UniquePtr<oa::RetiredPresenter>> retiredPresenters_;
	oa::Mutex retiredPresenterMutex_;
	oa::Vec<RetiredServiceState> retiredBorrowedServices_;
	oa::Mutex retiredBorrowedServiceMutex_;

	oa::Vec<oa::UniquePtr<oavk::Stream>> asyncStreamPool_;
	oa::Vec<oa::U32> asyncFreeStack_;
	oa::Spinlock asyncStreamPoolLock_;
	oa::Vec<GraphicsStreamSlot> graphicsStreamPool_;
	oa::Mutex graphicsStreamPoolMutex_;

	oavk::Stream transferStream_;
	oavk::Stream readbackStream_;
	oavk::Buffer readbackStaging_;
	oa::UniquePtr<oa::UploadRing> uploadRing_;
	oa::Mutex computeQueueMutex_;
	oa::Mutex asyncComputeQueueMutex_;
	oa::Mutex transferQueueMutex_;
	oa::Mutex graphicsQueueMutex_;
	oa::Mutex presentQueueMutex_;
	oa::Mutex transferStreamMutex_;
	oa::Mutex uploadRingMutex_;
	oa::Mutex readbackMutex_;
	oa::Mutex hostVisibleBufferCacheMutex_;
	oa::Vec<oa::HostVisibleBufferCacheEntry> hostVisibleBufferCache_;
	oa::U64 hostVisibleBufferCacheBytes_ = 0;
};
