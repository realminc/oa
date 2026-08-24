#pragma once

#include "engineImpl.h"

namespace oa {

// Sole private authority for oa::Engine implementation state and executable
// lowering services. The installed owner exposes semantic execution and
// capability observations only; Runtime implementation units borrow this
// short-lived access object instead of becoming friends of the public class.
class EngineAccess {
public:
	using Impl = Engine::Impl;
	using RetiredServiceCompleteFn = Impl::RetiredServiceCompleteFn;
	using RetiredServiceReleaseFn = Impl::RetiredServiceReleaseFn;

	explicit EngineAccess(Engine& inEngine) noexcept
		: engine_(inEngine)
		, impl_(inEngine.impl_.get())
	{}

	[[nodiscard]] static Impl& get(oa::Engine& inEngine) noexcept {
		return *inEngine.impl_;
	}

	[[nodiscard]] static const Impl& get(
		const oa::Engine& inEngine) noexcept
	{
		return *inEngine.impl_;
	}

	void selectActiveSession();
	void clearActiveSession();
	void retireImageDispatch(oa::RetiredImageDispatch&& inRetired);
	void collectRetiredImageDispatches();
	void retireExecutionPlan(oa::UniquePtr<oa::ExecutableGraph>&& inGraph);
	void collectRetiredExecutionPlans();
	[[nodiscard]] oa::Status completeRetiredExecutionPlans();
	void retireSessionBatch(
		oavk::Stream* inStream,
		const oa::Event& inCompletion,
		oa::Vec<oa::UniquePtr<oa::ExecutableGraph>>&& inGraphs);
	void collectRetiredSessionBatches();
	[[nodiscard]] oa::Status completeRetiredSessionBatches();
	[[nodiscard]] oavk::Stream* graphicsStreamForLease(
		oa::U32 inSlot, oa::U64 inGeneration) noexcept;
	[[nodiscard]] oa::Result<oa::Event> submitGraphicsStream(
		oa::U32 inSlot,
		oa::U64 inGeneration,
		oa::Span<const oa::Event> inDependencies);
	[[nodiscard]] oa::Status cancelGraphicsStream(
		oa::U32 inSlot, oa::U64 inGeneration);
	[[nodiscard]] oa::Status recycleGraphicsStream(
		oa::U32 inSlot, oa::U64 inGeneration, const oa::Event& inCompletion);
	[[nodiscard]] oa::Status abandonGraphicsStream(
		oa::U32 inSlot, oa::U64 inGeneration);
	void collectRetiredGraphicsStreams();
	[[nodiscard]] oa::Status completeGraphicsStreams();
	[[nodiscard]] void* queueSubmitMutex(void* inQueue) noexcept;
	void lockQueueSubmit(void* inQueue);
	void unlockQueueSubmit(void* inQueue);
	void retireUploadRing(oa::UniquePtr<oa::RetiredUploadRing>&& inRing);
	[[nodiscard]] oa::Status completeRetiredUploadRings();
	void retirePresenter(oa::UniquePtr<oa::RetiredPresenter>&& inPresenter);
	[[nodiscard]] oa::Status completeRetiredPresenters();
	void retireBorrowedService(
		void* inPayload,
		RetiredServiceCompleteFn inComplete,
		RetiredServiceReleaseFn inRelease);
	[[nodiscard]] oa::Status completeRetiredBorrowedServices();
	void detachRetiredBorrowedServices() noexcept;
	[[nodiscard]] oa::SharedPtr<oavk::Buffer> adoptBufferLease(
		oavk::Buffer&& inBuffer,
		oa::SharedPtr<oavk::Buffer> inBacking = {});

	[[nodiscard]] oa::Result<oavk::Buffer> allocBuffer(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocBuffer(
		oa::U64 inSize, oa::MemoryPlacement inPlacement);
	[[nodiscard]] oa::Result<oavk::Buffer> allocBufferDevice(oa::U64 inSize);
	[[nodiscard]] oa::Result<oavk::Buffer> allocBufferBar(oa::U64 inSize);
	[[nodiscard]] oa::Status uploadBuffer(
		const oavk::Buffer& inDst,
		oa::U64 inDstOffset,
		const void* inData,
		oa::U64 inSize);
	[[nodiscard]] oa::Status readbackBuffer(
		const oavk::Buffer& inSrc,
		oa::U64 inSrcOffset,
		void* outData,
		oa::U64 inSize);
	void freeBuffer(oavk::Buffer& inOutBuffer);
	[[nodiscard]] oa::Result<oa::Event> copyBufferAsync(
		const oavk::Buffer& inSrc,
		const oavk::Buffer& inDst,
		oa::U64 inSize);
	[[nodiscard]] static oa::U64 gemmCapsMask(const oa::Engine& inEngine);
	void logSelectedDevices();
	[[nodiscard]] oa::Status ensurePipeline(
		oa::StringView inName,
		oa::Span<const oa::U8> inSpirv,
		const oa::PipelineSpec& inSpec);
	[[nodiscard]] oavk::Stream* acquireStream();
	void releaseStream(oavk::Stream* inStream);
	[[nodiscard]] oa::U32 registerBuffer(oavk::Buffer& inOutBuffer);
	[[nodiscard]] oa::Status updateBufferDescriptor(const oavk::Buffer& inBuffer);
	void deregisterBuffer(oavk::Buffer& inOutBuffer);
	[[nodiscard]] oa::Status ensureAllEmbeddedLiboaPipelines();
	[[nodiscard]] oavk::Stream* acquireAsyncStream();
	void releaseAsyncStream(oavk::Stream* inStream);
	[[nodiscard]] oa::Status submitToQueue(
		void* inQueue, void* inSubmitInfo, void* inFence);
	[[nodiscard]] oa::Status submitToQueue2(
		void* inQueue, const void* inSubmitInfo);
	[[nodiscard]] oa::Status initialize(const oa::EngineConfig& inConfig);
	[[nodiscard]] oa::Status initializeImpl(const oa::EngineConfig& inConfig);

private:
	oa::Engine& engine_;
	Impl* impl_ = nullptr;
};

} // namespace oa
