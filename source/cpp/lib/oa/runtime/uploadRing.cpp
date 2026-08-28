#include <oa/runtime/uploadRing.h>

#include <oa/core/std/memory.h>
#include <oa/core/log.h>
#include <oa/core/std.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include "engine/engineAccess.h"
#include <oa/runtime/stream.h>

#include "uploadRingRetirement.h"

namespace {

struct PendingUploadCopy {
	oavk::Buffer dst;
	oavk::BufferCopyRegion region;
};

} // namespace

struct oa::UploadRing::Impl {
	oa::Engine* engine = nullptr;
	oavk::Buffer staging;
	oa::UploadRingConfig config;
	oa::U64 frameCapacity = 0;
	oa::U32 nextFrame = 0;
	oa::U32 activeFrame = 0;
	oa::Bool batchOpen = false;
	oa::Event lastCompletion;
	oa::Vector<oa::UniquePtr<oa::UploadFrame>> frames;
	oa::Vector<PendingUploadCopy> copies;
};

oa::UploadRing::UploadRing(oa::UploadRing&&) noexcept = default;
oa::UploadRing& oa::UploadRing::operator=(oa::UploadRing&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}
oa::UploadRing::~UploadRing() { abandon_(); }

oa::Result<oa::UploadRing> oa::UploadRing::create(oa::Engine& inEngine,	const oa::UploadRingConfig& inConfig) {
	if (inConfig.capacityBytes == 0 || inConfig.framesInFlight < 2 || inConfig.alignment == 0) {
		return oa::Status::invalidArgument("UploadRing: capacity/alignment must be non-zero and frames must be >= 2");
	}

	const oa::U64 frameCapacity =	(inConfig.capacityBytes / inConfig.framesInFlight / inConfig.alignment)	* inConfig.alignment;
	if (frameCapacity == 0) {
		return oa::Status::invalidArgument("UploadRing: capacity is too small for frame count");
	}

	oa::UploadRing result;
	result.impl_ = oa::makeUnique<Impl>();
	auto& impl = *result.impl_;
	impl.engine = &inEngine;
	impl.config = inConfig;
	impl.frameCapacity = frameCapacity;
	impl.config.capacityBytes = frameCapacity * inConfig.framesInFlight;

	auto staging = oa::EngineResourceAccess::allocBuffer(inEngine, impl.config.capacityBytes);
	if (!staging) return staging.getStatus();
	impl.staging = oa::move(*staging);
	if (impl.staging.mappedPtr == nullptr) {
		oa::EngineResourceAccess::freeBuffer(inEngine, impl.staging);
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing: staging allocation is not mapped");
	}

	impl.frames.reserve(inConfig.framesInFlight);
	// OA buffers are currently exclusive-sharing allocations. A dedicated
	// transfer family would therefore require an explicit release/acquire
	// ownership pair in every downstream consumer. Until that contract is part
	// of the graph, use the compute queue when the families differ. This keeps
	// uploads asynchronous without relying on undefined cross-family ownership.
	const bool sharedQueueFamily =
		oa::EngineDeviceAccess::get(inEngine).queues.transferQueueFamily
			== oa::EngineDeviceAccess::get(inEngine).queues.computeQueueFamily;
	const oa::U32 queueFamily = sharedQueueFamily
		? oa::EngineDeviceAccess::get(inEngine).queues.transferQueueFamily
		: oa::EngineDeviceAccess::get(inEngine).queues.computeQueueFamily;
	void* queue = sharedQueueFamily
		? oa::EngineDeviceAccess::get(inEngine).queues.transferQueue
		: oa::EngineDeviceAccess::get(inEngine).queues.computeQueue;
	for (oa::U32 index = 0; index < inConfig.framesInFlight; ++index) {
		auto stream = oavk::Stream::create(
			oa::EngineDeviceAccess::get(inEngine),
			queueFamily,
			queue);
		if (!stream) {
			// No work has been submitted yet. Close the partial ring through the
			// same status-bearing lifecycle boundary; preserve the initiating
			// stream-creation failure as the returned error.
			(void)result.close();
			return stream.getStatus();
		}
		auto frame = oa::makeUnique<oa::UploadFrame>();
		frame->stream = oa::move(*stream);
		frame->begin = static_cast<oa::U64>(index) * frameCapacity;
		frame->end = frame->begin + frameCapacity;
		frame->cursor = frame->begin;
		impl.frames.pushBack(oa::move(frame));
	}
	return result;
}

oa::Status oa::UploadRing::beginBatch() {
	if (!impl_ || !impl_->engine) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing is not initialized");
	}
	if (impl_->batchOpen) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing batch is already open");
	}
	impl_->activeFrame = impl_->nextFrame % impl_->config.framesInFlight;
	auto& frame = *impl_->frames[impl_->activeFrame];
	OA_RETURN_IF_ERROR(frame.stream.begin(oa::EngineDeviceAccess::get(*impl_->engine)));
	frame.cursor = frame.begin;
	impl_->copies.clear();
	impl_->batchOpen = true;
	return oa::Status::ok();
}

oa::Result<oa::UploadSlice> oa::UploadRing::reserve(oa::U64 inSize, oa::U64 inAlignment) {
	if (!impl_ || !impl_->batchOpen) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	if (inSize == 0) return oa::Status::invalidArgument("UploadRing: zero-byte reservation");
	const oa::U64 alignment = inAlignment == 0 ? impl_->config.alignment : inAlignment;
	auto& frame = *impl_->frames[impl_->activeFrame];
	const oa::U64 offset = alignUp(frame.cursor, oa::max<oa::U64>(alignment, 4));
	if (offset > frame.end || inSize > frame.end - offset) {
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"UploadRing: active frame capacity exhausted; submit or increase capacity");
	}
	frame.cursor = offset + inSize;
	auto* base = static_cast<oa::Byte*>(impl_->staging.mappedPtr);
	return oa::UploadSlice{.mapped = base + offset, .offset = offset, .size = inSize};
}

oa::Status oa::UploadRing::enqueueCopy(
	const oa::UploadSlice& inSlice,
	const oavk::Buffer& inDst,
	oa::U64 inDstOffset
) {
	if (!impl_ || !impl_->batchOpen) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	const auto& frame = *impl_->frames[impl_->activeFrame];
	auto* base = static_cast<oa::Byte*>(impl_->staging.mappedPtr);
	if (!inSlice.isValid() || inDst.buffer == nullptr
		|| inSlice.offset < frame.begin || inSlice.offset > frame.cursor
		|| inSlice.size > frame.cursor - inSlice.offset
		|| inSlice.mapped != base + inSlice.offset
		|| inDstOffset > inDst.size || inSlice.size > inDst.size - inDstOffset) {
		return oa::Status::invalidArgument("UploadRing: invalid copy range");
	}
	if ((inSlice.offset & 3U) != 0 || (inDstOffset & 3U) != 0) {
		return oa::Status::invalidArgument("UploadRing: vulkan buffer-copy offsets must be 4-byte aligned");
	}
	impl_->copies.pushBack(PendingUploadCopy{
		.dst = inDst,
		.region = {
			.srcOffset = inSlice.offset,
			.dstOffset = inDstOffset,
			.size = inSlice.size,
		},
	});
	return oa::Status::ok();
}

oa::Status oa::UploadRing::upload(
	const oavk::Buffer& inDst,
	oa::U64 inDstOffset,
	const void* inData,
	oa::U64 inSize,
	oa::U64 inAlignment
) {
	if (inData == nullptr) return oa::Status::invalidArgument("UploadRing: source is null");
	auto slice = reserve(inSize, inAlignment);
	if (!slice) return slice.getStatus();
	// The staging slice is published to a GPU transfer and is not consumed by
	// the CPU. memcpyStream selects the qualified non-temporal window and falls
	// back to the ordinary platform path outside it.
	oa::memcpyStream(slice->mapped, inData, static_cast<oa::Usize>(inSize));
	return enqueueCopy(*slice, inDst, inDstOffset);
}

oa::Result<oa::Event> oa::UploadRing::submit() {
	if (!impl_ || !impl_->batchOpen) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "UploadRing batch is not open");
	}
	auto& frame = *impl_->frames[impl_->activeFrame];
	if (!impl_->copies.empty() && !oa::EngineAllocatorAccess::get(
		*impl_->engine).flushHostBuffer(
		impl_->staging, frame.begin, frame.cursor - frame.begin)) {
		return oa::Status::error(oa::StatusCode::VulkanError, "UploadRing: mapped flush failed");
	}

	// Collapse adjacent jobs targeting the same destination into one
	// vkCmdCopyBuffer call with multiple regions. Different destinations still
	// share this command buffer and this single queue submission.
	oa::Usize begin = 0;
	while (begin < impl_->copies.size()) {
		const oavk::Buffer& dst = impl_->copies[begin].dst;
		oa::Vector<oavk::BufferCopyRegion> regions;
		oa::U64 barrierBegin = UINT64_MAX;
		oa::U64 barrierEnd = 0U;
		oa::Usize end = begin;
		while (end < impl_->copies.size()
			&& impl_->copies[end].dst.buffer == dst.buffer) {
			const oavk::BufferCopyRegion& region = impl_->copies[end].region;
			regions.pushBack(region);
			barrierBegin = oa::min(barrierBegin, region.dstOffset);
			barrierEnd = oa::max(barrierEnd, region.dstOffset + region.size);
			++end;
		}
		frame.stream.recordCopyBufferRegions(
			impl_->staging, dst,
			oa::Span<const oavk::BufferCopyRegion>(regions.data(), regions.size())
		);
		frame.stream.recordTransferWriteBarrier(dst, barrierBegin, barrierEnd - barrierBegin);
		begin = end;
	}
	// ring arenas protect staging reuse, but destination buffers can repeat
	// across adjacent batches. queue order alone is not a vulkan memory
	// dependency, so chain submissions through the previous upload completion.
	// This stays GPU-side and preserves host asynchrony.
	if (impl_->lastCompletion.isValid()) {
		const oavk::TimelineWait wait =
			oa::EventAccess::timelineWait(impl_->lastCompletion);
		OA_RETURN_IF_ERROR(frame.stream.submitWithDependencies(*impl_->engine, oa::Span<const oavk::TimelineWait>(&wait, 1)));
	} else {
		OA_RETURN_IF_ERROR(frame.stream.submit(*impl_->engine));
	}
	impl_->lastCompletion = frame.stream.completion(oa::EngineDeviceAccess::get(*impl_->engine));
	impl_->batchOpen = false;
	impl_->copies.clear();
	++impl_->nextFrame;
	return impl_->lastCompletion;
}

oa::Status oa::UploadRing::wait() {
	return impl_ && impl_->lastCompletion.isValid()
		? impl_->lastCompletion.wait()
		: oa::Status::ok();
}

oa::Status oa::UploadRing::close() {
	if (!impl_) return oa::Status::ok();
	oa::Status firstError = oa::Status::ok();
	const auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() && !inStatus.isOk()) firstError = inStatus;
	};
	if (impl_->engine && oa::EngineDeviceAccess::get(*impl_->engine).device) {
		if (impl_->batchOpen) {
			auto& active = *impl_->frames[impl_->activeFrame];
			retainError(active.stream.resetUnsubmitted(oa::EngineDeviceAccess::get(*impl_->engine)));
			impl_->batchOpen = false;
			impl_->copies.clear();
		}
		for (auto& frame : impl_->frames) {
			if (!frame) continue;
			retainError(frame->stream.synchronize(oa::EngineDeviceAccess::get(*impl_->engine)));
			frame->stream.destroy(oa::EngineDeviceAccess::get(*impl_->engine));
		}
		oa::EngineResourceAccess::freeBuffer(*impl_->engine, impl_->staging);
	}
	impl_.reset();
	return firstError;
}

void oa::UploadRing::abandon_() noexcept {
	if (!impl_) return;
	auto* engine = impl_->engine;
	if (engine == nullptr || oa::EngineDeviceAccess::get(*engine).device == nullptr) {
		impl_.reset();
		return;
	}
	if (impl_->batchOpen) {
		auto& active = *impl_->frames[impl_->activeFrame];
		if (const auto status = active.stream.resetUnsubmitted(oa::EngineDeviceAccess::get(*engine));
			not status.isOk())
		{
			OaLogError(
				oa::LogComponent::Runtime,
				"oa::UploadRing abandonment failed to cancel open batch: %s",
				status.getMessage().cStr()
			);
		}
		impl_->batchOpen = false;
		impl_->copies.clear();
	}

	auto retired = oa::makeUnique<oa::RetiredUploadRing>();
	retired->staging = oa::move(impl_->staging);
	retired->frames = oa::move(impl_->frames);
	oa::EngineAccess(*engine).retireUploadRing(oa::move(retired));
	impl_.reset();
}

oa::U64 oa::UploadRing::capacityBytes() const noexcept {
	return impl_ ? impl_->config.capacityBytes : 0;
}
oa::U64 oa::UploadRing::frameCapacityBytes() const noexcept {
	return impl_ ? impl_->frameCapacity : 0;
}
oa::U64 oa::UploadRing::bytesUsed() const noexcept {
	if (!impl_ || !impl_->batchOpen) return 0;
	const auto& frame = *impl_->frames[impl_->activeFrame];
	return frame.cursor - frame.begin;
}
oa::U32 oa::UploadRing::pendingCopyCount() const noexcept {
	return impl_ ? static_cast<oa::U32>(impl_->copies.size()) : 0;
}
bool oa::UploadRing::isBatchOpen() const noexcept {
	return impl_ && impl_->batchOpen;
}
