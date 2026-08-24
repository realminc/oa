#include <oa/vision/detection.h>

#include <oa/core/memory.h>
#include <oa/runtime/buffer.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/event.h>

#include "../ui/consumedBufferRetirement.h"

struct oa::DetectionBuffer::Impl {
	oa::Engine* runtime = nullptr;
	oavk::Buffer buffer;
	oa::Event consumerCompletion;
	oa::U32 count = 0;
	oa::U32 capacity = 0;
};

oa::DetectionBuffer::DetectionBuffer(oa::DetectionBuffer&& inOther) noexcept
	: impl_(oa::move(inOther.impl_)) {}

oa::DetectionBuffer& oa::DetectionBuffer::operator=(oa::DetectionBuffer&& inOther) noexcept {
	if (this != &inOther) {
		reset_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}

oa::DetectionBuffer::~DetectionBuffer() {
	reset_();
}

oa::Result<oa::DetectionBuffer> oa::DetectionBuffer::createHostUpload(
	oa::Engine& inRuntime,
	oa::U32 inCapacity) {
	if (inCapacity == 0) {
		return oa::Status::invalidArgument(
			"oa::DetectionBuffer::createHostUpload: capacity must be non-zero");
	}

	auto allocation = oa::EngineResourceAccess::allocBuffer(inRuntime,
		static_cast<oa::U64>(inCapacity) * sizeof(oa::Detection));
	if (!allocation.isOk()) return allocation.getStatus();

	oa::DetectionBuffer result;
	result.impl_ = oa::makeUnique<Impl>();
	result.impl_->runtime = &inRuntime;
	result.impl_->buffer = oa::move(*allocation);
	result.impl_->capacity = inCapacity;
	return result;
}

void oa::DetectionBuffer::reset_() noexcept {
	if (not impl_) return;
	oa::ConsumedBufferRetirement::releaseOrRetire(
		impl_->runtime, impl_->buffer, impl_->consumerCompletion);
	impl_.reset();
}

oa::Status oa::DetectionBuffer::upload(oa::Span<const oa::Detection> inDetections) {
	if (not impl_ or not impl_->runtime or !impl_->buffer.buffer
		or !impl_->buffer.mappedPtr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::DetectionBuffer::upload: buffer is not host-visible");
	}
	if (!isReady()) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::DetectionBuffer::upload: buffer is still consumed by the GPU");
	}
	if (inDetections.size() > impl_->capacity) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::DetectionBuffer::upload: detection count exceeds capacity");
	}

	const oa::Usize bytes = inDetections.size() * sizeof(oa::Detection);
	if (bytes > 0) {
		oa::memcpy(impl_->buffer.mappedPtr, inDetections.data(), bytes);
	}
	impl_->count = static_cast<oa::U32>(inDetections.size());
	impl_->consumerCompletion = {};
	return oa::Status::ok();
}

oa::Status oa::DetectionBuffer::markConsumed(const oa::Event& inCompletion) {
	if (not impl_ or impl_->runtime == nullptr or not isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::DetectionBuffer::markConsumed requires a valid buffer");
	}
	if (not impl_->runtime->ownsEvent(inCompletion)) {
		return oa::Status::invalidArgument(
			"oa::DetectionBuffer::markConsumed requires an event from its engine");
	}
	impl_->consumerCompletion = inCompletion.isComplete()
		? oa::Event{}
		: inCompletion;
	return oa::Status::ok();
}

bool oa::DetectionBuffer::isReady() const {
	if (not impl_ or not impl_->runtime or not isValid()) return false;
	return not impl_->consumerCompletion.isValid()
		or impl_->consumerCompletion.isComplete();
}

bool oa::DetectionBuffer::isValid() const noexcept {
	return impl_ and impl_->buffer.buffer != nullptr;
}

oa::U32 oa::DetectionBuffer::count() const noexcept {
	return impl_ ? impl_->count : 0U;
}

oa::U32 oa::DetectionBuffer::capacity() const noexcept {
	return impl_ ? impl_->capacity : 0U;
}

oa::U32 oa::DetectionBuffer::bindlessIndex() const noexcept {
	return impl_ ? impl_->buffer.bindlessIndex : OA_BINDLESS_INVALID;
}
