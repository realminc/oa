#include "graphicsStream.h"

#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/stream.h>
#include "engine/engineAccess.h"

namespace oa {

oa::Result<GraphicsStreamLease> GraphicsStreamLease::acquire(
	oa::Engine& inEngine)
{
	if (not inEngine.isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics stream acquisition requires a ready engine");
	}
	auto& impl = oa::EngineAccess::get(inEngine);
	if (impl.device_.queues.graphicsQueue == nullptr
		or impl.device_.queues.graphicsQueueFamily
			== oavk::EnumerationIndexUnset) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"engine was not created with graphics capability");
	}

	oa::EngineAccess(inEngine).collectRetiredGraphicsStreams();
	oa::ScopedLock lock(impl.graphicsStreamPoolMutex_);
	oa::U32 slotIndex = static_cast<oa::U32>(
		impl.graphicsStreamPool_.size());
	for (oa::U32 index = 0;
		index < static_cast<oa::U32>(impl.graphicsStreamPool_.size()); ++index) {
		if (impl.graphicsStreamPool_[index].state
			== oa::EngineAccess::Impl::GraphicsStreamSlotState::Free) {
			slotIndex = index;
			break;
		}
	}

	if (slotIndex == static_cast<oa::U32>(
			impl.graphicsStreamPool_.size())) {
		auto created = oavk::Stream::create(
			impl.device_,
			impl.device_.queues.graphicsQueueFamily,
			impl.device_.queues.graphicsQueue);
		if (not created.isOk()) return created.getStatus();
		oa::EngineAccess::Impl::GraphicsStreamSlot slot;
		slot.stream = oa::makeUnique<oavk::Stream>(oa::move(*created));
		impl.graphicsStreamPool_.pushBack(oa::move(slot));
	}

	auto& slot = impl.graphicsStreamPool_[slotIndex];
	if (not slot.stream) {
		slot.state = oa::EngineAccess::Impl::GraphicsStreamSlotState::Quarantined;
		return oa::Status::error(
			oa::StatusCode::Internal,
			"graphics stream pool contains an empty slot");
	}
	const oa::Status beginStatus = slot.stream->begin(impl.device_);
	if (not beginStatus.isOk()) {
		slot.state = oa::EngineAccess::Impl::GraphicsStreamSlotState::Quarantined;
		return beginStatus;
	}
	++slot.generation;
	if (slot.generation == 0) ++slot.generation;
	slot.completion = {};
	slot.state = oa::EngineAccess::Impl::GraphicsStreamSlotState::Recording;
	return GraphicsStreamLease(inEngine, slotIndex, slot.generation);
}

GraphicsStreamLease::GraphicsStreamLease(
	oa::Engine& inEngine,
	oa::U32 inSlot,
	oa::U64 inGeneration) noexcept
	: engine_(&inEngine)
	, slot_(inSlot)
	, generation_(inGeneration)
{}

GraphicsStreamLease::GraphicsStreamLease(
	GraphicsStreamLease&& inOther) noexcept
	: engine_(inOther.engine_)
	, slot_(inOther.slot_)
	, generation_(inOther.generation_)
{
	inOther.reset_();
}

GraphicsStreamLease& GraphicsStreamLease::operator=(
	GraphicsStreamLease&& inOther) noexcept
{
	if (this == &inOther) return *this;
	if (engine_ != nullptr) {
		const oa::Status status = close();
		if (not status.isOk()) {
			OaLogError(oa::LogComponent::Runtime,
				"graphics stream lease move-close failed: %s",
				status.toString().cStr());
		}
	}
	engine_ = inOther.engine_;
	slot_ = inOther.slot_;
	generation_ = inOther.generation_;
	inOther.reset_();
	return *this;
}

GraphicsStreamLease::~GraphicsStreamLease()
{
	if (engine_ == nullptr) return;
	const oa::Status status = close();
	if (not status.isOk()) {
		OaLogError(oa::LogComponent::Runtime,
			"graphics stream lease abandonment failed: %s",
			status.toString().cStr());
	}
}

bool GraphicsStreamLease::isValid() const noexcept
{
	return engine_ != nullptr and generation_ != 0;
}

oavk::Stream* GraphicsStreamLease::getStream() noexcept
{
	return engine_ != nullptr
		? oa::EngineAccess(*engine_).graphicsStreamForLease(slot_, generation_)
		: nullptr;
}

oa::Result<oa::Event> GraphicsStreamLease::submit()
{
	return submit(oa::Span<const oa::Event>{});
}

oa::Result<oa::Event> GraphicsStreamLease::submit(
	oa::Span<const oa::Event> inDependencies)
{
	if (engine_ == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	auto result = oa::EngineAccess(*engine_).submitGraphicsStream(
		slot_, generation_, inDependencies);
	return result;
}

oa::Status GraphicsStreamLease::cancel()
{
	if (engine_ == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	const oa::Status status = oa::EngineAccess(*engine_).cancelGraphicsStream(
		slot_, generation_);
	if (status.isOk()) reset_();
	return status;
}

oa::Status GraphicsStreamLease::recycle(const oa::Event& inCompletion)
{
	if (engine_ == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics stream lease is closed");
	}
	const oa::Status status = oa::EngineAccess(*engine_).recycleGraphicsStream(
		slot_, generation_, inCompletion);
	if (status.isOk()) reset_();
	return status;
}

oa::Status GraphicsStreamLease::close()
{
	if (engine_ == nullptr) return oa::Status::ok();
	oa::Engine* engine = engine_;
	const oa::U32 slot = slot_;
	const oa::U64 generation = generation_;
	reset_();
	return oa::EngineAccess(*engine).abandonGraphicsStream(slot, generation);
}

void GraphicsStreamLease::reset_() noexcept
{
	engine_ = nullptr;
	slot_ = 0;
	generation_ = 0;
}

} // namespace oa
