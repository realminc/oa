#pragma once

#include <oa/runtime/sync.h>

namespace oa {

class EventAccess {
public:
	[[nodiscard]] static oa::Event create(
		const oavk::Device& inDevice,
		const oavk::TimelineSemaphore& inSemaphore,
		oa::U64 inValue,
		oa::U32 inQueueFamily = oa::Event::UnknownQueueFamily) noexcept
	{
		oa::Event event;
		event.device_ = &inDevice;
		event.semaphore_ = inSemaphore.semaphore;
		event.value_ = inValue;
		event.queueFamily_ = inQueueFamily;
		return event;
	}

	[[nodiscard]] static oavk::TimelineWait timelineWait(
		const oa::Event& inEvent) noexcept
	{
		return {inEvent.semaphore_, inEvent.value_};
	}

	[[nodiscard]] static void* semaphoreHandle(
		const oa::Event& inEvent) noexcept
	{
		return inEvent.semaphore_;
	}

	[[nodiscard]] static oavk::TimelineSemaphore timelineSemaphore(
		const oa::Event& inEvent) noexcept
	{
		return {.semaphore = inEvent.semaphore_};
	}

	[[nodiscard]] static const void* deviceIdentity(
		const oa::Event& inEvent) noexcept
	{
		return inEvent.device_;
	}
};

} // namespace oa
