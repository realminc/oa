#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

class EventAccess;

// Cheap non-owning completion value shared by compute, image, video, capture,
// rendering, and training. It is a GPU dependency first and a host wait handle
// second. queue submission consumes it through private Runtime lowering; public
// callers wait only at a deliberate CPU boundary.
class Event {
public:
	static constexpr oa::U32 UnknownQueueFamily = UINT32_MAX;

	Event() = default;

	[[nodiscard]] bool isValid() const noexcept {
		return device_ != nullptr and semaphore_ != nullptr and value_ != 0U;
	}
	[[nodiscard]] oa::Status wait(oa::U64 inTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] oa::Bool isComplete() const;
	[[nodiscard]] oa::U64 value() const noexcept { return value_; }
	[[nodiscard]] bool hasQueueFamily() const noexcept {
		return queueFamily_ != UnknownQueueFamily;
	}
	// queue-family provenance is synchronization metadata, not a vulkan handle.
	// A different family still requires an explicit exclusive-resource
	// release/acquire ownership transfer in addition to an event dependency.
	[[nodiscard]] oa::U32 queueFamily() const noexcept { return queueFamily_; }
	[[nodiscard]] bool isSameCompletion(const oa::Event& inOther) const noexcept {
		return isValid() and inOther.isValid()
			and device_ == inOther.device_
			and semaphore_ == inOther.semaphore_
			and value_ == inOther.value_
			and queueFamily_ == inOther.queueFamily_;
	}

private:
	friend class EventAccess;

	const void* device_ = nullptr;
	void* semaphore_ = nullptr;
	oa::U64 value_ = 0U;
	oa::U32 queueFamily_ = UnknownQueueFamily;
};

} // namespace oa
