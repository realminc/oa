#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/event.h>

namespace oavk {

class Device;

class Fence {
public:
	void* fence = nullptr;

	[[nodiscard]] static oa::Result<Fence> create(
		const oavk::Device& inDevice, oa::Bool inSignaled = false);
	void destroy(const oavk::Device& inDevice);
	[[nodiscard]] oa::Status wait(
		const oavk::Device& inDevice, oa::U64 inTimeoutNs = UINT64_MAX);
	[[nodiscard]] oa::Bool isSignaled(const oavk::Device& inDevice) const;
	void reset(const oavk::Device& inDevice);
};

class TimelineSemaphore {
public:
	void* semaphore = nullptr;

	[[nodiscard]] static oa::Result<TimelineSemaphore> create(
		const oavk::Device& inDevice, oa::U64 inInitialValue = 0);
	void destroy(const oavk::Device& inDevice);
	[[nodiscard]] oa::Status wait(
		const oavk::Device& inDevice,
		oa::U64 inValue,
		oa::U64 inTimeoutNs = UINT64_MAX) const;
	[[nodiscard]] oa::U64 getValue(const oavk::Device& inDevice) const;
};

// Private GPU-side timeline dependency consumed by queue lowering.
struct TimelineWait {
	void* semaphore = nullptr;
	oa::U64 value = 0;
};

} // namespace oavk
