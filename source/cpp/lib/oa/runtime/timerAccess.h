#pragma once

#include <oa/core/status.h>
#include <oa/runtime/sync.h>

namespace oavk { class Stream; }

namespace oa {

class Timer;

// Private bridge from submission lowering to the universal public timer. Raw
// command streams and backend handles never enter the installed API.
class TimerAccess {
public:
	[[nodiscard]] static oa::Status beginDevice(
		oa::Timer& inTimer, oavk::Stream* inStream);
	[[nodiscard]] static oa::Status endDevice(
		oa::Timer& inTimer, oavk::Stream* inStream);
	[[nodiscard]] static oa::Status attachCompletion(
		oa::Timer& inTimer, const oa::Event& inCompletion);
	[[nodiscard]] static oa::Status completeSynchronously(oa::Timer& inTimer);
	static void cancelDevice(oa::Timer& inTimer) noexcept;
};

} // namespace oa
