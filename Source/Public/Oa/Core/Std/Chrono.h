#pragma once

// OaStd — std::chrono aliases and helpers (thin naming layer; types remain std).
// Monotonic intervals: OaSteadyClock / OaSteadyNow (matches OaTimestamp in time.h).
// Bench-oriented wall timer: OaHighResolutionClock / OaHighResolutionNow.

#include <chrono>
#include <cstdint>

using OaSteadyClock = std::chrono::steady_clock;
using OaSteadyTimePoint = OaSteadyClock::time_point;
using OaSteadyDuration = OaSteadyClock::duration;

using OaSystemClock = std::chrono::system_clock;
using OaSystemTimePoint = OaSystemClock::time_point;
using OaSystemDuration = OaSystemClock::duration;

using OaHighResolutionClock = std::chrono::high_resolution_clock;
using OaHighResolutionTimePoint = OaHighResolutionClock::time_point;
using OaHighResolutionDuration = OaHighResolutionClock::duration;

[[nodiscard]] inline OaSteadyTimePoint OaSteadyNow() noexcept {
	return OaSteadyClock::now();
}

[[nodiscard]] inline OaSystemTimePoint OaSystemNow() noexcept {
	return OaSystemClock::now();
}

[[nodiscard]] inline OaHighResolutionTimePoint OaHighResolutionNow() noexcept {
	return OaHighResolutionClock::now();
}

template <typename Rep, typename Period>
[[nodiscard]] inline double OaChronoToSeconds(std::chrono::duration<Rep, Period> InD) noexcept {
	return std::chrono::duration<double>(InD).count();
}

template <typename Rep, typename Period>
[[nodiscard]] inline double OaChronoToMilli(std::chrono::duration<Rep, Period> InD) noexcept {
	using namespace std::chrono;
	return duration<double, std::milli>(InD).count();
}

template <typename Rep, typename Period>
[[nodiscard]] inline double OaChronoToMicro(std::chrono::duration<Rep, Period> InD) noexcept {
	using namespace std::chrono;
	return duration<double, std::micro>(InD).count();
}

template <typename Rep, typename Period>
[[nodiscard]] inline std::int64_t OaChronoMillisCount(std::chrono::duration<Rep, Period> InD) noexcept {
	return std::chrono::duration_cast<std::chrono::milliseconds>(InD).count();
}

template <typename Rep, typename Period>
[[nodiscard]] inline std::int64_t OaChronoMicrosCount(std::chrono::duration<Rep, Period> InD) noexcept {
	return std::chrono::duration_cast<std::chrono::microseconds>(InD).count();
}
