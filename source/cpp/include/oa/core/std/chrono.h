#pragma once

#include <chrono>
#include <cstdint>

namespace oa {

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;
using SteadyDuration = SteadyClock::duration;

using SystemClock = std::chrono::system_clock;
using SystemTimePoint = SystemClock::time_point;
using SystemDuration = SystemClock::duration;

using HighResolutionClock = std::chrono::high_resolution_clock;
using HighResolutionTimePoint = HighResolutionClock::time_point;
using HighResolutionDuration = HighResolutionClock::duration;

[[nodiscard]] inline SteadyTimePoint steadyNow() noexcept { return SteadyClock::now(); }
[[nodiscard]] inline SystemTimePoint systemNow() noexcept { return SystemClock::now(); }
[[nodiscard]] inline HighResolutionTimePoint highResolutionNow() noexcept {
	return HighResolutionClock::now();
}

template<typename Rep, typename Period>
[[nodiscard]] inline double chronoToSeconds(
	std::chrono::duration<Rep, Period> inDuration) noexcept
{
	return std::chrono::duration<double>(inDuration).count();
}

template<typename Rep, typename Period>
[[nodiscard]] inline double chronoToMilli(
	std::chrono::duration<Rep, Period> inDuration) noexcept
{
	return std::chrono::duration<double, std::milli>(inDuration).count();
}

template<typename Rep, typename Period>
[[nodiscard]] inline double chronoToMicro(
	std::chrono::duration<Rep, Period> inDuration) noexcept
{
	return std::chrono::duration<double, std::micro>(inDuration).count();
}

template<typename Rep, typename Period>
[[nodiscard]] inline std::int64_t chronoMillisCount(
	std::chrono::duration<Rep, Period> inDuration) noexcept
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(inDuration).count();
}

template<typename Rep, typename Period>
[[nodiscard]] inline std::int64_t chronoMicrosCount(
	std::chrono::duration<Rep, Period> inDuration) noexcept
{
	return std::chrono::duration_cast<std::chrono::microseconds>(inDuration).count();
}

} // namespace oa
