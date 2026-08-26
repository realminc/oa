#include <oa/core/std/chrono.h>
#include <oa/core/assert.h>

#if defined(_WIN32)
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#else
	#include <time.h>
#endif

namespace {

constexpr oa::I64 kNanosecondsPerSecond = 1'000'000'000LL;

#if defined(_WIN32)

[[nodiscard]] oa::I64 scaledCounterNanoseconds(
	LONGLONG inCounter,
	LONGLONG inFrequency
) noexcept {
	OA_REQUIRE(inFrequency > 0);
	const LONGLONG seconds = inCounter / inFrequency;
	const LONGLONG remainder = inCounter % inFrequency;
	return static_cast<oa::I64>(seconds) * kNanosecondsPerSecond
		+ static_cast<oa::I64>(remainder * kNanosecondsPerSecond / inFrequency);
}

#else

[[nodiscard]] oa::I64 timespecNanoseconds(const timespec& inTime) noexcept {
	return static_cast<oa::I64>(inTime.tv_sec) * kNanosecondsPerSecond
		+ static_cast<oa::I64>(inTime.tv_nsec);
}

#endif

} // namespace

oa::SteadyTimePoint oa::SteadyClock::now() noexcept {
#if defined(_WIN32)
	LARGE_INTEGER counter{};
	LARGE_INTEGER frequency{};
	OA_REQUIRE(QueryPerformanceCounter(&counter) != 0);
	OA_REQUIRE(QueryPerformanceFrequency(&frequency) != 0);
	return oa::SteadyTimePoint(
		scaledCounterNanoseconds(counter.QuadPart, frequency.QuadPart));
#else
	timespec time{};
	OA_REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
	return oa::SteadyTimePoint(timespecNanoseconds(time));
#endif
}

oa::SystemTimePoint oa::SystemClock::now() noexcept {
#if defined(_WIN32)
	FILETIME fileTime{};
	GetSystemTimePreciseAsFileTime(&fileTime);
	ULARGE_INTEGER ticks{};
	ticks.LowPart = fileTime.dwLowDateTime;
	ticks.HighPart = fileTime.dwHighDateTime;
	constexpr oa::U64 kWindowsToUnixEpoch100Nanoseconds = 116'444'736'000'000'000ULL;
	OA_REQUIRE(ticks.QuadPart >= kWindowsToUnixEpoch100Nanoseconds);
	return oa::SystemTimePoint(static_cast<oa::I64>(
		(ticks.QuadPart - kWindowsToUnixEpoch100Nanoseconds) * 100ULL));
#else
	timespec time{};
	OA_REQUIRE(clock_gettime(CLOCK_REALTIME, &time) == 0);
	return oa::SystemTimePoint(timespecNanoseconds(time));
#endif
}
