#include <oa/core/std/chrono.h>
#include <oa/core/std/assert.h>

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
	OA_REQUIRE(inCounter >= 0);
	OA_REQUIRE(inFrequency > 0);
	const LONGLONG seconds = inCounter / inFrequency;
	const LONGLONG remainder = inCounter % inFrequency;
	const oa::I64 wholeNanoseconds = oa::detail::checkedI64Multiply(
		static_cast<oa::I64>(seconds), kNanosecondsPerSecond,
		"performance counter seconds overflow");
	const oa::I64 remainderNanoseconds = oa::detail::checkedI64Multiply(
		static_cast<oa::I64>(remainder), kNanosecondsPerSecond,
		"performance counter remainder overflow") / inFrequency;
	return oa::detail::checkedI64Add(
		wholeNanoseconds, remainderNanoseconds,
		"performance counter conversion overflow");
}

#else

[[nodiscard]] oa::I64 timespecNanoseconds(const timespec& inTime) noexcept {
	OA_REQUIRE(inTime.tv_nsec >= 0 and inTime.tv_nsec < kNanosecondsPerSecond);
	if constexpr (static_cast<time_t>(-1) > static_cast<time_t>(0)) {
		OA_REQUIRE(inTime.tv_sec <= static_cast<time_t>(oa::Limits<oa::I64>::max()));
	} else if constexpr (sizeof(time_t) > sizeof(oa::I64)) {
		OA_REQUIRE(inTime.tv_sec >= static_cast<time_t>(oa::Limits<oa::I64>::min()));
		OA_REQUIRE(inTime.tv_sec <= static_cast<time_t>(oa::Limits<oa::I64>::max()));
	}
	const oa::I64 seconds = static_cast<oa::I64>(inTime.tv_sec);
	const oa::I64 wholeNanoseconds = oa::detail::checkedI64Multiply(
		seconds, kNanosecondsPerSecond, "platform clock seconds overflow");
	return oa::detail::checkedI64Add(
		wholeNanoseconds, static_cast<oa::I64>(inTime.tv_nsec),
		"platform clock conversion overflow");
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
	const oa::U64 unixTicks = ticks.QuadPart - kWindowsToUnixEpoch100Nanoseconds;
	OA_REQUIRE(unixTicks <= static_cast<oa::U64>(oa::Limits<oa::I64>::max()) / 100ULL);
	return oa::SystemTimePoint(static_cast<oa::I64>(unixTicks * 100ULL));
#else
	timespec time{};
	OA_REQUIRE(clock_gettime(CLOCK_REALTIME, &time) == 0);
	return oa::SystemTimePoint(timespecNanoseconds(time));
#endif
}
