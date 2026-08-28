#include <oa/core/time.h>
#include <oa/core/assert.h>

#include <stdio.h>
#include <time.h>

namespace {

[[nodiscard]] time_t timeGm(tm* inTime) noexcept {
#if defined(_WIN32)
	return _mkgmtime64(inTime);
#else
	return timegm(inTime);
#endif
}

[[nodiscard]] tm gmTime(time_t inTime) noexcept {
	tm result{};
#if defined(_WIN32)
	OA_REQUIRE(gmtime_s(&result, &inTime) == 0);
#else
	OA_REQUIRE(gmtime_r(&inTime, &result) != nullptr);
#endif
	return result;
}

[[nodiscard]] time_t checkedTimeT(oa::I64 inSeconds) noexcept {
	if constexpr (static_cast<time_t>(-1) > static_cast<time_t>(0)) {
		OA_REQUIRE_MSG(inSeconds >= 0, "calendar time precedes the platform epoch");
	}
	const time_t result = static_cast<time_t>(inSeconds);
	OA_REQUIRE_MSG(
		static_cast<oa::I64>(result) == inSeconds,
		"calendar time is outside the platform time_t range");
	return result;
}

[[nodiscard]] oa::I64 checkedI64Time(time_t inSeconds) noexcept {
	if constexpr (static_cast<time_t>(-1) > static_cast<time_t>(0)) {
		OA_REQUIRE_MSG(
			inSeconds <= static_cast<time_t>(oa::Limits<oa::I64>::max()),
			"calendar result is outside the OA time range");
	}
	const oa::I64 result = static_cast<oa::I64>(inSeconds);
	OA_REQUIRE_MSG(
		static_cast<time_t>(result) == inSeconds,
		"calendar result is outside the OA time range");
	return result;
}

[[nodiscard]] oa::I64 floorEpochSeconds(oa::I64 inNanoseconds) noexcept {
	oa::I64 seconds = inNanoseconds / 1'000'000'000LL;
	if (inNanoseconds % 1'000'000'000LL < 0) {
		--seconds;
	}
	return seconds;
}

[[nodiscard]] tm systemTimeTm(oa::SystemTimePoint inTime) noexcept {
	return gmTime(checkedTimeT(floorEpochSeconds(inTime.nanosecondsSinceEpoch())));
}

} // namespace

oa::Datetime::Datetime(
	oa::I32 inYear,
	oa::I32 inMonth,
	oa::I32 inDay,
	oa::I32 inHour,
	oa::I32 inMinute,
	oa::I32 inSecond
) {
	OA_REQUIRE_MSG(inMonth >= 1 and inMonth <= 12, "Datetime month is outside 1..12");
	OA_REQUIRE_MSG(inDay >= 1 and inDay <= 31, "Datetime day is outside 1..31");
	OA_REQUIRE_MSG(inHour >= 0 and inHour <= 23, "Datetime hour is outside 0..23");
	OA_REQUIRE_MSG(inMinute >= 0 and inMinute <= 59, "Datetime minute is outside 0..59");
	OA_REQUIRE_MSG(inSecond >= 0 and inSecond <= 59, "Datetime second is outside 0..59");
	const oa::I64 yearOffset = static_cast<oa::I64>(inYear) - 1900LL;
	OA_REQUIRE_MSG(
		yearOffset >= oa::Limits<int>::min() and yearOffset <= oa::Limits<int>::max(),
		"Datetime year is outside the platform calendar range");

	tm time{};
	time.tm_year = static_cast<int>(yearOffset);
	time.tm_mon = inMonth - 1;
	time.tm_mday = inDay;
	time.tm_hour = inHour;
	time.tm_min = inMinute;
	time.tm_sec = inSecond;
	time.tm_isdst = 0;
	const tm requested = time;
	const time_t calendarSeconds = timeGm(&time);
	const tm normalized = gmTime(calendarSeconds);
	OA_REQUIRE_MSG(
		normalized.tm_year == requested.tm_year
			and normalized.tm_mon == requested.tm_mon
			and normalized.tm_mday == requested.tm_mday
			and normalized.tm_hour == requested.tm_hour
			and normalized.tm_min == requested.tm_min
			and normalized.tm_sec == requested.tm_sec,
		"Datetime fields do not form a representable UTC calendar time");
	const oa::I64 seconds = checkedI64Time(calendarSeconds);
	time_ = oa::SystemTimePoint(oa::detail::checkedI64Multiply(
		seconds, 1'000'000'000LL, "Datetime nanosecond conversion overflow"));
}

oa::Datetime oa::Datetime::now() noexcept {
	return oa::Datetime(oa::systemNow());
}

oa::I32 oa::Datetime::year() const { return systemTimeTm(time_).tm_year + 1900; }
oa::I32 oa::Datetime::month() const { return systemTimeTm(time_).tm_mon + 1; }
oa::I32 oa::Datetime::day() const { return systemTimeTm(time_).tm_mday; }
oa::I32 oa::Datetime::hour() const { return systemTimeTm(time_).tm_hour; }
oa::I32 oa::Datetime::minute() const { return systemTimeTm(time_).tm_min; }
oa::I32 oa::Datetime::second() const { return systemTimeTm(time_).tm_sec; }
oa::I32 oa::Datetime::dayOfWeek() const { return systemTimeTm(time_).tm_wday; }
oa::I32 oa::Datetime::dayOfYear() const { return systemTimeTm(time_).tm_yday + 1; }

oa::String oa::Datetime::format(const char* inFormat) const {
	OA_REQUIRE_MSG(inFormat != nullptr, "Datetime format requires a format string");
	tm time = systemTimeTm(time_);
	char buffer[128];
	const oa::Usize count = static_cast<oa::Usize>(
		::strftime(buffer, sizeof(buffer), inFormat, &time));
	return count == 0 ? oa::String() : oa::String(oa::StringView(buffer, count));
}

oa::String oa::Datetime::toIsoMicro() const {
	tm time = systemTimeTm(time_);
	char buffer[64];
	const oa::Usize count = static_cast<oa::Usize>(
		::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &time));
	if (count == 0) return oa::String();

	oa::String result(oa::StringView(buffer, count));
	char suffix[16];
	if (::snprintf(suffix, sizeof(suffix), ".%06dZ", static_cast<int>(microsecond())) > 0) {
		result += suffix;
	}
	return result;
}
