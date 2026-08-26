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

[[nodiscard]] tm systemTimeTm(oa::SystemTimePoint inTime) noexcept {
	return gmTime(static_cast<time_t>(
		inTime.nanosecondsSinceEpoch() / 1'000'000'000LL));
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
	tm time{};
	time.tm_year = inYear - 1900;
	time.tm_mon = inMonth - 1;
	time.tm_mday = inDay;
	time.tm_hour = inHour;
	time.tm_min = inMinute;
	time.tm_sec = inSecond;
	time.tm_isdst = 0;
	time_ = oa::SystemTimePoint(
		static_cast<oa::I64>(timeGm(&time)) * 1'000'000'000LL);
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
