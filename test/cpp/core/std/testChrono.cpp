#include "oaStdTest.h"

#include <type_traits>

static_assert(not std::is_same_v<oa::SteadyTimePoint, oa::SystemTimePoint>);
static_assert(std::is_same_v<oa::HighResolutionTimePoint, oa::SteadyTimePoint>);
static_assert(not std::is_constructible_v<oa::Datetime, oa::Timestamp>);
static_assert(oa::Duration::fromSeconds(2).nanoseconds() == 2'000'000'000LL);
static_assert(
	(oa::SteadyTimePoint(100) + oa::Duration::fromNanoseconds(25))
		.nanosecondsSinceEpoch() == 125);
static_assert(oa::Timestamp::fromMilliseconds(2).nanos() == 2'000'000LL);

TEST(OaStdChrono, DurationFactoriesAndArithmetic) {
	const oa::Duration duration = oa::Duration::fromSeconds(2)
		+ oa::Duration::fromMilliseconds(250)
		+ oa::Duration::fromMicroseconds(500);
	EXPECT_EQ(duration.nanoseconds(), 2'250'500'000LL);
	EXPECT_EQ(duration.microseconds(), 2'250'500LL);
	EXPECT_EQ(duration.milliseconds(), 2'250LL);
	EXPECT_DOUBLE_EQ(duration.toSeconds(), 2.2505);
	EXPECT_DOUBLE_EQ(duration.toMilliseconds(), 2250.5);
	EXPECT_DOUBLE_EQ(duration.toMicroseconds(), 2'250'500.0);
	EXPECT_EQ(
		duration - oa::Duration::fromMilliseconds(250),
		oa::Duration::fromMicroseconds(2'000'500));
}

TEST(OaStdChrono, CheckedConversionsAndArithmeticAcceptRepresentableBoundaries) {
	constexpr oa::I64 maximum = oa::Limits<oa::I64>::max();
	constexpr oa::I64 minimum = oa::Limits<oa::I64>::min();
	constexpr oa::I64 seconds = maximum / 1'000'000'000LL;
	EXPECT_EQ(
		oa::Duration::fromSeconds(seconds).nanoseconds(),
		seconds * 1'000'000'000LL);
	EXPECT_EQ(oa::Duration::fromDouble(1.25).nanoseconds(), 1'250'000'000LL);
	EXPECT_EQ(
		(oa::Duration::fromNanoseconds(maximum - 1)
			+ oa::Duration::fromNanoseconds(1)).nanoseconds(),
		maximum);
	EXPECT_EQ(
		(oa::Duration::fromNanoseconds(minimum + 1)
			- oa::Duration::fromNanoseconds(1)).nanoseconds(),
		minimum);

	oa::SteadyTimePoint steady(maximum - 1);
	steady += oa::Duration::fromNanoseconds(1);
	EXPECT_EQ(steady.nanosecondsSinceEpoch(), maximum);
	oa::SystemTimePoint system(minimum + 1);
	system -= oa::Duration::fromNanoseconds(1);
	EXPECT_EQ(system.nanosecondsSinceEpoch(), minimum);

	oa::Timestamp timestamp = oa::Timestamp::fromNanoseconds(maximum - 1);
	timestamp += oa::Timestamp::fromNanoseconds(1);
	EXPECT_EQ(timestamp.nanos(), maximum);
	EXPECT_EQ(
		oa::Datetime::fromUnixSeconds(1).unixNanoseconds(),
		1'000'000'000LL);
}

TEST(OaStdChrono, CheckedConversionsRejectOverflowAndNonFiniteInput) {
	constexpr oa::I64 maximum = oa::Limits<oa::I64>::max();
	constexpr oa::I64 minimum = oa::Limits<oa::I64>::min();
	EXPECT_DEATH(
		static_cast<void>(oa::Duration::fromSeconds(maximum)),
		"Duration seconds overflow");
	EXPECT_DEATH(
		static_cast<void>(oa::Duration::fromMilliseconds(minimum)),
		"Duration milliseconds overflow");
	EXPECT_DEATH(
		static_cast<void>(oa::Duration::fromDouble(oa::Limits<oa::F64>::infinity())),
		"time conversion requires a finite value");
	EXPECT_DEATH(
		static_cast<void>(oa::Duration::fromDouble(oa::Limits<oa::F64>::quietNaN())),
		"time conversion requires a finite value");
	EXPECT_DEATH(
		static_cast<void>(oa::Duration::fromDouble(0x1.0p63)),
		"Duration floating-point seconds overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::Duration::fromNanoseconds(maximum)
				+ oa::Duration::fromNanoseconds(1)),
		"Duration addition overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::Duration::fromNanoseconds(minimum)
				- oa::Duration::fromNanoseconds(1)),
		"Duration subtraction overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::SteadyTimePoint(maximum) + oa::Duration::fromNanoseconds(1)),
		"SteadyTimePoint addition overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::SystemTimePoint(maximum) - oa::SystemTimePoint(minimum)),
		"SystemTimePoint difference overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::Timestamp::fromNanoseconds(maximum)
				+ oa::Timestamp::fromNanoseconds(1)),
		"Timestamp addition overflow");
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime::fromUnixSeconds(maximum)),
		"Duration seconds overflow");
	EXPECT_DEATH(
		static_cast<void>(
			oa::Datetime::fromUnixNanoseconds(maximum).add(
				oa::Duration::fromNanoseconds(1))),
		"SystemTimePoint addition overflow");
}

TEST(OaStdChrono, DistinctTimePointsUseCommonDuration) {
	const oa::SteadyTimePoint steadyBegin(100);
	const oa::SteadyTimePoint steadyEnd = steadyBegin + oa::Duration::fromNanoseconds(25);
	EXPECT_EQ((steadyEnd - steadyBegin).nanoseconds(), 25);
	EXPECT_GT(steadyEnd, steadyBegin);

	const oa::SystemTimePoint systemBegin(500);
	const oa::SystemTimePoint systemEnd = systemBegin + oa::Duration::fromNanoseconds(50);
	EXPECT_EQ((systemEnd - systemBegin).nanoseconds(), 50);
	EXPECT_GT(systemEnd, systemBegin);
}

TEST(OaStdChrono, PlatformClocksHaveTheirDeclaredDomains) {
	const oa::SteadyTimePoint steadyBegin = oa::steadyNow();
	const oa::SteadyTimePoint steadyEnd = oa::steadyNow();
	EXPECT_GE(steadyEnd, steadyBegin);

	constexpr oa::I64 kJanuary2020Nanoseconds = 1'577'836'800'000'000'000LL;
	EXPECT_GT(oa::systemNow().nanosecondsSinceEpoch(), kJanuary2020Nanoseconds);
}

TEST(OaStdChrono, DatetimeUsesWallClockAndUtcCalendar) {
	const oa::Datetime leapDay(2024, 2, 29, 12, 34, 56);
	EXPECT_EQ(leapDay.year(), 2024);
	EXPECT_EQ(leapDay.month(), 2);
	EXPECT_EQ(leapDay.day(), 29);
	EXPECT_EQ(leapDay.hour(), 12);
	EXPECT_EQ(leapDay.minute(), 34);
	EXPECT_EQ(leapDay.second(), 56);
	EXPECT_EQ(leapDay.toIso(), "2024-02-29T12:34:56Z");
	EXPECT_EQ(
		leapDay.add(oa::Duration::fromSeconds(4)).toIso(),
		"2024-02-29T12:35:00Z");
	EXPECT_EQ(
		leapDay.diff(oa::Datetime(2024, 2, 29, 12, 34, 50)).toSeconds(),
		6.0);

	const oa::Datetime current = oa::Datetime::now();
	EXPECT_GE(current.year(), 2020);
	EXPECT_LT(current.year(), 2200);
}

TEST(OaStdChrono, DatetimeRejectsInvalidOrUnrepresentableCalendarInput) {
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime(2024, 13, 1)),
		"Datetime month is outside 1..12");
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime(2024, 2, 30)),
		"Datetime fields do not form a representable UTC calendar time");
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime(2024, 1, 1, 24)),
		"Datetime hour is outside 0..23");
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime(oa::Limits<oa::I32>::min(), 1, 1)),
		"Datetime year is outside the platform calendar range");
	EXPECT_DEATH(
		static_cast<void>(oa::Datetime(2024, 1, 1).format(nullptr)),
		"Datetime format requires a format string");
}

#if !defined(_WIN32)
TEST(OaStdChrono, DatetimeUsesFloorSecondsForPreEpochSubseconds) {
	const oa::Datetime beforeEpoch = oa::Datetime::fromUnixNanoseconds(-1);
	EXPECT_EQ(beforeEpoch.year(), 1969);
	EXPECT_EQ(beforeEpoch.month(), 12);
	EXPECT_EQ(beforeEpoch.day(), 31);
	EXPECT_EQ(beforeEpoch.hour(), 23);
	EXPECT_EQ(beforeEpoch.minute(), 59);
	EXPECT_EQ(beforeEpoch.second(), 59);
	EXPECT_EQ(beforeEpoch.microsecond(), 999'999);
	EXPECT_EQ(beforeEpoch.toIsoMicro(), "1969-12-31T23:59:59.999999Z");
}
#endif

TEST(OaStdChrono, FormatDurationRejectsNonFiniteAndOverflowingInput) {
	EXPECT_DEATH(
		static_cast<void>(oa::formatDuration(oa::Limits<oa::F64>::infinity())),
		"formatDuration requires a finite value");
	EXPECT_DEATH(
		static_cast<void>(oa::formatDuration(oa::Limits<oa::F64>::quietNaN())),
		"formatDuration requires a finite value");
	EXPECT_DEATH(
		static_cast<void>(oa::formatDuration(0x1.0p63)),
		"formatDuration seconds overflow");
}

TEST(OaStdChrono, DurationParserHandlesCompleteAndMixedCaseInput) {
	EXPECT_DOUBLE_EQ(oa::parseDuration("30"), 30.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration("1.5H"), 5400.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration(" 2 DAYS"), 172800.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration("."), 0.0);
}
