#include "oaStdTest.h"

#include <type_traits>

static_assert(not std::is_same_v<oa::SteadyTimePoint, oa::SystemTimePoint>);
static_assert(std::is_same_v<oa::HighResolutionTimePoint, oa::SteadyTimePoint>);
static_assert(not std::is_constructible_v<oa::Datetime, oa::Timestamp>);

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

TEST(OaStdChrono, DurationParserHandlesCompleteAndMixedCaseInput) {
	EXPECT_DOUBLE_EQ(oa::parseDuration("30"), 30.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration("1.5H"), 5400.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration(" 2 DAYS"), 172800.0);
	EXPECT_DOUBLE_EQ(oa::parseDuration("."), 0.0);
}
