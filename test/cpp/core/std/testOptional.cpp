#include "oaStdTest.h"

#include <optional>

TEST(Optional, ValueOr) {
	oa::Optional<int> empty;
	oa::Optional<int> filled(7);
	EXPECT_FALSE(empty.hasValue());
	EXPECT_TRUE(filled.hasValue());
	EXPECT_EQ(filled.value(), 7);
	EXPECT_EQ(empty.valueOr(99), 99);

	constexpr int kLoops = 200'000;
	oa::Optional<int> oa;
	std::optional<int> st;
	const auto t0 = oa::highResolutionNow();
	volatile long long sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.valueOr(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Optional::valueOr (empty) x200k", t0, t1,
		"std::optional::value_or (empty) x200k", t2);
	stdExpectGotInt("empty ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(Optional, ValueThrowsWhenEmpty) {
	oa::Optional<int> empty;
	EXPECT_THROW(static_cast<void>(empty.value()), std::bad_optional_access);
}

TEST(Optional, EmplaceAndReset) {
	oa::Optional<int> o;
	o.emplace(5);
	ASSERT_TRUE(o.hasValue());
	EXPECT_EQ(o.value(), 5);
	o.reset();
	EXPECT_FALSE(o.hasValue());
}

TEST(Optional, FromStdOptional) {
	std::optional<int> s(11);
	oa::Optional<int> o(s);
	ASSERT_TRUE(o.hasValue());
	EXPECT_EQ(o.value(), 11);
}

TEST(Optional, StdOptionalRoundTrip) {
	oa::Optional<int> o(3);
	std::optional<int> s = o.stdOptional();
	ASSERT_TRUE(s.has_value());
	EXPECT_EQ(*s, 3);
}

TEST(Optional, AssignNullopt) {
	oa::Optional<int> o(1);
	o = std::nullopt;
	EXPECT_FALSE(o.hasValue());
}

TEST(StdOptionalVsStd, ParallelEmplaceResetSequence) {
	oa::Optional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.hasValue(), st.has_value());
	oa.emplace(42);
	st.emplace(42);
	EXPECT_EQ(oa.value(), *st);
	oa.reset();
	st.reset();
	EXPECT_FALSE(oa.hasValue());
	EXPECT_FALSE(st.has_value());

	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		oa::Optional<int> x;
		x.emplace(i);
		x.reset();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		std::optional<int> x;
		x.emplace(i);
		x.reset();
	}
	const auto t2 = oa::highResolutionNow();
	stdEchoCurrentTest();
	stdExpectGotInt("optional has_value after reset", static_cast<long long>(st.has_value()),
		static_cast<long long>(oa.hasValue()));
	stdReportCompareMsLines(
		"oa::Optional emplace+reset x150k", stdWallMs(t1 - t0),
		"std::optional emplace+reset x150k", stdWallMs(t2 - t1));
}

TEST(StdOptionalVsStd, ValueOrMatchesStd) {
	oa::Optional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.valueOr(-1), st.value_or(-1));
	oa.emplace(7);
	st = 7;
	EXPECT_EQ(oa.valueOr(-1), st.value_or(-1));
	stdEchoCurrentTest();
	stdExpectGotInt("ValueOr with value", static_cast<long long>(st.value_or(-1)),
		static_cast<long long>(oa.valueOr(-1)));
}

TEST(StdOptionalVsStd, TimedValueOrWallUs) {
	constexpr int kLoops = 500'000;
	oa::Optional<int> oa;
	oa.emplace(123);
	std::optional<int> st(123);
	volatile long long sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.valueOr(i);
	}
	const auto t1 = oa::highResolutionNow();
	volatile long long sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Optional::valueOr (filled) x500k", t0, t1,
		"std::optional::value_or (filled) x500k", t2);
	stdExpectGotInt("optional ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}
