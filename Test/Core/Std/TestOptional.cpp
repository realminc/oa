#include "OaStdTest.h"

#include <optional>

TEST(OaStdOptional, ValueOr) {
	OaStdOptional<int> empty;
	OaStdOptional<int> filled(7);
	EXPECT_FALSE(empty.HasValue());
	EXPECT_TRUE(filled.HasValue());
	EXPECT_EQ(filled.Value(), 7);
	EXPECT_EQ(empty.ValueOr(99), 99);

	constexpr int kLoops = 200'000;
	OaStdOptional<int> oa;
	std::optional<int> st;
	const auto t0 = OaHighResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.ValueOr(i);
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdOptional::ValueOr (empty) x200k", t0, t1,
		"std::optional::value_or (empty) x200k", t2);
	OaStdExpectGotInt("empty ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(OaStdOptional, ValueThrowsWhenEmpty) {
	OaStdOptional<int> empty;
	EXPECT_THROW(static_cast<void>(empty.Value()), std::bad_optional_access);
}

TEST(OaStdOptional, EmplaceAndReset) {
	OaStdOptional<int> o;
	o.Emplace(5);
	ASSERT_TRUE(o.HasValue());
	EXPECT_EQ(o.Value(), 5);
	o.Reset();
	EXPECT_FALSE(o.HasValue());
}

TEST(OaStdOptional, FromStdOptional) {
	std::optional<int> s(11);
	OaStdOptional<int> o(s);
	ASSERT_TRUE(o.HasValue());
	EXPECT_EQ(o.Value(), 11);
}

TEST(OaStdOptional, StdOptionalRoundTrip) {
	OaStdOptional<int> o(3);
	std::optional<int> s = o.StdOptional();
	ASSERT_TRUE(s.has_value());
	EXPECT_EQ(*s, 3);
}

TEST(OaStdOptional, AssignNullopt) {
	OaStdOptional<int> o(1);
	o = std::nullopt;
	EXPECT_FALSE(o.HasValue());
}

TEST(OaStdOptionalVsStd, ParallelEmplaceResetSequence) {
	OaStdOptional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.HasValue(), st.has_value());
	oa.Emplace(42);
	st.emplace(42);
	EXPECT_EQ(oa.Value(), *st);
	oa.Reset();
	st.reset();
	EXPECT_FALSE(oa.HasValue());
	EXPECT_FALSE(st.has_value());

	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		OaStdOptional<int> x;
		x.Emplace(i);
		x.Reset();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < 150'000; ++i) {
		std::optional<int> x;
		x.emplace(i);
		x.reset();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("optional has_value after reset", static_cast<long long>(st.has_value()),
		static_cast<long long>(oa.HasValue()));
	OaStdReportCompareMsLines(
		"OaStdOptional emplace+reset x150k", OaStdWallMs(t1 - t0),
		"std::optional emplace+reset x150k", OaStdWallMs(t2 - t1));
}

TEST(OaStdOptionalVsStd, ValueOrMatchesStd) {
	OaStdOptional<int> oa;
	std::optional<int> st;
	EXPECT_EQ(oa.ValueOr(-1), st.value_or(-1));
	oa.Emplace(7);
	st = 7;
	EXPECT_EQ(oa.ValueOr(-1), st.value_or(-1));
	OaStdEchoCurrentTest();
	OaStdExpectGotInt("ValueOr with value", static_cast<long long>(st.value_or(-1)),
		static_cast<long long>(oa.ValueOr(-1)));
}

TEST(OaStdOptionalVsStd, TimedValueOrWallUs) {
	constexpr int kLoops = 500'000;
	OaStdOptional<int> oa;
	oa.Emplace(123);
	std::optional<int> st(123);
	volatile int sinkOa = 0;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa.ValueOr(i);
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.value_or(i);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdOptional::ValueOr (filled) x500k", t0, t1,
		"std::optional::value_or (filled) x500k", t2);
	OaStdExpectGotInt("optional ValueOr sum tail", static_cast<long long>(sinkSt % 1000),
		static_cast<long long>(sinkOa % 1000));
	EXPECT_EQ(sinkOa, sinkSt);
}
