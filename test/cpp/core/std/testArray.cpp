#include "oaStdTest.h"

#include <algorithm>
#include <random>

TEST(Array, FillAndIndex) {
	oa::Array<int, 4> arr;
	arr.fill(42);
	EXPECT_EQ(arr.size(), 4U);
	EXPECT_EQ(arr[0], 42);
	EXPECT_EQ(arr.back(), 42);

	oa::Array<int, 64> oa{};
	std::array<int, 64> st{};
	constexpr int kLoops = 80'000;
	const auto t0 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		oa.fill(static_cast<int>(n & 127));
	}
	const auto t1 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		st.fill(static_cast<int>(n & 127));
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array::fill x80k (64)", t0, t1, "std::array::fill x80k (64)", t2);
	stdExpectGotInt("fill spot check", static_cast<long long>(st[0]), static_cast<long long>(oa[0]));
}

TEST(Array, ValueInitialization) {
	constexpr oa::Array<oa::I32, 4> values{3, 5, 7, 11};
	static_assert(values.size() == 4);
	EXPECT_EQ(values[0], 3);
	EXPECT_EQ(values[3], 11);
}

TEST(Array, AtRejectsOutOfRange) {
	oa::Array<int, 2> arr{};
	EXPECT_EQ(arr.at(0), 0);
	EXPECT_DEATH(static_cast<void>(arr.at(2)), "OA contract failed: inIndex < N");

	constexpr int kLoops = 400'000;
	const auto t0 = oa::highResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += arr.at(0);
	}
	const auto t1 = oa::highResolutionNow();
	std::array<int, 2> st{};
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.at(0);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array::at(0) x400k", t0, t1, "std::array::at(0) x400k", t2);
	stdExpectGotInt("at(0) sum tail", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(Array, Swap) {
	oa::Array<int, 3> a{};
	oa::Array<int, 3> b{};
	a.fill(1);
	b.fill(2);
	a.swap(b);
	EXPECT_EQ(a[0], 2);
	EXPECT_EQ(b[0], 1);

	constexpr int kSwaps = 200'000;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		oa::Array<int, 3> x{};
		oa::Array<int, 3> y{};
		x[0] = 1;
		y[0] = 2;
		x.swap(y);
		(void)x[0];
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		std::array<int, 3> x{};
		std::array<int, 3> y{};
		x[0] = 1;
		y[0] = 2;
		x.swap(y);
		(void)x[0];
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array::swap x200k", t0, t1, "std::array::swap x200k", t2);
}

TEST(Array, RangeFor) {
	oa::Array<int, 3> arr{};
	arr[0] = 1;
	arr[1] = 2;
	arr[2] = 3;
	int sum = 0;
	for (int x : arr) {
		sum += x;
	}
	EXPECT_EQ(sum, 6);

	std::array<int, 3> st = {1, 2, 3};
	constexpr int kLoops = 300'000;
	const auto t0 = oa::highResolutionNow();
	volatile int sinkOa = 0;
	for (int n = 0; n < kLoops; ++n) {
		int s = 0;
		for (int x : arr) {
			s += x;
		}
		sinkOa += s;
	}
	const auto t1 = oa::highResolutionNow();
	volatile int sinkSt = 0;
	for (int n = 0; n < kLoops; ++n) {
		int s = 0;
		for (int x : st) {
			s += x;
		}
		sinkSt += s;
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array range-for sum x300k", t0, t1, "std::array range-for sum x300k", t2);
	stdExpectGotInt("range-for sum", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(Array, ZeroSize) {
	oa::Array<int, 0> z;
	EXPECT_EQ(z.size(), 0U);
	EXPECT_TRUE(z.empty());
	EXPECT_EQ(z.data(), nullptr);
	EXPECT_DEATH(static_cast<void>(z.at(0)), "OA contract failed: false");

	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < 500'000; ++i) {
		oa::Array<int, 0> zz;
		(void)zz.empty();
	}
	const auto t1 = oa::highResolutionNow();
	for (int i = 0; i < 500'000; ++i) {
		std::array<int, 0> zz;
		(void)zz.empty();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array<0>::Empty x500k", t0, t1, "std::array<0>::empty x500k", t2);
}

TEST(StdArrayVsStd, SameSequenceAsStdArray) {
	oa::Array<int, 16> oa{};
	std::array<int, 16> st{};
	std::minstd_rand rng(0xC0FFEEu);
	for (std::size_t i = 0; i < 16; ++i) {
		const int v = static_cast<int>(rng() & 0xFF);
		oa[static_cast<std::size_t>(i)] = v;
		st[i] = v;
	}
	for (std::size_t i = 0; i < 16; ++i) {
		EXPECT_EQ(oa[i], st[i]) << "i=" << i;
	}
	EXPECT_TRUE(std::equal(oa.data(), oa.data() + oa.size(), st.begin(), st.end()));
	stdEchoCurrentTest();
	stdExpectGotSize("StdArrayVsStd::size", st.size(), oa.size());
}

TEST(StdArrayVsStd, TimedIndexSumWallUs) {
	constexpr int kRounds = 400'000;
	oa::Array<int, 64> oa{};
	std::array<int, 64> st{};
	for (std::size_t i = 0; i < 64; ++i) {
		oa[static_cast<std::size_t>(i)] = static_cast<int>(i);
		st[i] = static_cast<int>(i);
	}
	volatile int sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		int s = 0;
		for (std::size_t i = 0; i < 64; ++i) {
			s += oa[static_cast<std::size_t>(i)];
		}
		sinkOa += s;
	}
	const auto t1 = oa::highResolutionNow();
	volatile int sinkSt = 0;
	for (int r = 0; r < kRounds; ++r) {
		int s = 0;
		for (std::size_t i = 0; i < 64; ++i) {
			s += st[i];
		}
		sinkSt += s;
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Array indexed sum x400k", t0, t1, "std::array indexed sum x400k", t2);
	stdExpectGotInt("array sum sanity", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}
