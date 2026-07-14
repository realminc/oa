#include "OaStdTest.h"

#include <algorithm>
#include <random>
#include <stdexcept>

TEST(OaStdArray, FillAndIndex) {
	OaStdArray<int, 4> arr;
	arr.Fill(42);
	EXPECT_EQ(arr.Size(), 4U);
	EXPECT_EQ(arr[0], 42);
	EXPECT_EQ(arr.Back(), 42);

	OaStdArray<int, 64> oa{};
	std::array<int, 64> st{};
	constexpr int kLoops = 80'000;
	const auto t0 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		oa.Fill(static_cast<int>(n & 127));
	}
	const auto t1 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		st.fill(static_cast<int>(n & 127));
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray::Fill x80k (64)", t0, t1, "std::array::fill x80k (64)", t2);
	OaStdExpectGotInt("fill spot check", static_cast<long long>(st[0]), static_cast<long long>(oa[0]));
}

TEST(OaStdArray, AtThrows) {
	OaStdArray<int, 2> arr{};
	EXPECT_EQ(arr.At(0), 0);
	EXPECT_THROW(static_cast<void>(arr.At(2)), std::out_of_range);

	constexpr int kLoops = 400'000;
	const auto t0 = OaHighResolutionNow();
	volatile int sinkOa = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += arr.At(0);
	}
	const auto t1 = OaHighResolutionNow();
	std::array<int, 2> st{};
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += st.at(0);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray::At(0) x400k", t0, t1, "std::array::at(0) x400k", t2);
	OaStdExpectGotInt("at(0) sum tail", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(OaStdArray, SwapAndStdArray) {
	OaStdArray<int, 3> a{};
	OaStdArray<int, 3> b{};
	a.Fill(1);
	b.Fill(2);
	a.Swap(b);
	EXPECT_EQ(a[0], 2);
	EXPECT_EQ(b[0], 1);
	std::array<int, 3> s = a.StdArray();
	EXPECT_EQ(s[0], 2);

	constexpr int kSwaps = 200'000;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		OaStdArray<int, 3> x{};
		OaStdArray<int, 3> y{};
		x[0] = 1;
		y[0] = 2;
		x.Swap(y);
		(void)x[0];
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < kSwaps; ++i) {
		std::array<int, 3> x{};
		std::array<int, 3> y{};
		x[0] = 1;
		y[0] = 2;
		x.swap(y);
		(void)x[0];
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray::Swap x200k", t0, t1, "std::array::swap x200k", t2);
}

TEST(OaStdArray, RangeFor) {
	OaStdArray<int, 3> arr{};
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
	const auto t0 = OaHighResolutionNow();
	volatile int sinkOa = 0;
	for (int n = 0; n < kLoops; ++n) {
		int s = 0;
		for (int x : arr) {
			s += x;
		}
		sinkOa += s;
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int n = 0; n < kLoops; ++n) {
		int s = 0;
		for (int x : st) {
			s += x;
		}
		sinkSt += s;
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray range-for sum x300k", t0, t1, "std::array range-for sum x300k", t2);
	OaStdExpectGotInt("range-for sum", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(OaStdArray, ZeroSize) {
	OaStdArray<int, 0> z;
	EXPECT_EQ(z.Size(), 0U);
	EXPECT_TRUE(z.Empty());
	EXPECT_EQ(z.Data(), nullptr);
	std::array<int, 0> s = z.StdArray();
	(void)s;

	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < 500'000; ++i) {
		OaStdArray<int, 0> zz;
		(void)zz.Empty();
	}
	const auto t1 = OaHighResolutionNow();
	for (int i = 0; i < 500'000; ++i) {
		std::array<int, 0> zz;
		(void)zz.empty();
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray<0>::Empty x500k", t0, t1, "std::array<0>::empty x500k", t2);
}

TEST(OaStdArrayVsStd, SameSequenceAsStdArray) {
	OaStdArray<int, 16> oa{};
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
	EXPECT_TRUE(std::equal(oa.Data(), oa.Data() + oa.Size(), st.begin(), st.end()));
	auto s2 = oa.StdArray();
	EXPECT_EQ(s2, st);
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("OaStdArrayVsStd::size", st.size(), oa.Size());
}

TEST(OaStdArrayVsStd, TimedIndexSumWallUs) {
	constexpr int kRounds = 400'000;
	OaStdArray<int, 64> oa{};
	std::array<int, 64> st{};
	for (std::size_t i = 0; i < 64; ++i) {
		oa[static_cast<std::size_t>(i)] = static_cast<int>(i);
		st[i] = static_cast<int>(i);
	}
	volatile int sinkOa = 0;
	const auto t0 = OaHighResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		int s = 0;
		for (std::size_t i = 0; i < 64; ++i) {
			s += oa[static_cast<std::size_t>(i)];
		}
		sinkOa += s;
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int r = 0; r < kRounds; ++r) {
		int s = 0;
		for (std::size_t i = 0; i < 64; ++i) {
			s += st[i];
		}
		sinkSt += s;
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdArray indexed sum x400k", t0, t1, "std::array indexed sum x400k", t2);
	OaStdExpectGotInt("array sum sanity", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}
