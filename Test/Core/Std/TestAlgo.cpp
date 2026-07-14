#include "OaStdTest.h"

#include <algorithm>
#include <random>
#include <vector>

TEST(OaStdAlgo, FindAndSort) {
	std::vector<int> v = {3, 1, 2};
	auto it = OaStdFind(v.begin(), v.end(), 2);
	ASSERT_NE(it, v.end());
	OaStdSort(v.begin(), v.end());
	EXPECT_EQ(v.front(), 1);
	EXPECT_EQ(v.back(), 3);

	constexpr int kRounds = 80'000;
	const auto t0 = OaHighResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		std::vector<int> w = {3, 1, 2};
		(void)OaStdFind(w.begin(), w.end(), 2);
		OaStdSort(w.begin(), w.end());
	}
	const auto t1 = OaHighResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		std::vector<int> w = {3, 1, 2};
		(void)std::find(w.begin(), w.end(), 2);
		std::sort(w.begin(), w.end());
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdFind+OaStdSort x80k", t0, t1, "std::find+std::sort x80k", t2);
	OaStdExpectGotInt("find+sort last back", 3, static_cast<long long>(v.back()));
}

TEST(OaStdAlgo, SpanFillAndFind) {
	int buf[4] = {1, 2, 3, 4};
	OaStdSpan<int> s(buf);
	OaStdFill(s, 0);
	EXPECT_EQ(buf[2], 0);
	buf[0] = 7;
	buf[1] = 8;
	auto it = OaStdFind(s, 8);
	ASSERT_NE(it, s.End());
	EXPECT_EQ(*it, 8);

	int oaBuf[64];
	int stBuf[64];
	constexpr int kLoops = 100'000;
	const auto t0 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		for (int i = 0; i < 64; ++i) {
			oaBuf[i] = i;
		}
		OaStdFill(OaStdSpan<int>(oaBuf), 0);
		oaBuf[0] = 7;
		oaBuf[1] = 8;
		(void)OaStdFind(OaStdSpan<int>(oaBuf), 8);
	}
	const auto t1 = OaHighResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		for (int i = 0; i < 64; ++i) {
			stBuf[i] = i;
		}
		std::fill(stBuf, stBuf + 64, 0);
		stBuf[0] = 7;
		stBuf[1] = 8;
		(void)std::find(stBuf, stBuf + 64, 8);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdFill+OaStdFind x100k", t0, t1, "std::fill+std::find x100k", t2);
	OaStdExpectGotInt("span find byte", 8, static_cast<long long>(*it));
}

TEST(OaStdAlgo, Clamp) {
	EXPECT_EQ(OaStdClamp(5, 0, 10), 5);
	EXPECT_EQ(OaStdClamp(-1, 0, 10), 0);
	EXPECT_EQ(OaStdClamp(99, 0, 10), 10);

	constexpr int kLoops = 2'000'000;
	volatile int sinkOa = 0;
	const auto t0 = OaHighResolutionNow();
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += OaStdClamp(i % 17 - 5, 0, 10);
	}
	const auto t1 = OaHighResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += std::clamp(i % 17 - 5, 0, 10);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdClamp x2M", t0, t1, "std::clamp x2M", t2);
	OaStdExpectGotInt("clamp sum tail", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(OaStdAlgoVsStd, SpanSortEqualToStdSort) {
	int oaBuf[32];
	int stBuf[32];
	std::minstd_rand rng(0x12345678u);
	for (int i = 0; i < 32; ++i) {
		const int v = static_cast<int>(rng() & 255);
		oaBuf[i] = v;
		stBuf[i] = v;
	}
	OaStdSort(OaStdSpan<int>(oaBuf));
	std::sort(stBuf, stBuf + 32);
	EXPECT_TRUE(OaStdEqual(OaStdSpan<int>(oaBuf), OaStdSpan<int>(stBuf)));

	const auto t0 = OaHighResolutionNow();
	for (int r = 0; r < 50'000; ++r) {
		for (int i = 0; i < 32; ++i) {
			oaBuf[i] = static_cast<int>((r + i) & 255);
		}
		OaStdSort(OaStdSpan<int>(oaBuf));
	}
	const auto t1 = OaHighResolutionNow();
	for (int r = 0; r < 50'000; ++r) {
		for (int i = 0; i < 32; ++i) {
			stBuf[i] = static_cast<int>((r + i) & 255);
		}
		std::sort(stBuf, stBuf + 32);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdSort(span 32) x50k", t0, t1, "std::sort(32) x50k", t2);
	OaStdExpectGotInt("sorted head", static_cast<long long>(stBuf[0]), static_cast<long long>(oaBuf[0]));
}

TEST(OaStdAlgoVsStd, SpanFindCountMatchStd) {
	int data[64];
	for (int i = 0; i < 64; ++i) {
		data[i] = i % 7;
	}
	OaStdSpan<int> sp(data);
	const int needle = 3;
	const auto oaIt = OaStdFind(sp, needle);
	const int* stIt = std::find(data, data + 64, needle);
	ASSERT_EQ(oaIt - sp.Data(), stIt - data);
	const auto oaCnt = OaStdCount(sp, needle);
	const auto stCnt = std::count(data, data + 64, needle);
	EXPECT_EQ(oaCnt, stCnt);

	const auto t0 = OaHighResolutionNow();
	volatile std::size_t sinkOa = 0;
	for (int r = 0; r < 40'000; ++r) {
		sinkOa += static_cast<std::size_t>(OaStdFind(sp, needle) - sp.Data());
		sinkOa += OaStdCount(sp, needle);
	}
	const auto t1 = OaHighResolutionNow();
	volatile std::size_t sinkSt = 0;
	for (int r = 0; r < 40'000; ++r) {
		sinkSt += static_cast<std::size_t>(std::find(data, data + 64, needle) - data);
		sinkSt += static_cast<std::size_t>(std::count(data, data + 64, needle));
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdFind+OaStdCount x40k", t0, t1, "std::find+std::count x40k", t2);
	OaStdExpectGotInt("OaStdCount vs std::count", static_cast<long long>(stCnt),
		static_cast<long long>(oaCnt));
	OaStdExpectGotInt("find+count sink", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(OaStdAlgoVsStd, TimedSortSpanWallUs) {
	constexpr int kN = 4096;
	constexpr int kRounds = 80;
	std::vector<int> base(kN);
	std::minstd_rand rng(0xDEADBEEFu);
	for (int i = 0; i < kN; ++i) {
		base[static_cast<std::size_t>(i)] = static_cast<int>(rng());
	}

	const auto t0 = OaHighResolutionNow();
	{
		std::vector<int> w = base;
		for (int r = 0; r < kRounds; ++r) {
			OaStdSort(OaStdSpan<int>(w.data(), w.size()));
			if (r + 1 < kRounds) {
				for (int i = 0; i < kN; ++i) {
					w[static_cast<std::size_t>(i)] ^= static_cast<int>(r + i);
				}
			}
		}
	}
	const auto t1 = OaHighResolutionNow();
	{
		std::vector<int> w = base;
		for (int r = 0; r < kRounds; ++r) {
			std::sort(w.begin(), w.end());
			if (r + 1 < kRounds) {
				for (int i = 0; i < kN; ++i) {
					w[static_cast<std::size_t>(i)] ^= static_cast<int>(r + i);
				}
			}
		}
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdSort(4096) x80 rounds", t0, t1, "std::sort(4096) x80 rounds", t2);
}

// Scalar two-argument OaStdMin / OaStdMax (added with the Format/Random gap-close).
TEST(OaStdAlgoMinMax, ScalarValues) {
	EXPECT_EQ(OaStdMin(3, 7), 3);
	EXPECT_EQ(OaStdMax(3, 7), 7);
	EXPECT_EQ(OaStdMin(7, 3), 3);
	EXPECT_EQ(OaStdMax(7, 3), 7);
}

TEST(OaStdAlgoMinMax, TieReturnsFirstArg) {
	int a = 5;
	int b = 5;
	EXPECT_EQ(&OaStdMin(a, b), &a);
	EXPECT_EQ(&OaStdMax(a, b), &a);
}

TEST(OaStdAlgoMinMax, CustomComparator) {
	auto greater = [](int x, int y) { return x > y; };
	// "min" under > selects the larger; "max" under > selects the smaller.
	EXPECT_EQ(OaStdMin(3, 7, greater), 7);
	EXPECT_EQ(OaStdMax(3, 7, greater), 3);
}
