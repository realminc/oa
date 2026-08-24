#include "oaStdTest.h"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace oa {

struct StdSortProbe {
	int value = 0;
};

} // namespace oa

TEST(StdAlgo, StandardSortSupportsOaTypesWithoutSwapAdlCollision) {
	std::array<oa::StdSortProbe, 3> values{{{3}, {1}, {2}}};
	std::sort(values.begin(), values.end(),
		[](const oa::StdSortProbe& inA, const oa::StdSortProbe& inB) {
			return inA.value < inB.value;
		});
	EXPECT_EQ(values[0].value, 1);
	EXPECT_EQ(values[1].value, 2);
	EXPECT_EQ(values[2].value, 3);
}

TEST(StdAlgo, FindAndSort) {
	std::vector<int> v = {3, 1, 2};
	auto it = oa::find(v.begin(), v.end(), 2);
	ASSERT_NE(it, v.end());
	oa::sort(v.begin(), v.end());
	EXPECT_EQ(v.front(), 1);
	EXPECT_EQ(v.back(), 3);

	constexpr int kRounds = 80'000;
	const auto t0 = oa::highResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		std::vector<int> w = {3, 1, 2};
		(void)oa::find(w.begin(), w.end(), 2);
		oa::sort(w.begin(), w.end());
	}
	const auto t1 = oa::highResolutionNow();
	for (int r = 0; r < kRounds; ++r) {
		std::vector<int> w = {3, 1, 2};
		(void)std::find(w.begin(), w.end(), 2);
		std::sort(w.begin(), w.end());
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::find+oa::sort x80k", t0, t1, "std::find+std::sort x80k", t2);
	stdExpectGotInt("find+sort last back", 3, static_cast<long long>(v.back()));
}

TEST(StdAlgo, SpanFillAndFind) {
	int buf[4] = {1, 2, 3, 4};
	oa::Span<int> s(buf);
	oa::fill(s, 0);
	EXPECT_EQ(buf[2], 0);
	buf[0] = 7;
	buf[1] = 8;
	auto it = oa::find(s, 8);
	ASSERT_NE(it, s.end());
	EXPECT_EQ(*it, 8);

	int oaBuf[64];
	int stBuf[64];
	constexpr int kLoops = 100'000;
	const auto t0 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		for (int i = 0; i < 64; ++i) {
			oaBuf[i] = i;
		}
		oa::fill(oa::Span<int>(oaBuf), 0);
		oaBuf[0] = 7;
		oaBuf[1] = 8;
		(void)oa::find(oa::Span<int>(oaBuf), 8);
	}
	const auto t1 = oa::highResolutionNow();
	for (int n = 0; n < kLoops; ++n) {
		for (int i = 0; i < 64; ++i) {
			stBuf[i] = i;
		}
		std::fill(stBuf, stBuf + 64, 0);
		stBuf[0] = 7;
		stBuf[1] = 8;
		(void)std::find(stBuf, stBuf + 64, 8);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::fill+oa::find x100k", t0, t1, "std::fill+std::find x100k", t2);
	stdExpectGotInt("span find byte", 8, static_cast<long long>(*it));
}

TEST(StdAlgo, Clamp) {
	EXPECT_EQ(oa::clamp(5, 0, 10), 5);
	EXPECT_EQ(oa::clamp(-1, 0, 10), 0);
	EXPECT_EQ(oa::clamp(99, 0, 10), 10);

	constexpr int kLoops = 2'000'000;
	volatile int sinkOa = 0;
	const auto t0 = oa::highResolutionNow();
	for (int i = 0; i < kLoops; ++i) {
		sinkOa += oa::clamp(i % 17 - 5, 0, 10);
	}
	const auto t1 = oa::highResolutionNow();
	volatile int sinkSt = 0;
	for (int i = 0; i < kLoops; ++i) {
		sinkSt += std::clamp(i % 17 - 5, 0, 10);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::clamp x2M", t0, t1, "std::clamp x2M", t2);
	stdExpectGotInt("clamp sum tail", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(StdAlgoVsStd, SpanSortEqualToStdSort) {
	int oaBuf[32];
	int stBuf[32];
	std::minstd_rand rng(0x12345678u);
	for (int i = 0; i < 32; ++i) {
		const int v = static_cast<int>(rng() & 255);
		oaBuf[i] = v;
		stBuf[i] = v;
	}
	oa::sort(oa::Span<int>(oaBuf));
	std::sort(stBuf, stBuf + 32);
	EXPECT_TRUE(oa::equal(oa::Span<int>(oaBuf), oa::Span<int>(stBuf)));

	const auto t0 = oa::highResolutionNow();
	for (int r = 0; r < 50'000; ++r) {
		for (int i = 0; i < 32; ++i) {
			oaBuf[i] = static_cast<int>((r + i) & 255);
		}
		oa::sort(oa::Span<int>(oaBuf));
	}
	const auto t1 = oa::highResolutionNow();
	for (int r = 0; r < 50'000; ++r) {
		for (int i = 0; i < 32; ++i) {
			stBuf[i] = static_cast<int>((r + i) & 255);
		}
		std::sort(stBuf, stBuf + 32);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::sort(span 32) x50k", t0, t1, "std::sort(32) x50k", t2);
	stdExpectGotInt("sorted head", static_cast<long long>(stBuf[0]), static_cast<long long>(oaBuf[0]));
}

TEST(StdAlgoVsStd, SpanFindCountMatchStd) {
	int data[64];
	for (int i = 0; i < 64; ++i) {
		data[i] = i % 7;
	}
	oa::Span<int> sp(data);
	const int needle = 3;
	const auto oaIt = oa::find(sp, needle);
	const int* stIt = std::find(data, data + 64, needle);
	ASSERT_EQ(oaIt - sp.data(), stIt - data);
	const auto oaCnt = oa::count(sp, needle);
	const auto stCnt = std::count(data, data + 64, needle);
	EXPECT_EQ(oaCnt, stCnt);

	const auto t0 = oa::highResolutionNow();
	volatile std::size_t sinkOa = 0;
	for (int r = 0; r < 40'000; ++r) {
		sinkOa += static_cast<std::size_t>(oa::find(sp, needle) - sp.data());
		sinkOa += oa::count(sp, needle);
	}
	const auto t1 = oa::highResolutionNow();
	volatile std::size_t sinkSt = 0;
	for (int r = 0; r < 40'000; ++r) {
		sinkSt += static_cast<std::size_t>(std::find(data, data + 64, needle) - data);
		sinkSt += static_cast<std::size_t>(std::count(data, data + 64, needle));
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::find+oa::count x40k", t0, t1, "std::find+std::count x40k", t2);
	stdExpectGotInt("oa::count vs std::count", static_cast<long long>(stCnt),
		static_cast<long long>(oaCnt));
	stdExpectGotInt("find+count sink", static_cast<long long>(sinkSt), static_cast<long long>(sinkOa));
	EXPECT_EQ(sinkOa, sinkSt);
}

TEST(StdAlgoVsStd, TimedSortSpanWallUs) {
	constexpr int kN = 4096;
	constexpr int kRounds = 80;
	std::vector<int> base(kN);
	std::minstd_rand rng(0xDEADBEEFu);
	for (int i = 0; i < kN; ++i) {
		base[static_cast<std::size_t>(i)] = static_cast<int>(rng());
	}

	const auto t0 = oa::highResolutionNow();
	{
		std::vector<int> w = base;
		for (int r = 0; r < kRounds; ++r) {
			oa::sort(oa::Span<int>(w.data(), w.size()));
			if (r + 1 < kRounds) {
				for (int i = 0; i < kN; ++i) {
					w[static_cast<std::size_t>(i)] ^= static_cast<int>(r + i);
				}
			}
		}
	}
	const auto t1 = oa::highResolutionNow();
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
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::sort(4096) x80 rounds", t0, t1, "std::sort(4096) x80 rounds", t2);
}

// Scalar two-argument oa::min / oa::max (added with the Format/Random gap-close).
TEST(StdAlgoMinMax, ScalarValues) {
	EXPECT_EQ(oa::min(3, 7), 3);
	EXPECT_EQ(oa::max(3, 7), 7);
	EXPECT_EQ(oa::min(7, 3), 3);
	EXPECT_EQ(oa::max(7, 3), 7);
}

TEST(StdAlgoMinMax, TieReturnsFirstArg) {
	int a = 5;
	int b = 5;
	EXPECT_EQ(&oa::min(a, b), &a);
	EXPECT_EQ(&oa::max(a, b), &a);
}

TEST(StdAlgoMinMax, CustomComparator) {
	auto greater = [](int x, int y) { return x > y; };
	// "min" under > selects the larger; "max" under > selects the smaller.
	EXPECT_EQ(oa::min(3, 7, greater), 7);
	EXPECT_EQ(oa::max(3, 7, greater), 3);
}
