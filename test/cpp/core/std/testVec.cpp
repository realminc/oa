#include "oaStdTest.h"

#include <algorithm>
#include <random>

TEST(StdVec, PushPop) {
	oa::Vec<int> v;
	v.pushBack(1);
	v.pushBack(2);
	ASSERT_EQ(v.size(), 2U);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 2);
	v.popBack();
	EXPECT_EQ(v.size(), 1U);

	constexpr int kStress = 120'000;
	const auto t0 = oa::highResolutionNow();
	std::size_t szOa = 0;
	{
		oa::Vec<int> w;
		w.reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kStress; ++n) {
			w.pushBack(static_cast<int>(rng()));
			if ((n & 7) == 7 && !w.empty()) {
				w.popBack();
			}
		}
		szOa = w.size();
	}
	const auto t1 = oa::highResolutionNow();
	std::size_t szSt = 0;
	{
		std::vector<int> w;
		w.reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kStress; ++n) {
			w.push_back(static_cast<int>(rng()));
			if ((n & 7) == 7 && !w.empty()) {
				w.pop_back();
			}
		}
		szSt = w.size();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Vec::push_back+pop_back (120k)", t0, t1,
		"std::vector::push_back+pop_back (120k)", t2);
	stdExpectGotSize("stress tail size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}

TEST(StdVecVsStd, ParallelPushSequenceIdentical) {
	oa::Vec<int> oa;
	oa.reserve(5000);
	const auto t0 = oa::highResolutionNow();
	std::minstd_rand rng(0xFACADEu);
	for (int i = 0; i < 5000; ++i) {
		const int x = static_cast<int>(rng() & 0x7fffffff);
		oa.pushBack(x);
	}
	const auto t1 = oa::highResolutionNow();
	std::vector<int> st;
	st.reserve(5000);
	std::minstd_rand rng2(0xFACADEu);
	for (int i = 0; i < 5000; ++i) {
		const int x = static_cast<int>(rng2() & 0x7fffffff);
		st.push_back(x);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Vec::pushBack x5000", t0, t1, "std::vector::push_back x5000", t2);
	ASSERT_EQ(oa.size(), st.size());
	stdExpectGotSize("StdVecVsStd::ParallelPush size", st.size(), oa.size());
	EXPECT_TRUE(std::equal(oa.data(), oa.data() + oa.size(), st.begin(), st.end()));
}

TEST(StdVecVsStd, ReserveClearPopPatternMatchesStd) {
	oa::Vec<int> oa;
	std::vector<int> st;
	oa.reserve(100);
	st.reserve(100);
	for (int i = 0; i < 80; ++i) {
		oa.pushBack(i);
		st.push_back(i);
	}
	while (oa.size() > 40) {
		oa.popBack();
		st.pop_back();
	}
	ASSERT_EQ(oa.size(), st.size());
	for (std::size_t i = 0; i < oa.size(); ++i) {
		EXPECT_EQ(oa[i], st[i]) << "i=" << i;
	}
	oa.clear();
	st.clear();
	EXPECT_EQ(oa.size(), 0U);
	EXPECT_TRUE(st.empty());

	const auto t0 = oa::highResolutionNow();
	{
		oa::Vec<int> v;
		v.reserve(100);
		for (int i = 0; i < 80; ++i) {
			v.pushBack(i);
		}
		while (v.size() > 40) {
			v.popBack();
		}
		(void)v.size();
	}
	const auto t1 = oa::highResolutionNow();
	{
		std::vector<int> v;
		v.reserve(100);
		for (int i = 0; i < 80; ++i) {
			v.push_back(i);
		}
		while (v.size() > 40) {
			v.pop_back();
		}
		(void)v.size();
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::Vec reserve/push/pop to size 40", t0, t1,
		"std::vector reserve/push/pop to size 40", t2);
}

TEST(StdVecVsStd, TimedPushPopWallUs) {
	constexpr int kIters = 200'000;
	auto runOa = [] {
		oa::Vec<int> v;
		v.reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kIters; ++n) {
			v.pushBack(static_cast<int>(rng()));
			if ((n & 7) == 7 && !v.empty()) {
				v.popBack();
			}
		}
		return v.size();
	};
	auto runStd = [] {
		std::vector<int> v;
		v.reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kIters; ++n) {
			v.push_back(static_cast<int>(rng()));
			if ((n & 7) == 7 && !v.empty()) {
				v.pop_back();
			}
		}
		return v.size();
	};

	const auto t0 = oa::highResolutionNow();
	const std::size_t szOa = runOa();
	const auto t1 = oa::highResolutionNow();
	const std::size_t szSt = runStd();
	const auto t2 = oa::highResolutionNow();

	stdReportCompareSequentialRuns(
		"oa::Vec::push_back+pop_back (200k)", t0, t1,
		"std::vector::push_back+pop_back (200k)", t2);
	stdExpectGotSize("vec final size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}
