#include "OaStdTest.h"

#include <algorithm>
#include <random>

TEST(OaStdVec, PushPop) {
	OaVec<int> v;
	v.PushBack(1);
	v.PushBack(2);
	ASSERT_EQ(v.Size(), 2U);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 2);
	v.PopBack();
	EXPECT_EQ(v.Size(), 1U);

	constexpr int kStress = 120'000;
	const auto t0 = OaHighResolutionNow();
	std::size_t szOa = 0;
	{
		OaVec<int> w;
		w.Reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kStress; ++n) {
			w.PushBack(static_cast<int>(rng()));
			if ((n & 7) == 7 && !w.Empty()) {
				w.PopBack();
			}
		}
		szOa = w.Size();
	}
	const auto t1 = OaHighResolutionNow();
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
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaVec::push_back+pop_back (120k)", t0, t1,
		"std::vector::push_back+pop_back (120k)", t2);
	OaStdExpectGotSize("stress tail size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}

TEST(OaStdVecVsStd, ParallelPushSequenceIdentical) {
	OaVec<int> oa;
	oa.Reserve(5000);
	const auto t0 = OaHighResolutionNow();
	std::minstd_rand rng(0xFACADEu);
	for (int i = 0; i < 5000; ++i) {
		const int x = static_cast<int>(rng() & 0x7fffffff);
		oa.PushBack(x);
	}
	const auto t1 = OaHighResolutionNow();
	std::vector<int> st;
	st.reserve(5000);
	std::minstd_rand rng2(0xFACADEu);
	for (int i = 0; i < 5000; ++i) {
		const int x = static_cast<int>(rng2() & 0x7fffffff);
		st.push_back(x);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaVec::PushBack x5000", t0, t1, "std::vector::push_back x5000", t2);
	ASSERT_EQ(oa.Size(), st.size());
	OaStdExpectGotSize("OaStdVecVsStd::ParallelPush size", st.size(), oa.Size());
	EXPECT_TRUE(std::equal(oa.Data(), oa.Data() + oa.Size(), st.begin(), st.end()));
}

TEST(OaStdVecVsStd, ReserveClearPopPatternMatchesStd) {
	OaVec<int> oa;
	std::vector<int> st;
	oa.Reserve(100);
	st.reserve(100);
	for (int i = 0; i < 80; ++i) {
		oa.PushBack(i);
		st.push_back(i);
	}
	while (oa.Size() > 40) {
		oa.PopBack();
		st.pop_back();
	}
	ASSERT_EQ(oa.Size(), st.size());
	for (std::size_t i = 0; i < oa.Size(); ++i) {
		EXPECT_EQ(oa[i], st[i]) << "i=" << i;
	}
	oa.Clear();
	st.clear();
	EXPECT_EQ(oa.Size(), 0U);
	EXPECT_TRUE(st.empty());

	const auto t0 = OaHighResolutionNow();
	{
		OaVec<int> v;
		v.Reserve(100);
		for (int i = 0; i < 80; ++i) {
			v.PushBack(i);
		}
		while (v.Size() > 40) {
			v.PopBack();
		}
		(void)v.Size();
	}
	const auto t1 = OaHighResolutionNow();
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
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaVec reserve/push/pop to size 40", t0, t1,
		"std::vector reserve/push/pop to size 40", t2);
}

TEST(OaStdVecVsStd, TimedPushPopWallUs) {
	constexpr int kIters = 200'000;
	auto runOa = [] {
		OaVec<int> v;
		v.Reserve(64);
		std::minstd_rand rng(0xBADC0DEu);
		for (int n = 0; n < kIters; ++n) {
			v.PushBack(static_cast<int>(rng()));
			if ((n & 7) == 7 && !v.Empty()) {
				v.PopBack();
			}
		}
		return v.Size();
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

	const auto t0 = OaHighResolutionNow();
	const std::size_t szOa = runOa();
	const auto t1 = OaHighResolutionNow();
	const std::size_t szSt = runStd();
	const auto t2 = OaHighResolutionNow();

	OaStdReportCompareSequentialRuns(
		"OaVec::push_back+pop_back (200k)", t0, t1,
		"std::vector::push_back+pop_back (200k)", t2);
	OaStdExpectGotSize("vec final size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}
