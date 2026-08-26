#include "oaStdTest.h"

#include <algorithm>
#include <random>

namespace {

struct LifetimeValue {
	static inline int alive = 0;
	static inline int moved = 0;

	int value = 0;

	LifetimeValue() { ++alive; }
	explicit LifetimeValue(int inValue) : value(inValue) { ++alive; }
	LifetimeValue(const LifetimeValue& inOther) : value(inOther.value) { ++alive; }
	LifetimeValue(LifetimeValue&& inOther) noexcept : value(inOther.value) {
		++alive;
		++moved;
		inOther.value = -1;
	}
	LifetimeValue& operator=(const LifetimeValue&) = default;
	LifetimeValue& operator=(LifetimeValue&&) = default;
	~LifetimeValue() { --alive; }
};

struct IntegralConstructibleValue {
	explicit IntegralConstructibleValue(unsigned int inValue = 0U)
		: value(inValue) {}

	unsigned int value = 0U;
};

} // namespace

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

TEST(StdVec, IntegralCountDoesNotBindElementPack) {
	oa::Vec<IntegralConstructibleValue> values(4U);
	ASSERT_EQ(values.size(), 4U);
	for (const auto& value : values) EXPECT_EQ(value.value, 0U);
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

TEST(StdVec, ReverseIteratorParity) {
	oa::Vec<int> values{1, 2, 3, 4};
	auto it = values.rbegin();
	EXPECT_EQ(*it, 4);
	EXPECT_EQ(it[1], 3);
	EXPECT_EQ(*(it + 2), 2);
	EXPECT_EQ(values.rend() - values.rbegin(), 4);
	EXPECT_TRUE(values.rbegin() < values.rend());
	++it;
	EXPECT_EQ(*it, 3);
	--it;
	EXPECT_EQ(*it, 4);

	const oa::Vec<int>& constValues = values;
	oa::Vec<int>::const_reverse_iterator constIt = values.rbegin();
	EXPECT_EQ(constIt, constValues.crbegin());
	EXPECT_EQ(*constIt, 4);
}

TEST(StdVec, CheckedAccessUsesAlwaysOnContract) {
	oa::Vec<int> values{7};
	EXPECT_EQ(values.at(0), 7);
	EXPECT_DEATH(static_cast<void>(values.at(1)), "OA contract failed: inidx < size\\(\\)");
}

TEST(StdVec, ReservedStorageTracksContiguousEndPointer) {
	oa::Vec<int> values;
	values.reserve(32);
	ASSERT_NE(values.data(), nullptr);
	EXPECT_EQ(values.begin(), values.end());
	EXPECT_EQ(values.capacity(), 32U);

	for (int value = 0; value < 32; ++value) {
		values.pushBack(value);
	}

	EXPECT_EQ(values.size(), 32U);
	EXPECT_EQ(values.end(), values.data() + 32);
	EXPECT_EQ(values.back(), 31);
}

TEST(StdVec, EmptyIteratorsAreStableNullSentinels) {
	oa::Vec<int> values;
	EXPECT_EQ(values.begin(), nullptr);
	EXPECT_EQ(values.end(), nullptr);
	EXPECT_EQ(values.cbegin(), nullptr);
	EXPECT_EQ(values.cend(), nullptr);

	const auto inserted = values.insert(values.cbegin(), 9);
	ASSERT_NE(inserted, nullptr);
	EXPECT_EQ(*inserted, 9);
	EXPECT_EQ(values.size(), 1U);
}

TEST(StdVec, NonTrivialGrowthOwnsEveryLifetimeOnce) {
	LifetimeValue::alive = 0;
	LifetimeValue::moved = 0;
	{
		oa::Vec<LifetimeValue> values;
		for (int index = 0; index < 100; ++index) {
			values.emplaceBack(index);
		}
		ASSERT_EQ(values.size(), 100U);
		EXPECT_EQ(LifetimeValue::alive, 100);
		EXPECT_GT(LifetimeValue::moved, 0);
		for (int index = 0; index < 100; ++index) {
			EXPECT_EQ(values[static_cast<oa::Usize>(index)].value, index);
		}
		values.resize(37);
		EXPECT_EQ(LifetimeValue::alive, 37);
	}
	EXPECT_EQ(LifetimeValue::alive, 0);
}
