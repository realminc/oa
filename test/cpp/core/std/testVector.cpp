#include "oaStdTest.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

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

struct alignas(64) OverAlignedValue {
	int value = 0;
};

struct ThrowingMoveValue {
	static inline int alive = 0;
	static inline int moves = 0;
	static inline int doubleDestructions = 0;
	static inline int throwOnMove = 0;

	int value = 0;
	bool live = true;

	explicit ThrowingMoveValue(int inValue = 0) : value(inValue) { ++alive; }
	ThrowingMoveValue(const ThrowingMoveValue&) = delete;
	ThrowingMoveValue& operator=(const ThrowingMoveValue&) = delete;
	ThrowingMoveValue(ThrowingMoveValue&& inOther) : value(inOther.value) {
		if (++moves == throwOnMove) {
			live = false;
			throw std::runtime_error("injected Vector move failure");
		}
		++alive;
		inOther.value = -1;
	}
	ThrowingMoveValue& operator=(ThrowingMoveValue&&) = delete;
	~ThrowingMoveValue() {
		if (!live) {
			++doubleDestructions;
			return;
		}
		live = false;
		--alive;
	}
};

struct ThrowingCopyValue {
	static inline int alive = 0;
	static inline int copies = 0;
	static inline int throwOnCopy = 0;

	explicit ThrowingCopyValue(int inValue = 0) : value(inValue) { ++alive; }
	ThrowingCopyValue(const ThrowingCopyValue& inOther) : value(inOther.value) {
		if (++copies == throwOnCopy) {
			throw std::runtime_error("injected Vector copy failure");
		}
		++alive;
	}
	ThrowingCopyValue(ThrowingCopyValue&&) = delete;
	ThrowingCopyValue& operator=(const ThrowingCopyValue&) = delete;
	ThrowingCopyValue& operator=(ThrowingCopyValue&&) = delete;
	~ThrowingCopyValue() { --alive; }

	int value = 0;
};

struct PaddedEqualityValue {
	char tag = 0;
	int value = 0;

	[[nodiscard]] bool operator==(const PaddedEqualityValue& inOther) const noexcept {
		return tag == inOther.tag && value == inOther.value;
	}
};

static_assert(std::is_trivially_copyable_v<PaddedEqualityValue>);

} // namespace

TEST(StdVector, PushPop) {
	oa::Vector<int> v;
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
		oa::Vector<int> w;
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
		"oa::Vector::push_back+pop_back (120k)", t0, t1,
		"std::vector::push_back+pop_back (120k)", t2);
	stdExpectGotSize("stress tail size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}

TEST(StdVector, IntegralCountDoesNotBindElementPack) {
	oa::Vector<IntegralConstructibleValue> values(4U);
	ASSERT_EQ(values.size(), 4U);
	for (const auto& value : values) EXPECT_EQ(value.value, 0U);
}

TEST(StdVectorVsStd, ParallelPushSequenceIdentical) {
	oa::Vector<int> oa;
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
		"oa::Vector::pushBack x5000", t0, t1, "std::vector::push_back x5000", t2);
	ASSERT_EQ(oa.size(), st.size());
	stdExpectGotSize("StdVectorVsStd::ParallelPush size", st.size(), oa.size());
	EXPECT_TRUE(std::equal(oa.data(), oa.data() + oa.size(), st.begin(), st.end()));
}

TEST(StdVectorVsStd, ReserveClearPopPatternMatchesStd) {
	oa::Vector<int> oa;
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
		oa::Vector<int> v;
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
		"oa::Vector reserve/push/pop to size 40", t0, t1,
		"std::vector reserve/push/pop to size 40", t2);
}

TEST(StdVectorVsStd, TimedPushPopWallUs) {
	constexpr int kIters = 200'000;
	auto runOa = [] {
		oa::Vector<int> v;
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
		"oa::Vector::push_back+pop_back (200k)", t0, t1,
		"std::vector::push_back+pop_back (200k)", t2);
	stdExpectGotSize("vec final size (same RNG)", szSt, szOa);
	EXPECT_EQ(szOa, szSt);
}

TEST(StdVector, ReverseIteratorParity) {
	oa::Vector<int> values{1, 2, 3, 4};
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

	const oa::Vector<int>& constValues = values;
	oa::Vector<int>::const_reverse_iterator constIt = values.rbegin();
	EXPECT_EQ(constIt, constValues.crbegin());
	EXPECT_EQ(*constIt, 4);
}

TEST(StdVector, CheckedAccessUsesAlwaysOnContract) {
	oa::Vector<int> values{7};
	EXPECT_EQ(values.at(0), 7);
	EXPECT_DEATH(static_cast<void>(values.at(1)), "OA contract failed: inidx < size\\(\\)");
}

TEST(StdVector, EmptyFrontBackAndPopUseAlwaysOnContract) {
	oa::Vector<int> values;
	const oa::Vector<int>& constValues = values;
	EXPECT_DEATH(static_cast<void>(values.front()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(static_cast<void>(values.back()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(static_cast<void>(constValues.front()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(static_cast<void>(constValues.back()), "OA contract failed: !empty\\(\\)");
	EXPECT_DEATH(values.popBack(), "OA contract failed: !empty\\(\\)");
}

TEST(StdVector, ReservedStorageTracksContiguousEndPointer) {
	oa::Vector<int> values;
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

TEST(StdVector, EmptyIteratorsAreStableNullSentinels) {
	oa::Vector<int> values;
	EXPECT_EQ(values.begin(), nullptr);
	EXPECT_EQ(values.end(), nullptr);
	EXPECT_EQ(values.cbegin(), nullptr);
	EXPECT_EQ(values.cend(), nullptr);

	const auto inserted = values.insert(values.cbegin(), 9);
	ASSERT_NE(inserted, nullptr);
	EXPECT_EQ(*inserted, 9);
	EXPECT_EQ(values.size(), 1U);
}

TEST(StdVector, NonTrivialGrowthOwnsEveryLifetimeOnce) {
	LifetimeValue::alive = 0;
	LifetimeValue::moved = 0;
	{
		oa::Vector<LifetimeValue> values;
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

TEST(StdVector, SelfAppendRebasesSourceAcrossOverAlignedGrowth) {
	oa::Vector<OverAlignedValue> values;
	values.reserve(3);
	values.pushBack({11});
	values.pushBack({22});
	values.pushBack({33});

	values.append(values.data(), values.size());

	ASSERT_EQ(values.size(), 6U);
	for (oa::Usize index = 0; index < values.size(); ++index) {
		EXPECT_EQ(values[index].value, values[index % 3U].value);
	}
}

TEST(StdVector, AppendRejectsUnusedOrPartiallyLiveOwnedStorage) {
	oa::Vector<OverAlignedValue> values;
	values.reserve(3);
	values.pushBack({11});
	values.pushBack({22});

	EXPECT_DEATH(values.append(values.data() + values.size(), 1),
		"OA contract failed: isWhollyLive");
	EXPECT_DEATH(values.append(values.data() + 1, 2),
		"OA contract failed: isWhollyLive");
}

TEST(StdVector, AppendRejectsWrappedSourceRange) {
	oa::Vector<int> values{1};
	const auto impossibleAddress = static_cast<oa::Usize>(-1) - 1U;
	const auto* const impossible = reinterpret_cast<const int*>(impossibleAddress);
	EXPECT_DEATH(values.append(impossible, 1),
		"OA contract failed: sourceBytes <=");
}

TEST(StdVector, AliasedPushBackSurvivesOverAlignedGrowth) {
	oa::Vector<OverAlignedValue> values;
	values.reserve(1);
	values.pushBack({73});

	values.pushBack(values[0]);

	ASSERT_EQ(values.size(), 2U);
	EXPECT_EQ(values[0].value, 73);
	EXPECT_EQ(values[1].value, 73);
}

TEST(StdVector, ThrowingMoveDuringGrowthDoesNotLeakOrDoubleDestroy) {
	ThrowingMoveValue::alive = 0;
	ThrowingMoveValue::moves = 0;
	ThrowingMoveValue::doubleDestructions = 0;
	ThrowingMoveValue::throwOnMove = 2;
	{
		oa::Vector<ThrowingMoveValue> values;
		values.reserve(2);
		values.emplaceBack(17);
		values.emplaceBack(29);

		EXPECT_THROW(values.reserve(4), std::runtime_error);
		EXPECT_EQ(values.size(), 2U);
		EXPECT_EQ(ThrowingMoveValue::alive, 2);
		EXPECT_EQ(ThrowingMoveValue::doubleDestructions, 0);
	}
	EXPECT_EQ(ThrowingMoveValue::alive, 0);
	EXPECT_EQ(ThrowingMoveValue::doubleDestructions, 0);
}

TEST(StdVector, ThrowingElementCopyDuringConstructionReleasesBuiltPrefix) {
	ThrowingCopyValue::alive = 0;
	ThrowingCopyValue::copies = 0;
	ThrowingCopyValue::throwOnCopy = 3;
	{
		ThrowingCopyValue source(41);
		EXPECT_THROW((oa::Vector<ThrowingCopyValue>(5, source)), std::runtime_error);
		EXPECT_EQ(ThrowingCopyValue::alive, 1);
	}
	EXPECT_EQ(ThrowingCopyValue::alive, 0);
}

TEST(StdVector, ThrowingRandomAccessRangeConstructionReleasesBuiltPrefix) {
	ThrowingCopyValue::alive = 0;
	ThrowingCopyValue::copies = 0;
	ThrowingCopyValue::throwOnCopy = 0;
	{
		ThrowingCopyValue source[] = {
			ThrowingCopyValue(11),
			ThrowingCopyValue(22),
			ThrowingCopyValue(33),
			ThrowingCopyValue(44),
		};
		ASSERT_EQ(ThrowingCopyValue::alive, 4);
		ThrowingCopyValue::copies = 0;
		ThrowingCopyValue::throwOnCopy = 3;

		EXPECT_THROW((oa::Vector<ThrowingCopyValue>(source, source + 4)), std::runtime_error);
		EXPECT_EQ(ThrowingCopyValue::alive, 4);
	}
	EXPECT_EQ(ThrowingCopyValue::alive, 0);
}

TEST(StdVector, AssignHandlesSelfRangesAndAliasedValues) {
	LifetimeValue::alive = 0;
	LifetimeValue::moved = 0;
	{
		oa::Vector<LifetimeValue> values;
		values.reserve(4);
		values.emplaceBack(10);
		values.emplaceBack(20);
		values.emplaceBack(30);

		values.assign(values.begin() + 1, values.end());
		ASSERT_EQ(values.size(), 2U);
		EXPECT_EQ(values[0].value, 20);
		EXPECT_EQ(values[1].value, 30);
		EXPECT_EQ(LifetimeValue::alive, 2);

		values.assign(4, values[1]);
		ASSERT_EQ(values.size(), 4U);
		for (const LifetimeValue& value : values) EXPECT_EQ(value.value, 30);
		EXPECT_EQ(LifetimeValue::alive, 4);

		oa::Vector<int> integers;
		integers.assign(3, 7);
		EXPECT_EQ(integers, (oa::Vector<int>{7, 7, 7}));
	}
	EXPECT_EQ(LifetimeValue::alive, 0);
}

TEST(StdVector, AssignRetainsOriginalWhenElementCopyThrows) {
	ThrowingCopyValue::alive = 0;
	ThrowingCopyValue::copies = 0;
	ThrowingCopyValue::throwOnCopy = 0;
	{
		ThrowingCopyValue source[] = {
			ThrowingCopyValue(1),
			ThrowingCopyValue(2),
			ThrowingCopyValue(3),
			ThrowingCopyValue(4),
		};
		oa::Vector<ThrowingCopyValue> values;
		values.reserve(2);
		values.emplaceBack(17);
		values.emplaceBack(29);
		const int baselineAlive = ThrowingCopyValue::alive;

		ThrowingCopyValue::copies = 0;
		ThrowingCopyValue::throwOnCopy = 2;
		EXPECT_THROW(values.assign(source, source + 4), std::runtime_error);
		ASSERT_EQ(values.size(), 2U);
		EXPECT_EQ(values[0].value, 17);
		EXPECT_EQ(values[1].value, 29);
		EXPECT_EQ(ThrowingCopyValue::alive, baselineAlive);

		ThrowingCopyValue::copies = 0;
		ThrowingCopyValue::throwOnCopy = 2;
		EXPECT_THROW(values.assign(4, values[0]), std::runtime_error);
		ASSERT_EQ(values.size(), 2U);
		EXPECT_EQ(values[0].value, 17);
		EXPECT_EQ(values[1].value, 29);
		EXPECT_EQ(ThrowingCopyValue::alive, baselineAlive);
	}
	EXPECT_EQ(ThrowingCopyValue::alive, 0);
}

TEST(StdVectorVsStd, EqualityUsesElementSemanticsForFloatingPointAndPadding) {
	const oa::Vector<float> oaNegativeZero{-0.0F};
	const oa::Vector<float> oaPositiveZero{0.0F};
	const std::vector<float> stdNegativeZero{-0.0F};
	const std::vector<float> stdPositiveZero{0.0F};
	EXPECT_EQ(oaNegativeZero == oaPositiveZero, stdNegativeZero == stdPositiveZero);

	const float nan = std::numeric_limits<float>::quiet_NaN();
	const oa::Vector<float> oaNanA{nan};
	const oa::Vector<float> oaNanB{nan};
	const std::vector<float> stdNanA{nan};
	const std::vector<float> stdNanB{nan};
	EXPECT_EQ(oaNanA == oaNanB, stdNanA == stdNanB);
	EXPECT_EQ(oaNanA == oaNanA, stdNanA == stdNanA);

	PaddedEqualityValue paddedA;
	PaddedEqualityValue paddedB;
	std::memset(&paddedA, 0xA5, sizeof(paddedA));
	std::memset(&paddedB, 0x5A, sizeof(paddedB));
	paddedA.tag = paddedB.tag = 'p';
	paddedA.value = paddedB.value = 73;
	const oa::Vector<PaddedEqualityValue> oaPaddedA{paddedA};
	const oa::Vector<PaddedEqualityValue> oaPaddedB{paddedB};
	const std::vector<PaddedEqualityValue> stdPaddedA{paddedA};
	const std::vector<PaddedEqualityValue> stdPaddedB{paddedB};
	EXPECT_EQ(oaPaddedA == oaPaddedB, stdPaddedA == stdPaddedB);
}

TEST(StdVectorVsStd, PointerValueInitializationProducesNullValues) {
	int sentinel = 7;
	oa::Vector<int*> oaValues(3);
	std::vector<int*> stdValues(3);
	oaValues.resize(6);
	stdValues.resize(6);
	ASSERT_EQ(oaValues.size(), stdValues.size());
	for (oa::Usize index = 0; index < oaValues.size(); ++index) {
		EXPECT_EQ(oaValues[index], stdValues[index]);
		EXPECT_EQ(oaValues[index], nullptr);
	}
	oaValues.pushBack(&sentinel);
	stdValues.push_back(&sentinel);
	EXPECT_EQ(oaValues.back(), stdValues.back());
}
