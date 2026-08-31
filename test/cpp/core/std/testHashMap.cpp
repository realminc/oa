#include "oaStdTest.h"

#include <oa/core/std/typeTraits.h>

#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

TEST(HashMap, TypesAliasMatchesOaStdHashMap) {
	oa::HashMap<int, int> m;
	m.emplace(7, 42);
	EXPECT_EQ(m.at(7), 42);
}

TEST(HashMap, IteratorKeyIsImmutable) {
	oa::HashMap<oa::I32, oa::I32> values;
	values.emplace(7, 9);
	auto iterator = values.begin();
	static_assert(oa::isSameV<decltype((iterator->first)), const oa::I32&>);
	EXPECT_EQ(iterator->first, 7);
	iterator->second = 11;
	EXPECT_EQ(values.at(7), 11);
}

TEST(HashMap, NativeStringHashMatchesViewAndRoutesEqualTextTogether) {
	const oa::String key("alpha");
	EXPECT_EQ(oa::KeyHash<oa::String>{}(key), oa::KeyHash<oa::StringView>{}(key.view()));

	oa::HashMap<oa::String, int> values;
	EXPECT_TRUE(values.emplace(key, 7).second);
	EXPECT_FALSE(values.emplace(oa::String("alpha"), 9).second);
	EXPECT_EQ(values.at(oa::String("alpha")), 7);
}

TEST(HashMap, NativeScalarEnumAndPointerHashAreStable) {
	enum class Key : oa::U32 { Example = 17 };
	int value = 0;
	EXPECT_EQ(oa::KeyHash<int>{}(42), oa::KeyHash<int>{}(42));
	EXPECT_EQ(oa::KeyHash<Key>{}(Key::Example), oa::KeyHash<Key>{}(Key::Example));
	EXPECT_EQ(oa::KeyHash<int*>{}(&value), oa::KeyHash<int*>{}(&value));
}

TEST(HashMap, EmplaceAt) {
	oa::HashMap<int, const char*> m;
	auto [it, inserted] = m.emplace(1, "a");
	ASSERT_TRUE(inserted);
	EXPECT_STREQ(m.at(1), "a");
	(void)it;
}

TEST(HashMap, RangeFor) {
	oa::HashMap<int, int> m;
	m.emplace(1, 10);
	m.emplace(2, 20);
	int sumK = 0;
	int sumV = 0;
	for (const auto& kv : m) {
		sumK += kv.first;
		sumV += kv.second;
	}
	EXPECT_EQ(sumK, 3);
	EXPECT_EQ(sumV, 30);
}

TEST(HashMap, DuplicateInsertEraseAndIteration) {
	oa::HashMap<int, int> m;
	EXPECT_TRUE(m.insert({2, 20}).second);
	EXPECT_FALSE(m.insert({2, 99}).second);
	EXPECT_EQ(m.at(2), 20);
	EXPECT_EQ(m.erase(2), 1u);
	EXPECT_FALSE(m.contains(2));
	m.reserve(64);
	for (int i = 0; i < 50; ++i) {
		m.emplace(i, i * 10);
	}
	int visited = 0;
	for (const auto& item : m) {
		EXPECT_EQ(item.second, item.first * 10);
		++visited;
	}
	EXPECT_EQ(visited, 50);
}

TEST(HashMap, MissingAtIsContractFailure) {
	oa::HashMap<int, int> map;
	EXPECT_DEATH(static_cast<void>(map.at(7)), "OA contract failed: it != end\\(\\)");
}

TEST(HashMap, InsertMovePair) {
	oa::HashMap<int, std::string> m;
	std::string v(32, 'z');
	auto [it, ok] = m.insert({1, std::move(v)});
	ASSERT_TRUE(ok);
	EXPECT_TRUE(v.empty());
	EXPECT_EQ(it->second.size(), 32u);
}

TEST(HashMap, IteratorPostfix) {
	oa::HashMap<int, int> m;
	m.emplace(42, 10);
	auto it = m.begin();
	auto old = it++;
	EXPECT_EQ(old->first, 42);
	EXPECT_EQ(it, m.end());
}

namespace {

struct CollidingHash {
	[[nodiscard]] oa::Usize operator()(int) const noexcept { return 3; }
};

struct ThrowingRehashValue {
	static inline int copiesUntilThrow = -1;
	int value = 0;

	explicit ThrowingRehashValue(int inValue = 0) : value(inValue) {}
	ThrowingRehashValue(const ThrowingRehashValue& inOther)
		: value(inOther.value) {
		if (copiesUntilThrow == 0) {
			throw std::runtime_error("injected HashMap value copy failure");
		}
		if (copiesUntilThrow > 0) --copiesUntilThrow;
	}
	ThrowingRehashValue(ThrowingRehashValue&&) noexcept = default;
	ThrowingRehashValue& operator=(const ThrowingRehashValue&) = default;
	ThrowingRehashValue& operator=(ThrowingRehashValue&&) noexcept = default;
};

struct ThrowingRehashHash {
	static inline int callsUntilThrow = -1;

	[[nodiscard]] oa::Usize operator()(int inValue) const {
		if (callsUntilThrow == 0) {
			throw std::runtime_error("injected HashMap hash failure");
		}
		if (callsUntilThrow > 0) --callsUntilThrow;
		return static_cast<oa::Usize>(inValue);
	}
};

struct ThrowingMoveHashPolicy {
	ThrowingMoveHashPolicy() = default;
	ThrowingMoveHashPolicy(const ThrowingMoveHashPolicy&) = default;
	ThrowingMoveHashPolicy(ThrowingMoveHashPolicy&&) noexcept(false) {}
	[[nodiscard]] oa::Usize operator()(int inValue) const noexcept {
		return static_cast<oa::Usize>(inValue);
	}
};

static_assert(!std::is_move_constructible_v<
	oa::HashMap<int, int, ThrowingMoveHashPolicy>>);

struct ReentrantEraseValue;
using ReentrantEraseMap = oa::HashMap<int, ReentrantEraseValue>;

struct ReentrantEraseValue {
	ReentrantEraseMap* owner = nullptr;
	int key = 0;
	bool* observedAbsent = nullptr;
	bool armed = false;

	ReentrantEraseValue() = default;
	ReentrantEraseValue(const ReentrantEraseValue&) = default;
	ReentrantEraseValue(ReentrantEraseValue&& inOther) noexcept
		: owner(inOther.owner)
		, key(inOther.key)
		, observedAbsent(inOther.observedAbsent)
		, armed(inOther.armed) {
		inOther.armed = false;
	}
	ReentrantEraseValue& operator=(const ReentrantEraseValue&) = default;
	ReentrantEraseValue& operator=(ReentrantEraseValue&&) noexcept = default;

	~ReentrantEraseValue() {
		if (armed) *observedAbsent = !owner->contains(key);
	}
};

} // namespace

TEST(HashMap, CollisionChainSurvivesEraseAndTombstoneReuse) {
	oa::HashMap<int, int, CollidingHash> map;
	for (int key = 0; key < 32; ++key) map.emplace(key, key * 7);

	EXPECT_EQ(map.erase(11), 1U);
	EXPECT_FALSE(map.contains(11));
	for (int key = 0; key < 32; ++key) {
		if (key != 11) EXPECT_EQ(map.at(key), key * 7);
	}

	EXPECT_TRUE(map.emplace(47, 329).second);
	EXPECT_EQ(map.at(47), 329);
	EXPECT_EQ(map.size(), 32U);
}

TEST(HashMap, RehashPreservesEntriesAfterErases) {
	oa::HashMap<int, int> map;
	for (int key = 0; key < 300; ++key) map.emplace(key, key + 5);
	for (int key = 0; key < 300; key += 3) EXPECT_EQ(map.erase(key), 1U);
	map.reserve(2'000);

	for (int key = 0; key < 300; ++key) {
		if (key % 3 == 0) EXPECT_FALSE(map.contains(key));
		else EXPECT_EQ(map.at(key), key + 5);
	}
}

TEST(HashMap, MoveLeavesSourceEmptyAndReusable) {
	oa::HashMap<int, int> source;
	for (int key = 0; key < 24; ++key) source.emplace(key, key * 5);
	EXPECT_EQ(source.erase(7), 1U);

	oa::HashMap<int, int> moved(oa::move(source));
	EXPECT_TRUE(source.empty());
	EXPECT_EQ(source.begin(), source.end());
	EXPECT_TRUE(source.emplace(100, 101).second);
	EXPECT_EQ(source.at(100), 101);
	EXPECT_EQ(moved.size(), 23U);
	EXPECT_EQ(moved.at(9), 45);

	oa::HashMap<int, int> destination;
	destination.emplace(-1, -1);
	destination = oa::move(moved);
	EXPECT_TRUE(moved.empty());
	EXPECT_TRUE(moved.emplace(200, 201).second);
	EXPECT_EQ(destination.size(), 23U);
	EXPECT_EQ(destination.at(9), 45);
}

TEST(HashMap, RehashValueConstructionFailurePreservesOriginalTable) {
	ThrowingRehashValue::copiesUntilThrow = -1;
	oa::HashMap<int, ThrowingRehashValue> map;
	for (int key = 0; key < 20; ++key) {
		map.emplace(key, ThrowingRehashValue(key * 11));
	}
	EXPECT_EQ(map.erase(6), 1U);

	ThrowingRehashValue::copiesUntilThrow = 2;
	EXPECT_THROW(map.reserve(1'024), std::runtime_error);
	ThrowingRehashValue::copiesUntilThrow = -1;

	EXPECT_EQ(map.size(), 19U);
	for (int key = 0; key < 20; ++key) {
		if (key == 6) EXPECT_FALSE(map.contains(key));
		else EXPECT_EQ(map.at(key).value, key * 11);
	}
}

TEST(HashMap, RehashHashFailurePreservesOriginalTable) {
	ThrowingRehashHash::callsUntilThrow = -1;
	oa::HashMap<int, int, ThrowingRehashHash> map;
	for (int key = 0; key < 20; ++key) map.emplace(key, key * 13);

	ThrowingRehashHash::callsUntilThrow = 3;
	EXPECT_THROW(map.reserve(1'024), std::runtime_error);
	ThrowingRehashHash::callsUntilThrow = -1;

	EXPECT_EQ(map.size(), 20U);
	for (int key = 0; key < 20; ++key) EXPECT_EQ(map.at(key), key * 13);
}

TEST(HashMap, ErasePublishesTombstoneBeforeValueDestruction) {
	ReentrantEraseMap map;
	map.reserve(8);
	map.emplace(3, ReentrantEraseValue{});
	bool observedAbsent = false;
	auto& value = map.at(3);
	value.owner = &map;
	value.key = 3;
	value.observedAbsent = &observedAbsent;
	value.armed = true;

	EXPECT_EQ(map.erase(3), 1U);
	EXPECT_TRUE(observedAbsent);
	EXPECT_TRUE(map.empty());
}

TEST(StdHashMapVsStd, AtMatchesUnorderedMapForSameKeys) {
	oa::HashMap<int, int> m;
	std::unordered_map<int, int> u;
	std::minstd_rand rng(0xBEEFu);
	for (int i = 0; i < 300; ++i) {
		const int v = static_cast<int>(rng());
		m.emplace(i, v);
		u.emplace(i, v);
	}
	stdEchoCurrentTest();
	stdExpectGotSize("hash_map size vs unordered_map", u.size(), m.size());
	EXPECT_EQ(m.size(), u.size());
	for (const auto& kv : u) {
		ASSERT_TRUE(m.contains(kv.first));
		EXPECT_EQ(m.at(kv.first), kv.second);
	}
}

TEST(StdHashMapVsStd, TimedInsertWallUs) {
	constexpr int kIters = 40'000;
	const auto t0 = oa::highResolutionNow();
	oa::HashMap<int, int> m;
	m.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		m.emplace(i, i ^ 31);
	}
	const auto t1 = oa::highResolutionNow();
	std::unordered_map<int, int> u;
	u.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		u.emplace(i, i ^ 31);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::HashMap::emplace x40k", t0, t1, "std::unordered_map::emplace x40k", t2);
	stdExpectGotSize("hash_map final size", u.size(), m.size());
	EXPECT_EQ(m.size(), u.size());
}
