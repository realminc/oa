#include "oaStdTest.h"

#include <random>
#include <string>
#include <unordered_map>

TEST(HashMap, TypesAliasMatchesOaStdHashMap) {
	oa::HashMap<int, int> m;
	m.emplace(7, 42);
	EXPECT_EQ(m.at(7), 42);
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

TEST(HashMap, DuplicateInsertEraseStdMap) {
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
	auto stdm = m.stdMap();
	EXPECT_EQ(stdm.size(), 50u);
	EXPECT_EQ(stdm.at(7), 70);
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
