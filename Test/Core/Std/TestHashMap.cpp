#include "OaStdTest.h"

#include <random>
#include <string>
#include <unordered_map>

TEST(OaHashMap, TypesAliasMatchesOaStdHashMap) {
	OaHashMap<int, int> m;
	m.Emplace(7, 42);
	EXPECT_EQ(m.At(7), 42);
}

TEST(OaStdHashMap, EmplaceAt) {
	OaStdHashMap<int, const char*> m;
	auto [it, inserted] = m.Emplace(1, "a");
	ASSERT_TRUE(inserted);
	EXPECT_STREQ(m.At(1), "a");
	(void)it;
}

TEST(OaStdHashMap, RangeFor) {
	OaStdHashMap<int, int> m;
	m.Emplace(1, 10);
	m.Emplace(2, 20);
	int sumK = 0;
	int sumV = 0;
	for (const auto& kv : m) {
		sumK += kv.first;
		sumV += kv.second;
	}
	EXPECT_EQ(sumK, 3);
	EXPECT_EQ(sumV, 30);
}

TEST(OaStdHashMap, DuplicateInsertEraseStdMap) {
	OaStdHashMap<int, int> m;
	EXPECT_TRUE(m.Insert({2, 20}).second);
	EXPECT_FALSE(m.Insert({2, 99}).second);
	EXPECT_EQ(m.At(2), 20);
	EXPECT_EQ(m.Erase(2), 1u);
	EXPECT_FALSE(m.Contains(2));
	m.Reserve(64);
	for (int i = 0; i < 50; ++i) {
		m.Emplace(i, i * 10);
	}
	auto stdm = m.StdMap();
	EXPECT_EQ(stdm.size(), 50u);
	EXPECT_EQ(stdm.at(7), 70);
}

TEST(OaStdHashMap, InsertMovePair) {
	OaStdHashMap<int, std::string> m;
	std::string v(32, 'z');
	auto [it, ok] = m.Insert({1, std::move(v)});
	ASSERT_TRUE(ok);
	EXPECT_TRUE(v.empty());
	EXPECT_EQ(it->second.size(), 32u);
}

TEST(OaStdHashMap, IteratorPostfix) {
	OaStdHashMap<int, int> m;
	m.Emplace(42, 10);
	auto it = m.Begin();
	auto old = it++;
	EXPECT_EQ(old->first, 42);
	EXPECT_EQ(it, m.End());
}

TEST(OaStdHashMapVsStd, AtMatchesUnorderedMapForSameKeys) {
	OaStdHashMap<int, int> m;
	std::unordered_map<int, int> u;
	std::minstd_rand rng(0xBEEFu);
	for (int i = 0; i < 300; ++i) {
		const int v = static_cast<int>(rng());
		m.Emplace(i, v);
		u.emplace(i, v);
	}
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("hash_map size vs unordered_map", u.size(), m.Size());
	EXPECT_EQ(m.Size(), u.size());
	for (const auto& kv : u) {
		ASSERT_TRUE(m.Contains(kv.first));
		EXPECT_EQ(m.At(kv.first), kv.second);
	}
}

TEST(OaStdHashMapVsStd, TimedInsertWallUs) {
	constexpr int kIters = 40'000;
	const auto t0 = OaHighResolutionNow();
	OaStdHashMap<int, int> m;
	m.Reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		m.Emplace(i, i ^ 31);
	}
	const auto t1 = OaHighResolutionNow();
	std::unordered_map<int, int> u;
	u.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		u.emplace(i, i ^ 31);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdHashMap::Emplace x40k", t0, t1, "std::unordered_map::emplace x40k", t2);
	OaStdExpectGotSize("hash_map final size", u.size(), m.Size());
	EXPECT_EQ(m.Size(), u.size());
}
