#include "OaStdTest.h"

#include <random>
#include <unordered_set>

TEST(OaStdHashSet, InsertContains) {
	OaStdHashSet<int> s;
	EXPECT_TRUE(s.Insert(5).second);
	EXPECT_TRUE(s.Contains(5));
}

TEST(OaStdHashSet, RangeFor) {
	OaStdHashSet<int> s;
	s.Insert(1);
	s.Insert(2);
	s.Insert(3);
	int sum = 0;
	for (int x : s) {
		sum += x;
	}
	EXPECT_EQ(sum, 6);
}

TEST(OaStdHashSet, DuplicateAndStdSet) {
	OaStdHashSet<int> s;
	EXPECT_TRUE(s.Insert(3).second);
	EXPECT_FALSE(s.Insert(3).second);
	EXPECT_EQ(s.Erase(3), 1u);
	for (int i = 0; i < 40; ++i) {
		s.Insert(i);
	}
	auto st = s.StdSet();
	EXPECT_EQ(st.size(), 40u);
	EXPECT_TRUE(st.count(17));
}

TEST(OaStdHashSet, IteratorPostfix) {
	OaStdHashSet<int> s;
	s.Insert(7);
	auto it = s.Begin();
	auto old = it++;
	EXPECT_EQ(*old, 7);
	EXPECT_EQ(it, s.End());
}

TEST(OaStdHashSetVsStd, SameInsertsAsUnorderedSet) {
	OaStdHashSet<int> oa;
	std::unordered_set<int> st;
	std::minstd_rand rng(0xF00Du);
	for (int n = 0; n < 200; ++n) {
		const int v = static_cast<int>(rng() % 400);
		oa.Insert(v);
		st.insert(v);
	}
	OaStdEchoCurrentTest();
	OaStdExpectGotSize("hash_set size vs unordered_set", st.size(), oa.Size());
	EXPECT_EQ(oa.Size(), st.size());
	for (int x : st) {
		EXPECT_TRUE(oa.Contains(x));
	}
}

TEST(OaStdHashSetVsStd, TimedInsertWallUs) {
	constexpr int kIters = 40'000;
	const auto t0 = OaHighResolutionNow();
	OaStdHashSet<int> oa;
	oa.Reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		oa.Insert(i ^ 13);
	}
	const auto t1 = OaHighResolutionNow();
	std::unordered_set<int> st;
	st.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		st.insert(i ^ 13);
	}
	const auto t2 = OaHighResolutionNow();
	OaStdReportCompareSequentialRuns(
		"OaStdHashSet::Insert x40k", t0, t1, "std::unordered_set::insert x40k", t2);
	OaStdExpectGotSize("hash_set final size", st.size(), oa.Size());
	EXPECT_EQ(oa.Size(), st.size());
}
