#include "oaStdTest.h"

#include <oa/core/std/typeTraits.h>

#include <random>
#include <unordered_set>

TEST(HashSet, InsertContains) {
	oa::HashSet<int> s;
	EXPECT_TRUE(s.insert(5).second);
	EXPECT_TRUE(s.contains(5));
}

TEST(HashSet, IteratorKeyIsImmutable) {
	oa::HashSet<oa::I32> values;
	values.insert(7);
	auto iterator = values.begin();
	static_assert(oa::IsSameV<decltype(*iterator), const oa::I32&>);
	EXPECT_EQ(*iterator, 7);
}

TEST(HashSet, RangeFor) {
	oa::HashSet<int> s;
	s.insert(1);
	s.insert(2);
	s.insert(3);
	int sum = 0;
	for (int x : s) {
		sum += x;
	}
	EXPECT_EQ(sum, 6);
}

TEST(HashSet, DuplicateEraseAndIteration) {
	oa::HashSet<int> s;
	EXPECT_TRUE(s.insert(3).second);
	EXPECT_FALSE(s.insert(3).second);
	EXPECT_EQ(s.erase(3), 1u);
	for (int i = 0; i < 40; ++i) {
		s.insert(i);
	}
	int visited = 0;
	for (int value : s) {
		EXPECT_TRUE(value >= 0 && value < 40);
		++visited;
	}
	EXPECT_EQ(visited, 40);
}

TEST(HashSet, IteratorPostfix) {
	oa::HashSet<int> s;
	s.insert(7);
	auto it = s.begin();
	auto old = it++;
	EXPECT_EQ(*old, 7);
	EXPECT_EQ(it, s.end());
}

TEST(HashSet, MoveLeavesSourceEmptyAndReusable) {
	oa::HashSet<int> source;
	for (int value = 0; value < 24; ++value) source.insert(value);
	EXPECT_EQ(source.erase(7), 1U);

	oa::HashSet<int> moved(oa::move(source));
	EXPECT_TRUE(source.empty());
	EXPECT_EQ(source.begin(), source.end());
	EXPECT_TRUE(source.insert(100).second);
	EXPECT_TRUE(source.contains(100));
	EXPECT_EQ(moved.size(), 23U);
	EXPECT_TRUE(moved.contains(9));

	oa::HashSet<int> destination;
	destination.insert(-1);
	destination = oa::move(moved);
	EXPECT_TRUE(moved.empty());
	EXPECT_TRUE(moved.insert(200).second);
	EXPECT_EQ(destination.size(), 23U);
	EXPECT_TRUE(destination.contains(9));
}

TEST(StdHashSetVsStd, SameInsertsAsUnorderedSet) {
	oa::HashSet<int> oa;
	std::unordered_set<int> st;
	std::minstd_rand rng(0xF00Du);
	for (int n = 0; n < 200; ++n) {
		const int v = static_cast<int>(rng() % 400);
		oa.insert(v);
		st.insert(v);
	}
	stdEchoCurrentTest();
	stdExpectGotSize("hash_set size vs unordered_set", st.size(), oa.size());
	EXPECT_EQ(oa.size(), st.size());
	for (int x : st) {
		EXPECT_TRUE(oa.contains(x));
	}
}

TEST(StdHashSetVsStd, TimedInsertWallUs) {
	constexpr int kIters = 40'000;
	const auto t0 = oa::highResolutionNow();
	oa::HashSet<int> oa;
	oa.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		oa.insert(i ^ 13);
	}
	const auto t1 = oa::highResolutionNow();
	std::unordered_set<int> st;
	st.reserve(static_cast<std::size_t>(kIters));
	for (int i = 0; i < kIters; ++i) {
		st.insert(i ^ 13);
	}
	const auto t2 = oa::highResolutionNow();
	stdReportCompareSequentialRuns(
		"oa::HashSet::insert x40k", t0, t1, "std::unordered_set::insert x40k", t2);
	stdExpectGotSize("hash_set final size", st.size(), oa.size());
	EXPECT_EQ(oa.size(), st.size());
}
