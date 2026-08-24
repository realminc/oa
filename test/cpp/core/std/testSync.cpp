#include "oaStdTest.h"

#include <oa/core/std/sync.h>

#include <thread>
#include <utility>
#include <vector>

TEST(Atomic, BasicOps) {
	oa::Atomic<int> a{0};
	EXPECT_EQ(a.load(), 0);
	a.store(5);
	EXPECT_EQ(a.load(), 5);
	EXPECT_EQ(a.exchange(9), 5);
	EXPECT_EQ(a.load(), 9);
	EXPECT_EQ(a.fetchAdd(3), 9);
	EXPECT_EQ(a.load(), 12);

	int expected = 12;
	EXPECT_TRUE(a.compareExchangeStrong(expected, 100));
	EXPECT_EQ(a.load(), 100);

	expected = 999;  // wrong → CAS fails, expected updated to actual
	EXPECT_FALSE(a.compareExchangeStrong(expected, 0));
	EXPECT_EQ(expected, 100);
}

TEST(Atomic, Operators) {
	oa::Atomic<int> a{0};
	++a;
	++a;
	EXPECT_EQ(a.load(), 2);
	a += 10;
	EXPECT_EQ(a.load(), 12);
	a = 7;
	EXPECT_EQ(static_cast<int>(a), 7);
}

TEST(Mutex, CountsConcurrently) {
	oa::Mutex m;
	long long counter = 0;
	auto worker = [&]() {
		for (int i = 0; i < 10000; ++i) {
			oa::ScopedLock<oa::Mutex> lk(m);
			++counter;
		}
	};
	std::vector<std::thread> ts;
	for (int i = 0; i < 4; ++i) {
		ts.emplace_back(worker);
	}
	for (auto& t : ts) {
		t.join();
	}
	EXPECT_EQ(counter, 40000);
}

TEST(SharedMutex, SharedAndExclusive) {
	oa::SharedMutex m;
	{
		oa::SharedLock<oa::SharedMutex> reader(m);
	}
	{
		oa::ScopedLock<oa::SharedMutex> writer(m);
	}
	SUCCEED();
}

TEST(UniqueLock, MoveAndDefer) {
	oa::Mutex m;
	oa::UniqueLock<oa::Mutex> lk(m);
	EXPECT_TRUE(lk.ownsLock());

	oa::UniqueLock<oa::Mutex> lk2(std::move(lk));
	EXPECT_TRUE(lk2.ownsLock());
	EXPECT_FALSE(lk.ownsLock());

	lk2.unlock();
	EXPECT_FALSE(lk2.ownsLock());
	lk2.lock();
	EXPECT_TRUE(lk2.ownsLock());
}
