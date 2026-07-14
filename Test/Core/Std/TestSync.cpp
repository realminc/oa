#include "OaStdTest.h"

#include <Oa/Core/Std/Sync.h>

#include <thread>
#include <utility>
#include <vector>

TEST(OaStdAtomic, BasicOps) {
	OaStdAtomic<int> a{0};
	EXPECT_EQ(a.Load(), 0);
	a.Store(5);
	EXPECT_EQ(a.Load(), 5);
	EXPECT_EQ(a.Exchange(9), 5);
	EXPECT_EQ(a.Load(), 9);
	EXPECT_EQ(a.FetchAdd(3), 9);
	EXPECT_EQ(a.Load(), 12);

	int expected = 12;
	EXPECT_TRUE(a.CompareExchangeStrong(expected, 100));
	EXPECT_EQ(a.Load(), 100);

	expected = 999;  // wrong → CAS fails, expected updated to actual
	EXPECT_FALSE(a.CompareExchangeStrong(expected, 0));
	EXPECT_EQ(expected, 100);
}

TEST(OaStdAtomic, Operators) {
	OaStdAtomic<int> a{0};
	++a;
	++a;
	EXPECT_EQ(a.Load(), 2);
	a += 10;
	EXPECT_EQ(a.Load(), 12);
	a = 7;
	EXPECT_EQ(static_cast<int>(a), 7);
}

TEST(OaStdMutex, CountsConcurrently) {
	OaStdMutex m;
	long long counter = 0;
	auto worker = [&]() {
		for (int i = 0; i < 10000; ++i) {
			OaStdScopedLock<OaStdMutex> lk(m);
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

TEST(OaStdSharedMutex, SharedAndExclusive) {
	OaStdSharedMutex m;
	{
		OaStdSharedLock<OaStdSharedMutex> reader(m);
	}
	{
		OaStdScopedLock<OaStdSharedMutex> writer(m);
	}
	SUCCEED();
}

TEST(OaStdUniqueLock, MoveAndDefer) {
	OaStdMutex m;
	OaStdUniqueLock<OaStdMutex> lk(m);
	EXPECT_TRUE(lk.OwnsLock());

	OaStdUniqueLock<OaStdMutex> lk2(std::move(lk));
	EXPECT_TRUE(lk2.OwnsLock());
	EXPECT_FALSE(lk.OwnsLock());

	lk2.Unlock();
	EXPECT_FALSE(lk2.OwnsLock());
	lk2.Lock();
	EXPECT_TRUE(lk2.OwnsLock());
}
