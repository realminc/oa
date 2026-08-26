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

TEST(Atomic, MemoryOrdersAndWeakCompareExchange) {
	oa::Atomic<int> value{1};
	value.store(2, oa::MemoryOrder::Release);
	EXPECT_EQ(value.load(oa::MemoryOrder::Acquire), 2);

	int expected = 2;
	while (not value.compareExchangeWeak(
		expected, 3, oa::MemoryOrder::AcquireRelease)) {
		expected = 2;
	}
	EXPECT_EQ(value.load(oa::MemoryOrder::Relaxed), 3);
	oa::atomicThreadFence(oa::MemoryOrder::Sequential);
}

TEST(Atomic, IntegralFetchAndDecrementOps) {
	oa::Atomic<unsigned> value{12U};
	EXPECT_EQ(value.fetchOr(3U), 12U);
	EXPECT_EQ(value.load(), 15U);
	EXPECT_EQ(value.fetchAnd(10U), 15U);
	EXPECT_EQ(value.load(), 10U);
	EXPECT_EQ(value.fetchXor(15U), 10U);
	EXPECT_EQ(value.load(), 5U);
	EXPECT_EQ(value.fetchSub(2U), 5U);
	EXPECT_EQ(value.load(), 3U);
	EXPECT_EQ(value--, 3U);
	EXPECT_EQ(--value, 1U);
	EXPECT_EQ(value -= 1U, 0U);
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

TEST(Mutex, TryLockReportsOwnership) {
	oa::Mutex mutex;
	EXPECT_TRUE(mutex.tryLock());
	EXPECT_FALSE(mutex.tryLock());
	mutex.unlock();
	EXPECT_TRUE(mutex.tryLock());
	mutex.unlock();
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

TEST(SharedMutex, ReadersExcludeWriter) {
	oa::SharedMutex mutex;
	oa::Atomic<bool> readerLocked{false};
	oa::Atomic<bool> releaseReader{false};

	std::thread reader([&] {
		mutex.lockShared();
		readerLocked.store(true, oa::MemoryOrder::Release);
		while (!releaseReader.load(oa::MemoryOrder::Acquire)) {
			std::this_thread::yield();
		}
		mutex.unlockShared();
	});
	while (!readerLocked.load(oa::MemoryOrder::Acquire)) {
		std::this_thread::yield();
	}

	EXPECT_TRUE(mutex.tryLockShared());
	mutex.unlockShared();
	EXPECT_FALSE(mutex.tryLock());
	releaseReader.store(true, oa::MemoryOrder::Release);
	reader.join();

	EXPECT_TRUE(mutex.tryLock());
	mutex.unlock();
}

TEST(Condition, PredicateWaitAndNotifyOne) {
	oa::Mutex mutex;
	oa::Condition condition;
	oa::Atomic<bool> entered{false};
	bool released = false;

	std::thread waiter([&] {
		oa::UniqueLock<oa::Mutex> lock(mutex);
		entered.store(true, oa::MemoryOrder::Release);
		condition.wait(lock, [&] { return released; });
	});
	while (!entered.load(oa::MemoryOrder::Acquire)) {
		std::this_thread::yield();
	}
	{
		oa::ScopedLock<oa::Mutex> lock(mutex);
		released = true;
	}
	condition.notifyOne();
	waiter.join();
	EXPECT_TRUE(released);
}

TEST(Condition, TimedWaitReportsTimeoutAndKeepsLock) {
	oa::Mutex mutex;
	oa::Condition condition;
	oa::UniqueLock<oa::Mutex> lock(mutex);

	EXPECT_FALSE(condition.waitFor(
		lock, oa::Duration::fromMilliseconds(2)));
	EXPECT_TRUE(lock.ownsLock());
}

TEST(Condition, TimedPredicateWaitObservesNotification) {
	oa::Mutex mutex;
	oa::Condition condition;
	bool released = false;
	std::thread notifier([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
		{
			oa::ScopedLock<oa::Mutex> lock(mutex);
			released = true;
		}
		condition.notifyOne();
	});

	oa::UniqueLock<oa::Mutex> lock(mutex);
	EXPECT_TRUE(condition.waitFor(
		lock, oa::Duration::fromMilliseconds(100), [&] { return released; }));
	lock.unlock();
	notifier.join();
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
