#include <gtest/gtest.h>

#include <Oa/Core/Thread.h>

#include <atomic>
#include <chrono>
#include <thread>

// =============================================================================
// OaChannel
// =============================================================================

TEST(Channel, SendRecv) {
	OaChannel<OaI32> ch(4);
	EXPECT_TRUE(ch.Send(42));
	EXPECT_TRUE(ch.Send(99));
	EXPECT_EQ(ch.Size(), 2);

	auto v1 = ch.Recv();
	auto v2 = ch.Recv();
	ASSERT_TRUE(v1.has_value());
	ASSERT_TRUE(v2.has_value());
	EXPECT_EQ(*v1, 42);
	EXPECT_EQ(*v2, 99);
	EXPECT_EQ(ch.Size(), 0);
}

TEST(Channel, TrySendRecv) {
	OaChannel<OaI32> ch(2);
	EXPECT_TRUE(ch.TrySend(1));
	EXPECT_TRUE(ch.TrySend(2));
	EXPECT_FALSE(ch.TrySend(3));

	auto v = ch.TryRecv();
	ASSERT_TRUE(v.has_value());
	EXPECT_EQ(*v, 1);

	EXPECT_TRUE(ch.TrySend(3));
}

TEST(Channel, CloseUnblocksRecv) {
	OaChannel<OaI32> ch(4);
	std::atomic<bool> gotNullopt{false};

	std::thread t([&] {
		auto v = ch.Recv();
		if (!v.has_value()) gotNullopt.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	ch.Close();
	t.join();
	EXPECT_TRUE(gotNullopt.load());
	EXPECT_TRUE(ch.IsClosed());
}

TEST(Channel, CloseUnblocksSend) {
	OaChannel<OaI32> ch(1);
	ch.Send(1);
	std::atomic<bool> sendFailed{false};

	std::thread t([&] {
		bool ok = ch.Send(2);
		if (!ok) sendFailed.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	ch.Close();
	t.join();
	EXPECT_TRUE(sendFailed.load());
}

TEST(Channel, Mpmc) {
	OaChannel<OaI32> ch(64);
	constexpr OaI32 kProducers = 4;
	constexpr OaI32 kPerProducer = 250;
	constexpr OaI32 kTotal = kProducers * kPerProducer;

	std::atomic<OaI32> sum{0};
	OaVec<std::thread> producers;
	OaVec<std::thread> consumers;

	for (OaI32 p = 0; p < kProducers; ++p) {
		producers.EmplaceBack([&, p] {
			for (OaI32 i = 0; i < kPerProducer; ++i) {
				ch.Send(p * kPerProducer + i + 1);
			}
		});
	}

	for (OaI32 c = 0; c < 2; ++c) {
		consumers.EmplaceBack([&] {
			while (true) {
				auto v = ch.Recv();
				if (!v.has_value()) break;
				sum.fetch_add(*v, std::memory_order_relaxed);
			}
		});
	}

	for (auto& t : producers) t.join();

	// Wait for all items to be consumed
	while (ch.Size() > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	ch.Close();

	for (auto& t : consumers) t.join();

	OaI32 expected = kTotal * (kTotal + 1) / 2;
	EXPECT_EQ(sum.load(), expected);
}

// =============================================================================
// OaTask
// =============================================================================

TEST(Task, CompleteWait) {
	auto task = OaMakeSharedPtr<OaTask<OaI32>>();
	std::thread t([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		task->Complete(42);
	});

	auto result = task->Wait();
	t.join();
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, 42);
	EXPECT_TRUE(task->IsDone());
	EXPECT_FALSE(task->HasFailed());
}

TEST(Task, Fail) {
	auto task = OaMakeSharedPtr<OaTask<OaI32>>();
	task->Fail(OaStatus::Error("test error"));
	EXPECT_TRUE(task->IsDone());
	EXPECT_TRUE(task->HasFailed());

	auto result = task->Wait();
	EXPECT_FALSE(result.has_value());
}

TEST(Task, TryGet) {
	auto task = OaMakeSharedPtr<OaTask<OaI32>>();
	EXPECT_FALSE(task->TryGet().has_value());

	task->Complete(7);
	auto v = task->TryGet();
	ASSERT_TRUE(v.has_value());
	EXPECT_EQ(*v, 7);
}

TEST(Task, Then) {
	auto task = OaMakeSharedPtr<OaTask<OaI32>>();
	auto doubled = task->Then([](OaI32 v) { return v * 2; });

	task->Complete(21);
	auto result = doubled->Wait();
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, 42);
}

TEST(Task, ThenAlreadyComplete) {
	auto task = OaMakeSharedPtr<OaTask<OaI32>>();
	task->Complete(10);
	auto tripled = task->Then([](OaI32 v) { return v * 3; });

	auto result = tripled->Wait();
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, 30);
}

TEST(Task, VoidTask) {
	auto task = OaMakeSharedPtr<OaTask<void>>();
	EXPECT_FALSE(task->IsDone());

	std::thread t([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		task->Complete();
	});

	task->Wait();
	t.join();
	EXPECT_TRUE(task->IsDone());
	EXPECT_FALSE(task->HasFailed());
}

// =============================================================================
// OaRwLock
// =============================================================================

TEST(RwLock, ConcurrentReads) {
	OaRwLock<OaI32> lock(42);
	std::atomic<OaI32> readCount{0};

	OaVec<std::thread> readers;
	for (OaI32 i = 0; i < 8; ++i) {
		readers.EmplaceBack([&] {
			auto guard = lock.Read();
			EXPECT_EQ(*guard, 42);
			readCount.fetch_add(1);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		});
	}

	for (auto& t : readers) t.join();
	EXPECT_EQ(readCount.load(), 8);
}

TEST(RwLock, ExclusiveWrite) {
	OaRwLock<OaI32> lock(0);

	OaVec<std::thread> writers;
	for (OaI32 i = 0; i < 4; ++i) {
		writers.EmplaceBack([&] {
			for (OaI32 j = 0; j < 100; ++j) {
				auto guard = lock.Write();
				++(*guard);
			}
		});
	}

	for (auto& t : writers) t.join();

	auto r = lock.Read();
	EXPECT_EQ(*r, 400);
}

// =============================================================================
// OaSpinlock
// =============================================================================

TEST(Spinlock, MutualExclusion) {
	OaSpinlock spin;
	OaI32 counter = 0;

	OaVec<std::thread> threads;
	for (OaI32 i = 0; i < 4; ++i) {
		threads.EmplaceBack([&] {
			for (OaI32 j = 0; j < 1000; ++j) {
				OaSpinlockGuard guard(spin);
				++counter;
			}
		});
	}

	for (auto& t : threads) t.join();
	EXPECT_EQ(counter, 4000);
}

TEST(Spinlock, TryLock) {
	OaSpinlock spin;
	EXPECT_TRUE(spin.TryLock());
	EXPECT_FALSE(spin.TryLock());
	spin.Unlock();
	EXPECT_TRUE(spin.TryLock());
	spin.Unlock();
}

// =============================================================================
// OaCpuTopology
// =============================================================================

TEST(CpuTopology, Detect) {
	auto topo = OaCpuTopology::Detect();
	EXPECT_GT(topo.NumLogicalCores, 0);
	EXPECT_GT(topo.NumPhysicalCores, 0);
	EXPECT_GE(topo.NumNumaNodes, 1);
	EXPECT_GE(topo.NumPackages, 1);
	EXPECT_EQ(static_cast<OaI32>(topo.Cores.Size()), topo.NumLogicalCores);

	for (auto& core : topo.Cores) {
		EXPECT_GE(core.Id, 0);
		EXPECT_NE(core.Type, OaCoreType::Unknown);
	}

	auto pcores = topo.GetPcoreIds();
	auto ecores = topo.GetEcoreIds();
	EXPECT_EQ(static_cast<OaI32>(pcores.Size() + ecores.Size()),
		topo.NumLogicalCores);

	topo.Print();
}

// =============================================================================
// OaThreadPool
// =============================================================================

TEST(ThreadPool, CreateShutdown) {
	auto pool = OaThreadPool::Create({.NumWorkers = 2, .PinToCores = false});
	EXPECT_EQ(pool.NumWorkers(), 2);
	EXPECT_TRUE(pool.IsRunning());
	pool.Shutdown();
	EXPECT_FALSE(pool.IsRunning());
}

TEST(ThreadPool, SubmitJobs) {
	auto pool = OaThreadPool::Create({.NumWorkers = 4, .PinToCores = false});
	std::atomic<OaI32> counter{0};

	for (OaI32 i = 0; i < 1000; ++i) {
		pool.Submit([&] { counter.fetch_add(1, std::memory_order_relaxed); });
	}

	pool.Shutdown();
	EXPECT_EQ(counter.load(), 1000);
}

TEST(ThreadPool, SubmitTask) {
	auto pool = OaThreadPool::Create({.NumWorkers = 2, .PinToCores = false});

	auto task = pool.SubmitTask([] { return 42; });
	auto result = task->Wait();
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, 42);

	pool.Shutdown();
}

TEST(ThreadPool, SubmitVoidTask) {
	auto pool = OaThreadPool::Create({.NumWorkers = 2, .PinToCores = false});
	std::atomic<bool> ran{false};

	auto task = pool.SubmitTask([&] { ran.store(true); });
	task->Wait();
	EXPECT_TRUE(ran.load());

	pool.Shutdown();
}

TEST(ThreadPool, WorkStealing) {
	auto pool = OaThreadPool::Create({.NumWorkers = 4, .PinToCores = false});
	std::atomic<OaI32> counter{0};

	for (OaI32 i = 0; i < 100; ++i) {
		pool.Submit([&] {
			std::this_thread::sleep_for(std::chrono::microseconds(100));
			counter.fetch_add(1, std::memory_order_relaxed);
		});
	}

	pool.Shutdown();
	EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPool, AutoDetectWorkers) {
	auto pool = OaThreadPool::Create({.PinToCores = false});
	EXPECT_GT(pool.NumWorkers(), 0);
	pool.Shutdown();
}

// Performance: submit 10k tasks, measure throughput
TEST(ThreadPool, ThroughputPerf) {
	auto pool = OaThreadPool::Create({.NumWorkers = 4, .PinToCores = false});
	constexpr OaI32 kJobs = 10000;
	std::atomic<OaI32> counter{0};

	auto start = std::chrono::steady_clock::now();
	for (OaI32 i = 0; i < kJobs; ++i) {
		pool.Submit([&] { counter.fetch_add(1, std::memory_order_relaxed); });
	}
	pool.Shutdown();
	auto end = std::chrono::steady_clock::now();

	EXPECT_EQ(counter.load(), kJobs);
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	fprintf(stderr, "  ThreadPool: %d jobs in %ldms (%.0f jobs/s)\n",
		kJobs, ms, ms > 0 ? kJobs * 1000.0 / ms : 0.0);
}
