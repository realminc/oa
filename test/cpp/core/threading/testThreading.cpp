#include <gtest/gtest.h>

#include <oa/core/thread.h>

#include <atomic>
#include <chrono>
#include <thread>

TEST(Thread, CreateMoveAndJoin) {
	oa::Atomic<bool> executed{false};
	auto created = oa::Thread::create([&] {
		executed.store(true, oa::MemoryOrder::Release);
	});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString().cStr();

	oa::Thread thread = oa::move(*created);
	EXPECT_TRUE(thread.joinable());
	const oa::Status status = thread.join();
	EXPECT_TRUE(status.isOk()) << status.toString().cStr();
	EXPECT_FALSE(thread.joinable());
	EXPECT_TRUE(executed.load(oa::MemoryOrder::Acquire));
}

TEST(Thread, RejectsEmptyEntryAndReportsHardware) {
	auto created = oa::Thread::create({});
	EXPECT_TRUE(created.isError());
	EXPECT_EQ(created.getStatus().getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_GE(oa::Thread::hardwareConcurrency(), 1U);
	oa::Thread::yield();
	oa::Thread::sleepFor(oa::Duration::fromNanoseconds(1));
}

// =============================================================================
// oa::Channel
// =============================================================================

TEST(Channel, SendRecv) {
	oa::Channel<oa::I32> ch(4);
	EXPECT_TRUE(ch.send(42));
	EXPECT_TRUE(ch.send(99));
	EXPECT_EQ(ch.size(), 2);

	auto v1 = ch.recv();
	auto v2 = ch.recv();
	ASSERT_TRUE(v1.hasValue());
	ASSERT_TRUE(v2.hasValue());
	EXPECT_EQ(*v1, 42);
	EXPECT_EQ(*v2, 99);
	EXPECT_EQ(ch.size(), 0);
}

TEST(Channel, TrySendRecv) {
	oa::Channel<oa::I32> ch(2);
	EXPECT_TRUE(ch.trySend(1));
	EXPECT_TRUE(ch.trySend(2));
	EXPECT_FALSE(ch.trySend(3));

	auto v = ch.tryRecv();
	ASSERT_TRUE(v.hasValue());
	EXPECT_EQ(*v, 1);

	EXPECT_TRUE(ch.trySend(3));
}

TEST(Channel, CloseUnblocksRecv) {
	oa::Channel<oa::I32> ch(4);
	std::atomic<bool> gotNullopt{false};

	std::thread t([&] {
		auto v = ch.recv();
		if (!v.hasValue()) gotNullopt.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	ch.close();
	t.join();
	EXPECT_TRUE(gotNullopt.load());
	EXPECT_TRUE(ch.isClosed());
}

TEST(Channel, CloseUnblocksSend) {
	oa::Channel<oa::I32> ch(1);
	ch.send(1);
	std::atomic<bool> sendFailed{false};

	std::thread t([&] {
		bool ok = ch.send(2);
		if (!ok) sendFailed.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	ch.close();
	t.join();
	EXPECT_TRUE(sendFailed.load());
}

TEST(Channel, Mpmc) {
	oa::Channel<oa::I32> ch(64);
	constexpr oa::I32 kProducers = 4;
	constexpr oa::I32 kPerProducer = 250;
	constexpr oa::I32 kTotal = kProducers * kPerProducer;

	std::atomic<oa::I32> sum{0};
	oa::Vector<std::thread> producers;
	oa::Vector<std::thread> consumers;

	for (oa::I32 p = 0; p < kProducers; ++p) {
		producers.emplaceBack([&, p] {
			for (oa::I32 i = 0; i < kPerProducer; ++i) {
				ch.send(p * kPerProducer + i + 1);
			}
		});
	}

	for (oa::I32 c = 0; c < 2; ++c) {
		consumers.emplaceBack([&] {
			while (true) {
				auto v = ch.recv();
				if (!v.hasValue()) break;
				sum.fetch_add(*v, std::memory_order_relaxed);
			}
		});
	}

	for (auto& t : producers) t.join();

	// wait for all items to be consumed
	while (ch.size() > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	ch.close();

	for (auto& t : consumers) t.join();

	oa::I32 expected = kTotal * (kTotal + 1) / 2;
	EXPECT_EQ(sum.load(), expected);
}

// =============================================================================
// oa::Task
// =============================================================================

TEST(Task, CompleteWait) {
	auto task = oa::makeShared<oa::Task<oa::I32>>();
	std::thread t([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		task->complete(42);
	});

	auto result = task->wait();
	t.join();
	ASSERT_TRUE(result.hasValue());
	EXPECT_EQ(*result, 42);
	EXPECT_TRUE(task->isDone());
	EXPECT_FALSE(task->hasFailed());
}

TEST(Task, fail) {
	auto task = oa::makeShared<oa::Task<oa::I32>>();
	task->fail(oa::Status::error("test error"));
	EXPECT_TRUE(task->isDone());
	EXPECT_TRUE(task->hasFailed());

	auto result = task->wait();
	EXPECT_FALSE(result.hasValue());
}

TEST(Task, tryGet) {
	auto task = oa::makeShared<oa::Task<oa::I32>>();
	EXPECT_FALSE(task->tryGet().hasValue());

	task->complete(7);
	auto v = task->tryGet();
	ASSERT_TRUE(v.hasValue());
	EXPECT_EQ(*v, 7);
}

TEST(Task, then) {
	auto task = oa::makeShared<oa::Task<oa::I32>>();
	auto doubled = task->then([](oa::I32 v) { return v * 2; });

	task->complete(21);
	auto result = doubled->wait();
	ASSERT_TRUE(result.hasValue());
	EXPECT_EQ(*result, 42);
}

TEST(Task, ThenAlreadyComplete) {
	auto task = oa::makeShared<oa::Task<oa::I32>>();
	task->complete(10);
	auto tripled = task->then([](oa::I32 v) { return v * 3; });

	auto result = tripled->wait();
	ASSERT_TRUE(result.hasValue());
	EXPECT_EQ(*result, 30);
}

TEST(Task, VoidTask) {
	auto task = oa::makeShared<oa::Task<void>>();
	EXPECT_FALSE(task->isDone());

	std::thread t([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		task->complete();
	});

	task->wait();
	t.join();
	EXPECT_TRUE(task->isDone());
	EXPECT_FALSE(task->hasFailed());
}

// =============================================================================
// oa::RwLock
// =============================================================================

TEST(RwLock, ConcurrentReads) {
	oa::RwLock<oa::I32> lock(42);
	std::atomic<oa::I32> readCount{0};

	oa::Vector<std::thread> readers;
	for (oa::I32 i = 0; i < 8; ++i) {
		readers.emplaceBack([&] {
			auto guard = lock.read();
			EXPECT_EQ(*guard, 42);
			readCount.fetch_add(1);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		});
	}

	for (auto& t : readers) t.join();
	EXPECT_EQ(readCount.load(), 8);
}

TEST(RwLock, ExclusiveWrite) {
	oa::RwLock<oa::I32> lock(0);

	oa::Vector<std::thread> writers;
	for (oa::I32 i = 0; i < 4; ++i) {
		writers.emplaceBack([&] {
			for (oa::I32 j = 0; j < 100; ++j) {
				auto guard = lock.write();
				++(*guard);
			}
		});
	}

	for (auto& t : writers) t.join();

	auto r = lock.read();
	EXPECT_EQ(*r, 400);
}

// =============================================================================
// oa::Spinlock
// =============================================================================

TEST(Spinlock, MutualExclusion) {
	oa::Spinlock spin;
	oa::I32 counter = 0;

	oa::Vector<std::thread> threads;
	for (oa::I32 i = 0; i < 4; ++i) {
		threads.emplaceBack([&] {
			for (oa::I32 j = 0; j < 1000; ++j) {
				oa::SpinlockGuard guard(spin);
				++counter;
			}
		});
	}

	for (auto& t : threads) t.join();
	EXPECT_EQ(counter, 4000);
}

TEST(Spinlock, tryLock) {
	oa::Spinlock spin;
	EXPECT_TRUE(spin.tryLock());
	EXPECT_FALSE(spin.tryLock());
	spin.unlock();
	EXPECT_TRUE(spin.tryLock());
	spin.unlock();
}

// =============================================================================
// oa::CpuTopology
// =============================================================================

TEST(CpuTopology, Detect) {
	auto topo = oa::CpuTopology::detect();
	EXPECT_GT(topo.numLogicalCores, 0);
	EXPECT_GT(topo.numPhysicalCores, 0);
	EXPECT_GE(topo.numNumaNodes, 1);
	EXPECT_GE(topo.numPackages, 1);
	EXPECT_EQ(static_cast<oa::I32>(topo.cores.size()), topo.numLogicalCores);

	for (auto& core : topo.cores) {
		EXPECT_GE(core.id, 0);
		EXPECT_NE(core.type, oa::CoreType::Unknown);
	}

	auto pcores = topo.getPcoreIds();
	auto ecores = topo.getEcoreIds();
	EXPECT_EQ(static_cast<oa::I32>(pcores.size() + ecores.size()),
		topo.numLogicalCores);

	topo.print();
}

// =============================================================================
// oa::ThreadPool
// =============================================================================

TEST(ThreadPool, CreateShutdown) {
	auto pool = oa::ThreadPool::create({.numWorkers = 2, .pinToCores = false});
	EXPECT_EQ(pool.numWorkers(), 2);
	EXPECT_TRUE(pool.isRunning());
	pool.shutdown();
	EXPECT_FALSE(pool.isRunning());
}

TEST(ThreadPool, AbandonDoesNotWaitForRunningJob) {
	struct JobGate {
		std::mutex mutex;
		std::condition_variable cv;
		bool started = false;
		bool release = false;
		bool finished = false;
		bool queuedRan = false;
	};
	auto gate = std::make_shared<JobGate>();
	oa::SharedPtr<oa::Task<oa::I32>> cancelledTask;
	auto destroyStart = std::chrono::steady_clock::now();
	{
		auto pool = oa::ThreadPool::create(
			{.numWorkers = 1, .pinToCores = false});
		pool.submit([gate] {
			std::unique_lock<std::mutex> lock(gate->mutex);
			gate->started = true;
			gate->cv.notify_all();
			(void)gate->cv.wait_for(
				lock,
				std::chrono::seconds(2),
				[gate] { return gate->release; });
			gate->finished = true;
			lock.unlock();
			gate->cv.notify_all();
		});
		pool.submit([gate] {
			std::lock_guard<std::mutex> lock(gate->mutex);
			gate->queuedRan = true;
		});
		cancelledTask = pool.submitTask([] { return 7; });

		std::unique_lock<std::mutex> lock(gate->mutex);
		ASSERT_TRUE(gate->cv.wait_for(
			lock,
			std::chrono::seconds(1),
			[gate] { return gate->started; }));
		destroyStart = std::chrono::steady_clock::now();
	}
	const auto destroyElapsed = std::chrono::steady_clock::now() - destroyStart;
	EXPECT_LT(destroyElapsed, std::chrono::milliseconds(500));
	ASSERT_TRUE(cancelledTask);
	EXPECT_TRUE(cancelledTask->isDone());
	EXPECT_TRUE(cancelledTask->hasFailed());
	EXPECT_EQ(cancelledTask->getError().getCode(), oa::StatusCode::Cancelled);

	std::unique_lock<std::mutex> lock(gate->mutex);
	gate->release = true;
	gate->cv.notify_all();
	EXPECT_TRUE(gate->cv.wait_for(
		lock,
		std::chrono::seconds(1),
		[gate] { return gate->finished; }));
	EXPECT_FALSE(gate->queuedRan);
}

TEST(ThreadPool, SubmitJobs) {
	auto pool = oa::ThreadPool::create({.numWorkers = 4, .pinToCores = false});
	std::atomic<oa::I32> counter{0};

	for (oa::I32 i = 0; i < 1000; ++i) {
		pool.submit([&] { counter.fetch_add(1, std::memory_order_relaxed); });
	}

	pool.shutdown();
	EXPECT_EQ(counter.load(), 1000);
}

TEST(ThreadPool, submitTask) {
	auto pool = oa::ThreadPool::create({.numWorkers = 2, .pinToCores = false});

	auto task = pool.submitTask([] { return 42; });
	auto result = task->wait();
	ASSERT_TRUE(result.hasValue());
	EXPECT_EQ(*result, 42);

	pool.shutdown();
}

TEST(ThreadPool, SubmitVoidTask) {
	auto pool = oa::ThreadPool::create({.numWorkers = 2, .pinToCores = false});
	std::atomic<bool> ran{false};

	auto task = pool.submitTask([&] { ran.store(true); });
	task->wait();
	EXPECT_TRUE(ran.load());

	pool.shutdown();
}

TEST(ThreadPool, WorkStealing) {
	auto pool = oa::ThreadPool::create({.numWorkers = 4, .pinToCores = false});
	std::atomic<oa::I32> counter{0};

	for (oa::I32 i = 0; i < 100; ++i) {
		pool.submit([&] {
			std::this_thread::sleep_for(std::chrono::microseconds(100));
			counter.fetch_add(1, std::memory_order_relaxed);
		});
	}

	pool.shutdown();
	EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPool, AutoDetectWorkers) {
	auto pool = oa::ThreadPool::create({.pinToCores = false});
	EXPECT_GT(pool.numWorkers(), 0);
	pool.shutdown();
}

// Performance: submit 10k tasks, measure throughput
TEST(ThreadPool, ThroughputPerf) {
	auto pool = oa::ThreadPool::create({.numWorkers = 4, .pinToCores = false});
	constexpr oa::I32 kJobs = 10000;
	std::atomic<oa::I32> counter{0};

	auto start = std::chrono::steady_clock::now();
	for (oa::I32 i = 0; i < kJobs; ++i) {
		pool.submit([&] { counter.fetch_add(1, std::memory_order_relaxed); });
	}
	pool.shutdown();
	auto end = std::chrono::steady_clock::now();

	EXPECT_EQ(counter.load(), kJobs);
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	fprintf(stderr, "  ThreadPool: %d jobs in %ldms (%.0f jobs/s)\n",
		kJobs, ms, ms > 0 ? kJobs * 1000.0 / ms : 0.0);
}
