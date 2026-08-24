#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>

#include <oa/core/types.h>
#include <oa/core/status.h>

// oa::CpuTopology — portable CPU core detection
//
// Detects P-cores / E-cores (Intel hybrid), CCX/CCD (AMD), big.lITTLE (ARM).
// Falls back to frequency-based heuristic when arch-specific info unavailable.
// Used by oa::ThreadPool for CPU affinity and NUMA-aware scheduling.

namespace oa {

enum class CoreType : oa::U8 {
	Performance = 0,
	Efficiency = 1,
	Unknown = 2
};

struct CoreInfo {
	oa::I32 id = -1;
	oa::I32 packageId = 0;
	oa::I32 numaNode = 0;
	oa::I32 maxFreqKhz = 0;
	CoreType type = CoreType::Unknown;
};

struct CpuTopology {
	oa::Vec<CoreInfo> cores;
	oa::I32 numPhysicalCores = 0;
	oa::I32 numLogicalCores = 0;
	oa::I32 numNumaNodes = 1;
	oa::I32 numPackages = 1;

	[[nodiscard]] static CpuTopology detect();

	[[nodiscard]] oa::Vec<oa::I32> getPcoreIds() const;
	[[nodiscard]] oa::Vec<oa::I32> getEcoreIds() const;
	[[nodiscard]] oa::Vec<oa::I32> getCoresOnNuma(oa::I32 inNode) const;

	void print() const;
};

// oa::Spinlock — PAUSE spinlock + SFENCE unlock
//
// Replaces std::mutex for short critical sections.
// Uncontended: ~1-2ns (vs ~18-22ns for futex).
// Contended: PAUSE loop stays in userspace (no kernel call).

struct Spinlock {
	std::atomic<oa::U32> state{0};

	OA_FORCEINLINE void lock() {
		oa::U32 expected = 0;
		if (OA_LIKELY(
			state.compare_exchange_weak(expected, 1,
				std::memory_order_acquire,
				std::memory_order_relaxed))) return;
		lockSlow();
	}

	OA_FORCEINLINE void unlock() {
		std::atomic_thread_fence(std::memory_order_release);
		state.store(0, std::memory_order_release);
	}

	[[nodiscard]] OA_FORCEINLINE bool tryLock() {
		oa::U32 expected = 0;
		return state.compare_exchange_weak(expected, 1,
			std::memory_order_acquire,
			std::memory_order_relaxed);
	}

private:
	OA_NOINLINE void lockSlow() {
		for (;;) {
		#if defined(__i386__) || defined(__x86_64__)
			__asm__ __volatile__("pause" ::: "memory");
		#elif defined(__aarch64__) || defined(__arm__)
			__asm__ __volatile__("yield" ::: "memory");
		#else
			std::atomic_signal_fence(std::memory_order_seq_cst);
		#endif
			oa::U32 expected = 0;
			if (state.load(std::memory_order_relaxed) == 0 &&
				state.compare_exchange_weak(expected, 1,
					std::memory_order_acquire,
					std::memory_order_relaxed)) return;
		}
	}
};

struct SpinlockGuard {
	Spinlock& lock_;

	explicit SpinlockGuard(Spinlock& inLock) : lock_(inLock) {
		lock_.lock();
	}
	~SpinlockGuard() {
		lock_.unlock();
	}

	SpinlockGuard(const SpinlockGuard&) = delete;
	SpinlockGuard& operator=(const SpinlockGuard&) = delete;
};

// oa::RwLock<T> — reader-writer lock wrapping a value.
//
// Multiple concurrent readers OR one exclusive writer.
// Uses std::shared_mutex under the hood.
//
// usage:
//   oa::RwLock<oa::Vec<oa::Matrix>> cache;
//   { auto r = cache.read();  use(*r); }           // shared
//   { auto w = cache.write(); w->push_back(t); }   // exclusive

template <typename T>
struct RwLock {
	RwLock() = default;
	explicit RwLock(T inValue) : value_(std::move(inValue)) {}

	struct ReadGuard {
		const T& operator*() const { return value_; }
		const T* operator->() const { return &value_; }
		~ReadGuard() { lock_.unlock_shared(); }

	private:
		friend struct RwLock;
		ReadGuard(const T& inValue, std::shared_mutex& inLock)
			: value_(inValue), lock_(inLock) {
			lock_.lock_shared();
		}
		const T& value_;
		std::shared_mutex& lock_;
	};

	struct WriteGuard {
		T& operator*() { return value_; }
		T* operator->() { return &value_; }
		~WriteGuard() { lock_.unlock(); }

	private:
		friend struct RwLock;
		WriteGuard(T& inValue, std::shared_mutex& inLock)
			: value_(inValue), lock_(inLock) {
			lock_.lock();
		}
		T& value_;
		std::shared_mutex& lock_;
	};

	[[nodiscard]] ReadGuard read() const { return ReadGuard(value_, mutex_); }
	[[nodiscard]] WriteGuard write() { return WriteGuard(value_, mutex_); }

private:
	T value_{};
	mutable std::shared_mutex mutex_;
};

// oa::Channel<T> — bounded MPMC channel (multiple-producer, multiple-consumer).
//
// Bounded ring buffer with blocking send/recv and non-blocking trySend/tryRecv.
// Supports graceful shutdown via close() which unblocks all waiters.
//
// usage:
//   oa::Channel<oa::Matrix> ch(16);  // capacity 16
//   ch.send(tensor);                 // blocks if full
//   auto t = ch.recv();              // blocks if empty, returns nullopt if closed
//   ch.close();                      // unblocks all waiters

template <typename T>
struct Channel {
	explicit Channel(oa::I32 inCapacity)
		: capacity_(inCapacity), buffer_(inCapacity) {}

	bool send(T inValue) {
		std::unique_lock<std::mutex> lk(mutex_);
		notFull_.wait(lk, [this] { return count_ < capacity_ || closed_.load(std::memory_order_relaxed); });
		if (closed_.load(std::memory_order_relaxed)) return false;
		buffer_[writePos_ % capacity_] = std::move(inValue);
		++writePos_;
		++count_;
		lk.unlock();
		notEmpty_.notify_one();
		return true;
	}

	bool trySend(T inValue) {
		std::unique_lock<std::mutex> lk(mutex_);
		if (count_ >= capacity_ || closed_.load(std::memory_order_relaxed)) return false;
		buffer_[writePos_ % capacity_] = std::move(inValue);
		++writePos_;
		++count_;
		lk.unlock();
		notEmpty_.notify_one();
		return true;
	}

	oa::Optional<T> recv() {
		std::unique_lock<std::mutex> lk(mutex_);
		notEmpty_.wait(lk, [this] { return count_ > 0 || closed_.load(std::memory_order_relaxed); });
		if (count_ == 0) return {};
		T val = std::move(buffer_[readPos_ % capacity_]);
		++readPos_;
		--count_;
		lk.unlock();
		notFull_.notify_one();
		return val;
	}

	oa::Optional<T> tryRecv() {
		std::unique_lock<std::mutex> lk(mutex_);
		if (count_ == 0) return {};
		T val = std::move(buffer_[readPos_ % capacity_]);
		++readPos_;
		--count_;
		lk.unlock();
		notFull_.notify_one();
		return val;
	}

	void close() {
		{
			std::lock_guard<std::mutex> lk(mutex_);
			closed_.store(true, std::memory_order_release);
		}
		notEmpty_.notify_all();
		notFull_.notify_all();
	}

	[[nodiscard]] bool isClosed() const {
		return closed_.load(std::memory_order_acquire);
	}

	[[nodiscard]] oa::I32 size() const {
		std::lock_guard<std::mutex> lk(mutex_);
		return count_;
	}

private:
	oa::I32 capacity_;
	oa::Vec<T> buffer_;
	oa::I32 readPos_ = 0;
	oa::I32 writePos_ = 0;
	oa::I32 count_ = 0;
	std::atomic<bool> closed_{false};
	mutable std::mutex mutex_;
	std::condition_variable notEmpty_;
	std::condition_variable notFull_;
};

// oa::Task<T> — lightweight async result (future/promise).
//
// producer calls complete(value) or fail(status) exactly once.
// consumer calls wait() or tryGet() to retrieve the result.
// Supports then() for continuation chaining.
//
// usage:
//   auto task = oa::makeShared<oa::Task<oa::Matrix>>();
//   pool.submitTask([task] { task->complete(computeSomething()); });
//   auto result = task->wait();  // blocks until complete

template <typename T>
struct Task {
	void complete(T inValue) {
		{
			std::lock_guard<std::mutex> lk(mutex_);
			value_ = std::move(inValue);
			done_.store(true, std::memory_order_release);
		}
		cv_.notify_all();
		runContinuation();
	}

	void fail(oa::Status inError) {
		{
			std::lock_guard<std::mutex> lk(mutex_);
			error_ = inError;
			failed_.store(true, std::memory_order_release);
			done_.store(true, std::memory_order_release);
		}
		cv_.notify_all();
	}

	oa::Optional<T> wait() {
		std::unique_lock<std::mutex> lk(mutex_);
		cv_.wait(lk, [this] { return done_.load(std::memory_order_relaxed); });
		if (failed_.load(std::memory_order_relaxed)) return {};
		return value_;
	}

	oa::Optional<T> tryGet() {
		std::lock_guard<std::mutex> lk(mutex_);
		if (!done_.load(std::memory_order_relaxed) || failed_.load(std::memory_order_relaxed)) return {};
		return value_;
	}

	[[nodiscard]] bool isDone() const { return done_.load(std::memory_order_acquire); }
	[[nodiscard]] bool hasFailed() const { return failed_.load(std::memory_order_acquire); }

	[[nodiscard]] oa::Status getError() const {
		std::lock_guard<std::mutex> lk(mutex_);
		return error_;
	}

	template <typename F>
	auto then(F inFunc) -> oa::SharedPtr<Task<decltype(inFunc(std::declval<T>()))>> {
		using R = decltype(inFunc(std::declval<T>()));
		auto next = oa::makeShared<Task<R>>();
		{
			std::lock_guard<std::mutex> lk(mutex_);
			if (done_.load(std::memory_order_relaxed) && !failed_.load(std::memory_order_relaxed)) {
				next->complete(inFunc(*value_));
				return next;
			}
			continuation_ = [next, f = std::move(inFunc)](T val) {
				next->complete(f(std::move(val)));
			};
		}
		return next;
	}

private:
	void runContinuation() {
		std::function<void(T)> cont;
		{
			std::lock_guard<std::mutex> lk(mutex_);
			if (!continuation_) return;
			cont = std::move(continuation_);
		}
		if (cont && value_) cont(std::move(*value_));
	}

	mutable std::mutex mutex_;
	std::condition_variable cv_;
	oa::Optional<T> value_;
	oa::Status error_;
	std::atomic<bool> done_{false};
	std::atomic<bool> failed_{false};
	std::function<void(T)> continuation_;
};

template <>
struct Task<void> {
	void complete() {
		{
			std::lock_guard<std::mutex> lk(mutex_);
			done_.store(true, std::memory_order_release);
		}
		cv_.notify_all();
	}

	void fail(oa::Status inError) {
		{
			std::lock_guard<std::mutex> lk(mutex_);
			error_ = inError;
			failed_.store(true, std::memory_order_release);
			done_.store(true, std::memory_order_release);
		}
		cv_.notify_all();
	}

	void wait() {
		std::unique_lock<std::mutex> lk(mutex_);
		cv_.wait(lk, [this] { return done_.load(std::memory_order_relaxed); });
	}

	[[nodiscard]] bool isDone() const { return done_.load(std::memory_order_acquire); }
	[[nodiscard]] bool hasFailed() const { return failed_.load(std::memory_order_acquire); }

	[[nodiscard]] oa::Status getError() const {
		std::lock_guard<std::mutex> lk(mutex_);
		return error_;
	}

private:
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	oa::Status error_;
	std::atomic<bool> done_{false};
	std::atomic<bool> failed_{false};
};

// oa::ThreadPool — work-stealing thread pool with CPU affinity.
//
// Workers each own a bounded channel. submit() round-robins jobs.
// When a worker's own queue is empty, it steals from siblings.
// Optional CPU pinning via oa::CpuTopology (P-cores preferred).
//
// usage:
//   auto pool = oa::ThreadPool::create();        // auto-detect cores
//   pool.submit([] { doWork(); });               // fire-and-forget
//   auto t = pool.submitTask([] { return 42; }); // get future
//   auto val = t->wait();                        // blocks, returns 42
//   pool.shutdown();                             // drains and waits explicitly

struct ThreadPoolConfig {
	oa::I32 numWorkers = 0;
	oa::Bool pinToCores = true;
	oa::Bool useTopology = true;
	oa::Vec<oa::I32> coreIds;
};

struct ThreadPool {
	[[nodiscard]] static ThreadPool create(
		const ThreadPoolConfig& inConfig = {});
	// Explicit graceful boundary. Stops intake, drains queued jobs, and waits
	// for every worker. Destruction only requests cancellation and never joins.
	void shutdown();
	~ThreadPool();

	ThreadPool(ThreadPool&& inOther) noexcept;
	ThreadPool& operator=(ThreadPool&& inOther) noexcept;
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	void submit(std::function<void()> inJob);

	template <typename F>
	auto submitTask(F inFunc) -> oa::SharedPtr<Task<decltype(inFunc())>> {
		using R = decltype(inFunc());
		auto task = oa::makeShared<Task<R>>();
		submitJob_({
			.run = [task, f = std::move(inFunc)]() mutable {
				if constexpr (std::is_void_v<R>) {
					f();
					task->complete();
				} else {
					task->complete(f());
				}
			},
			.cancel = [task] {
				task->fail(oa::Status::cancelled("Thread pool abandoned"));
			},
		});
		return task;
	}

	[[nodiscard]] oa::I32 numWorkers() const;
	[[nodiscard]] bool isRunning() const;
	[[nodiscard]] const CpuTopology& getTopology() const;

private:
	ThreadPool() = default;

	struct Job {
		std::function<void()> run;
		std::function<void()> cancel;
	};
	static constexpr oa::I32 kQueueCapacity = 256;
	struct State;
	static void workerLoop(
		oa::SharedPtr<State> inState,
		oa::I32 inWorkerId,
		oa::I32 inCoreId,
		oa::Bool inPinToCore);
	void submitJob_(Job inJob);
	void abandon_() noexcept;

	oa::SharedPtr<State> state_;
	CpuTopology topology_;
};

} // namespace oa
