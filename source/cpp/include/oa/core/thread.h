#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

// oa::CpuTopology — portable CPU core detection
//
// Detects P-cores / E-cores (Intel hybrid), CCX/CCD (AMD), big.lITTLE (ARM).
// Falls back to frequency-based heuristic when arch-specific info unavailable.
// Used by oa::ThreadPool for CPU affinity and NUMA-aware scheduling.

namespace oa {

// Native thread owner. Creation is explicit and fallible; destruction never
// waits or detaches implicitly. A live thread must be joined or detached by
// its owning session before this value is destroyed.
class Thread {
public:
	Thread() noexcept = default;
	~Thread() noexcept;
	Thread(Thread&& inOther) noexcept;
	Thread& operator=(Thread&& inOther) noexcept;
	Thread(const Thread&) = delete;
	Thread& operator=(const Thread&) = delete;

	[[nodiscard]] static oa::Result<Thread> create(oa::Fn<void()> inEntry);
	[[nodiscard]] static oa::U32 hardwareConcurrency() noexcept;
	static void yield() noexcept;
	static void sleepFor(oa::Duration inDuration) noexcept;

	[[nodiscard]] bool joinable() const noexcept { return joinable_; }
	[[nodiscard]] oa::Status join() noexcept;
	[[nodiscard]] oa::Status detach() noexcept;

private:
	static constexpr unsigned kStorageSize = 16;
	alignas(16) unsigned char storage_[kStorageSize]{};
	bool joinable_ = false;
};

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
	oa::Vector<CoreInfo> cores;
	oa::I32 numPhysicalCores = 0;
	oa::I32 numLogicalCores = 0;
	oa::I32 numNumaNodes = 1;
	oa::I32 numPackages = 1;

	[[nodiscard]] static CpuTopology detect();

	[[nodiscard]] oa::Vector<oa::I32> getPcoreIds() const;
	[[nodiscard]] oa::Vector<oa::I32> getEcoreIds() const;
	[[nodiscard]] oa::Vector<oa::I32> getCoresOnNuma(oa::I32 inNode) const;

	void print() const;
};

// oa::Spinlock — PAUSE spinlock + SFENCE unlock
//
// Replaces std::mutex for short critical sections.
// Uncontended: ~1-2ns (vs ~18-22ns for futex).
// Contended: PAUSE loop stays in userspace (no kernel call).

struct Spinlock {
	oa::Atomic<oa::U32> state{0};

	OA_FORCEINLINE void lock() {
		oa::U32 expected = 0;
		if (OA_LIKELY(state.compareExchangeWeak(
			expected, 1, oa::MemoryOrder::Acquire))) return;
		lockSlow();
	}

	OA_FORCEINLINE void unlock() {
		oa::atomicThreadFence(oa::MemoryOrder::Release);
		state.store(0, oa::MemoryOrder::Release);
	}

	[[nodiscard]] OA_FORCEINLINE bool tryLock() {
		oa::U32 expected = 0;
		return state.compareExchangeWeak(
			expected, 1, oa::MemoryOrder::Acquire);
	}

private:
	OA_NOINLINE void lockSlow() {
		for (;;) {
		#if defined(__i386__) || defined(__x86_64__)
			__asm__ __volatile__("pause" ::: "memory");
		#elif defined(__aarch64__) || defined(__arm__)
			__asm__ __volatile__("yield" ::: "memory");
		#else
			__atomic_signal_fence(__ATOMIC_SEQ_CST);
		#endif
			oa::U32 expected = 0;
			if (state.load(oa::MemoryOrder::Relaxed) == 0 &&
				state.compareExchangeWeak(
					expected, 1, oa::MemoryOrder::Acquire)) return;
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
// Uses oa::SharedMutex under the hood.
//
// usage:
//   oa::RwLock<oa::Vector<oa::Matrix>> cache;
//   { auto r = cache.read();  use(*r); }           // shared
//   { auto w = cache.write(); w->push_back(t); }   // exclusive

template <typename T>
struct RwLock {
	RwLock() = default;
	explicit RwLock(T inValue) : value_(oa::move(inValue)) {}

	struct ReadGuard {
		const T& operator*() const { return value_; }
		const T* operator->() const { return &value_; }
		~ReadGuard() { lock_.unlockShared(); }

	private:
		friend struct RwLock;
		ReadGuard(const T& inValue, oa::SharedMutex& inLock)
			: value_(inValue), lock_(inLock) {
			lock_.lockShared();
		}
		const T& value_;
		oa::SharedMutex& lock_;
	};

	struct WriteGuard {
		T& operator*() { return value_; }
		T* operator->() { return &value_; }
		~WriteGuard() { lock_.unlock(); }

	private:
		friend struct RwLock;
		WriteGuard(T& inValue, oa::SharedMutex& inLock)
			: value_(inValue), lock_(inLock) {
			lock_.lock();
		}
		T& value_;
		oa::SharedMutex& lock_;
	};

	[[nodiscard]] ReadGuard read() const { return ReadGuard(value_, mutex_); }
	[[nodiscard]] WriteGuard write() { return WriteGuard(value_, mutex_); }

private:
	T value_{};
	mutable oa::SharedMutex mutex_;
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
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		notFull_.wait(lk, [this] {
			return count_ < capacity_
				|| closed_.load(oa::MemoryOrder::Relaxed);
		});
		if (closed_.load(oa::MemoryOrder::Relaxed)) return false;
		buffer_[writePos_ % capacity_] = oa::move(inValue);
		++writePos_;
		++count_;
		lk.unlock();
		notEmpty_.notifyOne();
		return true;
	}

	bool trySend(T inValue) {
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		if (count_ >= capacity_
			|| closed_.load(oa::MemoryOrder::Relaxed)) return false;
		buffer_[writePos_ % capacity_] = oa::move(inValue);
		++writePos_;
		++count_;
		lk.unlock();
		notEmpty_.notifyOne();
		return true;
	}

	oa::Optional<T> recv() {
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		notEmpty_.wait(lk, [this] {
			return count_ > 0 || closed_.load(oa::MemoryOrder::Relaxed);
		});
		if (count_ == 0) return {};
		T val = oa::move(buffer_[readPos_ % capacity_]);
		++readPos_;
		--count_;
		lk.unlock();
		notFull_.notifyOne();
		return val;
	}

	oa::Optional<T> tryRecv() {
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		if (count_ == 0) return {};
		T val = oa::move(buffer_[readPos_ % capacity_]);
		++readPos_;
		--count_;
		lk.unlock();
		notFull_.notifyOne();
		return val;
	}

	void close() {
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			closed_.store(true, oa::MemoryOrder::Release);
		}
		notEmpty_.notifyAll();
		notFull_.notifyAll();
	}

	[[nodiscard]] bool isClosed() const {
		return closed_.load(oa::MemoryOrder::Acquire);
	}

	[[nodiscard]] oa::I32 size() const {
		oa::ScopedLock<oa::Mutex> lk(mutex_);
		return count_;
	}

private:
	oa::I32 capacity_;
	oa::Vector<T> buffer_;
	oa::I32 readPos_ = 0;
	oa::I32 writePos_ = 0;
	oa::I32 count_ = 0;
	oa::Atomic<bool> closed_{false};
	mutable oa::Mutex mutex_;
	oa::Condition notEmpty_;
	oa::Condition notFull_;
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
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			value_ = oa::move(inValue);
			done_.store(true, oa::MemoryOrder::Release);
		}
		cv_.notifyAll();
		runContinuation();
	}

	void fail(oa::Status inError) {
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			error_ = inError;
			failed_.store(true, oa::MemoryOrder::Release);
			done_.store(true, oa::MemoryOrder::Release);
		}
		cv_.notifyAll();
	}

	oa::Optional<T> wait() {
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		cv_.wait(lk, [this] { return done_.load(oa::MemoryOrder::Relaxed); });
		if (failed_.load(oa::MemoryOrder::Relaxed)) return {};
		return value_;
	}

	oa::Optional<T> tryGet() {
		oa::ScopedLock<oa::Mutex> lk(mutex_);
		if (!done_.load(oa::MemoryOrder::Relaxed)
			|| failed_.load(oa::MemoryOrder::Relaxed)) return {};
		return value_;
	}

	[[nodiscard]] bool isDone() const { return done_.load(oa::MemoryOrder::Acquire); }
	[[nodiscard]] bool hasFailed() const { return failed_.load(oa::MemoryOrder::Acquire); }

	[[nodiscard]] oa::Status getError() const {
		oa::ScopedLock<oa::Mutex> lk(mutex_);
		return error_;
	}

	template <typename F>
	auto then(F inFunc) -> oa::SharedPtr<Task<decltype(inFunc(oa::declval<T>()))>> {
		using R = decltype(inFunc(oa::declval<T>()));
		auto next = oa::makeShared<Task<R>>();
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			if (done_.load(oa::MemoryOrder::Relaxed)
				&& !failed_.load(oa::MemoryOrder::Relaxed)) {
				next->complete(inFunc(*value_));
				return next;
			}
			continuation_ = [next, f = oa::move(inFunc)](T val) {
				next->complete(f(oa::move(val)));
			};
		}
		return next;
	}

private:
	void runContinuation() {
		oa::Fn<void(T)> cont;
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			if (!continuation_) return;
			cont = oa::move(continuation_);
		}
		if (cont && value_) cont(oa::move(*value_));
	}

	mutable oa::Mutex mutex_;
	oa::Condition cv_;
	oa::Optional<T> value_;
	oa::Status error_;
	oa::Atomic<bool> done_{false};
	oa::Atomic<bool> failed_{false};
	oa::Fn<void(T)> continuation_;
};

template <>
struct Task<void> {
	void complete() {
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			done_.store(true, oa::MemoryOrder::Release);
		}
		cv_.notifyAll();
	}

	void fail(oa::Status inError) {
		{
			oa::ScopedLock<oa::Mutex> lk(mutex_);
			error_ = inError;
			failed_.store(true, oa::MemoryOrder::Release);
			done_.store(true, oa::MemoryOrder::Release);
		}
		cv_.notifyAll();
	}

	void wait() {
		oa::UniqueLock<oa::Mutex> lk(mutex_);
		cv_.wait(lk, [this] { return done_.load(oa::MemoryOrder::Relaxed); });
	}

	[[nodiscard]] bool isDone() const { return done_.load(oa::MemoryOrder::Acquire); }
	[[nodiscard]] bool hasFailed() const { return failed_.load(oa::MemoryOrder::Acquire); }

	[[nodiscard]] oa::Status getError() const {
		oa::ScopedLock<oa::Mutex> lk(mutex_);
		return error_;
	}

private:
	mutable oa::Mutex mutex_;
	oa::Condition cv_;
	oa::Status error_;
	oa::Atomic<bool> done_{false};
	oa::Atomic<bool> failed_{false};
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
	oa::Vector<oa::I32> coreIds;
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

	void submit(oa::Fn<void()> inJob);

	template <typename F>
	auto submitTask(F inFunc) -> oa::SharedPtr<Task<decltype(inFunc())>> {
		using R = decltype(inFunc());
		auto task = oa::makeShared<Task<R>>();
		submitJob_({
			.run = [task, f = oa::move(inFunc)]() mutable {
				if constexpr (oa::isVoidV<R>) {
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
		oa::Fn<void()> run;
		oa::Fn<void()> cancel;
	};
	static constexpr oa::I32 kQueueCapacity = 256;
	struct State;
	static void workerLoop(
		oa::SharedPtr<State> inState,
		oa::I32 inWorkerId,
		oa::I32 inCoreId,
		oa::Bool inPinToCore
	);
	void submitJob_(Job inJob);
	void abandon_() noexcept;

	oa::SharedPtr<State> state_;
	CpuTopology topology_;
};

} // namespace oa
