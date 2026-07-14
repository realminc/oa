#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

// OaCpuTopology — portable CPU core detection
//
// Detects P-cores / E-cores (Intel hybrid), CCX/CCD (AMD), big.LITTLE (ARM).
// Falls back to frequency-based heuristic when arch-specific info unavailable.
// Used by OaThreadPool for CPU affinity and NUMA-aware scheduling.

enum class OaCoreType : OaU8 {
	Performance = 0,
	Efficiency = 1,
	Unknown = 2
};

struct OaCoreInfo {
	OaI32 Id = -1;
	OaI32 PackageId = 0;
	OaI32 NumaNode = 0;
	OaI32 MaxFreqKhz = 0;
	OaCoreType Type = OaCoreType::Unknown;
};

struct OaCpuTopology {
	OaVec<OaCoreInfo> Cores;
	OaI32 NumPhysicalCores = 0;
	OaI32 NumLogicalCores = 0;
	OaI32 NumNumaNodes = 1;
	OaI32 NumPackages = 1;

	[[nodiscard]] static OaCpuTopology Detect();

	[[nodiscard]] OaVec<OaI32> GetPcoreIds() const;
	[[nodiscard]] OaVec<OaI32> GetEcoreIds() const;
	[[nodiscard]] OaVec<OaI32> GetCoresOnNuma(OaI32 InNode) const;

	void Print() const;
};

// OaSpinlock — PAUSE spinlock + SFENCE unlock
//
// Replaces std::mutex for short critical sections.
// Uncontended: ~1-2ns (vs ~18-22ns for futex).
// Contended: PAUSE loop stays in userspace (no kernel call).

struct OaSpinlock {
	std::atomic<OaU32> State{0};

	OA_FORCEINLINE void Lock() {
		OaU32 expected = 0;
		if (OA_LIKELY(
			State.compare_exchange_weak(expected, 1,
				std::memory_order_acquire,
				std::memory_order_relaxed))) return;
		LockSlow();
	}

	OA_FORCEINLINE void Unlock() {
		std::atomic_thread_fence(std::memory_order_release);
		State.store(0, std::memory_order_release);
	}

	[[nodiscard]] OA_FORCEINLINE bool TryLock() {
		OaU32 expected = 0;
		return State.compare_exchange_weak(expected, 1,
			std::memory_order_acquire,
			std::memory_order_relaxed);
	}

private:
	OA_NOINLINE void LockSlow() {
		for (;;) {
		#if defined(__i386__) || defined(__x86_64__)
			__asm__ __volatile__("pause" ::: "memory");
		#elif defined(__aarch64__) || defined(__arm__)
			__asm__ __volatile__("yield" ::: "memory");
		#else
			std::atomic_signal_fence(std::memory_order_seq_cst);
		#endif
			OaU32 expected = 0;
			if (State.load(std::memory_order_relaxed) == 0 &&
				State.compare_exchange_weak(expected, 1,
					std::memory_order_acquire,
					std::memory_order_relaxed)) return;
		}
	}
};

struct OaSpinlockGuard {
	OaSpinlock& Lock_;

	explicit OaSpinlockGuard(OaSpinlock& InLock) : Lock_(InLock) {
		Lock_.Lock();
	}
	~OaSpinlockGuard() {
		Lock_.Unlock();
	}

	OaSpinlockGuard(const OaSpinlockGuard&) = delete;
	OaSpinlockGuard& operator=(const OaSpinlockGuard&) = delete;
};

// OaRwLock<T> — reader-writer lock wrapping a value.
//
// Multiple concurrent readers OR one exclusive writer.
// Uses std::shared_mutex under the hood.
//
// Usage:
//   OaRwLock<OaVec<OaMatrix>> cache;
//   { auto r = cache.Read();  use(*r); }           // shared
//   { auto w = cache.Write(); w->push_back(t); }   // exclusive

template <typename T>
struct OaRwLock {
	OaRwLock() = default;
	explicit OaRwLock(T InValue) : Value_(std::move(InValue)) {}

	struct ReadGuard {
		const T& operator*() const { return Value_; }
		const T* operator->() const { return &Value_; }
		~ReadGuard() { Lock_.unlock_shared(); }

	private:
		friend struct OaRwLock;
		ReadGuard(const T& InValue, std::shared_mutex& InLock)
			: Value_(InValue), Lock_(InLock) {
			Lock_.lock_shared();
		}
		const T& Value_;
		std::shared_mutex& Lock_;
	};

	struct WriteGuard {
		T& operator*() { return Value_; }
		T* operator->() { return &Value_; }
		~WriteGuard() { Lock_.unlock(); }

	private:
		friend struct OaRwLock;
		WriteGuard(T& InValue, std::shared_mutex& InLock)
			: Value_(InValue), Lock_(InLock) {
			Lock_.lock();
		}
		T& Value_;
		std::shared_mutex& Lock_;
	};

	[[nodiscard]] ReadGuard Read() const { return ReadGuard(Value_, Mutex_); }
	[[nodiscard]] WriteGuard Write() { return WriteGuard(Value_, Mutex_); }

private:
	T Value_{};
	mutable std::shared_mutex Mutex_;
};

// OaChannel<T> — bounded MPMC channel (multiple-producer, multiple-consumer).
//
// Bounded ring buffer with blocking Send/Recv and non-blocking TrySend/TryRecv.
// Supports graceful shutdown via Close() which unblocks all waiters.
//
// Usage:
//   OaChannel<OaMatrix> ch(16);  // capacity 16
//   ch.Send(tensor);             // blocks if full
//   auto t = ch.Recv();          // blocks if empty, returns nullopt if closed
//   ch.Close();                  // unblocks all waiters

template <typename T>
struct OaChannel {
	explicit OaChannel(OaI32 InCapacity)
		: Capacity_(InCapacity), Buffer_(InCapacity) {}

	bool Send(T InValue) {
		std::unique_lock<std::mutex> lk(Mutex_);
		NotFull_.wait(lk, [this] { return Count_ < Capacity_ || Closed_.load(std::memory_order_relaxed); });
		if (Closed_.load(std::memory_order_relaxed)) return false;
		Buffer_[WritePos_ % Capacity_] = std::move(InValue);
		++WritePos_;
		++Count_;
		lk.unlock();
		NotEmpty_.notify_one();
		return true;
	}

	bool TrySend(T InValue) {
		std::unique_lock<std::mutex> lk(Mutex_);
		if (Count_ >= Capacity_ || Closed_.load(std::memory_order_relaxed)) return false;
		Buffer_[WritePos_ % Capacity_] = std::move(InValue);
		++WritePos_;
		++Count_;
		lk.unlock();
		NotEmpty_.notify_one();
		return true;
	}

	OaOpt<T> Recv() {
		std::unique_lock<std::mutex> lk(Mutex_);
		NotEmpty_.wait(lk, [this] { return Count_ > 0 || Closed_.load(std::memory_order_relaxed); });
		if (Count_ == 0) return {};
		T val = std::move(Buffer_[ReadPos_ % Capacity_]);
		++ReadPos_;
		--Count_;
		lk.unlock();
		NotFull_.notify_one();
		return val;
	}

	OaOpt<T> TryRecv() {
		std::unique_lock<std::mutex> lk(Mutex_);
		if (Count_ == 0) return {};
		T val = std::move(Buffer_[ReadPos_ % Capacity_]);
		++ReadPos_;
		--Count_;
		lk.unlock();
		NotFull_.notify_one();
		return val;
	}

	void Close() {
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			Closed_.store(true, std::memory_order_release);
		}
		NotEmpty_.notify_all();
		NotFull_.notify_all();
	}

	[[nodiscard]] bool IsClosed() const {
		return Closed_.load(std::memory_order_acquire);
	}

	[[nodiscard]] OaI32 Size() const {
		std::lock_guard<std::mutex> lk(Mutex_);
		return Count_;
	}

private:
	OaI32 Capacity_;
	OaVec<T> Buffer_;
	OaI32 ReadPos_ = 0;
	OaI32 WritePos_ = 0;
	OaI32 Count_ = 0;
	std::atomic<bool> Closed_{false};
	mutable std::mutex Mutex_;
	std::condition_variable NotEmpty_;
	std::condition_variable NotFull_;
};

// OaTask<T> — lightweight async result (future/promise).
//
// Producer calls Complete(value) or Fail(status) exactly once.
// Consumer calls Wait() or TryGet() to retrieve the result.
// Supports Then() for continuation chaining.
//
// Usage:
//   auto task = OaMakeSharedPtr<OaTask<OaMatrix>>();
//   pool.Submit([task] { task->Complete(ComputeSomething()); });
//   auto result = task->Wait();  // blocks until complete

template <typename T>
struct OaTask {
	void Complete(T InValue) {
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			Value_ = std::move(InValue);
			Done_.store(true, std::memory_order_release);
		}
		Cv_.notify_all();
		RunContinuation();
	}

	void Fail(OaStatus InError) {
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			Error_ = InError;
			Failed_.store(true, std::memory_order_release);
			Done_.store(true, std::memory_order_release);
		}
		Cv_.notify_all();
	}

	OaOpt<T> Wait() {
		std::unique_lock<std::mutex> lk(Mutex_);
		Cv_.wait(lk, [this] { return Done_.load(std::memory_order_relaxed); });
		if (Failed_.load(std::memory_order_relaxed)) return {};
		return Value_;
	}

	OaOpt<T> TryGet() {
		std::lock_guard<std::mutex> lk(Mutex_);
		if (!Done_.load(std::memory_order_relaxed) || Failed_.load(std::memory_order_relaxed)) return {};
		return Value_;
	}

	[[nodiscard]] bool IsDone() const { return Done_.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasFailed() const { return Failed_.load(std::memory_order_acquire); }

	[[nodiscard]] OaStatus GetError() const {
		std::lock_guard<std::mutex> lk(Mutex_);
		return Error_;
	}

	template <typename F>
	auto Then(F InFunc) -> OaSharedPtr<OaTask<decltype(InFunc(std::declval<T>()))>> {
		using R = decltype(InFunc(std::declval<T>()));
		auto next = OaMakeSharedPtr<OaTask<R>>();
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			if (Done_.load(std::memory_order_relaxed) && !Failed_.load(std::memory_order_relaxed)) {
				next->Complete(InFunc(*Value_));
				return next;
			}
			Continuation_ = [next, f = std::move(InFunc)](T val) {
				next->Complete(f(std::move(val)));
			};
		}
		return next;
	}

private:
	void RunContinuation() {
		std::function<void(T)> cont;
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			if (!Continuation_) return;
			cont = std::move(Continuation_);
		}
		if (cont && Value_) cont(std::move(*Value_));
	}

	mutable std::mutex Mutex_;
	std::condition_variable Cv_;
	OaOpt<T> Value_;
	OaStatus Error_;
	std::atomic<bool> Done_{false};
	std::atomic<bool> Failed_{false};
	std::function<void(T)> Continuation_;
};

template <>
struct OaTask<void> {
	void Complete() {
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			Done_.store(true, std::memory_order_release);
		}
		Cv_.notify_all();
	}

	void Fail(OaStatus InError) {
		{
			std::lock_guard<std::mutex> lk(Mutex_);
			Error_ = InError;
			Failed_.store(true, std::memory_order_release);
			Done_.store(true, std::memory_order_release);
		}
		Cv_.notify_all();
	}

	void Wait() {
		std::unique_lock<std::mutex> lk(Mutex_);
		Cv_.wait(lk, [this] { return Done_.load(std::memory_order_relaxed); });
	}

	[[nodiscard]] bool IsDone() const { return Done_.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasFailed() const { return Failed_.load(std::memory_order_acquire); }

	[[nodiscard]] OaStatus GetError() const {
		std::lock_guard<std::mutex> lk(Mutex_);
		return Error_;
	}

private:
	mutable std::mutex Mutex_;
	std::condition_variable Cv_;
	OaStatus Error_;
	std::atomic<bool> Done_{false};
	std::atomic<bool> Failed_{false};
};

// OaThreadPool — work-stealing thread pool with CPU affinity.
//
// Workers each own a bounded channel. Submit() round-robins jobs.
// When a worker's own queue is empty, it steals from siblings.
// Optional CPU pinning via OaCpuTopology (P-cores preferred).
//
// Usage:
//   auto pool = OaThreadPool::Create();        // auto-detect cores
//   pool.Submit([] { DoWork(); });              // fire-and-forget
//   auto t = pool.SubmitTask([] { return 42; }); // get future
//   auto val = t->Wait();                       // blocks, returns 42
//   pool.Shutdown();                            // joins all workers

struct OaThreadPoolConfig {
	OaI32 NumWorkers = 0;
	OaBool PinToCores = true;
	OaBool UseTopology = true;
	OaVec<OaI32> CoreIds;
};

struct OaThreadPool {
	[[nodiscard]] static OaThreadPool Create(
		const OaThreadPoolConfig& InConfig = {});
	void Shutdown();
	~OaThreadPool();

	OaThreadPool(OaThreadPool&& InOther) noexcept;
	OaThreadPool& operator=(OaThreadPool&& InOther) noexcept;
	OaThreadPool(const OaThreadPool&) = delete;
	OaThreadPool& operator=(const OaThreadPool&) = delete;

	void Submit(std::function<void()> InJob);

	template <typename F>
	auto SubmitTask(F InFunc) -> OaSharedPtr<OaTask<decltype(InFunc())>> {
		using R = decltype(InFunc());
		auto task = OaMakeSharedPtr<OaTask<R>>();
		Submit([task, f = std::move(InFunc)]() mutable {
			if constexpr (std::is_void_v<R>) {
				f();
				task->Complete();
			} else {
				task->Complete(f());
			}
		});
		return task;
	}

	[[nodiscard]] OaI32 NumWorkers() const;
	[[nodiscard]] bool IsRunning() const;
	[[nodiscard]] const OaCpuTopology& GetTopology() const;

private:
	OaThreadPool() = default;

	using Job = std::function<void()>;
	static constexpr OaI32 kQueueCapacity = 256;

	OaVec<std::thread> Workers_;
	OaVec<OaSharedPtr<OaChannel<Job>>> Queues_;
	std::atomic<OaI32> NextWorker_{0};
	std::atomic<bool> Running_{false};
	OaCpuTopology Topology_;
	OaVec<OaI32> WorkerCoreIds_;
};
