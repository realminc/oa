#pragma once

// Atomic / Mutex / SharedMutex / ScopedLock / … —
// PascalCase synchronization surface.
//
// HONEST NOTE: atomics are compiler builtins and mutexes are kernel/futex
// shims. There is no meaningful clean-room implementation — reimplementing a
// CAS or a mutex "ourselves" would just re-expose the same __atomic_* / futex
// syscalls with more bugs. So these deliberately WRAP <atomic>/<mutex> and
// only provide the OA-consistent naming. native() exposes the underlying std
// object for the rare boundary (e.g. std::condition_variable). See OA standard library.md.

#include <atomic>
#include <mutex>
#include <shared_mutex>

namespace oa {

// ── Atomic<T> ──────────────────────────────────────────────────────────
template<typename T>
class Atomic {
public:
	using ValueType = T;

	Atomic() noexcept = default;
	constexpr Atomic(T inDesired) noexcept : value_(inDesired) {}
	Atomic(const Atomic&)            = delete;
	Atomic& operator=(const Atomic&) = delete;

	[[nodiscard]] T load(std::memory_order inOrder = std::memory_order_seq_cst) const noexcept {
		return value_.load(inOrder);
	}
	void store(T inDesired, std::memory_order inOrder = std::memory_order_seq_cst) noexcept {
		value_.store(inDesired, inOrder);
	}
	T exchange(T inDesired, std::memory_order inOrder = std::memory_order_seq_cst) noexcept {
		return value_.exchange(inDesired, inOrder);
	}
	bool compareExchangeStrong(T& inExpected, T inDesired,
	                           std::memory_order inOrder = std::memory_order_seq_cst) noexcept {
		return value_.compare_exchange_strong(inExpected, inDesired, inOrder);
	}
	bool compareExchangeWeak(T& inExpected, T inDesired,
	                         std::memory_order inOrder = std::memory_order_seq_cst) noexcept {
		return value_.compare_exchange_weak(inExpected, inDesired, inOrder);
	}

	// Integral/pointer only (instantiated on use, mirroring std::atomic).
	T fetchAdd(T inArg, std::memory_order inOrder = std::memory_order_seq_cst) noexcept { return value_.fetch_add(inArg, inOrder); }
	T fetchSub(T inArg, std::memory_order inOrder = std::memory_order_seq_cst) noexcept { return value_.fetch_sub(inArg, inOrder); }
	T fetchOr (T inArg, std::memory_order inOrder = std::memory_order_seq_cst) noexcept { return value_.fetch_or(inArg, inOrder); }
	T fetchAnd(T inArg, std::memory_order inOrder = std::memory_order_seq_cst) noexcept { return value_.fetch_and(inArg, inOrder); }
	T fetchXor(T inArg, std::memory_order inOrder = std::memory_order_seq_cst) noexcept { return value_.fetch_xor(inArg, inOrder); }

	// Ergonomic operators (match std::atomic).
	operator T() const noexcept { return load(); }
	T operator=(T inDesired) noexcept { store(inDesired); return inDesired; }
	T operator++()    noexcept { return ++value_; }
	T operator++(int) noexcept { return value_++; }
	T operator--()    noexcept { return --value_; }
	T operator--(int) noexcept { return value_--; }
	T operator+=(T inArg) noexcept { return value_ += inArg; }
	T operator-=(T inArg) noexcept { return value_ -= inArg; }

	[[nodiscard]] std::atomic<T>&       native()       noexcept { return value_; }
	[[nodiscard]] const std::atomic<T>& native() const noexcept { return value_; }

private:
	std::atomic<T> value_;
};

// ── Mutex ──────────────────────────────────────────────────────────────
class Mutex {
public:
	Mutex() = default;
	Mutex(const Mutex&)            = delete;
	Mutex& operator=(const Mutex&) = delete;

	void lock()            { mutex_.lock(); }
	[[nodiscard]] bool tryLock() { return mutex_.try_lock(); }
	void unlock()          { mutex_.unlock(); }

	[[nodiscard]] std::mutex& native() noexcept { return mutex_; }

private:
	std::mutex mutex_;
};

// ── SharedMutex (reader/writer) ────────────────────────────────────────
class SharedMutex {
public:
	SharedMutex() = default;
	SharedMutex(const SharedMutex&)            = delete;
	SharedMutex& operator=(const SharedMutex&) = delete;

	void lock()               { mutex_.lock(); }
	[[nodiscard]] bool tryLock()    { return mutex_.try_lock(); }
	void unlock()             { mutex_.unlock(); }

	void lockShared()         { mutex_.lock_shared(); }
	[[nodiscard]] bool tryLockShared() { return mutex_.try_lock_shared(); }
	void unlockShared()       { mutex_.unlock_shared(); }

	[[nodiscard]] std::shared_mutex& native() noexcept { return mutex_; }

private:
	std::shared_mutex mutex_;
};

// ── ScopedLock<Mutex> — RAII exclusive lock (non-movable) ───────────────
template<typename Mutex>
class ScopedLock {
public:
	explicit ScopedLock(Mutex& inMutex) : mutex_(inMutex) { mutex_.lock(); }
	~ScopedLock() { mutex_.unlock(); }
	ScopedLock(const ScopedLock&)            = delete;
	ScopedLock& operator=(const ScopedLock&) = delete;
private:
	Mutex& mutex_;
};

// ── SharedLock<Mutex> — RAII shared (reader) lock ───────────────────────
template<typename Mutex>
class SharedLock {
public:
	explicit SharedLock(Mutex& inMutex) : mutex_(inMutex) { mutex_.lockShared(); }
	~SharedLock() { mutex_.unlockShared(); }
	SharedLock(const SharedLock&)            = delete;
	SharedLock& operator=(const SharedLock&) = delete;
private:
	Mutex& mutex_;
};

// ── UniqueLock<Mutex> — movable, deferrable exclusive lock ──────────────
template<typename Mutex>
class UniqueLock {
public:
	UniqueLock() = default;
	explicit UniqueLock(Mutex& inMutex) : mutex_(&inMutex), owns_(true) { mutex_->lock(); }
	~UniqueLock() { if (owns_ && mutex_ != nullptr) { mutex_->unlock(); } }

	UniqueLock(UniqueLock&& inOther) noexcept
		: mutex_(inOther.mutex_), owns_(inOther.owns_) {
		inOther.mutex_ = nullptr;
		inOther.owns_  = false;
	}
	UniqueLock& operator=(UniqueLock&& inOther) noexcept {
		if (this != &inOther) {
			if (owns_ && mutex_ != nullptr) { mutex_->unlock(); }
			mutex_ = inOther.mutex_;
			owns_  = inOther.owns_;
			inOther.mutex_ = nullptr;
			inOther.owns_  = false;
		}
		return *this;
	}
	UniqueLock(const UniqueLock&)            = delete;
	UniqueLock& operator=(const UniqueLock&) = delete;

	void lock()   { if (mutex_ != nullptr && !owns_) { mutex_->lock();   owns_ = true;  } }
	void unlock() { if (mutex_ != nullptr && owns_)  { mutex_->unlock(); owns_ = false; } }
	[[nodiscard]] bool ownsLock() const noexcept { return owns_; }

private:
	Mutex* mutex_ = nullptr;
	bool   owns_  = false;
};

} // namespace oa
