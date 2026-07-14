#pragma once

// OaStdAtomic / OaStdMutex / OaStdSharedMutex / OaStdScopedLock / … —
// PascalCase synchronization surface.
//
// HONEST NOTE: atomics are compiler builtins and mutexes are kernel/futex
// shims. There is no meaningful clean-room implementation — reimplementing a
// CAS or a mutex "ourselves" would just re-expose the same __atomic_* / futex
// syscalls with more bugs. So these deliberately WRAP <atomic>/<mutex> and
// only provide the OA-consistent naming. Native() exposes the underlying std
// object for the rare boundary (e.g. std::condition_variable). See OaStd.md.

#include <atomic>
#include <mutex>
#include <shared_mutex>

// ── OaStdAtomic<T> ──────────────────────────────────────────────────────────
template<typename T>
class OaStdAtomic {
public:
	using ValueType = T;

	OaStdAtomic() noexcept = default;
	constexpr OaStdAtomic(T InDesired) noexcept : Value_(InDesired) {}
	OaStdAtomic(const OaStdAtomic&)            = delete;
	OaStdAtomic& operator=(const OaStdAtomic&) = delete;

	[[nodiscard]] T Load(std::memory_order InOrder = std::memory_order_seq_cst) const noexcept {
		return Value_.load(InOrder);
	}
	void Store(T InDesired, std::memory_order InOrder = std::memory_order_seq_cst) noexcept {
		Value_.store(InDesired, InOrder);
	}
	T Exchange(T InDesired, std::memory_order InOrder = std::memory_order_seq_cst) noexcept {
		return Value_.exchange(InDesired, InOrder);
	}
	bool CompareExchangeStrong(T& InExpected, T InDesired,
	                           std::memory_order InOrder = std::memory_order_seq_cst) noexcept {
		return Value_.compare_exchange_strong(InExpected, InDesired, InOrder);
	}
	bool CompareExchangeWeak(T& InExpected, T InDesired,
	                         std::memory_order InOrder = std::memory_order_seq_cst) noexcept {
		return Value_.compare_exchange_weak(InExpected, InDesired, InOrder);
	}

	// Integral/pointer only (instantiated on use, mirroring std::atomic).
	T FetchAdd(T InArg, std::memory_order InOrder = std::memory_order_seq_cst) noexcept { return Value_.fetch_add(InArg, InOrder); }
	T FetchSub(T InArg, std::memory_order InOrder = std::memory_order_seq_cst) noexcept { return Value_.fetch_sub(InArg, InOrder); }
	T FetchOr (T InArg, std::memory_order InOrder = std::memory_order_seq_cst) noexcept { return Value_.fetch_or(InArg, InOrder); }
	T FetchAnd(T InArg, std::memory_order InOrder = std::memory_order_seq_cst) noexcept { return Value_.fetch_and(InArg, InOrder); }
	T FetchXor(T InArg, std::memory_order InOrder = std::memory_order_seq_cst) noexcept { return Value_.fetch_xor(InArg, InOrder); }

	// Ergonomic operators (match std::atomic).
	operator T() const noexcept { return Load(); }
	T operator=(T InDesired) noexcept { Store(InDesired); return InDesired; }
	T operator++()    noexcept { return ++Value_; }
	T operator++(int) noexcept { return Value_++; }
	T operator--()    noexcept { return --Value_; }
	T operator--(int) noexcept { return Value_--; }
	T operator+=(T InArg) noexcept { return Value_ += InArg; }
	T operator-=(T InArg) noexcept { return Value_ -= InArg; }

	[[nodiscard]] std::atomic<T>&       Native()       noexcept { return Value_; }
	[[nodiscard]] const std::atomic<T>& Native() const noexcept { return Value_; }

private:
	std::atomic<T> Value_;
};

// ── OaStdMutex ──────────────────────────────────────────────────────────────
class OaStdMutex {
public:
	OaStdMutex() = default;
	OaStdMutex(const OaStdMutex&)            = delete;
	OaStdMutex& operator=(const OaStdMutex&) = delete;

	void Lock()            { M_.lock(); }
	[[nodiscard]] bool TryLock() { return M_.try_lock(); }
	void Unlock()          { M_.unlock(); }

	[[nodiscard]] std::mutex& Native() noexcept { return M_; }

private:
	std::mutex M_;
};

// ── OaStdSharedMutex (reader/writer) ────────────────────────────────────────
class OaStdSharedMutex {
public:
	OaStdSharedMutex() = default;
	OaStdSharedMutex(const OaStdSharedMutex&)            = delete;
	OaStdSharedMutex& operator=(const OaStdSharedMutex&) = delete;

	void Lock()               { M_.lock(); }
	[[nodiscard]] bool TryLock()    { return M_.try_lock(); }
	void Unlock()             { M_.unlock(); }

	void LockShared()         { M_.lock_shared(); }
	[[nodiscard]] bool TryLockShared() { return M_.try_lock_shared(); }
	void UnlockShared()       { M_.unlock_shared(); }

	[[nodiscard]] std::shared_mutex& Native() noexcept { return M_; }

private:
	std::shared_mutex M_;
};

// ── OaStdScopedLock<Mutex> — RAII exclusive lock (non-movable) ───────────────
template<typename Mutex>
class OaStdScopedLock {
public:
	explicit OaStdScopedLock(Mutex& InMutex) : Mutex_(InMutex) { Mutex_.Lock(); }
	~OaStdScopedLock() { Mutex_.Unlock(); }
	OaStdScopedLock(const OaStdScopedLock&)            = delete;
	OaStdScopedLock& operator=(const OaStdScopedLock&) = delete;
private:
	Mutex& Mutex_;
};

// ── OaStdSharedLock<Mutex> — RAII shared (reader) lock ───────────────────────
template<typename Mutex>
class OaStdSharedLock {
public:
	explicit OaStdSharedLock(Mutex& InMutex) : Mutex_(InMutex) { Mutex_.LockShared(); }
	~OaStdSharedLock() { Mutex_.UnlockShared(); }
	OaStdSharedLock(const OaStdSharedLock&)            = delete;
	OaStdSharedLock& operator=(const OaStdSharedLock&) = delete;
private:
	Mutex& Mutex_;
};

// ── OaStdUniqueLock<Mutex> — movable, deferrable exclusive lock ──────────────
template<typename Mutex>
class OaStdUniqueLock {
public:
	OaStdUniqueLock() = default;
	explicit OaStdUniqueLock(Mutex& InMutex) : Mutex_(&InMutex), Owns_(true) { Mutex_->Lock(); }
	~OaStdUniqueLock() { if (Owns_ && Mutex_ != nullptr) { Mutex_->Unlock(); } }

	OaStdUniqueLock(OaStdUniqueLock&& InOther) noexcept
		: Mutex_(InOther.Mutex_), Owns_(InOther.Owns_) {
		InOther.Mutex_ = nullptr;
		InOther.Owns_  = false;
	}
	OaStdUniqueLock& operator=(OaStdUniqueLock&& InOther) noexcept {
		if (this != &InOther) {
			if (Owns_ && Mutex_ != nullptr) { Mutex_->Unlock(); }
			Mutex_ = InOther.Mutex_;
			Owns_  = InOther.Owns_;
			InOther.Mutex_ = nullptr;
			InOther.Owns_  = false;
		}
		return *this;
	}
	OaStdUniqueLock(const OaStdUniqueLock&)            = delete;
	OaStdUniqueLock& operator=(const OaStdUniqueLock&) = delete;

	void Lock()   { if (Mutex_ != nullptr && !Owns_) { Mutex_->Lock();   Owns_ = true;  } }
	void Unlock() { if (Mutex_ != nullptr && Owns_)  { Mutex_->Unlock(); Owns_ = false; } }
	[[nodiscard]] bool OwnsLock() const noexcept { return Owns_; }

private:
	Mutex* Mutex_ = nullptr;
	bool   Owns_  = false;
};

// Short public aliases (canonical names stay OaStd*; see OaStd.md naming).
template<typename T>     using OaAtomic     = OaStdAtomic<T>;
using                          OaMutex       = OaStdMutex;
using                          OaSharedMutex = OaStdSharedMutex;
template<typename Mutex> using OaScopedLock  = OaStdScopedLock<Mutex>;
template<typename Mutex> using OaSharedLock  = OaStdSharedLock<Mutex>;
template<typename Mutex> using OaUniqueLock  = OaStdUniqueLock<Mutex>;
