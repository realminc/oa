#pragma once

// OA synchronization values. Platform synchronization objects remain hidden
// behind fixed, aligned storage; no hosted C++ mutex type or ABI crosses this
// public boundary.

#include <oa/core/std/atomic.h>
#include <oa/core/std/chrono.h>

namespace oa {

class Condition;

class Mutex {
public:
	Mutex() noexcept;
	~Mutex() noexcept;
	Mutex(const Mutex&) = delete;
	Mutex& operator=(const Mutex&) = delete;

	void lock() noexcept;
	[[nodiscard]] bool tryLock() noexcept;
	void unlock() noexcept;

private:
	friend class Condition;
	static constexpr unsigned kStorageSize = 64;
	alignas(16) unsigned char storage_[kStorageSize]{};
};

class SharedMutex {
public:
	SharedMutex() noexcept;
	~SharedMutex() noexcept;
	SharedMutex(const SharedMutex&) = delete;
	SharedMutex& operator=(const SharedMutex&) = delete;

	void lock() noexcept;
	[[nodiscard]] bool tryLock() noexcept;
	void unlock() noexcept;

	void lockShared() noexcept;
	[[nodiscard]] bool tryLockShared() noexcept;
	void unlockShared() noexcept;

private:
	static constexpr unsigned kStorageSize = 64;
	alignas(16) unsigned char storage_[kStorageSize]{};
};

template<typename Lock>
class ScopedLock {
public:
	explicit ScopedLock(Lock& inLock) noexcept : lock_(inLock) { lock_.lock(); }
	~ScopedLock() { lock_.unlock(); }
	ScopedLock(const ScopedLock&) = delete;
	ScopedLock& operator=(const ScopedLock&) = delete;

private:
	Lock& lock_;
};

template<typename Lock>
class SharedLock {
public:
	explicit SharedLock(Lock& inLock) noexcept : lock_(inLock) {
		lock_.lockShared();
	}
	~SharedLock() { lock_.unlockShared(); }
	SharedLock(const SharedLock&) = delete;
	SharedLock& operator=(const SharedLock&) = delete;

private:
	Lock& lock_;
};

template<typename Lock>
class UniqueLock {
public:
	UniqueLock() = default;
	explicit UniqueLock(Lock& inLock) noexcept : lock_(&inLock), owns_(true) {
		lock_->lock();
	}
	~UniqueLock() {
		if (owns_ && lock_ != nullptr) lock_->unlock();
	}

	UniqueLock(UniqueLock&& inOther) noexcept
		: lock_(inOther.lock_), owns_(inOther.owns_) {
		inOther.lock_ = nullptr;
		inOther.owns_ = false;
	}
	UniqueLock& operator=(UniqueLock&& inOther) noexcept {
		if (this == &inOther) return *this;
		if (owns_ && lock_ != nullptr) lock_->unlock();
		lock_ = inOther.lock_;
		owns_ = inOther.owns_;
		inOther.lock_ = nullptr;
		inOther.owns_ = false;
		return *this;
	}
	UniqueLock(const UniqueLock&) = delete;
	UniqueLock& operator=(const UniqueLock&) = delete;

	void lock() noexcept {
		if (lock_ != nullptr && !owns_) {
			lock_->lock();
			owns_ = true;
		}
	}
	void unlock() noexcept {
		if (lock_ != nullptr && owns_) {
			lock_->unlock();
			owns_ = false;
		}
	}
	[[nodiscard]] bool ownsLock() const noexcept { return owns_; }

private:
	friend class Condition;
	Lock* lock_ = nullptr;
	bool owns_ = false;
};

class Condition {
public:
	Condition() noexcept;
	~Condition() noexcept;
	Condition(const Condition&) = delete;
	Condition& operator=(const Condition&) = delete;

	void wait(UniqueLock<Mutex>& inLock) noexcept;
	[[nodiscard]] bool waitFor(
		UniqueLock<Mutex>& inLock,
		oa::Duration inDuration
	) noexcept;
	template<typename Predicate>
	void wait(UniqueLock<Mutex>& inLock, Predicate inPredicate) noexcept {
		while (!inPredicate()) wait(inLock);
	}
	template<typename Predicate>
	[[nodiscard]] bool waitFor(
		UniqueLock<Mutex>& inLock,
		oa::Duration inDuration,
		Predicate inPredicate
	) noexcept {
		const oa::SteadyTimePoint deadline = oa::steadyNow() + inDuration;
		while (!inPredicate()) {
			const oa::SteadyTimePoint current = oa::steadyNow();
			if (current >= deadline) return inPredicate();
			if (!waitFor(inLock, deadline - current)) return inPredicate();
		}
		return true;
	}
	void notifyOne() noexcept;
	void notifyAll() noexcept;

private:
	static constexpr unsigned kStorageSize = 64;
	alignas(16) unsigned char storage_[kStorageSize]{};
};

} // namespace oa
