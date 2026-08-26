#include <oa/core/std/sync.h>
#include <oa/core/assert.h>

#if defined(_WIN32)
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#elif defined(__APPLE__)
	#include <errno.h>
	#include <pthread.h>
#else
	#include <errno.h>
	#include <pthread.h>
#endif

namespace {

#if defined(_WIN32)

using NativeMutex = SRWLOCK;
using NativeSharedMutex = SRWLOCK;
using NativeCondition = CONDITION_VARIABLE;

#else

using NativeMutex = pthread_mutex_t;
using NativeSharedMutex = pthread_rwlock_t;
using NativeCondition = pthread_cond_t;

#endif

static_assert(sizeof(NativeMutex) <= 64);
static_assert(alignof(NativeMutex) <= 16);
static_assert(sizeof(NativeSharedMutex) <= 64);
static_assert(alignof(NativeSharedMutex) <= 16);
static_assert(sizeof(NativeCondition) <= 64);
static_assert(alignof(NativeCondition) <= 16);

template<typename T>
[[nodiscard]] T* native(void* inStorage) noexcept {
	return reinterpret_cast<T*>(inStorage);
}

} // namespace

oa::Mutex::Mutex() noexcept {
#if defined(_WIN32)
	InitializeSRWLock(native<NativeMutex>(storage_));
#else
	OA_REQUIRE(pthread_mutex_init(native<NativeMutex>(storage_), nullptr) == 0);
#endif
}

oa::Mutex::~Mutex() noexcept {
#if !defined(_WIN32)
	OA_REQUIRE(pthread_mutex_destroy(native<NativeMutex>(storage_)) == 0);
#endif
}

void oa::Mutex::lock() noexcept {
#if defined(_WIN32)
	AcquireSRWLockExclusive(native<NativeMutex>(storage_));
#else
	OA_REQUIRE(pthread_mutex_lock(native<NativeMutex>(storage_)) == 0);
#endif
}

bool oa::Mutex::tryLock() noexcept {
#if defined(_WIN32)
	return TryAcquireSRWLockExclusive(native<NativeMutex>(storage_)) != 0;
#else
	const int result = pthread_mutex_trylock(native<NativeMutex>(storage_));
	OA_REQUIRE(result == 0 || result == EBUSY);
	return result == 0;
#endif
}

void oa::Mutex::unlock() noexcept {
#if defined(_WIN32)
	ReleaseSRWLockExclusive(native<NativeMutex>(storage_));
#else
	OA_REQUIRE(pthread_mutex_unlock(native<NativeMutex>(storage_)) == 0);
#endif
}

oa::SharedMutex::SharedMutex() noexcept {
#if defined(_WIN32)
	InitializeSRWLock(native<NativeSharedMutex>(storage_));
#else
	OA_REQUIRE(pthread_rwlock_init(
		native<NativeSharedMutex>(storage_), nullptr) == 0);
#endif
}

oa::SharedMutex::~SharedMutex() noexcept {
#if !defined(_WIN32)
	OA_REQUIRE(pthread_rwlock_destroy(native<NativeSharedMutex>(storage_)) == 0);
#endif
}

void oa::SharedMutex::lock() noexcept {
#if defined(_WIN32)
	AcquireSRWLockExclusive(native<NativeSharedMutex>(storage_));
#else
	OA_REQUIRE(pthread_rwlock_wrlock(native<NativeSharedMutex>(storage_)) == 0);
#endif
}

bool oa::SharedMutex::tryLock() noexcept {
#if defined(_WIN32)
	return TryAcquireSRWLockExclusive(native<NativeSharedMutex>(storage_)) != 0;
#else
	const int result = pthread_rwlock_trywrlock(native<NativeSharedMutex>(storage_));
	OA_REQUIRE(result == 0 || result == EBUSY);
	return result == 0;
#endif
}

void oa::SharedMutex::unlock() noexcept {
#if defined(_WIN32)
	ReleaseSRWLockExclusive(native<NativeSharedMutex>(storage_));
#else
	OA_REQUIRE(pthread_rwlock_unlock(native<NativeSharedMutex>(storage_)) == 0);
#endif
}

void oa::SharedMutex::lockShared() noexcept {
#if defined(_WIN32)
	AcquireSRWLockShared(native<NativeSharedMutex>(storage_));
#else
	OA_REQUIRE(pthread_rwlock_rdlock(native<NativeSharedMutex>(storage_)) == 0);
#endif
}

bool oa::SharedMutex::tryLockShared() noexcept {
#if defined(_WIN32)
	return TryAcquireSRWLockShared(native<NativeSharedMutex>(storage_)) != 0;
#else
	const int result = pthread_rwlock_tryrdlock(native<NativeSharedMutex>(storage_));
	OA_REQUIRE(result == 0 || result == EBUSY);
	return result == 0;
#endif
}

void oa::SharedMutex::unlockShared() noexcept {
#if defined(_WIN32)
	ReleaseSRWLockShared(native<NativeSharedMutex>(storage_));
#else
	OA_REQUIRE(pthread_rwlock_unlock(native<NativeSharedMutex>(storage_)) == 0);
#endif
}

oa::Condition::Condition() noexcept {
#if defined(_WIN32)
	InitializeConditionVariable(native<NativeCondition>(storage_));
#elif defined(__linux__) || defined(__ANDROID__)
	pthread_condattr_t attributes;
	OA_REQUIRE(pthread_condattr_init(&attributes) == 0);
	OA_REQUIRE(pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC) == 0);
	OA_REQUIRE(pthread_cond_init(
		native<NativeCondition>(storage_), &attributes) == 0);
	OA_REQUIRE(pthread_condattr_destroy(&attributes) == 0);
#else
	OA_REQUIRE(pthread_cond_init(native<NativeCondition>(storage_), nullptr) == 0);
#endif
}

bool oa::Condition::waitFor(
	oa::UniqueLock<oa::Mutex>& inLock,
	oa::Duration inDuration
) noexcept {
	OA_REQUIRE(inLock.owns_ && inLock.lock_ != nullptr);
	if (inDuration.nanoseconds() <= 0) return false;
#if defined(_WIN32)
	const oa::I64 milliseconds =
		(inDuration.nanoseconds() + 999'999LL) / 1'000'000LL;
	const DWORD timeout = milliseconds >= static_cast<oa::I64>(INFINITE - 1U)
		? INFINITE - 1U : static_cast<DWORD>(milliseconds);
	if (SleepConditionVariableSRW(
		native<NativeCondition>(storage_),
		native<NativeMutex>(inLock.lock_->storage_),
		timeout, 0) != 0) {
		return true;
	}
	const DWORD error = GetLastError();
	OA_REQUIRE(error == ERROR_TIMEOUT);
	return false;
#else
	timespec deadline{};
	#if defined(__linux__) || defined(__ANDROID__)
		OA_REQUIRE(clock_gettime(CLOCK_MONOTONIC, &deadline) == 0);
	#else
		OA_REQUIRE(clock_gettime(CLOCK_REALTIME, &deadline) == 0);
	#endif
	const oa::I64 durationNanos = inDuration.nanoseconds();
	deadline.tv_sec += static_cast<time_t>(durationNanos / 1'000'000'000LL);
	deadline.tv_nsec += static_cast<long>(durationNanos % 1'000'000'000LL);
	if (deadline.tv_nsec >= 1'000'000'000L) {
		++deadline.tv_sec;
		deadline.tv_nsec -= 1'000'000'000L;
	}
	const int result = pthread_cond_timedwait(
		native<NativeCondition>(storage_),
		native<NativeMutex>(inLock.lock_->storage_),
		&deadline);
	OA_REQUIRE(result == 0 || result == ETIMEDOUT);
	return result == 0;
#endif
}

oa::Condition::~Condition() noexcept {
#if !defined(_WIN32)
	OA_REQUIRE(pthread_cond_destroy(native<NativeCondition>(storage_)) == 0);
#endif
}

void oa::Condition::wait(oa::UniqueLock<oa::Mutex>& inLock) noexcept {
	OA_REQUIRE(inLock.owns_ && inLock.lock_ != nullptr);
#if defined(_WIN32)
	OA_REQUIRE(SleepConditionVariableSRW(
		native<NativeCondition>(storage_),
		native<NativeMutex>(inLock.lock_->storage_),
		INFINITE, 0) != 0);
#else
	OA_REQUIRE(pthread_cond_wait(
		native<NativeCondition>(storage_),
		native<NativeMutex>(inLock.lock_->storage_)) == 0);
#endif
}

void oa::Condition::notifyOne() noexcept {
#if defined(_WIN32)
	WakeConditionVariable(native<NativeCondition>(storage_));
#else
	OA_REQUIRE(pthread_cond_signal(native<NativeCondition>(storage_)) == 0);
#endif
}

void oa::Condition::notifyAll() noexcept {
#if defined(_WIN32)
	WakeAllConditionVariable(native<NativeCondition>(storage_));
#else
	OA_REQUIRE(pthread_cond_broadcast(native<NativeCondition>(storage_)) == 0);
#endif
}
