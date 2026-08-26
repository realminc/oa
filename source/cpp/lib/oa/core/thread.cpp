#include <oa/core/thread.h>

#include <oa/core/assert.h>

#if defined(OA_PLATFORM_WINDOWS)
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <process.h>
	#include <windows.h>
#elif defined(OA_PLATFORM_APPLE)
	#include <errno.h>
	#include <pthread.h>
	#include <sched.h>
	#include <sys/sysctl.h>
	#include <time.h>
#else
	#include <errno.h>
	#include <pthread.h>
	#include <sched.h>
	#include <time.h>
	#include <unistd.h>
#endif

namespace {

#if defined(OA_PLATFORM_WINDOWS)
using NativeThread = HANDLE;
#else
using NativeThread = pthread_t;
#endif

static_assert(sizeof(NativeThread) <= 16);
static_assert(alignof(NativeThread) <= 16);

template<typename T>
[[nodiscard]] T* native(void* inStorage) noexcept {
	return reinterpret_cast<T*>(inStorage);
}

struct ThreadEntry {
	oa::Fn<void()> function;
};

#if defined(OA_PLATFORM_WINDOWS)
unsigned __stdcall runThread(void* inPayload) {
#else
void* runThread(void* inPayload) {
#endif
	oa::UniquePtr<ThreadEntry> entry(static_cast<ThreadEntry*>(inPayload));
	entry->function();
#if defined(OA_PLATFORM_WINDOWS)
	return 0U;
#else
	return nullptr;
#endif
}

} // namespace

oa::Thread::~Thread() noexcept {
	OA_REQUIRE(!joinable_);
}

oa::Thread::Thread(oa::Thread&& inOther) noexcept
	: joinable_(inOther.joinable_) {
	for (unsigned index = 0; index < kStorageSize; ++index) {
		storage_[index] = inOther.storage_[index];
	}
	inOther.joinable_ = false;
}

oa::Thread& oa::Thread::operator=(oa::Thread&& inOther) noexcept {
	if (this == &inOther) return *this;
	OA_REQUIRE(!joinable_);
	for (unsigned index = 0; index < kStorageSize; ++index) {
		storage_[index] = inOther.storage_[index];
	}
	joinable_ = inOther.joinable_;
	inOther.joinable_ = false;
	return *this;
}

oa::Result<oa::Thread> oa::Thread::create(oa::Fn<void()> inEntry) {
	if (!inEntry) {
		return oa::Status::invalidArgument("oa::Thread requires an entry function");
	}
	auto entry = oa::makeUnique<ThreadEntry>(ThreadEntry{oa::move(inEntry)});
	oa::Thread result;
#if defined(OA_PLATFORM_WINDOWS)
	const auto handle = _beginthreadex(
		nullptr, 0U, &runThread, entry.get(), 0U, nullptr);
	if (handle == 0U) {
		return oa::Status::error(
			oa::StatusCode::Unavailable, "_beginthreadex failed");
	}
	*native<NativeThread>(result.storage_) = reinterpret_cast<HANDLE>(handle);
#else
	NativeThread thread{};
	const int error = pthread_create(&thread, nullptr, &runThread, entry.get());
	if (error != 0) {
		return oa::Status::error(
			oa::StatusCode::Unavailable, "pthread_create failed");
	}
	*native<NativeThread>(result.storage_) = thread;
#endif
	(void)entry.release();
	result.joinable_ = true;
	return result;
}

oa::Status oa::Thread::join() noexcept {
	if (!joinable_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition, "oa::Thread is not joinable");
	}
#if defined(OA_PLATFORM_WINDOWS)
	HANDLE handle = *native<NativeThread>(storage_);
	if (WaitForSingleObject(handle, INFINITE) != WAIT_OBJECT_0) {
		CloseHandle(handle);
		joinable_ = false;
		return oa::Status::error(oa::StatusCode::Internal, "thread join failed");
	}
	CloseHandle(handle);
#else
	const int error = pthread_join(*native<NativeThread>(storage_), nullptr);
	if (error != 0) {
		(void)pthread_detach(*native<NativeThread>(storage_));
		joinable_ = false;
		return oa::Status::error(oa::StatusCode::Internal, "pthread_join failed");
	}
#endif
	joinable_ = false;
	return oa::Status::ok();
}

oa::Status oa::Thread::detach() noexcept {
	if (!joinable_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition, "oa::Thread is not joinable");
	}
#if defined(OA_PLATFORM_WINDOWS)
	CloseHandle(*native<NativeThread>(storage_));
#else
	if (pthread_detach(*native<NativeThread>(storage_)) != 0) {
		return oa::Status::error(oa::StatusCode::Internal, "pthread_detach failed");
	}
#endif
	joinable_ = false;
	return oa::Status::ok();
}

oa::U32 oa::Thread::hardwareConcurrency() noexcept {
#if defined(OA_PLATFORM_WINDOWS)
	const DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
	return count > 0U ? static_cast<oa::U32>(count) : 1U;
#elif defined(OA_PLATFORM_APPLE)
	int count = 0;
	size_t size = sizeof(count);
	if (sysctlbyname("hw.logicalcpu", &count, &size, nullptr, 0) == 0 && count > 0) {
		return static_cast<oa::U32>(count);
	}
	return 1U;
#else
	const long count = sysconf(_SC_NPROCESSORS_ONLN);
	return count > 0 ? static_cast<oa::U32>(count) : 1U;
#endif
}

void oa::Thread::yield() noexcept {
#if defined(OA_PLATFORM_WINDOWS)
	(void)SwitchToThread();
#else
	(void)sched_yield();
#endif
}

void oa::Thread::sleepFor(oa::Duration inDuration) noexcept {
	if (inDuration.nanoseconds() <= 0) return;
#if defined(OA_PLATFORM_WINDOWS)
	const oa::I64 milliseconds =
		(inDuration.nanoseconds() + 999'999LL) / 1'000'000LL;
	Sleep(milliseconds >= static_cast<oa::I64>(INFINITE - 1U)
		? INFINITE - 1U : static_cast<DWORD>(milliseconds));
#else
	timespec remaining{
		static_cast<time_t>(inDuration.nanoseconds() / 1'000'000'000LL),
		static_cast<long>(inDuration.nanoseconds() % 1'000'000'000LL)
	};
	while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {}
#endif
}
