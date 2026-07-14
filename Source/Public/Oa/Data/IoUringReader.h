// IoUringReader — Linux io_uring async file reader
//
// Direct pread into pinned memory. Uses Linux io_uring to submit async
// pread() SQEs that read directly from NVMe into pinned memory buffers.
// No page fault latency, no intermediate kernel page cache copy.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

#ifdef OA_PLATFORM_LINUX

class OaIoUringReader {
public:
	/// Open file for async reads. InQueueDepth is the io_uring SQ size.
	OaIoUringReader(const OaString& InPath, OaI32 InQueueDepth = 32);
	~OaIoUringReader();

	// Non-copyable, non-movable
	OaIoUringReader(const OaIoUringReader&) = delete;
	OaIoUringReader& operator=(const OaIoUringReader&) = delete;

	/// Submit async read of InBytes from InFileOffset into InDst (pinned memory).
	[[nodiscard]] OaStatus SubmitRead(void* InDst, OaI64 InFileOffset, OaI64 InBytes);

	/// Wait for at least InMinComplete completions. Returns number completed.
	[[nodiscard]] OaI32 WaitCompletions(OaI32 InMinComplete = 1);

	/// File size in bytes
	[[nodiscard]] OaI64 FileSize() const { return FileSize_; }

	/// Whether the reader is valid (file opened, ring initialized)
	[[nodiscard]] bool IsValid() const { return Fd_ >= 0 && RingInitialized_; }

private:
	void* Ring_ = nullptr;  // struct io_uring* (opaque to avoid liburing.h in public header)
	int Fd_ = -1;
	OaI64 FileSize_ = 0;
	bool RingInitialized_ = false;
};

#endif // OA_PLATFORM_LINUX
