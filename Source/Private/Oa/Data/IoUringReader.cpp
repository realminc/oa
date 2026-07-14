// io_uring async reader — direct pread into pinned memory
// Uses Linux io_uring to submit async pread() SQEs that read directly from
// NVMe into pinned memory. No page faults, no kernel page cache middleman.
// Queue depth of 32 means 32 concurrent reads in-flight.

#include <Oa/Data/IoUringReader.h>
#include <cstdio>

#ifdef OA_PLATFORM_LINUX

#include <liburing.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

OaIoUringReader::OaIoUringReader(const OaString& InPath, OaI32 InQueueDepth) {
	// Open file with O_DIRECT for bypass of page cache
	Fd_ = open(InPath.c_str(), O_RDONLY | O_DIRECT);
	if (Fd_ < 0) {
		// Fallback: try without O_DIRECT (some filesystems don't support it)
		Fd_ = open(InPath.c_str(), O_RDONLY);
		if (Fd_ < 0) {
			fprintf(stderr, "[OaIoUringReader] Failed to open: %s\n", InPath.c_str());
			return;
		}
	}

	struct stat st;
	if (fstat(Fd_, &st) != 0) {
		fprintf(stderr, "[OaIoUringReader] Failed to stat: %s\n", InPath.c_str());
		close(Fd_);
		Fd_ = -1;
		return;
	}
	FileSize_ = static_cast<OaI64>(st.st_size);

	// Initialize io_uring
	auto* ring = new struct io_uring;
	int ret = io_uring_queue_init(static_cast<unsigned>(InQueueDepth), ring, 0);
	if (ret < 0) {
		fprintf(stderr, "[OaIoUringReader] io_uring_queue_init failed: %d\n", ret);
		delete ring;
		close(Fd_);
		Fd_ = -1;
		return;
	}

	Ring_ = ring;
	RingInitialized_ = true;
}

OaIoUringReader::~OaIoUringReader() {
	if (RingInitialized_ && Ring_ != nullptr) {
		io_uring_queue_exit(static_cast<struct io_uring*>(Ring_));
		delete static_cast<struct io_uring*>(Ring_);
	}
	if (Fd_ >= 0) {
		close(Fd_);
	}
}

OaStatus OaIoUringReader::SubmitRead(void* InDst, OaI64 InFileOffset, OaI64 InBytes) {
	if (!RingInitialized_) {
		return OaStatus::Error("io_uring not initialized");
	}

	auto* ring = static_cast<struct io_uring*>(Ring_);
	struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
	if (sqe == nullptr) {
		return OaStatus::Error("io_uring SQ full — no free SQE");
	}

	io_uring_prep_read(sqe, Fd_, InDst,
		static_cast<unsigned>(InBytes), static_cast<__u64>(InFileOffset));
	io_uring_sqe_set_data(sqe, InDst);  // Tag with destination for tracking

	int ret = io_uring_submit(ring);
	if (ret < 0) {
		return OaStatus::Error("io_uring_submit failed");
	}

	return OaStatus::Ok();
}

OaI32 OaIoUringReader::WaitCompletions(OaI32 InMinComplete) {
	if (!RingInitialized_) return 0;

	auto* ring = static_cast<struct io_uring*>(Ring_);
	struct io_uring_cqe* cqe = nullptr;
	OaI32 completed = 0;

	for (OaI32 i = 0; i < InMinComplete; ++i) {
		int ret = io_uring_wait_cqe(ring, &cqe);
		if (ret < 0) break;
		if (cqe->res < 0) {
			fprintf(stderr, "[OaIoUringReader] Read failed: %d\n", cqe->res);
		}
		io_uring_cqe_seen(ring, cqe);
		completed++;
	}

	// Drain any additional completions without waiting
	while (io_uring_peek_cqe(ring, &cqe) == 0) {
		io_uring_cqe_seen(ring, cqe);
		completed++;
	}

	return completed;
}

#endif // OA_PLATFORM_LINUX
