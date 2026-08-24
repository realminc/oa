// OA Runtime — persistent mapped CPU-to-GPU upload ring.
//
// The ring amortizes staging allocation, mapping, command recording and queue
// submission across many uploads. Each frame owns one fixed arena and one
// persistent transfer stream. Reusing a frame waits only when the producer
// laps work that is still in flight.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/sync.h>

namespace oa { class Engine; }
namespace oavk { class Buffer; }

namespace oa {

struct UploadRingConfig {
	oa::U64 capacityBytes = 64ULL * 1024ULL * 1024ULL;
	oa::U32 framesInFlight = 4;
	oa::U64 alignment = 256;
};

struct UploadSlice {
	void* mapped = nullptr;
	oa::U64 offset = 0;
	oa::U64 size = 0;

	[[nodiscard]] bool isValid() const noexcept {
		return mapped != nullptr && size != 0;
	}
};

class UploadRing {
public:
	UploadRing() = default;
	UploadRing(UploadRing&&) noexcept;
	UploadRing& operator=(UploadRing&&) noexcept;
	UploadRing(const UploadRing&) = delete;
	UploadRing& operator=(const UploadRing&) = delete;
	~UploadRing();

	[[nodiscard]] static oa::Result<oa::UploadRing> create(oa::Engine& inEngine, const oa::UploadRingConfig& inConfig = {});

	// Begin one producer batch. This may wait only for the frame slot being
	// recycled; other in-flight frames remain asynchronous.
	[[nodiscard]] oa::Status beginBatch();
	[[nodiscard]] oa::Result<oa::UploadSlice> reserve(oa::U64 inSize, oa::U64 inAlignment = 0);
	[[nodiscard]] oa::Status enqueueCopy(
		const oa::UploadSlice& inSlice,
		const oavk::Buffer& inDst,
		oa::U64 inDstOffset = 0
	);
	[[nodiscard]] oa::Status upload(
		const oavk::Buffer& inDst,
		oa::U64 inDstOffset,
		const void* inData,
		oa::U64 inSize,
		oa::U64 inAlignment = 0
	);

	// Flushes mapped writes, records all queued copies into one command buffer,
	// and submits once. The returned token should be chained into GPU consumers;
	// host waiting is only required before CPU access or explicit close.
	[[nodiscard]] oa::Result<oa::Event> submit();
	[[nodiscard]] oa::Status wait();
	// Explicit host completion boundary. Destruction itself never submits or
	// waits: it cancels an open batch and transfers submitted frame/staging
	// lifetime to the engine for retirement at engine close.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] oa::U64 capacityBytes() const noexcept;
	[[nodiscard]] oa::U64 frameCapacityBytes() const noexcept;
	[[nodiscard]] oa::U64 bytesUsed() const noexcept;
	[[nodiscard]] oa::U32 pendingCopyCount() const noexcept;
	[[nodiscard]] bool isBatchOpen() const noexcept;

private:
	struct Impl;
	void abandon_() noexcept;
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
