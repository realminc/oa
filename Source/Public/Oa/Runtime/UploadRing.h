// OA Runtime — persistent mapped CPU-to-GPU upload ring.
//
// The ring amortizes staging allocation, mapping, command recording and queue
// submission across many uploads. Each frame owns one fixed arena and one
// persistent transfer stream. Reusing a frame waits only when the producer
// laps work that is still in flight.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;
class OaVkBuffer;

struct OaUploadRingConfig {
	OaU64 CapacityBytes = 64ULL * 1024ULL * 1024ULL;
	OaU32 FramesInFlight = 4;
	OaU64 Alignment = 256;
};

struct OaUploadSlice {
	void* Mapped = nullptr;
	OaU64 Offset = 0;
	OaU64 Size = 0;

	[[nodiscard]] bool IsValid() const noexcept {
		return Mapped != nullptr && Size != 0;
	}
};

class OaUploadRing {
public:
	OaUploadRing() = default;
	OaUploadRing(OaUploadRing&&) noexcept;
	OaUploadRing& operator=(OaUploadRing&&) noexcept;
	OaUploadRing(const OaUploadRing&) = delete;
	OaUploadRing& operator=(const OaUploadRing&) = delete;
	~OaUploadRing();

	[[nodiscard]] static OaResult<OaUploadRing> Create(
		OaEngine& InEngine,
		const OaUploadRingConfig& InConfig = {});

	// Begin one producer batch. This may wait only for the frame slot being
	// recycled; other in-flight frames remain asynchronous.
	[[nodiscard]] OaStatus BeginBatch();
	[[nodiscard]] OaResult<OaUploadSlice> Reserve(
		OaU64 InSize, OaU64 InAlignment = 0);
	[[nodiscard]] OaStatus EnqueueCopy(
		const OaUploadSlice& InSlice,
		const OaVkBuffer& InDst,
		OaU64 InDstOffset = 0);
	[[nodiscard]] OaStatus Upload(
		const OaVkBuffer& InDst,
		OaU64 InDstOffset,
		const void* InData,
		OaU64 InSize,
		OaU64 InAlignment = 0);

	// Flushes mapped writes, records all queued copies into one command buffer,
	// and submits once. The returned token should be chained into GPU consumers;
	// host waiting is only required before CPU access or explicit close.
	[[nodiscard]] OaResult<OaCompletionToken> Submit();
	[[nodiscard]] OaStatus Wait();
	// Explicit host completion boundary. Destruction itself never submits or
	// waits: it cancels an open batch and transfers submitted frame/staging
	// lifetime to the engine for retirement at engine close.
	[[nodiscard]] OaStatus Close();
	void Destroy();

	[[nodiscard]] OaU64 CapacityBytes() const noexcept;
	[[nodiscard]] OaU64 FrameCapacityBytes() const noexcept;
	[[nodiscard]] OaU64 BytesUsed() const noexcept;
	[[nodiscard]] OaU32 PendingCopyCount() const noexcept;
	[[nodiscard]] bool IsBatchOpen() const noexcept;

private:
	struct Impl;
	void Abandon_() noexcept;
	OaUniquePtr<Impl> Impl_;
};
