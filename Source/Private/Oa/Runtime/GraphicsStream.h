// Private engine-owned graphics recording lease.
//
// This is the narrow low-level encoder seam for private Render code. It is not
// an installed public API: GetStream() deliberately exposes the existing raw
// command encoder only inside OA implementation code. The engine owns the
// stream, queue, timeline, pool slot, and retirement.
//
// Acquisition never waits for an older submission. Submit returns the exact
// timeline event for the recorded command buffer. Recycle() accepts only that
// exact completed event. Close() and destruction never submit or wait: they
// cancel unsubmitted recording or transfer in-flight work to engine retirement.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;
class OaVkStream;

class OaGraphicsStreamLease {
public:
	OaGraphicsStreamLease() = default;
	OaGraphicsStreamLease(const OaGraphicsStreamLease&) = delete;
	OaGraphicsStreamLease& operator=(const OaGraphicsStreamLease&) = delete;
	OaGraphicsStreamLease(OaGraphicsStreamLease&& InOther) noexcept;
	OaGraphicsStreamLease& operator=(OaGraphicsStreamLease&& InOther) noexcept;
	~OaGraphicsStreamLease();

	[[nodiscard]] static OaResult<OaGraphicsStreamLease> Acquire(
		OaEngine& InEngine);

	[[nodiscard]] bool IsValid() const noexcept;
	// Recording access only. Callers must submit through this lease rather than
	// OaVkStream::Submit(), so exact-event and retirement state stay canonical.
	[[nodiscard]] OaVkStream* GetStream() noexcept;

	[[nodiscard]] OaResult<OaEvent> Submit();
	// Dependencies must belong to this engine and originate from the graphics
	// queue family. A timeline wait orders same-family work; it does not perform
	// an exclusive-resource queue-family ownership transfer.
	[[nodiscard]] OaResult<OaEvent> Submit(
		OaSpan<const OaEvent> InDependencies);

	[[nodiscard]] OaStatus Cancel();
	[[nodiscard]] OaStatus Recycle(const OaEvent& InCompletion);
	[[nodiscard]] OaStatus Close();

private:
	friend class OaEngine;

	OaGraphicsStreamLease(
		OaEngine& InEngine, OaU32 InSlot, OaU64 InGeneration) noexcept;
	void Reset_() noexcept;

	OaEngine* Engine_ = nullptr;
	OaU32 Slot_ = 0;
	OaU64 Generation_ = 0;
};
