// Private engine-owned graphics recording lease.
//
// This is the narrow low-level encoder seam for private Render code. It is not
// an installed public API: getStream() deliberately exposes the existing raw
// command encoder only inside OA implementation code. The engine owns the
// stream, queue, timeline, pool slot, and retirement.
//
// Acquisition never waits for an older submission. submit returns the exact
// timeline event for the recorded command buffer. recycle() accepts only that
// exact completed event. close() and destruction never submit or wait: they
// cancel unsubmitted recording or transfer in-flight work to engine retirement.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/sync.h>

namespace oa { class Engine; }
namespace oavk { class Stream; }

namespace oa {

class GraphicsStreamLease {
public:
	GraphicsStreamLease() = default;
	GraphicsStreamLease(const GraphicsStreamLease&) = delete;
	GraphicsStreamLease& operator=(const GraphicsStreamLease&) = delete;
	GraphicsStreamLease(GraphicsStreamLease&& inOther) noexcept;
	GraphicsStreamLease& operator=(GraphicsStreamLease&& inOther) noexcept;
	~GraphicsStreamLease();

	[[nodiscard]] static oa::Result<GraphicsStreamLease> acquire(
		oa::Engine& inEngine);

	[[nodiscard]] bool isValid() const noexcept;
	// Recording access only. Callers must submit through this lease rather than
	// oavk::Stream::submit(), so exact-event and retirement state stay canonical.
	[[nodiscard]] oavk::Stream* getStream() noexcept;

	[[nodiscard]] oa::Result<oa::Event> submit();
	// dependencies must belong to this engine and originate from the graphics
	// queue family. A timeline wait orders same-family work; it does not perform
	// an exclusive-resource queue-family ownership transfer.
	[[nodiscard]] oa::Result<oa::Event> submit(
		oa::Span<const oa::Event> inDependencies);

	[[nodiscard]] oa::Status cancel();
	[[nodiscard]] oa::Status recycle(const oa::Event& inCompletion);
	[[nodiscard]] oa::Status close();

private:
	friend class oa::Engine;

	GraphicsStreamLease(
		oa::Engine& inEngine, oa::U32 inSlot, oa::U64 inGeneration) noexcept;
	void reset_() noexcept;

	oa::Engine* engine_ = nullptr;
	oa::U32 slot_ = 0;
	oa::U64 generation_ = 0;
};

} // namespace oa
