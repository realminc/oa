// OA private vulkan compute stream.
//
// persistent, reusable command recording + async submission unit.
// Owns a VkCommandPool + VkCommandBuffer + VkFence for its entire lifetime.
// Replaces oavk::Batch — no per-submit create/destroy overhead.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/sync.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/dispatchDesc.h>

namespace oa { class Engine; }
struct VklDeviceTable;

namespace oavk {

class Device;

// API-neutral buffer-copy region. Keeping this out of vulkan-facing call sites
// lets upload producers batch metadata and payload transfers without building
// VkBufferCopy arrays themselves.
struct BufferCopyRegion {
	oa::U64 srcOffset = 0;
	oa::U64 dstOffset = 0;
	oa::U64 size = 0;
};

class Stream {
public:

	// Data, class members.
	void* commandPool = nullptr;
	void* commandBuffer = nullptr;
	// Borrowed immutable dispatch owned by the device that created this stream.
	const VklDeviceTable* deviceDispatch = nullptr;
	TimelineSemaphore timelineSem;
	oa::U64 timelineValue = 0;
	oa::Vector<void*> pendingPools;
	void* queue = nullptr;
	oa::U32 queueFamily = 0;
	oa::Bool recording = false;
	oa::Bool submitted = false;
	// When true, recordBufferBarrier() is a no-op. Used by graph-controlled
	// recording paths that emit precise per-buffer barriers themselves.
	oa::Bool suppressAutoBarrier = false;

	// Constructors.
	Stream() = default;

	// Destructors.
	~Stream() = default;

	// Methods.
	[[nodiscard]] static oa::Result<Stream> create(
		const oavk::Device& inDevice, oa::U32 inQueueFamily, void* inQueue
	);
	[[nodiscard]] static oa::Result<Stream> createCompute(const Device& inDevice);
	void destroy(const oavk::Device& inDevice);
	// reset a recording/executable command buffer that was never submitted.
	// This is the cancellation edge for an abandoned execution-session lease;
	// pending command buffers must complete through their returned oa::Event.
	[[nodiscard]] oa::Status resetUnsubmitted(const oavk::Device& inDevice);

	// ─── recording ────────────────────────────────────────────────────────
	[[nodiscard]] oa::Status begin(const oavk::Device& inDevice);

	// Dispatch + automatic full barrier after (existing behavior). Raw buffers
	// have no dtype metadata; inStorageDtype selects their exact storage ABI.
	[[nodiscard]] oa::Status record(
		oa::Engine& inRt, oa::StringView inPipeline,
		oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX, oa::U32 inGroupsY = 1, oa::U32 inGroupsZ = 1
	);

	// Dispatch only, no barrier — used by oa::ExecutableGraph for precise barriers
	[[nodiscard]] oa::Status recordDispatch(
		oa::Engine& inRt, oa::StringView inPipeline,
		oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX, oa::U32 inGroupsY = 1, oa::U32 inGroupsZ = 1
	);
	// Canonical stream encoder for the engine's selected local device.
	[[nodiscard]] oa::Status recordDispatchDesc(oa::Engine& inRt, const oa::ComputeDispatchDesc& inDesc);

	// indirect dispatch — workgroup counts read from inIndirectBuffer at inOffset.
	// Buffer must have indirect-dispatch usage and contain an aligned
	// VkDispatchIndirectCommand struct (3 x uint32) on the selected device.
	[[nodiscard]] oa::Status recordDispatchIndirect(
		oa::Engine& inRt, oa::StringView inPipeline,
		oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset = 0
	);

	void recordCopyBuffer(const oavk::Buffer& inSrc, const oavk::Buffer& inDst, oa::U64 inSize);
	void recordCopyBufferRegions(
		const oavk::Buffer& inSrc,
		const oavk::Buffer& inDst,
		oa::Span<const BufferCopyRegion> inRegions
	);
	// Make prior device writes to a copy source visible to the next transfer
	// read. alias-backed sources use a global memory dependency; ordinary
	// buffers retain the exact source range. Flushed host writes are made
	// visible by queue submission and do not require this barrier.
	void recordTransferReadBarrier(
		const oavk::Buffer& inSrc,
		oa::U64 inOffset,
		oa::U64 inSize
	);
	// Make prior transfer writes visible to later same-queue buffer accesses
	// and to host reads after submission completion. alias-backed destinations
	// use a global memory dependency; ordinary buffers retain the exact range.
	// A consumer on another queue in the same family must also wait the
	// submission's timeline event. A different family additionally needs an
	// explicit ownership transfer.
	void recordTransferWriteBarrier(
		const oavk::Buffer& inDst,
		oa::U64 inOffset,
		oa::U64 inSize
	);
	void recordBufferBarrier();
	// One graph/batch-final visibility edge from all prior device writes to
	// mapped host readback. Device-only intermediates do not emit it individually.
	void recordHostReadbackBarrier();
	// Emit precise per-buffer barriers. inBufs/inCount specify which buffers need
	// COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ|WRITE synchronization.
	void recordBufferMemoryBarriers(const oavk::Buffer* inBufs, oa::U32 inCount);

	// ─── Submission ───────────────────────────────────────────────────────
	[[nodiscard]] oa::Status submit(oa::Engine& inRt);

	// submit with a GPU-side dependency: wait on inWaitSem reaching inWaitValue
	// before executing this command buffer. Used for cross-queue sync.
	[[nodiscard]] oa::Status submitWithDependency(
		oa::Engine& inRt,
		const TimelineSemaphore& inWaitSem,
		oa::U64 inWaitValue
	);
	[[nodiscard]] oa::Status submitWithDependencies(
		oa::Engine& inRt,
		oa::Span<const TimelineWait> inWaits
	);

	[[nodiscard]] oa::Status synchronize(const oavk::Device& inDevice);
	[[nodiscard]] oa::Status submitAndWait(oa::Engine& inRt);
	[[nodiscard]] oa::Bool isComplete(const oavk::Device& inDevice) const;
	[[nodiscard]] oa::Event completion(const oavk::Device& inDevice) const {
		return submitted
			? oa::EventAccess::create(
				inDevice, timelineSem, timelineValue, queueFamily)
			: oa::Event();
	}

	// ─── Single-Shot ──────────────────────────────────────────────────────
	[[nodiscard]] static oa::Status runOnce(
		oa::Engine& inRt, oa::StringView inPipeline,
		oa::Span<oavk::Buffer> inBufs, const void* inPush, oa::U32 inPushSize,
		oa::ScalarType inStorageDtype,
		oa::U32 inGroupsX, oa::U32 inGroupsY = 1, oa::U32 inGroupsZ = 1
	);

	// Operators.
	Stream(Stream&& inOther) noexcept;
	Stream& operator=(Stream&& inOther) noexcept;
	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;
};

} // namespace oavk
