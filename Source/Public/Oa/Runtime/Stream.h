// OA VULKAN - Compute Stream
//
// Persistent, reusable command recording + async submission unit.
// Owns a VkCommandPool + VkCommandBuffer + VkFence for its entire lifetime.
// Replaces OaVkBatch — no per-submit create/destroy overhead.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/DispatchDesc.h>

class OaEngine;
class OaVkDevice;

// API-neutral buffer-copy region. Keeping this out of Vulkan-facing call sites
// lets upload producers batch metadata and payload transfers without building
// VkBufferCopy arrays themselves.
struct OaBufferCopyRegion {
	OaU64 SrcOffset = 0;
	OaU64 DstOffset = 0;
	OaU64 Size = 0;
};

class OaVkStream {
public:

	// Data, class members.
	void* CommandPool = nullptr;
	void* CommandBuffer = nullptr;
	OaVkTimelineSemaphore TimelineSem;
	OaU64 TimelineValue = 0;
	OaVec<void*> PendingPools;
	void* Queue = nullptr;
	OaU32 QueueFamily = 0;
	OaBool Recording = false;
	OaBool Submitted = false;
	// When true, RecordBufferBarrier() is a no-op. Used by graph-controlled
	// recording paths that emit precise per-buffer barriers themselves.
	OaBool SuppressAutoBarrier = false;
	// VkDeviceMesh: queue submit targets this node (0 = engine primary / default pool).
	OaU32 MeshNodeIndex = 0;

	// Constructors.
	OaVkStream() = default;

	// Destructors.
	~OaVkStream() = default;

	// Methods.
	[[nodiscard]] static OaResult<OaVkStream> Create(
		const OaVkDevice& InDevice, OaU32 InQueueFamily, void* InQueue
	);
	[[nodiscard]] static OaResult<OaVkStream> CreateCompute(const OaVkDevice& InDevice);
	void Destroy(const OaVkDevice& InDevice);
	// Reset a recording/executable command buffer that was never submitted.
	// This is the cancellation edge for an abandoned execution-session lease;
	// pending command buffers must complete through their returned OaEvent.
	[[nodiscard]] OaStatus ResetUnsubmitted(const OaVkDevice& InDevice);

	// ─── Recording ────────────────────────────────────────────────────────
	[[nodiscard]] OaStatus Begin(const OaVkDevice& InDevice);

	// Dispatch + automatic full barrier after (existing behavior)
	[[nodiscard]] OaStatus Record(
		OaEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Dispatch only, no barrier — used by OaComputeGraph for precise barriers
	[[nodiscard]] OaStatus RecordDispatch(
		OaEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);
	// Canonical stream encoder. Direct, indirect, primary-device and mesh-node
	// wrappers all lower to this exact descriptor contract.
	[[nodiscard]] OaStatus RecordDispatchDesc(OaEngine& InRt, const OaComputeDispatchDesc& InDesc);

	// Same as RecordDispatch but resolves pipeline + bindless on mesh node InNodeIndex.
	[[nodiscard]] OaStatus RecordDispatchOnNode(
		OaEngine& InRt, OaU32 InNodeIndex, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Indirect dispatch — workgroup counts read from InIndirectBuffer at InOffset.
	// Buffer must contain a VkDispatchIndirectCommand struct (3 x uint32).
	[[nodiscard]] OaStatus RecordDispatchIndirect(
		OaEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer, OaU64 InOffset = 0
	);

	void RecordCopyBuffer(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize);
	void RecordCopyBufferRegions(
		const OaVkBuffer& InSrc,
		const OaVkBuffer& InDst,
		OaSpan<const OaBufferCopyRegion> InRegions
	);
	// Make prior device writes to a copy source visible to the next transfer
	// read. Alias-backed sources use a global memory dependency; ordinary
	// buffers retain the exact source range. Flushed host writes are made
	// visible by queue submission and do not require this barrier.
	void RecordTransferReadBarrier(
		const OaVkBuffer& InSrc,
		OaU64 InOffset,
		OaU64 InSize
	);
	// Make prior transfer writes visible to later same-queue buffer accesses
	// and to host reads after submission completion. Alias-backed destinations
	// use a global memory dependency; ordinary buffers retain the exact range.
	// A consumer on another queue in the same family must also wait the
	// submission's timeline event. A different family additionally needs an
	// explicit ownership transfer.
	void RecordTransferWriteBarrier(
		const OaVkBuffer& InDst,
		OaU64 InOffset,
		OaU64 InSize
	);
	void RecordBufferBarrier();
	// One graph/batch-final visibility edge from all prior device writes to
	// mapped host readback. Device-only intermediates do not emit it individually.
	void RecordHostReadbackBarrier();
	// Emit precise per-buffer barriers. InBufs/InCount specify which buffers need
	// COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ|WRITE synchronization.
	void RecordBufferMemoryBarriers(const OaVkBuffer* InBufs, OaU32 InCount);

	// ─── Submission ───────────────────────────────────────────────────────
	// For mesh node streams (MeshNodeIndex != 0), if InDispatchAlreadyLoadedForNode,
	// caller holds DeviceLoadLock and global dispatch matches that node (see RunOn).
	[[nodiscard]] OaStatus Submit(OaEngine& InRt, OaBool InDispatchAlreadyLoadedForNode = false);

	// Submit with a GPU-side dependency: wait on InWaitSem reaching InWaitValue
	// before executing this command buffer. Used for cross-queue sync.
	[[nodiscard]] OaStatus SubmitWithDependency(
		OaEngine& InRt,
		const OaVkTimelineSemaphore& InWaitSem,
		OaU64 InWaitValue,
		OaBool InDispatchAlreadyLoadedForNode = false
	);
	[[nodiscard]] OaStatus SubmitWithDependencies(
		OaEngine& InRt,
		OaSpan<const OaVkTimelineWait> InWaits,
		OaBool InDispatchAlreadyLoadedForNode = false
	);

	[[nodiscard]] OaStatus Synchronize(const OaVkDevice& InDevice);
	[[nodiscard]] OaStatus SubmitAndWait(OaEngine& InRt, OaBool InDispatchAlreadyLoadedForNode = false);
	[[nodiscard]] OaBool IsComplete(const OaVkDevice& InDevice) const;
	[[nodiscard]] OaCompletionToken Completion(const OaVkDevice& InDevice) const {
		return Submitted
			? OaCompletionToken(InDevice, TimelineSem, TimelineValue, QueueFamily)
			: OaCompletionToken();
	}

	// ─── Single-Shot ──────────────────────────────────────────────────────
	[[nodiscard]] static OaStatus RunOnce(
		OaEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Operators.
	OaVkStream(OaVkStream&& InOther) noexcept;
	OaVkStream& operator=(OaVkStream&& InOther) noexcept;
	OaVkStream(const OaVkStream&) = delete;
	OaVkStream& operator=(const OaVkStream&) = delete;
};
