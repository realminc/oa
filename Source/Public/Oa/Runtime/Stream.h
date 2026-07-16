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

class OaComputeEngine;
class OaVkDevice;

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

	// ─── Recording ────────────────────────────────────────────────────────
	[[nodiscard]] OaStatus Begin(const OaVkDevice& InDevice);

	// Dispatch + automatic full barrier after (existing behavior)
	[[nodiscard]] OaStatus Record(
		OaComputeEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Dispatch only, no barrier — used by OaComputeGraph for precise barriers
	[[nodiscard]] OaStatus RecordDispatch(
		OaComputeEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);
	// Canonical stream encoder. Direct, indirect, primary-device and mesh-node
	// wrappers all lower to this exact descriptor contract.
	[[nodiscard]] OaStatus RecordDispatchDesc(
		OaComputeEngine& InRt, const OaComputeDispatchDesc& InDesc);

	// Same as RecordDispatch but resolves pipeline + bindless on mesh node InNodeIndex.
	[[nodiscard]] OaStatus RecordDispatchOnNode(
		OaComputeEngine& InRt, OaU32 InNodeIndex, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Indirect dispatch — workgroup counts read from InIndirectBuffer at InOffset.
	// Buffer must contain a VkDispatchIndirectCommand struct (3 x uint32).
	[[nodiscard]] OaStatus RecordDispatchIndirect(
		OaComputeEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer, OaU64 InOffset = 0
	);

	void RecordCopyBuffer(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize);
	void RecordBufferBarrier();
	// One graph/batch-final visibility edge for mapped host readback. Device-only
	// intermediate graphs must not emit this barrier individually.
	void RecordHostReadbackBarrier();
	// Emit precise per-buffer barriers. InBufs/InCount specify which buffers need
	// COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ|WRITE synchronization.
	void RecordBufferMemoryBarriers(const OaVkBuffer* InBufs, OaU32 InCount);

	// ─── Submission ───────────────────────────────────────────────────────
	// For mesh node streams (MeshNodeIndex != 0), if InDispatchAlreadyLoadedForNode,
	// caller holds DeviceLoadLock and global dispatch matches that node (see RunOn).
	[[nodiscard]] OaStatus Submit(OaComputeEngine& InRt, OaBool InDispatchAlreadyLoadedForNode = false);

	// Submit with a GPU-side dependency: wait on InWaitSem reaching InWaitValue
	// before executing this command buffer. Used for cross-queue sync.
	[[nodiscard]] OaStatus SubmitWithDependency(
		OaComputeEngine& InRt,
		const OaVkTimelineSemaphore& InWaitSem,
		OaU64 InWaitValue,
		OaBool InDispatchAlreadyLoadedForNode = false
	);
	[[nodiscard]] OaStatus SubmitWithDependencies(
		OaComputeEngine& InRt,
		OaSpan<const OaVkTimelineWait> InWaits,
		OaBool InDispatchAlreadyLoadedForNode = false
	);

	[[nodiscard]] OaStatus Synchronize(const OaVkDevice& InDevice);
	[[nodiscard]] OaStatus SubmitAndWait(OaComputeEngine& InRt, OaBool InDispatchAlreadyLoadedForNode = false);
	[[nodiscard]] OaBool IsComplete(const OaVkDevice& InDevice) const;
	[[nodiscard]] OaCompletionToken Completion(const OaVkDevice& InDevice) const {
		return Submitted
			? OaCompletionToken(InDevice, TimelineSem, TimelineValue)
			: OaCompletionToken();
	}

	// ─── Single-Shot ──────────────────────────────────────────────────────
	[[nodiscard]] static OaStatus RunOnce(
		OaComputeEngine& InRt, OaStringView InPipeline,
		OaSpan<OaVkBuffer> InBufs, const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1
	);

	// Operators.
	OaVkStream(OaVkStream&& InOther) noexcept;
	OaVkStream& operator=(OaVkStream&& InOther) noexcept;
	OaVkStream(const OaVkStream&) = delete;
	OaVkStream& operator=(const OaVkStream&) = delete;
};
