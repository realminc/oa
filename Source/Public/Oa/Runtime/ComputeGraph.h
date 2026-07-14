#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Sync.h>

class OaComputeEngine;
class OaVkDevice;
class OaDeviceMesh;
class OaScheduler;

enum class OaQueueHint : OaU8 {
	Compute,
	AsyncCompute,
	Transfer,
};

// Compute graph node — one dispatch with buffer access annotations.
// Push constant data is copied inline (max 128 bytes per Vulkan spec).
class OaComputeNode {
public:
	OaString Shader;
	OaVec<OaVkBuffer> Buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> BufferOwners;
	OaVec<OaBufferAccess> Access;
	alignas(16) OaU8 PushData[128] = {};
	OaU32 PushSize = 0;
	// Storage dtype the shader variant must match: 0 = FP32 (4-byte OaLoad/OaStore),
	// 1 = BF16/FP16 (2-byte). Derived from the operand tensors at record time (see
	// OaFnContext::Add) so the dispatch always picks the pipeline variant that matches
	// its actual buffers — never a global mode.
	OaU32 Dtype = 0;
	OaU32 GroupsX = 1;
	OaU32 GroupsY = 1;
	OaU32 GroupsZ = 1;
	OaVkBuffer IndirectBuffer;
	OaU64 IndirectOffset = 0;
	OaBool Indirect = false;
	OaQueueHint Queue = OaQueueHint::Compute;
	OaU32 NodeIndex = 0;  // Which mesh node runs this dispatch
};

// Per-buffer lifetime within the graph: [FirstAccess, LastAccess] node indices.
class OaBufferLifetime {
public:
	void* Buffer = nullptr;
	OaU64 Size = 0;
	OaU32 FirstAccess = 0;
	OaU32 LastAccess = 0;
};

// Group of buffers whose lifetimes don't overlap — can share one allocation.
class OaAliasGroup {
public:
	OaVec<OaBufferLifetime> Members;
	OaU64 RequiredSize = 0;
};

// Stats from a compiled graph.
class OaGraphStats {
public:
	OaU32 DispatchCount = 0;
	OaU32 BarrierCount = 0;        // total VkBufferMemoryBarrier2 entries emitted after PR-3 coalescing
	OaU32 DescriptorSetCount = 0;
	OaU64 TotalBufferBytes = 0;
	OaU64 PotentialAliasSavings = 0;

	// PR-3 G v3 — graph-optimizer observability.
	// Populated by Compile() (OaLlamaCppVulkanLessons.md §12).
	OaU32 GroupCount    = 0;       // # barrier-groups (=  # barrier emissions)
	OaU32 MaxGroupSize  = 0;       // largest group size
	OaU32 ReorderCount  = 0;       // # nodes pulled forward by the v2 look-ahead pass
	OaU32 NodeCountStat = 0;       // mirrors DispatchCount for AvgGroupSize math (= NodeCountStat / GroupCount)
	OaU32 WarBarrierCount = 0;     // execution-only read -> write dependencies
	OaU32 IndirectBarrierCount = 0;// dependencies consumed as indirect commands
};

// DAG-scheduled compute dispatcher with compile/replay semantics. This is the
// active graph used by the runtime and ML/autograd paths. It owns precise
// per-buffer synchronization, indirect-dispatch hazards and compiled command
// buffer cache identity.
//
// Phase 1 — One-shot execution:
//   graph.Add("RmsNorm", {x, out, w}, {Read, Write, Read}, &pc, sizeof(pc), groups);
//   graph.Add("Silu",    {out, act},  {Read, Write},       &pc2, sizeof(pc2), groups2);
//   graph.Execute(rt);
//
// Phase 2 — Compile & Replay:
//   graph.Add(...);  // build once
//   graph.Compile(rt);  // record into secondary command buffer once
//   for (int step = 0; step < 10000; ++step) {
//       UploadBatch(...);
//       graph.Replay(rt);  // zero CPU recording overhead
//   }
//   graph.Destroy(rt.Device);
//
// Phase 3 — Memory aliasing analysis:
//   auto groups = graph.ComputeAliasGroups();
//   // groups tells you which buffers can share memory
class OaComputeGraph {
public:
	// ─── Construction ─────────────────────────────────────────────────────
	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1);

	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY = 1, OaU32 InGroupsZ = 1);

	// Queue-annotated node — routes to async compute or transfer queue.
	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
		OaQueueHint InQueue
	);

	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
		OaQueueHint InQueue
	);

	// Device-targeted node — explicit mesh node assignment.
	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
		OaU32 InNodeIndex
	);

	void Add(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		OaU32 InGroupsX, OaU32 InGroupsY, OaU32 InGroupsZ,
		OaU32 InNodeIndex
	);

	// GPU-driven: workgroup counts read from InIndirectBuffer at InOffset.
	void AddIndirect(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer, OaU64 InOffset = 0);

	void AddIndirect(
		OaStringView InShader,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush, OaU32 InPushSize,
		const OaVkBuffer& InIndirectBuffer, OaU64 InOffset = 0);

	// ─── BLAS Integration ─────────────────────────────────────────────────
	// Optimized GEMM dispatch with automatic kernel selection.
	// Selects from: GemmCmSgBf16, GemmCmWgBf16, GemmTiled, or GemmNaive based on (M, N, K, device caps).
	// C = A @ B^T  (B stored transposed — OA convention)
	// NOTE: Requires InRt for device capability detection.
	void AddGemm(
		OaComputeEngine& InRt,
		OaVkBuffer InA,
		OaVkBuffer InB,
		OaVkBuffer OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);

	// ─── Phase 1: One-shot execution ──────────────────────────────────────
	[[nodiscard]] OaStatus Execute(OaComputeEngine& InRt);

	// Multi-queue execution: nodes with AsyncCompute hint go to the async queue.
	// Cross-queue dependencies are synchronized via timeline semaphores.
	[[nodiscard]] OaStatus ExecuteMultiQueue(OaComputeEngine& InRt);

	// Multi-device execution: dispatches routed to mesh nodes by NodeIndex.
	// Single-node mesh delegates to Execute(). Multi-node collects dispatches
	// per device, submits in parallel, cross-node deps via timeline semaphores.
	[[nodiscard]] OaStatus ExecuteDistributed(OaComputeEngine& InRt);

	// Auto-assign NodeIndex to each compute node based on scheduler routing.
	// Called by OaParallelStrategy after building the graph topology.
	void MapToMesh(OaDeviceMesh& InMesh, OaScheduler& InScheduler);

	// ─── Phase 2: Compile & Replay ────────────────────────────────────────
	// Record all dispatches + minimal barriers into a secondary command buffer.
	// Push constants, descriptor sets, and pipeline binds are baked in.
	[[nodiscard]] OaStatus Compile(OaComputeEngine& InRt);

	// Execute the pre-compiled secondary command buffer via vkCmdExecuteCommands.
	// The graph topology, buffers, and push constants must not have changed.
	[[nodiscard]] OaStatus Replay(OaComputeEngine& InRt);
	// Replay and expose the submitted GPU completion without forcing a host
	// wait. This is the graph edge used by image/video/capture consumers.
	[[nodiscard]] OaResult<OaCompletionToken> ReplayAsync(OaComputeEngine& InRt);
	[[nodiscard]] OaCompletionToken LastCompletion(const OaVkDevice& InDevice) const;

	// Wait for the most recent Replay() submission to complete on the GPU.
	// Only needed if Replay() was called without a following Sync().
	[[nodiscard]] OaStatus WaitForPendingReplay(const OaVkDevice& InDevice);

	// Record compiled secondary into a primary command buffer that is already recording.
	// Use to chain multiple compiled graphs before a single SubmitAndWait (same queue).
	[[nodiscard]] OaStatus RecordReplay(OaComputeEngine& InRt, void* InPrimaryCommandBuffer) const;

	[[nodiscard]] bool IsCompiled() const { return Compiled_; }

	// ─── Phase 3: Memory aliasing analysis ────────────────────────────────
	// Per-buffer first/last access within the graph.
	[[nodiscard]] OaVec<OaBufferLifetime> ComputeLifetimes() const;

	// Non-overlapping buffer groups that can share one VkDeviceMemory allocation.
	[[nodiscard]] OaVec<OaAliasGroup> ComputeAliasGroups() const;

	// ─── Queries ──────────────────────────────────────────────────────────
	[[nodiscard]] OaGraphStats GetStats() const;
	[[nodiscard]] OaU32 NodeCount() const { return static_cast<OaU32>(Nodes_.Size()); }
	[[nodiscard]] OaSpan<OaComputeNode> Nodes() { return {Nodes_.Data(), Nodes_.Size()}; }
	[[nodiscard]] OaSpan<const OaComputeNode> Nodes() const { return {Nodes_.Data(), Nodes_.Size()}; }

	// Stamp the just-recorded node's storage dtype (see OaComputeNode::Dtype). Called
	// right after an Add() by the record-time dtype derivation so a dispatch selects the
	// pipeline variant matching its operand tensors.
	void SetLastNodeDtype(OaU32 InDtype) { if (not Nodes_.Empty()) { Nodes_.Back().Dtype = InDtype; } }

	// ─── Lifecycle ────────────────────────────────────────────────────────
	// Invalidates compiled state only. Nodes are preserved.
	void Invalidate(const OaVkDevice& InDevice);

	// Clears nodes + compiled state.
	void Reset();
	void Reset(const OaVkDevice& InDevice);

	// Clears nodes but keeps the command pool + secondary CB. Compiled state
	// is invalidated (Compiled_ = false) so the next Compile() will reset the
	// CB and re-record. Descriptor pools from the previous compilation are
	// cleaned up in Compile(). Use this instead of Reset(device) when the
	// graph will be re-compiled with new nodes — avoids vkCreateCommandPool
	// + vkAllocateCommandBuffers per call.
	void ClearNodes();

	// Destroys all Vulkan objects (descriptor pools, command pool, secondary CB).
	void Destroy(const OaVkDevice& InDevice);

private:
	OaVec<OaComputeNode> Nodes_;

	// Compiled state (Phase 2 - CPU-driven)
	void* SecondaryPool_ = nullptr;
	void* SecondaryCb_ = nullptr;
	void* PrimaryPool_ = nullptr;      // dedicated pool for PrimaryCb_
	void* PrimaryCb_ = nullptr;        // pre-built primary wrapping SecondaryCb_
	OaVkTimelineSemaphore ReplayTimelineSem_;  // for cached replay submit+wait
	OaU64 ReplayTimelineValue_ = 0;
	OaVec<void*> DescriptorPools_;
	OaU32 QueueFamily_ = 0;
	bool Compiled_ = false;
	OaU64 LastCompileHash_ = 0;  // FNV-1a hash of node list; 0 = never compiled
	
	// Compile-time stats (PR-3 G v3). Populated by Compile().
	OaU32 BarrierCount_  = 0;   // total VkBufferMemoryBarrier2 entries emitted
	OaU32 GroupCount_    = 0;   // # barrier-coalescing groups (= # barrier emissions)
	OaU32 MaxGroupSize_  = 0;   // largest group size
	OaU32 ReorderCount_  = 0;   // # nodes pulled forward by v2 look-ahead reorder
	OaU32 WarBarrierCount_ = 0;
	OaU32 IndirectBarrierCount_ = 0;

	// FNV-1a hash over all node fields that affect the compiled secondary CB:
	// shader/dtype, VkBuffer handles, bindless indices, accesses, push constants,
	// direct/indirect dispatch state, queue hint, and target device node.
	// Used by Compile() to skip re-recording when the same op sequence repeats.
	[[nodiscard]] OaU64 ComputeNodeHash() const;
};
