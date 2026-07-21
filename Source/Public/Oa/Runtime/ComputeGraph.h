#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/DispatchDesc.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Runtime/Timestamp.h>

class OaEngine;
class OaVkDevice;
class OaDeviceMesh;
class OaScheduler;
class OaMatrix;

// Compute graph node — one dispatch with buffer access annotations.
// Push constant data is copied inline (max 128 bytes per Vulkan spec).
class OaComputeNode {
public:
	OaString Operation;
	OaVec<OaSemanticOperationId> SemanticOperations;
	OaU64 ImplementationId = 0;
	OaU64 OperationContractHash = 0;
	OaU64 ProblemContractHash = 0;
	OaU64 KernelContentHash = 0;
	OaString Shader;
	OaVec<OaVkBuffer> Buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> BufferOwners;
	OaVec<OaBufferAccess> Access;
	alignas(16) OaU8 PushData[128] = {};
	OaU32 PushSize = 0;
	// Storage dtype the shader variant must match: 0 = FP32 (4-byte OaLoad/OaStore),
	// 1 = BF16/FP16 (2-byte). Derived from the operand tensors at record time so
	// the dispatch always picks the pipeline variant that matches
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
	OaU32 ResourceOrder = 0;
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
	OaU32 BarrierCount = 0;        // total exact resource/global dependencies
	OaU32 DescriptorSetCount = 0;
	OaU64 TotalBufferBytes = 0;
	OaU64 PotentialAliasSavings = 0;

	OaU32 WarBarrierCount = 0;     // execution-only read -> write dependencies
	OaU32 IndirectBarrierCount = 0;// dependencies consumed as indirect commands
	OaU32 AliasBarrierCount = 0;   // global dependencies between buffer aliases
	OaU32 HostBarrierCount = 0;    // graph-final compute -> host visibility edges
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
	// Canonical append path. Compatibility overloads lower to this descriptor.
	void Add(const OaComputeDispatchDesc& InDesc);

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

	// ─── Phase 1: One-shot execution ──────────────────────────────────────
	[[nodiscard]] OaStatus Execute(OaEngine& InRt);

	// Multi-queue execution: nodes with AsyncCompute hint go to the async queue.
	// Cross-queue dependencies are synchronized via timeline semaphores.
	[[nodiscard]] OaStatus ExecuteMultiQueue(OaEngine& InRt);

	// Experimental multi-device route. Single-node delegates to Execute(); the
	// multi-node path is not a supported peer-transport/event contract and lacks
	// physical two-device acceptance evidence.
	[[nodiscard]] OaStatus ExecuteDistributed(OaEngine& InRt);

	// Experimental static NodeIndex assignment through OaScheduler.
	void MapToMesh(OaDeviceMesh& InMesh, OaScheduler& InScheduler);

	// ─── Phase 2: Compile & Replay ────────────────────────────────────────
	// Record all dispatches + minimal barriers into a secondary command buffer.
	// Push constants, descriptor sets, and pipeline binds are baked in.
	[[nodiscard]] OaStatus Compile(OaEngine& InRt);

	// Execute the pre-compiled secondary command buffer via vkCmdExecuteCommands.
	// The graph topology, buffers, and push constants must not have changed.
	[[nodiscard]] OaStatus Replay(OaEngine& InRt);
	// Replay and expose the submitted GPU completion without forcing a host
	// wait. This is the graph edge used by image/video/capture consumers.
	[[nodiscard]] OaResult<OaCompletionToken> ReplayAsync(OaEngine& InRt);
	[[nodiscard]] OaCompletionToken LastCompletion(const OaVkDevice& InDevice) const;

	// Embed one timestamp pair around the complete reusable primary program.
	// Timed replay is intentionally single-flight: callers must WaitForPendingReplay()
	// before submitting the same program again so query-pool reuse is well-defined.
	void SetReplayTimingEnabled(OaBool InEnabled) noexcept {
		if (ReplayTimingEnabled_ == InEnabled) return;
		ReplayTimingEnabled_ = InEnabled;
		Compiled_ = false;
	}
	[[nodiscard]] OaBool ReplayTimingEnabled() const noexcept {
		return ReplayTimingEnabled_;
	}
	[[nodiscard]] OaF64 LastReplayGpuMs() const noexcept {
		return LastReplayGpuMs_;
	}

	// Wait for the most recent Replay() submission to complete on the GPU.
	// Only needed if Replay() was called without a following Sync().
	[[nodiscard]] OaStatus WaitForPendingReplay(const OaVkDevice& InDevice);

	// Record compiled secondary into a primary command buffer that is already recording.
	// Use to chain multiple compiled graphs before a single SubmitAndWait (same queue).
	[[nodiscard]] OaStatus RecordReplay(OaEngine& InRt, void* InPrimaryCommandBuffer) const;

	[[nodiscard]] bool IsCompiled() const { return Compiled_; }
	[[nodiscard]] OaBool LastCompileReused() const noexcept { return LastCompileReused_; }

	// A standalone graph defaults to host-visible completion because callers may
	// wait and immediately read mapped output. Context-owned batches disable this
	// on their secondary graphs and emit one host edge at the actual batch
	// boundary instead of draining every intermediate graph.
	void SetHostReadbackRequired(OaBool InRequired) noexcept {
		if (HostReadbackRequired_ == InRequired) return;
		HostReadbackRequired_ = InRequired;
		Compiled_ = false;
	}
	[[nodiscard]] OaBool HostReadbackRequired() const noexcept {
		return HostReadbackRequired_;
	}

	// ─── Phase 3: Memory aliasing analysis ────────────────────────────────
	// Per-buffer first/last access within the graph.
	[[nodiscard]] OaVec<OaBufferLifetime> ComputeLifetimes() const;

	// Non-overlapping buffer groups that can share one VkDeviceMemory allocation.
	[[nodiscard]] OaVec<OaAliasGroup> ComputeAliasGroups() const;

	// Materialize alias backing for an explicit set of exclusively owned,
	// graph-internal transient matrices. External inputs/outputs are never
	// inferred as eligible. The matrices and all node owners are rebound to
	// distinct VkBuffer identities over shared allocations, which releases the
	// original physical allocations instead of merely adding an alias arena.
	[[nodiscard]] OaStatus MaterializeAliases(
		OaEngine& InRt, OaSpan<OaMatrix*> InEligible);
	// Transactional capture may retain the original graph, semantic bindings,
	// and stable slots until the aliased graph compiles. Those known references
	// are permitted explicitly; every unaccounted owner still fails closed.
	[[nodiscard]] OaStatus MaterializeAliases(
		OaEngine& InRt,
		OaSpan<OaMatrix*> InEligible,
		OaSpan<const OaU32> InPermittedAdditionalOwners);
	[[nodiscard]] OaU64 MaterializedAliasSavings() const noexcept {
		return MaterializedAliasSavings_;
	}

	// ─── Queries ──────────────────────────────────────────────────────────
	[[nodiscard]] OaGraphStats GetStats() const;
	// Deterministic, handle-free execution evidence. Resource ids are assigned
	// by first graph appearance, so identical captures produce identical JSON
	// even when Vulkan handles and bindless slots differ between processes.
	[[nodiscard]] OaString DebugReportJson(OaStringView InName = "") const;
	[[nodiscard]] OaU32 NodeCount() const { return static_cast<OaU32>(Nodes_.Size()); }
	[[nodiscard]] OaSpan<OaComputeNode> Nodes() { return {Nodes_.Data(), Nodes_.Size()}; }
	[[nodiscard]] OaSpan<const OaComputeNode> Nodes() const { return {Nodes_.Data(), Nodes_.Size()}; }

	// Copy only immutable dispatch descriptions. Compiled Vulkan state is never
	// shared between graphs. This is the capture primitive for reusable programs:
	// the context keeps its eager recording graph while the destination owns an
	// independently compiled plan and its strong buffer references.
	[[nodiscard]] OaStatus CopyNodesFrom(const OaComputeGraph& InSource);

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

	// Release the strong references retained by the compiled command buffer.
	// Call only after GPU completion. Cached VkBuffers remain alive in the engine
	// allocator and can be reacquired by the next identical graph, allowing its
	// node hash to match and the recorded secondary command buffer to replay.
	void ReleaseCompletedBufferOwners();

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
	OaVkTimestamp ReplayTimestamp_;
	OaU64 ReplayTimestampReadValue_ = 0;
	OaF64 LastReplayGpuMs_ = 0.0;
	OaBool ReplayTimingEnabled_ = false;
	OaVec<void*> DescriptorPools_;
	OaU32 QueueFamily_ = 0;
	bool Compiled_ = false;
	OaU64 LastCompileHash_ = 0;  // FNV-1a hash of node list; 0 = never compiled
	OaBool LastCompileReused_ = false;
	// A compiled command buffer embeds resource bindings. Retain matrix-owned
	// buffers after ClearNodes() so cache reuse cannot outlive those resources.
	OaVec<OaSharedPtr<OaVkBuffer>> CompiledBufferOwners_;

	// Compile-time synchronization stats.
	OaU32 BarrierCount_  = 0;   // total resource/global dependencies emitted
	OaU32 WarBarrierCount_ = 0;
	OaU32 IndirectBarrierCount_ = 0;
	OaU32 AliasBarrierCount_ = 0;
	OaBool HostReadbackRequired_ = true;

	// Allocator-backed transient arena. View deleters retain their backing owner,
	// so graph nodes and rebound matrices can safely outlive this graph object.
	OaVec<OaSharedPtr<OaVkBuffer>> AliasOwners_;
	OaU64 MaterializedAliasSavings_ = 0;
	void DestroyAliasArena();

	// FNV-1a hash over all node fields that affect the compiled secondary CB:
	// shader/dtype, VkBuffer handles, bindless indices, accesses, push constants,
	// direct/indirect dispatch state, queue hint, and target device node.
	// Used by Compile() to skip re-recording when the same op sequence repeats.
	[[nodiscard]] OaU64 ComputeNodeHash() const;
};
