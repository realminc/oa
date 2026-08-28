#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/bufferAccess.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/dispatchDesc.h>
#include <oa/runtime/sync.h>
#include <oa/runtime/timestamp.h>

namespace oavk { class Device; }
namespace oa {

class Engine;
class Matrix;

// Private compute graph node — one dispatch with buffer access annotations.
// Push constant data is copied inline (max 128 bytes per vulkan spec).
class ComputeNode {
public:
	oa::String operation;
	oa::Vector<oa::U32> semanticOps;
	oa::U64 implementationId = 0;
	oa::U64 opContractHash = 0;
	oa::U64 problemContractHash = 0;
	oa::U64 kernelContentHash = 0;
	oa::KernelSelectionKind kernelSelection = oa::KernelSelectionKind::Unspecified;
	oa::String shader;
	oa::Vector<oavk::Buffer> buffers;
	oa::Vector<oa::SharedPtr<oavk::Buffer>> bufferOwners;
	oa::Vector<oa::BufferAccess> access;
	alignas(16) oa::U8 pushData[128] = {};
	oa::U32 pushSize = 0;
	// Storage dtype the shader variant must match: 0 = FP32 (4-byte oaLoad/oaStore),
	// 1 = BF16/FP16 (2-byte). Derived from the operand tensors at record time so
	// the dispatch always picks the pipeline variant that matches
	// its actual buffers — never a global mode.
	oa::U32 dtype = 0;
	oa::U32 groupsX = 1;
	oa::U32 groupsY = 1;
	oa::U32 groupsZ = 1;
	oavk::Buffer indirectBuffer;
	oa::U64 indirectOffset = 0;
	oa::Bool indirect = false;
	oa::QueueHint queue = oa::QueueHint::Compute;
};

// Per-buffer lifetime within the graph: [firstAccess, lastAccess] node indices.
class BufferLifetime {
public:
	void* buffer = nullptr;
	oa::U64 size = 0;
	oa::U32 firstAccess = 0;
	oa::U32 lastAccess = 0;
	oa::U32 resourceOrder = 0;
};

// Group of buffers whose lifetimes don't overlap — can share one allocation.
class AliasGroup {
public:
	oa::Vector<oa::BufferLifetime> members;
	oa::U64 requiredSize = 0;
};

// DAG-scheduled compute dispatcher with compile/replay semantics. This is the
// active graph used by the runtime and ML/autograd paths. It owns precise
// per-buffer synchronization, indirect-dispatch hazards and compiled command
// buffer cache identity.
//
// phase 1 — One-shot execution:
//   graph.add("RmsNorm", {x, out, w}, {Read, Write, Read}, &pc, sizeof(pc), groups);
//   graph.add("Silu",    {out, act},  {Read, Write},       &pc2, sizeof(pc2), groups2);
//   graph.execute(rt);
//
// phase 2 — compile & replay:
//   graph.add(...);  // build once
//   graph.compile(rt);  // record into secondary command buffer once
//   for (int step = 0; step < 10000; ++step) {
//       uploadBatch(...);
//       graph.replay(rt);  // zero CPU recording overhead
//   }
//   graph.waitForPendingReplay(rt);  // explicit completion before scope exit
//
// phase 3 — memory aliasing analysis:
//   auto groups = graph.computeAliasGroups();
//   // groups tells you which buffers can share memory
class ExecutableGraph {
public:
	ExecutableGraph() = default;
	~ExecutableGraph();
	ExecutableGraph(const ExecutableGraph&) = delete;
	ExecutableGraph& operator=(const ExecutableGraph&) = delete;
	ExecutableGraph(ExecutableGraph&&) = delete;
	ExecutableGraph& operator=(ExecutableGraph&&) = delete;

	// ─── Construction ─────────────────────────────────────────────────────
	// Canonical append path. Compatibility overloads lower to this descriptor.
	void add(const oa::ComputeDispatchDesc& inDesc);

	void add(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		oa::U32 inGroupsX, oa::U32 inGroupsY = 1, oa::U32 inGroupsZ = 1);

	void add(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		oa::U32 inGroupsX, oa::U32 inGroupsY = 1, oa::U32 inGroupsZ = 1);

	// queue-annotated node — routes to async compute or transfer queue.
	void add(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ,
		oa::QueueHint inQueue
	);

	void add(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		oa::U32 inGroupsX, oa::U32 inGroupsY, oa::U32 inGroupsZ,
		oa::QueueHint inQueue
	);

	// GPU-driven: workgroup counts read from inIndirectBuffer at inOffset.
	// The buffer must have indirect-dispatch usage and contain one aligned
	// three-u32 command within its logical range.
	void addIndirect(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset = 0);

	void addIndirect(
		oa::StringView inShader,
		oa::Span<oavk::Buffer> inBuffers,
		oa::Span<oa::SharedPtr<oavk::Buffer>> inBufferOwners,
		oa::Span<oa::BufferAccess> inAccess,
		const void* inPush, oa::U32 inPushSize,
		const oavk::Buffer& inIndirectBuffer, oa::U64 inOffset = 0);

	// ─── phase 1: One-shot execution ──────────────────────────────────────
	[[nodiscard]] oa::Status execute(oa::Engine& inRt);

	// Multi-queue execution: nodes with AsyncCompute hint go to the async queue.
	// Cross-queue dependencies are synchronized via timeline semaphores.
	[[nodiscard]] oa::Status executeMultiQueue(oa::Engine& inRt);

	// ─── phase 2: compile & replay ────────────────────────────────────────
	// Record all dispatches + minimal barriers into a secondary command buffer.
	// Push constants, descriptor sets, and pipeline binds are baked in.
	[[nodiscard]] oa::Status compile(oa::Engine& inRt);

	// execute the pre-compiled secondary command buffer via vkCmdExecuteCommands.
	// The graph topology, buffers, and push constants must not have changed.
	[[nodiscard]] oa::Status replay(oa::Engine& inRt);
	// replay and expose the submitted GPU completion without forcing a host
	// wait. This is the graph edge used by image/video/capture consumers.
	[[nodiscard]] oa::Result<oa::Event> replayAsync(oa::Engine& inRt);
	[[nodiscard]] oa::Event lastCompletion(const oa::Engine& inEngine) const;

	// Embed one timestamp pair around the complete reusable primary program.
	// Timed replay is intentionally single-flight: callers must waitForPendingReplay()
	// before submitting the same program again so query-pool reuse is well-defined.
	void setReplayTimingEnabled(oa::Bool inEnabled) noexcept {
		if (replayTimingEnabled_ == inEnabled) return;
		replayTimingEnabled_ = inEnabled;
		compiled_ = false;
	}
	[[nodiscard]] oa::Bool replayTimingEnabled() const noexcept {
		return replayTimingEnabled_;
	}
	[[nodiscard]] oa::F64 lastReplayGpuMs() const noexcept {
		return lastReplayGpuMs_;
	}

	// wait for the most recent replay() submission to complete on the GPU.
	// Only needed if replay() was called without a following sync().
	[[nodiscard]] oa::Status waitForPendingReplay(const oa::Engine& inEngine);

	// Record compiled secondary into a primary command buffer that is already recording.
	// Use to chain multiple compiled graphs before a single submitAndWait (same queue).
	[[nodiscard]] oa::Status recordReplay(oa::Engine& inRt, void* inPrimaryCommandBuffer) const;

	[[nodiscard]] bool isCompiled() const { return compiled_; }
	[[nodiscard]] oa::Bool lastCompileReused() const noexcept { return lastCompileReused_; }

	// A standalone graph defaults to host-visible completion because callers may
	// wait and immediately read mapped output. context-owned batches disable this
	// on their secondary graphs and emit one host edge at the actual batch
	// boundary instead of draining every intermediate graph.
	void setHostReadbackRequired(oa::Bool inRequired) noexcept {
		if (hostReadbackRequired_ == inRequired) return;
		hostReadbackRequired_ = inRequired;
		compiled_ = false;
	}
	[[nodiscard]] oa::Bool hostReadbackRequired() const noexcept {
		return hostReadbackRequired_;
	}

	// ─── phase 3: memory aliasing analysis ────────────────────────────────
	// Per-buffer first/last access within the graph.
	[[nodiscard]] oa::Vector<oa::BufferLifetime> computeLifetimes() const;

	// Non-overlapping buffer groups that can share one VkDeviceMemory allocation.
	[[nodiscard]] oa::Vector<oa::AliasGroup> computeAliasGroups() const;

	// Materialize alias backing for an explicit set of exclusively owned,
	// graph-internal transient matrices. external inputs/outputs are never
	// inferred as eligible. The matrices and all node owners are rebound to
	// distinct VkBuffer identities over shared allocations, which releases the
	// original physical allocations instead of merely adding an alias arena.
	[[nodiscard]] oa::Status materializeAliases(
		oa::Engine& inRt, oa::Span<oa::Matrix*> inEligible);
	// Transactional capture may retain the original graph, semantic bindings,
	// and stable slots until the aliased graph compiles. Those known references
	// are permitted explicitly; every unaccounted owner still fails closed.
	[[nodiscard]] oa::Status materializeAliases(
		oa::Engine& inRt,
		oa::Span<oa::Matrix*> inEligible,
		oa::Span<const oa::U32> inPermittedAdditionalOwners);
	[[nodiscard]] oa::U64 materializedAliasSavings() const noexcept {
		return materializedAliasSavings_;
	}

	// ─── Queries ──────────────────────────────────────────────────────────
	[[nodiscard]] oa::GraphStats getStats() const;
	// Deterministic, handle-free execution evidence. resource ids are assigned
	// by first graph appearance, so identical captures produce identical JSON
	// even when vulkan handles and bindless slots differ between processes.
	[[nodiscard]] oa::String debugReportJson(oa::StringView inName = "") const;
	[[nodiscard]] oa::U32 nodeCount() const { return static_cast<oa::U32>(nodes_.size()); }
	[[nodiscard]] oa::Span<oa::ComputeNode> nodes() { return {nodes_.data(), nodes_.size()}; }
	[[nodiscard]] oa::Span<const oa::ComputeNode> nodes() const { return {nodes_.data(), nodes_.size()}; }

	// Copy only immutable dispatch descriptions. Compiled vulkan state is never
	// shared between graphs. This is the capture primitive for reusable programs:
	// the context keeps its eager recording graph while the destination owns an
	// independently compiled plan and its strong buffer references.
	[[nodiscard]] oa::Status copyNodesFrom(const oa::ExecutableGraph& inSource);

	// ─── Lifecycle ────────────────────────────────────────────────────────
	// Invalidates compiled state only. nodes are preserved.
	void invalidate(const oa::Engine& inEngine);

	// Clears nodes + compiled state.
	void reset();
	void reset(const oa::Engine& inEngine);

	// Clears nodes but keeps the command pool + secondary CB. Compiled state
	// is invalidated (compiled_ = false) so the next compile() will reset the
	// CB and re-record. Descriptor pools from the previous compilation are
	// cleaned up in compile(). Use this instead of reset(device) when the
	// graph will be re-compiled with new nodes — avoids vkCreateCommandPool
	// + vkAllocateCommandBuffers per call.
	void clearNodes();

	// release the strong references retained by the compiled command buffer.
	// call only after GPU completion. Cached VkBuffers remain alive in the engine
	// allocator and can be reacquired by the next identical graph, allowing its
	// node hash to match and the recorded secondary command buffer to replay.
	void releaseCompletedBufferOwners();

private:
	[[nodiscard]] oa::Status bindEngine_(oa::Engine& inEngine);
	[[nodiscard]] bool hasDeviceState_() const noexcept;
	void release_() noexcept;
	void swapState_(oa::ExecutableGraph& inOther) noexcept;
	[[nodiscard]] oa::Event lastCompletionDevice_(const oavk::Device& inDevice) const;
	[[nodiscard]] oa::Status waitForPendingReplayDevice_(const oavk::Device& inDevice);
	void invalidateDevice_(const oavk::Device& inDevice);
	void resetDevice_(const oavk::Device& inDevice);
	void destroyDevice_(const oavk::Device& inDevice);

	oa::Vector<oa::ComputeNode> nodes_;
	oa::Engine* owner_ = nullptr;

	// Compiled state (phase 2 - CPU-driven)
	void* secondaryPool_ = nullptr;
	void* secondaryCb_ = nullptr;
	void* primaryPool_ = nullptr;      // dedicated pool for primaryCb_
	void* primaryCb_ = nullptr;        // pre-built primary wrapping secondaryCb_
	oavk::TimelineSemaphore replayTimelineSem_;  // for cached replay submit+wait
	oa::U64 replayTimelineValue_ = 0;
	oavk::Timestamp replayTimestamp_;
	oa::U64 replayTimestampReadValue_ = 0;
	oa::F64 lastReplayGpuMs_ = 0.0;
	oa::Bool replayTimingEnabled_ = false;
	oa::Vector<void*> descriptorPools_;
	oa::U32 queueFamily_ = 0;
	bool compiled_ = false;
	oa::U64 lastCompileHash_ = 0;  // FNV-1a hash of node list; 0 = never compiled
	oa::Bool lastCompileReused_ = false;
	// A compiled command buffer embeds resource bindings. Retain matrix-owned
	// buffers after clearNodes() so cache reuse cannot outlive those resources.
	oa::Vector<oa::SharedPtr<oavk::Buffer>> compiledBufferOwners_;

	// compile-time synchronization stats.
	oa::U32 barrierCount_  = 0;   // total resource/global dependencies emitted
	oa::U32 warBarrierCount_ = 0;
	oa::U32 indirectBarrierCount_ = 0;
	oa::U32 aliasBarrierCount_ = 0;
	oa::Bool hostReadbackRequired_ = true;

	// allocator-backed transient arena. view deleters retain their backing owner,
	// so graph nodes and rebound matrices can safely outlive this graph object.
	oa::Vector<oa::SharedPtr<oavk::Buffer>> aliasOwners_;
	oa::U64 materializedAliasSavings_ = 0;
	void destroyAliasArena();

	// FNV-1a hash over all node fields that affect the compiled secondary CB:
	// shader/dtype, VkBuffer handles, bindless indices, accesses, push constants,
	// direct/indirect dispatch state and queue hint.
	// Used by compile() to skip re-recording when the same op sequence repeats.
	[[nodiscard]] oa::U64 computeNodeHash() const;
};

} // namespace oa
