#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>

#include <atomic>
#include <chrono>

#include <vulkan/vulkan.h>

static bool OaContextAccessReads(OaBufferAccess InAccess) {
	return InAccess == OaBufferAccess::Read or InAccess == OaBufferAccess::ReadWrite;
}

static bool OaContextAccessWrites(OaBufferAccess InAccess) {
	return InAccess == OaBufferAccess::Write or InAccess == OaBufferAccess::ReadWrite;
}

static OaContextBatchBufferState* OaContextFindBatchState(
	OaVec<OaContextBatchBufferState>& InStates, void* InBuffer)
{
	for (auto& state : InStates) {
		if (state.Buffer.Buffer == InBuffer) return &state;
	}
	return nullptr;
}

static void OaContextMergeBatchState(
	OaVec<OaContextBatchBufferState>& InStates,
	const OaVkBuffer& InBuffer,
	OaBool InRead,
	OaBool InWrite,
	OaBool InIndirectRead)
{
	if (not InBuffer.Buffer) return;
	auto* state = OaContextFindBatchState(InStates, InBuffer.Buffer);
	if (not state) {
		OaContextBatchBufferState value;
		value.Buffer = InBuffer;
		InStates.PushBack(value);
		state = &InStates.Back();
	}
	state->Read = state->Read or InRead;
	state->Write = state->Write or InWrite;
	state->IndirectRead = state->IndirectRead or InIndirectRead;
}

// Carry exact buffer access state across secondary CBs. A global memory barrier
// here serialized unrelated work and defeated the graph's per-buffer planner.
// The primary now emits only RAW/WAR/WAW dependencies for buffers touched by
// both the pending batch and the incoming graph.
static OaU32 OaContextBarrierBetweenSecondaryCbs(
	void* InPrimaryCb,
	OaVec<OaContextBatchBufferState>& InPending,
	const OaComputeGraph& InIncoming)
{
	OaVec<OaContextBatchBufferState> incoming;
	for (const auto& node : InIncoming.Nodes()) {
		for (OaU32 i = 0; i < static_cast<OaU32>(node.Buffers.Size()); ++i) {
			OaContextMergeBatchState(
				incoming, node.Buffers[i], OaContextAccessReads(node.Access[i]),
				OaContextAccessWrites(node.Access[i]), false);
		}
		if (node.Indirect) {
			OaContextMergeBatchState(
				incoming, node.IndirectBuffer, true, false, true);
		}
	}

	OaVec<VkBufferMemoryBarrier2> barriers;
	for (const auto& current : incoming) {
		auto* previous = OaContextFindBatchState(InPending, current.Buffer.Buffer);
		if (not previous) continue;
		const bool hazard =
			(previous->Write and (current.Read or current.Write))
			or (previous->Read and current.Write);
		if (not hazard) continue;

		VkBufferMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (previous->IndirectRead) {
			barrier.srcStageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		barrier.srcAccessMask = 0;
		if (previous->Read) barrier.srcAccessMask |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (previous->Write) barrier.srcAccessMask |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (previous->IndirectRead) barrier.srcAccessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		if (current.IndirectRead) {
			barrier.dstStageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		}
		barrier.dstAccessMask = 0;
		if (current.Read) barrier.dstAccessMask |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		if (current.Write) barrier.dstAccessMask |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		if (current.IndirectRead) barrier.dstAccessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = static_cast<VkBuffer>(current.Buffer.Buffer);
		barrier.offset = 0;
		barrier.size = VK_WHOLE_SIZE;
		barriers.PushBack(barrier);
	}

	if (not barriers.Empty()) {
		VkDependencyInfo dep{};
		dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.bufferMemoryBarrierCount = static_cast<OaU32>(barriers.Size());
		dep.pBufferMemoryBarriers = barriers.Data();
		vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(InPrimaryCb), &dep);
	}

	for (const auto& current : incoming) {
		auto* previous = OaContextFindBatchState(InPending, current.Buffer.Buffer);
		if (previous) {
			*previous = current;
		} else {
			InPending.PushBack(current);
		}
	}
	return static_cast<OaU32>(barriers.Size());
}

static OaF64 OaContextElapsedMs(
	const std::chrono::steady_clock::time_point& InBegin) noexcept
{
	return std::chrono::duration<OaF64, std::milli>(
		std::chrono::steady_clock::now() - InBegin).count();
}

static void OaContextMaybeLogGraph(const OaComputeGraph& InGraph) {
	static std::atomic<OaU32> sLoggedGraphs{0};

	const OaI64 requested = OaEnvFlag::GetInt("OA_LOG_CONTEXT_GRAPH", 0);
	if (requested <= 0) {
		return;
	}

	const OaU32 limit = static_cast<OaU32>(requested);
	const OaU32 index = sLoggedGraphs.fetch_add(1, std::memory_order_relaxed);
	if (index >= limit) {
		return;
	}

	const auto nodes = InGraph.Nodes();
	OA_LOG_INFO(OaLogComponent::Core,
		"OaContext graph #%u: %u node(s)", index + 1U,
		static_cast<OaU32>(nodes.Size()));
	for (OaU32 i = 0; i < static_cast<OaU32>(nodes.Size()); ++i) {
		const auto& node = nodes[i];
		OA_LOG_INFO(OaLogComponent::Core,
			"  [%02u] %-32s groups=(%u,%u,%u) buffers=%u push=%u",
			i,
			node.Shader.c_str(),
			node.GroupsX,
			node.GroupsY,
			node.GroupsZ,
			static_cast<OaU32>(node.Buffers.Size()),
			node.PushSize);
	}
}

OaStatus OaContext::Execute() {
	assert(Engine_ and "Engine is null");
	assert(Graph_ and "Graph is null");

	if (Graph_->NodeCount() == 0) {
		Executed_ = true;
		return OaStatus::Ok();
	}
	if (not Engine_->IsComputeBatchActive()) {
		ExecutionStats_ = OaContextExecutionStats{};
	}

	OaContextMaybeLogGraph(*Graph_);
	ExecutionStats_.NodeCount += Graph_->NodeCount();
	++ExecutionStats_.GraphCount;

	if (Engine_->IsComputeBatchActive()) {
		// The primary batch emits one host-visibility edge immediately before
		// submission. Intermediate secondary graphs remain device-only.
		Graph_->SetHostReadbackRequired(false);
		// A compute batch owns the primary command buffer. Record our secondary
		// CB into it (chains via vkCmdExecuteCommands) so the eventual
		// FlushComputeBatch submits both this context's work and any other
		// batch-recorded work in one queue submission, in order, sharing one
		// fence. Avoids racing two streams against the same buffers.
		if (!Graph_->IsCompiled()) {
			const auto compileBegin = std::chrono::steady_clock::now();
			auto compileStatus = Graph_->Compile(*Engine_);
			ExecutionStats_.CompileMs += OaContextElapsedMs(compileBegin);
			if (Graph_->LastCompileReused()) ++ExecutionStats_.CompileCacheHits;
			if (not compileStatus.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaContext::Execute compile failed: %s",
					compileStatus.GetMessage().c_str());
				// Recover: drop the un-compilable graph (not yet recorded into the
				// primary CB) so the failed node doesn't brick subsequent Executes.
				Graph_->Reset(Engine_->Device);
				return compileStatus;
			}
		}
		auto* stream = Engine_->ActiveComputeBatchStream();

		// Secondary-command-buffer boundaries provide no implicit dependency.
		// Insert the boundary barrier immediately before a following graph, not
		// unconditionally after every graph. The common one-graph training step
		// therefore has no useless trailing compute -> compute barrier.
		auto previousBatchStates = BatchBufferStates_;
		ExecutionStats_.BoundaryBarrierCount += OaContextBarrierBetweenSecondaryCbs(
			stream->CommandBuffer, BatchBufferStates_, *Graph_);
		const auto recordBegin = std::chrono::steady_clock::now();
		auto recordStatus = Graph_->RecordReplay(*Engine_, stream->CommandBuffer);
		ExecutionStats_.RecordMs += OaContextElapsedMs(recordBegin);
		if (not recordStatus.IsOk()) {
			BatchBufferStates_ = std::move(previousBatchStates);
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Execute RecordReplay failed: %s",
				recordStatus.GetMessage().c_str());
			return recordStatus;
		}

		// CRITICAL: the primary CB now references this secondary CB. Resetting
		// the graph here would invalidate the descriptor pools + secondary
		// CB before the batch submission consumes them — segfault inside the
		// driver. Park the graph in DeferredGraphs_ and hand the context
		// a fresh one so subsequent recording lands in a new graph; Sync()
		// drains and resets the deferred set after the batch completes.
		DeferredGraphs_.PushBack(Graph_);
		if (not ReusableGraphs_.Empty()) {
			Graph_ = ReusableGraphs_.Back();
			ReusableGraphs_.PopBack();
		} else {
			Graph_ = new OaComputeGraph();
		}
		Executed_ = true;
		return OaStatus::Ok();
	}

	// No batch active — compile, replay, submit, wait. Each Execute owns
	// its own command-buffer submission; safe to Reset immediately.
	Graph_->SetHostReadbackRequired(true);
	if (!Graph_->IsCompiled()) {
		const auto compileBegin = std::chrono::steady_clock::now();
		auto compileStatus = Graph_->Compile(*Engine_);
		ExecutionStats_.CompileMs += OaContextElapsedMs(compileBegin);
		if (Graph_->LastCompileReused()) ++ExecutionStats_.CompileCacheHits;
		if (not compileStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Execute compile failed: %s",
				compileStatus.GetMessage().c_str());
			// Recover the context: drop the un-compilable graph so the bad node
			// doesn't poison every subsequent Execute. Without this, a single
			// missing/failed kernel (e.g. an op with no shader) permanently
			// bricks the shared default context — later ops silently read zeros.
			Graph_->Reset(Engine_->Device);
			return compileStatus;
		}
	}
	const auto submitBegin = std::chrono::steady_clock::now();
	auto status = Graph_->Replay(*Engine_);
	ExecutionStats_.SubmitMs += OaContextElapsedMs(submitBegin);
	if (not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Execute failed: %s",
			status.GetMessage().c_str());
		Graph_->Reset(Engine_->Device);
		return status;
	}
	// Wait for the non-blocking Replay() submission to complete before
	// clearing nodes. The next Compile() will free descriptor pools that
	// the GPU may still be referencing — must ensure GPU is done first.
	const auto waitBegin = std::chrono::steady_clock::now();
	OA_RETURN_IF_ERROR(Graph_->WaitForPendingReplay(Engine_->Device));
	ExecutionStats_.WaitMs += OaContextElapsedMs(waitBegin);
	ExecutionStats_.HostBarrierCount = 1;
	Graph_->ReleaseCompletedBufferOwners();
	// Keep the command pool + secondary CB for reuse — ClearNodes() just
	// clears the node list and marks not-compiled. The next Compile() will
	// reset the CB and re-record, avoiding vkCreateCommandPool +
	// vkAllocateCommandBuffers per call (~0.05ms savings).
	Graph_->ClearNodes();
	Executed_ = true;
	return OaStatus::Ok();
}

OaStatus OaContext::ExecuteAsync(OaGpuTimer* InTimer) {
	auto completion = ExecuteAsyncToken(InTimer);
	return completion.IsOk() ? OaStatus::Ok() : completion.GetStatus();
}

OaResult<OaCompletionToken> OaContext::ExecuteAsyncToken(OaGpuTimer* InTimer) {
	OA_RETURN_IF_ERROR(ExecuteInAsyncBatch(InTimer));
	return FlushAsyncBatchToken();
}

OaStatus OaContext::BeginAsyncBatch() {
	assert(Engine_ and "Engine is null");
	if (Engine_->IsComputeBatchActive()) {
		return OaStatus::Ok();
	}
	ExecutionStats_ = OaContextExecutionStats{};
	return Engine_->BeginComputeBatch();
}

OaStatus OaContext::ExecuteInAsyncBatch(OaGpuTimer* InTimer) {
	assert(Engine_ and "Engine is null");
	assert(Graph_ and "Graph is null");

	if (Graph_->NodeCount() == 0) {
		Executed_ = true;
		return OaStatus::Ok();
	}

	OA_RETURN_IF_ERROR(BeginAsyncBatch());

	if (InTimer) {
		InTimer->Begin(*Engine_);
	}
	auto executeStatus = Execute();
	if (InTimer) {
		InTimer->End(*Engine_);
	}
	if (not executeStatus.IsOk()) {
		return executeStatus;
	}

	return OaStatus::Ok();
}

OaStatus OaContext::FlushAsyncBatch() {
	assert(Engine_ and "Engine is null");
	if (not Engine_->IsComputeBatchActive()) {
		return OaStatus::Ok();
	}
	const auto submitBegin = std::chrono::steady_clock::now();
	auto status = Engine_->FlushComputeBatch();
	ExecutionStats_.SubmitMs += OaContextElapsedMs(submitBegin);
	ExecutionStats_.HostBarrierCount = 1;
	BatchBufferStates_.Clear();
	return status;
}

OaResult<OaCompletionToken> OaContext::FlushAsyncBatchToken() {
	assert(Engine_ and "Engine is null");
	if (not Engine_->IsComputeBatchActive()) {
		return Engine_->LastComputeBatchCompletion();
	}
	const auto submitBegin = std::chrono::steady_clock::now();
	const auto status = Engine_->FlushComputeBatch();
	ExecutionStats_.SubmitMs += OaContextElapsedMs(submitBegin);
	ExecutionStats_.HostBarrierCount = 1;
	BatchBufferStates_.Clear();
	OA_RETURN_IF_ERROR(status);
	return Engine_->LastComputeBatchCompletion();
}

OaBool OaContext::IsAsyncBatchActive() const noexcept {
	return Engine_ ? Engine_->IsComputeBatchActive() : false;
}

OaStatus OaContext::Sync() {
	assert(Engine_ and "Engine is null");

	// Wait for any pending non-batch Replay() submission. Replay() now
	// submits without blocking (fire-and-forget) — same-queue ordering
	// ensures GPU executes in submission order. We wait here before
	// returning to the caller.
	if (Graph_) {
		OA_RETURN_IF_ERROR(Graph_->WaitForPendingReplay(Engine_->Device));
	}

	// If the caller recorded several Execute() calls into one context-owned
	// async batch, submit that primary CB before waiting.
	auto flushStatus = FlushAsyncBatch();
	if (not flushStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Sync flush failed: %s",
			flushStatus.GetMessage().c_str());
		return flushStatus;
	}

	// Wait for GPU completion (flush current batch and sync device)
	const auto waitBegin = std::chrono::steady_clock::now();
	auto status = Engine_->SyncCurrentBatch();
	ExecutionStats_.WaitMs += OaContextElapsedMs(waitBegin);

	// GPU is done with the batch. Preserve compiled secondary CBs and their
	// hashes so a structurally identical next step can skip command recording.
	// Restore the graphs in execution order: G0 becomes active, then G1..Gn are
	// popped from ReusableGraphs_ as subsequent Execute calls park each graph.
	if (not DeferredGraphs_.Empty()) {
		for (auto* graph : DeferredGraphs_) {
			graph->ClearNodes();
			graph->ReleaseCompletedBufferOwners();
		}
		if (Graph_->NodeCount() == 0) {
			Graph_->Destroy(Engine_->Device);
			delete Graph_;
			Graph_ = DeferredGraphs_[0];
			for (OaUsize i = DeferredGraphs_.Size(); i > 1; --i) {
				ReusableGraphs_.PushBack(DeferredGraphs_[i - 1]);
			}
		} else {
			for (OaUsize i = DeferredGraphs_.Size(); i > 0; --i) {
				ReusableGraphs_.PushBack(DeferredGraphs_[i - 1]);
			}
		}
	}
	DeferredGraphs_.Clear();
	BatchBufferStates_.Clear();

	// Step 3b.4: drain a pending RecordPresent. By construction we're now
	// past the compute fence wait, so any buffer that RecordBlit pointed at
	// is GPU-coherent and safe to vkCmdCopyBufferToImage from.
	FlushPendingPresent();

	if (not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Sync failed: %s",
			status.GetMessage().c_str());
		return status;
	}
	if (OaEnvFlag::IsSet("OA_LOG_RUNTIME_PHASES")) {
		OA_LOG_INFO(OaLogComponent::Core,
			"Runtime phases: nodes=%u graphs=%u cache_hits=%u boundary_barriers=%u "
			"host_barriers=%u compile=%.3f ms record=%.3f ms submit=%.3f ms wait=%.3f ms",
			ExecutionStats_.NodeCount, ExecutionStats_.GraphCount,
			ExecutionStats_.CompileCacheHits, ExecutionStats_.BoundaryBarrierCount,
			ExecutionStats_.HostBarrierCount, ExecutionStats_.CompileMs,
			ExecutionStats_.RecordMs, ExecutionStats_.SubmitMs, ExecutionStats_.WaitMs);
	}

	return OaStatus::Ok();
}

OaU32 OaContext::MaxAsyncSubmissions() const noexcept {
	return Engine_ ? Engine_->ComputeBatchRingSize() : 1U;
}

void OaContext::Clear() {
	assert(Graph_ and "Graph is null");
	
	// Clear recorded operations for reuse. Keep the command pool + secondary
	// CB so the next Compile() avoids vkCreateCommandPool + vkAllocateCommandBuffers.
	if (Engine_) {
		Graph_->ClearNodes();
	} else {
		Graph_->Reset();
	}
	Executed_ = false;
	if (not Engine_ or not Engine_->IsComputeBatchActive()) {
		BatchBufferStates_.Clear();
		ExecutionStats_ = OaContextExecutionStats{};
	}
}

OaU32 OaContext::NodeCount() const noexcept {
	assert(Graph_ and "Graph is null");
	return Graph_->NodeCount();
}
