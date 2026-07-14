#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>

#include <atomic>

#include <vulkan/vulkan.h>

// Insert a compute→compute pipeline barrier on the active primary CB. Used
// between consecutive RecordReplay secondary CBs so the next graph's reads
// see the previous graph's writes — without this, the optimizer's gradient
// reads race the backward's accumulation writes and training diverges.
static void OaContextBarrierBetweenSecondaryCbs(void* InPrimaryCb) {
	VkMemoryBarrier2 bar{};
	bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
		| VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
	bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	bar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
		| VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

	VkDependencyInfo dep{};
	dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dep.memoryBarrierCount = 1;
	dep.pMemoryBarriers = &bar;
	vkCmdPipelineBarrier2(static_cast<VkCommandBuffer>(InPrimaryCb), &dep);
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
	assert(Runtime_ and "Runtime is null");
	assert(Graph_ and "Graph is null");

	if (Graph_->NodeCount() == 0) {
		Executed_ = true;
		return OaStatus::Ok();
	}

	OaContextMaybeLogGraph(*Graph_);

	if (Runtime_->IsComputeBatchActive()) {
		// A compute batch owns the primary command buffer. Record our secondary
		// CB into it (chains via vkCmdExecuteCommands) so the eventual
		// FlushComputeBatch submits both this context's work and any other
		// batch-recorded work in one queue submission, in order, sharing one
		// fence. Avoids racing two streams against the same buffers.
		if (!Graph_->IsCompiled()) {
			auto compileStatus = Graph_->Compile(*Runtime_);
			if (not compileStatus.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaContext::Execute compile failed: %s",
					compileStatus.GetMessage().c_str());
				// Recover: drop the un-compilable graph (not yet recorded into the
				// primary CB) so the failed node doesn't brick subsequent Executes.
				Graph_->Reset(Runtime_->Device);
				return compileStatus;
			}
		}
		auto* stream = Runtime_->ActiveComputeBatchStream();

		// Secondary-command-buffer boundaries provide no implicit dependency.
		// Insert the boundary barrier immediately before a following graph, not
		// unconditionally after every graph. The common one-graph training step
		// therefore has no useless trailing compute -> compute barrier.
		if (!DeferredGraphs_.Empty()) {
			OaContextBarrierBetweenSecondaryCbs(stream->CommandBuffer);
		}
		auto recordStatus = Graph_->RecordReplay(*Runtime_, stream->CommandBuffer);
		if (not recordStatus.IsOk()) {
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
		Graph_ = new OaComputeGraph();
		Executed_ = true;
		return OaStatus::Ok();
	}

	// No batch active — compile, replay, submit, wait. Each Execute owns
	// its own command-buffer submission; safe to Reset immediately.
	if (!Graph_->IsCompiled()) {
		auto compileStatus = Graph_->Compile(*Runtime_);
		if (not compileStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Execute compile failed: %s",
				compileStatus.GetMessage().c_str());
			// Recover the context: drop the un-compilable graph so the bad node
			// doesn't poison every subsequent Execute. Without this, a single
			// missing/failed kernel (e.g. an op with no shader) permanently
			// bricks the shared default context — later ops silently read zeros.
			Graph_->Reset(Runtime_->Device);
			return compileStatus;
		}
	}
	auto status = Graph_->Replay(*Runtime_);
	if (not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Execute failed: %s",
			status.GetMessage().c_str());
		Graph_->Reset(Runtime_->Device);
		return status;
	}
	// Wait for the non-blocking Replay() submission to complete before
	// clearing nodes. The next Compile() will free descriptor pools that
	// the GPU may still be referencing — must ensure GPU is done first.
	OA_RETURN_IF_ERROR(Graph_->WaitForPendingReplay(Runtime_->Device));
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
	assert(Runtime_ and "Runtime is null");
	if (Runtime_->IsComputeBatchActive()) {
		return OaStatus::Ok();
	}
	return Runtime_->BeginComputeBatch();
}

OaStatus OaContext::ExecuteInAsyncBatch(OaGpuTimer* InTimer) {
	assert(Runtime_ and "Runtime is null");
	assert(Graph_ and "Graph is null");

	if (Graph_->NodeCount() == 0) {
		Executed_ = true;
		return OaStatus::Ok();
	}

	OA_RETURN_IF_ERROR(BeginAsyncBatch());

	if (InTimer) {
		InTimer->Begin(*Runtime_);
	}
	auto executeStatus = Execute();
	if (InTimer) {
		InTimer->End(*Runtime_);
	}
	if (not executeStatus.IsOk()) {
		return executeStatus;
	}

	return OaStatus::Ok();
}

OaStatus OaContext::FlushAsyncBatch() {
	assert(Runtime_ and "Runtime is null");
	if (not Runtime_->IsComputeBatchActive()) {
		return OaStatus::Ok();
	}
	return Runtime_->FlushComputeBatch();
}

OaResult<OaCompletionToken> OaContext::FlushAsyncBatchToken() {
	assert(Runtime_ and "Runtime is null");
	if (not Runtime_->IsComputeBatchActive()) {
		return Runtime_->LastComputeBatchCompletion();
	}
	OA_RETURN_IF_ERROR(Runtime_->FlushComputeBatch());
	return Runtime_->LastComputeBatchCompletion();
}

OaBool OaContext::IsAsyncBatchActive() const noexcept {
	return Runtime_ ? Runtime_->IsComputeBatchActive() : false;
}

OaStatus OaContext::Sync() {
	assert(Runtime_ and "Runtime is null");

	// Wait for any pending non-batch Replay() submission. Replay() now
	// submits without blocking (fire-and-forget) — same-queue ordering
	// ensures GPU executes in submission order. We wait here before
	// returning to the caller.
	if (Graph_) {
		OA_RETURN_IF_ERROR(Graph_->WaitForPendingReplay(Runtime_->Device));
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
	auto status = Runtime_->SyncCurrentBatch();

	// GPU is done with the batch. The deferred graphs' secondary CBs and
	// descriptor pools are no longer referenced — Reset + delete safely.
	for (auto* graph : DeferredGraphs_) {
		graph->Reset(Runtime_->Device);
		delete graph;
	}
	DeferredGraphs_.Clear();

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

	return OaStatus::Ok();
}

OaU32 OaContext::MaxAsyncSubmissions() const noexcept {
	return Runtime_ ? Runtime_->ComputeBatchRingSize() : 1U;
}

void OaContext::Clear() {
	assert(Graph_ and "Graph is null");
	
	// Clear recorded operations for reuse. Keep the command pool + secondary
	// CB so the next Compile() avoids vkCreateCommandPool + vkAllocateCommandBuffers.
	if (Runtime_) {
		Graph_->ClearNodes();
	} else {
		Graph_->Reset();
	}
	Executed_ = false;
}

OaU32 OaContext::NodeCount() const noexcept {
	assert(Graph_ and "Graph is null");
	return Graph_->NodeCount();
}
