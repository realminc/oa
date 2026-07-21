#include <Oa/Runtime/Context.h>
#include "ContextImpl.h"
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>

#include <atomic>
#include <chrono>

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
	assert(Impl_->Engine_ and "Engine is null");
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext recording transaction aborted: %s",
			recordingStatus.GetMessage().c_str());
		return recordingStatus;
	}
	auto* graph = Impl_->Execution_.Graph();
	assert(graph and "Graph is null");
	const auto loweringStatus = Impl_->Execution_.ValidateLowering();
	if (not loweringStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext semantic lowering validation failed: %s",
			loweringStatus.GetMessage().c_str());
		Impl_->Execution_.DiscardActiveRecording();
		return loweringStatus;
	}

	if (graph->NodeCount() == 0) {
		Impl_->Execution_.MarkExecuted();
		return OaStatus::Ok();
	}
	if (not Impl_->Execution_.IsBatchActive()) {
		Impl_->Execution_.ResetStats();
	}

	OaContextMaybeLogGraph(*graph);
	Impl_->Execution_.Stats().NodeCount += graph->NodeCount();
	++Impl_->Execution_.Stats().GraphCount;

	if (Impl_->Execution_.IsBatchActive()) {
		// The primary batch emits one host-visibility edge immediately before
		// submission. Intermediate secondary graphs remain device-only.
		graph->SetHostReadbackRequired(false);
		// A compute batch owns the primary command buffer. Record our secondary
		// CB into it (chains via vkCmdExecuteCommands) so the eventual
		// SubmitBatch submits only this execution session's recorded work in one
		// queue submission. No engine-global ambient batch can absorb work from a
		// different context.
		if (!graph->IsCompiled()) {
			const auto compileBegin = std::chrono::steady_clock::now();
			auto compileStatus = graph->Compile(*Impl_->Engine_);
			Impl_->Execution_.Stats().CompileMs += OaContextElapsedMs(compileBegin);
			if (graph->LastCompileReused()) ++Impl_->Execution_.Stats().CompileCacheHits;
			if (not compileStatus.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaContext::Execute compile failed: %s",
					compileStatus.GetMessage().c_str());
				// Recover: drop the un-compilable graph (not yet recorded into the
				// primary CB) so the failed node doesn't brick subsequent Executes.
				Impl_->Execution_.DiscardActiveRecording();
				return compileStatus;
			}
		}
		auto* stream = Impl_->Execution_.ActiveBatchStream();

		// Secondary-command-buffer boundaries provide no implicit dependency.
		// Insert the boundary barrier immediately before a following graph, not
		// unconditionally after every graph. The common one-graph training step
		// therefore has no useless trailing compute -> compute barrier.
		const auto recordBegin = std::chrono::steady_clock::now();
		auto recordResult = Impl_->Execution_.RecordActiveGraphInBatch(
			stream->CommandBuffer);
		Impl_->Execution_.Stats().RecordMs += OaContextElapsedMs(recordBegin);
		if (not recordResult.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Execute RecordReplay failed: %s",
				recordResult.GetStatus().GetMessage().c_str());
			return recordResult.GetStatus();
		}
		Impl_->Execution_.Stats().BoundaryBarrierCount += recordResult.GetValue();

		// CRITICAL: the primary CB now references this secondary CB. Resetting
		// the graph here would invalidate the descriptor pools + secondary
		// CB before the batch submission consumes them — segfault inside the
		// driver. The execution session parks the graph and hands recording a
		// fresh one; Sync() reclaims the deferred set after completion.
		return OaStatus::Ok();
	}

	// No batch active — compile, replay, submit, wait. Each Execute owns
	// its own command-buffer submission; safe to Reset immediately.
	graph->SetHostReadbackRequired(true);
	if (!graph->IsCompiled()) {
		const auto compileBegin = std::chrono::steady_clock::now();
		auto compileStatus = graph->Compile(*Impl_->Engine_);
		Impl_->Execution_.Stats().CompileMs += OaContextElapsedMs(compileBegin);
		if (graph->LastCompileReused()) ++Impl_->Execution_.Stats().CompileCacheHits;
		if (not compileStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Execute compile failed: %s",
				compileStatus.GetMessage().c_str());
			// Recover the context: drop the un-compilable graph so the bad node
			// doesn't poison every subsequent Execute. Without this, a single
			// missing/failed kernel (e.g. an op with no shader) permanently
			// bricks the shared default context — later ops silently read zeros.
			Impl_->Execution_.DiscardActiveRecording();
			return compileStatus;
		}
	}
	const auto submitBegin = std::chrono::steady_clock::now();
	auto status = graph->Replay(*Impl_->Engine_);
	Impl_->Execution_.Stats().SubmitMs += OaContextElapsedMs(submitBegin);
	if (not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Execute failed: %s",
			status.GetMessage().c_str());
		Impl_->Execution_.DiscardActiveRecording();
		return status;
	}
	// Wait for the non-blocking Replay() submission to complete before
	// clearing nodes. The next Compile() will free descriptor pools that
	// the GPU may still be referencing — must ensure GPU is done first.
	const auto waitBegin = std::chrono::steady_clock::now();
	OA_RETURN_IF_ERROR(graph->WaitForPendingReplay(Impl_->Engine_->Device));
	Impl_->Execution_.Stats().WaitMs += OaContextElapsedMs(waitBegin);
	Impl_->Execution_.Stats().HostBarrierCount = 1;
	graph->ReleaseCompletedBufferOwners();
	// Keep the command pool + secondary CB for reuse — ClearNodes() just
	// clears the node list and marks not-compiled. The next Compile() will
	// reset the CB and re-record, avoiding vkCreateCommandPool +
	// vkAllocateCommandBuffers per call (~0.05ms savings).
	graph->ClearNodes();
	Impl_->Execution_.MarkExecuted();
	return OaStatus::Ok();
}

OaResult<OaEvent> OaContext::Submit(OaGpuTimer* InTimer) {
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) return recordingStatus;
	// The execution session has one reclamation arena. Keep the first explicit
	// contract deliberately strict until it owns multiple independent
	// in-flight arenas: one submitted event must be consumed before
	// this context can submit another explicit workload.
	if (Impl_->Execution_.PendingEvent().IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaContext::Submit requires Wait() for the previous explicit event");
	}
	OA_RETURN_IF_ERROR(ExecuteInAsyncBatch(InTimer));
	return SubmitBatch();
}

OaStatus OaContext::BeginAsyncBatch() {
	assert(Impl_->Engine_ and "Engine is null");
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) return recordingStatus;
	if (Impl_->Execution_.IsBatchActive()) {
		return OaStatus::Ok();
	}
	Impl_->Execution_.ResetStats();
	return Impl_->Execution_.BeginBatch();
}

OaStatus OaContext::ExecuteInAsyncBatch(OaGpuTimer* InTimer) {
	assert(Impl_->Engine_ and "Engine is null");
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) return recordingStatus;
	auto* graph = Impl_->Execution_.Graph();
	assert(graph and "Graph is null");

	if (graph->NodeCount() == 0) {
		// Execute() still validates a semantic-only recording and rejects a
		// missing lowering. Do not erase that contract merely because no Vulkan
		// node was authored, and do not acquire a batch stream for an empty graph.
		return Execute();
	}

	OA_RETURN_IF_ERROR(BeginAsyncBatch());
	auto* stream = Impl_->Execution_.ActiveBatchStream();

	if (InTimer) {
		InTimer->Begin(stream);
	}
	auto executeStatus = Execute();
	if (InTimer) {
		InTimer->End(stream);
	}
	if (not executeStatus.IsOk()) {
		return executeStatus;
	}

	return OaStatus::Ok();
}

OaResult<OaEvent> OaContext::SubmitBatch() {
	assert(Impl_->Engine_ and "Engine is null");
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) return recordingStatus;
	const auto submitBegin = std::chrono::steady_clock::now();
	auto completion = Impl_->Execution_.SubmitBatch();
	Impl_->Execution_.Stats().SubmitMs += OaContextElapsedMs(submitBegin);
	if (completion.IsOk()) Impl_->Execution_.Stats().HostBarrierCount = 1;
	return completion;
}

OaBool OaContext::IsAsyncBatchActive() const noexcept {
	return Impl_->Execution_.IsBatchActive();
}

OaStatus OaContext::Wait(const OaEvent& InEvent) {
	const auto waitBegin = std::chrono::steady_clock::now();
	const auto status = Impl_->Execution_.Wait(InEvent);
	Impl_->Execution_.Stats().WaitMs += OaContextElapsedMs(waitBegin);
	return status;
}

OaStatus OaContext::Sync() {
	assert(Impl_->Engine_ and "Engine is null");
	const auto recordingStatus = Impl_->Execution_.ConsumeRecordingFailure();
	if (not recordingStatus.IsOk()) return recordingStatus;

	// Wait for any pending non-batch Replay() submission. Replay() now
	// submits without blocking (fire-and-forget) — same-queue ordering
	// ensures GPU executes in submission order. We wait here before
	// returning to the caller.
	if (auto* graph = Impl_->Execution_.Graph()) {
		OA_RETURN_IF_ERROR(graph->WaitForPendingReplay(Impl_->Engine_->Device));
	}

	if (Impl_->Execution_.IsBatchActive()) {
		auto completion = SubmitBatch();
		if (not completion.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaContext::Sync submit failed: %s",
				completion.GetStatus().GetMessage().c_str());
			return completion.GetStatus();
		}
		const auto waitStatus = Wait(completion.GetValue());
		if (not waitStatus.IsOk()) return waitStatus;
	} else if (Impl_->Execution_.PendingEvent().IsValid()) {
		const auto waitStatus = Wait(Impl_->Execution_.PendingEvent());
		if (not waitStatus.IsOk()) return waitStatus;
	}
	if (OaEnvFlag::IsSet("OA_LOG_RUNTIME_PHASES")) {
		OA_LOG_INFO(OaLogComponent::Core,
			"Runtime phases: nodes=%u graphs=%u cache_hits=%u boundary_barriers=%u "
			"host_barriers=%u compile=%.3f ms record=%.3f ms submit=%.3f ms wait=%.3f ms",
			Impl_->Execution_.Stats().NodeCount, Impl_->Execution_.Stats().GraphCount,
			Impl_->Execution_.Stats().CompileCacheHits, Impl_->Execution_.Stats().BoundaryBarrierCount,
			Impl_->Execution_.Stats().HostBarrierCount, Impl_->Execution_.Stats().CompileMs,
			Impl_->Execution_.Stats().RecordMs, Impl_->Execution_.Stats().SubmitMs, Impl_->Execution_.Stats().WaitMs);
	}

	return OaStatus::Ok();
}

OaU32 OaContext::MaxAsyncSubmissions() const noexcept {
	return 1U;
}

void OaContext::Clear() {
	Impl_->Execution_.Clear();
	if (not Impl_->Execution_.IsBatchActive()) {
		Impl_->Execution_.ClearBatchHazards();
		Impl_->Execution_.ResetStats();
	}
}

OaU32 OaContext::NodeCount() const noexcept {
	return Impl_->Execution_.NodeCount();
}
