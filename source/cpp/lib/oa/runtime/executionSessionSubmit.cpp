#include "executionSession.h"
#include "timerAccess.h"
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/timer.h>
#include <oa/runtime/stream.h>
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/chrono.h>

static oa::F64 executionSessionElapsedMs(
	oa::SteadyTimePoint inBegin) noexcept
{
	return (oa::steadyNow() - inBegin).toMilliseconds();
}

static void executionSessionMaybeLogGraph(const oa::ExecutableGraph& inGraph) {
	static oa::Atomic<oa::U32> sLoggedGraphs{0};

	const oa::I64 requested = oa::EnvFlag::getInt("OA_LOG_SESSION_GRAPH", 0);
	if (requested <= 0) {
		return;
	}

	const oa::U32 limit = static_cast<oa::U32>(requested);
	const oa::U32 index = sLoggedGraphs.fetchAdd(1, oa::MemoryOrder::Relaxed);
	if (index >= limit) {
		return;
	}

	const auto nodes = inGraph.nodes();
	OaLogInfo(oa::LogComponent::Compute,
		"oa::ExecutionSession graph #{}: {} node(s)", index + 1U,
		static_cast<oa::U32>(nodes.size()));
	for (oa::U32 i = 0; i < static_cast<oa::U32>(nodes.size()); ++i) {
		const auto& node = nodes[i];
		OaLogInfo(oa::LogComponent::Compute,
			"  [{:02}] {:<32} groups=({},{},{}) buffers={} push={}",
			i,
			node.shader.cStr(),
			node.groupsX,
			node.groupsY,
			node.groupsZ,
			static_cast<oa::U32>(node.buffers.size()),
			node.pushSize);
	}
}

oa::Status oa::ExecutionSession::recordActiveGraph_() {
	OA_ASSERT(engine_ and "Engine is null");
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutionSession recording transaction aborted: {}",
			recordingStatus.getMessage().cStr());
		return recordingStatus;
	}
	auto* activeGraph = graph();
	OA_ASSERT(activeGraph and "graph is null");
	const auto loweringStatus = validateLowering();
	if (not loweringStatus.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutionSession semantic lowering validation failed: {}",
			loweringStatus.getMessage().cStr());
		discardActiveRecording();
		return loweringStatus;
	}

	if (activeGraph->nodeCount() == 0) {
		markExecuted();
		return oa::Status::ok();
	}
	OA_ASSERT(isBatchActive()
		and "non-empty session recording requires an explicit submission");

	executionSessionMaybeLogGraph(*activeGraph);
	stats().nodeCount += activeGraph->nodeCount();

	// The primary batch emits one host-visibility edge immediately before
	// submission. Intermediate secondary graphs remain device-only.
	activeGraph->setHostReadbackRequired(false);
	// The execution session owns the primary command buffer. Record this
	// session's secondary graph into it; no engine-global ambient batch can
	// absorb work from another session.
	if (!activeGraph->isCompiled()) {
		const auto compileBegin = oa::steadyNow();
		auto compileStatus = activeGraph->compile(*engine_);
		stats().compileMs += executionSessionElapsedMs(compileBegin);
		if (activeGraph->lastCompileReused()) ++stats().compileCacheHits;
		if (not compileStatus.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"oa::ExecutionSession graph compile failed: {}",
				compileStatus.getMessage().cStr());
			// The primary command buffer does not reference this graph yet, so
			// discarding the rejected recording is safe.
			discardActiveRecording();
			return compileStatus;
		}
	}
	auto* stream = activeBatchStream();

	// Secondary-command-buffer boundaries provide no implicit dependency.
	// Insert the boundary barrier immediately before a following graph.
	const auto recordBegin = oa::steadyNow();
	auto recordResult = recordActiveGraphInBatch_(
		stream->commandBuffer);
	stats().recordMs += executionSessionElapsedMs(recordBegin);
	if (not recordResult.isOk()) {
		OaLogError(oa::LogComponent::Compute,
			"oa::ExecutionSession graph recording failed: {}",
			recordResult.getStatus().getMessage().cStr());
		return recordResult.getStatus();
	}
	stats().boundaryBarrierCount += recordResult.getValue();
	const auto graphStats = activeGraph->getStats();
	stats().dispatchCount += graphStats.dispatchCount;
	stats().intraGraphBarrierCount += graphStats.barrierCount;
	stats().warBarrierCount += graphStats.warBarrierCount;
	stats().indirectBarrierCount += graphStats.indirectBarrierCount;
	stats().aliasBarrierCount += graphStats.aliasBarrierCount;
	stats().descriptorSetCount += graphStats.descriptorSetCount;
	stats().kernelSelectionCount += graphStats.kernelSelectionCount;
	stats().kernelFallbackCount += graphStats.kernelFallbackCount;
	stats().precisionFallbackCount += graphStats.precisionFallbackCount;
	stats().layoutFallbackCount += graphStats.layoutFallbackCount;
	stats().naiveFallbackCount += graphStats.naiveFallbackCount;
	stats().referencedBufferBytes += graphStats.totalBufferBytes;
	++stats().graphCount;

	// The primary command buffer now references the secondary graph. rotation
	// retains its resources until wait() consumes the exact submission event.
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::ExecutionSession::submit(oa::Timer* inTimer) {
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) return recordingStatus;
	// The execution session has one reclamation arena. Keep the first explicit
	// contract deliberately strict until it owns multiple independent
	// in-flight arenas: one submitted event must be consumed before
	// this session can submit another explicit workload.
	if (pendingEvent().isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ExecutionSession::submit requires wait() for the previous explicit event");
	}
	const auto recordStatus = recordSubmission_(inTimer);
	if (not recordStatus.isOk()) {
		const auto abandonStatus = abandon();
		if (inTimer) oa::TimerAccess::cancelDevice(*inTimer);
		if (not abandonStatus.isOk()) {
			OaLogError(oa::LogComponent::Compute,
				"failed submission cleanup also failed: {}",
				abandonStatus.getMessage().cStr());
		}
		return recordStatus;
	}
	auto completion = submitRecorded_();
	if (not completion.isOk()) {
		if (inTimer) oa::TimerAccess::cancelDevice(*inTimer);
		return completion.getStatus();
	}
	if (inTimer) {
		const auto attachStatus = oa::TimerAccess::attachCompletion(
			*inTimer, completion.getValue());
		if (not attachStatus.isOk()) {
			// Submission already succeeded: never hide its event behind an
			// auxiliary profiling failure. The timer becomes unavailable, while
			// the caller still owns the exact completion boundary.
			OaLogError(oa::LogComponent::Compute,
				"device timer completion attachment failed: {}",
				attachStatus.getMessage().cStr());
		}
	}
	return completion;
}

oa::Status oa::ExecutionSession::beginSubmission_() {
	OA_ASSERT(engine_ and "Engine is null");
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) return recordingStatus;
	if (isBatchActive()) {
		return oa::Status::ok();
	}
	resetStats();
	return beginBatch_();
}

oa::Status oa::ExecutionSession::recordSubmission_(oa::Timer* inTimer) {
	OA_ASSERT(engine_ and "Engine is null");
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) return recordingStatus;
	auto* activeGraph = graph();
	OA_ASSERT(activeGraph and "graph is null");

	if (activeGraph->nodeCount() == 0) {
		// Recording validation still rejects a semantic operation with a
		// missing lowering. Do not erase that contract merely because no vulkan
		// node was authored, and do not acquire a batch stream for an empty graph.
		OA_RETURN_IF_ERROR(recordActiveGraph_());
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ExecutionSession::submit requires recorded device work");
	}

	OA_RETURN_IF_ERROR(beginSubmission_());
	auto* stream = activeBatchStream();

	if (inTimer) {
		OA_RETURN_IF_ERROR(oa::TimerAccess::beginDevice(*inTimer, stream));
	}
	auto executeStatus = recordActiveGraph_();
	if (inTimer) {
		const auto timerEnd = oa::TimerAccess::endDevice(*inTimer, stream);
		if (executeStatus.isOk() and not timerEnd.isOk()) return timerEnd;
	}
	if (not executeStatus.isOk()) {
		return executeStatus;
	}

	return oa::Status::ok();
}

oa::Result<oa::Event> oa::ExecutionSession::submitRecorded_() {
	OA_ASSERT(engine_ and "Engine is null");
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) return recordingStatus;
	const auto submitBegin = oa::steadyNow();
	auto completion = submitBatch_();
	stats().submitMs += executionSessionElapsedMs(submitBegin);
	if (completion.isOk()) {
		stats().hostBarrierCount = 1;
		++stats().submissionCount;
	}
	return completion;
}

oa::Bool oa::ExecutionSession::isPendingEvent(
	const oa::Event& inEvent) const noexcept
{
	return pendingEvent_.isValid()
		and pendingEvent_.isSameCompletion(inEvent);
}

oa::Status oa::ExecutionSession::wait(const oa::Event& inEvent) {
	const auto waitBegin = oa::steadyNow();
	const auto status = waitBatch_(inEvent);
	stats().waitMs += executionSessionElapsedMs(waitBegin);
	if (status.isOk() and oa::EnvFlag::isSet("OA_LOG_RUNTIME_PHASES")) {
		OaLogInfo(oa::LogComponent::Compute,
			"Runtime phases: nodes={} dispatches={} graphs={} submissions={} cache_hits={} "
			"barriers={} boundary_barriers={} host_barriers={} war={} indirect={} alias={} "
			"descriptor_sets={} referenced_bytes={} compile={:.3f} ms record={:.3f} ms "
			"submit={:.3f} ms wait={:.3f} ms",
			stats().nodeCount, stats().dispatchCount,
			stats().graphCount, stats().submissionCount,
			stats().compileCacheHits,
			stats().intraGraphBarrierCount,
			stats().boundaryBarrierCount,
			stats().hostBarrierCount,
			stats().warBarrierCount,
			stats().indirectBarrierCount,
			stats().aliasBarrierCount,
			stats().descriptorSetCount,
			static_cast<unsigned long long>(stats().referencedBufferBytes),
			stats().compileMs,
			stats().recordMs, stats().submitMs,
			stats().waitMs);
	}
	return status;
}

oa::Status oa::ExecutionSession::submitAndWait(oa::Timer* inTimer)
{
	if (pendingEvent_.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::ExecutionSession completion requires the previous event to be consumed");
	}
	if (nodeCount() == 0U) {
		return recordActiveGraph_();
	}
	auto submitted = submit(inTimer);
	if (not submitted.isOk()) return submitted.getStatus();
	return wait(submitted.getValue());
}

oa::Status oa::ExecutionSession::validateCapture() {
	const auto recordingStatus = consumeRecordingFailure();
	if (not recordingStatus.isOk()) {
		return recordingStatus;
	}
	OA_RETURN_IF_ERROR(validateLowering());
	if (nodeCount() == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::capture requires recorded device work");
	}
	return oa::Status::ok();
}
