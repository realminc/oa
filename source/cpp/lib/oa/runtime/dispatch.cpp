#include <oa/runtime/dispatch.h>
#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/stream.h>
#include <vkl/vkl.h>
#include <oa/core/thread.h>
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/validation.h>

static oa::Result<oa::StringView> resolveKernelName(oa::U64 inKernelId) {
	const oa::ComputeKernel* kernel = oa::computeKernelFindByPackedId(inKernelId);
	if (!kernel) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "dispatch: unknown kernel id");
	}
	return oa::StringView(kernel->name);
}

oa::Status oavk::Dispatch::run(
	oa::Engine& inRuntime,
	oa::StringView inPipelineName,
	oa::Span<oavk::Buffer> inBuffers,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ
) {
	// NOTE: Debug logging and validation removed from hot path - was causing 2.5x slowdown.
	// in Debug builds, oa::Validation::isEnabled() does atomic load on EVERY validation check.
	// With ~21 dispatches per training step × 6 validates per dispatch = 126 atomic loads/step.
	// Validation should be done at higher-level API boundaries, not in the dispatch hot path.
	
	OA_DEBUG_COUNTER_INC(dispatch_count);

	return oavk::Stream::runOnce(
		inRuntime, inPipelineName, inBuffers,
		inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ);
}

oa::Status oavk::Dispatch::run(
	oa::Engine& inRuntime,
	oa::U64 inKernelId,
	oa::Span<oavk::Buffer> inBuffers,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ
) {
	auto kernelName = resolveKernelName(inKernelId);
	if (!kernelName) {
		return kernelName.getStatus();
	}
	return run(
		inRuntime, *kernelName, inBuffers, inPushData, inPushSize,
		inStorageDtype, inGroupsX, inGroupsY, inGroupsZ);
}

oa::Status oavk::Dispatch::runIndirect(
	oa::Engine& inRuntime,
	oa::StringView inPipelineName,
	oa::Span<oavk::Buffer> inBuffers,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	const oavk::Buffer& inIndirectBuffer,
	oa::U64 inOffset
) {
	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRuntime);
	if (!stream) {
		return oa::Status::error(oa::StatusCode::VulkanError, "failed to acquire stream for indirect");
	}
	oa::Status status = stream->begin(oa::EngineDeviceAccess::get(inRuntime));
	if (not status.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(inRuntime, stream);
		return status;
	}
	status = stream->recordDispatchIndirect(
		inRuntime, inPipelineName, inBuffers,
		inPushData, inPushSize, inStorageDtype, inIndirectBuffer, inOffset);
	if (not status.isOk()) {
		(void)stream->resetUnsubmitted(oa::EngineDeviceAccess::get(inRuntime));
		oa::EngineSubmissionAccess::releaseStream(inRuntime, stream);
		return status;
	}
	status = stream->submitAndWait(inRuntime);
	oa::EngineSubmissionAccess::releaseStream(inRuntime, stream);
	return status;
}

oa::Result<oavk::Batch> oavk::Dispatch::beginBatch(oa::Engine& inRuntime) {
	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(inRuntime);
	if (!stream) {
		return oa::Status::error(oa::StatusCode::VulkanError, "failed to acquire stream for batch");
	}

	const auto begin = stream->begin(oa::EngineDeviceAccess::get(inRuntime));
	if (not begin.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(inRuntime, stream);
		return begin;
	}

	oavk::Batch batch;
	batch.stream = stream;
	return batch;
}

oa::Status oavk::Dispatch::record(
	oavk::Batch& inBatch,
	oa::Engine& inRuntime,
	oa::StringView inPipelineName,
	oa::Span<oavk::Buffer> inBuffers,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ
) {
	if (not inBatch.stream) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Record: batch has no active stream");
	}
	OA_VALIDATE(inPushSize <= 128U, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Record '%.*s': push size %u exceeds 128-byte vulkan minimum guarantee",
		static_cast<int>(inPipelineName.size()), inPipelineName.data(), inPushSize);
	OA_VALIDATE(inPushSize == 0U or inPushData != nullptr, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
		"Record '%.*s': null push data with push size %u",
		static_cast<int>(inPipelineName.size()), inPipelineName.data(), inPushSize);
	for (oa::Usize i = 0; i < inBuffers.size(); ++i) {
		OA_VALIDATE(inBuffers[i].buffer != nullptr, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
			"Record '%.*s': buffer[%zu] has null VkBuffer",
			static_cast<int>(inPipelineName.size()), inPipelineName.data(), i);
		OA_VALIDATE(inBuffers[i].bindlessIndex != UINT32_MAX, oa::ValidationSeverity::Error, oa::LogComponent::Compute,
			"Record '%.*s': buffer[%zu] not registered in bindless heap",
			static_cast<int>(inPipelineName.size()), inPipelineName.data(), i);
	}

	const auto status = inBatch.stream->record(
		inRuntime, inPipelineName, inBuffers,
		inPushData, inPushSize,
		inStorageDtype,
		inGroupsX, inGroupsY, inGroupsZ
	);
	if (not status.isOk()) {
		(void)inBatch.stream->resetUnsubmitted(oa::EngineDeviceAccess::get(inRuntime));
		oa::EngineSubmissionAccess::releaseStream(inRuntime, inBatch.stream);
		inBatch.stream = nullptr;
	}
	return status;
}

oa::Status oavk::Dispatch::record(
	oavk::Batch& inBatch,
	oa::Engine& inRuntime,
	oa::U64 inKernelId,
	oa::Span<oavk::Buffer> inBuffers,
	const void* inPushData,
	oa::U32 inPushSize,
	oa::ScalarType inStorageDtype,
	oa::U32 inGroupsX,
	oa::U32 inGroupsY,
	oa::U32 inGroupsZ
) {
	auto kernelName = resolveKernelName(inKernelId);
	if (!kernelName) {
		return kernelName.getStatus();
	}
	return record(
		inBatch, inRuntime, *kernelName, inBuffers, inPushData, inPushSize,
		inStorageDtype, inGroupsX, inGroupsY, inGroupsZ);
}

oa::Status oavk::Dispatch::flush(oavk::Batch& inBatch, oa::Engine& inRuntime) {
	if (not inBatch.stream) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"flush: batch has no active stream");
	}
	oa::Status status = inBatch.stream->submitAndWait(inRuntime);
	oa::EngineSubmissionAccess::releaseStream(inRuntime, inBatch.stream);
	inBatch.stream = nullptr;
	return status;
}
