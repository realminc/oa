#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Scheduler.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/Thread.h>
#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Validation.h>

static OaResult<OaStringView> ResolveKernelName(OaKernelId InKernelId) {
	const OaComputeKernel* kernel = OaComputeKernelFindByPackedId(InKernelId);
	if (!kernel) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "dispatch: unknown kernel id");
	}
	return OaStringView(kernel->Name);
}

OaStatus OaVkDispatch::Run(
	OaComputeEngine& InRuntime,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	// NOTE: Debug logging and validation removed from hot path - was causing 2.5x slowdown.
	// In Debug builds, OaValidation::IsEnabled() does atomic load on EVERY validation check.
	// With ~21 dispatches per training step × 6 validates per dispatch = 126 atomic loads/step.
	// Validation should be done at higher-level API boundaries, not in the dispatch hot path.
	
	OA_DEBUG_COUNTER_INC(dispatch_count);

	if (InRuntime.ComputeBatchStream_) {
		OaVkBatch batch;
		batch.Stream = InRuntime.ComputeBatchStream_;
		return Record(batch, InRuntime, InPipelineName, InBuffers,
			InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
	}
	return OaVkStream::RunOnce(
		InRuntime, InPipelineName, InBuffers,
		InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ);
}

OaStatus OaVkDispatch::Run(
	OaComputeEngine& InRuntime,
	OaKernelId InKernelId,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	auto kernelName = ResolveKernelName(InKernelId);
	if (!kernelName) {
		return kernelName.GetStatus();
	}
	return Run(InRuntime, *kernelName, InBuffers, InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
}

OaStatus OaVkDispatch::RunIndirect(
	OaComputeEngine& InRuntime,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	const OaVkBuffer& InIndirectBuffer,
	OaU64 InOffset
) {
	OaVkStream* stream = InRuntime.AcquireStream();
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError, "failed to acquire stream for indirect");
	}
	OA_RETURN_IF_ERROR(stream->Begin(InRuntime.Device));
	OA_RETURN_IF_ERROR(stream->RecordDispatchIndirect(
		InRuntime, InPipelineName, InBuffers,
		InPushData, InPushSize, InIndirectBuffer, InOffset));
	OaStatus status = stream->SubmitAndWait(InRuntime);
	InRuntime.ReleaseStream(stream);
	return status;
}

OaResult<OaVkBatch> OaVkDispatch::BeginBatch(OaComputeEngine& InRuntime) {
	OaVkStream* stream = InRuntime.AcquireStream();
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError, "failed to acquire stream for batch");
	}

	OA_RETURN_IF_ERROR(stream->Begin(InRuntime.Device));

	OaVkBatch batch;
	batch.Stream = stream;
	return batch;
}

OaStatus OaVkDispatch::Record(
	OaVkBatch& InBatch,
	OaComputeEngine& InRuntime,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	OA_VALIDATE(InBatch.Stream != nullptr, OaValidationSeverity::Error, OaLogComponent::Core,
		"Record '%.*s': null batch stream — call BeginBatch first",
		static_cast<int>(InPipelineName.size()), InPipelineName.data());
	OA_VALIDATE(InGroupsX > 0U, OaValidationSeverity::Error, OaLogComponent::Core,
		"Record '%.*s': InGroupsX=0",
		static_cast<int>(InPipelineName.size()), InPipelineName.data());
	OA_VALIDATE(InPushSize <= 128U, OaValidationSeverity::Error, OaLogComponent::Core,
		"Record '%.*s': push size %u exceeds 128-byte Vulkan minimum guarantee",
		static_cast<int>(InPipelineName.size()), InPipelineName.data(), InPushSize);
	OA_VALIDATE(InPushSize == 0U or InPushData != nullptr, OaValidationSeverity::Error, OaLogComponent::Core,
		"Record '%.*s': null push data with push size %u",
		static_cast<int>(InPipelineName.size()), InPipelineName.data(), InPushSize);
	for (OaUsize i = 0; i < InBuffers.size(); ++i) {
		OA_VALIDATE(InBuffers[i].Buffer != nullptr, OaValidationSeverity::Error, OaLogComponent::Core,
			"Record '%.*s': buffer[%zu] has null VkBuffer",
			static_cast<int>(InPipelineName.size()), InPipelineName.data(), i);
		OA_VALIDATE(InBuffers[i].BindlessIndex != UINT32_MAX, OaValidationSeverity::Error, OaLogComponent::Core,
			"Record '%.*s': buffer[%zu] not registered in bindless heap",
			static_cast<int>(InPipelineName.size()), InPipelineName.data(), i);
	}

	return InBatch.Stream->Record(
		InRuntime, InPipelineName, InBuffers,
		InPushData, InPushSize,
		InGroupsX, InGroupsY, InGroupsZ
	);
}

OaStatus OaVkDispatch::Record(
	OaVkBatch& InBatch,
	OaComputeEngine& InRuntime,
	OaKernelId InKernelId,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	auto kernelName = ResolveKernelName(InKernelId);
	if (!kernelName) {
		return kernelName.GetStatus();
	}
	return Record(InBatch, InRuntime, *kernelName, InBuffers, InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
}

OaStatus OaVkDispatch::Flush(OaVkBatch& InBatch, OaComputeEngine& InRuntime) {
	OaStatus status = InBatch.Stream->SubmitAndWait(InRuntime);
	InRuntime.ReleaseStream(InBatch.Stream);
	InBatch.Stream = nullptr;
	return status;
}

// ─── Multi-Device Dispatch Stubs ────────────────────────────────────────────

OaStatus OaVkDispatch::Run(
	OaComputeEngine& InRuntime,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ,
	const OaDispatchHint& InHint
) {
	// Phase 1 stub: ignore hint, dispatch to primary device.
	(void)InHint;
	return Run(InRuntime, InPipelineName, InBuffers,
		InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
}

OaStatus OaVkDispatch::RunOn(
	OaComputeEngine& InRuntime,
	OaU32 InNodeIndex,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	OA_VALIDATE_NODE_INDEX(InNodeIndex, InRuntime.DeviceCount(), "RunOn");
	if (InNodeIndex == 0) {
		return Run(InRuntime, InPipelineName, InBuffers,
			InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ);
	}
	OA_VALIDATE(InRuntime.IsMultiDevice(), OaValidationSeverity::Error, OaLogComponent::Core,
		"RunOn: non-zero node index %u requires multi-device engine", InNodeIndex);
	auto* node = InRuntime.GetNode(InNodeIndex);
	OA_VALIDATE(node != nullptr, OaValidationSeverity::Error, OaLogComponent::Core,
		"RunOn: node %u not found", InNodeIndex);
	for (OaU32 i = 0; i < static_cast<OaU32>(InBuffers.size()); ++i) {
		OA_VALIDATE(InBuffers[i].NodeIndex == InNodeIndex, OaValidationSeverity::Error, OaLogComponent::Core,
			"RunOn: buffer[%u] is on node %u, expected node %u",
			i, InBuffers[i].NodeIndex, InNodeIndex);
	}

	OaDeviceMesh* mesh = InRuntime.GetMesh();
	if (!mesh) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "RunOn: multi-device without mesh");
	}

	OaSpinlockGuard guard(*mesh->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	struct OaRestorePrimaryVkDispatch {
		OaComputeEngine& Rt;
		explicit OaRestorePrimaryVkDispatch(OaComputeEngine& InRt) : Rt(InRt) {}
		~OaRestorePrimaryVkDispatch() {
			OaVkLoadDevice(static_cast<VkDevice>(Rt.Device.Device));
		}
	} restorePrimary{InRuntime};

	OaVkStream* stream = InRuntime.AcquireStreamOn(InNodeIndex);
	if (!stream) {
		return OaStatus::Error(OaStatusCode::VulkanError, "RunOn: failed to acquire stream on node");
	}

	OA_RETURN_IF_ERROR(stream->Begin(node->Device));
	OA_RETURN_IF_ERROR(stream->RecordDispatchOnNode(
		InRuntime, InNodeIndex, InPipelineName, InBuffers,
		InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ));
	OaStatus status = stream->SubmitAndWait(InRuntime, true);
	InRuntime.ReleaseStreamOn(InNodeIndex, stream);
	return status;
}

OaResult<OaDispatchTicket> OaVkDispatch::RunAsync(
	OaComputeEngine& InRuntime,
	OaStringView InPipelineName,
	OaSpan<OaVkBuffer> InBuffers,
	const void* InPushData,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	// Async path must not defer into an engine compute batch (ticket would lie about completion).
	OA_RETURN_IF_ERROR(OaVkStream::RunOnce(
		InRuntime, InPipelineName, InBuffers,
		InPushData, InPushSize, InGroupsX, InGroupsY, InGroupsZ));
	OaDispatchTicket ticket;
	ticket.NodeIndex = 0;
	return ticket;
}
