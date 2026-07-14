// OaPrefetch.cpp — Double-Buffer Async Staging Pipeline

#include <Oa/Core/Prefetch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Log.h>

// ============================================================================
// OaPrefetchPipeline Implementation
// ============================================================================

struct OaPrefetchPipeline::Impl {
	OaVkBuffer StagingBuffers[2];  // Double-buffered host-visible staging
	OaVkBuffer DeviceBuffer;       // Device-local target
	OaU64 BufferSize = 0;
	OaU32 CurrentStaging = 0;      // 0 or 1
	OaBool Initialized = false;
	OaBool CopyInProgress = false;
	
	// Background copy state
	OaSharedPtr<OaTask<void>> CopyTask;
	const void* PendingData = nullptr;
	OaU64 PendingSize = 0;
};

OaResult<OaPrefetchPipeline> OaPrefetchPipeline::Create(
	OaComputeEngine& InRt,
	OaU64 InBufferSize)
{
	OaPrefetchPipeline pipeline;
	pipeline.Impl_ = OaMakeUniquePtr<Impl>();
	
	// Allocate device-local buffer
	auto deviceResult = InRt.AllocBufferDevice(InBufferSize);
	if (not deviceResult.IsOk()) {
		return OaResult<OaPrefetchPipeline>(deviceResult.GetStatus());
	}
	pipeline.Impl_->DeviceBuffer = std::move(deviceResult).GetValue();
	
	// Allocate two host-visible staging buffers
	for (OaU32 i = 0; i < 2; ++i) {
		auto stagingResult = InRt.AllocBuffer(InBufferSize);
		if (not stagingResult.IsOk()) {
			// Cleanup on failure
			InRt.FreeBuffer(pipeline.Impl_->DeviceBuffer);
			for (OaU32 j = 0; j < i; ++j) {
				InRt.FreeBuffer(pipeline.Impl_->StagingBuffers[j]);
			}
			return OaResult<OaPrefetchPipeline>(stagingResult.GetStatus());
		}
		pipeline.Impl_->StagingBuffers[i] = std::move(stagingResult).GetValue();
	}
	
	pipeline.Impl_->BufferSize = InBufferSize;
	pipeline.Impl_->CurrentStaging = 0;
	pipeline.Impl_->Initialized = true;
	
	OA_LOG_INFO(OaLogComponent::Core,
		"OaPrefetchPipeline created: %.2f MB staging (x2) + device",
		static_cast<double>(InBufferSize) / (1024.0 * 1024.0));
	
	return OaResult<OaPrefetchPipeline>(std::move(pipeline));
}

void OaPrefetchPipeline::Destroy(OaComputeEngine& InRt) {
	if (not Impl_ or not Impl_->Initialized) {
		return;
	}
	
	// Wait for any pending copy
	if (Impl_->CopyInProgress and Impl_->CopyTask and Impl_->CopyTask->IsDone()) {
		Impl_->CopyTask->Wait();
	}
	
	InRt.FreeBuffer(Impl_->DeviceBuffer);
	InRt.FreeBuffer(Impl_->StagingBuffers[0]);
	InRt.FreeBuffer(Impl_->StagingBuffers[1]);
	
	Impl_->Initialized = false;
}

OaStatus OaPrefetchPipeline::BeginCopy(const void* InData, OaU64 InSize) {
	if (not Impl_ or not Impl_->Initialized) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "Pipeline not initialized");
	}
	
	if (InSize > Impl_->BufferSize) {
		return OaStatus::InvalidArgument("Data size exceeds buffer capacity");
	}
	
	if (Impl_->CopyInProgress) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "Copy already in progress");
	}
	
	// Store copy parameters
	Impl_->PendingData = InData;
	Impl_->PendingSize = InSize;
	Impl_->CopyInProgress = true;
	
	// Start background copy to staging buffer
	OaVkBuffer& staging = Impl_->StagingBuffers[Impl_->CurrentStaging];
	auto pool = OaThreadPool::Create();
	
	Impl_->CopyTask = pool.SubmitTask([staging, InData, InSize]() {
		if (staging.MappedPtr) {
			OaMemcpy(staging.MappedPtr, InData, InSize);
		}
	});
	
	return OaStatus::Ok();
}

OaResult<OaVkBuffer> OaPrefetchPipeline::Wait() {
	if (not Impl_ or not Impl_->Initialized) {
		return OaResult<OaVkBuffer>(
			OaStatus::Error(OaStatusCode::FailedPrecondition, "Pipeline not initialized"));
	}
	
	if (not Impl_->CopyInProgress) {
		return OaResult<OaVkBuffer>(
			OaStatus::Error(OaStatusCode::FailedPrecondition, "No copy in progress"));
	}
	
	// Wait for background copy to complete
	if (Impl_->CopyTask and Impl_->CopyTask->IsDone()) {
		Impl_->CopyTask->Wait();
	}
	
	Impl_->CopyInProgress = false;
	
	// Return device buffer (caller will use CopyBufferAsync separately if needed)
	return OaResult<OaVkBuffer>(Impl_->DeviceBuffer);
}

void OaPrefetchPipeline::Swap() {
	if (not Impl_ or not Impl_->Initialized) {
		return;
	}
	
	// Swap staging buffers
	Impl_->CurrentStaging = 1 - Impl_->CurrentStaging;
}

bool OaPrefetchPipeline::IsBusy() const {
	if (not Impl_) {
		return false;
	}
	return Impl_->CopyInProgress;
}

OaU64 OaPrefetchPipeline::BufferSize() const {
	if (not Impl_) {
		return 0;
	}
	return Impl_->BufferSize;
}

// ============================================================================
// Convenience API
// ============================================================================

OaStatus OaPrefetchSync(
	OaComputeEngine& InRt,
	const void* InData,
	OaU64 InSize,
	OaVkBuffer OutDeviceBuffer)
{
	// Allocate temporary staging buffer
	auto stagingResult = InRt.AllocBuffer(InSize);
	if (not stagingResult.IsOk()) {
		return stagingResult.GetStatus();
	}
	OaVkBuffer staging = std::move(stagingResult).GetValue();
	
	// Copy CPU → staging
	if (not staging.MappedPtr) {
		InRt.FreeBuffer(staging);
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "Staging buffer not mapped");
	}
	OaMemcpy(staging.MappedPtr, InData, InSize);
	
	// Copy staging → device (synchronous)
	OaStatus status = InRt.CopyBufferAsync(staging, OutDeviceBuffer, InSize);
	if (not status.IsOk()) {
		InRt.FreeBuffer(staging);
		return status;
	}
	
	// Wait for transfer
	status = InRt.WaitTransfer();
	InRt.FreeBuffer(staging);
	
	return status;
}

OaStatus OaPrefetchAsync(
	OaComputeEngine& InRt,
	const void* InData,
	OaU64 InSize,
	OaVkBuffer OutDeviceBuffer)
{
	// Allocate temporary staging buffer
	auto stagingResult = InRt.AllocBuffer(InSize);
	if (not stagingResult.IsOk()) {
		return stagingResult.GetStatus();
	}
	OaVkBuffer staging = std::move(stagingResult).GetValue();
	
	// Copy CPU → staging
	if (not staging.MappedPtr) {
		InRt.FreeBuffer(staging);
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "Staging buffer not mapped");
	}
	OaMemcpy(staging.MappedPtr, InData, InSize);
	
	// Start async copy staging → device
	OaStatus status = InRt.CopyBufferAsync(staging, OutDeviceBuffer, InSize);
	
	// Note: staging buffer leaks here - caller must track and free after Wait
	// This is a limitation of the simple API
	
	return status;
}

OaStatus OaPrefetchWait(OaComputeEngine& InRt) {
	return InRt.WaitTransfer();
}
