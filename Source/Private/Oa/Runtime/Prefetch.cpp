#include <Oa/Runtime/Prefetch.h>

#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/UploadRing.h>
#include <Oa/Core/Log.h>

#include <algorithm>

struct OaPrefetchPipeline::Impl {
	OaEngine* Engine = nullptr;
	OaUploadRing UploadRing;
	OaVkBuffer DeviceBuffer;
	OaU64 BufferSize = 0;
	OaBool CopyPending = false;
};

OaPrefetchPipeline::OaPrefetchPipeline(OaPrefetchPipeline&&) noexcept = default;
OaPrefetchPipeline& OaPrefetchPipeline::operator=(OaPrefetchPipeline&& InOther) noexcept {
	if (this != &InOther) {
		Abandon_();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}
OaPrefetchPipeline::~OaPrefetchPipeline() {
	Abandon_();
}

OaResult<OaPrefetchPipeline> OaPrefetchPipeline::Create(
	OaEngine& InRt,
	OaU64 InBufferSize)
{
	if (InBufferSize == 0) {
		return OaStatus::InvalidArgument("PrefetchPipeline: buffer size must be non-zero");
	}
	OaPrefetchPipeline pipeline;
	pipeline.Impl_ = OaMakeUniquePtr<Impl>();
	pipeline.Impl_->Engine = &InRt;
	pipeline.Impl_->BufferSize = InBufferSize;

	auto device = InRt.AllocBufferDevice(InBufferSize);
	if (!device) return device.GetStatus();
	pipeline.Impl_->DeviceBuffer = OaStdMove(*device);

	constexpr OaU64 kAlignment = 256;
	const OaU64 frameBytes =
		((InBufferSize + kAlignment - 1) / kAlignment) * kAlignment;
	auto ring = OaUploadRing::Create(InRt, OaUploadRingConfig{
		.CapacityBytes = frameBytes * 2,
		.FramesInFlight = 2,
		.Alignment = kAlignment,
	});
	if (!ring) {
		InRt.FreeBuffer(pipeline.Impl_->DeviceBuffer);
		return ring.GetStatus();
	}
	pipeline.Impl_->UploadRing = OaStdMove(*ring);
	return pipeline;
}

void OaPrefetchPipeline::Destroy(OaEngine& InRt) {
	if (!Impl_) return;
	if (const auto status = Impl_->UploadRing.Close(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaPrefetchPipeline::Destroy: upload close failed: %s",
			status.GetMessage().c_str());
	}
	InRt.FreeBuffer(Impl_->DeviceBuffer);
	Impl_.reset();
}

void OaPrefetchPipeline::Abandon_() noexcept {
	if (!Impl_) return;
	if (Impl_->Engine && Impl_->Engine->Device.Device) {
		// The compatibility facade has no submitted-but-unwaited state: BeginCopy
		// only records, while Wait submits and completes. Abandon the ring first so
		// an open recording is cancelled before its destination buffer is released.
		Impl_->UploadRing = OaUploadRing{};
		Impl_->Engine->FreeBuffer(Impl_->DeviceBuffer);
	}
	Impl_.reset();
}

OaStatus OaPrefetchPipeline::BeginCopy(const void* InData, OaU64 InSize) {
	if (!Impl_ || !Impl_->Engine) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"PrefetchPipeline is not initialized");
	}
	if (Impl_->CopyPending) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"PrefetchPipeline already has a pending copy");
	}
	if (InData == nullptr || InSize == 0 || InSize > Impl_->BufferSize) {
		return OaStatus::InvalidArgument("PrefetchPipeline: invalid source range");
	}
	OA_RETURN_IF_ERROR(Impl_->UploadRing.BeginBatch());
	OA_RETURN_IF_ERROR(Impl_->UploadRing.Upload(
		Impl_->DeviceBuffer, 0, InData, InSize));
	Impl_->CopyPending = true;
	return OaStatus::Ok();
}

OaResult<OaVkBuffer> OaPrefetchPipeline::Wait() {
	if (!Impl_ || !Impl_->CopyPending) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"PrefetchPipeline has no pending copy");
	}
	auto completion = Impl_->UploadRing.Submit();
	if (!completion) return completion.GetStatus();
	if (completion->IsValid()) {
		const OaStatus status = completion->Wait();
		if (!status.IsOk()) return status;
	}
	Impl_->CopyPending = false;
	return Impl_->DeviceBuffer;
}

void OaPrefetchPipeline::Swap() {
	// OaUploadRing rotates frames after every submission.
}

bool OaPrefetchPipeline::IsBusy() const {
	return Impl_ && Impl_->CopyPending;
}

OaU64 OaPrefetchPipeline::BufferSize() const {
	return Impl_ ? Impl_->BufferSize : 0;
}
