#include <Oa/Runtime/Prefetch.h>

#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/UploadRing.h>

#include <algorithm>

struct OaPrefetchPipeline::Impl {
	OaComputeEngine* Engine = nullptr;
	OaUploadRing UploadRing;
	OaVkBuffer DeviceBuffer;
	OaU64 BufferSize = 0;
	OaBool CopyPending = false;
};

OaPrefetchPipeline::OaPrefetchPipeline(OaPrefetchPipeline&&) noexcept = default;
OaPrefetchPipeline& OaPrefetchPipeline::operator=(OaPrefetchPipeline&& InOther) noexcept {
	if (this != &InOther) {
		if (Impl_ && Impl_->Engine) Destroy(*Impl_->Engine);
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}
OaPrefetchPipeline::~OaPrefetchPipeline() {
	if (Impl_ && Impl_->Engine) Destroy(*Impl_->Engine);
}

OaResult<OaPrefetchPipeline> OaPrefetchPipeline::Create(
	OaComputeEngine& InRt,
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

void OaPrefetchPipeline::Destroy(OaComputeEngine& InRt) {
	if (!Impl_) return;
	(void)Impl_->UploadRing.Wait();
	Impl_->UploadRing.Destroy();
	InRt.FreeBuffer(Impl_->DeviceBuffer);
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
