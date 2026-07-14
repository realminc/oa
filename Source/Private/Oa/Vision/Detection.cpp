#include <Oa/Vision/Detection.h>

#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Engine.h>

OaDetectionBuffer::OaDetectionBuffer(OaDetectionBuffer&& InOther) noexcept
	: Runtime_(InOther.Runtime_)
	, Buffer_(InOther.Buffer_)
	, ConsumerSemaphore_(InOther.ConsumerSemaphore_)
	, ConsumerValue_(InOther.ConsumerValue_)
	, Count_(InOther.Count_)
	, Capacity_(InOther.Capacity_)
{
	InOther.Runtime_ = nullptr;
	InOther.Buffer_ = {};
	InOther.ConsumerSemaphore_ = {};
	InOther.ConsumerValue_ = 0;
	InOther.Count_ = 0;
	InOther.Capacity_ = 0;
}

OaDetectionBuffer& OaDetectionBuffer::operator=(OaDetectionBuffer&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Runtime_ = InOther.Runtime_;
		Buffer_ = InOther.Buffer_;
		ConsumerSemaphore_ = InOther.ConsumerSemaphore_;
		ConsumerValue_ = InOther.ConsumerValue_;
		Count_ = InOther.Count_;
		Capacity_ = InOther.Capacity_;

		InOther.Runtime_ = nullptr;
		InOther.Buffer_ = {};
		InOther.ConsumerSemaphore_ = {};
		InOther.ConsumerValue_ = 0;
		InOther.Count_ = 0;
		InOther.Capacity_ = 0;
	}
	return *this;
}

OaDetectionBuffer::~OaDetectionBuffer() {
	Destroy();
}

OaResult<OaDetectionBuffer> OaDetectionBuffer::CreateHostUpload(
	OaComputeEngine& InRuntime,
	OaU32 InCapacity) {
	if (InCapacity == 0) {
		return OaStatus::InvalidArgument(
			"OaDetectionBuffer::CreateHostUpload: capacity must be non-zero");
	}

	auto allocation = InRuntime.AllocBuffer(
		static_cast<OaU64>(InCapacity) * sizeof(OaDetection));
	if (!allocation.IsOk()) return allocation.GetStatus();

	OaDetectionBuffer result;
	result.Runtime_ = &InRuntime;
	result.Buffer_ = OaStdMove(*allocation);
	result.Capacity_ = InCapacity;
	return result;
}

void OaDetectionBuffer::Destroy() {
	if (Runtime_ && Buffer_.Buffer) {
		Runtime_->FreeBuffer(Buffer_);
	}
	Runtime_ = nullptr;
	Buffer_ = {};
	ConsumerSemaphore_ = {};
	ConsumerValue_ = 0;
	Count_ = 0;
	Capacity_ = 0;
}

OaStatus OaDetectionBuffer::Upload(OaSpan<const OaDetection> InDetections) {
	if (!Runtime_ || !Buffer_.Buffer || !Buffer_.MappedPtr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaDetectionBuffer::Upload: buffer is not host-visible");
	}
	if (!IsReady()) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"OaDetectionBuffer::Upload: buffer is still consumed by the GPU");
	}
	if (InDetections.Size() > Capacity_) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaDetectionBuffer::Upload: detection count exceeds capacity");
	}

	const OaUsize bytes = InDetections.Size() * sizeof(OaDetection);
	if (bytes > 0) {
		OaMemcpy(Buffer_.MappedPtr, InDetections.Data(), bytes);
	}
	Count_ = static_cast<OaU32>(InDetections.Size());
	ConsumerSemaphore_ = {};
	ConsumerValue_ = 0;
	return OaStatus::Ok();
}

void OaDetectionBuffer::MarkConsumed(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (!IsValid() || InSemaphore.Semaphore == nullptr || InValue == 0) return;
	ConsumerSemaphore_ = InSemaphore;
	ConsumerValue_ = InValue;
}

bool OaDetectionBuffer::IsReady() const {
	if (!Runtime_ || !IsValid()) return false;
	return ConsumerSemaphore_.Semaphore == nullptr
		|| ConsumerSemaphore_.GetValue(Runtime_->Device) >= ConsumerValue_;
}
