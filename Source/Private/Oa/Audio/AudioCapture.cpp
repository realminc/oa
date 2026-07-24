#include <Oa/Audio/AudioCapture.h>

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>
#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

namespace {

OaU64 MonotonicMicroseconds()
{
	return static_cast<OaU64>(std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

struct OaAudioCapture::Impl {
	OaEngine* Engine = nullptr;
	ma_device Device = {};
	OaAudioCaptureConfig Config = {};
	OaVec<OaF32> Ring;
	OaVec<OaU64> RingFrameIndices;
	OaU64 CapacityFrames = 0U;
	std::atomic<OaU64> WriteFrame = 0U;
	std::atomic<OaU64> ReadFrame = 0U;
	std::atomic<OaU64> CapturedFrame = 0U;
	std::atomic<OaU64> DroppedFrames = 0U;
	std::atomic<OaU64> EpochUs = 0U;
	std::atomic<bool> Started = false;
	std::atomic<bool> AcceptingCallbacks = false;
	bool Initialized = false;
};

namespace {

void CaptureCallback(
	ma_device* InDevice,
	void*,
	const void* InInput,
	ma_uint32 InFrameCount)
{
	auto* impl = static_cast<OaAudioCapture::Impl*>(InDevice->pUserData);
	if (impl == nullptr || InInput == nullptr || InFrameCount == 0U) return;
	if (not impl->AcceptingCallbacks.load(std::memory_order_acquire)) return;
	const OaU64 write = impl->WriteFrame.load(std::memory_order_relaxed);
	const OaU64 read = impl->ReadFrame.load(std::memory_order_acquire);
	const OaU64 freeFrames = impl->CapacityFrames - std::min(impl->CapacityFrames, write - read);
	const OaU64 accepted = std::min<OaU64>(freeFrames, InFrameCount);
	const OaU64 dropped = static_cast<OaU64>(InFrameCount) - accepted;
	const OaU64 capturedFirst = impl->CapturedFrame.fetch_add(
		InFrameCount, std::memory_order_relaxed);
	if (dropped > 0U) impl->DroppedFrames.fetch_add(dropped, std::memory_order_relaxed);
	if (accepted == 0U) return;

	OaU64 epoch = impl->EpochUs.load(std::memory_order_relaxed);
	if (epoch == 0U) {
		const OaU64 durationUs = static_cast<OaU64>(InFrameCount) * 1'000'000ULL
			/ impl->Config.SampleRate;
		epoch = MonotonicMicroseconds() - durationUs;
		impl->EpochUs.store(epoch, std::memory_order_release);
	}

	const OaF32* input = static_cast<const OaF32*>(InInput);
	const OaU64 channels = impl->Config.ChannelCount;
	const OaU64 first = write % impl->CapacityFrames;
	const OaU64 firstFrames = std::min(accepted, impl->CapacityFrames - first);
	std::memcpy(
		impl->Ring.Data() + first * channels,
		input,
		static_cast<OaUsize>(firstFrames * channels * sizeof(OaF32)));
	if (firstFrames < accepted) {
		std::memcpy(
			impl->Ring.Data(),
			input + firstFrames * channels,
			static_cast<OaUsize>((accepted - firstFrames) * channels * sizeof(OaF32)));
	}
	for (OaU64 frame = 0U; frame < accepted; ++frame) {
		impl->RingFrameIndices[(write + frame) % impl->CapacityFrames]
			= capturedFirst + frame;
	}
	impl->WriteFrame.store(write + accepted, std::memory_order_release);
}

} // namespace

OaAudioCapture::OaAudioCapture(OaAudioCapture&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_))
{}

OaAudioCapture& OaAudioCapture::operator=(OaAudioCapture&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}

OaAudioCapture::~OaAudioCapture()
{
	Abandon_();
}

void OaAudioCapture::Abandon_() noexcept
{
	if (not Impl_) return;
	OaEngine* engine = Impl_->Engine;
	if (engine == nullptr) {
		Impl_.Reset();
		return;
	}
	Impl_->AcceptingCallbacks.store(false, std::memory_order_release);
	auto retired = OaMakeUniquePtr<OaAudioCapture>(OaStdMove(*this));
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retired.Release(),
		&OaAudioCapture::CompleteRetired_,
		&OaAudioCapture::ReleaseRetired_);
}

OaStatus OaAudioCapture::CompleteRetired_(void* InPayload)
{
	auto* capture = static_cast<OaAudioCapture*>(InPayload);
	return capture ? capture->Close() : OaStatus::Ok();
}

void OaAudioCapture::ReleaseRetired_(void* InPayload)
{
	OaUniquePtr<OaAudioCapture> capture(
		static_cast<OaAudioCapture*>(InPayload));
}

OaResult<OaAudioCapture> OaAudioCapture::Open(
	OaEngine& InEngine,
	const OaAudioCaptureConfig& InConfig)
{
	if (InConfig.SampleRate == 0U || InConfig.ChannelCount == 0U
		|| InConfig.ChannelCount > 8U || InConfig.RingMilliseconds < 20U) {
		return OaStatus::InvalidArgument(
			"OaAudioCapture requires a sample rate, 1..8 channels and at least 20 ms of ring storage");
	}
	OaAudioCapture capture;
	capture.Impl_ = OaMakeUniquePtr<Impl>();
	auto& impl = *capture.Impl_;
	impl.Engine = &InEngine;
	impl.Config = InConfig;
	impl.CapacityFrames = std::max<OaU64>(
		1U, static_cast<OaU64>(InConfig.SampleRate) * InConfig.RingMilliseconds / 1000U);
	impl.Ring.Resize(static_cast<OaUsize>(impl.CapacityFrames * InConfig.ChannelCount));
	impl.RingFrameIndices.Resize(static_cast<OaUsize>(impl.CapacityFrames));

	ma_device_config config = ma_device_config_init(ma_device_type_capture);
	config.capture.format = ma_format_f32;
	config.capture.channels = InConfig.ChannelCount;
	config.sampleRate = InConfig.SampleRate;
	config.dataCallback = CaptureCallback;
	config.pUserData = &impl;
	const ma_result result = ma_device_init(nullptr, &config, &impl.Device);
	if (result != MA_SUCCESS) {
		capture.Impl_.Reset();
		return OaStatus::Error(OaStatusCode::Unavailable,
			OaString("OaAudioCapture device open failed: ") + ma_result_description(result));
	}
	impl.Initialized = true;
	return capture;
}

OaStatus OaAudioCapture::Start()
{
	if (!Impl_ || !Impl_->Initialized) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaAudioCapture is not open");
	}
	if (Impl_->Started.load(std::memory_order_acquire)) return OaStatus::Ok();
	Impl_->WriteFrame.store(0U, std::memory_order_relaxed);
	Impl_->ReadFrame.store(0U, std::memory_order_relaxed);
	Impl_->CapturedFrame.store(0U, std::memory_order_relaxed);
	Impl_->DroppedFrames.store(0U, std::memory_order_relaxed);
	Impl_->EpochUs.store(0U, std::memory_order_release);
	Impl_->AcceptingCallbacks.store(true, std::memory_order_release);
	const ma_result result = ma_device_start(&Impl_->Device);
	if (result != MA_SUCCESS) {
		Impl_->AcceptingCallbacks.store(false, std::memory_order_release);
		return OaStatus::Error(OaStatusCode::Unavailable,
			OaString("OaAudioCapture start failed: ") + ma_result_description(result));
	}
	Impl_->Started.store(true, std::memory_order_release);
	return OaStatus::Ok();
}

OaStatus OaAudioCapture::Stop()
{
	if (!Impl_ || !Impl_->Initialized) return OaStatus::Ok();
	Impl_->AcceptingCallbacks.store(false, std::memory_order_release);
	if (!Impl_->Started.exchange(false, std::memory_order_acq_rel)) return OaStatus::Ok();
	const ma_result result = ma_device_stop(&Impl_->Device);
	if (result != MA_SUCCESS) {
		return OaStatus::Error(OaStatusCode::Internal,
			OaString("OaAudioCapture stop failed: ") + ma_result_description(result));
	}
	return OaStatus::Ok();
}

bool OaAudioCapture::Poll(OaAudioCaptureChunk& OutChunk, OaU32 InMaxFrames)
{
	if (!Impl_ || InMaxFrames == 0U) return false;
	const OaU64 read = Impl_->ReadFrame.load(std::memory_order_relaxed);
	const OaU64 write = Impl_->WriteFrame.load(std::memory_order_acquire);
	OaU64 count = std::min<OaU64>(write - read, InMaxFrames);
	if (count == 0U) return false;
	const OaU64 physicalFirst = Impl_->RingFrameIndices[read % Impl_->CapacityFrames];
	// Dropped frames create a real clock gap. Do not merge samples across it:
	// returning the next run separately lets the recorder insert exact silence.
	for (OaU64 frame = 1U; frame < count; ++frame) {
		if (Impl_->RingFrameIndices[(read + frame) % Impl_->CapacityFrames]
			!= physicalFirst + frame) {
			count = frame;
			break;
		}
	}
	const OaU64 channels = Impl_->Config.ChannelCount;
	OutChunk = {};
	OutChunk.Interleaved.Resize(static_cast<OaUsize>(count * channels));
	const OaU64 first = read % Impl_->CapacityFrames;
	const OaU64 firstFrames = std::min(count, Impl_->CapacityFrames - first);
	std::memcpy(
		OutChunk.Interleaved.Data(),
		Impl_->Ring.Data() + first * channels,
		static_cast<OaUsize>(firstFrames * channels * sizeof(OaF32)));
	if (firstFrames < count) {
		std::memcpy(
			OutChunk.Interleaved.Data() + firstFrames * channels,
			Impl_->Ring.Data(),
			static_cast<OaUsize>((count - firstFrames) * channels * sizeof(OaF32)));
	}
	OutChunk.SampleRate = Impl_->Config.SampleRate;
	OutChunk.ChannelCount = Impl_->Config.ChannelCount;
	OutChunk.FrameCount = count;
	OutChunk.FirstFrameIndex = physicalFirst;
	const OaU64 epoch = Impl_->EpochUs.load(std::memory_order_acquire);
	OutChunk.PresentationTimestamp = epoch
		+ physicalFirst * 1'000'000ULL / Impl_->Config.SampleRate;
	Impl_->ReadFrame.store(read + count, std::memory_order_release);
	return true;
}

OaStatus OaAudioCapture::Close() {
	if (not Impl_) return OaStatus::Ok();
	const OaStatus status = Stop();
	if (Impl_->Initialized) ma_device_uninit(&Impl_->Device);
	Impl_.Reset();
	return status;
}

void OaAudioCapture::Destroy()
{
	if (const auto status = Close(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaAudioCapture::Destroy: shutdown failed: %s",
			status.ToString().c_str());
	}
}

bool OaAudioCapture::IsStarted() const noexcept
{
	return Impl_ && Impl_->Started.load(std::memory_order_acquire);
}

OaU64 OaAudioCapture::DroppedFrameCount() const noexcept
{
	return Impl_ ? Impl_->DroppedFrames.load(std::memory_order_relaxed) : 0U;
}
