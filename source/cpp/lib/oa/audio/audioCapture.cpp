#include <oa/audio/audioCapture.h>

#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

namespace {

oa::U64 monotonicMicroseconds()
{
	return static_cast<oa::U64>(std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

struct oa::AudioCapture::Impl {
	oa::Engine* engine = nullptr;
	ma_device device = {};
	oa::AudioCaptureConfig config = {};
	oa::Vec<oa::F32> ring;
	oa::Vec<oa::U64> ringFrameIndices;
	oa::U64 capacityFrames = 0U;
	std::atomic<oa::U64> writeFrame = 0U;
	std::atomic<oa::U64> readFrame = 0U;
	std::atomic<oa::U64> capturedFrame = 0U;
	std::atomic<oa::U64> droppedFrames = 0U;
	std::atomic<oa::U64> epochUs = 0U;
	std::atomic<bool> started = false;
	std::atomic<bool> acceptingCallbacks = false;
	bool initialized = false;
};

namespace {

void captureCallback(
	ma_device* inDevice,
	void*,
	const void* inInput,
	ma_uint32 inFrameCount)
{
	auto* impl = static_cast<oa::AudioCapture::Impl*>(inDevice->pUserData);
	if (impl == nullptr || inInput == nullptr || inFrameCount == 0U) return;
	if (not impl->acceptingCallbacks.load(std::memory_order_acquire)) return;
	const oa::U64 write = impl->writeFrame.load(std::memory_order_relaxed);
	const oa::U64 read = impl->readFrame.load(std::memory_order_acquire);
	const oa::U64 freeFrames = impl->capacityFrames - std::min(impl->capacityFrames, write - read);
	const oa::U64 accepted = std::min<oa::U64>(freeFrames, inFrameCount);
	const oa::U64 dropped = static_cast<oa::U64>(inFrameCount) - accepted;
	const oa::U64 capturedFirst = impl->capturedFrame.fetch_add(inFrameCount, std::memory_order_relaxed);
	if (dropped > 0U) impl->droppedFrames.fetch_add(dropped, std::memory_order_relaxed);
	if (accepted == 0U) return;

	oa::U64 epoch = impl->epochUs.load(std::memory_order_relaxed);
	if (epoch == 0U) {
		const oa::U64 durationUs = static_cast<oa::U64>(inFrameCount) * 1'000'000ULL
			/ impl->config.sampleRate;
		epoch = monotonicMicroseconds() - durationUs;
		impl->epochUs.store(epoch, std::memory_order_release);
	}

	const oa::F32* input = static_cast<const oa::F32*>(inInput);
	const oa::U64 channels = impl->config.channelCount;
	const oa::U64 first = write % impl->capacityFrames;
	const oa::U64 firstFrames = std::min(accepted, impl->capacityFrames - first);
	std::memcpy(
		impl->ring.data() + first * channels,
		input,
		static_cast<oa::Usize>(firstFrames * channels * sizeof(oa::F32)));
	if (firstFrames < accepted) {
		std::memcpy(
			impl->ring.data(),
			input + firstFrames * channels,
			static_cast<oa::Usize>((accepted - firstFrames) * channels * sizeof(oa::F32)));
	}
	for (oa::U64 frame = 0U; frame < accepted; ++frame) {
		impl->ringFrameIndices[(write + frame) % impl->capacityFrames]
			= capturedFirst + frame;
	}
	impl->writeFrame.store(write + accepted, std::memory_order_release);
}

} // namespace

oa::AudioCapture::AudioCapture(oa::AudioCapture&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
{}

oa::AudioCapture& oa::AudioCapture::operator=(oa::AudioCapture&& inOther) noexcept
{
	if (this != &inOther) {
		abandon_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}

oa::AudioCapture::~AudioCapture()
{
	abandon_();
}

void oa::AudioCapture::abandon_() noexcept
{
	if (not impl_) return;
	oa::Engine* engine = impl_->engine;
	if (engine == nullptr) {
		impl_.reset();
		return;
	}
	impl_->acceptingCallbacks.store(false, std::memory_order_release);
	auto retired = oa::makeUnique<oa::AudioCapture>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::AudioCapture::completeRetired_,
		&oa::AudioCapture::releaseRetired_);
}

oa::Status oa::AudioCapture::completeRetired_(void* inPayload)
{
	auto* capture = static_cast<oa::AudioCapture*>(inPayload);
	return capture ? capture->close() : oa::Status::ok();
}

void oa::AudioCapture::releaseRetired_(void* inPayload)
{
	oa::UniquePtr<oa::AudioCapture> capture(
		static_cast<oa::AudioCapture*>(inPayload));
}

oa::Result<oa::AudioCapture> oa::AudioCapture::open(
	oa::Engine& inEngine,
	const oa::AudioCaptureConfig& inConfig)
{
	if (inConfig.sampleRate == 0U || inConfig.channelCount == 0U
		|| inConfig.channelCount > 8U || inConfig.ringMilliseconds < 20U) {
		return oa::Status::invalidArgument(
			"oa::AudioCapture requires a sample rate, 1..8 channels and at least 20 ms of ring storage");
	}
	oa::AudioCapture capture;
	capture.impl_ = oa::makeUnique<Impl>();
	auto& impl = *capture.impl_;
	impl.engine = &inEngine;
	impl.config = inConfig;
	impl.capacityFrames = std::max<oa::U64>(
		1U, static_cast<oa::U64>(inConfig.sampleRate) * inConfig.ringMilliseconds / 1000U);
	impl.ring.resize(static_cast<oa::Usize>(impl.capacityFrames * inConfig.channelCount));
	impl.ringFrameIndices.resize(static_cast<oa::Usize>(impl.capacityFrames));

	ma_device_config config = ma_device_config_init(ma_device_type_capture);
	config.capture.format = ma_format_f32;
	config.capture.channels = inConfig.channelCount;
	config.sampleRate = inConfig.sampleRate;
	config.dataCallback = captureCallback;
	config.pUserData = &impl;
	const ma_result result = ma_device_init(nullptr, &config, &impl.device);
	if (result != MA_SUCCESS) {
		capture.impl_.reset();
		return oa::Status::error(oa::StatusCode::Unavailable,
			oa::String("oa::AudioCapture device open failed: ") + ma_result_description(result));
	}
	impl.initialized = true;
	return capture;
}

oa::Status oa::AudioCapture::start()
{
	if (!impl_ || !impl_->initialized) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::AudioCapture is not open");
	}
	if (impl_->started.load(std::memory_order_acquire)) return oa::Status::ok();
	impl_->writeFrame.store(0U, std::memory_order_relaxed);
	impl_->readFrame.store(0U, std::memory_order_relaxed);
	impl_->capturedFrame.store(0U, std::memory_order_relaxed);
	impl_->droppedFrames.store(0U, std::memory_order_relaxed);
	impl_->epochUs.store(0U, std::memory_order_release);
	impl_->acceptingCallbacks.store(true, std::memory_order_release);
	const ma_result result = ma_device_start(&impl_->device);
	if (result != MA_SUCCESS) {
		impl_->acceptingCallbacks.store(false, std::memory_order_release);
		return oa::Status::error(oa::StatusCode::Unavailable,
			oa::String("oa::AudioCapture start failed: ") + ma_result_description(result));
	}
	impl_->started.store(true, std::memory_order_release);
	return oa::Status::ok();
}

oa::Status oa::AudioCapture::stop()
{
	if (!impl_ || !impl_->initialized) return oa::Status::ok();
	impl_->acceptingCallbacks.store(false, std::memory_order_release);
	if (!impl_->started.exchange(false, std::memory_order_acq_rel)) return oa::Status::ok();
	const ma_result result = ma_device_stop(&impl_->device);
	if (result != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("oa::AudioCapture stop failed: ") + ma_result_description(result));
	}
	return oa::Status::ok();
}

bool oa::AudioCapture::poll(oa::AudioCaptureChunk& outChunk, oa::U32 inMaxFrames) {
	if (!impl_ || inMaxFrames == 0U) return false;
	const oa::U64 read = impl_->readFrame.load(std::memory_order_relaxed);
	const oa::U64 write = impl_->writeFrame.load(std::memory_order_acquire);
	oa::U64 count = std::min<oa::U64>(write - read, inMaxFrames);
	if (count == 0U) return false;
	const oa::U64 physicalFirst = impl_->ringFrameIndices[read % impl_->capacityFrames];
	// Dropped frames create a real clock gap. Do not merge samples across it:
	// returning the next run separately lets the recorder insert exact silence.
	for (oa::U64 frame = 1U; frame < count; ++frame) {
		if (impl_->ringFrameIndices[(read + frame) % impl_->capacityFrames]
			!= physicalFirst + frame) {
			count = frame;
			break;
		}
	}
	const oa::U64 channels = impl_->config.channelCount;
	outChunk = {};
	outChunk.interleaved.resize(static_cast<oa::Usize>(count * channels));
	const oa::U64 first = read % impl_->capacityFrames;
	const oa::U64 firstFrames = std::min(count, impl_->capacityFrames - first);
	std::memcpy(
		outChunk.interleaved.data(),
		impl_->ring.data() + first * channels,
		static_cast<oa::Usize>(firstFrames * channels * sizeof(oa::F32)));
	if (firstFrames < count) {
		std::memcpy(
			outChunk.interleaved.data() + firstFrames * channels,
			impl_->ring.data(),
			static_cast<oa::Usize>((count - firstFrames) * channels * sizeof(oa::F32)));
	}
	outChunk.sampleRate = impl_->config.sampleRate;
	outChunk.channelCount = impl_->config.channelCount;
	outChunk.frameCount = count;
	outChunk.firstFrameIndex = physicalFirst;
	const oa::U64 epoch = impl_->epochUs.load(std::memory_order_acquire);
	outChunk.presentationTimestamp = epoch
		+ physicalFirst * 1'000'000ULL / impl_->config.sampleRate;
	impl_->readFrame.store(read + count, std::memory_order_release);
	return true;
}

oa::Status oa::AudioCapture::close() {
	if (not impl_) return oa::Status::ok();
	const oa::Status status = stop();
	if (impl_->initialized) ma_device_uninit(&impl_->device);
	impl_.reset();
	return status;
}

bool oa::AudioCapture::isStarted() const noexcept
{
	return impl_ && impl_->started.load(std::memory_order_acquire);
}

oa::U64 oa::AudioCapture::droppedFrameCount() const noexcept
{
	return impl_ ? impl_->droppedFrames.load(std::memory_order_relaxed) : 0U;
}
