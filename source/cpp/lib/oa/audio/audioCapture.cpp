#include <oa/audio/audioCapture.h>

#include <oa/core/log.h>
#include <oa/core/memory.h>
#include <oa/core/std/chrono.h>
#include <oa/runtime/engine.h>
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <miniaudio.h>

namespace {

oa::U64 monotonicMicroseconds()
{
	return static_cast<oa::U64>(oa::steadyNow().nanosecondsSinceEpoch() / 1'000LL);
}

} // namespace

struct oa::AudioCapture::Impl {
	oa::Engine* engine = nullptr;
	ma_device device = {};
	oa::AudioCaptureConfig config = {};
	oa::Vec<oa::F32> ring;
	oa::Vec<oa::U64> ringFrameIndices;
	oa::U64 capacityFrames = 0U;
	oa::Atomic<oa::U64> writeFrame{0U};
	oa::Atomic<oa::U64> readFrame{0U};
	oa::Atomic<oa::U64> capturedFrame{0U};
	oa::Atomic<oa::U64> droppedFrames{0U};
	oa::Atomic<oa::U64> epochUs{0U};
	oa::Atomic<oa::Bool> started{false};
	oa::Atomic<oa::Bool> acceptingCallbacks{false};
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
	if (not impl->acceptingCallbacks.load(oa::MemoryOrder::Acquire)) return;
	const oa::U64 write = impl->writeFrame.load(oa::MemoryOrder::Relaxed);
	const oa::U64 read = impl->readFrame.load(oa::MemoryOrder::Acquire);
	const oa::U64 freeFrames = impl->capacityFrames
		- oa::min(impl->capacityFrames, write - read);
	const oa::U64 accepted = oa::min<oa::U64>(freeFrames, inFrameCount);
	const oa::U64 dropped = static_cast<oa::U64>(inFrameCount) - accepted;
	const oa::U64 capturedFirst = impl->capturedFrame.fetchAdd(
		inFrameCount, oa::MemoryOrder::Relaxed);
	if (dropped > 0U) {
		impl->droppedFrames.fetchAdd(dropped, oa::MemoryOrder::Relaxed);
	}
	if (accepted == 0U) return;

	oa::U64 epoch = impl->epochUs.load(oa::MemoryOrder::Relaxed);
	if (epoch == 0U) {
		const oa::U64 durationUs = static_cast<oa::U64>(inFrameCount) * 1'000'000ULL
			/ impl->config.sampleRate;
		epoch = monotonicMicroseconds() - durationUs;
		impl->epochUs.store(epoch, oa::MemoryOrder::Release);
	}

	const oa::F32* input = static_cast<const oa::F32*>(inInput);
	const oa::U64 channels = impl->config.channelCount;
	const oa::U64 first = write % impl->capacityFrames;
	const oa::U64 firstFrames = oa::min(accepted, impl->capacityFrames - first);
	oa::memcpy(
		impl->ring.data() + first * channels,
		input,
		static_cast<oa::Usize>(firstFrames * channels * sizeof(oa::F32)));
	if (firstFrames < accepted) {
		oa::memcpy(
			impl->ring.data(),
			input + firstFrames * channels,
			static_cast<oa::Usize>((accepted - firstFrames) * channels * sizeof(oa::F32)));
	}
	for (oa::U64 frame = 0U; frame < accepted; ++frame) {
		impl->ringFrameIndices[(write + frame) % impl->capacityFrames]
			= capturedFirst + frame;
	}
	impl->writeFrame.store(write + accepted, oa::MemoryOrder::Release);
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
	impl_->acceptingCallbacks.store(false, oa::MemoryOrder::Release);
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
	impl.capacityFrames = oa::max<oa::U64>(
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
	if (impl_->started.load(oa::MemoryOrder::Acquire)) return oa::Status::ok();
	impl_->writeFrame.store(0U, oa::MemoryOrder::Relaxed);
	impl_->readFrame.store(0U, oa::MemoryOrder::Relaxed);
	impl_->capturedFrame.store(0U, oa::MemoryOrder::Relaxed);
	impl_->droppedFrames.store(0U, oa::MemoryOrder::Relaxed);
	impl_->epochUs.store(0U, oa::MemoryOrder::Release);
	impl_->acceptingCallbacks.store(true, oa::MemoryOrder::Release);
	const ma_result result = ma_device_start(&impl_->device);
	if (result != MA_SUCCESS) {
		impl_->acceptingCallbacks.store(false, oa::MemoryOrder::Release);
		return oa::Status::error(oa::StatusCode::Unavailable,
			oa::String("oa::AudioCapture start failed: ") + ma_result_description(result));
	}
	impl_->started.store(true, oa::MemoryOrder::Release);
	return oa::Status::ok();
}

oa::Status oa::AudioCapture::stop()
{
	if (!impl_ || !impl_->initialized) return oa::Status::ok();
	impl_->acceptingCallbacks.store(false, oa::MemoryOrder::Release);
	if (!impl_->started.exchange(false, oa::MemoryOrder::AcquireRelease)) {
		return oa::Status::ok();
	}
	const ma_result result = ma_device_stop(&impl_->device);
	if (result != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("oa::AudioCapture stop failed: ") + ma_result_description(result));
	}
	return oa::Status::ok();
}

bool oa::AudioCapture::poll(oa::AudioCaptureChunk& outChunk, oa::U32 inMaxFrames) {
	if (!impl_ || inMaxFrames == 0U) return false;
	const oa::U64 read = impl_->readFrame.load(oa::MemoryOrder::Relaxed);
	const oa::U64 write = impl_->writeFrame.load(oa::MemoryOrder::Acquire);
	oa::U64 count = oa::min<oa::U64>(write - read, inMaxFrames);
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
	const oa::U64 firstFrames = oa::min(count, impl_->capacityFrames - first);
	oa::memcpy(
		outChunk.interleaved.data(),
		impl_->ring.data() + first * channels,
		static_cast<oa::Usize>(firstFrames * channels * sizeof(oa::F32)));
	if (firstFrames < count) {
		oa::memcpy(
			outChunk.interleaved.data() + firstFrames * channels,
			impl_->ring.data(),
			static_cast<oa::Usize>((count - firstFrames) * channels * sizeof(oa::F32)));
	}
	outChunk.sampleRate = impl_->config.sampleRate;
	outChunk.channelCount = impl_->config.channelCount;
	outChunk.frameCount = count;
	outChunk.firstFrameIndex = physicalFirst;
	const oa::U64 epoch = impl_->epochUs.load(oa::MemoryOrder::Acquire);
	outChunk.presentationTimestamp = epoch
		+ physicalFirst * 1'000'000ULL / impl_->config.sampleRate;
	impl_->readFrame.store(read + count, oa::MemoryOrder::Release);
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
	return impl_ && impl_->started.load(oa::MemoryOrder::Acquire);
}

oa::U64 oa::AudioCapture::droppedFrameCount() const noexcept
{
	return impl_ ? impl_->droppedFrames.load(oa::MemoryOrder::Relaxed) : 0U;
}
