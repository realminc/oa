#include <oa/audio/audioPlayer.h>

#include <oa/runtime/engine.h>
#include "../core/logAccess.h"
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <miniaudio.h>


struct oa::AudioPlayer::Impl {
	oa::Engine* engine = nullptr;
	oa::AudioPlayerConfig config;
	oa::Vec<oa::F32> ring;
	oa::U64 capacityFrames = 0U;
	oa::Atomic<oa::U64> writeFrame{0U};
	oa::Atomic<oa::U64> readFrame{0U};
	oa::Atomic<oa::U64> positionFrame{0U};
	oa::Atomic<oa::U64> underrunFrames{0U};
	oa::Atomic<bool> playing{false};
	oa::Atomic<bool> eos{false};
	oa::Atomic<bool> stop{false};
	oa::Atomic<bool> loop{false};
	oa::Atomic<oa::I64> seekRequestUs{-1};
	oa::Atomic<oa::U64> seekSerial{0U};
	oa::Atomic<oa::U64> appliedSeekSerial{0U};
	oa::Mutex wakeMutex;
	oa::Condition wake;
	oa::Thread decodeThread;
	ma_device device = {};
	bool deviceInitialized = false;
	oa::U32 sampleRate = 0U;
	oa::U32 channels = 0U;
	oa::U64 duration = 0U;
	ma_decoder decoder = {};
	bool decoderInitialized = false;
};

namespace {

void playbackCallback(ma_device* inDevice, void* out, const void*, ma_uint32 inFrames) {
	auto* impl = static_cast<oa::AudioPlayer::Impl*>(inDevice->pUserData);
	auto* output = static_cast<oa::F32*>(out);
	const oa::U64 samples = static_cast<oa::U64>(inFrames) * impl->channels;
	oa::memzero(output, static_cast<oa::Usize>(samples * sizeof(oa::F32)));
	if (not impl->playing.load(oa::MemoryOrder::Acquire)) return;

	const oa::U64 read = impl->readFrame.load(oa::MemoryOrder::Relaxed);
	const oa::U64 write = impl->writeFrame.load(oa::MemoryOrder::Acquire);
	const oa::U64 count = oa::min<oa::U64>(write - read, inFrames);
	const oa::U64 first = read % impl->capacityFrames;
	const oa::U64 firstFrames = oa::min(count, impl->capacityFrames - first);
	oa::memcpy(output,
		impl->ring.data() + first * impl->channels,
		static_cast<oa::Usize>(firstFrames * impl->channels * sizeof(oa::F32)));
	if (firstFrames < count) {
		oa::memcpy(output + firstFrames * impl->channels,
			impl->ring.data(),
			static_cast<oa::Usize>((count - firstFrames) * impl->channels * sizeof(oa::F32)));
	}
	impl->readFrame.store(read + count, oa::MemoryOrder::Release);
	impl->positionFrame.fetchAdd(count, oa::MemoryOrder::Relaxed);
	if (count < inFrames) {
		impl->underrunFrames.fetchAdd(inFrames - count, oa::MemoryOrder::Relaxed);
	}
	impl->wake.notifyOne();
}

bool pushFrames(oa::AudioPlayer::Impl& inImpl, const oa::F32* inSamples, oa::U64 inFrames) {
	oa::U64 consumed = 0U;
	while (consumed < inFrames and not inImpl.stop.load(oa::MemoryOrder::Acquire)) {
		if (inImpl.seekSerial.load(oa::MemoryOrder::Acquire)
			!= inImpl.appliedSeekSerial.load(oa::MemoryOrder::Relaxed)) {
			return false;
		}
		const oa::U64 write = inImpl.writeFrame.load(oa::MemoryOrder::Relaxed);
		const oa::U64 read = inImpl.readFrame.load(oa::MemoryOrder::Acquire);
		const oa::U64 free = inImpl.capacityFrames
			- oa::min(inImpl.capacityFrames, write - read);
		if (free == 0U) {
			oa::UniqueLock<oa::Mutex> lock(inImpl.wakeMutex);
			(void)inImpl.wake.waitFor(lock, oa::Duration::fromMilliseconds(2));
			continue;
		}
		const oa::U64 count = oa::min(free, inFrames - consumed);
		const oa::U64 first = write % inImpl.capacityFrames;
		const oa::U64 firstFrames = oa::min(count, inImpl.capacityFrames - first);
		oa::memcpy(inImpl.ring.data() + first * inImpl.channels,
			inSamples + consumed * inImpl.channels,
			static_cast<oa::Usize>(firstFrames * inImpl.channels * sizeof(oa::F32)));
		if (firstFrames < count) {
			oa::memcpy(inImpl.ring.data(),
				inSamples + (consumed + firstFrames) * inImpl.channels,
				static_cast<oa::Usize>((count - firstFrames) * inImpl.channels * sizeof(oa::F32)));
		}
		inImpl.writeFrame.store(write + count, oa::MemoryOrder::Release);
		consumed += count;
	}
	return consumed == inFrames;
}

bool seekDecoder(oa::AudioPlayer::Impl& inImpl, oa::U64 inTimestampUs) {
	if (inImpl.sampleRate == 0U) return false;
	const oa::U64 frame = inTimestampUs > oa::Limits<oa::U64>::max() / inImpl.sampleRate
		? oa::Limits<oa::U64>::max()
		: inTimestampUs * inImpl.sampleRate / 1'000'000ULL;
	if (ma_decoder_seek_to_pcm_frame(&inImpl.decoder, frame) != MA_SUCCESS) return false;
	inImpl.readFrame.store(0U, oa::MemoryOrder::Relaxed);
	inImpl.writeFrame.store(0U, oa::MemoryOrder::Relaxed);
	inImpl.positionFrame.store(frame, oa::MemoryOrder::Relaxed);
	inImpl.eos.store(false, oa::MemoryOrder::Release);
	return true;
}

void decodeLoop(oa::AudioPlayer::Impl* inImpl) {
	constexpr oa::U64 kDecodeFrames = 4096U;
	oa::Vec<oa::F32> decoded;
	decoded.resize(static_cast<oa::Usize>(kDecodeFrames * inImpl->channels));
	while (not inImpl->stop.load(oa::MemoryOrder::Acquire)) {
		const oa::U64 requestSerial = inImpl->seekSerial.load(oa::MemoryOrder::Acquire);
		if (requestSerial != inImpl->appliedSeekSerial.load(oa::MemoryOrder::Relaxed)) {
			const oa::I64 requested = inImpl->seekRequestUs.load(oa::MemoryOrder::Relaxed);
			(void)seekDecoder(*inImpl, static_cast<oa::U64>(oa::max<oa::I64>(0, requested)));
			inImpl->appliedSeekSerial.store(requestSerial, oa::MemoryOrder::Release);
			inImpl->wake.notifyAll();
		}
		if (not inImpl->playing.load(oa::MemoryOrder::Acquire)) {
			oa::UniqueLock<oa::Mutex> lock(inImpl->wakeMutex);
			(void)inImpl->wake.waitFor(lock, oa::Duration::fromMilliseconds(10));
			continue;
		}
		ma_uint64 frameCount = 0U;
		const ma_result readResult = ma_decoder_read_pcm_frames(&inImpl->decoder, decoded.data(), kDecodeFrames, &frameCount);
		if (frameCount > 0U and not pushFrames(*inImpl, decoded.data(), frameCount)) {
			continue;
		}
		if (readResult == MA_AT_END or frameCount == 0U) {
			if (inImpl->loop.load(oa::MemoryOrder::Acquire)) {
				(void)seekDecoder(*inImpl, 0U);
				continue;
			}
			while (
				not inImpl->stop.load(oa::MemoryOrder::Acquire)
				and inImpl->readFrame.load(oa::MemoryOrder::Acquire)
					< inImpl->writeFrame.load(oa::MemoryOrder::Acquire)
				) {
				if (inImpl->seekSerial.load(oa::MemoryOrder::Acquire)
					!= inImpl->appliedSeekSerial.load(oa::MemoryOrder::Relaxed)) {
					break;
				}
				oa::UniqueLock<oa::Mutex> lock(inImpl->wakeMutex);
				(void)inImpl->wake.waitFor(lock, oa::Duration::fromMilliseconds(2));
			}
			if (inImpl->seekSerial.load(oa::MemoryOrder::Acquire)
				!= inImpl->appliedSeekSerial.load(oa::MemoryOrder::Relaxed)) {
				continue;
			}
			inImpl->eos.store(true, oa::MemoryOrder::Release);
			inImpl->playing.store(false, oa::MemoryOrder::Release);
			continue;
		}
		if (readResult != MA_SUCCESS) {
			inImpl->eos.store(true, oa::MemoryOrder::Release);
			inImpl->playing.store(false, oa::MemoryOrder::Release);
		}
	}
}

} // namespace

oa::AudioPlayer::AudioPlayer(oa::AudioPlayer&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
{}

oa::AudioPlayer& oa::AudioPlayer::operator=(oa::AudioPlayer&& inOther) noexcept {
	if (this != &inOther) {
		(void)close();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}

oa::AudioPlayer::~AudioPlayer() { abandon_(); }

void oa::AudioPlayer::abandon_() noexcept {
	if (not impl_) return;
	oa::Engine* engine = impl_->engine;
	if (engine == nullptr) {
		impl_.reset();
		return;
	}
	impl_->stop.store(true, oa::MemoryOrder::Release);
	impl_->playing.store(false, oa::MemoryOrder::Release);
	impl_->wake.notifyAll();
	auto retired = oa::makeUnique<oa::AudioPlayer>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::AudioPlayer::completeRetired_,
		&oa::AudioPlayer::releaseRetired_);
}

oa::Status oa::AudioPlayer::completeRetired_(void* inPayload) {
	auto* stream = static_cast<oa::AudioPlayer*>(inPayload);
	return stream ? stream->close() : oa::Status::ok();
}

void oa::AudioPlayer::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::AudioPlayer> stream(static_cast<oa::AudioPlayer*>(inPayload));
}

oa::Result<oa::AudioPlayer> oa::AudioPlayer::open(
	oa::Engine& inEngine,
	const oa::AudioPlayerConfig& inConfig
) {
	if (inConfig.uri.empty() || inConfig.ringMilliseconds < 40U) {
		return oa::Status::invalidArgument(
			"oa::AudioPlayer requires a URI and at least 40 ms of ring storage");
	}
	oa::AudioPlayer stream;
	stream.impl_ = oa::makeUnique<Impl>();
	auto& impl = *stream.impl_;
	impl.engine = &inEngine;
	impl.config = inConfig;
	impl.loop = inConfig.loop;
	ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 0U, 0U);
	if (ma_decoder_init_file(inConfig.uri.cStr(), &decoderConfig, &impl.decoder) != MA_SUCCESS) {
		(void)stream.close();
		return oa::Status::error(oa::StatusCode::Unavailable,	"oa::AudioPlayer could not open WAV, FLAC or MP3 source");
	}
	impl.decoderInitialized = true;
	ma_format outputFormat = ma_format_unknown;
	ma_uint32 outputChannels = 0U;
	ma_uint32 outputSampleRate = 0U;
	if (ma_decoder_get_data_format(&impl.decoder, &outputFormat, &outputChannels,
		&outputSampleRate, nullptr, 0U) != MA_SUCCESS
		or outputFormat != ma_format_f32) {
		(void)stream.close();
		return oa::Status::error("oa::AudioPlayer could not query the native decoder format");
	}
	impl.sampleRate = outputSampleRate;
	impl.channels = outputChannels;
	if (impl.sampleRate == 0U || impl.channels == 0U || impl.channels > 8U) {
		(void)stream.close();
		return oa::Status::error(oa::StatusCode::Unimplemented,	"oa::AudioPlayer supports 1..8 channel streams with a declared sample rate");
	}
	ma_uint64 totalFrames = 0U;
	if (ma_decoder_get_length_in_pcm_frames(&impl.decoder, &totalFrames) == MA_SUCCESS
		and totalFrames <= oa::Limits<oa::U64>::max() / 1'000'000ULL) {
		impl.duration = totalFrames * 1'000'000ULL / impl.sampleRate;
	}
	impl.capacityFrames = oa::max<oa::U64>(1U, static_cast<oa::U64>(impl.sampleRate) * inConfig.ringMilliseconds / 1000U);
	impl.ring.resize(static_cast<oa::Usize>(impl.capacityFrames * impl.channels));

	ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
	deviceConfig.playback.format = ma_format_f32;
	deviceConfig.playback.channels = impl.channels;
	deviceConfig.sampleRate = impl.sampleRate;
	deviceConfig.dataCallback = playbackCallback;
	deviceConfig.pUserData = &impl;
	if (ma_device_init(nullptr, &deviceConfig, &impl.device) != MA_SUCCESS) {
		(void)stream.close();
		return oa::Status::error(oa::StatusCode::Unavailable,	"oa::AudioPlayer could not open the playback device");
	}
	impl.deviceInitialized = true;
	if (ma_device_start(&impl.device) != MA_SUCCESS) {
		(void)stream.close();
		return oa::Status::error(oa::StatusCode::Unavailable,	"oa::AudioPlayer could not start the playback device");
	}
	const oa::LogSelection logSelection = oa::LogAccess::currentSelection();
	auto decodeThread = oa::Thread::create([&impl, logSelection] {
		oa::LogAccess::Scope logScope(logSelection);
		decodeLoop(&impl);
	});
	if (decodeThread.isError()) {
		(void)stream.close();
		return decodeThread.getStatus();
	}
	impl.decodeThread = oa::move(*decodeThread);
	return stream;
}

oa::Result<oa::AudioPlayer> oa::AudioPlayer::open(
	oa::Engine& inEngine,
	oa::StringView inUri
) {
	oa::AudioPlayerConfig config;
	config.uri = oa::String(inUri);
	return open(inEngine, config);
}

oa::Status oa::AudioPlayer::play() {
	if (not impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::AudioPlayer::play called on a closed stream");
	if (impl_->eos.load(oa::MemoryOrder::Acquire)) {
		OA_RETURN_IF_ERROR(seek(0U));
	}
	impl_->eos.store(false, oa::MemoryOrder::Release);
	impl_->playing.store(true, oa::MemoryOrder::Release);
	impl_->wake.notifyAll();
	return oa::Status::ok();
}

void oa::AudioPlayer::pause() {
	if (impl_) impl_->playing.store(false, oa::MemoryOrder::Release);
}

oa::Status oa::AudioPlayer::seek(oa::U64 inTimestampUs) {
	if (not impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::AudioPlayer::seek called on a closed stream");
	// The decoder resets both monotonic SPSC indices when it applies a seek.
	// Stop the device first so its callback cannot observe one reset index and
	// one old index (which would look like an unsigned ring overflow).
	const bool resume = impl_->playing.exchange(false, oa::MemoryOrder::AcquireRelease);
	if (impl_->deviceInitialized and ma_device_stop(&impl_->device) != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"oa::AudioPlayer could not stop playback for seek");
	}
	impl_->seekRequestUs.store(static_cast<oa::I64>(inTimestampUs), oa::MemoryOrder::Release);
	const oa::U64 serial = impl_->seekSerial.fetchAdd(
		1U, oa::MemoryOrder::AcquireRelease) + 1U;
	impl_->wake.notifyAll();
	oa::UniqueLock<oa::Mutex> lock(impl_->wakeMutex);
	const bool applied = impl_->wake.waitFor(
		lock, oa::Duration::fromSeconds(2), [&] {
		return impl_->appliedSeekSerial.load(oa::MemoryOrder::Acquire) >= serial;
	});
	if (impl_->deviceInitialized and ma_device_start(&impl_->device) != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"oa::AudioPlayer could not restart playback after seek");
	}
	if (applied and resume) {
		impl_->playing.store(true, oa::MemoryOrder::Release);
		impl_->wake.notifyAll();
	}
	return applied ? oa::Status::ok()
		: oa::Status::error(oa::StatusCode::DeadlineExceeded, "oa::AudioPlayer seek timed out");
}

void oa::AudioPlayer::setLoop(bool inLoop) {
	if (impl_) impl_->loop.store(inLoop, oa::MemoryOrder::Release);
}

oa::Status oa::AudioPlayer::close() {
	if (not impl_) return oa::Status::ok();
	impl_->stop.store(true, oa::MemoryOrder::Release);
	impl_->playing.store(false, oa::MemoryOrder::Release);
	impl_->wake.notifyAll();
	oa::Status status = oa::Status::ok();
	if (impl_->deviceInitialized) {
		const ma_result result = ma_device_stop(&impl_->device);
		if (result != MA_SUCCESS) {
			status = oa::Status::error(oa::StatusCode::Internal,
				oa::String("oa::AudioPlayer stop failed: ")
					+ ma_result_description(result));
		}
	}
	if (impl_->decodeThread.joinable()) {
		const oa::Status joinStatus = impl_->decodeThread.join();
		if (status.isOk() and joinStatus.isError()) status = joinStatus;
	}
	if (impl_->deviceInitialized) ma_device_uninit(&impl_->device);
	if (impl_->decoderInitialized) ma_decoder_uninit(&impl_->decoder);
	impl_.reset();
	return status;
}

bool oa::AudioPlayer::isOpen() const noexcept { return impl_ != nullptr; }
bool oa::AudioPlayer::isPlaying() const noexcept { return impl_ and impl_->playing.load(); }
bool oa::AudioPlayer::isEos() const noexcept { return not impl_ or impl_->eos.load(); }
oa::U32 oa::AudioPlayer::sampleRate() const noexcept { return impl_ ? impl_->sampleRate : 0U; }
oa::U32 oa::AudioPlayer::channelCount() const noexcept { return impl_ ? impl_->channels : 0U; }
oa::U64 oa::AudioPlayer::durationUs() const noexcept { return impl_ ? impl_->duration : 0U; }
oa::U64 oa::AudioPlayer::positionUs() const noexcept {
	return impl_ and impl_->sampleRate > 0U
		? impl_->positionFrame.load(oa::MemoryOrder::Relaxed) * 1'000'000ULL / impl_->sampleRate
		: 0U;
}
oa::U64 oa::AudioPlayer::underrunFrameCount() const noexcept {
	return impl_ ? impl_->underrunFrames.load(oa::MemoryOrder::Relaxed) : 0U;
}
