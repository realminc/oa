#include <oa/audio/audioPlayer.h>

#include <oa/runtime/engine.h>
#include "../core/logAccess.h"
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

struct oa::AudioPlayer::Impl {
	oa::Engine* engine = nullptr;
	oa::AudioPlayerConfig config;
	oa::Vec<oa::F32> ring;
	oa::U64 capacityFrames = 0U;
	std::atomic<oa::U64> writeFrame = 0U;
	std::atomic<oa::U64> readFrame = 0U;
	std::atomic<oa::U64> positionFrame = 0U;
	std::atomic<oa::U64> underrunFrames = 0U;
	std::atomic<bool> playing = false;
	std::atomic<bool> eos = false;
	std::atomic<bool> stop = false;
	std::atomic<bool> loop = false;
	std::atomic<oa::I64> seekRequestUs = -1;
	std::atomic<oa::U64> seekSerial = 0U;
	std::atomic<oa::U64> appliedSeekSerial = 0U;
	std::mutex wakeMutex;
	std::condition_variable wake;
	std::thread decodeThread;
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
	std::memset(output, 0, static_cast<oa::Usize>(samples * sizeof(oa::F32)));
	if (not impl->playing.load(std::memory_order_acquire)) return;

	const oa::U64 read = impl->readFrame.load(std::memory_order_relaxed);
	const oa::U64 write = impl->writeFrame.load(std::memory_order_acquire);
	const oa::U64 count = std::min<oa::U64>(write - read, inFrames);
	const oa::U64 first = read % impl->capacityFrames;
	const oa::U64 firstFrames = std::min(count, impl->capacityFrames - first);
	std::memcpy(output,
		impl->ring.data() + first * impl->channels,
		static_cast<oa::Usize>(firstFrames * impl->channels * sizeof(oa::F32)));
	if (firstFrames < count) {
		std::memcpy(output + firstFrames * impl->channels,
			impl->ring.data(),
			static_cast<oa::Usize>((count - firstFrames) * impl->channels * sizeof(oa::F32)));
	}
	impl->readFrame.store(read + count, std::memory_order_release);
	impl->positionFrame.fetch_add(count, std::memory_order_relaxed);
	if (count < inFrames) {
		impl->underrunFrames.fetch_add(inFrames - count, std::memory_order_relaxed);
	}
	impl->wake.notify_one();
}

bool pushFrames(oa::AudioPlayer::Impl& inImpl, const oa::F32* inSamples, oa::U64 inFrames) {
	oa::U64 consumed = 0U;
	while (consumed < inFrames and not inImpl.stop.load(std::memory_order_acquire)) {
		if (inImpl.seekSerial.load(std::memory_order_acquire)
			!= inImpl.appliedSeekSerial.load(std::memory_order_relaxed)) {
			return false;
		}
		const oa::U64 write = inImpl.writeFrame.load(std::memory_order_relaxed);
		const oa::U64 read = inImpl.readFrame.load(std::memory_order_acquire);
		const oa::U64 free = inImpl.capacityFrames
			- std::min(inImpl.capacityFrames, write - read);
		if (free == 0U) {
			std::unique_lock lock(inImpl.wakeMutex);
			inImpl.wake.wait_for(lock, std::chrono::milliseconds(2));
			continue;
		}
		const oa::U64 count = std::min(free, inFrames - consumed);
		const oa::U64 first = write % inImpl.capacityFrames;
		const oa::U64 firstFrames = std::min(count, inImpl.capacityFrames - first);
		std::memcpy(inImpl.ring.data() + first * inImpl.channels,
			inSamples + consumed * inImpl.channels,
			static_cast<oa::Usize>(firstFrames * inImpl.channels * sizeof(oa::F32)));
		if (firstFrames < count) {
			std::memcpy(inImpl.ring.data(),
				inSamples + (consumed + firstFrames) * inImpl.channels,
				static_cast<oa::Usize>((count - firstFrames) * inImpl.channels * sizeof(oa::F32)));
		}
		inImpl.writeFrame.store(write + count, std::memory_order_release);
		consumed += count;
	}
	return consumed == inFrames;
}

bool seekDecoder(oa::AudioPlayer::Impl& inImpl, oa::U64 inTimestampUs) {
	if (inImpl.sampleRate == 0U) return false;
	const oa::U64 frame = inTimestampUs > std::numeric_limits<oa::U64>::max() / inImpl.sampleRate
		? std::numeric_limits<oa::U64>::max()
		: inTimestampUs * inImpl.sampleRate / 1'000'000ULL;
	if (ma_decoder_seek_to_pcm_frame(&inImpl.decoder, frame) != MA_SUCCESS) return false;
	inImpl.readFrame.store(0U, std::memory_order_relaxed);
	inImpl.writeFrame.store(0U, std::memory_order_relaxed);
	inImpl.positionFrame.store(frame, std::memory_order_relaxed);
	inImpl.eos.store(false, std::memory_order_release);
	return true;
}

void decodeLoop(oa::AudioPlayer::Impl* inImpl) {
	constexpr oa::U64 kDecodeFrames = 4096U;
	oa::Vec<oa::F32> decoded;
	decoded.resize(static_cast<oa::Usize>(kDecodeFrames * inImpl->channels));
	while (not inImpl->stop.load(std::memory_order_acquire)) {
		const oa::U64 requestSerial = inImpl->seekSerial.load(std::memory_order_acquire);
		if (requestSerial != inImpl->appliedSeekSerial.load(std::memory_order_relaxed)) {
			const oa::I64 requested = inImpl->seekRequestUs.load(std::memory_order_relaxed);
			(void)seekDecoder(*inImpl, static_cast<oa::U64>(std::max<oa::I64>(0, requested)));
			inImpl->appliedSeekSerial.store(requestSerial, std::memory_order_release);
			inImpl->wake.notify_all();
		}
		if (not inImpl->playing.load(std::memory_order_acquire)) {
			std::unique_lock lock(inImpl->wakeMutex);
			inImpl->wake.wait_for(lock, std::chrono::milliseconds(10));
			continue;
		}
		ma_uint64 frameCount = 0U;
		const ma_result readResult = ma_decoder_read_pcm_frames(&inImpl->decoder, decoded.data(), kDecodeFrames, &frameCount);
		if (frameCount > 0U and not pushFrames(*inImpl, decoded.data(), frameCount)) {
			continue;
		}
		if (readResult == MA_AT_END or frameCount == 0U) {
			if (inImpl->loop.load(std::memory_order_acquire)) {
				(void)seekDecoder(*inImpl, 0U);
				continue;
			}
			while (
				not inImpl->stop.load(std::memory_order_acquire)
				and inImpl->readFrame.load(std::memory_order_acquire)
					< inImpl->writeFrame.load(std::memory_order_acquire)
				) {
				if (inImpl->seekSerial.load(std::memory_order_acquire)
					!= inImpl->appliedSeekSerial.load(std::memory_order_relaxed)) {
					break;
				}
				std::unique_lock lock(inImpl->wakeMutex);
				inImpl->wake.wait_for(lock, std::chrono::milliseconds(2));
			}
			if (inImpl->seekSerial.load(std::memory_order_acquire)
				!= inImpl->appliedSeekSerial.load(std::memory_order_relaxed)) {
				continue;
			}
			inImpl->eos.store(true, std::memory_order_release);
			inImpl->playing.store(false, std::memory_order_release);
			continue;
		}
		if (readResult != MA_SUCCESS) {
			inImpl->eos.store(true, std::memory_order_release);
			inImpl->playing.store(false, std::memory_order_release);
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
	impl_->stop.store(true, std::memory_order_release);
	impl_->playing.store(false, std::memory_order_release);
	impl_->wake.notify_all();
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
		and totalFrames <= std::numeric_limits<oa::U64>::max() / 1'000'000ULL) {
		impl.duration = totalFrames * 1'000'000ULL / impl.sampleRate;
	}
	impl.capacityFrames = std::max<oa::U64>(1U, static_cast<oa::U64>(impl.sampleRate) * inConfig.ringMilliseconds / 1000U);
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
	impl.decodeThread = std::thread([&impl, logSelection] {
		oa::LogAccess::Scope logScope(logSelection);
		decodeLoop(&impl);
	});
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
	if (impl_->eos.load(std::memory_order_acquire)) {
		OA_RETURN_IF_ERROR(seek(0U));
	}
	impl_->eos.store(false, std::memory_order_release);
	impl_->playing.store(true, std::memory_order_release);
	impl_->wake.notify_all();
	return oa::Status::ok();
}

void oa::AudioPlayer::pause() {
	if (impl_) impl_->playing.store(false, std::memory_order_release);
}

oa::Status oa::AudioPlayer::seek(oa::U64 inTimestampUs) {
	if (not impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::AudioPlayer::seek called on a closed stream");
	// The decoder resets both monotonic SPSC indices when it applies a seek.
	// Stop the device first so its callback cannot observe one reset index and
	// one old index (which would look like an unsigned ring overflow).
	const bool resume = impl_->playing.exchange(false, std::memory_order_acq_rel);
	if (impl_->deviceInitialized and ma_device_stop(&impl_->device) != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"oa::AudioPlayer could not stop playback for seek");
	}
	impl_->seekRequestUs.store(static_cast<oa::I64>(inTimestampUs), std::memory_order_release);
	const oa::U64 serial = impl_->seekSerial.fetch_add(1U, std::memory_order_acq_rel) + 1U;
	impl_->wake.notify_all();
	std::unique_lock lock(impl_->wakeMutex);
	const bool applied = impl_->wake.wait_for(lock, std::chrono::seconds(2), [&] {
		return impl_->appliedSeekSerial.load(std::memory_order_acquire) >= serial;
	});
	if (impl_->deviceInitialized and ma_device_start(&impl_->device) != MA_SUCCESS) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"oa::AudioPlayer could not restart playback after seek");
	}
	if (applied and resume) {
		impl_->playing.store(true, std::memory_order_release);
		impl_->wake.notify_all();
	}
	return applied ? oa::Status::ok()
		: oa::Status::error(oa::StatusCode::DeadlineExceeded, "oa::AudioPlayer seek timed out");
}

void oa::AudioPlayer::setLoop(bool inLoop) {
	if (impl_) impl_->loop.store(inLoop, std::memory_order_release);
}

oa::Status oa::AudioPlayer::close() {
	if (not impl_) return oa::Status::ok();
	impl_->stop.store(true, std::memory_order_release);
	impl_->playing.store(false, std::memory_order_release);
	impl_->wake.notify_all();
	oa::Status status = oa::Status::ok();
	if (impl_->deviceInitialized) {
		const ma_result result = ma_device_stop(&impl_->device);
		if (result != MA_SUCCESS) {
			status = oa::Status::error(oa::StatusCode::Internal,
				oa::String("oa::AudioPlayer stop failed: ")
					+ ma_result_description(result));
		}
	}
	if (impl_->decodeThread.joinable()) impl_->decodeThread.join();
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
		? impl_->positionFrame.load(std::memory_order_relaxed) * 1'000'000ULL / impl_->sampleRate
		: 0U;
}
oa::U64 oa::AudioPlayer::underrunFrameCount() const noexcept {
	return impl_ ? impl_->underrunFrames.load(std::memory_order_relaxed) : 0U;
}
