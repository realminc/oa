#include <Oa/Audio/AudioStream.h>

#include <Oa/Runtime/Engine.h>
#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"

#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

struct OaAudioStream::Impl {
	OaEngine* Engine = nullptr;
	OaAudioStreamConfig Config;
	OaVec<OaF32> Ring;
	OaU64 CapacityFrames = 0U;
	std::atomic<OaU64> WriteFrame = 0U;
	std::atomic<OaU64> ReadFrame = 0U;
	std::atomic<OaU64> PositionFrame = 0U;
	std::atomic<OaU64> UnderrunFrames = 0U;
	std::atomic<bool> Playing = false;
	std::atomic<bool> Eos = false;
	std::atomic<bool> Stop = false;
	std::atomic<bool> Loop = false;
	std::atomic<OaI64> SeekRequestUs = -1;
	std::atomic<OaU64> SeekSerial = 0U;
	std::atomic<OaU64> AppliedSeekSerial = 0U;
	std::mutex WakeMutex;
	std::condition_variable Wake;
	std::thread DecodeThread;
	ma_device Device = {};
	bool DeviceInitialized = false;
	OaU32 SampleRate = 0U;
	OaU32 Channels = 0U;
	OaU64 Duration = 0U;
	ma_decoder Decoder = {};
	bool DecoderInitialized = false;
};

namespace {

void PlaybackCallback(ma_device* InDevice, void* Out, const void*, ma_uint32 InFrames) {
	auto* impl = static_cast<OaAudioStream::Impl*>(InDevice->pUserData);
	auto* output = static_cast<OaF32*>(Out);
	const OaU64 samples = static_cast<OaU64>(InFrames) * impl->Channels;
	std::memset(output, 0, static_cast<OaUsize>(samples * sizeof(OaF32)));
	if (not impl->Playing.load(std::memory_order_acquire)) return;

	const OaU64 read = impl->ReadFrame.load(std::memory_order_relaxed);
	const OaU64 write = impl->WriteFrame.load(std::memory_order_acquire);
	const OaU64 count = std::min<OaU64>(write - read, InFrames);
	const OaU64 first = read % impl->CapacityFrames;
	const OaU64 firstFrames = std::min(count, impl->CapacityFrames - first);
	std::memcpy(output,
		impl->Ring.Data() + first * impl->Channels,
		static_cast<OaUsize>(firstFrames * impl->Channels * sizeof(OaF32)));
	if (firstFrames < count) {
		std::memcpy(output + firstFrames * impl->Channels,
			impl->Ring.Data(),
			static_cast<OaUsize>((count - firstFrames) * impl->Channels * sizeof(OaF32)));
	}
	impl->ReadFrame.store(read + count, std::memory_order_release);
	impl->PositionFrame.fetch_add(count, std::memory_order_relaxed);
	if (count < InFrames) {
		impl->UnderrunFrames.fetch_add(InFrames - count, std::memory_order_relaxed);
	}
	impl->Wake.notify_one();
}

bool PushFrames(OaAudioStream::Impl& InImpl, const OaF32* InSamples, OaU64 InFrames) {
	OaU64 consumed = 0U;
	while (consumed < InFrames and not InImpl.Stop.load(std::memory_order_acquire)) {
		if (InImpl.SeekSerial.load(std::memory_order_acquire)
			!= InImpl.AppliedSeekSerial.load(std::memory_order_relaxed)) {
			return false;
		}
		const OaU64 write = InImpl.WriteFrame.load(std::memory_order_relaxed);
		const OaU64 read = InImpl.ReadFrame.load(std::memory_order_acquire);
		const OaU64 free = InImpl.CapacityFrames
			- std::min(InImpl.CapacityFrames, write - read);
		if (free == 0U) {
			std::unique_lock lock(InImpl.WakeMutex);
			InImpl.Wake.wait_for(lock, std::chrono::milliseconds(2));
			continue;
		}
		const OaU64 count = std::min(free, InFrames - consumed);
		const OaU64 first = write % InImpl.CapacityFrames;
		const OaU64 firstFrames = std::min(count, InImpl.CapacityFrames - first);
		std::memcpy(InImpl.Ring.Data() + first * InImpl.Channels,
			InSamples + consumed * InImpl.Channels,
			static_cast<OaUsize>(firstFrames * InImpl.Channels * sizeof(OaF32)));
		if (firstFrames < count) {
			std::memcpy(InImpl.Ring.Data(),
				InSamples + (consumed + firstFrames) * InImpl.Channels,
				static_cast<OaUsize>((count - firstFrames) * InImpl.Channels * sizeof(OaF32)));
		}
		InImpl.WriteFrame.store(write + count, std::memory_order_release);
		consumed += count;
	}
	return consumed == InFrames;
}

bool SeekDecoder(OaAudioStream::Impl& InImpl, OaU64 InTimestampUs) {
	if (InImpl.SampleRate == 0U) return false;
	const OaU64 frame = InTimestampUs > std::numeric_limits<OaU64>::max() / InImpl.SampleRate
		? std::numeric_limits<OaU64>::max()
		: InTimestampUs * InImpl.SampleRate / 1'000'000ULL;
	if (ma_decoder_seek_to_pcm_frame(&InImpl.Decoder, frame) != MA_SUCCESS) return false;
	InImpl.ReadFrame.store(0U, std::memory_order_relaxed);
	InImpl.WriteFrame.store(0U, std::memory_order_relaxed);
	InImpl.PositionFrame.store(frame, std::memory_order_relaxed);
	InImpl.Eos.store(false, std::memory_order_release);
	return true;
}

void DecodeLoop(OaAudioStream::Impl* InImpl) {
	constexpr OaU64 kDecodeFrames = 4096U;
	OaVec<OaF32> decoded;
	decoded.Resize(static_cast<OaUsize>(kDecodeFrames * InImpl->Channels));
	while (not InImpl->Stop.load(std::memory_order_acquire)) {
		const OaU64 requestSerial = InImpl->SeekSerial.load(std::memory_order_acquire);
		if (requestSerial != InImpl->AppliedSeekSerial.load(std::memory_order_relaxed)) {
			const OaI64 requested = InImpl->SeekRequestUs.load(std::memory_order_relaxed);
			(void)SeekDecoder(*InImpl, static_cast<OaU64>(std::max<OaI64>(0, requested)));
			InImpl->AppliedSeekSerial.store(requestSerial, std::memory_order_release);
			InImpl->Wake.notify_all();
		}
		if (not InImpl->Playing.load(std::memory_order_acquire)) {
			std::unique_lock lock(InImpl->WakeMutex);
			InImpl->Wake.wait_for(lock, std::chrono::milliseconds(10));
			continue;
		}
		ma_uint64 frameCount = 0U;
		const ma_result readResult = ma_decoder_read_pcm_frames(&InImpl->Decoder, decoded.Data(), kDecodeFrames, &frameCount);
		if (frameCount > 0U and not PushFrames(*InImpl, decoded.Data(), frameCount)) {
			continue;
		}
		if (readResult == MA_AT_END or frameCount == 0U) {
			if (InImpl->Loop.load(std::memory_order_acquire)) {
				(void)SeekDecoder(*InImpl, 0U);
				continue;
			}
			while (
				not InImpl->Stop.load(std::memory_order_acquire)
				and InImpl->ReadFrame.load(std::memory_order_acquire)
					< InImpl->WriteFrame.load(std::memory_order_acquire)
				) {
				if (InImpl->SeekSerial.load(std::memory_order_acquire)
					!= InImpl->AppliedSeekSerial.load(std::memory_order_relaxed)) {
					break;
				}
				std::unique_lock lock(InImpl->WakeMutex);
				InImpl->Wake.wait_for(lock, std::chrono::milliseconds(2));
			}
			if (InImpl->SeekSerial.load(std::memory_order_acquire)
				!= InImpl->AppliedSeekSerial.load(std::memory_order_relaxed)) {
				continue;
			}
			InImpl->Eos.store(true, std::memory_order_release);
			InImpl->Playing.store(false, std::memory_order_release);
			continue;
		}
		if (readResult != MA_SUCCESS) {
			InImpl->Eos.store(true, std::memory_order_release);
			InImpl->Playing.store(false, std::memory_order_release);
		}
	}
}

} // namespace

OaAudioStream::OaAudioStream(OaAudioStream&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_)) {}

OaAudioStream& OaAudioStream::operator=(OaAudioStream&& InOther) noexcept {
	if (this != &InOther) {
		(void)Close();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}

OaAudioStream::~OaAudioStream() { Abandon_(); }

void OaAudioStream::Abandon_() noexcept {
	if (not Impl_) return;
	OaEngine* engine = Impl_->Engine;
	if (engine == nullptr) {
		Impl_.Reset();
		return;
	}
	Impl_->Stop.store(true, std::memory_order_release);
	Impl_->Playing.store(false, std::memory_order_release);
	Impl_->Wake.notify_all();
	auto retired = OaMakeUniquePtr<OaAudioStream>(OaStdMove(*this));
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retired.Release(),
		&OaAudioStream::CompleteRetired_,
		&OaAudioStream::ReleaseRetired_);
}

OaStatus OaAudioStream::CompleteRetired_(void* InPayload) {
	auto* stream = static_cast<OaAudioStream*>(InPayload);
	return stream ? stream->Close() : OaStatus::Ok();
}

void OaAudioStream::ReleaseRetired_(void* InPayload) {
	OaUniquePtr<OaAudioStream> stream(static_cast<OaAudioStream*>(InPayload));
}

OaResult<OaAudioStream> OaAudioStream::Open(OaEngine& InEngine,	const OaAudioStreamConfig& InConfig) {
	if (InConfig.Uri.Empty() || InConfig.RingMilliseconds < 40U) {
		return OaStatus::InvalidArgument(
			"OaAudioStream requires a URI and at least 40 ms of ring storage");
	}
	OaAudioStream stream;
	stream.Impl_ = OaMakeUniquePtr<Impl>();
	auto& impl = *stream.Impl_;
	impl.Engine = &InEngine;
	impl.Config = InConfig;
	impl.Loop = InConfig.Loop;
	ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 0U, 0U);
	if (ma_decoder_init_file(InConfig.Uri.CStr(), &decoderConfig, &impl.Decoder) != MA_SUCCESS) {
		(void)stream.Close();
		return OaStatus::Error(OaStatusCode::Unavailable,	"OaAudioStream could not open WAV, FLAC or MP3 source");
	}
	impl.DecoderInitialized = true;
	ma_format outputFormat = ma_format_unknown;
	ma_uint32 outputChannels = 0U;
	ma_uint32 outputSampleRate = 0U;
	if (ma_decoder_get_data_format(&impl.Decoder, &outputFormat, &outputChannels,
		&outputSampleRate, nullptr, 0U) != MA_SUCCESS
		or outputFormat != ma_format_f32) {
		(void)stream.Close();
		return OaStatus::Error("OaAudioStream could not query the native decoder format");
	}
	impl.SampleRate = outputSampleRate;
	impl.Channels = outputChannels;
	if (impl.SampleRate == 0U || impl.Channels == 0U || impl.Channels > 8U) {
		(void)stream.Close();
		return OaStatus::Error(OaStatusCode::Unimplemented,	"OaAudioStream supports 1..8 channel streams with a declared sample rate");
	}
	ma_uint64 totalFrames = 0U;
	if (ma_decoder_get_length_in_pcm_frames(&impl.Decoder, &totalFrames) == MA_SUCCESS
		and totalFrames <= std::numeric_limits<OaU64>::max() / 1'000'000ULL) {
		impl.Duration = totalFrames * 1'000'000ULL / impl.SampleRate;
	}
	impl.CapacityFrames = std::max<OaU64>(1U, static_cast<OaU64>(impl.SampleRate) * InConfig.RingMilliseconds / 1000U);
	impl.Ring.Resize(static_cast<OaUsize>(impl.CapacityFrames * impl.Channels));

	ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
	deviceConfig.playback.format = ma_format_f32;
	deviceConfig.playback.channels = impl.Channels;
	deviceConfig.sampleRate = impl.SampleRate;
	deviceConfig.dataCallback = PlaybackCallback;
	deviceConfig.pUserData = &impl;
	if (ma_device_init(nullptr, &deviceConfig, &impl.Device) != MA_SUCCESS) {
		(void)stream.Close();
		return OaStatus::Error(OaStatusCode::Unavailable,	"OaAudioStream could not open the playback device");
	}
	impl.DeviceInitialized = true;
	if (ma_device_start(&impl.Device) != MA_SUCCESS) {
		(void)stream.Close();
		return OaStatus::Error(OaStatusCode::Unavailable,	"OaAudioStream could not start the playback device");
	}
	impl.DecodeThread = std::thread(DecodeLoop, &impl);
	return stream;
}

OaResult<OaAudioStream> OaAudioStream::Open(OaEngine& InEngine,	OaStringView InUri) {
	OaAudioStreamConfig config;
	config.Uri = OaString(InUri);
	return Open(InEngine, config);
}

OaStatus OaAudioStream::Play() {
	if (not Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaAudioStream::Play called on a closed stream");
	if (Impl_->Eos.load(std::memory_order_acquire)) {
		OA_RETURN_IF_ERROR(Seek(0U));
	}
	Impl_->Eos.store(false, std::memory_order_release);
	Impl_->Playing.store(true, std::memory_order_release);
	Impl_->Wake.notify_all();
	return OaStatus::Ok();
}

void OaAudioStream::Pause() {
	if (Impl_) Impl_->Playing.store(false, std::memory_order_release);
}

OaStatus OaAudioStream::Seek(OaU64 InTimestampUs) {
	if (not Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition,
		"OaAudioStream::Seek called on a closed stream");
	// The decoder resets both monotonic SPSC indices when it applies a seek.
	// Stop the device first so its callback cannot observe one reset index and
	// one old index (which would look like an unsigned ring overflow).
	const bool resume = Impl_->Playing.exchange(false, std::memory_order_acq_rel);
	if (Impl_->DeviceInitialized and ma_device_stop(&Impl_->Device) != MA_SUCCESS) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"OaAudioStream could not stop playback for seek");
	}
	Impl_->SeekRequestUs.store(static_cast<OaI64>(InTimestampUs), std::memory_order_release);
	const OaU64 serial = Impl_->SeekSerial.fetch_add(1U, std::memory_order_acq_rel) + 1U;
	Impl_->Wake.notify_all();
	std::unique_lock lock(Impl_->WakeMutex);
	const bool applied = Impl_->Wake.wait_for(lock, std::chrono::seconds(2), [&] {
		return Impl_->AppliedSeekSerial.load(std::memory_order_acquire) >= serial;
	});
	if (Impl_->DeviceInitialized and ma_device_start(&Impl_->Device) != MA_SUCCESS) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"OaAudioStream could not restart playback after seek");
	}
	if (applied and resume) {
		Impl_->Playing.store(true, std::memory_order_release);
		Impl_->Wake.notify_all();
	}
	return applied ? OaStatus::Ok()
		: OaStatus::Error(OaStatusCode::DeadlineExceeded, "OaAudioStream seek timed out");
}

void OaAudioStream::SetLoop(bool InLoop) {
	if (Impl_) Impl_->Loop.store(InLoop, std::memory_order_release);
}

OaStatus OaAudioStream::Close() {
	if (not Impl_) return OaStatus::Ok();
	Impl_->Stop.store(true, std::memory_order_release);
	Impl_->Playing.store(false, std::memory_order_release);
	Impl_->Wake.notify_all();
	OaStatus status = OaStatus::Ok();
	if (Impl_->DeviceInitialized) {
		const ma_result result = ma_device_stop(&Impl_->Device);
		if (result != MA_SUCCESS) {
			status = OaStatus::Error(OaStatusCode::Internal,
				OaString("OaAudioStream stop failed: ")
					+ ma_result_description(result));
		}
	}
	if (Impl_->DecodeThread.joinable()) Impl_->DecodeThread.join();
	if (Impl_->DeviceInitialized) ma_device_uninit(&Impl_->Device);
	if (Impl_->DecoderInitialized) ma_decoder_uninit(&Impl_->Decoder);
	Impl_.Reset();
	return status;
}

bool OaAudioStream::IsOpen() const noexcept { return Impl_ != nullptr; }
bool OaAudioStream::IsPlaying() const noexcept { return Impl_ and Impl_->Playing.load(); }
bool OaAudioStream::IsEos() const noexcept { return not Impl_ or Impl_->Eos.load(); }
OaU32 OaAudioStream::SampleRate() const noexcept { return Impl_ ? Impl_->SampleRate : 0U; }
OaU32 OaAudioStream::ChannelCount() const noexcept { return Impl_ ? Impl_->Channels : 0U; }
OaU64 OaAudioStream::DurationUs() const noexcept { return Impl_ ? Impl_->Duration : 0U; }
OaU64 OaAudioStream::PositionUs() const noexcept {
	return Impl_ and Impl_->SampleRate > 0U
		? Impl_->PositionFrame.load(std::memory_order_relaxed) * 1'000'000ULL / Impl_->SampleRate
		: 0U;
}
OaU64 OaAudioStream::UnderrunFrameCount() const noexcept {
	return Impl_ ? Impl_->UnderrunFrames.load(std::memory_order_relaxed) : 0U;
}
