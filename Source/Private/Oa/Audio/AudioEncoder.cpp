// OaAudioEncoder — validated WAV-F32 encoder and synchronous GPU readback sink.

#include <Oa/Audio/AudioEncoder.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Context.h>

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

void WriteU16Le(OaU8* Out, OaU16 InValue) {
	Out[0] = static_cast<OaU8>(InValue & 0xFFU);
	Out[1] = static_cast<OaU8>((InValue >> 8U) & 0xFFU);
}

void WriteU32Le(OaU8* Out, OaU32 InValue) {
	Out[0] = static_cast<OaU8>(InValue & 0xFFU);
	Out[1] = static_cast<OaU8>((InValue >> 8U) & 0xFFU);
	Out[2] = static_cast<OaU8>((InValue >> 16U) & 0xFFU);
	Out[3] = static_cast<OaU8>((InValue >> 24U) & 0xFFU);
}

} // namespace

OaResult<OaVec<OaU8>> OaAudioEncoder::EncodeWavF32(OaSpan<const OaF32> InSamples,	OaU32 InSampleRate, OaU32 InChannelCount) {
	if (InSampleRate == 0) {
		return OaStatus::InvalidArgument("OaAudioEncoder: sample rate must be > 0");
	}
	if (InChannelCount == 0 || InChannelCount > std::numeric_limits<OaU16>::max()) {
		return OaStatus::InvalidArgument("OaAudioEncoder: channel count must be in [1, 65535]");
	}
	if (InSamples.Empty() || (InSamples.Size() % InChannelCount) != 0) {
		return OaStatus::InvalidArgument(
			"OaAudioEncoder: samples must contain complete, non-empty interleaved frames");
	}

	constexpr OaU64 kBytesPerSample = sizeof(OaF32);
	constexpr OaU64 kHeaderBytes = 46;
	const OaU64 dataBytes64 = static_cast<OaU64>(InSamples.Size()) * kBytesPerSample;
	if (dataBytes64 > std::numeric_limits<OaU32>::max() - 38ULL) {
		return OaStatus::InvalidArgument("OaAudioEncoder: WAV exceeds the RIFF 32-bit size limit");
	}
	const OaU64 byteRate64 = static_cast<OaU64>(InSampleRate) * InChannelCount * kBytesPerSample;
	const OaU64 blockAlign64 = static_cast<OaU64>(InChannelCount) * kBytesPerSample;
	if (byteRate64 > std::numeric_limits<OaU32>::max() ||
		blockAlign64 > std::numeric_limits<OaU16>::max()) {
		return OaStatus::InvalidArgument("OaAudioEncoder: WAV rate/channel product overflows the header");
	}

	const OaU32 dataBytes = static_cast<OaU32>(dataBytes64);
	OaVec<OaU8> out;
	out.Resize(static_cast<OaUsize>(kHeaderBytes + dataBytes64));
	OaU8* p = out.Data();

	p[0] = 'R'; p[1] = 'I'; p[2] = 'F'; p[3] = 'F';
	WriteU32Le(p + 4, 38U + dataBytes);
	p[8] = 'W'; p[9] = 'A'; p[10] = 'V'; p[11] = 'E';
	p[12] = 'f'; p[13] = 'm'; p[14] = 't'; p[15] = ' ';
	WriteU32Le(p + 16, 18U);
	WriteU16Le(p + 20, 3U); // WAVE_FORMAT_IEEE_FLOAT
	WriteU16Le(p + 22, static_cast<OaU16>(InChannelCount));
	WriteU32Le(p + 24, InSampleRate);
	WriteU32Le(p + 28, static_cast<OaU32>(byteRate64));
	WriteU16Le(p + 32, static_cast<OaU16>(blockAlign64));
	WriteU16Le(p + 34, 32U);
	WriteU16Le(p + 36, 0U);
	p[38] = 'd'; p[39] = 'a'; p[40] = 't'; p[41] = 'a';
	WriteU32Le(p + 42, dataBytes);
	OaMemcpy(p + kHeaderBytes, InSamples.Data(), static_cast<OaUsize>(dataBytes));
	return out;
}

OaResult<OaVec<OaU8>> OaAudioEncoder::EncodeWavF32(const OaAudio& InAudio) {
	if (not InAudio.Validate() || InAudio.IsEmpty()) {
		return OaStatus::InvalidArgument("OaAudioEncoder: expected valid non-empty audio");
	}
	const OaMatrix& InBuffer = InAudio.AsMatrix();
	const OaMatrixShape shape = InBuffer.GetShape();
	if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0) {
		return OaStatus::InvalidArgument("OaAudioEncoder: expected non-empty [Channels, Samples] audio");
	}
	if (InBuffer.GetDtype() != OaScalarType::Float32) {
		return OaStatus::InvalidArgument("OaAudioEncoder: audio buffer must be Float32");
	}

	auto& ctx = OaContext::GetDefault();
	if (auto status = ctx.Execute(); not status.IsOk()) return status;
	if (auto status = ctx.Sync(); not status.IsOk()) return status;

	const OaU64 channels = static_cast<OaU64>(shape[0]);
	const OaU64 samples = static_cast<OaU64>(shape[1]);
	if (channels > std::numeric_limits<OaUsize>::max() / samples) {
		return OaStatus::InvalidArgument("OaAudioEncoder: audio shape overflows host storage");
	}
	OaVec<OaF32> interleaved(static_cast<OaUsize>(channels * samples));
	const OaF32* planar = InBuffer.DataAs<OaF32>();
	for (OaU64 sample = 0; sample < samples; ++sample) {
		for (OaU64 channel = 0; channel < channels; ++channel) {
			interleaved[static_cast<OaUsize>(sample * channels + channel)] = planar[static_cast<OaUsize>(channel * samples + sample)];
		}
	}
	return EncodeWavF32(
		OaSpan<const OaF32>(interleaved.Data(), interleaved.Size()),
		InAudio.SampleRate(),
		static_cast<OaU32>(channels)
	);
}

OaStatus OaAudioEncoder::SaveWavF32(const OaPath& InPath,	const OaAudio& InAudio) {
	if (InPath.Empty()) {
		return OaStatus::InvalidArgument("OaAudioEncoder: output path is empty");
	}
	auto encoded = EncodeWavF32(InAudio);
	if (not encoded.IsOk()) return encoded.GetStatus();
	const auto& bytes = encoded.GetValue();
	return OaFilesystem::WriteBinary(InPath, OaSpan<const OaU8>(bytes.Data(), bytes.Size()));
}

struct OaAudioStreamEncoder::Impl {
	OaAudioEncodeProfile Profile = {};
	OaVec<OaU8> CodecConfig;
	OaVec<OaF32> Pending;
	OaI64 NextInputFrame = 0;
	OaU32 PrimingFrames = 0U;
};

namespace {

OaI16 QuantizePcmS16(OaF32 InSample) {
	// A device/caller can hand the recorder non-finite FP32. Never feed NaN to
	// lrint: silence NaN and saturate infinities deterministically.
	if (std::isnan(InSample)) return 0;
	if (InSample >= 1.0F) return std::numeric_limits<OaI16>::max();
	if (InSample <= -1.0F) return std::numeric_limits<OaI16>::min();
	const OaF32 clamped = std::clamp(InSample, -1.0F, 1.0F);
	const OaF32 scaled = clamped < 0.0F ? clamped * 32768.0F : clamped * 32767.0F;
	return static_cast<OaI16>(std::lrint(scaled));
}

void EmitPcmPacket(OaAudioStreamEncoder::Impl& InImpl, OaU32 InFrames, OaVec<OaEncodedAudioPacket>& OutPackets) {
	const OaU32 channels = InImpl.Profile.ChannelCount;
	const OaUsize samples = static_cast<OaUsize>(InFrames) * channels;
	OaEncodedAudioPacket packet;
	packet.Bitstream.Resize(samples * sizeof(OaI16));
	for (OaUsize i = 0U; i < samples; ++i) {
		const OaU16 value = static_cast<OaU16>(QuantizePcmS16(InImpl.Pending[i]));
		packet.Bitstream[i * 2U] = static_cast<OaU8>(value & 0xFFU);
		packet.Bitstream[i * 2U + 1U] = static_cast<OaU8>(value >> 8U);
	}
	packet.PresentationFrame = InImpl.NextInputFrame;
	packet.DurationFrames = InFrames;
	InImpl.NextInputFrame += InFrames;
	OutPackets.PushBack(OaStdMove(packet));
	const OaUsize remaining = InImpl.Pending.Size() - samples;
	if (remaining > 0U) {
		std::memmove(InImpl.Pending.Data(), InImpl.Pending.Data() + samples,
			remaining * sizeof(OaF32)
		);
	}
	InImpl.Pending.Resize(remaining);
}

} // namespace

OaAudioStreamEncoder::OaAudioStreamEncoder(OaAudioStreamEncoder&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_))
{}

OaAudioStreamEncoder& OaAudioStreamEncoder::operator=(OaAudioStreamEncoder&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}

OaAudioStreamEncoder::~OaAudioStreamEncoder()
{
	Destroy();
}

OaResult<OaAudioStreamEncoder> OaAudioStreamEncoder::Create(
	const OaAudioEncodeProfile& InProfile)
{
	if (InProfile.SampleRate == 0U || InProfile.ChannelCount == 0U
		|| InProfile.ChannelCount > 8U || InProfile.FramesPerPacket == 0U) {
		return OaStatus::InvalidArgument(
			"Audio encoder requires a sample rate, 1..8 channels and a packet size");
	}
	if (InProfile.Codec != OaAudioCodec::PcmS16) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"Requested audio codec is not implemented by OA");
	}
	if (InProfile.FramesPerPacket
		> std::numeric_limits<OaUsize>::max() / InProfile.ChannelCount / sizeof(OaI16)) {
		return OaStatus::InvalidArgument("Audio packet shape exceeds host address space");
	}
	OaAudioStreamEncoder encoder;
	encoder.Impl_ = OaMakeUniquePtr<Impl>();
	encoder.Impl_->Profile = InProfile;
	return encoder;
}

OaStatus OaAudioStreamEncoder::Encode(
	OaSpan<const OaF32> InInterleaved,
	OaVec<OaEncodedAudioPacket>& OutPackets)
{
	if (!Impl_) return OaStatus::Error(OaStatusCode::FailedPrecondition, "Audio encoder is not open");
	if (InInterleaved.Empty() || InInterleaved.Size() % Impl_->Profile.ChannelCount != 0U) {
		return OaStatus::InvalidArgument("Audio input must contain complete interleaved frames");
	}
	const OaUsize oldSize = Impl_->Pending.Size();
	if (InInterleaved.Size() > std::numeric_limits<OaUsize>::max() - oldSize) {
		return OaStatus::InvalidArgument("Audio input exceeds host address space");
	}
	Impl_->Pending.Resize(oldSize + InInterleaved.Size());
	OaMemcpy(Impl_->Pending.Data() + oldSize, InInterleaved.Data(),
		InInterleaved.Size() * sizeof(OaF32));
	const OaUsize frameSamples = static_cast<OaUsize>(Impl_->Profile.FramesPerPacket)
		* Impl_->Profile.ChannelCount;
	while (Impl_->Pending.Size() >= frameSamples) {
		EmitPcmPacket(*Impl_, Impl_->Profile.FramesPerPacket, OutPackets);
	}
	return OaStatus::Ok();
}

OaStatus OaAudioStreamEncoder::Flush(OaVec<OaEncodedAudioPacket>& OutPackets)
{
	if (!Impl_) return OaStatus::Ok();
	if (!Impl_->Pending.Empty()) {
		const OaU32 frames = static_cast<OaU32>(
			Impl_->Pending.Size() / Impl_->Profile.ChannelCount);
		EmitPcmPacket(*Impl_, frames, OutPackets);
	}
	return OaStatus::Ok();
}

void OaAudioStreamEncoder::Destroy()
{
	if (!Impl_) return;
	Impl_.Reset();
}

const OaAudioEncodeProfile& OaAudioStreamEncoder::GetProfile() const noexcept
{
	static const OaAudioEncodeProfile empty = {};
	return Impl_ ? Impl_->Profile : empty;
}

OaSpan<const OaU8> OaAudioStreamEncoder::GetCodecConfig() const noexcept
{
	return Impl_ ? OaSpan<const OaU8>(Impl_->CodecConfig.Data(), Impl_->CodecConfig.Size())
		: OaSpan<const OaU8>();
}

OaU32 OaAudioStreamEncoder::GetPrimingFrames() const noexcept
{
	return Impl_ ? Impl_->PrimingFrames : 0U;
}
