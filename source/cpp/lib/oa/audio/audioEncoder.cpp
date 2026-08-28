// oa::FnAudio WAV-F32 sinks and the native streaming audio encode session.

#include <oa/audio/audioEncoder.h>
#include <oa/audio/fnAudio.h>
#include <oa/core/filesystem.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/runtime/executionSession.h>

namespace {

void writeU16Le(oa::U8* out, oa::U16 inValue) {
	out[0] = static_cast<oa::U8>(inValue & 0xFFU);
	out[1] = static_cast<oa::U8>((inValue >> 8U) & 0xFFU);
}

void writeU32Le(oa::U8* out, oa::U32 inValue) {
	out[0] = static_cast<oa::U8>(inValue & 0xFFU);
	out[1] = static_cast<oa::U8>((inValue >> 8U) & 0xFFU);
	out[2] = static_cast<oa::U8>((inValue >> 16U) & 0xFFU);
	out[3] = static_cast<oa::U8>((inValue >> 24U) & 0xFFU);
}

} // namespace

oa::Result<oa::Vector<oa::U8>> oa::FnAudio::encodeInterleavedWavF32(
	oa::Span<const oa::F32> inSamples,
	oa::U32 inSampleRate,
	oa::U32 inChannelCount)
{
	if (inSampleRate == 0) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeInterleavedWavF32: sample rate must be > 0");
	}
	if (inChannelCount == 0 || inChannelCount > oa::Limits<oa::U16>::max()) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeInterleavedWavF32: channel count must be in [1, 65535]");
	}
	if (inSamples.empty() || (inSamples.size() % inChannelCount) != 0) {
		return oa::Status::invalidArgument(
			"oa::FnAudio::encodeInterleavedWavF32: samples must contain complete, non-empty interleaved frames");
	}

	constexpr oa::U64 kBytesPerSample = sizeof(oa::F32);
	constexpr oa::U64 kHeaderBytes = 46;
	const oa::U64 dataBytes64 = static_cast<oa::U64>(inSamples.size()) * kBytesPerSample;
	if (dataBytes64 > oa::Limits<oa::U32>::max() - 38ULL) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeInterleavedWavF32: WAV exceeds the RIFF 32-bit size limit");
	}
	const oa::U64 byteRate64 = static_cast<oa::U64>(inSampleRate) * inChannelCount * kBytesPerSample;
	const oa::U64 blockAlign64 = static_cast<oa::U64>(inChannelCount) * kBytesPerSample;
	if (byteRate64 > oa::Limits<oa::U32>::max() ||
		blockAlign64 > oa::Limits<oa::U16>::max()) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeInterleavedWavF32: WAV rate/channel product overflows the header");
	}

	const oa::U32 dataBytes = static_cast<oa::U32>(dataBytes64);
	oa::Vector<oa::U8> out;
	out.resize(static_cast<oa::Usize>(kHeaderBytes + dataBytes64));
	oa::U8* p = out.data();

	p[0] = 'R'; p[1] = 'I'; p[2] = 'F'; p[3] = 'F';
	writeU32Le(p + 4, 38U + dataBytes);
	p[8] = 'W'; p[9] = 'A'; p[10] = 'V'; p[11] = 'E';
	p[12] = 'f'; p[13] = 'm'; p[14] = 't'; p[15] = ' ';
	writeU32Le(p + 16, 18U);
	writeU16Le(p + 20, 3U); // WAVE_FORMAT_IEEE_FLOAT
	writeU16Le(p + 22, static_cast<oa::U16>(inChannelCount));
	writeU32Le(p + 24, inSampleRate);
	writeU32Le(p + 28, static_cast<oa::U32>(byteRate64));
	writeU16Le(p + 32, static_cast<oa::U16>(blockAlign64));
	writeU16Le(p + 34, 32U);
	writeU16Le(p + 36, 0U);
	p[38] = 'd'; p[39] = 'a'; p[40] = 't'; p[41] = 'a';
	writeU32Le(p + 42, dataBytes);
	oa::memcpy(p + kHeaderBytes, inSamples.data(), static_cast<oa::Usize>(dataBytes));
	return out;
}

oa::Result<oa::Vector<oa::U8>> oa::FnAudio::encodeWavF32(const oa::Audio& inAudio) {
	if (not inAudio.validate() || inAudio.isEmpty()) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeWavF32: expected valid non-empty audio");
	}
	const oa::Matrix& inBuffer = inAudio.asMatrix();
	const oa::MatrixShape shape = inBuffer.getShape();
	if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeWavF32: expected non-empty [channels, samples] audio");
	}
	if (inBuffer.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeWavF32: audio buffer must be Float32");
	}

	auto& ctx = oa::ExecutionSession::getActive();
	if (auto status = ctx.submitAndWait(); not status.isOk()) {
		return status;
	}

	const oa::U64 channels = static_cast<oa::U64>(shape[0]);
	const oa::U64 samples = static_cast<oa::U64>(shape[1]);
	if (channels > oa::Limits<oa::Usize>::max() / samples) {
		return oa::Status::invalidArgument("oa::FnAudio::encodeWavF32: audio shape overflows host storage");
	}
	oa::Vector<oa::F32> interleaved(static_cast<oa::Usize>(channels * samples));
	const oa::F32* planar = inBuffer.dataAs<oa::F32>();
	for (oa::U64 sample = 0; sample < samples; ++sample) {
		for (oa::U64 channel = 0; channel < channels; ++channel) {
			interleaved[static_cast<oa::Usize>(sample * channels + channel)] = planar[static_cast<oa::Usize>(channel * samples + sample)];
		}
	}
	return encodeInterleavedWavF32(
		oa::Span<const oa::F32>(interleaved.data(), interleaved.size()),
		inAudio.sampleRate(),
		static_cast<oa::U32>(channels)
	);
}

oa::Status oa::FnAudio::saveWavF32(
	const oa::Path& inPath,
	const oa::Audio& inAudio)
{
	if (inPath.empty()) {
		return oa::Status::invalidArgument("oa::FnAudio::saveWavF32: output path is empty");
	}
	auto encoded = encodeWavF32(inAudio);
	if (not encoded.isOk()) return encoded.getStatus();
	const auto& bytes = encoded.getValue();
	return oa::Filesystem::writeBinary(inPath, oa::Span<const oa::U8>(bytes.data(), bytes.size()));
}

struct oa::AudioEncoder::Impl {
	oa::AudioEncodeProfile profile = {};
	oa::Vector<oa::U8> codecConfig;
	oa::Vector<oa::F32> pending;
	oa::I64 nextInputFrame = 0;
	oa::U32 primingFrames = 0U;
};

namespace {

oa::I16 quantizePcmS16(oa::F32 inSample) {
	// A device/caller can hand the recorder non-finite FP32. Never feed NaN to
	// lrint: silence NaN and saturate infinities deterministically.
	if (oa::isNan(inSample)) return 0;
	if (inSample >= 1.0F) return oa::Limits<oa::I16>::max();
	if (inSample <= -1.0F) return oa::Limits<oa::I16>::min();
	const oa::F32 clamped = oa::clamp(inSample, -1.0F, 1.0F);
	const oa::F32 scaled = clamped < 0.0F ? clamped * 32768.0F : clamped * 32767.0F;
	return static_cast<oa::I16>(oa::lround(scaled));
}

void emitPcmPacket(
	oa::AudioEncoder::Impl& inImpl,
	oa::U32 inFrames,
	oa::Vector<oa::EncodedAudioPacket>& outPackets
) {
	const oa::U32 channels = inImpl.profile.channelCount;
	const oa::Usize samples = static_cast<oa::Usize>(inFrames) * channels;
	oa::EncodedAudioPacket packet;
	packet.bitstream.resize(samples * sizeof(oa::I16));
	for (oa::Usize i = 0U; i < samples; ++i) {
		const oa::U16 value = static_cast<oa::U16>(quantizePcmS16(inImpl.pending[i]));
		packet.bitstream[i * 2U] = static_cast<oa::U8>(value & 0xFFU);
		packet.bitstream[i * 2U + 1U] = static_cast<oa::U8>(value >> 8U);
	}
	packet.presentationFrame = inImpl.nextInputFrame;
	packet.durationFrames = inFrames;
	inImpl.nextInputFrame += inFrames;
	outPackets.pushBack(oa::move(packet));
	const oa::Usize remaining = inImpl.pending.size() - samples;
	if (remaining > 0U) {
		oa::memmove(inImpl.pending.data(), inImpl.pending.data() + samples,
			remaining * sizeof(oa::F32)
		);
	}
	inImpl.pending.resize(remaining);
}

} // namespace

oa::AudioEncoder::AudioEncoder() = default;

oa::AudioEncoder::AudioEncoder(oa::AudioEncoder&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
{}

oa::AudioEncoder& oa::AudioEncoder::operator=(oa::AudioEncoder&& inOther) noexcept
{
	if (this != &inOther) {
		impl_.reset();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}

oa::AudioEncoder::~AudioEncoder()
{
	impl_.reset();
}

oa::Result<oa::AudioEncoder> oa::AudioEncoder::create(
	const oa::AudioEncodeProfile& inProfile)
{
	if (inProfile.sampleRate == 0U || inProfile.channelCount == 0U
		|| inProfile.channelCount > 8U || inProfile.framesPerPacket == 0U) {
		return oa::Status::invalidArgument(
			"Audio encoder requires a sample rate, 1..8 channels and a packet size");
	}
	if (inProfile.codec != oa::AudioCodec::PcmS16) {
		return oa::Status::error(oa::StatusCode::Unimplemented,
			"Requested audio codec is not implemented by OA");
	}
	if (inProfile.framesPerPacket
		> oa::Limits<oa::Usize>::max() / inProfile.channelCount / sizeof(oa::I16)) {
		return oa::Status::invalidArgument("Audio packet shape exceeds host address space");
	}
	oa::AudioEncoder encoder;
	encoder.impl_ = oa::makeUnique<Impl>();
	encoder.impl_->profile = inProfile;
	return encoder;
}

oa::Status oa::AudioEncoder::encode(
	oa::Span<const oa::F32> inInterleaved,
	oa::Vector<oa::EncodedAudioPacket>& outPackets)
{
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition, "Audio encoder is not open");
	if (inInterleaved.empty() || inInterleaved.size() % impl_->profile.channelCount != 0U) {
		return oa::Status::invalidArgument("Audio input must contain complete interleaved frames");
	}
	const oa::Usize oldSize = impl_->pending.size();
	if (inInterleaved.size() > oa::Limits<oa::Usize>::max() - oldSize) {
		return oa::Status::invalidArgument("Audio input exceeds host address space");
	}
	impl_->pending.resize(oldSize + inInterleaved.size());
	oa::memcpy(impl_->pending.data() + oldSize, inInterleaved.data(),
		inInterleaved.size() * sizeof(oa::F32));
	const oa::Usize frameSamples = static_cast<oa::Usize>(impl_->profile.framesPerPacket)
		* impl_->profile.channelCount;
	while (impl_->pending.size() >= frameSamples) {
		emitPcmPacket(*impl_, impl_->profile.framesPerPacket, outPackets);
	}
	return oa::Status::ok();
}

oa::Status oa::AudioEncoder::flush(oa::Vector<oa::EncodedAudioPacket>& outPackets)
{
	if (!impl_) return oa::Status::ok();
	if (!impl_->pending.empty()) {
		const oa::U32 frames = static_cast<oa::U32>(
			impl_->pending.size() / impl_->profile.channelCount);
		emitPcmPacket(*impl_, frames, outPackets);
	}
	return oa::Status::ok();
}

oa::Status oa::AudioEncoder::close()
{
	impl_.reset();
	return oa::Status::ok();
}

const oa::AudioEncodeProfile& oa::AudioEncoder::getProfile() const noexcept
{
	static const oa::AudioEncodeProfile empty = {};
	return impl_ ? impl_->profile : empty;
}

oa::Span<const oa::U8> oa::AudioEncoder::getCodecConfig() const noexcept
{
	return impl_ ? oa::Span<const oa::U8>(impl_->codecConfig.data(), impl_->codecConfig.size())
		: oa::Span<const oa::U8>();
}

oa::U32 oa::AudioEncoder::getPrimingFrames() const noexcept
{
	return impl_ ? impl_->primingFrames : 0U;
}
