// OaAudioEncoder — lossless file output and native streaming codec boundaries.
// GPU processing remains device-resident until this explicit file/codec edge;
// WAV-F32 and signed 16-bit PCM are currently implemented.

#pragma once

#include <Oa/Audio/Type.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std/Path.h>
#include <Oa/Core/Std/UniquePtr.h>

struct OaAudioEncoder {
	// Build a WAV byte buffer from interleaved F32 samples (LRLR...).
	[[nodiscard]] static OaResult<OaVec<OaU8>> EncodeWavF32(
		OaSpan<const OaF32> InSamples,
		OaU32 InSampleRate,
		OaU32 InChannelCount);

	// Execute pending GPU work, read planar [Channels, Samples] F32, interleave,
	// and encode. This is intentionally a synchronous file/codec boundary.
	[[nodiscard]] static OaResult<OaVec<OaU8>> EncodeWavF32(
		const OaAudio& InAudio);

	[[nodiscard]] static OaStatus SaveWavF32(
		const OaPath& InPath,
		const OaAudio& InAudio);
};

struct OaEncodedAudioPacket {
	OaVec<OaU8> Bitstream;
	OaI64 PresentationFrame = 0;
	OaU32 DurationFrames = 0U;
};

struct OaAudioEncodeProfile {
	OaAudioCodec Codec = OaAudioCodec::PcmS16;
	OaU32 SampleRate = 48'000U;
	OaU32 ChannelCount = 2U;
	OaU32 FramesPerPacket = 1'024U;
};

// OA-native streaming audio encoder used by recording/container sinks.
// PcmS16 is uncompressed, deterministic and has no external codec dependency.
// Converting FP32 capture samples to 16-bit PCM is intentionally quantizing;
// additional codecs plug into this boundary rather than hiding in the muxer.
class OaAudioStreamEncoder {
public:
	struct Impl;

	OaAudioStreamEncoder() = default;
	OaAudioStreamEncoder(OaAudioStreamEncoder&& InOther) noexcept;
	OaAudioStreamEncoder& operator=(OaAudioStreamEncoder&& InOther) noexcept;
	OaAudioStreamEncoder(const OaAudioStreamEncoder&) = delete;
	OaAudioStreamEncoder& operator=(const OaAudioStreamEncoder&) = delete;
	~OaAudioStreamEncoder();

	[[nodiscard]] static OaResult<OaAudioStreamEncoder> Create(
		const OaAudioEncodeProfile& InProfile);
	[[nodiscard]] OaStatus Encode(
		OaSpan<const OaF32> InInterleaved,
		OaVec<OaEncodedAudioPacket>& OutPackets);
	[[nodiscard]] OaStatus Flush(OaVec<OaEncodedAudioPacket>& OutPackets);
	void Destroy();

	[[nodiscard]] const OaAudioEncodeProfile& GetProfile() const noexcept;
	[[nodiscard]] OaSpan<const OaU8> GetCodecConfig() const noexcept;
	[[nodiscard]] OaU32 GetPrimingFrames() const noexcept;
	[[nodiscard]] bool IsOpen() const noexcept { return Impl_ != nullptr; }

private:
	OaUniquePtr<Impl> Impl_;
};
