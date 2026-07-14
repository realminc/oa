// OaAudioDecoder — CPU-side audio file decode via miniaudio.
// Supports miniaudio's built-in WAV, FLAC, and MP3 decoders. Ogg Vorbis
// requires a separately configured backend and is not enabled in OA today.
// Output is always interleaved F32 on host, then uploaded to GPU as
// OaAudioBuffer [Channels, Samples] F32 via OaFnMatrix::Empty + host copy.
//
// Encode lives in <Oa/Audio/AudioEncoder.h>.

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Audio/Type.h>

// ─── OaAudioDecodeResult ──────────────────────────────────────────────────────
struct OaAudioDecodeResult {
	OaMatrix Buffer;       // [Channels, Samples] F32 on GPU
	OaU32          SampleRate   = 0;
	OaU32          ChannelCount = 0;
	OaU64          SampleCount  = 0;   // per channel

	[[nodiscard]] bool IsValid() const noexcept {
		return SampleRate > 0 && ChannelCount > 0 && SampleCount > 0 &&
			Buffer.GetShape().Rank == 2 && Buffer.GetDtype() == OaScalarType::Float32;
	}
	[[nodiscard]] OaF64 DurationSeconds() const noexcept {
		return SampleRate > 0 ? double(SampleCount) / double(SampleRate) : 0.0;
	}

	// Metadata POD for the OaFnAudio ops (layout inferred from channel count).
	[[nodiscard]] OaAudioMeta Meta() const noexcept {
		return OaAudioMeta{
			.SampleRate   = SampleRate,
			.ChannelCount = ChannelCount,
			.SampleCount  = SampleCount,
			.Layout       = OaLayoutForChannels(ChannelCount),
		};
	}
};

// ─── OaAudioDecoder ───────────────────────────────────────────────────────────
// Stateless — all methods are static.
struct OaAudioDecoder {
		// Decode a file from disk (WAV/FLAC/MP3).
	// Returns [Channels, Samples] F32 GPU buffer.
	[[nodiscard]] static OaResult<OaAudioDecodeResult> LoadFile(
		const char* InPath);

	[[nodiscard]] static OaResult<OaAudioDecodeResult> LoadFile(
		const OaString& InPath) { return LoadFile(InPath.c_str()); }

	// Decode from memory (e.g. embedded asset or downloaded buffer).
	[[nodiscard]] static OaResult<OaAudioDecodeResult> LoadMemory(
		OaSpan<const OaU8> InData);
};
