// OaAudioDecoder — CPU-side audio file decode via miniaudio.
// Supports miniaudio's built-in WAV, FLAC, and MP3 decoders. Ogg Vorbis
// requires a separately configured backend and is not enabled in OA today.
// Output is always interleaved F32 on host, then uploaded as one semantic
// OaAudio value backed by a planar [Channels, Samples] GPU matrix.
//
// Encode lives in <Oa/Audio/AudioEncoder.h>.

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Audio/Type.h>

// ─── OaAudioDecoder ───────────────────────────────────────────────────────────
// Stateless — all methods are static.
struct OaAudioDecoder {
	// Decode a file from disk (WAV/FLAC/MP3).
	[[nodiscard]] static OaResult<OaAudio> LoadFile(
		const OaPath& InPath);

	// Decode from memory (e.g. embedded asset or downloaded buffer).
	[[nodiscard]] static OaResult<OaAudio> LoadMemory(
		OaSpan<const OaU8> InData);
};
