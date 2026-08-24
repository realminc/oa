// oa::FnAudio — GPU-accelerated audio DSP operations.
//
// Stateless audio processing functions. Waveform operations return new
// oa::Audio values and preserve or explicitly transform their metadata; feature
// and measurement operations return oa::Matrix. Bodies record through the active
// engine's private recorder; no engine/runtime parameters appear here.
//
// usage: #include <oa/audio.h>

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/std/path.h>
#include <oa/audio/type.h>

namespace oa {

namespace FnAudio {
	// Schema-owned declarations. The generated source fragment remains private;
	// installation maps only this declaration family to the stable SDK path.
	#include <oa/audio/fnaudio/fnAudio.gen.h>

	// Stateless CPU codec boundaries. Decode uploads planar Float32
	// storage through the active engine. encoding is an explicit synchronous
	// host boundary; the semantic overload completes and reads back its input.
	[[nodiscard]] oa::Result<Audio> decodeFile(const oa::Path& inPath);
	[[nodiscard]] oa::Result<Audio> decodeMemory(oa::Span<const oa::U8> inData);
	[[nodiscard]] oa::Result<oa::Vec<oa::U8>> encodeInterleavedWavF32(
		oa::Span<const oa::F32> inSamples,
		oa::U32 inSampleRate,
		oa::U32 inChannelCount
	);
	[[nodiscard]] oa::Result<oa::Vec<oa::U8>> encodeWavF32(const Audio& inAudio);
	[[nodiscard]] oa::Status saveWavF32(const oa::Path& inPath, const Audio& inAudio);

	// Configuration-object conveniences are authored here because their bodies
	// are pure forwarding and introduce no second operation signature owner.
	[[nodiscard]] inline Audio normalize(const Audio& inAudio, const NormalizeAudioConfig& inCfg) {
		return normalize(inAudio, inCfg.targetDb, inCfg.mode);
	}
	[[nodiscard]] inline Audio resample(const Audio& inAudio,	const ResampleConfig& inCfg) {
		return resample(inAudio, inCfg.outRate, inCfg.filterHalfWidth);
	}

	// ── Planned extensions ───────────────────────────────────────────────────
	// Declarations are added together with their implementation — nothing on
	// this surface exists without a dispatch path and an oracle test.
	//
	//   0.8 ("audio v2"): IStft, GriffinLim, convolve (overlap-save)

} // namespace FnAudio

} // namespace oa
