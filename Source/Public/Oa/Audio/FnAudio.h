// OaFnAudio — GPU-accelerated audio DSP operations.
//
// Stateless audio processing functions. Waveform operations return new
// OaAudio values and preserve or explicitly transform their metadata; feature
// and measurement operations return OaMatrix. Bodies record to
// OaContext::GetDefault(); no engine/runtime parameters appear here.
//
// Usage: #include <Oa/Audio.h>
// Namespace: OaFnAudio

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Audio/Type.h>

// ─── OaFnAudio Namespace ──────────────────────────────────────────────────────
namespace OaFnAudio {
	// Schema-owned declarations. The generated source fragment remains private;
	// installation maps only this declaration family to the stable SDK path.
	#include <Oa/Audio/FnAudio/FnAudio.gen.h>

	// Configuration-object conveniences are authored here because their bodies
	// are pure forwarding and introduce no second operation signature owner.
	[[nodiscard]] inline OaAudio Normalize(
		const OaAudio& InAudio,
		const OaNormalizeAudioConfig& InCfg) {
		return Normalize(InAudio, InCfg.TargetDb, InCfg.Mode);
	}
	[[nodiscard]] inline OaAudio Resample(
		const OaAudio& InAudio,
		const OaResampleConfig& InCfg) {
		return Resample(InAudio, InCfg.OutRate, InCfg.FilterHalfWidth);
	}

	// ── Planned extensions ───────────────────────────────────────────────────
	// Declarations are added together with their implementation — nothing on
	// this surface exists without a dispatch path and an oracle test.
	//
	//   0.8 (“audio v2”): IStft, GriffinLim, Convolve (overlap-save)

} // namespace OaFnAudio
