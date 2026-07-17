// OaFnAudio — GPU-accelerated audio DSP operations.
//
// Stateless audio processing functions. All ops return new buffers (no
// in-place mutation) and follow the OaFnMatrix auto-context formula:
// plain `OaMatrix Op(const OaMatrix&, params…)`, body records to
// OaContext::GetDefault(). No engine/runtime parameters anywhere.
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

	// ─── Generated Audio Operations ───────────────────────────────────────────
	// Source of truth: Tools/FnAutogen/Schema/Audio/*.toml
	// Regenerate via: python3 Tools/FnAutogen/oafnautogen.py --schema-dir Tools/FnAutogen/Schema/Audio --live
	//
	// Declares (Signal): Mix(InA, InB, GainA, GainB) — weighted sum.
	#include "../../../Private/Oa/Audio/FnAudio/FnAudio.gen.h"

	// ── Transform (hand-written, Transform/FnAudioTransform.cpp) ─────────────

	// STFT magnitude spectrogram → [Channels, Frames, FftSize/2+1] F32.
	// Periodic Hann/Hamming/Blackman or rectangular window; Center pads
	// FftSize/2 zeros per side. FftSize is pow2 in [16, 1024].
	// Invalid configuration returns an empty matrix and logs the reason.
	[[nodiscard]] OaMatrix Stft(const OaAudioBuffer& InBuf, const OaStftConfig& InCfg = {});

	// Mel spectrogram → [Channels, NumMels, Frames] F32 (Whisper/CLAP layout).
	// HTK mel scale, magnitude input, optional log and per-channel normalization.
	[[nodiscard]] OaMatrix MelSpectrogram(const OaAudioBuffer& InBuf, OaU32 InSampleRate, const OaMelConfig& InCfg = {});
	[[nodiscard]] inline OaMatrix MelSpectrogram(
		const OaAudioBuffer& InBuf, const OaAudioMeta& InMeta, const OaMelConfig& InCfg = {}) {
		return MelSpectrogram(InBuf, InMeta.SampleRate, InCfg);
	}

	// MFCC → [Channels, NumCoeffs, Frames] F32.
	// Orthonormal DCT-II of the log-mel spectrogram (librosa convention),
	// computed as a single GEMM ride on the verified MatMulNt.
	[[nodiscard]] OaMatrix Mfcc(const OaAudioBuffer& InBuf, OaU32 InSampleRate, const OaMfccConfig& InCfg = {});
	[[nodiscard]] inline OaMatrix Mfcc(
		const OaAudioBuffer& InBuf, const OaAudioMeta& InMeta, const OaMfccConfig& InCfg = {}) {
		return Mfcc(InBuf, InMeta.SampleRate, InCfg);
	}

	// ── Signal (hand-written, Signal/FnAudioSignal.cpp) ───────────────────────
	// All composed from verified FnMatrix kernels — no audio-specific dispatch.

	// Peak (Mode 0) or RMS (Mode 1) normalization to InTargetDb.
	// Global level across all channels; performs one scalar host readback.
	[[nodiscard]] OaMatrix Normalize(const OaMatrix& InA, OaF32 InTargetDb = -3.0F, OaU8 InMode = 0);
	[[nodiscard]] inline OaMatrix Normalize(
		const OaMatrix& InA, const OaNormalizeAudioConfig& InCfg) {
		return Normalize(InA, InCfg.TargetDb, InCfg.Mode);
	}

	// Windowed-sinc sample-rate conversion with anti-aliasing.
	// Output: [Channels, Samples·OutRate/InRate] F32. Rates are gcd-reduced
	// internally; the AudioResample kernel gathers ±FilterHalfWidth taps.
	[[nodiscard]] OaMatrix Resample(
		const OaMatrix& InA, OaU32 InInRate = 48000, OaU32 InOutRate = 16000,
		OaU32 InFilterHalfWidth = 64
	);
	[[nodiscard]] inline OaMatrix Resample(
		const OaMatrix& InA, const OaResampleConfig& InCfg) {
		return Resample(InA, InCfg.InRate, InCfg.OutRate, InCfg.FilterHalfWidth);
	}

	// Scalar gain in dB.
	[[nodiscard]] OaMatrix Gain(const OaAudioBuffer& InBuf, OaF32 InGainDb);

	// Hard clip to [InMin, InMax].
	[[nodiscard]] OaMatrix Clip(const OaAudioBuffer& InBuf, OaF32 InMin = -1.0F, OaF32 InMax = 1.0F);

	// 20·log10(max(|x|, floor)) — amplitude to dB with a silence floor.
	[[nodiscard]] OaMatrix AmplitudeToDb(const OaAudioBuffer& InBuf, OaF32 InFloorDb = -100.0F);

	// Pre-emphasis filter: y[n] = x[n] − α·x[n−1] (y[0] = x[0]).
	[[nodiscard]] OaMatrix PreEmphasis(const OaAudioBuffer& InBuf, OaF32 InAlpha = 0.97F);

	// Average channels → [1, Samples].
	[[nodiscard]] OaMatrix ToMono(const OaAudioBuffer& InBuf);

	// Linear fade-in/fade-out envelope (sample counts per edge).
	[[nodiscard]] OaMatrix Fade(const OaAudioBuffer& InBuf, OaU64 InFadeInSamples, OaU64 InFadeOutSamples);

	// Peak-preserving display envelope. Reduces [Channels, Samples] to
	// [Bins, 2] F32 where each row is {minimum, maximum}. The reduction stays
	// on the GPU and is intended for waveform viewers and timeline scrubbing.
	[[nodiscard]] OaMatrix WaveformEnvelope(
		const OaAudioBuffer& InBuf,
		OaU32 InBins = 2048U);

	// ── Generic (hand-written, FnAudio.cpp) ───────────────────────────────────

	// Raw waveform as OaMatrix [Channels, 1, SampleCount].
	// Zero-copy reshape: wraps OaAudioBuffer without re-allocating.
	// Useful for models that take raw audio (EnCodec, WaveNet).
	[[nodiscard]] OaResult<OaMatrix> ToMatrix(const OaAudioBuffer& InBuf);

	// ── Planned extensions ───────────────────────────────────────────────────
	// Declarations are added together with their implementation — nothing on
	// this surface exists without a dispatch path and an oracle test.
	//
	//   0.8 (“audio v2”): IStft, GriffinLim, Convolve (overlap-save)

} // namespace OaFnAudio
