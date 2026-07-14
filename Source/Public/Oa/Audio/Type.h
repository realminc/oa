// OaAudio types — buffer alias, channel layout, unified metadata, DSP configs.
// Single source of truth for the audio type surface; included by
// AudioDecoder.h / AudioEncoder.h / FnAudio.h (use <Oa/Audio.h> as umbrella).

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>

// ─── OaAudioBuffer ────────────────────────────────────────────────────────────
// Planar float32 audio: [Channels, Samples]. Plain OaMatrix alias so all
// OaFnMatrix ops apply directly.
using OaAudioBuffer = OaMatrix;

// Encoded audio formats owned by OA. Keep this independent from containers:
// the same elementary stream can be written to MP4, Matroska or a raw sink.
enum class OaAudioCodec : OaU8 {
	PcmS16 = 0,
};

// ─── OaChannelLayout ──────────────────────────────────────────────────────────
enum class OaChannelLayout : OaU8 {
	Mono        = 0,
	Stereo      = 1,
	Surround5_1 = 2,
	Surround7_1 = 3,
	Unknown     = 255,
};

// Channel layout → expected channel count (0 = unknown layout).
[[nodiscard]] constexpr OaI32 OaChannelsForLayout(OaChannelLayout InLayout) {
	switch (InLayout) {
		case OaChannelLayout::Mono:        return 1;
		case OaChannelLayout::Stereo:      return 2;
		case OaChannelLayout::Surround5_1: return 6;
		case OaChannelLayout::Surround7_1: return 8;
		case OaChannelLayout::Unknown:     return 0;
	}
	return 0;
}

// Best-effort layout for a raw channel count (defaults to Mono when ambiguous).
[[nodiscard]] constexpr OaChannelLayout OaLayoutForChannels(OaU32 InChannels) {
	switch (InChannels) {
		case 1:  return OaChannelLayout::Mono;
		case 2:  return OaChannelLayout::Stereo;
		case 6:  return OaChannelLayout::Surround5_1;
		case 8:  return OaChannelLayout::Surround7_1;
		default: return OaChannelLayout::Unknown;
	}
}

// ─── OaAudioMeta ──────────────────────────────────────────────────────────────
// POD metadata — no heap, cheap to copy, pass by value alongside OaAudioBuffer.
struct OaAudioMeta {
	OaU32           SampleRate   = 44100;
	OaU32           ChannelCount = 1;
	OaU64           SampleCount  = 0;   // per channel
	OaChannelLayout Layout       = OaChannelLayout::Mono;

	[[nodiscard]] OaF64 DurationSeconds() const noexcept {
		return SampleRate > 0 ? double(SampleCount) / double(SampleRate) : 0.0;
	}
};

// ─── STFT Configuration ───────────────────────────────────────────────────────
struct OaStftConfig {
	OaU32 FftSize = 1024;  // Must be power of 2
	OaU32 HopSize = 256;   // Samples between frames
	OaU32 WinSize = 0;     // Window length (0 = FftSize, otherwise ≤ FftSize)
	OaU8  Window  = 0;     // 0=Hann, 1=Hamming, 2=Blackman, 3=Rect
	bool  Center  = true;  // Pad input by FftSize/2 on each side
};

// ─── Mel Spectrogram Configuration ────────────────────────────────────────────
struct OaMelConfig {
	OaU32 FftSize   = 1024;
	OaU32 HopSize   = 256;
	OaU32 NumMels   = 80;    // Number of mel bins
	OaF32 FMin      = 0.0F;  // Lowest frequency (Hz)
	OaF32 FMax      = 0.0F;  // Highest frequency (0 = SampleRate/2)
	bool  LogScale  = true;  // Apply log(mel + 1e-9)
	bool  Normalize = false; // Per-channel instance normalization
};

// ─── MFCC Configuration ───────────────────────────────────────────────────────
struct OaMfccConfig {
	OaU32 NumCoeffs = 13;  // Number of cepstral coefficients
	OaMelConfig Mel;       // Mel spectrogram config
};

// ─── Resample Configuration ───────────────────────────────────────────────────
struct OaResampleConfig {
	OaU32 InRate = 48000;        // Input sample rate
	OaU32 OutRate = 16000;       // Output sample rate
	OaU32 FilterHalfWidth = 64; // Sinc filter half-width in output samples
};

// ─── Audio Normalization Configuration ────────────────────────────────────────
struct OaNormalizeAudioConfig {
	OaU8  Mode     = 0;      // 0=peak (max abs), 1=RMS
	OaF32 TargetDb = -3.0F;  // Target level in dB
};
