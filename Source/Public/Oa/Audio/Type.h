// OaAudio value, channel layout, and DSP configuration types.

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Std/Utility.h>
#include <Oa/Core/Types.h>

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

// ─── OaAudio ─────────────────────────────────────────────────────────────────
// Semantic planar float32 audio value composed over OaMatrix. The matrix view
// is [Channels, Samples]; sample rate and layout stay attached to that storage.
class OaAudio {
public:
	OaAudio() = default;

	OaAudio(OaMatrix InData, OaU32 InSampleRate, OaChannelLayout InLayout)
		: Data_(OaStdMove(InData))
		, SampleRate_(InSampleRate)
		, Layout_(InLayout)
	{}

	[[nodiscard]] const OaMatrix& AsMatrix() const noexcept { return Data_; }
	[[nodiscard]] OaMatrix& AsMatrix() noexcept { return Data_; }

	[[nodiscard]] OaI32 Channels() const noexcept {
		const OaMatrixShape shape = Data_.GetShape();
		return shape.Rank == 0 ? 0 : static_cast<OaI32>(shape[0]);
	}
	[[nodiscard]] OaI64 Samples() const noexcept {
		const OaMatrixShape shape = Data_.GetShape();
		return shape.Rank < 2 ? 0 : shape[1];
	}
	[[nodiscard]] OaU32 SampleRate() const noexcept { return SampleRate_; }
	[[nodiscard]] OaChannelLayout Layout() const noexcept { return Layout_; }
	[[nodiscard]] OaScalarType GetDtype() const noexcept {
		return Data_.GetDtype();
	}
	[[nodiscard]] bool IsEmpty() const noexcept {
		return Data_.GetShape().Rank == 0;
	}
	[[nodiscard]] OaF64 DurationSeconds() const noexcept {
		return SampleRate_ > 0
			? static_cast<OaF64>(Samples()) / static_cast<OaF64>(SampleRate_)
			: 0.0;
	}
	[[nodiscard]] bool Validate() const noexcept {
		const OaMatrixShape shape = Data_.GetShape();
		if (shape.Rank == 0) return true;
		if (shape.Rank != 2 || shape[0] <= 0 || shape[1] <= 0
			|| SampleRate_ == 0 || Data_.GetDtype() != OaScalarType::Float32) {
			return false;
		}
		const OaI32 expectedChannels = OaChannelsForLayout(Layout_);
		return expectedChannels == 0
			|| static_cast<OaI32>(shape[0]) == expectedChannels;
	}

private:
	OaMatrix Data_;
	OaU32 SampleRate_ = 44'100U;
	OaChannelLayout Layout_ = OaChannelLayout::Mono;
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
	OaU32 OutRate = 16000;       // Output sample rate
	OaU32 FilterHalfWidth = 64; // Sinc filter half-width in output samples
};

// ─── Audio Normalization Configuration ────────────────────────────────────────
struct OaNormalizeAudioConfig {
	OaU8  Mode     = 0;      // 0=peak (max abs), 1=RMS
	OaF32 TargetDb = -3.0F;  // Target level in dB
};
