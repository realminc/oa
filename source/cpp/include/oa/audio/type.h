// oa::Audio value, channel layout, and DSP configuration types.

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/std/utility.h>
#include <oa/core/types.h>

namespace oa {

// Encoded audio formats owned by OA. Keep this independent from containers:
// the same elementary stream can be written to MP4, Matroska or a raw sink.
enum class AudioCodec : oa::U8 {
	PcmS16 = 0,
};

// ─── ChannelLayout ────────────────────────────────────────────────────────────
enum class AudioChannelLayout : oa::U8 {
	Mono       = 0,
	Stereo     = 1,
	Stereo21   = 2,
	Surround51 = 3,
	Surround71 = 4,
	Unknown    = 255,
};

// Channel layout → expected channel count (0 = unknown layout).
[[nodiscard]] constexpr oa::I32 channelsForLayout(AudioChannelLayout inLayout) {
	switch (inLayout) {
		case AudioChannelLayout::Mono:       return 1;
		case AudioChannelLayout::Stereo:     return 2;
		case AudioChannelLayout::Stereo21:   return 3;
		case AudioChannelLayout::Surround51: return 6;
		case AudioChannelLayout::Surround71: return 8;
		case AudioChannelLayout::Unknown:    return 0;
	}
	return 0;
}

// Best-effort layout for a raw channel count. Three channels remain unknown:
// channel count alone cannot distinguish L/R/LFE (2.1) from L/C/R (3.0).
[[nodiscard]] constexpr AudioChannelLayout layoutForChannels(oa::U32 inChannels) {
	switch (inChannels) {
		case 1:  return AudioChannelLayout::Mono;
		case 2:  return AudioChannelLayout::Stereo;
		case 6:  return AudioChannelLayout::Surround51;
		case 8:  return AudioChannelLayout::Surround71;
		default: return AudioChannelLayout::Unknown;
	}
}

// ─── Audio ───────────────────────────────────────────────────────────────────
// Semantic planar float32 audio value composed over oa::Matrix. The matrix view
// is [channels, samples]; sample rate and layout stay attached to that storage.
class Audio {
public:
	Audio() = default;

	Audio(Matrix inData, oa::U32 inSampleRate, AudioChannelLayout inLayout)
		: data_(oa::move(inData))
		, sampleRate_(inSampleRate)
		, layout_(inLayout)
	{}

	[[nodiscard]] const Matrix& asMatrix() const noexcept { return data_; }
	[[nodiscard]] Matrix& asMatrix() noexcept { return data_; }

	[[nodiscard]] oa::I32 channels() const noexcept {
		const MatrixShape shape = data_.getShape();
		return shape.rank == 0 ? 0 : static_cast<oa::I32>(shape[0]);
	}
	[[nodiscard]] oa::I64 samples() const noexcept {
		const MatrixShape shape = data_.getShape();
		return shape.rank < 2 ? 0 : shape[1];
	}
	[[nodiscard]] oa::U32 sampleRate() const noexcept { return sampleRate_; }
	[[nodiscard]] AudioChannelLayout layout() const noexcept { return layout_; }
	[[nodiscard]] oa::ScalarType getDtype() const noexcept { return data_.getDtype();	}
	[[nodiscard]] bool isEmpty() const noexcept {	return data_.getShape().rank == 0; }
	[[nodiscard]] oa::F64 durationSeconds() const noexcept {
		return sampleRate_ > 0
			? static_cast<oa::F64>(samples()) / static_cast<oa::F64>(sampleRate_)
			: 0.0;
	}
	[[nodiscard]] bool validate() const noexcept {
		const MatrixShape shape = data_.getShape();
		if (shape.rank == 0) return true;
		if (shape.rank != 2 || shape[0] <= 0 || shape[1] <= 0	|| sampleRate_ == 0 || data_.getDtype() != oa::ScalarType::Float32) {
			return false;
		}
		const oa::I32 expectedChannels = channelsForLayout(layout_);
		return expectedChannels == 0 || static_cast<oa::I32>(shape[0]) == expectedChannels;
	}

private:
	Matrix data_;
	oa::U32 sampleRate_ = 44'100U;
	AudioChannelLayout layout_ = AudioChannelLayout::Mono;
};

// ─── STFT Configuration ───────────────────────────────────────────────────────
struct StftConfig {
	oa::U32 fftSize = 1024;  // Must be power of 2
	oa::U32 hopSize = 256;   // samples between frames
	oa::U32 winSize = 0;     // Window length (0 = fftSize, otherwise ≤ fftSize)
	oa::U8  window  = 0;     // 0=Hann, 1=Hamming, 2=Blackman, 3=Rect
	bool  center  = true;  // pad input by fftSize/2 on each side
};

// ─── Mel Spectrogram Configuration ────────────────────────────────────────────
struct MelConfig {
	oa::U32 fftSize   = 1024;
	oa::U32 hopSize   = 256;
	oa::U32 numMels   = 80;    // Number of mel bins
	oa::F32 fMin      = 0.0F;  // Lowest frequency (Hz)
	oa::F32 fMax      = 0.0F;  // Highest frequency (0 = sampleRate/2)
	bool  logScale  = true;  // apply log(mel + 1e-9)
	bool  normalize = false; // Per-channel instance normalization
};

// ─── MFCC Configuration ───────────────────────────────────────────────────────
struct MfccConfig {
	oa::U32 numCoeffs = 13;  // Number of cepstral coefficients
	MelConfig mel;         // Mel spectrogram config
};

// ─── Resample Configuration ───────────────────────────────────────────────────
struct ResampleConfig {
	oa::U32 outRate = 16000;       // output sample rate
	oa::U32 filterHalfWidth = 64; // Sinc filter half-width in output samples
};

// ─── Audio Normalization Configuration ────────────────────────────────────────
struct NormalizeAudioConfig {
	oa::U8  mode     = 0;      // 0=peak (max abs), 1=RMS
	oa::F32 targetDb = -3.0F;  // target level in dB
};

// ─── Biquad Coefficients ─────────────────────────────────────────────────────
// Real, a0-normalized coefficients for
// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
//                  - a1*y[n-1] - a2*y[n-2].
// biquad() applies each channel independently with zero initial state.
struct BiquadCoefficients {
	oa::F32 b0 = 1.0F;
	oa::F32 b1 = 0.0F;
	oa::F32 b2 = 0.0F;
	oa::F32 a1 = 0.0F;
	oa::F32 a2 = 0.0F;
};

} // namespace oa
