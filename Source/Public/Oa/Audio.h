// OaAudio — GPU-accelerated audio processing. Public umbrella.
//
// OaAudioBuffer  — GPU tensor [Channels, Samples] F32, plain OaMatrix alias.
// OaAudioMeta    — POD metadata (rate, channels, samples, layout).
// OaAudio        — semantic wrapper: OaMatrix + rate + layout (like OaImage).
// OaAudioDecoder — CPU decode (WAV/FLAC/MP3 via miniaudio) → GPU upload.
// OaAudioEncoder — WAV-F32 and streaming AAC-LC codec boundaries.
// OaAudioCapture — timestamped real-time F32 device input.
// OaFnAudio      — stateless DSP ops on the OaFnMatrix auto-context pattern.
//
// Typical ML pipeline:
//   auto dec = OaAudioDecoder::LoadFile("speech.wav").Unwrap();
//   auto mel = OaFnAudio::MelSpectrogram(dec.Buffer, dec.Meta());  // → OaMatrix
//   model.Forward(mel);

#pragma once

#include <Oa/Audio/Type.h>
#include <Oa/Audio/AudioDecoder.h>
#include <Oa/Audio/AudioEncoder.h>
#include <Oa/Audio/AudioCapture.h>
#include <Oa/Audio/AudioStream.h>
#include <Oa/Audio/FnAudio.h>

// ─── OaAudio ──────────────────────────────────────────────────────────────────
// Semantic audio wrapper composed over OaMatrix, following OaImage.
// Inline: accessors + layout validation only.

class OaAudio {
public:
	OaAudio() = default;

	// Build from an existing OaMatrix. Shape should be [Channels, Samples].
	OaAudio(OaMatrix InData, OaU32 InSampleRate, OaChannelLayout InLayout)
		: Data_(std::move(InData))
		, SampleRate_(InSampleRate)
		, Layout_(InLayout)
	{}

	// Access underlying tensor
	[[nodiscard]] const OaMatrix& AsMatrix() const { return Data_; }
	[[nodiscard]]       OaMatrix& AsMatrix()       { return Data_; }

	// Audio dimensions
	[[nodiscard]] OaI32 Channels() const {
		const OaMatrixShape shape = Data_.GetShape();
		return shape.Rank == 0 ? 0 : static_cast<OaI32>(shape[0]);
	}
	[[nodiscard]] OaI64 Samples() const {
		const OaMatrixShape shape = Data_.GetShape();
		return shape.Rank < 2 ? 0 : shape[1];
	}

	// Semantic metadata
	[[nodiscard]] OaU32           SampleRate() const { return SampleRate_; }
	[[nodiscard]] OaChannelLayout Layout()     const { return Layout_; }
	[[nodiscard]] OaScalarType    GetDtype()   const { return Data_.GetDtype(); }
	[[nodiscard]] bool            IsEmpty()    const { return Data_.GetShape().Rank == 0; }

	// Validation: rank-2 [Channels, Samples] with channel count matching layout.
	[[nodiscard]] bool Validate() const {
		const OaMatrixShape shape = Data_.GetShape();
		if (shape.Rank == 0) return true;  // empty audio is trivially valid
		if (shape.Rank != 2) return false;
		if (shape[0] <= 0 || shape[1] <= 0 || SampleRate_ == 0) return false;
		if (Data_.GetDtype() != OaScalarType::Float32) return false;
		const OaI32 expectedChannels = OaChannelsForLayout(Layout_);
		return expectedChannels == 0 || static_cast<OaI32>(shape[0]) == expectedChannels;
	}

private:
	OaMatrix        Data_;
	OaU32           SampleRate_ = 44100;
	OaChannelLayout Layout_     = OaChannelLayout::Mono;
};
