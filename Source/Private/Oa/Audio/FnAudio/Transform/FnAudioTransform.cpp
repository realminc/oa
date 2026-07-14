// FnAudioTransform.cpp — hand-written OaFnAudio Transform implementations
// (schema bodies = manual_context; the .gen.{h,cpp} units carry only the
// schema-derived doc comments for these ops).
//
// Design rule: compose from existing
// verified kernels wherever possible. Stft and MelFilterbank dispatch their
// dedicated audio kernels; Mfcc is a pure GEMM ride (DCT-II matrix × log-mel).

#include <Oa/Audio/FnAudio.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/KernelRegistry.h>
#include <cmath>
#include <limits>

namespace OaFnAudio {

static OaU32 DivCeil(OaU32 InA, OaU32 InB) { return (InA + InB - 1) / InB; }

static bool IsPow2(OaU32 n) { return n != 0 && (n & (n - 1)) == 0; }

// Frames for a non-centered STFT over InSamples (the kernel zero-pads reads
// past the buffer end, so a short tail still yields a valid final frame).
static OaU32 StftFrameCount(OaU32 InSamples, OaU32 InFftSize, OaU32 InHopSize) {
	if (InSamples < InFftSize) return 1;
	return 1 + (InSamples - InFftSize) / InHopSize;
}

// Build mel filterbank on CPU (HTK mel scale, triangular filters), upload.
// Shape: [NumMels, FreqBins].
static OaMatrix BuildMelFilterbank(const OaMelConfig& InCfg, OaU32 InSampleRate) {
	const OaU32 fftSize  = InCfg.FftSize;
	const OaU32 freqBins = fftSize / 2 + 1;
	const OaU32 numMels  = InCfg.NumMels;

	const OaF32 sampleRate = static_cast<OaF32>(InSampleRate);
	const OaF32 fMax       = InCfg.FMax > 0.0F ? InCfg.FMax : sampleRate * 0.5F;
	const OaF32 fMin       = InCfg.FMin;

	// Hz ↔ Mel conversions (HTK: 2595 log10(1 + f/700))
	auto hzToMel = [](float f) { return 2595.0F * std::log10(1.0F + f / 700.0F); };
	auto melToHz = [](float m) { return 700.0F * (std::pow(10.0F, m / 2595.0F) - 1.0F); };

	// Triangular filters [numMels, freqBins]
	OaVec<OaF32> fb(numMels * freqBins, 0.0F);
	float melMin = hzToMel(fMin);
	float melMax = hzToMel(fMax);

	OaVec<float> melPts(numMels + 2);
	for (OaU32 i = 0; i < numMels + 2; ++i) {
		melPts[i] = melToHz(melMin + float(i) * (melMax - melMin) / float(numMels + 1));
	}

	const float binHz = sampleRate / static_cast<float>(fftSize);
	for (OaU32 m = 0; m < numMels; ++m) {
		for (OaU32 k = 0; k < freqBins; ++k) {
			float hz = float(k) * binHz;
			float lo = melPts[m], ctr = melPts[m+1], hi = melPts[m+2];
			float w = 0.0F;
			if (hz >= lo && hz <= ctr && ctr > lo)      w = (hz - lo) / (ctr - lo);
			else if (hz > ctr && hz <= hi && hi > ctr)  w = (hi - hz) / (hi - ctr);
			fb[m * freqBins + k] = w;
		}
	}

	auto fbMat = OaFnMatrix::Empty(
		OaMatrixShape{static_cast<OaI64>(numMels), static_cast<OaI64>(freqBins)},
		OaScalarType::Float32);
	OaMemcpy(fbMat.DataAs<OaF32>(), fb.Data(), fb.Size() * sizeof(OaF32));
	return fbMat;
}

static bool IsValidMelConfig(const OaMelConfig& InCfg, OaU32 InSampleRate) {
	const OaF32 nyquist = static_cast<OaF32>(InSampleRate) * 0.5F;
	const OaF32 fMax = InCfg.FMax > 0.0F ? InCfg.FMax : nyquist;
	return InSampleRate > 0 && IsPow2(InCfg.FftSize) &&
		InCfg.FftSize >= 16 && InCfg.FftSize <= 1024 && InCfg.HopSize > 0 &&
		InCfg.NumMels > 0 && InCfg.NumMels <= 4096 &&
		std::isfinite(InCfg.FMin) && std::isfinite(fMax) &&
		InCfg.FMin >= 0.0F && InCfg.FMin < fMax && fMax <= nyquist;
}

// Orthonormal DCT-II matrix [NumCoeffs, NumMels] (scipy dct norm='ortho' —
// what librosa MFCC uses).
static OaMatrix BuildDctIiMatrix(OaU32 InNumCoeffs, OaU32 InNumMels) {
	OaVec<OaF32> d(InNumCoeffs * InNumMels, 0.0F);
	const OaF32 scale0 = std::sqrt(1.0F / static_cast<OaF32>(InNumMels));
	const OaF32 scale  = std::sqrt(2.0F / static_cast<OaF32>(InNumMels));
	for (OaU32 k = 0; k < InNumCoeffs; ++k) {
		for (OaU32 m = 0; m < InNumMels; ++m) {
			const OaF32 v = std::cos(
				static_cast<OaF32>(M_PI) / static_cast<OaF32>(InNumMels)
				* (static_cast<OaF32>(m) + 0.5F) * static_cast<OaF32>(k));
			d[k * InNumMels + m] = v * (k == 0 ? scale0 : scale);
		}
	}
	auto dct = OaFnMatrix::Empty(
		OaMatrixShape{static_cast<OaI64>(InNumCoeffs), static_cast<OaI64>(InNumMels)},
		OaScalarType::Float32);
	OaMemcpy(dct.DataAs<OaF32>(), d.Data(), d.Size() * sizeof(OaF32));
	return dct;
}

// Mel spectrogram in the MelFilterbank kernel's native layout [C, Frames, Mels].
// Public MelSpectrogram/Mfcc transpose/reshape from here.
static OaMatrix MelNative(const OaAudioBuffer& InBuf, OaU32 InSampleRate, const OaMelConfig& InCfg)
{
	if (!IsValidMelConfig(InCfg, InSampleRate)) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::MelSpectrogram: invalid sample rate or mel configuration");
		return {};
	}
	OaStftConfig stftCfg{};
	stftCfg.FftSize = InCfg.FftSize;
	stftCfg.HopSize = InCfg.HopSize;
	stftCfg.WinSize = InCfg.FftSize;
	OaMatrix spec = Stft(InBuf, stftCfg);   // [C, Frames, FreqBins]
	if (spec.GetShape().Rank != 3) return {};

	const OaU32 channels = static_cast<OaU32>(spec.GetShape()[0]);
	const OaU32 frames   = static_cast<OaU32>(spec.GetShape()[1]);
	const OaU32 freqBins = static_cast<OaU32>(spec.GetShape()[2]);
	const OaU32 numMels  = InCfg.NumMels;

	OaMatrix fb  = BuildMelFilterbank(InCfg, InSampleRate);
	OaMatrix out = OaFnMatrix::Empty(
		OaMatrixShape{channels, frames, numMels}, OaScalarType::Float32);

	auto& ctx = OaContext::GetDefault();
	struct {
		OaU32 Channels; OaU32 Frames; OaU32 FreqBins; OaU32 NumMels; OaU32 LogScale;
	} push{.Channels = channels, .Frames = frames, .FreqBins = freqBins,
	       .NumMels = numMels, .LogScale = InCfg.LogScale ? 1U : 0U};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write, OaBufferAccess::Read};
	ctx.Add("MelFilterbank", {&spec, &out, &fb}, access, &push, sizeof(push),
	        DivCeil(numMels, 32), frames, channels);
	return out;
}


OaMatrix Stft(const OaAudioBuffer& InBuf, const OaStftConfig& InCfg) {
	const auto& shape = InBuf.GetShape();
	if (shape.Rank != 2) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Stft: expected [Channels, Samples], rank=%d", shape.Rank);
		return {};
	}
	if (!IsPow2(InCfg.FftSize) || InCfg.FftSize < 16 || InCfg.FftSize > 1024) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Stft: FftSize must be pow2 in [16, 1024], got %u", InCfg.FftSize);
		return {};
	}
	if (InCfg.HopSize == 0) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Stft: HopSize must be > 0");
		return {};
	}
	const OaU32 winSize = InCfg.WinSize == 0 ? InCfg.FftSize : InCfg.WinSize;
	if (winSize > InCfg.FftSize || InCfg.Window > 3) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaFnAudio::Stft: WinSize must be 0 or <= FftSize and Window in [0, 3]");
		return {};
	}
	if (shape[0] <= 0 || shape[1] <= 0 || InBuf.GetDtype() != OaScalarType::Float32 ||
		shape[0] > std::numeric_limits<OaU32>::max() || shape[1] > std::numeric_limits<OaU32>::max()) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Stft: expected non-empty u32-addressable F32 audio");
		return {};
	}

	// Center: pad FftSize/2 zeros on both sides (librosa convention), composed
	// from Zeros + Concat — no dedicated pad kernel needed.
	OaMatrix src = InBuf;
	if (InCfg.Center) {
		const OaI64 pad = static_cast<OaI64>(InCfg.FftSize) / 2;
		auto z1 = OaFnMatrix::Zeros(OaMatrixShape{shape[0], pad}, OaScalarType::Float32);
		auto z2 = OaFnMatrix::Zeros(OaMatrixShape{shape[0], pad}, OaScalarType::Float32);
		OaMatrix parts[] = {z1, InBuf, z2};
		src = OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 3), 1);
	}

	const OaU32 channels = static_cast<OaU32>(src.GetShape()[0]);
	const OaU32 samples  = static_cast<OaU32>(src.GetShape()[1]);
	const OaU32 frames   = StftFrameCount(samples, InCfg.FftSize, InCfg.HopSize);
	const OaU32 freqBins = InCfg.FftSize / 2 + 1;

	OaMatrix out = OaFnMatrix::Empty(
		OaMatrixShape{channels, frames, freqBins}, OaScalarType::Float32);

	auto& ctx = OaContext::GetDefault();
	struct {
		OaU32 FftSize; OaU32 HopSize; OaU32 WinSize; OaU32 Window;
		OaU32 Channels; OaU32 Samples; OaU32 Frames; OaU32 FreqBins;
	} push{.FftSize = InCfg.FftSize, .HopSize = InCfg.HopSize,
	       .WinSize = winSize, .Window = InCfg.Window, .Channels = channels,
	       .Samples = samples, .Frames = frames, .FreqBins = freqBins};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	ctx.Add("Stft", {&src, &out}, access, &push, sizeof(push), frames, 1, channels);
	return out;
}

OaMatrix MelSpectrogram(const OaAudioBuffer& InBuf, OaU32 InSampleRate, const OaMelConfig& InCfg) {
	OaMatrix mel = MelNative(InBuf, InSampleRate, InCfg);   // [C, Frames, Mels]
	if (mel.GetShape().Rank != 3) return {};
	// Whisper/CLAP layout: [C, NumMels, Frames].
	OaMatrix out = OaFnMatrix::Transpose(mel, 1, 2);
	if (!InCfg.Normalize) return out;

	const OaI64 channels = out.GetShape()[0];
	const OaI64 valuesPerChannel = out.GetShape()[1] * out.GetShape()[2];
	OaMatrix flat = OaFnMatrix::Reshape(out, OaMatrixShape{channels, valuesPerChannel});
	OaMatrix mean = OaFnMatrix::Mean(flat, 1);
	OaMatrix centered = OaFnMatrix::Sub(flat, mean);
	OaMatrix variance = OaFnMatrix::Mean(OaFnMatrix::Mul(centered, centered), 1);
	OaMatrix normalized = OaFnMatrix::Div(
		centered, OaFnMatrix::Sqrt(OaFnMatrix::AddScalar(variance, 1e-5F)));
	return OaFnMatrix::Reshape(normalized, out.GetShape());
}

OaMatrix Mfcc(const OaAudioBuffer& InBuf, OaU32 InSampleRate, const OaMfccConfig& InCfg) {
	if (InCfg.NumCoeffs == 0 || InCfg.NumCoeffs > InCfg.Mel.NumMels) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaFnAudio::Mfcc: NumCoeffs must be in [1, NumMels]");
		return {};
	}
	OaMelConfig melCfg = InCfg.Mel;
	melCfg.LogScale = true;   // MFCC is DCT of the LOG mel spectrogram
	OaMatrix mel = MelNative(InBuf, InSampleRate, melCfg);  // [C, Frames, Mels]
	if (mel.GetShape().Rank != 3) return {};

	const OaI64 channels = mel.GetShape()[0];
	const OaI64 frames   = mel.GetShape()[1];
	const OaI64 numMels  = mel.GetShape()[2];
	const OaI64 numCoeffs = static_cast<OaI64>(InCfg.NumCoeffs);

	// [C*F, M] × DCTᵀ[M, K] via MatMulNt → [C*F, K], then back to [C, K, F].
	OaMatrix dct  = BuildDctIiMatrix(InCfg.NumCoeffs, static_cast<OaU32>(numMels));
	OaMatrix flat = OaFnMatrix::Reshape(mel, OaMatrixShape{channels * frames, numMels});
	OaMatrix coef = OaFnMatrix::MatMulNt(flat, dct);
	OaMatrix c3   = OaFnMatrix::Reshape(coef, OaMatrixShape{channels, frames, numCoeffs});
	return OaFnMatrix::Transpose(c3, 1, 2);
}

} // namespace OaFnAudio
