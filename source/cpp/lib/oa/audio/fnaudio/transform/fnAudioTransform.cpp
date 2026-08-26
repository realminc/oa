// fnAudioTransform.cpp — hand-written oa::FnAudio transform implementations.
// Their schemas own the declarations and contracts; this file owns the
// manual_session bodies.
//
// Design rule: compose from existing
// verified kernels wherever possible. Stft and MelFilterbank dispatch their
// dedicated audio kernels; Mfcc is a pure GEMM ride (DCT-II matrix × log-mel).

#include <oa/audio/fnAudio.h>
#include <oa/core/log.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/memory.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

namespace FnAudio {

static oa::U32 divCeil(oa::U32 inA, oa::U32 inB) { return (inA + inB - 1) / inB; }

static bool isPow2(oa::U32 n) { return n != 0 && (n & (n - 1)) == 0; }

// frames for a non-centered STFT over inSamples (the kernel zero-pads reads
// past the buffer end, so a short tail still yields a valid final frame).
static oa::U32 stftFrameCount(oa::U32 inSamples, oa::U32 inFftSize, oa::U32 inHopSize) {
	if (inSamples < inFftSize) return 1;
	return 1 + (inSamples - inFftSize) / inHopSize;
}

// Build mel filterbank on CPU (HTK mel scale, triangular filters), upload.
// Shape: [numMels, freqBins].
static oa::Matrix buildMelFilterbank(const MelConfig& inCfg, oa::U32 inSampleRate) {
	const oa::U32 fftSize  = inCfg.fftSize;
	const oa::U32 freqBins = fftSize / 2 + 1;
	const oa::U32 numMels  = inCfg.numMels;

	const oa::F32 sampleRate = static_cast<oa::F32>(inSampleRate);
	const oa::F32 fMax       = inCfg.fMax > 0.0F ? inCfg.fMax : sampleRate * 0.5F;
	const oa::F32 fMin       = inCfg.fMin;

	// Hz ↔ Mel conversions (HTK: 2595 log10(1 + f/700))
	auto hzToMel = [](float f) { return 2595.0F * oa::log10(1.0F + f / 700.0F); };
	auto melToHz = [](float m) { return 700.0F * (oa::pow(10.0F, m / 2595.0F) - 1.0F); };

	// Triangular filters [numMels, freqBins]
	oa::Vec<oa::F32> fb(numMels * freqBins, 0.0F);
	float melMin = hzToMel(fMin);
	float melMax = hzToMel(fMax);

	oa::Vec<float> melPts(numMels + 2);
	for (oa::U32 i = 0; i < numMels + 2; ++i) {
		melPts[i] = melToHz(melMin + float(i) * (melMax - melMin) / float(numMels + 1));
	}

	const float binHz = sampleRate / static_cast<float>(fftSize);
	for (oa::U32 m = 0; m < numMels; ++m) {
		for (oa::U32 k = 0; k < freqBins; ++k) {
			float hz = float(k) * binHz;
			float lo = melPts[m], ctr = melPts[m+1], hi = melPts[m+2];
			float w = 0.0F;
			if (hz >= lo && hz <= ctr && ctr > lo)      w = (hz - lo) / (ctr - lo);
			else if (hz > ctr && hz <= hi && hi > ctr)  w = (hi - hz) / (hi - ctr);
			fb[m * freqBins + k] = w;
		}
	}

	auto fbMat = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(numMels), static_cast<oa::I64>(freqBins)},
		oa::ScalarType::Float32);
	oa::memcpy(fbMat.dataAs<oa::F32>(), fb.data(), fb.size() * sizeof(oa::F32));
	return fbMat;
}

static bool isValidMelConfig(const MelConfig& inCfg, oa::U32 inSampleRate) {
	const oa::F32 nyquist = static_cast<oa::F32>(inSampleRate) * 0.5F;
	const oa::F32 fMax = inCfg.fMax > 0.0F ? inCfg.fMax : nyquist;
	return inSampleRate > 0 && isPow2(inCfg.fftSize) &&
		inCfg.fftSize >= 16 && inCfg.fftSize <= 1024 && inCfg.hopSize > 0 &&
		inCfg.numMels > 0 && inCfg.numMels <= 4096 &&
		oa::isFinite(inCfg.fMin) && oa::isFinite(fMax) &&
		inCfg.fMin >= 0.0F && inCfg.fMin < fMax && fMax <= nyquist;
}

// Orthonormal DCT-II matrix [numCoeffs, numMels] (scipy dct norm='ortho' —
// what librosa MFCC uses).
static oa::Matrix buildDctIiMatrix(oa::U32 inNumCoeffs, oa::U32 inNumMels) {
	oa::Vec<oa::F32> d(inNumCoeffs * inNumMels, 0.0F);
	const oa::F32 scale0 = oa::sqrt(1.0F / static_cast<oa::F32>(inNumMels));
	const oa::F32 scale  = oa::sqrt(2.0F / static_cast<oa::F32>(inNumMels));
	for (oa::U32 k = 0; k < inNumCoeffs; ++k) {
		for (oa::U32 m = 0; m < inNumMels; ++m) {
			const oa::F32 v = oa::cos(
				oa::PiF / static_cast<oa::F32>(inNumMels)
				* (static_cast<oa::F32>(m) + 0.5F) * static_cast<oa::F32>(k));
			d[k * inNumMels + m] = v * (k == 0 ? scale0 : scale);
		}
	}
	auto dct = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(inNumCoeffs), static_cast<oa::I64>(inNumMels)},
		oa::ScalarType::Float32);
	oa::memcpy(dct.dataAs<oa::F32>(), d.data(), d.size() * sizeof(oa::F32));
	return dct;
}

// Mel spectrogram in the MelFilterbank kernel's native layout [C, frames, Mels].
// Public MelSpectrogram/Mfcc transpose/reshape from here.
static oa::Matrix melNative(const Audio& inAudio, const MelConfig& inCfg)
{
	const oa::U32 inSampleRate = inAudio.sampleRate();
	if (!isValidMelConfig(inCfg, inSampleRate)) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::melSpectrogram: invalid sample rate or mel configuration");
		return {};
	}
	StftConfig stftCfg{};
	stftCfg.fftSize = inCfg.fftSize;
	stftCfg.hopSize = inCfg.hopSize;
	stftCfg.winSize = inCfg.fftSize;
	oa::Matrix spec = stft(inAudio, stftCfg);   // [C, frames, freqBins]
	if (spec.getShape().rank != 3) return {};

	const oa::U32 channels = static_cast<oa::U32>(spec.getShape()[0]);
	const oa::U32 frames   = static_cast<oa::U32>(spec.getShape()[1]);
	const oa::U32 freqBins = static_cast<oa::U32>(spec.getShape()[2]);
	const oa::U32 numMels  = inCfg.numMels;

	oa::Matrix fb  = buildMelFilterbank(inCfg, inSampleRate);
	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{channels, frames, numMels}, oa::ScalarType::Float32);

	auto& ctx = oa::ExecutionSession::getActive();
	struct {
		oa::U32 channels; oa::U32 frames; oa::U32 freqBins; oa::U32 numMels; oa::U32 logScale;
	} push{.channels = channels, .frames = frames, .freqBins = freqBins,
	       .numMels = numMels, .logScale = inCfg.logScale ? 1U : 0U};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write, oa::BufferAccess::Read};
	ctx.add( "MelFilterbank", {&spec, &out, &fb}, access, &push, sizeof(push),
	        divCeil(numMels, 32), frames, channels);
	return out;
}


oa::Matrix stft(const Audio& inAudio, const StftConfig& inCfg) {
	if (not inAudio.validate() || inAudio.isEmpty()) {
		OaLogError(
			oa::LogComponent::Audio,
			"oa::FnAudio::stft: expected valid non-empty audio");
		return {};
	}
	const oa::Matrix& inBuf = inAudio.asMatrix();
	const auto& shape = inBuf.getShape();
	if (shape.rank != 2) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::stft: expected [channels, samples], rank=%d", shape.rank);
		return {};
	}
	if (!isPow2(inCfg.fftSize) || inCfg.fftSize < 16 || inCfg.fftSize > 1024) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::stft: fftSize must be pow2 in [16, 1024], got %u", inCfg.fftSize);
		return {};
	}
	if (inCfg.hopSize == 0) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::stft: hopSize must be > 0");
		return {};
	}
	const oa::U32 winSize = inCfg.winSize == 0 ? inCfg.fftSize : inCfg.winSize;
	if (winSize > inCfg.fftSize || inCfg.window > 3) {
		OaLogError(oa::LogComponent::Audio,
			"oa::FnAudio::stft: winSize must be 0 or <= fftSize and Window in [0, 3]");
		return {};
	}
	if (shape[0] <= 0 || shape[1] <= 0 || inBuf.getDtype() != oa::ScalarType::Float32 ||
		shape[0] > oa::Limits<oa::U32>::max() || shape[1] > oa::Limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::stft: expected non-empty u32-addressable F32 audio");
		return {};
	}

	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	// Center: pad fftSize/2 zeros on both sides (librosa convention), composed
	// from Zeros + Concat — no dedicated pad kernel needed.
	oa::Matrix src = inBuf;
	if (inCfg.center) {
		const oa::I64 pad = static_cast<oa::I64>(inCfg.fftSize) / 2;
		auto z1 = oa::FnMatrix::zeros(oa::MatrixShape{shape[0], pad}, oa::ScalarType::Float32);
		auto z2 = oa::FnMatrix::zeros(oa::MatrixShape{shape[0], pad}, oa::ScalarType::Float32);
		oa::Matrix parts[] = {z1, inBuf, z2};
		src = oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 3), 1);
	}

	const oa::U32 channels = static_cast<oa::U32>(src.getShape()[0]);
	const oa::U32 samples  = static_cast<oa::U32>(src.getShape()[1]);
	const oa::U32 frames   = stftFrameCount(samples, inCfg.fftSize, inCfg.hopSize);
	const oa::U32 freqBins = inCfg.fftSize / 2 + 1;

	oa::Matrix out = oa::FnMatrix::empty(
		oa::MatrixShape{channels, frames, freqBins}, oa::ScalarType::Float32);

	struct {
		oa::U32 fftSize; oa::U32 hopSize; oa::U32 winSize; oa::U32 window;
		oa::U32 channels; oa::U32 samples; oa::U32 frames; oa::U32 freqBins;
	} push{.fftSize = inCfg.fftSize, .hopSize = inCfg.hopSize,
	       .winSize = winSize, .window = inCfg.window, .channels = channels,
	       .samples = samples, .frames = frames, .freqBins = freqBins};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "Stft", {&src, &out}, access, &push, sizeof(push), frames, 1, channels);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::stft,
		{&inBuf}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"fftSize", inCfg.fftSize),
			oa::OpAttribute::fromUnsignedInteger(
				"hopSize", inCfg.hopSize),
			oa::OpAttribute::fromUnsignedInteger(
				"winSize", inCfg.winSize),
			oa::OpAttribute::fromUnsignedInteger(
				"window", inCfg.window),
			oa::OpAttribute::fromBoolean("center", inCfg.center),
		});
	if (not status.isOk()) return {};
	return out;
}

oa::Matrix melSpectrogram(
	const Audio& inAudio, const MelConfig& inCfg) {
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix mel = melNative(inAudio, inCfg);   // [C, frames, Mels]
	if (mel.getShape().rank != 3) return {};
	// Whisper/CLAP layout: [C, numMels, frames].
	oa::Matrix out = oa::FnMatrix::transpose(mel, 1, 2);
	if (inCfg.normalize) {
		const oa::I64 channels = out.getShape()[0];
		const oa::I64 valuesPerChannel =
			out.getShape()[1] * out.getShape()[2];
		oa::Matrix flat = oa::FnMatrix::reshape(
			out, oa::MatrixShape{channels, valuesPerChannel});
		oa::Matrix mean = oa::FnMatrix::mean(flat, 1);
		oa::Matrix centered = oa::FnMatrix::sub(flat, mean);
		oa::Matrix variance = oa::FnMatrix::mean(
			oa::FnMatrix::mul(centered, centered), 1);
		oa::Matrix normalized = oa::FnMatrix::div(
			centered,
			oa::FnMatrix::sqrt(oa::FnMatrix::addScalar(variance, 1e-5F)));
		out = oa::FnMatrix::reshape(normalized, out.getShape());
	}
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::melSpectrogram,
		{&input}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"fftSize", inCfg.fftSize),
			oa::OpAttribute::fromUnsignedInteger(
				"hopSize", inCfg.hopSize),
			oa::OpAttribute::fromUnsignedInteger(
				"numMels", inCfg.numMels),
			oa::OpAttribute::fromFloat("fMin", inCfg.fMin),
			oa::OpAttribute::fromFloat("fMax", inCfg.fMax),
			oa::OpAttribute::fromBoolean(
				"logScale", inCfg.logScale),
			oa::OpAttribute::fromBoolean(
				"normalize", inCfg.normalize),
		});
	if (not status.isOk()) return {};
	return out;
}

oa::Matrix mfcc(const Audio& inAudio, const MfccConfig& inCfg) {
	if (inCfg.numCoeffs == 0 || inCfg.numCoeffs > inCfg.mel.numMels) {
		OaLogError(oa::LogComponent::Audio, "oa::FnAudio::mfcc: numCoeffs must be in [1, numMels]");
		return {};
	}
	if (not inAudio.validate() || inAudio.isEmpty()) return {};
	const oa::Matrix& input = inAudio.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	MelConfig melCfg = inCfg.mel;
	melCfg.logScale = true;   // MFCC is DCT of the LOG mel spectrogram
	oa::Matrix mel = melNative(inAudio, melCfg);  // [C, frames, Mels]
	if (mel.getShape().rank != 3) return {};

	const oa::I64 channels = mel.getShape()[0];
	const oa::I64 frames   = mel.getShape()[1];
	const oa::I64 numMels  = mel.getShape()[2];
	const oa::I64 numCoeffs = static_cast<oa::I64>(inCfg.numCoeffs);

	// [C*F, M] × DCTᵀ[M, K] via MatMulNt → [C*F, K], then back to [C, K, F].
	oa::Matrix dct  = buildDctIiMatrix(inCfg.numCoeffs, static_cast<oa::U32>(numMels));
	oa::Matrix flat = oa::FnMatrix::reshape(mel, oa::MatrixShape{channels * frames, numMels});
	oa::Matrix coef = oa::FnMatrix::matMulNt(flat, dct);
	oa::Matrix c3   = oa::FnMatrix::reshape(coef, oa::MatrixShape{channels, frames, numCoeffs});
	oa::Matrix out = oa::FnMatrix::transpose(c3, 1, 2);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnAudio::mfcc,
		{&input}, {&out},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"numCoeffs", inCfg.numCoeffs),
			oa::OpAttribute::fromUnsignedInteger(
				"fftSize", inCfg.mel.fftSize),
			oa::OpAttribute::fromUnsignedInteger(
				"hopSize", inCfg.mel.hopSize),
			oa::OpAttribute::fromUnsignedInteger(
				"numMels", inCfg.mel.numMels),
			oa::OpAttribute::fromFloat("fMin", inCfg.mel.fMin),
			oa::OpAttribute::fromFloat("fMax", inCfg.mel.fMax),
			oa::OpAttribute::fromBoolean(
				"logScale", inCfg.mel.logScale),
			oa::OpAttribute::fromBoolean(
				"normalize", inCfg.mel.normalize),
		});
	if (not status.isOk()) return {};
	return out;
}

} // namespace FnAudio

} // namespace oa
