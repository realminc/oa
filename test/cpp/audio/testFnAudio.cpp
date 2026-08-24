// TestFnAudio — oa::FnAudio DSP ops vs in-test CPU oracles.
// Validation gates: STFT matches a float64 CPU DFT
// reference ≤1e-3 relative; Parseval energy holds; mel/MFCC match the CPU
// reference chain; signal ops match closed forms. Synthetic fixtures only.

#include "../oaTest.h"

#include <oa/audio.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/semanticGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

oa::Engine* GRt = nullptr;

class TestFnAudio : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		oa::EngineConfig cfg{};
		cfg.appName = "TestFnAudio";
		auto r = oa::Engine::create(cfg);
		ASSERT_TRUE(r.isOk()) << r.getStatus().getMessage();
		static oa::UniquePtr<oa::Engine> rt = std::move(*r);
		GRt = rt.get();
	}

	static void sync() {
		auto& ctx = oa::ExecutionSession::getActive();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	}
};

std::vector<oa::F32> makeSine(oa::U32 inRate, oa::F32 inFreqHz, oa::U32 inSamples)
{
	std::vector<oa::F32> x(inSamples);
	for (oa::U32 i = 0; i < inSamples; ++i) {
		x[i] = std::sin(2.0F * 3.14159265F * inFreqHz * float(i) / float(inRate));
	}
	return x;
}

// Tone + deterministic broadband noise floor (LCG). The noise keeps every mel
// band's energy far above FP32 STFT error, so log-mel comparisons stay
// relative instead of amplifying near-silence noise.
std::vector<oa::F32> makeToneWithNoise(oa::U32 inRate, oa::F32 inFreqHz, oa::U32 inSamples)
{
	auto x = makeSine(inRate, inFreqHz, inSamples);
	oa::U32 state = 0x12345678U;
	for (oa::U32 i = 0; i < inSamples; ++i) {
		state = state * 1664525U + 1013904223U;
		const oa::F32 r = (static_cast<oa::F32>(state >> 8) / 8388608.0F) - 1.0F;  // [-1, 1)
		x[i] = 0.7F * x[i] + 0.05F * r;
	}
	return x;
}

oa::BiquadCoefficients makeLowpassBiquad(
	oa::F32 inSampleRate,
	oa::F32 inFrequency,
	oa::F32 inQ) {
	constexpr oa::F32 pi = 3.14159265358979323846F;
	const oa::F32 omega = 2.0F * pi * inFrequency / inSampleRate;
	const oa::F32 cosine = std::cos(omega);
	const oa::F32 alpha = std::sin(omega) / (2.0F * inQ);
	const oa::F32 a0 = 1.0F + alpha;
	return oa::BiquadCoefficients{
		.b0 = ((1.0F - cosine) * 0.5F) / a0,
		.b1 = (1.0F - cosine) / a0,
		.b2 = ((1.0F - cosine) * 0.5F) / a0,
		.a1 = (-2.0F * cosine) / a0,
		.a2 = (1.0F - alpha) / a0,
	};
}

oa::BiquadCoefficients makeHighpassBiquad(
	oa::F32 inSampleRate,
	oa::F32 inFrequency,
	oa::F32 inQ) {
	constexpr oa::F32 pi = 3.14159265358979323846F;
	const oa::F32 omega = 2.0F * pi * inFrequency / inSampleRate;
	const oa::F32 cosine = std::cos(omega);
	const oa::F32 alpha = std::sin(omega) / (2.0F * inQ);
	const oa::F32 a0 = 1.0F + alpha;
	return oa::BiquadCoefficients{
		.b0 = ((1.0F + cosine) * 0.5F) / a0,
		.b1 = -(1.0F + cosine) / a0,
		.b2 = ((1.0F + cosine) * 0.5F) / a0,
		.a1 = (-2.0F * cosine) / a0,
		.a2 = (1.0F - alpha) / a0,
	};
}

std::vector<oa::F32> cpuBiquadPlanar(
	const std::vector<oa::F32>& inSamples,
	oa::U32 inChannels,
	oa::U32 inSamplesPerChannel,
	const oa::BiquadCoefficients& inCoefficients) {
	std::vector<oa::F32> output(inSamples.size(), 0.0F);
	for (oa::U32 channel = 0; channel < inChannels; ++channel) {
		oa::F32 state0 = 0.0F;
		oa::F32 state1 = 0.0F;
		for (oa::U32 sample = 0; sample < inSamplesPerChannel; ++sample) {
			const oa::U32 index = channel * inSamplesPerChannel + sample;
			const oa::F32 x = inSamples[index];
			const oa::F32 y = inCoefficients.b0 * x + state0;
			const oa::F32 next0 = inCoefficients.b1 * x
				- inCoefficients.a1 * y + state1;
			const oa::F32 next1 = inCoefficients.b2 * x
				- inCoefficients.a2 * y;
			output[index] = y;
			state0 = next0;
			state1 = next1;
		}
	}
	return output;
}

std::vector<oa::F32> cpuReverbPlanar(
	const std::vector<oa::F32>& inSamples,
	oa::U32 inChannels,
	oa::U32 inSamplesPerChannel,
	oa::U32 inSampleRate,
	oa::F32 inDecaySeconds,
	oa::F32 inWet) {
	constexpr std::array<double, 4> combDelaySeconds{
		0.0297, 0.0371, 0.0411, 0.0437};
	constexpr std::array<double, 2> allpassDelaySeconds{0.0050, 0.0017};
	constexpr oa::F32 allpassGain = 0.7F;
	const oa::U32 tailSamples = static_cast<oa::U32>(std::ceil(
		static_cast<double>(inSampleRate) * inDecaySeconds));
	const oa::U32 outSamples = inSamplesPerChannel + tailSamples;

	std::array<std::vector<oa::F32>, 4> combs;
	double normalizationDenominator = 0.0;
	for (oa::Usize comb = 0; comb < combs.size(); ++comb) {
		const oa::U32 delay = std::max<oa::U32>(1U, static_cast<oa::U32>(
			std::llround(combDelaySeconds[comb] * inSampleRate)));
		const oa::F32 feedback = static_cast<oa::F32>(std::pow(
			0.001, (static_cast<double>(delay) / inSampleRate) / inDecaySeconds));
		normalizationDenominator += 1.0 / (1.0 - feedback);
		combs[comb].assign(
			static_cast<size_t>(inChannels) * outSamples, 0.0F);
		for (oa::U32 channel = 0; channel < inChannels; ++channel) {
			for (oa::U32 sample = delay; sample < outSamples; ++sample) {
				const oa::U32 source = sample - delay;
				const oa::F32 input = source < inSamplesPerChannel
					? inSamples[channel * inSamplesPerChannel + source] : 0.0F;
				combs[comb][channel * outSamples + sample] = input
					+ feedback * combs[comb][channel * outSamples + source];
			}
		}
	}

	const oa::F32 normalization = static_cast<oa::F32>(
		1.0 / normalizationDenominator);
	std::vector<oa::F32> diffused(
		static_cast<size_t>(inChannels) * outSamples, 0.0F);
	for (oa::U32 index = 0; index < diffused.size(); ++index) {
		for (const auto& comb : combs) diffused[index] += comb[index];
		diffused[index] *= normalization;
	}
	for (const double delaySeconds : allpassDelaySeconds) {
		const oa::U32 delay = std::max<oa::U32>(1U, static_cast<oa::U32>(
			std::llround(delaySeconds * inSampleRate)));
		std::vector<oa::F32> output(diffused.size(), 0.0F);
		for (oa::U32 channel = 0; channel < inChannels; ++channel) {
			const oa::U32 base = channel * outSamples;
			for (oa::U32 sample = 0; sample < outSamples; ++sample) {
				const oa::F32 delayedInput = sample >= delay
					? diffused[base + sample - delay] : 0.0F;
				const oa::F32 delayedOutput = sample >= delay
					? output[base + sample - delay] : 0.0F;
				output[base + sample] = -allpassGain * diffused[base + sample]
					+ delayedInput + allpassGain * delayedOutput;
			}
		}
		diffused = std::move(output);
	}

	std::vector<oa::F32> output(diffused.size(), 0.0F);
	for (oa::U32 channel = 0; channel < inChannels; ++channel) {
		for (oa::U32 sample = 0; sample < outSamples; ++sample) {
			const oa::F32 dry = sample < inSamplesPerChannel
				? inSamples[channel * inSamplesPerChannel + sample] : 0.0F;
			const oa::U32 index = channel * outSamples + sample;
			output[index] = dry + (diffused[index] - dry) * inWet;
		}
	}
	return output;
}

oa::Audio uploadMono(
	const std::vector<oa::F32>& inX,
	oa::U32 inSampleRate = 48'000U)
{
	auto m = oa::FnMatrix::empty(
		oa::MatrixShape{1, static_cast<oa::I64>(inX.size())}, oa::ScalarType::Float32);
	oa::memcpy(m.dataAs<oa::F32>(), inX.data(), inX.size() * sizeof(oa::F32));
	return oa::Audio(oa::move(m), inSampleRate, oa::AudioChannelLayout::Mono);
}

// Float64 STFT magnitude reference: periodic Hann, naive DFT, optional
// librosa-style center padding. layout [frames][bins].
void cpuStftMono(const std::vector<oa::F32>& inX, oa::U32 inFft, oa::U32 inHop, bool inCenter,
                 std::vector<double>& outMag, oa::U32& outFrames, oa::U32& outBins,
				 oa::U8 inWindow = 0, oa::U32 inWinSize = 0)
{
	if (inWinSize == 0) inWinSize = inFft;
	std::vector<double> s;
	if (inCenter) s.assign(inFft / 2, 0.0);
	s.insert(s.end(), inX.begin(), inX.end());
	if (inCenter) s.insert(s.end(), inFft / 2, 0.0);

	outBins   = inFft / 2 + 1;
	outFrames = s.size() >= inFft
		? 1 + static_cast<oa::U32>((s.size() - inFft) / inHop) : 1;
	outMag.assign(static_cast<size_t>(outFrames) * outBins, 0.0);

	const double pi = 3.14159265358979323846;
	for (oa::U32 f = 0; f < outFrames; ++f) {
		for (oa::U32 k = 0; k < outBins; ++k) {
			double re = 0.0, im = 0.0;
			for (oa::U32 n = 0; n < inFft; ++n) {
				const size_t idx = static_cast<size_t>(f) * inHop + n;
				const double v  = idx < s.size() ? s[idx] : 0.0;
				double w = 0.0;
				if (n < inWinSize) {
					const double phase = 2.0 * pi * n / inWinSize;
					if (inWindow == 1) w = 0.54 - 0.46 * std::cos(phase);
					else if (inWindow == 2) w = 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
					else if (inWindow == 3) w = 1.0;
					else w = 0.5 * (1.0 - std::cos(phase));
				}
				const double a  = -2.0 * pi * double(k) * double(n) / double(inFft);
				re += v * w * std::cos(a);
				im += v * w * std::sin(a);
			}
			outMag[static_cast<size_t>(f) * outBins + k] = std::sqrt(re * re + im * im);
		}
	}
}

// CPU HTK mel filterbank [Mels][bins] — mirrors the implementation so the
// GPU apply + layout are what's under test; filter construction sanity is
// covered by the peak-bin check in MelSinePeaksAtExpectedBin.
std::vector<double> cpuMelFilterbank(oa::U32 inFft, oa::U32 inRate, oa::U32 inMels)
{
	const oa::U32 bins = inFft / 2 + 1;
	auto hzToMel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
	auto melToHz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
	const double melMin = hzToMel(0.0), melMax = hzToMel(inRate * 0.5);

	std::vector<double> pts(inMels + 2);
	for (oa::U32 i = 0; i < inMels + 2; ++i) {
		pts[i] = melToHz(melMin + double(i) * (melMax - melMin) / double(inMels + 1));
	}
	std::vector<double> fb(static_cast<size_t>(inMels) * bins, 0.0);
	const double binHz = double(inRate) / double(inFft);
	for (oa::U32 m = 0; m < inMels; ++m) {
		for (oa::U32 k = 0; k < bins; ++k) {
			const double hz = k * binHz;
			const double lo = pts[m], ctr = pts[m+1], hi = pts[m+2];
			double w = 0.0;
			if (hz >= lo && hz <= ctr && ctr > lo)      w = (hz - lo) / (ctr - lo);
			else if (hz > ctr && hz <= hi && hi > ctr)  w = (hi - hz) / (hi - ctr);
			fb[static_cast<size_t>(m) * bins + k] = w;
		}
	}
	return fb;
}

} // namespace

// ─── STFT ─────────────────────────────────────────────────────────────────────

TEST_VK(TestFnAudio, StftMatchesCpuDft)
{
	const oa::U32 rate = 22050, fft = 512, hop = 128, n = 2048;
	auto x = makeSine(rate, 440.0F, n);
	auto buf = uploadMono(x, rate);

	oa::StftConfig cfg{};
	cfg.fftSize = fft; cfg.hopSize = hop; cfg.winSize = fft; cfg.center = true;
	oa::Matrix spec = oa::FnAudio::stft(buf, cfg);
	sync();

	oa::U32 frames = 0, bins = 0;
	std::vector<double> ref;
	cpuStftMono(x, fft, hop, true, ref, frames, bins);

	ASSERT_EQ(spec.getShape().rank, 3);
	ASSERT_EQ(spec.getShape()[0], 1);
	ASSERT_EQ(spec.getShape()[1], static_cast<oa::I64>(frames));
	ASSERT_EQ(spec.getShape()[2], static_cast<oa::I64>(bins));

	for (oa::U32 f = 0; f < frames; ++f) {
		for (oa::U32 k = 0; k < bins; ++k) {
			const oa::I64 idx = static_cast<oa::I64>(f) * bins + k;
			const double cpu = ref[static_cast<size_t>(idx)];
			const double gpu = spec.at(idx);
			ASSERT_NEAR(gpu, cpu, 1e-3 * std::max(1.0, cpu))
				<< "frame " << f << " bin " << k;
		}
	}
}

TEST_VK(TestFnAudio, StftParsevalEnergy)
{
	const oa::U32 rate = 22050, fft = 256, hop = 256, n = 1024;  // disjoint frames
	auto x = makeSine(rate, 993.0F, n);   // deliberately off-bin frequency
	auto buf = uploadMono(x, rate);

	oa::StftConfig cfg{};
	cfg.fftSize = fft; cfg.hopSize = hop; cfg.winSize = fft; cfg.center = false;
	oa::Matrix spec = oa::FnAudio::stft(buf, cfg);
	sync();

	const oa::U32 frames = static_cast<oa::U32>(spec.getShape()[1]);
	const oa::U32 bins   = static_cast<oa::U32>(spec.getShape()[2]);
	const double pi = 3.14159265358979323846;

	for (oa::U32 f = 0; f < frames; ++f) {
		// Frequency-domain energy: Σ|X_k|² over the full N bins — interior
		// positive-frequency bins count twice (conjugate symmetry).
		double freqEnergy = 0.0;
		for (oa::U32 k = 0; k < bins; ++k) {
			const double m = spec.at(static_cast<oa::I64>(f) * bins + k);
			const double mult = (k == 0 || k == fft / 2) ? 1.0 : 2.0;
			freqEnergy += mult * m * m;
		}
		// time-domain energy of the windowed frame, scaled by N (DFT Parseval).
		double timeEnergy = 0.0;
		for (oa::U32 nn = 0; nn < fft; ++nn) {
			const size_t idx = static_cast<size_t>(f) * hop + nn;
			const double v = idx < x.size() ? x[idx] : 0.0;
			const double w = 0.5 * (1.0 - std::cos(2.0 * pi * nn / fft));
			timeEnergy += (v * w) * (v * w);
		}
		timeEnergy *= double(fft);
		ASSERT_NEAR(freqEnergy, timeEnergy, 1e-3 * std::max(1.0, timeEnergy))
			<< "frame " << f;
	}
}

TEST_VK(TestFnAudio, StftSupportsConfiguredWindowsAndShortInput)
{
	const oa::U32 rate = 16000, fft = 256, hop = 64, win = 128, n = 91;
	auto x = makeToneWithNoise(rate, 700.0F, n);
	auto buf = uploadMono(x, rate);

	for (oa::U8 window : {oa::U8(1), oa::U8(2), oa::U8(3)}) {
		oa::StftConfig cfg{};
		cfg.fftSize = fft;
		cfg.hopSize = hop;
		cfg.winSize = win;
		cfg.window = window;
		cfg.center = false;
		oa::Matrix spec = oa::FnAudio::stft(buf, cfg);
		sync();

		oa::U32 frames = 0, bins = 0;
		std::vector<double> ref;
		cpuStftMono(x, fft, hop, false, ref, frames, bins, window, win);
		ASSERT_EQ(spec.getShape()[1], 1);
		for (oa::U32 k = 0; k < bins; ++k) {
			ASSERT_NEAR(spec.at(k), ref[k], 1e-3 * std::max(1.0, ref[k]))
				<< "window " << unsigned(window) << " bin " << k;
		}
	}
}

// ─── Mel / MFCC ───────────────────────────────────────────────────────────────

TEST_VK(TestFnAudio, MelSpectrogramMatchesCpuChain)
{
	const oa::U32 rate = 16000, fft = 512, hop = 160, mels = 80, n = 4000;
	auto x = makeToneWithNoise(rate, 440.0F, n);
	auto buf = uploadMono(x, rate);

	oa::MelConfig cfg{};
	cfg.fftSize = fft; cfg.hopSize = hop; cfg.numMels = mels; cfg.logScale = true;
	oa::Matrix mel = oa::FnAudio::melSpectrogram(buf, cfg);
	sync();

	oa::U32 frames = 0, bins = 0;
	std::vector<double> spec;
	cpuStftMono(x, fft, hop, true, spec, frames, bins);
	auto fb = cpuMelFilterbank(fft, rate, mels);

	// Whisper layout: [1, Mels, frames]
	ASSERT_EQ(mel.getShape().rank, 3);
	ASSERT_EQ(mel.getShape()[0], 1);
	ASSERT_EQ(mel.getShape()[1], static_cast<oa::I64>(mels));
	ASSERT_EQ(mel.getShape()[2], static_cast<oa::I64>(frames));

	for (oa::U32 m = 0; m < mels; ++m) {
		for (oa::U32 f = 0; f < frames; ++f) {
			double acc = 0.0;
			for (oa::U32 k = 0; k < bins; ++k) {
				acc += spec[static_cast<size_t>(f) * bins + k]
				     * fb[static_cast<size_t>(m) * bins + k];
			}
			const double cpu = std::log(acc + 1e-9);
			const double gpu = mel.at(static_cast<oa::I64>(m) * frames + f);
			ASSERT_NEAR(gpu, cpu, 1e-2 * std::max(1.0, std::abs(cpu)))
				<< "mel " << m << " frame " << f;
		}
	}
}

TEST_VK(TestFnAudio, MfccMatchesCpuChain)
{
	const oa::U32 rate = 16000, fft = 512, hop = 160, mels = 40, coeffs = 13, n = 3200;
	auto x = makeToneWithNoise(rate, 440.0F, n);
	auto buf = uploadMono(x, rate);

	oa::MfccConfig cfg{};
	cfg.numCoeffs = coeffs;
	cfg.mel.fftSize = fft; cfg.mel.hopSize = hop; cfg.mel.numMels = mels;
	oa::Matrix mfcc = oa::FnAudio::mfcc(buf, cfg);
	sync();

	oa::U32 frames = 0, bins = 0;
	std::vector<double> spec;
	cpuStftMono(x, fft, hop, true, spec, frames, bins);
	auto fb = cpuMelFilterbank(fft, rate, mels);

	ASSERT_EQ(mfcc.getShape().rank, 3);
	ASSERT_EQ(mfcc.getShape()[0], 1);
	ASSERT_EQ(mfcc.getShape()[1], static_cast<oa::I64>(coeffs));
	ASSERT_EQ(mfcc.getShape()[2], static_cast<oa::I64>(frames));

	const double pi = 3.14159265358979323846;
	std::vector<double> logMel(static_cast<size_t>(frames) * mels);
	for (oa::U32 f = 0; f < frames; ++f) {
		for (oa::U32 m = 0; m < mels; ++m) {
			double acc = 0.0;
			for (oa::U32 k = 0; k < bins; ++k) {
				acc += spec[static_cast<size_t>(f) * bins + k]
				     * fb[static_cast<size_t>(m) * bins + k];
			}
			logMel[static_cast<size_t>(f) * mels + m] = std::log(acc + 1e-9);
		}
	}
	for (oa::U32 k = 0; k < coeffs; ++k) {
		const double scale = k == 0 ? std::sqrt(1.0 / mels) : std::sqrt(2.0 / mels);
		for (oa::U32 f = 0; f < frames; ++f) {
			double acc = 0.0;
			for (oa::U32 m = 0; m < mels; ++m) {
				acc += logMel[static_cast<size_t>(f) * mels + m]
				     * std::cos(pi / mels * (m + 0.5) * k);
			}
			const double cpu = acc * scale;
			const double gpu = mfcc.at(static_cast<oa::I64>(k) * frames + f);
			ASSERT_NEAR(gpu, cpu, 1e-2 * std::max(1.0, std::abs(cpu)))
				<< "coeff " << k << " frame " << f;
		}
	}
}

TEST_VK(TestFnAudio, MelNormalizationIsPerChannelZeroMeanUnitVariance)
{
	const oa::U32 rate = 16000;
	auto x = makeToneWithNoise(rate, 440.0F, 3200);
	auto buf = uploadMono(x, rate);
	oa::MelConfig cfg{};
	cfg.fftSize = 256;
	cfg.hopSize = 80;
	cfg.numMels = 32;
	cfg.normalize = true;
	oa::Matrix mel = oa::FnAudio::melSpectrogram(buf, cfg);
	sync();

	double mean = 0.0;
	for (oa::I64 i = 0; i < mel.numElements(); ++i) mean += mel.at(i);
	mean /= static_cast<double>(mel.numElements());
	double variance = 0.0;
	for (oa::I64 i = 0; i < mel.numElements(); ++i) {
		const double d = mel.at(i) - mean;
		variance += d * d;
	}
	variance /= static_cast<double>(mel.numElements());
	EXPECT_NEAR(mean, 0.0, 1e-4);
	EXPECT_NEAR(variance, 1.0, 1e-3);
}

// ─── Signal ops ───────────────────────────────────────────────────────────────

TEST_VK(TestFnAudio, WaveformEnvelopePreservesMultichannelPeaks)
{
	const oa::F32 samples[] = {
		-0.5F, 0.2F, -0.1F, 0.7F, -0.9F, 0.3F, 0.1F, 0.4F,
		 0.1F, 0.8F, -0.4F, 0.2F, -0.2F, 0.6F, -0.7F, 0.5F,
	};
	oa::Matrix input = oa::FnMatrix::empty(
		oa::MatrixShape{2, 8}, oa::ScalarType::Float32);
	oa::memcpy(input.dataAs<oa::F32>(), samples, sizeof(samples));
	oa::Audio audio(
		oa::move(input), 48'000U, oa::AudioChannelLayout::Stereo);
	oa::Matrix envelope = oa::FnAudio::waveformEnvelope(audio, 4U);
	sync();
	ASSERT_EQ(envelope.getShape(), (oa::MatrixShape{4, 2}));
	const oa::F32 expected[] = {
		-0.5F, 0.8F,
		-0.4F, 0.7F,
		-0.9F, 0.6F,
		-0.7F, 0.5F,
	};
	for (oa::I64 i = 0; i < 8; ++i) {
		EXPECT_FLOAT_EQ(envelope.at(i), expected[i]) << "envelope element " << i;
	}
}

TEST_VK(TestFnAudio, NormalizePeakHitsTarget)
{
	std::vector<oa::F32> x(1024);
	for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25F * float(i) / float(x.size());
	auto buf = uploadMono(x);

	oa::Audio normalized = oa::FnAudio::normalize(buf, -3.0F, 0);
	const oa::Matrix& out = normalized.asMatrix();
	sync();
	EXPECT_EQ(normalized.sampleRate(), buf.sampleRate());
	EXPECT_EQ(normalized.layout(), buf.layout());
	oa::F32 peak = 0.0F;
	for (oa::I64 i = 0; i < 1024; ++i) peak = std::max(peak, std::abs(out.at(i)));
	EXPECT_NEAR(peak, std::pow(10.0F, -3.0F / 20.0F), 1e-4F);
}

TEST_VK(TestFnAudio, NormalizeRmsHitsTarget)
{
	auto x = makeSine(48000, 440.0F, 4800);
	auto buf = uploadMono(x);

	oa::Audio normalized = oa::FnAudio::normalize(buf, -20.0F, 1);
	const oa::Matrix& out = normalized.asMatrix();
	sync();
	double sq = 0.0;
	for (oa::I64 i = 0; i < 4800; ++i) { const double v = out.at(i); sq += v * v; }
	const double rms = std::sqrt(sq / 4800.0);
	EXPECT_NEAR(rms, std::pow(10.0, -20.0 / 20.0), 1e-4);
}

TEST_VK(TestFnAudio, NormalizeSilenceRemainsFiniteSilence)
{
	auto silence = oa::FnMatrix::zeros(oa::MatrixShape{2, 128}, oa::ScalarType::Float32);
	oa::Audio audio(
		oa::move(silence), 48'000U, oa::AudioChannelLayout::Stereo);
	oa::Audio normalized = oa::FnAudio::normalize(audio, -3.0F, 0);
	const oa::Matrix& out = normalized.asMatrix();
	sync();
	for (oa::I64 i = 0; i < out.numElements(); ++i) EXPECT_FLOAT_EQ(out.at(i), 0.0F);
}

TEST_VK(TestFnAudio, GainPlus6DbDoubles)
{
	auto x = makeSine(48000, 440.0F, 512);
	auto buf = uploadMono(x);
	oa::Audio gained = oa::FnAudio::gain(buf, 20.0F * std::log10(2.0F));
	const oa::Matrix& out = gained.asMatrix();
	sync();
	for (oa::I64 i = 0; i < 512; ++i) {
		ASSERT_NEAR(out.at(i), 2.0F * x[static_cast<size_t>(i)], 1e-5F) << "i=" << i;
	}
}

TEST_VK(TestFnAudio, ClipClampsRange)
{
	std::vector<oa::F32> x = {-2.0F, -0.6F, -0.5F, 0.0F, 0.4F, 0.5F, 0.9F, 3.0F};
	auto buf = uploadMono(x);
	oa::Audio clipped = oa::FnAudio::clip(buf, -0.5F, 0.5F);
	const oa::Matrix& out = clipped.asMatrix();
	sync();
	const oa::F32 expect[] = {-0.5F, -0.5F, -0.5F, 0.0F, 0.4F, 0.5F, 0.5F, 0.5F};
	for (oa::I64 i = 0; i < 8; ++i) ASSERT_FLOAT_EQ(out.at(i), expect[i]) << "i=" << i;
}

TEST_VK(TestFnAudio, SaturateMatchesCpuWaveshaper)
{
	const std::vector<oa::F32> samples{
		-2.0F, -1.0F, -0.25F, 0.0F, 0.125F, 0.75F, 1.5F};
	auto matrix = oa::FnMatrix::empty(
		oa::MatrixShape{1, static_cast<oa::I64>(samples.size())},
		oa::ScalarType::Float32);
	oa::memcpy(matrix.dataAs<oa::F32>(), samples.data(), samples.size() * sizeof(oa::F32));
	oa::Audio audio(oa::move(matrix), 48'000U, oa::AudioChannelLayout::Mono);

	constexpr oa::F32 driveDb = 6.0F;
	constexpr oa::F32 mix = 0.7F;
	oa::Audio saturated = oa::FnAudio::saturate(audio, driveDb, mix);
	sync();

	ASSERT_TRUE(saturated.validate());
	EXPECT_EQ(saturated.sampleRate(), audio.sampleRate());
	EXPECT_EQ(saturated.layout(), audio.layout());
	EXPECT_EQ(saturated.asMatrix().getShape(), audio.asMatrix().getShape());
	const oa::F32 drive = std::pow(10.0F, driveDb / 20.0F);
	for (oa::U32 i = 0; i < samples.size(); ++i) {
		const oa::F32 expected = samples[i]
			+ (std::tanh(samples[i] * drive) - samples[i]) * mix;
		EXPECT_NEAR(saturated.asMatrix().at(i), expected, 2.0e-6F)
			<< "sample " << i;
	}
}

TEST_VK(TestFnAudio, ReverbMatchesSchroederCpuOracleAndOwnsOneOperation)
{
	constexpr oa::U32 channels = 2U;
	constexpr oa::U32 samplesPerChannel = 257U;
	constexpr oa::U32 sampleRate = 8'000U;
	constexpr oa::F32 decaySeconds = 0.125F;
	constexpr oa::F32 wet = 0.4F;
	std::vector<oa::F32> samples(channels * samplesPerChannel, 0.0F);
	samples[0] = 1.0F;
	samples[samplesPerChannel + 17U] = -0.75F;
	auto matrix = oa::FnMatrix::empty(
		oa::MatrixShape{channels, samplesPerChannel}, oa::ScalarType::Float32);
	oa::memcpy(
		matrix.dataAs<oa::F32>(), samples.data(),
		samples.size() * sizeof(oa::F32));
	oa::Audio audio(oa::move(matrix), sampleRate, oa::AudioChannelLayout::Stereo);
	const auto expected = cpuReverbPlanar(
		samples, channels, samplesPerChannel, sampleRate, decaySeconds, wet);

	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	oa::Audio reverberated = oa::FnAudio::reverb(audio, decaySeconds, wet);
	ASSERT_TRUE(reverberated.validate());
	EXPECT_EQ(reverberated.sampleRate(), sampleRate);
	EXPECT_EQ(reverberated.layout(), oa::AudioChannelLayout::Stereo);
	EXPECT_EQ(reverberated.samples(), 1'257);

	const auto* semantic = ctx.semanticGraph();
	const auto* executable = ctx.graph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_NE(executable, nullptr);
	ASSERT_EQ(semantic->operationCount(), 1U);
	ASSERT_EQ(executable->nodeCount(), 8U);
	EXPECT_EQ(semantic->operations()[0].name, "oa::FnAudio::reverb");
	ASSERT_EQ(semantic->operations()[0].attributes.size(), 2U);
	const char* expectedShaders[] = {
		"AudioReverbComb", "AudioReverbComb", "AudioReverbComb",
		"AudioReverbComb", "AudioReverbSum", "AudioReverbAllpass",
		"AudioReverbAllpass", "AudioReverbMix",
	};
	for (oa::U32 index = 0; index < executable->nodeCount(); ++index) {
		const auto& node = executable->nodes()[index];
		EXPECT_EQ(node.shader, expectedShaders[index]);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
		EXPECT_EQ(node.operation, "oa::FnAudio::reverb");
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnAudio::reverb.hash);
	}

	sync();
	ASSERT_EQ(reverberated.asMatrix().numElements(), expected.size());
	for (oa::U32 index = 0; index < expected.size(); ++index) {
		EXPECT_NEAR(reverberated.asMatrix().at(index), expected[index], 3.0e-5F)
			<< "index " << index;
	}
	ctx.clear();
}

TEST_VK(TestFnAudio, BiquadMatchesCpuAcrossBlocksAndChannels)
{
	constexpr oa::U32 channels = 2U;
	constexpr oa::U32 samplesPerChannel = 777U;
	std::vector<oa::F32> samples;
	samples.reserve(channels * samplesPerChannel);
	const auto left = makeToneWithNoise(48'000U, 440.0F, samplesPerChannel);
	const auto right = makeToneWithNoise(48'000U, 3'000.0F, samplesPerChannel);
	samples.insert(samples.end(), left.begin(), left.end());
	samples.insert(samples.end(), right.begin(), right.end());
	auto matrix = oa::FnMatrix::empty(
		oa::MatrixShape{channels, samplesPerChannel}, oa::ScalarType::Float32);
	oa::memcpy(
		matrix.dataAs<oa::F32>(), samples.data(),
		samples.size() * sizeof(oa::F32));
	oa::Audio audio(oa::move(matrix), 48'000U, oa::AudioChannelLayout::Stereo);
	const auto coefficients = makeLowpassBiquad(
		48'000.0F, 2'000.0F, 0.7071067811865475F);
	const auto expected = cpuBiquadPlanar(
		samples, channels, samplesPerChannel, coefficients);

	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	oa::Audio filtered = oa::FnAudio::biquad(audio, coefficients);
	ASSERT_TRUE(filtered.validate());
	ASSERT_EQ(filtered.sampleRate(), audio.sampleRate());
	ASSERT_EQ(filtered.layout(), audio.layout());
	ASSERT_EQ(filtered.asMatrix().getShape(), audio.asMatrix().getShape());

	const auto* semantic = ctx.semanticGraph();
	const auto* executable = ctx.graph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_NE(executable, nullptr);
	ASSERT_EQ(semantic->operationCount(), 1U);
	ASSERT_EQ(executable->nodeCount(), 3U);
	EXPECT_EQ(semantic->operations()[0].name, "oa::FnAudio::biquad");
	ASSERT_EQ(semantic->operations()[0].attributes.size(), 5U);
	EXPECT_EQ(executable->nodes()[0].shader, "AudioBiquadBlockSummary");
	EXPECT_EQ(executable->nodes()[1].shader, "AudioBiquadBlockScan");
	EXPECT_EQ(executable->nodes()[2].shader, "AudioBiquadApply");
	for (const auto& node : executable->nodes()) {
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
		EXPECT_EQ(node.operation, "oa::FnAudio::biquad");
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnAudio::biquad.hash);
	}

	sync();
	for (oa::U32 index = 0; index < expected.size(); ++index) {
		EXPECT_NEAR(filtered.asMatrix().at(index), expected[index], 2.0e-5F)
			<< "index " << index;
	}
	ctx.clear();
	oa::Audio identity = oa::FnAudio::biquad(audio);
	sync();
	for (oa::U32 index = 0; index < samples.size(); ++index) {
		EXPECT_FLOAT_EQ(identity.asMatrix().at(index), samples[index])
			<< "identity index " << index;
	}
	ctx.clear();
}

TEST_VK(TestFnAudio, SosFilterMatchesCpuCascadeAndOwnsOneOperation)
{
	constexpr oa::U32 channels = 2U;
	constexpr oa::U32 samplesPerChannel = 1'031U;
	std::vector<oa::F32> samples;
	samples.reserve(channels * samplesPerChannel);
	const auto left = makeToneWithNoise(48'000U, 440.0F, samplesPerChannel);
	const auto right = makeToneWithNoise(48'000U, 6'000.0F, samplesPerChannel);
	samples.insert(samples.end(), left.begin(), left.end());
	samples.insert(samples.end(), right.begin(), right.end());
	auto matrix = oa::FnMatrix::empty(
		oa::MatrixShape{channels, samplesPerChannel}, oa::ScalarType::Float32);
	oa::memcpy(
		matrix.dataAs<oa::F32>(), samples.data(),
		samples.size() * sizeof(oa::F32));
	oa::Audio audio(oa::move(matrix), 48'000U, oa::AudioChannelLayout::Stereo);
	const oa::BiquadCoefficients sections[] = {
		makeHighpassBiquad(48'000.0F, 120.0F, 0.7071067811865475F),
		makeLowpassBiquad(48'000.0F, 8'000.0F, 0.7071067811865475F),
		makeLowpassBiquad(48'000.0F, 5'000.0F, 0.9F),
	};
	auto expected = samples;
	for (const auto& section : sections) {
		expected = cpuBiquadPlanar(
			expected, channels, samplesPerChannel, section);
	}

	auto& ctx = oa::ExecutionSession::getActive();
	ctx.clear();
	oa::Audio filtered = oa::FnAudio::sosFilter(audio, sections);
	ASSERT_TRUE(filtered.validate());
	ASSERT_EQ(filtered.sampleRate(), audio.sampleRate());
	ASSERT_EQ(filtered.layout(), audio.layout());
	ASSERT_EQ(filtered.asMatrix().getShape(), audio.asMatrix().getShape());

	const auto* semantic = ctx.semanticGraph();
	const auto* executable = ctx.graph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_NE(executable, nullptr);
	ASSERT_EQ(semantic->operationCount(), 1U);
	ASSERT_EQ(executable->nodeCount(), 9U);
	const auto& operation = semantic->operations()[0];
	EXPECT_EQ(operation.name, "oa::FnAudio::sosFilter");
	ASSERT_EQ(operation.attributes.size(), 2U);
	EXPECT_EQ(operation.attributes[0].name, "sectionCount");
	EXPECT_EQ(operation.attributes[0].unsignedInteger, 3U);
	EXPECT_EQ(operation.attributes[1].name, "coefficientHash");
	EXPECT_NE(operation.attributes[1].unsignedInteger, 0U);
	const char* expectedShaders[] = {
		"AudioBiquadBlockSummary",
		"AudioBiquadBlockScan",
		"AudioBiquadApply",
	};
	for (oa::U32 index = 0U; index < executable->nodeCount(); ++index) {
		const auto& node = executable->nodes()[index];
		EXPECT_EQ(node.shader, expectedShaders[index % 3U]);
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
		EXPECT_EQ(node.operation, "oa::FnAudio::sosFilter");
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnAudio::sosFilter.hash);
	}

	sync();
	for (oa::U32 index = 0; index < expected.size(); ++index) {
		EXPECT_NEAR(filtered.asMatrix().at(index), expected[index], 5.0e-5F)
			<< "index " << index;
	}
	ctx.clear();
}

TEST_VK(TestFnAudio, AmplitudeToDbKnownValues)
{
	std::vector<oa::F32> x = {1.0F, 0.1F, -0.1F, 0.0F};
	auto buf = uploadMono(x);
	oa::Matrix out = oa::FnAudio::amplitudeToDb(buf, -100.0F);
	sync();
	EXPECT_NEAR(out.at(0),    0.0F, 1e-3F);
	EXPECT_NEAR(out.at(1),  -20.0F, 1e-3F);
	EXPECT_NEAR(out.at(2),  -20.0F, 1e-3F);
	EXPECT_NEAR(out.at(3), -100.0F, 1e-3F);   // silence hits the floor
}

TEST_VK(TestFnAudio, ToMonoAveragesChannels)
{
	const oa::I64 n = 256;
	std::vector<oa::F32> host(static_cast<size_t>(2 * n));
	for (oa::I64 i = 0; i < n; ++i) { host[static_cast<size_t>(i)] = 1.0F; host[static_cast<size_t>(n + i)] = 0.0F; }
	auto buf = oa::FnMatrix::empty(oa::MatrixShape{2, n}, oa::ScalarType::Float32);
	oa::memcpy(buf.dataAs<oa::F32>(), host.data(), host.size() * sizeof(oa::F32));

	oa::Audio audio(
		oa::move(buf), 48'000U, oa::AudioChannelLayout::Stereo);
	oa::Audio monoAudio = oa::FnAudio::toMono(audio);
	const oa::Matrix& mono = monoAudio.asMatrix();
	sync();
	EXPECT_EQ(monoAudio.sampleRate(), 48'000U);
	EXPECT_EQ(monoAudio.layout(), oa::AudioChannelLayout::Mono);
	ASSERT_EQ(mono.getShape()[0], 1);
	ASSERT_EQ(mono.getShape()[1], n);
	for (oa::I64 i = 0; i < n; ++i) ASSERT_NEAR(mono.at(i), 0.5F, 1e-5F) << "i=" << i;
}

TEST_VK(TestFnAudio, ResampleMatchesCpuSinc)
{
	const oa::U32 inRate = 48000;
	const oa::U32 outRate = 16000;
	const oa::U32 halfW = 32;
	const oa::U32 n = 4800;
	auto x = makeToneWithNoise(inRate, 440.0F, n);
	auto buf = uploadMono(x, inRate);

	oa::Audio resampled = oa::FnAudio::resample(buf, outRate, halfW);
	const oa::Matrix& out = resampled.asMatrix();
	sync();
	EXPECT_EQ(resampled.sampleRate(), outRate);
	EXPECT_EQ(resampled.layout(), oa::AudioChannelLayout::Mono);

	const oa::U32 inR = 3;    // 48000/16000 gcd-reduced
	const oa::U32 outR = 1;
	const oa::U32 outSamples = (n * outR) / inR;
	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], static_cast<oa::I64>(outSamples));

	// CPU mirror of the shader math in double.
	const double pi = 3.14159265358979323846;
	const double ratio = double(outR) / double(inR);
	const double scale = std::min(1.0, ratio);
	for (oa::U32 j = 0; j < outSamples; ++j) {
		const oa::U32 q = j / outR;
		const oa::U32 r = j % outR;
		const oa::U32 num = r * inR;
		const oa::U32 n0 = (q * inR) + (num / outR);
		const double frac = double(num % outR) / double(outR);
		double acc = 0.0;
		double ksum = 0.0;
		for (int tap = -int(halfW); tap <= int(halfW); ++tap) {
			const double t = double(tap) - frac;
			if (std::abs(t) > double(halfW)) {
				continue;
			}
			const double window = 0.5 * (1.0 + std::cos(pi * t / double(halfW)));
			const double st = scale * t;
			const double sinc = std::abs(st) < 1e-12 ? 1.0 : std::sin(pi * st) / (pi * st);
			const double kernel = scale * sinc * window;
			ksum += kernel;
			const int nn = int(n0) + tap;
			if (nn >= 0 && nn < int(n)) {
				acc += double(x[static_cast<size_t>(nn)]) * kernel;
			}
		}
		const double cpu = ksum > 1e-7 ? acc / ksum : 0.0;
		ASSERT_NEAR(out.at(static_cast<oa::I64>(j)), cpu, 1e-4)
			<< "out sample " << j;
	}
}

TEST_VK(TestFnAudio, ResampleIdentityPreservesSamples)
{
	auto x = makeToneWithNoise(48000, 440.0F, 257);
	auto buf = uploadMono(x, 48'000U);
	oa::Audio resampled = oa::FnAudio::resample(
		buf, oa::ResampleConfig{.outRate = 48'000U, .filterHalfWidth = 32});
	const oa::Matrix& out = resampled.asMatrix();
	sync();
	ASSERT_EQ(out.getShape(), buf.asMatrix().getShape());
	for (oa::I64 i = 0; i < out.numElements(); ++i) EXPECT_FLOAT_EQ(out.at(i), x[static_cast<size_t>(i)]);
}

TEST_VK(TestFnAudio, ResampleDownsampleSuppressesOutOfBandTone)
{
	const oa::U32 inRate = 48000, outRate = 16000, n = 4800;
	auto x = makeSine(inRate, 12000.0F, n); // above the 8 kHz output Nyquist
	auto buf = uploadMono(x, inRate);
	oa::Audio resampled = oa::FnAudio::resample(buf, outRate, 64);
	const oa::Matrix& out = resampled.asMatrix();
	sync();
	double squareSum = 0.0;
	const oa::I64 margin = 64;
	for (oa::I64 i = margin; i < out.getShape()[1] - margin; ++i) {
		const double v = out.at(i);
		squareSum += v * v;
	}
	const double rms = std::sqrt(squareSum / double(out.getShape()[1] - 2 * margin));
	EXPECT_LT(rms, 0.02);
}

TEST_VK(TestFnAudio, InvalidConfigurationsReturnEmpty)
{
	auto buf = uploadMono(makeSine(16000, 440.0F, 512), 16'000U);
	oa::StftConfig stft{};
	stft.fftSize = 300;
	EXPECT_TRUE(oa::FnAudio::stft(buf, stft).isEmpty());

	oa::MelConfig mel{};
	mel.fMin = 9000.0F;
	EXPECT_TRUE(oa::FnAudio::melSpectrogram(buf, mel).isEmpty());

	oa::MfccConfig mfcc{};
	mfcc.numCoeffs = mfcc.mel.numMels + 1;
	EXPECT_TRUE(oa::FnAudio::mfcc(buf, mfcc).isEmpty());

	EXPECT_TRUE(oa::FnAudio::normalize(buf, -3.0F, 2).isEmpty());
	EXPECT_TRUE(oa::FnAudio::resample(buf, 0, 32).isEmpty());
	EXPECT_TRUE(oa::FnAudio::resample(buf, 16000, 2048).isEmpty());
	EXPECT_TRUE(oa::FnAudio::clip(buf, 1.0F, -1.0F).isEmpty());
	EXPECT_TRUE(oa::FnAudio::saturate(buf, 61.0F, 1.0F).isEmpty());
	EXPECT_TRUE(oa::FnAudio::saturate(buf, 0.0F, 1.01F).isEmpty());
	EXPECT_TRUE(oa::FnAudio::reverb(buf, 0.09F, 0.5F).isEmpty());
	EXPECT_TRUE(oa::FnAudio::reverb(buf, 1.0F, 1.01F).isEmpty());
	oa::BiquadCoefficients unstable{};
	unstable.a2 = 1.0F;
	EXPECT_TRUE(oa::FnAudio::biquad(buf, unstable).isEmpty());
	oa::BiquadCoefficients nonFinite{};
	nonFinite.b0 = std::numeric_limits<oa::F32>::quiet_NaN();
	EXPECT_TRUE(oa::FnAudio::biquad(buf, nonFinite).isEmpty());
	EXPECT_TRUE(oa::FnAudio::sosFilter(buf, {}).isEmpty());
	oa::BiquadCoefficients unstableSections[2]{};
	unstableSections[1].a2 = 1.0F;
	EXPECT_TRUE(oa::FnAudio::sosFilter(buf, unstableSections).isEmpty());
	oa::BiquadCoefficients tooManySections[65]{};
	EXPECT_TRUE(oa::FnAudio::sosFilter(buf, tooManySections).isEmpty());
}

TEST_VK(TestFnAudio, PreEmphasisMatchesClosedForm)
{
	const oa::U32 n = 512;
	const oa::F32 alpha = 0.97F;
	auto x = makeToneWithNoise(22050, 440.0F, n);
	auto buf = uploadMono(x, 22'050U);

	oa::Audio emphasized = oa::FnAudio::preEmphasis(buf, alpha);
	const oa::Matrix& out = emphasized.asMatrix();
	sync();

	ASSERT_EQ(out.getShape()[0], 1);
	ASSERT_EQ(out.getShape()[1], static_cast<oa::I64>(n));
	ASSERT_NEAR(out.at(0), x[0], 1e-6F);   // y[0] = x[0]
	for (oa::U32 i = 1; i < n; ++i) {
		const oa::F32 expect = x[i] - (alpha * x[i - 1]);
		ASSERT_NEAR(out.at(static_cast<oa::I64>(i)), expect, 1e-5F) << "i=" << i;
	}
}

TEST_VK(TestFnAudio, FadeAppliesLinearEnvelope)
{
	const oa::I64 n = 100;
	std::vector<oa::F32> x(static_cast<size_t>(n), 1.0F);
	auto buf = uploadMono(x);
	oa::Audio faded = oa::FnAudio::fade(buf, 10, 10);
	const oa::Matrix& out = faded.asMatrix();
	sync();
	EXPECT_FLOAT_EQ(out.at(0), 0.0F);
	EXPECT_NEAR(out.at(5),  0.5F, 1e-5F);
	EXPECT_FLOAT_EQ(out.at(50), 1.0F);
	EXPECT_NEAR(out.at(n - 6), 0.5F, 1e-5F);
	EXPECT_FLOAT_EQ(out.at(n - 1), 0.0F);
}

TEST_VK(TestFnAudio, ComposedLoweringHasOneSchemaOwner)
{
	auto& ctx = oa::ExecutionSession::getActive();
	auto audio = uploadMono(makeToneWithNoise(16'000U, 440.0F, 1024U));
	ctx.clear();

	oa::MelConfig config{};
	config.fftSize = 256U;
	config.hopSize = 64U;
	config.numMels = 24U;
	config.logScale = true;
	config.normalize = true;
	auto output = oa::FnAudio::melSpectrogram(audio, config);
	ASSERT_FALSE(output.isEmpty());

	const auto* semantic = ctx.semanticGraph();
	const auto* executable = ctx.graph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_NE(executable, nullptr);
	ASSERT_EQ(semantic->operationCount(), 1U);
	ASSERT_GT(executable->nodeCount(), 2U);
	const auto& operation = semantic->operations()[0];
	EXPECT_EQ(operation.name, "oa::FnAudio::melSpectrogram");
	ASSERT_EQ(operation.attributes.size(), 7U);
	EXPECT_EQ(operation.attributes[0].name, "fftSize");
	EXPECT_EQ(operation.attributes[0].unsignedInteger, config.fftSize);
	EXPECT_EQ(operation.attributes[6].name, "normalize");
	EXPECT_TRUE(operation.attributes[6].boolean);
	for (const auto& node : executable->nodes()) {
		ASSERT_EQ(node.semanticOps.size(), 1U);
		EXPECT_EQ(node.semanticOps[0], 0U);
		EXPECT_EQ(node.operation, "oa::FnAudio::melSpectrogram");
		EXPECT_EQ(
			node.opContractHash,
			oa::detail::opRegistry::FnAudio::melSpectrogram.hash);
	}
	ctx.clear();
}

TEST_VK(TestFnAudio, InvalidOperationDoesNotPoisonNextLowering)
{
	auto& ctx = oa::ExecutionSession::getActive();
	auto audio = uploadMono({0.25F, -0.5F, 0.75F, -1.0F});
	ctx.clear();

	EXPECT_TRUE(oa::FnAudio::normalize(audio, -3.0F, 2U).isEmpty());
	EXPECT_EQ(ctx.semanticGraph()->operationCount(), 0U);
	EXPECT_EQ(ctx.graph()->nodeCount(), 0U);

	auto gained = oa::FnAudio::gain(audio, 6.0F);
	ASSERT_FALSE(gained.isEmpty());
	ASSERT_EQ(ctx.semanticGraph()->operationCount(), 1U);
	EXPECT_EQ(ctx.semanticGraph()->operations()[0].name, "oa::FnAudio::gain");
	EXPECT_TRUE(ctx.semanticGraph()->validate().isOk());
	ctx.clear();
}

TEST_VK(TestFnAudio, ComposedLoweringRetainsNestedAutograd)
{
	auto& ctx = oa::ExecutionSession::getActive();
	auto audio = uploadMono({0.1F, -0.2F, 0.4F, 0.8F});
	audio.asMatrix().setRequiresGrad(true);
	ctx.clear();

	oa::GradientTape tape;
	auto normalized = oa::FnAudio::normalize(audio, -3.0F, 1U);
	ASSERT_FALSE(normalized.isEmpty());
	ASSERT_TRUE(normalized.asMatrix().requiresGrad());
	ASSERT_TRUE(normalized.asMatrix().getGradFn());
	auto loss = oa::FnMatrix::sum(normalized.asMatrix(), -1);
	tape.backward(loss);
	sync();

	const auto& gradient = audio.asMatrix().gradMatrix();
	ASSERT_EQ(gradient.numElements(), 4);
	bool hasNonZero = false;
	for (oa::I64 index = 0; index < gradient.numElements(); ++index) {
		const auto value = gradient.at(index);
		EXPECT_TRUE(std::isfinite(value));
		hasNonZero = hasNonZero or std::abs(value) > 1.0e-5F;
	}
	EXPECT_TRUE(hasNonZero);
	ctx.clear();
}
