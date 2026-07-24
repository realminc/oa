// TestFnAudio — OaFnAudio DSP ops vs in-test CPU oracles.
// Validation gates: STFT matches a float64 CPU DFT
// reference ≤1e-3 relative; Parseval energy holds; mel/MFCC match the CPU
// reference chain; signal ops match closed forms. Synthetic fixtures only.

#include "../OaTest.h"

#include <Oa/Audio.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cmath>
#include <vector>

namespace {

OaEngine* GRt = nullptr;

class TestFnAudio : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		OaEngineConfig cfg{};
		cfg.AppName = "TestFnAudio";
		auto r = OaEngine::Create(cfg);
		ASSERT_TRUE(r.IsOk()) << r.GetStatus().GetMessage();
		static OaUniquePtr<OaEngine> rt = std::move(*r);
		GRt = rt.get();
	}

	static void Sync() {
		auto& ctx = OaContext::GetDefault();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
};

std::vector<OaF32> MakeSine(OaU32 InRate, OaF32 InFreqHz, OaU32 InSamples)
{
	std::vector<OaF32> x(InSamples);
	for (OaU32 i = 0; i < InSamples; ++i) {
		x[i] = std::sin(2.0F * 3.14159265F * InFreqHz * float(i) / float(InRate));
	}
	return x;
}

// Tone + deterministic broadband noise floor (LCG). The noise keeps every mel
// band's energy far above FP32 STFT error, so log-mel comparisons stay
// relative instead of amplifying near-silence noise.
std::vector<OaF32> MakeToneWithNoise(OaU32 InRate, OaF32 InFreqHz, OaU32 InSamples)
{
	auto x = MakeSine(InRate, InFreqHz, InSamples);
	OaU32 state = 0x12345678U;
	for (OaU32 i = 0; i < InSamples; ++i) {
		state = state * 1664525U + 1013904223U;
		const OaF32 r = (static_cast<OaF32>(state >> 8) / 8388608.0F) - 1.0F;  // [-1, 1)
		x[i] = 0.7F * x[i] + 0.05F * r;
	}
	return x;
}

OaAudio UploadMono(
	const std::vector<OaF32>& InX,
	OaU32 InSampleRate = 48'000U)
{
	auto m = OaFnMatrix::Empty(
		OaMatrixShape{1, static_cast<OaI64>(InX.size())}, OaScalarType::Float32);
	OaMemcpy(m.DataAs<OaF32>(), InX.data(), InX.size() * sizeof(OaF32));
	return OaAudio(OaStdMove(m), InSampleRate, OaChannelLayout::Mono);
}

// Float64 STFT magnitude reference: periodic Hann, naive DFT, optional
// librosa-style center padding. Layout [Frames][Bins].
void CpuStftMono(const std::vector<OaF32>& InX, OaU32 InFft, OaU32 InHop, bool InCenter,
                 std::vector<double>& OutMag, OaU32& OutFrames, OaU32& OutBins,
				 OaU8 InWindow = 0, OaU32 InWinSize = 0)
{
	if (InWinSize == 0) InWinSize = InFft;
	std::vector<double> s;
	if (InCenter) s.assign(InFft / 2, 0.0);
	s.insert(s.end(), InX.begin(), InX.end());
	if (InCenter) s.insert(s.end(), InFft / 2, 0.0);

	OutBins   = InFft / 2 + 1;
	OutFrames = s.size() >= InFft
		? 1 + static_cast<OaU32>((s.size() - InFft) / InHop) : 1;
	OutMag.assign(static_cast<size_t>(OutFrames) * OutBins, 0.0);

	const double pi = 3.14159265358979323846;
	for (OaU32 f = 0; f < OutFrames; ++f) {
		for (OaU32 k = 0; k < OutBins; ++k) {
			double re = 0.0, im = 0.0;
			for (OaU32 n = 0; n < InFft; ++n) {
				const size_t idx = static_cast<size_t>(f) * InHop + n;
				const double v  = idx < s.size() ? s[idx] : 0.0;
				double w = 0.0;
				if (n < InWinSize) {
					const double phase = 2.0 * pi * n / InWinSize;
					if (InWindow == 1) w = 0.54 - 0.46 * std::cos(phase);
					else if (InWindow == 2) w = 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
					else if (InWindow == 3) w = 1.0;
					else w = 0.5 * (1.0 - std::cos(phase));
				}
				const double a  = -2.0 * pi * double(k) * double(n) / double(InFft);
				re += v * w * std::cos(a);
				im += v * w * std::sin(a);
			}
			OutMag[static_cast<size_t>(f) * OutBins + k] = std::sqrt(re * re + im * im);
		}
	}
}

// CPU HTK mel filterbank [Mels][Bins] — mirrors the implementation so the
// GPU apply + layout are what's under test; filter construction sanity is
// covered by the peak-bin check in MelSinePeaksAtExpectedBin.
std::vector<double> CpuMelFilterbank(OaU32 InFft, OaU32 InRate, OaU32 InMels)
{
	const OaU32 bins = InFft / 2 + 1;
	auto hzToMel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
	auto melToHz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
	const double melMin = hzToMel(0.0), melMax = hzToMel(InRate * 0.5);

	std::vector<double> pts(InMels + 2);
	for (OaU32 i = 0; i < InMels + 2; ++i) {
		pts[i] = melToHz(melMin + double(i) * (melMax - melMin) / double(InMels + 1));
	}
	std::vector<double> fb(static_cast<size_t>(InMels) * bins, 0.0);
	const double binHz = double(InRate) / double(InFft);
	for (OaU32 m = 0; m < InMels; ++m) {
		for (OaU32 k = 0; k < bins; ++k) {
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
	const OaU32 rate = 22050, fft = 512, hop = 128, n = 2048;
	auto x = MakeSine(rate, 440.0F, n);
	auto buf = UploadMono(x, rate);

	OaStftConfig cfg{};
	cfg.FftSize = fft; cfg.HopSize = hop; cfg.WinSize = fft; cfg.Center = true;
	OaMatrix spec = OaFnAudio::Stft(buf, cfg);
	Sync();

	OaU32 frames = 0, bins = 0;
	std::vector<double> ref;
	CpuStftMono(x, fft, hop, true, ref, frames, bins);

	ASSERT_EQ(spec.GetShape().Rank, 3);
	ASSERT_EQ(spec.GetShape()[0], 1);
	ASSERT_EQ(spec.GetShape()[1], static_cast<OaI64>(frames));
	ASSERT_EQ(spec.GetShape()[2], static_cast<OaI64>(bins));

	for (OaU32 f = 0; f < frames; ++f) {
		for (OaU32 k = 0; k < bins; ++k) {
			const OaI64 idx = static_cast<OaI64>(f) * bins + k;
			const double cpu = ref[static_cast<size_t>(idx)];
			const double gpu = spec.At(idx);
			ASSERT_NEAR(gpu, cpu, 1e-3 * std::max(1.0, cpu))
				<< "frame " << f << " bin " << k;
		}
	}
}

TEST_VK(TestFnAudio, StftParsevalEnergy)
{
	const OaU32 rate = 22050, fft = 256, hop = 256, n = 1024;  // disjoint frames
	auto x = MakeSine(rate, 993.0F, n);   // deliberately off-bin frequency
	auto buf = UploadMono(x, rate);

	OaStftConfig cfg{};
	cfg.FftSize = fft; cfg.HopSize = hop; cfg.WinSize = fft; cfg.Center = false;
	OaMatrix spec = OaFnAudio::Stft(buf, cfg);
	Sync();

	const OaU32 frames = static_cast<OaU32>(spec.GetShape()[1]);
	const OaU32 bins   = static_cast<OaU32>(spec.GetShape()[2]);
	const double pi = 3.14159265358979323846;

	for (OaU32 f = 0; f < frames; ++f) {
		// Frequency-domain energy: Σ|X_k|² over the full N bins — interior
		// positive-frequency bins count twice (conjugate symmetry).
		double freqEnergy = 0.0;
		for (OaU32 k = 0; k < bins; ++k) {
			const double m = spec.At(static_cast<OaI64>(f) * bins + k);
			const double mult = (k == 0 || k == fft / 2) ? 1.0 : 2.0;
			freqEnergy += mult * m * m;
		}
		// Time-domain energy of the windowed frame, scaled by N (DFT Parseval).
		double timeEnergy = 0.0;
		for (OaU32 nn = 0; nn < fft; ++nn) {
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
	const OaU32 rate = 16000, fft = 256, hop = 64, win = 128, n = 91;
	auto x = MakeToneWithNoise(rate, 700.0F, n);
	auto buf = UploadMono(x, rate);

	for (OaU8 window : {OaU8(1), OaU8(2), OaU8(3)}) {
		OaStftConfig cfg{};
		cfg.FftSize = fft;
		cfg.HopSize = hop;
		cfg.WinSize = win;
		cfg.Window = window;
		cfg.Center = false;
		OaMatrix spec = OaFnAudio::Stft(buf, cfg);
		Sync();

		OaU32 frames = 0, bins = 0;
		std::vector<double> ref;
		CpuStftMono(x, fft, hop, false, ref, frames, bins, window, win);
		ASSERT_EQ(spec.GetShape()[1], 1);
		for (OaU32 k = 0; k < bins; ++k) {
			ASSERT_NEAR(spec.At(k), ref[k], 1e-3 * std::max(1.0, ref[k]))
				<< "window " << unsigned(window) << " bin " << k;
		}
	}
}

// ─── Mel / MFCC ───────────────────────────────────────────────────────────────

TEST_VK(TestFnAudio, MelSpectrogramMatchesCpuChain)
{
	const OaU32 rate = 16000, fft = 512, hop = 160, mels = 80, n = 4000;
	auto x = MakeToneWithNoise(rate, 440.0F, n);
	auto buf = UploadMono(x, rate);

	OaMelConfig cfg{};
	cfg.FftSize = fft; cfg.HopSize = hop; cfg.NumMels = mels; cfg.LogScale = true;
	OaMatrix mel = OaFnAudio::MelSpectrogram(buf, cfg);
	Sync();

	OaU32 frames = 0, bins = 0;
	std::vector<double> spec;
	CpuStftMono(x, fft, hop, true, spec, frames, bins);
	auto fb = CpuMelFilterbank(fft, rate, mels);

	// Whisper layout: [1, Mels, Frames]
	ASSERT_EQ(mel.GetShape().Rank, 3);
	ASSERT_EQ(mel.GetShape()[0], 1);
	ASSERT_EQ(mel.GetShape()[1], static_cast<OaI64>(mels));
	ASSERT_EQ(mel.GetShape()[2], static_cast<OaI64>(frames));

	for (OaU32 m = 0; m < mels; ++m) {
		for (OaU32 f = 0; f < frames; ++f) {
			double acc = 0.0;
			for (OaU32 k = 0; k < bins; ++k) {
				acc += spec[static_cast<size_t>(f) * bins + k]
				     * fb[static_cast<size_t>(m) * bins + k];
			}
			const double cpu = std::log(acc + 1e-9);
			const double gpu = mel.At(static_cast<OaI64>(m) * frames + f);
			ASSERT_NEAR(gpu, cpu, 1e-2 * std::max(1.0, std::abs(cpu)))
				<< "mel " << m << " frame " << f;
		}
	}
}

TEST_VK(TestFnAudio, MfccMatchesCpuChain)
{
	const OaU32 rate = 16000, fft = 512, hop = 160, mels = 40, coeffs = 13, n = 3200;
	auto x = MakeToneWithNoise(rate, 440.0F, n);
	auto buf = UploadMono(x, rate);

	OaMfccConfig cfg{};
	cfg.NumCoeffs = coeffs;
	cfg.Mel.FftSize = fft; cfg.Mel.HopSize = hop; cfg.Mel.NumMels = mels;
	OaMatrix mfcc = OaFnAudio::Mfcc(buf, cfg);
	Sync();

	OaU32 frames = 0, bins = 0;
	std::vector<double> spec;
	CpuStftMono(x, fft, hop, true, spec, frames, bins);
	auto fb = CpuMelFilterbank(fft, rate, mels);

	ASSERT_EQ(mfcc.GetShape().Rank, 3);
	ASSERT_EQ(mfcc.GetShape()[0], 1);
	ASSERT_EQ(mfcc.GetShape()[1], static_cast<OaI64>(coeffs));
	ASSERT_EQ(mfcc.GetShape()[2], static_cast<OaI64>(frames));

	const double pi = 3.14159265358979323846;
	std::vector<double> logMel(static_cast<size_t>(frames) * mels);
	for (OaU32 f = 0; f < frames; ++f) {
		for (OaU32 m = 0; m < mels; ++m) {
			double acc = 0.0;
			for (OaU32 k = 0; k < bins; ++k) {
				acc += spec[static_cast<size_t>(f) * bins + k]
				     * fb[static_cast<size_t>(m) * bins + k];
			}
			logMel[static_cast<size_t>(f) * mels + m] = std::log(acc + 1e-9);
		}
	}
	for (OaU32 k = 0; k < coeffs; ++k) {
		const double scale = k == 0 ? std::sqrt(1.0 / mels) : std::sqrt(2.0 / mels);
		for (OaU32 f = 0; f < frames; ++f) {
			double acc = 0.0;
			for (OaU32 m = 0; m < mels; ++m) {
				acc += logMel[static_cast<size_t>(f) * mels + m]
				     * std::cos(pi / mels * (m + 0.5) * k);
			}
			const double cpu = acc * scale;
			const double gpu = mfcc.At(static_cast<OaI64>(k) * frames + f);
			ASSERT_NEAR(gpu, cpu, 1e-2 * std::max(1.0, std::abs(cpu)))
				<< "coeff " << k << " frame " << f;
		}
	}
}

TEST_VK(TestFnAudio, MelNormalizationIsPerChannelZeroMeanUnitVariance)
{
	const OaU32 rate = 16000;
	auto x = MakeToneWithNoise(rate, 440.0F, 3200);
	auto buf = UploadMono(x, rate);
	OaMelConfig cfg{};
	cfg.FftSize = 256;
	cfg.HopSize = 80;
	cfg.NumMels = 32;
	cfg.Normalize = true;
	OaMatrix mel = OaFnAudio::MelSpectrogram(buf, cfg);
	Sync();

	double mean = 0.0;
	for (OaI64 i = 0; i < mel.NumElements(); ++i) mean += mel.At(i);
	mean /= static_cast<double>(mel.NumElements());
	double variance = 0.0;
	for (OaI64 i = 0; i < mel.NumElements(); ++i) {
		const double d = mel.At(i) - mean;
		variance += d * d;
	}
	variance /= static_cast<double>(mel.NumElements());
	EXPECT_NEAR(mean, 0.0, 1e-4);
	EXPECT_NEAR(variance, 1.0, 1e-3);
}

// ─── Signal ops ───────────────────────────────────────────────────────────────

TEST_VK(TestFnAudio, WaveformEnvelopePreservesMultichannelPeaks)
{
	const OaF32 samples[] = {
		-0.5F, 0.2F, -0.1F, 0.7F, -0.9F, 0.3F, 0.1F, 0.4F,
		 0.1F, 0.8F, -0.4F, 0.2F, -0.2F, 0.6F, -0.7F, 0.5F,
	};
	OaMatrix input = OaFnMatrix::Empty(
		OaMatrixShape{2, 8}, OaScalarType::Float32);
	OaMemcpy(input.DataAs<OaF32>(), samples, sizeof(samples));
	OaAudio audio(
		OaStdMove(input), 48'000U, OaChannelLayout::Stereo);
	OaMatrix envelope = OaFnAudio::WaveformEnvelope(audio, 4U);
	Sync();
	ASSERT_EQ(envelope.GetShape(), (OaMatrixShape{4, 2}));
	const OaF32 expected[] = {
		-0.5F, 0.8F,
		-0.4F, 0.7F,
		-0.9F, 0.6F,
		-0.7F, 0.5F,
	};
	for (OaI64 i = 0; i < 8; ++i) {
		EXPECT_FLOAT_EQ(envelope.At(i), expected[i]) << "envelope element " << i;
	}
}

TEST_VK(TestFnAudio, NormalizePeakHitsTarget)
{
	std::vector<OaF32> x(1024);
	for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25F * float(i) / float(x.size());
	auto buf = UploadMono(x);

	OaAudio normalized = OaFnAudio::Normalize(buf, -3.0F, 0);
	const OaMatrix& out = normalized.AsMatrix();
	Sync();
	EXPECT_EQ(normalized.SampleRate(), buf.SampleRate());
	EXPECT_EQ(normalized.Layout(), buf.Layout());
	OaF32 peak = 0.0F;
	for (OaI64 i = 0; i < 1024; ++i) peak = std::max(peak, std::abs(out.At(i)));
	EXPECT_NEAR(peak, std::pow(10.0F, -3.0F / 20.0F), 1e-4F);
}

TEST_VK(TestFnAudio, NormalizeRmsHitsTarget)
{
	auto x = MakeSine(48000, 440.0F, 4800);
	auto buf = UploadMono(x);

	OaAudio normalized = OaFnAudio::Normalize(buf, -20.0F, 1);
	const OaMatrix& out = normalized.AsMatrix();
	Sync();
	double sq = 0.0;
	for (OaI64 i = 0; i < 4800; ++i) { const double v = out.At(i); sq += v * v; }
	const double rms = std::sqrt(sq / 4800.0);
	EXPECT_NEAR(rms, std::pow(10.0, -20.0 / 20.0), 1e-4);
}

TEST_VK(TestFnAudio, NormalizeSilenceRemainsFiniteSilence)
{
	auto silence = OaFnMatrix::Zeros(OaMatrixShape{2, 128}, OaScalarType::Float32);
	OaAudio audio(
		OaStdMove(silence), 48'000U, OaChannelLayout::Stereo);
	OaAudio normalized = OaFnAudio::Normalize(audio, -3.0F, 0);
	const OaMatrix& out = normalized.AsMatrix();
	Sync();
	for (OaI64 i = 0; i < out.NumElements(); ++i) EXPECT_FLOAT_EQ(out.At(i), 0.0F);
}

TEST_VK(TestFnAudio, GainPlus6DbDoubles)
{
	auto x = MakeSine(48000, 440.0F, 512);
	auto buf = UploadMono(x);
	OaAudio gained = OaFnAudio::Gain(buf, 20.0F * std::log10(2.0F));
	const OaMatrix& out = gained.AsMatrix();
	Sync();
	for (OaI64 i = 0; i < 512; ++i) {
		ASSERT_NEAR(out.At(i), 2.0F * x[static_cast<size_t>(i)], 1e-5F) << "i=" << i;
	}
}

TEST_VK(TestFnAudio, ClipClampsRange)
{
	std::vector<OaF32> x = {-2.0F, -0.6F, -0.5F, 0.0F, 0.4F, 0.5F, 0.9F, 3.0F};
	auto buf = UploadMono(x);
	OaAudio clipped = OaFnAudio::Clip(buf, -0.5F, 0.5F);
	const OaMatrix& out = clipped.AsMatrix();
	Sync();
	const OaF32 expect[] = {-0.5F, -0.5F, -0.5F, 0.0F, 0.4F, 0.5F, 0.5F, 0.5F};
	for (OaI64 i = 0; i < 8; ++i) ASSERT_FLOAT_EQ(out.At(i), expect[i]) << "i=" << i;
}

TEST_VK(TestFnAudio, AmplitudeToDbKnownValues)
{
	std::vector<OaF32> x = {1.0F, 0.1F, -0.1F, 0.0F};
	auto buf = UploadMono(x);
	OaMatrix out = OaFnAudio::AmplitudeToDb(buf, -100.0F);
	Sync();
	EXPECT_NEAR(out.At(0),    0.0F, 1e-3F);
	EXPECT_NEAR(out.At(1),  -20.0F, 1e-3F);
	EXPECT_NEAR(out.At(2),  -20.0F, 1e-3F);
	EXPECT_NEAR(out.At(3), -100.0F, 1e-3F);   // silence hits the floor
}

TEST_VK(TestFnAudio, ToMonoAveragesChannels)
{
	const OaI64 n = 256;
	std::vector<OaF32> host(static_cast<size_t>(2 * n));
	for (OaI64 i = 0; i < n; ++i) { host[static_cast<size_t>(i)] = 1.0F; host[static_cast<size_t>(n + i)] = 0.0F; }
	auto buf = OaFnMatrix::Empty(OaMatrixShape{2, n}, OaScalarType::Float32);
	OaMemcpy(buf.DataAs<OaF32>(), host.data(), host.size() * sizeof(OaF32));

	OaAudio audio(
		OaStdMove(buf), 48'000U, OaChannelLayout::Stereo);
	OaAudio monoAudio = OaFnAudio::ToMono(audio);
	const OaMatrix& mono = monoAudio.AsMatrix();
	Sync();
	EXPECT_EQ(monoAudio.SampleRate(), 48'000U);
	EXPECT_EQ(monoAudio.Layout(), OaChannelLayout::Mono);
	ASSERT_EQ(mono.GetShape()[0], 1);
	ASSERT_EQ(mono.GetShape()[1], n);
	for (OaI64 i = 0; i < n; ++i) ASSERT_NEAR(mono.At(i), 0.5F, 1e-5F) << "i=" << i;
}

TEST_VK(TestFnAudio, ResampleMatchesCpuSinc)
{
	const OaU32 inRate = 48000;
	const OaU32 outRate = 16000;
	const OaU32 halfW = 32;
	const OaU32 n = 4800;
	auto x = MakeToneWithNoise(inRate, 440.0F, n);
	auto buf = UploadMono(x, inRate);

	OaAudio resampled = OaFnAudio::Resample(buf, outRate, halfW);
	const OaMatrix& out = resampled.AsMatrix();
	Sync();
	EXPECT_EQ(resampled.SampleRate(), outRate);
	EXPECT_EQ(resampled.Layout(), OaChannelLayout::Mono);

	const OaU32 inR = 3;    // 48000/16000 gcd-reduced
	const OaU32 outR = 1;
	const OaU32 outSamples = (n * outR) / inR;
	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], static_cast<OaI64>(outSamples));

	// CPU mirror of the shader math in double.
	const double pi = 3.14159265358979323846;
	const double ratio = double(outR) / double(inR);
	const double scale = std::min(1.0, ratio);
	for (OaU32 j = 0; j < outSamples; ++j) {
		const OaU32 q = j / outR;
		const OaU32 r = j % outR;
		const OaU32 num = r * inR;
		const OaU32 n0 = (q * inR) + (num / outR);
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
		ASSERT_NEAR(out.At(static_cast<OaI64>(j)), cpu, 1e-4)
			<< "out sample " << j;
	}
}

TEST_VK(TestFnAudio, ResampleIdentityPreservesSamples)
{
	auto x = MakeToneWithNoise(48000, 440.0F, 257);
	auto buf = UploadMono(x, 48'000U);
	OaAudio resampled = OaFnAudio::Resample(
		buf, OaResampleConfig{.OutRate = 48'000U, .FilterHalfWidth = 32});
	const OaMatrix& out = resampled.AsMatrix();
	Sync();
	ASSERT_EQ(out.GetShape(), buf.AsMatrix().GetShape());
	for (OaI64 i = 0; i < out.NumElements(); ++i) EXPECT_FLOAT_EQ(out.At(i), x[static_cast<size_t>(i)]);
}

TEST_VK(TestFnAudio, ResampleDownsampleSuppressesOutOfBandTone)
{
	const OaU32 inRate = 48000, outRate = 16000, n = 4800;
	auto x = MakeSine(inRate, 12000.0F, n); // above the 8 kHz output Nyquist
	auto buf = UploadMono(x, inRate);
	OaAudio resampled = OaFnAudio::Resample(buf, outRate, 64);
	const OaMatrix& out = resampled.AsMatrix();
	Sync();
	double squareSum = 0.0;
	const OaI64 margin = 64;
	for (OaI64 i = margin; i < out.GetShape()[1] - margin; ++i) {
		const double v = out.At(i);
		squareSum += v * v;
	}
	const double rms = std::sqrt(squareSum / double(out.GetShape()[1] - 2 * margin));
	EXPECT_LT(rms, 0.02);
}

TEST_VK(TestFnAudio, InvalidConfigurationsReturnEmpty)
{
	auto buf = UploadMono(MakeSine(16000, 440.0F, 512), 16'000U);
	OaStftConfig stft{};
	stft.FftSize = 300;
	EXPECT_TRUE(OaFnAudio::Stft(buf, stft).IsEmpty());

	OaMelConfig mel{};
	mel.FMin = 9000.0F;
	EXPECT_TRUE(OaFnAudio::MelSpectrogram(buf, mel).IsEmpty());

	OaMfccConfig mfcc{};
	mfcc.NumCoeffs = mfcc.Mel.NumMels + 1;
	EXPECT_TRUE(OaFnAudio::Mfcc(buf, mfcc).IsEmpty());

	EXPECT_TRUE(OaFnAudio::Normalize(buf, -3.0F, 2).IsEmpty());
	EXPECT_TRUE(OaFnAudio::Resample(buf, 0, 32).IsEmpty());
	EXPECT_TRUE(OaFnAudio::Resample(buf, 16000, 2048).IsEmpty());
	EXPECT_TRUE(OaFnAudio::Clip(buf, 1.0F, -1.0F).IsEmpty());
}

TEST_VK(TestFnAudio, PreEmphasisMatchesClosedForm)
{
	const OaU32 n = 512;
	const OaF32 alpha = 0.97F;
	auto x = MakeToneWithNoise(22050, 440.0F, n);
	auto buf = UploadMono(x, 22'050U);

	OaAudio emphasized = OaFnAudio::PreEmphasis(buf, alpha);
	const OaMatrix& out = emphasized.AsMatrix();
	Sync();

	ASSERT_EQ(out.GetShape()[0], 1);
	ASSERT_EQ(out.GetShape()[1], static_cast<OaI64>(n));
	ASSERT_NEAR(out.At(0), x[0], 1e-6F);   // y[0] = x[0]
	for (OaU32 i = 1; i < n; ++i) {
		const OaF32 expect = x[i] - (alpha * x[i - 1]);
		ASSERT_NEAR(out.At(static_cast<OaI64>(i)), expect, 1e-5F) << "i=" << i;
	}
}

TEST_VK(TestFnAudio, FadeAppliesLinearEnvelope)
{
	const OaI64 n = 100;
	std::vector<OaF32> x(static_cast<size_t>(n), 1.0F);
	auto buf = UploadMono(x);
	OaAudio faded = OaFnAudio::Fade(buf, 10, 10);
	const OaMatrix& out = faded.AsMatrix();
	Sync();
	EXPECT_FLOAT_EQ(out.At(0), 0.0F);
	EXPECT_NEAR(out.At(5),  0.5F, 1e-5F);
	EXPECT_FLOAT_EQ(out.At(50), 1.0F);
	EXPECT_NEAR(out.At(n - 6), 0.5F, 1e-5F);
	EXPECT_FLOAT_EQ(out.At(n - 1), 0.0F);
}
