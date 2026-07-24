// TestAudioDecoder — OaAudioDecoder / OaAudioEncoder / OaFnAudio tests.
// Synthetic WAV fixtures plus one attributed real speech asset.
// Validation gates: WAV F32 round-trip is
// BIT-EXACT; deinterleave is sample-exact per channel; GPU ops assert
// against in-test CPU oracles.

#include "../../OaTest.h"

#include <Oa/Audio.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cmath>
#include <chrono>
#include <limits>
#include <thread>

namespace {

OaEngine* GRt = nullptr;

class TestAudioDecoder : public ::testing::Test {
protected:
	static void SetUpTestSuite() {
		GRt = OaEngine::GetGlobal();
		ASSERT_NE(GRt, nullptr)
			<< "OaVkTestEnvironment did not create the suite engine";
	}

	// Flush + sync the default context so .At() reads committed values.
	static void Sync() {
		auto& ctx = OaContext::GetDefault();
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
	}
};

OaVec<OaF32> MakeSineWave(OaU32 InSampleRate, OaF32 InFreqHz, OaF32 InDurationS)
{
	const OaU32 n = static_cast<OaU32>(static_cast<OaF32>(InSampleRate) * InDurationS);
	OaVec<OaF32> samples;
	samples.Resize(n);
	for (OaU32 i = 0; i < n; ++i) {
		samples[i] = std::sin(2.0F * 3.14159265F * InFreqHz * float(i) / float(InSampleRate));
	}
	return samples;
}

} // namespace

// ─── WAV encode → decode round trip ───────────────────────────────────────────

TEST_VK(TestAudioDecoder, WavRoundTripMonoBitExact)
{
	const OaU32 sampleRate = 44100;
	auto sine = MakeSineWave(sampleRate, 440.0F, 1.0F);

	auto wavResult = OaAudioEncoder::EncodeWavF32(
		OaSpan<const OaF32>(sine.Data(), sine.Size()), sampleRate, 1);
	ASSERT_TRUE(wavResult.IsOk()) << wavResult.GetStatus().GetMessage();
	const auto& wavBytes = wavResult.GetValue();
	ASSERT_GT(wavBytes.Size(), 46u);

	auto res = OaAudioDecoder::LoadMemory(OaSpan<const OaU8>(wavBytes.Data(), wavBytes.Size()));
	ASSERT_TRUE(res.IsOk()) << res.GetStatus().GetMessage();

	auto& dec = res.GetValue();
	EXPECT_EQ(dec.SampleRate(), sampleRate);
	EXPECT_EQ(dec.Channels(), 1);
	EXPECT_EQ(dec.Samples(), static_cast<OaI64>(sine.Size()));

	ASSERT_FALSE(dec.IsEmpty());
	ASSERT_EQ(dec.AsMatrix().GetShape()[0], 1);
	ASSERT_EQ(dec.AsMatrix().GetShape()[1], static_cast<OaI64>(sine.Size()));

	// F32 WAV (format 3) → miniaudio f32 output is a pass-through: bit-exact.
	for (OaUsize i = 0; i < sine.Size(); ++i) {
		ASSERT_EQ(dec.AsMatrix().At(static_cast<OaI64>(i)), sine[i]) << "sample " << i;
	}
}

TEST_VK(TestAudioDecoder, WavRoundTripStereoDeinterleaves)
{
	const OaU32 sampleRate = 22050;
	auto left  = MakeSineWave(sampleRate, 440.0F, 0.5F);
	auto right = MakeSineWave(sampleRate, 880.0F, 0.5F);

	OaVec<OaF32> interleaved;
	interleaved.Resize(left.Size() * 2);
	for (OaUsize i = 0; i < left.Size(); ++i) {
		interleaved[i * 2]     = left[i];
		interleaved[i * 2 + 1] = right[i];
	}

	auto wavResult = OaAudioEncoder::EncodeWavF32(
		OaSpan<const OaF32>(interleaved.Data(), interleaved.Size()), sampleRate, 2);
	ASSERT_TRUE(wavResult.IsOk()) << wavResult.GetStatus().GetMessage();
	const auto& wavBytes = wavResult.GetValue();

	auto res = OaAudioDecoder::LoadMemory(OaSpan<const OaU8>(wavBytes.Data(), wavBytes.Size()));
	ASSERT_TRUE(res.IsOk()) << res.GetStatus().GetMessage();

	auto& dec = res.GetValue();
	EXPECT_EQ(dec.Channels(), 2);
	ASSERT_EQ(dec.AsMatrix().GetShape()[0], 2);
	ASSERT_EQ(dec.AsMatrix().GetShape()[1], static_cast<OaI64>(left.Size()));

	// Planar [C, S]: row 0 = left, row 1 = right, both bit-exact.
	const OaI64 n = static_cast<OaI64>(left.Size());
	for (OaI64 i = 0; i < n; ++i) {
		ASSERT_EQ(dec.AsMatrix().At(i), left[static_cast<OaUsize>(i)]) << "L sample " << i;
		ASSERT_EQ(dec.AsMatrix().At(n + i), right[static_cast<OaUsize>(i)]) << "R sample " << i;
	}
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

TEST_VK(TestAudioDecoder, DecodeResultMeta)
{
	const OaU32 sampleRate = 44100;
	auto sine = MakeSineWave(sampleRate, 440.0F, 1.0F);

	OaVec<OaF32> interleaved;
	interleaved.Resize(sine.Size() * 2);
	for (OaUsize i = 0; i < sine.Size(); ++i) {
		interleaved[i * 2] = interleaved[i * 2 + 1] = sine[i];
	}
	auto wavResult = OaAudioEncoder::EncodeWavF32(
		OaSpan<const OaF32>(interleaved.Data(), interleaved.Size()), sampleRate, 2);
	ASSERT_TRUE(wavResult.IsOk()) << wavResult.GetStatus().GetMessage();
	const auto& wavBytes = wavResult.GetValue();
	auto res = OaAudioDecoder::LoadMemory(OaSpan<const OaU8>(wavBytes.Data(), wavBytes.Size()));
	ASSERT_TRUE(res.IsOk()) << res.GetStatus().GetMessage();

	const OaAudio& audio = res.GetValue();
	EXPECT_EQ(audio.SampleRate(), sampleRate);
	EXPECT_EQ(audio.Channels(), 2);
	EXPECT_EQ(audio.Samples(), static_cast<OaI64>(sine.Size()));
	EXPECT_EQ(audio.Layout(), OaChannelLayout::Stereo);
	EXPECT_NEAR(audio.DurationSeconds(), 1.0, 1e-6);
}

TEST_VK(TestAudioDecoder, RejectsMalformedInputs)
{
	OaAudioCaptureConfig invalidCapture;
	invalidCapture.SampleRate = 0U;
	EXPECT_FALSE(OaAudioCapture::Open(*GRt, invalidCapture).IsOk());
	OaAudioStreamConfig invalidStream;
	EXPECT_FALSE(OaAudioStream::Open(*GRt, invalidStream).IsOk());
	invalidCapture.SampleRate = 48'000U;
	invalidCapture.RingMilliseconds = 1U;
	EXPECT_FALSE(OaAudioCapture::Open(*GRt, invalidCapture).IsOk());

	EXPECT_FALSE(OaAudioDecoder::LoadFile(nullptr).IsOk());
	EXPECT_FALSE(OaAudioDecoder::LoadFile("").IsOk());
	EXPECT_FALSE(OaAudioDecoder::LoadMemory({}).IsOk());

	const OaU8 garbage[] = {0x00, 0x11, 0x22, 0x33};
	EXPECT_FALSE(OaAudioDecoder::LoadMemory(OaSpan<const OaU8>(garbage, 4)).IsOk());

	EXPECT_FALSE(OaAudioEncoder::EncodeWavF32({}, 48000, 1).IsOk());
	const OaF32 incomplete[] = {0.0F, 0.1F, 0.2F};
	EXPECT_FALSE(OaAudioEncoder::EncodeWavF32(
		OaSpan<const OaF32>(incomplete, 3), 48000, 2).IsOk());
	EXPECT_FALSE(OaAudioEncoder::EncodeWavF32(
		OaSpan<const OaF32>(incomplete, 3), 0, 1).IsOk());
}

TEST_VK(TestAudioDecoder, AudioSessionCloseIsIdempotent)
{
	OaAudioCapture capture;
	OaAudioStream stream;
	EXPECT_TRUE(capture.Close().IsOk());
	EXPECT_TRUE(capture.Close().IsOk());
	EXPECT_TRUE(stream.Close().IsOk());
	EXPECT_TRUE(stream.Close().IsOk());
}

TEST_VK(TestAudioDecoder, StreamingPcmIsDeterministicForNonFiniteInput)
{
	OaAudioEncodeProfile profile;
	profile.SampleRate = 48'000U;
	profile.ChannelCount = 1U;
	profile.FramesPerPacket = 4U;
	auto created = OaAudioStreamEncoder::Create(profile);
	ASSERT_TRUE(created.IsOk());
	OaAudioStreamEncoder encoder = OaStdMove(*created);
	const OaF32 samples[] = {
		-std::numeric_limits<OaF32>::infinity(),
		std::numeric_limits<OaF32>::quiet_NaN(),
		std::numeric_limits<OaF32>::infinity(),
		0.5F,
	};
	OaVec<OaEncodedAudioPacket> packets;
	ASSERT_TRUE(encoder.Encode(OaSpan<const OaF32>(samples, 4U), packets).IsOk());
	ASSERT_EQ(packets.Size(), 1U);
	ASSERT_EQ(packets[0].Bitstream.Size(), 8U);
	const OaU8 expected[] = {0x00U, 0x80U, 0x00U, 0x00U,
		0xFFU, 0x7FU, 0x00U, 0x40U};
	for (OaUsize i = 0U; i < sizeof(expected); ++i) {
		EXPECT_EQ(packets[0].Bitstream[i], expected[i]) << i;
	}
}

TEST_VK(TestAudioDecoder, AudioStreamPlayPauseSeek)
{
	OaAudioStreamConfig config;
	config.Uri = OaTestAssetPath("Audio/0_jackson_0.wav").String();
	config.RingMilliseconds = 100U;
	auto opened = OaAudioStream::Open(*GRt, config);
	if (not opened.IsOk()) {
		GTEST_SKIP() << "No playback device/media backend: "
			<< opened.GetStatus().GetMessage();
	}
	OaAudioStream stream = OaStdMove(*opened);
	EXPECT_EQ(stream.SampleRate(), 8000U);
	EXPECT_EQ(stream.ChannelCount(), 1U);
	EXPECT_GT(stream.DurationUs(), 0U);
	ASSERT_TRUE(stream.Play().IsOk());
	// A low-rate source can inherit a 1024-frame device quantum (128 ms at
	// 8 kHz), and backend startup may miss the first callback deadline. Give
	// the real device two complete quanta instead of baking workstation timing
	// into the playback contract.
	for (OaI32 retry = 0; retry < 200 and stream.PositionUs() == 0U; ++retry) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	stream.Pause();
	EXPECT_GT(stream.PositionUs(), 0U);
	ASSERT_TRUE(stream.Seek(10'000U).IsOk());
	EXPECT_GE(stream.PositionUs(), 10'000U);
	ASSERT_TRUE(stream.Close().IsOk());
	EXPECT_FALSE(stream.IsOpen());
}

TEST_VK(TestAudioDecoder, RealSpeechDecodeProcessSaveReload)
{
	const OaPath wavPath = OaTestAssetPath("Audio/0_jackson_0.wav");
	const OaPath inputPath = OaTestAssetPath("Audio/0_jackson_0.flac");
	const OaPath mp3Path = OaTestAssetPath("Audio/0_jackson_0.mp3");
	auto decoded = OaAudioDecoder::LoadFile(inputPath);
	ASSERT_TRUE(decoded.IsOk()) << decoded.GetStatus().GetMessage();
	auto wavDecoded = OaAudioDecoder::LoadFile(wavPath);
	ASSERT_TRUE(wavDecoded.IsOk()) << wavDecoded.GetStatus().GetMessage();
	ASSERT_EQ(decoded->SampleRate(), 8000U);
	ASSERT_EQ(decoded->Channels(), 1);
	ASSERT_GT(decoded->Samples(), 4000);
	ASSERT_EQ(decoded->Samples(), wavDecoded->Samples());
	for (OaI64 i = 0; i < decoded->Samples(); i += 31) {
		ASSERT_FLOAT_EQ(decoded->AsMatrix().At(i),
			wavDecoded->AsMatrix().At(i)) << "lossless sample " << i;
	}
	auto mp3Decoded = OaAudioDecoder::LoadFile(mp3Path);
	ASSERT_TRUE(mp3Decoded.IsOk()) << mp3Decoded.GetStatus().GetMessage();
	EXPECT_EQ(mp3Decoded->SampleRate(), 8000U);
	EXPECT_EQ(mp3Decoded->Channels(), 1);
	EXPECT_GT(mp3Decoded->Samples(), 4000);
	double mp3SquareSum = 0.0;
	for (OaI64 i = 0; i < mp3Decoded->Samples(); ++i) {
		const double sample = mp3Decoded->AsMatrix().At(i);
		EXPECT_TRUE(std::isfinite(sample));
		mp3SquareSum += sample * sample;
	}
	EXPECT_GT(std::sqrt(mp3SquareSum / static_cast<double>(mp3Decoded->Samples())), 0.01);

	OaResampleConfig resampleCfg{};
	resampleCfg.OutRate = 16000;
	resampleCfg.FilterHalfWidth = 32;
	OaAudio processed = OaFnAudio::Normalize(
		OaFnAudio::Resample(*decoded, resampleCfg),
		OaNormalizeAudioConfig{.Mode = 0, .TargetDb = -6.0F});

	OaMelConfig melCfg{};
	melCfg.FftSize = 256;
	melCfg.HopSize = 80;
	melCfg.NumMels = 40;
	melCfg.Normalize = true;
	OaMatrix mel = OaFnAudio::MelSpectrogram(processed, melCfg);
	ASSERT_EQ(mel.GetShape().Rank, 3);
	ASSERT_EQ(mel.GetShape()[0], 1);
	ASSERT_EQ(mel.GetShape()[1], 40);

	const OaPath outputPath = OaPaths::Temp() / "oa_audio_real_e2e.wav";
	ASSERT_TRUE(OaAudioEncoder::SaveWavF32(outputPath, processed).IsOk());
	auto reloaded = OaAudioDecoder::LoadFile(outputPath);
	ASSERT_TRUE(reloaded.IsOk()) << reloaded.GetStatus().GetMessage();
	EXPECT_EQ(reloaded->SampleRate(), 16000U);
	EXPECT_EQ(reloaded->Channels(), 1);
	EXPECT_EQ(reloaded->Samples(), processed.Samples());
	EXPECT_TRUE(OaFilesystem::RemoveFile(outputPath).IsOk());

	for (OaI64 i = 0; i < mel.NumElements(); i += 97) {
		EXPECT_TRUE(std::isfinite(mel.At(i))) << "mel element " << i;
	}
}

// ─── OaFnAudio ────────────────────────────────────────────────────────────────

TEST_VK(TestAudioDecoder, AudioComposesMatrixWithoutAlias)
{
	OaAudio audio(
		OaFnMatrix::Zeros(OaMatrixShape{2, 512}),
		48'000U,
		OaChannelLayout::Stereo);
	ASSERT_TRUE(audio.Validate());
	const OaMatrix& m = audio.AsMatrix();
	EXPECT_EQ(m.GetShape()[0], 2);
	EXPECT_EQ(m.GetShape()[1], 512);
	EXPECT_EQ(audio.SampleRate(), 48'000U);
	EXPECT_EQ(audio.Layout(), OaChannelLayout::Stereo);
}

TEST_VK(TestAudioDecoder, MixMatchesCpuOracle)
{
	const OaI64 n = 1024;
	auto a = MakeSineWave(48000, 440.0F, 1024.0F / 48000.0F);
	auto b = MakeSineWave(48000, 880.0F, 1024.0F / 48000.0F);
	ASSERT_GE(static_cast<OaI64>(a.Size()), n);

	auto ma = OaFnMatrix::Empty(OaMatrixShape{1, n}, OaScalarType::Float32);
	auto mb = OaFnMatrix::Empty(OaMatrixShape{1, n}, OaScalarType::Float32);
	OaMemcpy(ma.DataAs<OaF32>(), a.Data(), static_cast<OaUsize>(n) * sizeof(OaF32));
	OaMemcpy(mb.DataAs<OaF32>(), b.Data(), static_cast<OaUsize>(n) * sizeof(OaF32));

	const OaF32 gainA = 0.75F, gainB = 0.25F;
	OaAudio audioA(OaStdMove(ma), 48'000U, OaChannelLayout::Mono);
	OaAudio audioB(OaStdMove(mb), 48'000U, OaChannelLayout::Mono);
	OaAudio mixedAudio = OaFnAudio::Mix(audioA, audioB, gainA, gainB);
	const OaMatrix& mixed = mixedAudio.AsMatrix();
	Sync();

	for (OaI64 i = 0; i < n; ++i) {
		const OaF32 expect = gainA * a[static_cast<OaUsize>(i)] + gainB * b[static_cast<OaUsize>(i)];
		ASSERT_NEAR(mixed.At(i), expect, 1e-6F) << "sample " << i;
	}
}
