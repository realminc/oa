// TestAudioDecoder — oa::FnAudio codec/DSP and audio session tests.
// Synthetic WAV fixtures plus one attributed real speech asset.
// Validation gates: WAV F32 round-trip is
// BIT-EXACT; deinterleave is sample-exact per channel; GPU ops assert
// against in-test CPU oracles.

#include "../../oaTest.h"

#include <oa/audio.h>
#include <oa/core/filesystem.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <cmath>
#include <chrono>
#include <limits>
#include <thread>

namespace {

oa::Engine* gRt = nullptr;

class TestAudioDecoder : public ::testing::Test {
protected:
	static void setUpTestSuite() {
		gRt = testEnginePtr();
		ASSERT_NE(gRt, nullptr)
			<< "VkTestEnvironment did not create the suite engine";
	}

	// flush + sync the default context so .at() reads committed values.
	static void sync() {
		auto& ctx = oa::ExecutionSession::getActive();
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	}
};

oa::Vec<oa::F32> makeSineWave(oa::U32 inSampleRate, oa::F32 inFreqHz, oa::F32 inDurationS)
{
	const oa::U32 n = static_cast<oa::U32>(static_cast<oa::F32>(inSampleRate) * inDurationS);
	oa::Vec<oa::F32> samples;
	samples.resize(n);
	for (oa::U32 i = 0; i < n; ++i) {
		samples[i] = std::sin(2.0F * 3.14159265F * inFreqHz * float(i) / float(inSampleRate));
	}
	return samples;
}

} // namespace

// ─── WAV encode → decode round trip ───────────────────────────────────────────

TEST_VK(TestAudioDecoder, WavRoundTripMonoBitExact)
{
	const oa::U32 sampleRate = 44100;
	auto sine = makeSineWave(sampleRate, 440.0F, 1.0F);

	auto wavResult = oa::FnAudio::encodeInterleavedWavF32(
		oa::Span<const oa::F32>(sine.data(), sine.size()), sampleRate, 1);
	ASSERT_TRUE(wavResult.isOk()) << wavResult.getStatus().getMessage();
	const auto& wavBytes = wavResult.getValue();
	ASSERT_GT(wavBytes.size(), 46u);

	auto res = oa::FnAudio::decodeMemory(oa::Span<const oa::U8>(wavBytes.data(), wavBytes.size()));
	ASSERT_TRUE(res.isOk()) << res.getStatus().getMessage();

	auto& dec = res.getValue();
	EXPECT_EQ(dec.sampleRate(), sampleRate);
	EXPECT_EQ(dec.channels(), 1);
	EXPECT_EQ(dec.samples(), static_cast<oa::I64>(sine.size()));

	ASSERT_FALSE(dec.isEmpty());
	ASSERT_EQ(dec.asMatrix().getShape()[0], 1);
	ASSERT_EQ(dec.asMatrix().getShape()[1], static_cast<oa::I64>(sine.size()));

	// F32 WAV (format 3) → miniaudio f32 output is a pass-through: bit-exact.
	for (oa::Usize i = 0; i < sine.size(); ++i) {
		ASSERT_EQ(dec.asMatrix().at(static_cast<oa::I64>(i)), sine[i]) << "sample " << i;
	}
}

TEST_VK(TestAudioDecoder, WavRoundTripStereoDeinterleaves)
{
	const oa::U32 sampleRate = 22050;
	auto left  = makeSineWave(sampleRate, 440.0F, 0.5F);
	auto right = makeSineWave(sampleRate, 880.0F, 0.5F);

	oa::Vec<oa::F32> interleaved;
	interleaved.resize(left.size() * 2);
	for (oa::Usize i = 0; i < left.size(); ++i) {
		interleaved[i * 2]     = left[i];
		interleaved[i * 2 + 1] = right[i];
	}

	auto wavResult = oa::FnAudio::encodeInterleavedWavF32(
		oa::Span<const oa::F32>(interleaved.data(), interleaved.size()), sampleRate, 2);
	ASSERT_TRUE(wavResult.isOk()) << wavResult.getStatus().getMessage();
	const auto& wavBytes = wavResult.getValue();

	auto res = oa::FnAudio::decodeMemory(oa::Span<const oa::U8>(wavBytes.data(), wavBytes.size()));
	ASSERT_TRUE(res.isOk()) << res.getStatus().getMessage();

	auto& dec = res.getValue();
	EXPECT_EQ(dec.channels(), 2);
	ASSERT_EQ(dec.asMatrix().getShape()[0], 2);
	ASSERT_EQ(dec.asMatrix().getShape()[1], static_cast<oa::I64>(left.size()));

	// Planar [C, S]: row 0 = left, row 1 = right, both bit-exact.
	const oa::I64 n = static_cast<oa::I64>(left.size());
	for (oa::I64 i = 0; i < n; ++i) {
		ASSERT_EQ(dec.asMatrix().at(i), left[static_cast<oa::Usize>(i)]) << "L sample " << i;
		ASSERT_EQ(dec.asMatrix().at(n + i), right[static_cast<oa::Usize>(i)]) << "R sample " << i;
	}
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

TEST_VK(TestAudioDecoder, DecodeResultMeta)
{
	const oa::U32 sampleRate = 44100;
	auto sine = makeSineWave(sampleRate, 440.0F, 1.0F);

	oa::Vec<oa::F32> interleaved;
	interleaved.resize(sine.size() * 2);
	for (oa::Usize i = 0; i < sine.size(); ++i) {
		interleaved[i * 2] = interleaved[i * 2 + 1] = sine[i];
	}
	auto wavResult = oa::FnAudio::encodeInterleavedWavF32(
		oa::Span<const oa::F32>(interleaved.data(), interleaved.size()), sampleRate, 2);
	ASSERT_TRUE(wavResult.isOk()) << wavResult.getStatus().getMessage();
	const auto& wavBytes = wavResult.getValue();
	auto res = oa::FnAudio::decodeMemory(oa::Span<const oa::U8>(wavBytes.data(), wavBytes.size()));
	ASSERT_TRUE(res.isOk()) << res.getStatus().getMessage();

	const oa::Audio& audio = res.getValue();
	EXPECT_EQ(audio.sampleRate(), sampleRate);
	EXPECT_EQ(audio.channels(), 2);
	EXPECT_EQ(audio.samples(), static_cast<oa::I64>(sine.size()));
	EXPECT_EQ(audio.layout(), oa::AudioChannelLayout::Stereo);
	EXPECT_NEAR(audio.durationSeconds(), 1.0, 1e-6);
}

TEST_VK(TestAudioDecoder, RejectsMalformedInputs)
{
	oa::AudioCaptureConfig invalidCapture;
	invalidCapture.sampleRate = 0U;
	EXPECT_FALSE(oa::AudioCapture::open(*gRt, invalidCapture).isOk());
	oa::AudioPlayerConfig invalidStream;
	EXPECT_FALSE(oa::AudioPlayer::open(*gRt, invalidStream).isOk());
	invalidCapture.sampleRate = 48'000U;
	invalidCapture.ringMilliseconds = 1U;
	EXPECT_FALSE(oa::AudioCapture::open(*gRt, invalidCapture).isOk());

	EXPECT_FALSE(oa::FnAudio::decodeFile(nullptr).isOk());
	EXPECT_FALSE(oa::FnAudio::decodeFile("").isOk());
	EXPECT_FALSE(oa::FnAudio::decodeMemory({}).isOk());

	const oa::U8 garbage[] = {0x00, 0x11, 0x22, 0x33};
	EXPECT_FALSE(oa::FnAudio::decodeMemory(oa::Span<const oa::U8>(garbage, 4)).isOk());

	EXPECT_FALSE(oa::FnAudio::encodeInterleavedWavF32({}, 48000, 1).isOk());
	const oa::F32 incomplete[] = {0.0F, 0.1F, 0.2F};
	EXPECT_FALSE(oa::FnAudio::encodeInterleavedWavF32(
		oa::Span<const oa::F32>(incomplete, 3), 48000, 2).isOk());
	EXPECT_FALSE(oa::FnAudio::encodeInterleavedWavF32(
		oa::Span<const oa::F32>(incomplete, 3), 0, 1).isOk());
}

TEST_VK(TestAudioDecoder, AudioSessionCloseIsIdempotent)
{
	oa::AudioCapture capture;
	oa::AudioPlayer stream;
	EXPECT_TRUE(capture.close().isOk());
	EXPECT_TRUE(capture.close().isOk());
	EXPECT_TRUE(stream.close().isOk());
	EXPECT_TRUE(stream.close().isOk());
}

TEST_VK(TestAudioDecoder, StreamingPcmIsDeterministicForNonFiniteInput)
{
	oa::AudioEncodeProfile profile;
	profile.sampleRate = 48'000U;
	profile.channelCount = 1U;
	profile.framesPerPacket = 4U;
	auto created = oa::AudioEncoder::create(profile);
	ASSERT_TRUE(created.isOk());
	oa::AudioEncoder encoder = oa::move(*created);
	const oa::F32 samples[] = {
		-std::numeric_limits<oa::F32>::infinity(),
		std::numeric_limits<oa::F32>::quiet_NaN(),
		std::numeric_limits<oa::F32>::infinity(),
		0.5F,
	};
	oa::Vec<oa::EncodedAudioPacket> packets;
	ASSERT_TRUE(encoder.encode(oa::Span<const oa::F32>(samples, 4U), packets).isOk());
	ASSERT_EQ(packets.size(), 1U);
	ASSERT_EQ(packets[0].bitstream.size(), 8U);
	const oa::U8 expected[] = {0x00U, 0x80U, 0x00U, 0x00U,
		0xFFU, 0x7FU, 0x00U, 0x40U};
	for (oa::Usize i = 0U; i < sizeof(expected); ++i) {
		EXPECT_EQ(packets[0].bitstream[i], expected[i]) << i;
	}
	EXPECT_TRUE(encoder.close().isOk());
	EXPECT_TRUE(encoder.close().isOk());
	EXPECT_FALSE(encoder.isOpen());
}

TEST_VK(TestAudioDecoder, AudioPlayerPlayPauseSeek)
{
	oa::AudioPlayerConfig config;
	config.uri = testAssetPath("audio/oaNarration.wav").string();
	config.ringMilliseconds = 100U;
	auto opened = oa::AudioPlayer::open(*gRt, config);
	if (not opened.isOk()) {
		GTEST_SKIP() << "No playback device/media backend: "
			<< opened.getStatus().getMessage();
	}
	oa::AudioPlayer stream = oa::move(*opened);
	EXPECT_EQ(stream.sampleRate(), 24000U);
	EXPECT_EQ(stream.channelCount(), 1U);
	EXPECT_GT(stream.durationUs(), 0U);
	ASSERT_TRUE(stream.play().isOk());
	// Backend startup may miss the first callback deadline. Give the real device
	// enough time for several complete quanta instead of baking workstation
	// timing into the playback contract.
	for (oa::I32 retry = 0; retry < 200 and stream.positionUs() == 0U; ++retry) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	stream.pause();
	EXPECT_GT(stream.positionUs(), 0U);
	ASSERT_TRUE(stream.seek(10'000U).isOk());
	EXPECT_GE(stream.positionUs(), 10'000U);
	ASSERT_TRUE(stream.close().isOk());
	EXPECT_FALSE(stream.isOpen());
}

TEST_VK(TestAudioDecoder, RealNarrationDecodeProcessSaveReload)
{
	const oa::Path wavPath = testAssetPath("audio/oaNarration.wav");
	const oa::Path inputPath = testAssetPath("audio/oaNarration.flac");
	const oa::Path mp3Path = testAssetPath("audio/oaNarration.mp3");
	auto decoded = oa::FnAudio::decodeFile(inputPath);
	ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().getMessage();
	auto wavDecoded = oa::FnAudio::decodeFile(wavPath);
	ASSERT_TRUE(wavDecoded.isOk()) << wavDecoded.getStatus().getMessage();
	ASSERT_EQ(decoded->sampleRate(), 24000U);
	ASSERT_EQ(decoded->channels(), 1);
	ASSERT_GT(decoded->samples(), 240000);
	ASSERT_EQ(decoded->samples(), wavDecoded->samples());
	for (oa::I64 i = 0; i < decoded->samples(); i += 31) {
		ASSERT_FLOAT_EQ(decoded->asMatrix().at(i),
			wavDecoded->asMatrix().at(i)) << "lossless sample " << i;
	}
	auto mp3Decoded = oa::FnAudio::decodeFile(mp3Path);
	ASSERT_TRUE(mp3Decoded.isOk()) << mp3Decoded.getStatus().getMessage();
	EXPECT_EQ(mp3Decoded->sampleRate(), 24000U);
	EXPECT_EQ(mp3Decoded->channels(), 1);
	EXPECT_GT(mp3Decoded->samples(), 240000);
	double mp3SquareSum = 0.0;
	for (oa::I64 i = 0; i < mp3Decoded->samples(); ++i) {
		const double sample = mp3Decoded->asMatrix().at(i);
		EXPECT_TRUE(std::isfinite(sample));
		mp3SquareSum += sample * sample;
	}
	EXPECT_GT(std::sqrt(mp3SquareSum / static_cast<double>(mp3Decoded->samples())), 0.01);

	oa::ResampleConfig resampleCfg{};
	resampleCfg.outRate = 16000;
	resampleCfg.filterHalfWidth = 32;
	oa::Audio processed = oa::FnAudio::normalize(
		oa::FnAudio::resample(*decoded, resampleCfg),
		oa::NormalizeAudioConfig{.mode = 0, .targetDb = -6.0F});

	oa::MelConfig melCfg{};
	melCfg.fftSize = 256;
	melCfg.hopSize = 80;
	melCfg.numMels = 40;
	melCfg.normalize = true;
	oa::Matrix mel = oa::FnAudio::melSpectrogram(processed, melCfg);
	ASSERT_EQ(mel.getShape().rank, 3);
	ASSERT_EQ(mel.getShape()[0], 1);
	ASSERT_EQ(mel.getShape()[1], 40);

	const oa::Path outputPath = oa::Paths::temp() / "oa_audio_real_e2e.wav";
	ASSERT_TRUE(oa::FnAudio::saveWavF32(outputPath, processed).isOk());
	auto reloaded = oa::FnAudio::decodeFile(outputPath);
	ASSERT_TRUE(reloaded.isOk()) << reloaded.getStatus().getMessage();
	EXPECT_EQ(reloaded->sampleRate(), 16000U);
	EXPECT_EQ(reloaded->channels(), 1);
	EXPECT_EQ(reloaded->samples(), processed.samples());
	EXPECT_TRUE(oa::Filesystem::removeFile(outputPath).isOk());

	for (oa::I64 i = 0; i < mel.numElements(); i += 97) {
		EXPECT_TRUE(std::isfinite(mel.at(i))) << "mel element " << i;
	}
}

// ─── oa::FnAudio ────────────────────────────────────────────────────────────────

TEST_VK(TestAudioDecoder, AudioComposesMatrixWithoutAlias)
{
	oa::Audio audio(
		oa::FnMatrix::zeros(oa::MatrixShape{2, 512}),
		48'000U,
		oa::AudioChannelLayout::Stereo);
	ASSERT_TRUE(audio.validate());
	const oa::Matrix& m = audio.asMatrix();
	EXPECT_EQ(m.getShape()[0], 2);
	EXPECT_EQ(m.getShape()[1], 512);
	EXPECT_EQ(audio.sampleRate(), 48'000U);
	EXPECT_EQ(audio.layout(), oa::AudioChannelLayout::Stereo);
}

TEST_VK(TestAudioDecoder, MixMatchesCpuOracle)
{
	const oa::I64 n = 1024;
	auto a = makeSineWave(48000, 440.0F, 1024.0F / 48000.0F);
	auto b = makeSineWave(48000, 880.0F, 1024.0F / 48000.0F);
	ASSERT_GE(static_cast<oa::I64>(a.size()), n);

	auto ma = oa::FnMatrix::empty(oa::MatrixShape{1, n}, oa::ScalarType::Float32);
	auto mb = oa::FnMatrix::empty(oa::MatrixShape{1, n}, oa::ScalarType::Float32);
	oa::memcpy(ma.dataAs<oa::F32>(), a.data(), static_cast<oa::Usize>(n) * sizeof(oa::F32));
	oa::memcpy(mb.dataAs<oa::F32>(), b.data(), static_cast<oa::Usize>(n) * sizeof(oa::F32));

	const oa::F32 gainA = 0.75F, gainB = 0.25F;
	oa::Audio audioA(oa::move(ma), 48'000U, oa::AudioChannelLayout::Mono);
	oa::Audio audioB(oa::move(mb), 48'000U, oa::AudioChannelLayout::Mono);
	oa::Audio mixedAudio = oa::FnAudio::mix(audioA, audioB, gainA, gainB);
	const oa::Matrix& mixed = mixedAudio.asMatrix();
	sync();

	for (oa::I64 i = 0; i < n; ++i) {
		const oa::F32 expect = gainA * a[static_cast<oa::Usize>(i)] + gainB * b[static_cast<oa::Usize>(i)];
		ASSERT_NEAR(mixed.at(i), expect, 1e-6F) << "sample " << i;
	}
}
