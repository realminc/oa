// OaAudio tests — Core audio wrapper composed over OaMatrix.
// Phase 6: Audio wrapper (requires Vulkan since OA is GPU-only).

#include "../OaTest.h"

#include <Oa/Audio.h>
#include <Oa/Core/FnMatrix.h>

// ─── OaAudio Construction ─────────────────────────────────────────────────────

TEST(OaAudio, ConstructMono) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 16000});
	OaAudio audio(std::move(data), 16000, OaChannelLayout::Mono);
	EXPECT_TRUE(audio.Validate());
	EXPECT_EQ(audio.Layout(), OaChannelLayout::Mono);
	EXPECT_EQ(audio.SampleRate(), 16000);
	EXPECT_EQ(audio.Channels(), 1);
	EXPECT_EQ(audio.Samples(), 16000);
}

TEST(OaAudio, ConstructStereo) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{2, 16000});
	OaAudio audio(std::move(data), 44100, OaChannelLayout::Stereo);
	EXPECT_TRUE(audio.Validate());
	EXPECT_EQ(audio.Layout(), OaChannelLayout::Stereo);
	EXPECT_EQ(audio.SampleRate(), 44100);
	EXPECT_EQ(audio.Channels(), 2);
	EXPECT_EQ(audio.Samples(), 16000);
}

TEST(OaAudio, InvalidRank) {
	// OaAudio expects rank 2: [Channels, Samples]
	auto data = OaFnMatrix::Zeros(OaMatrixShape{16000});
	OaAudio audio(std::move(data), 16000, OaChannelLayout::Mono);
	EXPECT_FALSE(audio.Validate());
}

TEST(OaAudio, InvalidChannelCount) {
	// Channel count must match layout
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 16000});
	OaAudio audio(std::move(data), 16000, OaChannelLayout::Stereo);
	EXPECT_FALSE(audio.Validate());
}

TEST(OaAudio, InvalidSampleRate) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{1, 16000});
	OaAudio audio(std::move(data), 0, OaChannelLayout::Mono);
	EXPECT_FALSE(audio.Validate());
}

TEST(OaAudio, UnknownLayoutAllowsCustomChannelCount) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{3, 16000});
	OaAudio audio(std::move(data), 48000, OaChannelLayout::Unknown);
	EXPECT_TRUE(audio.Validate());
	EXPECT_EQ(OaLayoutForChannels(3), OaChannelLayout::Unknown);
}

TEST(OaAudio, AsMatrixRoundTrip) {
	auto data = OaFnMatrix::Zeros(OaMatrixShape{2, 16000});
	OaAudio audio(std::move(data), 44100, OaChannelLayout::Stereo);
	EXPECT_TRUE(audio.Validate());

	// Access underlying tensor
	const OaMatrix& mat = audio.AsMatrix();
	EXPECT_EQ(mat.GetShape().Rank, 2);
	EXPECT_EQ(mat.GetShape()[0], 2);
	EXPECT_EQ(mat.GetShape()[1], 16000);
}

TEST(OaAudio, DefaultConstructed) {
	OaAudio audio;
	EXPECT_TRUE(audio.IsEmpty());
	EXPECT_EQ(audio.Channels(), 0);
	EXPECT_EQ(audio.Samples(), 0);
}
