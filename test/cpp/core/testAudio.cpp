// oa::Audio tests — Core audio wrapper composed over oa::Matrix.
// phase 6: Audio wrapper (requires vulkan since OA is GPU-only).

#include "../oaTest.h"

#include <oa/audio.h>
#include <oa/core/fnMatrix.h>

// ─── oa::Audio Construction ─────────────────────────────────────────────────────

TEST(Audio, ConstructMono) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 16000});
	oa::Audio audio(std::move(data), 16000, oa::AudioChannelLayout::Mono);
	EXPECT_TRUE(audio.validate());
	EXPECT_EQ(audio.layout(), oa::AudioChannelLayout::Mono);
	EXPECT_EQ(audio.sampleRate(), 16000);
	EXPECT_EQ(audio.channels(), 1);
	EXPECT_EQ(audio.samples(), 16000);
}

TEST(Audio, ConstructStereo) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{2, 16000});
	oa::Audio audio(std::move(data), 44100, oa::AudioChannelLayout::Stereo);
	EXPECT_TRUE(audio.validate());
	EXPECT_EQ(audio.layout(), oa::AudioChannelLayout::Stereo);
	EXPECT_EQ(audio.sampleRate(), 44100);
	EXPECT_EQ(audio.channels(), 2);
	EXPECT_EQ(audio.samples(), 16000);
}

TEST(Audio, InvalidRank) {
	// oa::Audio expects rank 2: [channels, samples]
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{16000});
	oa::Audio audio(std::move(data), 16000, oa::AudioChannelLayout::Mono);
	EXPECT_FALSE(audio.validate());
}

TEST(Audio, InvalidChannelCount) {
	// Channel count must match layout
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 16000});
	oa::Audio audio(std::move(data), 16000, oa::AudioChannelLayout::Stereo);
	EXPECT_FALSE(audio.validate());
}

TEST(Audio, ConstructStereo21) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 16000});
	oa::Audio audio(std::move(data), 48000, oa::AudioChannelLayout::Stereo21);
	EXPECT_TRUE(audio.validate());
	EXPECT_EQ(audio.channels(), 3);
	EXPECT_EQ(oa::channelsForLayout(audio.layout()), 3);
}

TEST(Audio, InvalidSampleRate) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{1, 16000});
	oa::Audio audio(std::move(data), 0, oa::AudioChannelLayout::Mono);
	EXPECT_FALSE(audio.validate());
}

TEST(Audio, UnknownLayoutAllowsCustomChannelCount) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{3, 16000});
	oa::Audio audio(std::move(data), 48000, oa::AudioChannelLayout::Unknown);
	EXPECT_TRUE(audio.validate());
	EXPECT_EQ(oa::layoutForChannels(3), oa::AudioChannelLayout::Unknown);
	EXPECT_EQ(oa::channelsForLayout(oa::AudioChannelLayout::Stereo21), 3);
}

TEST(Audio, AsMatrixRoundTrip) {
	auto data = oa::FnMatrix::zeros(oa::MatrixShape{2, 16000});
	oa::Audio audio(std::move(data), 44100, oa::AudioChannelLayout::Stereo);
	EXPECT_TRUE(audio.validate());

	// access underlying tensor
	const oa::Matrix& mat = audio.asMatrix();
	EXPECT_EQ(mat.getShape().rank, 2);
	EXPECT_EQ(mat.getShape()[0], 2);
	EXPECT_EQ(mat.getShape()[1], 16000);
}

TEST(Audio, DefaultConstructed) {
	oa::Audio audio;
	EXPECT_TRUE(audio.isEmpty());
	EXPECT_EQ(audio.channels(), 0);
	EXPECT_EQ(audio.samples(), 0);
}
