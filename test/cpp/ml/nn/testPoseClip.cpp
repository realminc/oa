#include "../../oaTest.h"

#include <anim/poseClip.h>

TEST(PoseClip, BinaryRoundTripAndTxtExport) {
	oa::Vec<oa::F32> samples;
	constexpr oa::U32 kFrames = 3;
	constexpr oa::U32 kPoseDim = 5;
	for (oa::U32 i = 0; i < kFrames * kPoseDim; ++i) {
		samples.pushBack(static_cast<oa::F32>(i) * 0.25f - 1.0f);
	}

	auto clipResult = oa::PoseClip::create(
		kFrames,
		kPoseDim,
		60.0f,
		7,
		oa::Span<const oa::F32>(samples.data(), samples.size()),
		0x3u);
	ASSERT_TRUE(clipResult.isOk()) << clipResult.getStatus().toString();
	const oa::PoseClip clip = *clipResult;

	const oa::Path dir = oa::Paths::temp() / "oa_poseclip_test";
	ASSERT_TRUE(oa::Filesystem::createDirectories(dir).isOk());
	const oa::Path binPath = dir / "fake_gait.3danim";
	const oa::Path txtPath = dir / "fake_gait.txt";

	ASSERT_TRUE(clip.write3dAnim(binPath).isOk());
	auto loadedResult = oa::PoseClip::read3dAnim(binPath);
	ASSERT_TRUE(loadedResult.isOk()) << loadedResult.getStatus().toString();
	const oa::PoseClip loaded = *loadedResult;

	EXPECT_TRUE(loaded.isValid());
	EXPECT_EQ(loaded.version, oa::PoseClip::formatVersion);
	EXPECT_EQ(loaded.flags, 0x3u);
	EXPECT_EQ(loaded.frameCount, kFrames);
	EXPECT_EQ(loaded.poseDim, kPoseDim);
	EXPECT_FLOAT_EQ(loaded.fps, 60.0f);
	EXPECT_EQ(loaded.skeletonId, 7u);
	ASSERT_EQ(loaded.samples.size(), samples.size());
	for (oa::Usize i = 0; i < samples.size(); ++i) {
		EXPECT_FLOAT_EQ(loaded.samples[i], samples[i]) << "sample " << i;
	}

	ASSERT_TRUE(loaded.writeTxt(txtPath).isOk());
	auto text = oa::Filesystem::readText(txtPath);
	ASSERT_TRUE(text.isOk()) << text.getStatus().toString();
	EXPECT_NE(text->view().find("frames 3"), oa::String::Npos);
	EXPECT_NE(text->view().find("-1"), oa::String::Npos);

	(void)oa::Filesystem::removeDirectory(dir, true);
}

TEST(PoseClip, RejectsBadSampleCount) {
	oa::F32 values[3] = { 1.0f, 2.0f, 3.0f };
	auto result = oa::PoseClip::create(
		2,
		2,
		30.0f,
		0,
		oa::Span<const oa::F32>(values, 3));
	EXPECT_FALSE(result.isOk());
}

