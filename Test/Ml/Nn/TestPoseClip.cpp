#include "../../OaTest.h"

#include <Anim/PoseClip.h>

TEST(PoseClip, BinaryRoundTripAndTxtExport) {
	OaVec<OaF32> samples;
	constexpr OaU32 kFrames = 3;
	constexpr OaU32 kPoseDim = 5;
	for (OaU32 i = 0; i < kFrames * kPoseDim; ++i) {
		samples.PushBack(static_cast<OaF32>(i) * 0.25f - 1.0f);
	}

	auto clipResult = OaPoseClip::Create(
		kFrames,
		kPoseDim,
		60.0f,
		7,
		OaSpan<const OaF32>(samples.Data(), samples.Size()),
		0x3u);
	ASSERT_TRUE(clipResult.IsOk()) << clipResult.GetStatus().ToString();
	const OaPoseClip clip = *clipResult;

	const OaPath dir = OaPaths::Temp() / "oa_poseclip_test";
	ASSERT_TRUE(OaFilesystem::CreateDirectories(dir).IsOk());
	const OaPath binPath = dir / "fake_gait.3danim";
	const OaPath txtPath = dir / "fake_gait.txt";

	ASSERT_TRUE(clip.Write3dAnim(binPath).IsOk());
	auto loadedResult = OaPoseClip::Read3dAnim(binPath);
	ASSERT_TRUE(loadedResult.IsOk()) << loadedResult.GetStatus().ToString();
	const OaPoseClip loaded = *loadedResult;

	EXPECT_TRUE(loaded.IsValid());
	EXPECT_EQ(loaded.Version, OaPoseClip::FormatVersion);
	EXPECT_EQ(loaded.Flags, 0x3u);
	EXPECT_EQ(loaded.FrameCount, kFrames);
	EXPECT_EQ(loaded.PoseDim, kPoseDim);
	EXPECT_FLOAT_EQ(loaded.Fps, 60.0f);
	EXPECT_EQ(loaded.SkeletonId, 7u);
	ASSERT_EQ(loaded.Samples.Size(), samples.Size());
	for (OaUsize i = 0; i < samples.Size(); ++i) {
		EXPECT_FLOAT_EQ(loaded.Samples[i], samples[i]) << "sample " << i;
	}

	ASSERT_TRUE(loaded.WriteTxt(txtPath).IsOk());
	auto text = OaFilesystem::ReadText(txtPath);
	ASSERT_TRUE(text.IsOk()) << text.GetStatus().ToString();
	EXPECT_NE(text->View().find("frames 3"), OaStdString::npos);
	EXPECT_NE(text->View().find("-1"), OaStdString::npos);

	(void)OaFilesystem::RemoveDirectory(dir, true);
}

TEST(PoseClip, RejectsBadSampleCount) {
	OaF32 values[3] = { 1.0f, 2.0f, 3.0f };
	auto result = OaPoseClip::Create(
		2,
		2,
		30.0f,
		0,
		OaSpan<const OaF32>(values, 3));
	EXPECT_FALSE(result.IsOk());
}

