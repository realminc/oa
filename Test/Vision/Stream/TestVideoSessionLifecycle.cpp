#include "../../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Vision/Video.h>
#include <Oa/Vision/VideoDecoder.h>

TEST(VideoSessionLifecycle, AbandonedDecodedSourceRetiresAtEngineClose)
{
	OaEngineConfig engineConfig = OaTestEngineConfig(OaPrecision::FP32);
	engineConfig.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(engineConfig);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);

	if (not OaVideoDecoder::IsCodecSupported(*engine, OaVideoCodec::H264)) {
		ASSERT_TRUE(engine->Close().IsOk());
		GTEST_SKIP() << "Vulkan Video H.264 decode not supported";
	}

	OaVideoConfig config;
	config.Uri = OaTestAssetPath("Video/shibuya_720p_h264_high_8bit_420.mp4").String();
	config.Audio = false;
	config.StartPlaying = false;
	config.Loop = false;
	{
		auto opened = OaVideo::Open(*engine, config);
		ASSERT_TRUE(opened.IsOk()) << opened.GetStatus().ToString();
		auto video = OaStdMove(*opened);
		ASSERT_NE(video.CurrentFrame().ImageView, VK_NULL_HANDLE);
		ASSERT_TRUE(video.CurrentFrame().Ready.IsValid());
		video.MarkCurrentFrameConsumed(video.CurrentFrame().Ready);
		// No Close: decoder, pool, stream, and exact consumer event move intact
		// to the engine-owned composed-service retirement queue.
	}

	ASSERT_TRUE(engine->Close().IsOk());
}
