#include "../../oaTest.h"
#include "../videoTestSupport.h"

#include <oa/runtime/engine.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoDecoder.h>

TEST(VideoSessionLifecycle, AbandonedDecodedSourceRetiresAtEngineClose)
{
	oa::EngineConfig engineConfig = testEngineConfig(oa::Precision::FP32);
	engineConfig.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(engineConfig);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);

	if (not testVideoDecodeSupported(*engine, oa::VideoCodec::H264)) {
		ASSERT_TRUE(engine->close().isOk());
		GTEST_SKIP() << "vulkan Video H.264 decode not supported";
	}

	oa::VideoPlayerConfig config;
	config.uri = testAssetPath("video/shibuya720pH264HighEightBit420.mp4").string();
	config.audio = false;
	config.startPlaying = false;
	config.loop = false;
	{
		auto opened = oa::VideoPlayer::open(*engine, config);
		ASSERT_TRUE(opened.isOk()) << opened.getStatus().toString();
		auto video = oa::move(*opened);
		ASSERT_NE(video.currentFrame().imageView, VK_NULL_HANDLE);
		ASSERT_TRUE(video.currentFrame().ready.isValid());
		video.markCurrentFrameConsumed(video.currentFrame().ready);
		// No Close: decoder, pool, stream, and exact consumer event move intact
		// to the engine-owned composed-service retirement queue.
	}

	ASSERT_TRUE(engine->close().isOk());
}
