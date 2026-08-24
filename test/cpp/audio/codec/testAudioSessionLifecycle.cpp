#include "../../oaTest.h"

#include <oa/audio/audioCapture.h>
#include <oa/audio/audioPlayer.h>
#include <oa/runtime/engine.h>

#include <chrono>
#include <thread>

TEST(AudioSessionLifecycle, AbandonedLiveSessionsRetireAtEngineClose)
{
	oa::EngineConfig engineConfig = testEngineConfig(oa::Precision::FP32);
	engineConfig.preloadEmbeddedPipelines = false;
	engineConfig.enablePipelineCache = false;
	auto engineResult = oa::Engine::create(engineConfig);
	ASSERT_TRUE(engineResult.isOk()) << engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);

	oa::String captureSkipReason;
	{
		auto opened = oa::AudioCapture::open(*engine);
		if (not opened.isOk()) {
			captureSkipReason = opened.getStatus().toString();
		} else {
			auto capture = oa::move(*opened);
			const auto started = capture.start();
			if (not started.isOk()) {
				captureSkipReason = started.toString();
			} else {
				EXPECT_TRUE(capture.isStarted());
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
			// No Stop/Close: the callback device moves intact to engine retirement.
		}
	}

	oa::String playbackSkipReason;
	{
		oa::AudioPlayerConfig config;
		config.uri = testAssetPath("audio/oaNarration.wav").string();
		config.ringMilliseconds = 100U;
		auto opened = oa::AudioPlayer::open(*engine, config);
		if (not opened.isOk()) {
			playbackSkipReason = opened.getStatus().toString();
		} else {
			auto stream = oa::move(*opened);
			const auto playing = stream.play();
			if (not playing.isOk()) {
				playbackSkipReason = playing.toString();
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
			// No Close: callback and decode-thread state move to engine retirement.
		}
	}

	ASSERT_TRUE(engine->close().isOk());
	if (not captureSkipReason.empty() or not playbackSkipReason.empty()) {
		GTEST_SKIP()
			<< "capture: " << captureSkipReason.cStr()
			<< "; playback: " << playbackSkipReason.cStr();
	}
}
