#include "../../OaTest.h"

#include <Oa/Audio/AudioCapture.h>
#include <Oa/Audio/AudioStream.h>
#include <Oa/Runtime/Engine.h>

#include <chrono>
#include <thread>

TEST(AudioSessionLifecycle, AbandonedLiveSessionsRetireAtEngineClose)
{
	OaEngineConfig engineConfig = OaTestEngineConfig(OaPrecision::FP32);
	engineConfig.PreloadEmbeddedPipelines = false;
	engineConfig.EnablePipelineCache = false;
	auto engineResult = OaEngine::Create(engineConfig);
	ASSERT_TRUE(engineResult.IsOk()) << engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);

	OaString captureSkipReason;
	{
		auto opened = OaAudioCapture::Open(*engine);
		if (not opened.IsOk()) {
			captureSkipReason = opened.GetStatus().ToString();
		} else {
			auto capture = OaStdMove(*opened);
			const auto started = capture.Start();
			if (not started.IsOk()) {
				captureSkipReason = started.ToString();
			} else {
				EXPECT_TRUE(capture.IsStarted());
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
			// No Stop/Close: the callback device moves intact to engine retirement.
		}
	}

	OaString playbackSkipReason;
	{
		OaAudioStreamConfig config;
		config.Uri = OaTestAssetPath("Audio/0_jackson_0.wav").String();
		config.RingMilliseconds = 100U;
		auto opened = OaAudioStream::Open(*engine, config);
		if (not opened.IsOk()) {
			playbackSkipReason = opened.GetStatus().ToString();
		} else {
			auto stream = OaStdMove(*opened);
			const auto playing = stream.Play();
			if (not playing.IsOk()) {
				playbackSkipReason = playing.ToString();
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
			// No Close: callback and decode-thread state move to engine retirement.
		}
	}

	ASSERT_TRUE(engine->Close().IsOk());
	if (not captureSkipReason.empty() or not playbackSkipReason.empty()) {
		GTEST_SKIP()
			<< "capture: " << captureSkipReason.c_str()
			<< "; playback: " << playbackSkipReason.c_str();
	}
}
