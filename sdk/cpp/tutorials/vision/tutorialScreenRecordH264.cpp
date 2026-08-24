// OA Tutorial — Wayland screen capture -> vulkan Video H.264 -> MP4.
//
// usage:
//   TutorialScreenRecordH264 [output.mp4] [seconds] [audio: 0|1]
//
// The Wayland portal displays its monitor/window picker. The first negotiated
// frame fixes the recording extent; input timestamps are normalized to zero.

#include <oa/runtime/engine.h>
#include <oa/audio/audioCapture.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/videoRecorder.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

volatile std::sig_atomic_t stopRequested = 0;

void requestStop(int) noexcept {
	stopRequested = 1;
}

}

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_screen_capture.mp4";
	const double seconds = argc > 2 ? std::atof(argv[2]) : 10.0;
	const bool wantAudio = argc <= 3 or std::atoi(argv[3]) != 0;
	if (seconds <= 0.0) {
		std::fprintf(stderr, "Duration must be positive\n");
		return 1;
	}
	std::signal(SIGINT, requestStop);
	std::signal(SIGTERM, requestStop);

	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::None;
	engineConfig.selectForThread = true;
	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	oa::Engine& engine = *engineResult.getValue();

	if (not oa::ScreenCapture::isSupported()) {
		std::fprintf(stderr, "This build has no libportal/PipeWire screen backend\n");
		return 1;
	}
	// Declare the completion-event producer before capture so capture drains
	// deferred DMA-BUF releases while the recorder timelines are still alive.
	oa::VideoRecorder recorder;
	std::printf("Select a monitor or window in the Wayland portal...\n");
	auto captureResult = oa::ScreenCapture::open(engine);
	if (not captureResult.isOk()) {
		std::fprintf(stderr, "Screen capture failed: %s\n",
			captureResult.getStatus().toString().cStr());
		return 1;
	}
	oa::ScreenCapture capture = oa::move(*captureResult);
	oa::AudioCapture audioCapture;
	bool audioEnabled = false;
	if (wantAudio) {
		auto audioResult = oa::AudioCapture::open(engine);
		if (audioResult.isOk()) {
			audioCapture = oa::move(*audioResult);
			auto start = audioCapture.start();
			if (start.isOk()) audioEnabled = true;
			else std::fprintf(stderr, "Audio capture unavailable: %s; recording video only\n",
				start.toString().cStr());
		} else {
			std::fprintf(stderr, "Audio capture unavailable: %s; recording video only\n",
				audioResult.getStatus().toString().cStr());
		}
	}

	bool recorderOpen = false;
	oa::U32 frameCount = 0;
	auto drainAudio = [&]() -> oa::Status {
		if (not audioEnabled or not recorderOpen) return oa::Status::ok();
		oa::AudioCaptureChunk chunk;
		while (audioCapture.poll(chunk)) OA_RETURN_IF_ERROR(recorder.writeAudio(chunk));
		return oa::Status::ok();
	};
	const auto deadline = std::chrono::steady_clock::now()
		+ std::chrono::duration<double>(seconds);
	while (not stopRequested and std::chrono::steady_clock::now() < deadline) {
		oa::VideoFrame frame;
		if (not capture.poll(frame)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		if (not recorderOpen) {
			oa::VideoRecorderConfig config;
			config.outputPath = output;
			config.encode.codec = oa::VideoCodec::H264;
			config.encode.width = frame.width;
			config.encode.height = frame.height;
			config.encode.frameRate = 30;
			config.encode.gopSize = 60;
			config.audioEnabled = audioEnabled;
			auto result = oa::VideoRecorder::create(engine, config);
			if (not result.isOk()) {
				std::fprintf(stderr, "Recorder creation failed: %s\n",
					result.getStatus().toString().cStr());
				capture.release(frame);
				return 1;
			}
			recorder = oa::move(*result);
			recorderOpen = true;
			std::printf("Recording %ux%u%s to %s for %.1f seconds\n",
				frame.width, frame.height, audioEnabled ? " + AAC audio" : "",
				output, seconds);
		}
		auto audioStatus = drainAudio();
		if (not audioStatus.isOk()) {
			std::fprintf(stderr, "Record audio failed: %s\n", audioStatus.toString().cStr());
			capture.release(frame);
			return 1;
		}
		oa::Event consumed;
		auto status = recorder.writeAsync(frame, consumed);
		capture.release(frame, consumed);
		if (not status.isOk()) {
			std::fprintf(stderr, "Record frame failed: %s\n", status.toString().cStr());
			return 1;
		}
		++frameCount;
	}

	if (not recorderOpen or frameCount == 0U) {
		std::fprintf(stderr, "portal stream produced no frames\n");
		return 1;
	}
	if (audioEnabled) {
		(void)audioCapture.stop();
		auto audioStatus = drainAudio();
		if (not audioStatus.isOk()) {
			std::fprintf(stderr, "Final audio drain failed: %s\n", audioStatus.toString().cStr());
			return 1;
		}
	}
	auto finalStatus = recorder.finalize();
	if (not finalStatus.isOk()) {
		std::fprintf(stderr, "finalize failed: %s\n", finalStatus.toString().cStr());
		return 1;
	}
	std::printf("Saved %u frames%s to %s\n", frameCount,
		audioEnabled ? " with synchronized AAC audio" : "", output);
	return 0;
}
