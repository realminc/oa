// OA Tutorial — Wayland screen capture -> vulkan Video H.264 -> MP4.
//
// usage:
//   TutorialScreenRecordH264 [output.mp4] [seconds] [audio: 0|1]
//
// The Wayland portal displays its monitor/window picker. The first negotiated
// frame fixes the recording extent; input timestamps are normalized to zero.

#include <oa/runtime/engine.h>
#include <oa/core/thread.h>
#include <oa/audio/audioCapture.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/videoRecorder.h>

#include <signal.h>
#include <stdlib.h>

namespace {

volatile ::sig_atomic_t stopRequested = 0;

void requestStop(int) noexcept {
	stopRequested = 1;
}

}

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_screen_capture.mp4";
	const double seconds = argc > 2 ? ::atof(argv[2]) : 10.0;
	const bool wantAudio = argc <= 3 or ::atoi(argv[3]) != 0;
	if (seconds <= 0.0) {
		oa::print(oa::PrintStream::Error, "Duration must be positive");
		return 1;
	}
	::signal(SIGINT, requestStop);
	::signal(SIGTERM, requestStop);

	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::None;
	engineConfig.selectForThread = true;
	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()) {
		oa::print(oa::PrintStream::Error, "Engine creation failed: {}",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	oa::Engine& engine = *engineResult.getValue();

	if (not oa::ScreenCapture::isSupported()) {
		oa::print(oa::PrintStream::Error, "This build has no libportal/PipeWire screen backend");
		return 1;
	}
	// Declare the completion-event producer before capture so capture drains
	// deferred DMA-BUF releases while the recorder timelines are still alive.
	oa::VideoRecorder recorder;
	oa::print("Select a monitor or window in the Wayland portal...");
	auto captureResult = oa::ScreenCapture::open(engine);
	if (not captureResult.isOk()) {
		oa::print(oa::PrintStream::Error, "Screen capture failed: {}",
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
			else oa::print(oa::PrintStream::Error, "Audio capture unavailable: {}; recording video only",
				start.toString().cStr());
		} else {
			oa::print(oa::PrintStream::Error, "Audio capture unavailable: {}; recording video only",
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
	const auto deadline = oa::steadyNow() + oa::Duration::fromDouble(seconds);
	while (not stopRequested and oa::steadyNow() < deadline) {
		oa::VideoFrame frame;
		if (not capture.poll(frame)) {
			oa::Thread::sleepFor(oa::Duration::fromMilliseconds(1));
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
				oa::print(oa::PrintStream::Error, "Recorder creation failed: {}",
					result.getStatus().toString().cStr());
				capture.release(frame);
				return 1;
			}
			recorder = oa::move(*result);
			recorderOpen = true;
			oa::print("Recording {}x{}{} to {} for {:.1f} seconds",
				frame.width, frame.height, audioEnabled ? " + AAC audio" : "",
				output, seconds);
		}
		auto audioStatus = drainAudio();
		if (not audioStatus.isOk()) {
			oa::print(oa::PrintStream::Error, "Record audio failed: {}", audioStatus.toString().cStr());
			capture.release(frame);
			return 1;
		}
		oa::Event consumed;
		auto status = recorder.writeAsync(frame, consumed);
		capture.release(frame, consumed);
		if (not status.isOk()) {
			oa::print(oa::PrintStream::Error, "Record frame failed: {}", status.toString().cStr());
			return 1;
		}
		++frameCount;
	}

	if (not recorderOpen or frameCount == 0U) {
		oa::print(oa::PrintStream::Error, "portal stream produced no frames");
		return 1;
	}
	if (audioEnabled) {
		(void)audioCapture.stop();
		auto audioStatus = drainAudio();
		if (not audioStatus.isOk()) {
			oa::print(oa::PrintStream::Error, "Final audio drain failed: {}", audioStatus.toString().cStr());
			return 1;
		}
	}
	auto finalStatus = recorder.finalize();
	if (not finalStatus.isOk()) {
		oa::print(oa::PrintStream::Error, "finalize failed: {}", finalStatus.toString().cStr());
		return 1;
	}
	oa::print("Saved {} frames{} to {}", frameCount,
		audioEnabled ? " with synchronized AAC audio" : "", output);
	return 0;
}
