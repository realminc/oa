// OA Tutorial — SDL3 camera capture -> vulkan Video H.264 -> MP4.
//
// usage:
//   TutorialCameraRecordH264 [output.mp4] [seconds] [device-index]

// The camera backend publishes the same oa::VideoFrame contract as screen and
// file sources. The recorder is OA's native vulkan Video + MP4 path; no
// ffmpeg subprocess participates in capture or encoding.

#include <oa/runtime/engine.h>
#include <oa/vision/cameraCapture.h>
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
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_camera_capture.mp4";
	const double seconds = argc > 2 ? std::atof(argv[2]) : 10.0;
	const oa::I32 deviceIndex = argc > 3 ? std::atoi(argv[3]) : 0;
	if (seconds <= 0.0 or deviceIndex < 0) {
		std::fprintf(stderr, "Duration must be positive and device index non-negative\n");
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

	// capture holds non-owning completion events returned by the recorder.
	// Destruction order must drain capture before recorder timeline ownership.
	oa::VideoRecorder recorder;
	oa::CameraCapture capture;
	oa::CameraCaptureConfig captureConfig;
	captureConfig.deviceIndex = deviceIndex;
	auto captureResult = oa::CameraCapture::open(engine, captureConfig);
	if (not captureResult.isOk()) {
		std::fprintf(stderr, "camera capture failed: %s\n",
			captureResult.getStatus().toString().cStr());
		return 1;
	}
	capture = oa::move(*captureResult);

	oa::VideoRecorderConfig recorderConfig;
	recorderConfig.outputPath = output;
	recorderConfig.encode.codec = oa::VideoCodec::H264;
	recorderConfig.encode.width = static_cast<oa::U32>(capture.width());
	recorderConfig.encode.height = static_cast<oa::U32>(capture.height());
	recorderConfig.encode.frameRate = static_cast<oa::U32>(capture.fps());
	recorderConfig.encode.gopSize = recorderConfig.encode.frameRate * 2U;
	auto recorderResult = oa::VideoRecorder::create(engine, recorderConfig);
	if (not recorderResult.isOk()) {
		std::fprintf(stderr, "Recorder creation failed: %s\n",
			recorderResult.getStatus().toString().cStr());
		return 1;
	}
	recorder = oa::move(*recorderResult);

	oa::U64 firstPts = 0;
	oa::U32 frameCount = 0;
	const auto deadline = std::chrono::steady_clock::now()
		+ std::chrono::duration<double>(seconds);
	std::printf("Recording camera %d at %dx%d @ %d fps to %s for %.1f seconds\n",
		deviceIndex, capture.width(), capture.height(), capture.fps(), output, seconds);
	while (not stopRequested and std::chrono::steady_clock::now() < deadline) {
		oa::VideoFrame frame;
		if (not capture.pollFrame(frame)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		if (frameCount == 0U) firstPts = frame.presentationTimestamp;
		frame.presentationTimestamp -= firstPts;
		oa::Event consumed;
		auto status = recorder.writeAsync(frame, consumed);
		capture.release(frame, consumed);
		if (not status.isOk()) {
			std::fprintf(stderr, "Record frame failed: %s\n", status.toString().cStr());
			return 1;
		}
		++frameCount;
	}

	if (frameCount == 0U) {
		std::fprintf(stderr, "camera produced no frames\n");
		return 1;
	}
	auto finalStatus = recorder.finalize();
	if (not finalStatus.isOk()) {
		std::fprintf(stderr, "finalize failed: %s\n", finalStatus.toString().cStr());
		return 1;
	}
	std::printf("Saved %u frames to %s\n", frameCount, output);
	return 0;
}
