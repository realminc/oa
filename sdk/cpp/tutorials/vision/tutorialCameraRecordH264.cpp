// OA Tutorial — SDL3 camera capture -> vulkan Video H.264 -> MP4.
//
// usage:
//   TutorialCameraRecordH264 [output.mp4] [seconds] [device-index]

// The camera backend publishes the same oa::VideoFrame contract as screen and
// file sources. The recorder is OA's native vulkan Video + MP4 path; no
// ffmpeg subprocess participates in capture or encoding.

#include <oa/runtime/engine.h>
#include <oa/core/thread.h>
#include <oa/vision/cameraCapture.h>
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
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_camera_capture.mp4";
	const double seconds = argc > 2 ? ::atof(argv[2]) : 10.0;
	const oa::I32 deviceIndex = argc > 3 ? ::atoi(argv[3]) : 0;
	if (seconds <= 0.0 or deviceIndex < 0) {
		oa::print(oa::PrintStream::Error, "Duration must be positive and device index non-negative");
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

	// capture holds non-owning completion events returned by the recorder.
	// Destruction order must drain capture before recorder timeline ownership.
	oa::VideoRecorder recorder;
	oa::CameraCapture capture;
	oa::CameraCaptureConfig captureConfig;
	captureConfig.deviceIndex = deviceIndex;
	auto captureResult = oa::CameraCapture::open(engine, captureConfig);
	if (not captureResult.isOk()) {
		oa::print(oa::PrintStream::Error, "camera capture failed: {}",
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
		oa::print(oa::PrintStream::Error, "Recorder creation failed: {}",
			recorderResult.getStatus().toString().cStr());
		return 1;
	}
	recorder = oa::move(*recorderResult);

	oa::U64 firstPts = 0;
	oa::U32 frameCount = 0;
	const auto deadline = oa::steadyNow() + oa::Duration::fromDouble(seconds);
	oa::print("Recording camera {} at {}x{} @ {} fps to {} for {:.1f} seconds",
		deviceIndex, capture.width(), capture.height(), capture.fps(), output, seconds);
	while (not stopRequested and oa::steadyNow() < deadline) {
		oa::VideoFrame frame;
		if (not capture.pollFrame(frame)) {
			oa::Thread::sleepFor(oa::Duration::fromMilliseconds(1));
			continue;
		}
		if (frameCount == 0U) firstPts = frame.presentationTimestamp;
		frame.presentationTimestamp -= firstPts;
		oa::Event consumed;
		auto status = recorder.writeAsync(frame, consumed);
		capture.release(frame, consumed);
		if (not status.isOk()) {
			oa::print(oa::PrintStream::Error, "Record frame failed: {}", status.toString().cStr());
			return 1;
		}
		++frameCount;
	}

	if (frameCount == 0U) {
		oa::print(oa::PrintStream::Error, "camera produced no frames");
		return 1;
	}
	auto finalStatus = recorder.finalize();
	if (not finalStatus.isOk()) {
		oa::print(oa::PrintStream::Error, "finalize failed: {}", finalStatus.toString().cStr());
		return 1;
	}
	oa::print("Saved {} frames to {}", frameCount, output);
	return 0;
}
