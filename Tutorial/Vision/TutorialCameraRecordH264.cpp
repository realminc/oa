// OA Tutorial — SDL3 camera capture -> Vulkan Video H.264 -> MP4.
//
// Usage:
//   TutorialCameraRecordH264 [output.mp4] [seconds] [device-index]

// The camera backend publishes the same OaVideoFrame contract as screen and
// file sources. The recorder is OA's native Vulkan Video + MP4 path; no
// ffmpeg subprocess participates in capture or encoding.

#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Capture.h>
#include <Oa/Vision/VideoRecorder.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_camera_capture.mp4";
	const double seconds = argc > 2 ? std::atof(argv[2]) : 10.0;
	const OaI32 deviceIndex = argc > 3 ? std::atoi(argv[3]) : 0;
	if (seconds <= 0.0 or deviceIndex < 0) {
		std::fprintf(stderr, "Duration must be positive and device index non-negative\n");
		return 1;
	}

	OaEngineConfig engineConfig;
	engineConfig.PresentationMode = OaPresentationMode::None;
	engineConfig.RegisterAsGlobal = true;
	auto engineResult = OaEngine::Create(engineConfig);
	if (not engineResult.IsOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			engineResult.GetStatus().ToString().c_str());
		return 1;
	}
	OaEngine& engine = *engineResult.GetValue();

	// Capture holds non-owning completion tokens returned by the recorder.
	// Destruction order must drain capture before recorder timeline ownership.
	OaVideoRecorder recorder;
	OaCapture capture;
	OaCaptureConfig captureConfig;
	captureConfig.DeviceIndex = deviceIndex;
	auto captureStatus = capture.Init(engine, captureConfig);
	if (not captureStatus.IsOk()) {
		std::fprintf(stderr, "Camera capture failed: %s\n",
			captureStatus.ToString().c_str());
		return 1;
	}

	OaVideoRecorderConfig recorderConfig;
	recorderConfig.OutputPath = output;
	recorderConfig.Encode.Codec = OaVideoCodec::H264;
	recorderConfig.Encode.Width = static_cast<OaU32>(capture.Width());
	recorderConfig.Encode.Height = static_cast<OaU32>(capture.Height());
	recorderConfig.Encode.FrameRate = static_cast<OaU32>(capture.Fps());
	recorderConfig.Encode.GopSize = recorderConfig.Encode.FrameRate * 2U;
	auto recorderResult = OaVideoRecorder::Create(engine, recorderConfig);
	if (not recorderResult.IsOk()) {
		std::fprintf(stderr, "Recorder creation failed: %s\n",
			recorderResult.GetStatus().ToString().c_str());
		return 1;
	}
	recorder = OaStdMove(*recorderResult);

	OaU64 firstPts = 0;
	OaU32 frameCount = 0;
	const auto deadline = std::chrono::steady_clock::now()
		+ std::chrono::duration<double>(seconds);
	std::printf("Recording camera %d at %dx%d @ %d fps to %s for %.1f seconds\n",
		deviceIndex, capture.Width(), capture.Height(), capture.Fps(), output, seconds);
	while (std::chrono::steady_clock::now() < deadline) {
		OaVideoFrame frame;
		if (not capture.PollFrame(frame)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		if (frameCount == 0U) firstPts = frame.PresentationTimestamp;
		frame.PresentationTimestamp -= firstPts;
		OaCompletionToken consumed;
		auto status = recorder.WriteAsync(frame, consumed);
		capture.Release(frame, consumed);
		if (not status.IsOk()) {
			std::fprintf(stderr, "Record frame failed: %s\n", status.ToString().c_str());
			return 1;
		}
		++frameCount;
	}

	if (frameCount == 0U) {
		std::fprintf(stderr, "Camera produced no frames\n");
		return 1;
	}
	auto finalStatus = recorder.Finalize();
	if (not finalStatus.IsOk()) {
		std::fprintf(stderr, "Finalize failed: %s\n", finalStatus.ToString().c_str());
		return 1;
	}
	std::printf("Saved %u frames to %s\n", frameCount, output);
	return 0;
}
