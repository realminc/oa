// OA Tutorial — Wayland screen capture -> Vulkan Video H.264 -> MP4.
//
// Usage:
//   TutorialScreenRecordH264 [output.mp4] [seconds] [audio: 0|1]
//
// The Wayland portal displays its monitor/window picker. The first negotiated
// frame fixes the recording extent; input timestamps are normalized to zero.

#include <Oa/Runtime/Engine.h>
#include <Oa/Audio/AudioCapture.h>
#include <Oa/Vision/ScreenCapture.h>
#include <Oa/Vision/VideoRecorder.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
	const char* output = argc > 1 ? argv[1] : "/tmp/oa_screen_capture.mp4";
	const double seconds = argc > 2 ? std::atof(argv[2]) : 10.0;
	const bool wantAudio = argc <= 3 or std::atoi(argv[3]) != 0;
	if (seconds <= 0.0) {
		std::fprintf(stderr, "Duration must be positive\n");
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

	if (not OaScreenCapture::IsSupported()) {
		std::fprintf(stderr, "This build has no libportal/PipeWire screen backend\n");
		return 1;
	}
	// Declare the completion-token producer before capture so capture drains
	// deferred DMA-BUF releases while the recorder timelines are still alive.
	OaVideoRecorder recorder;
	std::printf("Select a monitor or window in the Wayland portal...\n");
	auto captureResult = OaScreenCapture::Open(engine);
	if (not captureResult.IsOk()) {
		std::fprintf(stderr, "Screen capture failed: %s\n",
			captureResult.GetStatus().ToString().c_str());
		return 1;
	}
	OaScreenCapture capture = OaStdMove(*captureResult);
	OaAudioCapture audioCapture;
	bool audioEnabled = false;
	if (wantAudio) {
		auto audioResult = OaAudioCapture::Open(engine);
		if (audioResult.IsOk()) {
			audioCapture = OaStdMove(*audioResult);
			auto start = audioCapture.Start();
			if (start.IsOk()) audioEnabled = true;
			else std::fprintf(stderr, "Audio capture unavailable: %s; recording video only\n",
				start.ToString().c_str());
		} else {
			std::fprintf(stderr, "Audio capture unavailable: %s; recording video only\n",
				audioResult.GetStatus().ToString().c_str());
		}
	}

	bool recorderOpen = false;
	OaU32 frameCount = 0;
	auto DrainAudio = [&]() -> OaStatus {
		if (not audioEnabled or not recorderOpen) return OaStatus::Ok();
		OaAudioCaptureChunk chunk;
		while (audioCapture.Poll(chunk)) OA_RETURN_IF_ERROR(recorder.WriteAudio(chunk));
		return OaStatus::Ok();
	};
	const auto deadline = std::chrono::steady_clock::now()
		+ std::chrono::duration<double>(seconds);
	while (std::chrono::steady_clock::now() < deadline) {
		OaVideoFrame frame;
		if (not capture.Poll(frame)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		if (not recorderOpen) {
			OaVideoRecorderConfig config;
			config.OutputPath = output;
			config.Encode.Codec = OaVideoCodec::H264;
			config.Encode.Width = frame.Width;
			config.Encode.Height = frame.Height;
			config.Encode.FrameRate = 30;
			config.Encode.GopSize = 60;
			config.AudioEnabled = audioEnabled;
			auto result = OaVideoRecorder::Create(engine, config);
			if (not result.IsOk()) {
				std::fprintf(stderr, "Recorder creation failed: %s\n",
					result.GetStatus().ToString().c_str());
				capture.Release(frame);
				return 1;
			}
			recorder = OaStdMove(*result);
			recorderOpen = true;
			std::printf("Recording %ux%u%s to %s for %.1f seconds\n",
				frame.Width, frame.Height, audioEnabled ? " + AAC audio" : "",
				output, seconds);
		}
		auto audioStatus = DrainAudio();
		if (not audioStatus.IsOk()) {
			std::fprintf(stderr, "Record audio failed: %s\n", audioStatus.ToString().c_str());
			capture.Release(frame);
			return 1;
		}
		OaCompletionToken consumed;
		auto status = recorder.WriteAsync(frame, consumed);
		capture.Release(frame, consumed);
		if (not status.IsOk()) {
			std::fprintf(stderr, "Record frame failed: %s\n", status.ToString().c_str());
			return 1;
		}
		++frameCount;
	}

	if (not recorderOpen or frameCount == 0U) {
		std::fprintf(stderr, "Portal stream produced no frames\n");
		return 1;
	}
	if (audioEnabled) {
		(void)audioCapture.Stop();
		auto audioStatus = DrainAudio();
		if (not audioStatus.IsOk()) {
			std::fprintf(stderr, "Final audio drain failed: %s\n", audioStatus.ToString().c_str());
			return 1;
		}
	}
	auto finalStatus = recorder.Finalize();
	if (not finalStatus.IsOk()) {
		std::fprintf(stderr, "Finalize failed: %s\n", finalStatus.ToString().c_str());
		return 1;
	}
	std::printf("Saved %u frames%s to %s\n", frameCount,
		audioEnabled ? " with synchronized AAC audio" : "", output);
	return 0;
}
