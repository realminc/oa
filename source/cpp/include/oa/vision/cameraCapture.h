// OA Vision — live camera capture through the platform SDL3 backend.
//
// On Linux SDL selects the available camera backend (normally PipeWire or
// V4L2). Frames are exposed through the same VideoFrame contract used by
// decode and screen capture. The current portable path converts to RGBA8 on
// the host and uploads into a bounded vulkan-visible ring.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/vision/videoDecoder.h>
#include <oa/runtime/event.h>

namespace oa {

class Engine;

struct CameraCaptureConfig {
	oa::I32 deviceIndex = 0;
	oa::I32 width = 1280;
	oa::I32 height = 720;
	oa::I32 fps = 30;
	oa::I32 ringFrames = 4;
	// Linux: try V4L2 MMAP -> EXPBUF -> vulkan DMA-BUF import before SDL.
	// Empty devicePath resolves to /dev/video<deviceIndex>.
	oa::String devicePath;
	bool preferDmaBuf = true;
	oa::U32 reconnectAttempts = 8;
	oa::U32 reconnectBackoffMs = 100;
};

class CameraCapture {
public:
	CameraCapture();
	CameraCapture(const CameraCapture&) = delete;
	CameraCapture& operator=(const CameraCapture&) = delete;
	CameraCapture(CameraCapture&&) noexcept;
	CameraCapture& operator=(CameraCapture&&) noexcept;
	~CameraCapture();

	[[nodiscard]] static Result<CameraCapture> open(
		Engine& inEngine,
		const CameraCaptureConfig& inConfig = {});
	// Explicit completion and release boundary for producer-owned images and
	// mapped ring slots still referenced by GPU consumers.
	[[nodiscard]] Status close();

	// acquire and upload the newest available frame. Returns false when the
	// camera has not produced another frame yet.
	bool poll();
	bool pollFrame(VideoFrame& frame);
	// release a frame after its final consumer. The no-token overload declares
	// immediate reuse; pass the exact completion when GPU work remains live.
	void release(const VideoFrame& frame);
	void release(const VideoFrame& frame, const Event& consumed);

	[[nodiscard]] I32 width() const noexcept { return width_; }
	[[nodiscard]] I32 height() const noexcept { return height_; }
	[[nodiscard]] I32 fps() const noexcept { return fps_; }
	[[nodiscard]] bool isStreaming() const noexcept { return streaming_; }
	[[nodiscard]] bool usesDmaBuf() const noexcept;
	[[nodiscard]] U64 formatGeneration() const noexcept;
	[[nodiscard]] U64 reconnectCount() const noexcept;

private:
	struct Impl;
	[[nodiscard]] Status init_(Engine& inEngine, const CameraCaptureConfig& inConfig);
	void abandon_() noexcept;
	static Status completeRetired_(void* payload);
	static void releaseRetired_(void* payload);
	UniquePtr<Impl> impl_;
	I32 width_ = 0;
	I32 height_ = 0;
	I32 fps_ = 0;
	bool streaming_ = false;
	U64 latestTimestampUs_ = 0;
};

} // namespace oa
