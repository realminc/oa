// OA Vision — live camera capture through the platform SDL3 backend.
//
// On Linux SDL selects the available camera backend (normally PipeWire or
// V4L2). Frames are exposed through the same OaVideoFrame contract used by
// decode and screen capture. The current portable path converts to RGBA8 on
// the host and uploads into a bounded Vulkan-visible ring.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;
class OaVkBuffer;

struct OaCameraCaptureConfig {
	OaI32 DeviceIndex = 0;
	OaI32 Width = 1280;
	OaI32 Height = 720;
	OaI32 Fps = 30;
	OaI32 RingFrames = 4;
	// Linux: try V4L2 MMAP -> EXPBUF -> Vulkan DMA-BUF import before SDL.
	// Empty DevicePath resolves to /dev/video<DeviceIndex>.
	OaString DevicePath;
	bool PreferDmaBuf = true;
	OaU32 ReconnectAttempts = 8;
	OaU32 ReconnectBackoffMs = 100;
};

class OaCameraCapture {
public:
	OaCameraCapture() = default;
	OaCameraCapture(const OaCameraCapture&) = delete;
	OaCameraCapture& operator=(const OaCameraCapture&) = delete;
	OaCameraCapture(OaCameraCapture&&) noexcept;
	OaCameraCapture& operator=(OaCameraCapture&&) noexcept;
	~OaCameraCapture();

	[[nodiscard]] OaStatus Init(
		OaEngine& InRt,
		const OaCameraCaptureConfig& InConfig = {});
	// Explicit completion and release boundary for producer-owned images and
	// mapped ring slots still referenced by GPU consumers.
	[[nodiscard]] OaStatus Close();
	// Compatibility wrapper that logs Close() failures. Prefer Close() where
	// the shutdown result can be propagated.
	void Destroy();

	// Acquire and upload the newest available frame. Returns false when the
	// camera has not produced another frame yet.
	bool Poll();
	bool PollFrame(OaVideoFrame& OutFrame);
	// Release a frame after its final consumer. The no-token overload declares
	// immediate reuse; pass the exact completion when GPU work remains live.
	void Release(const OaVideoFrame& InFrame);
	void Release(const OaVideoFrame& InFrame, const OaCompletionToken& InConsumed);

	[[nodiscard]] OaVkBuffer* LatestFrame() noexcept;
	[[nodiscard]] const OaVkBuffer* LatestFrame() const noexcept;

	[[nodiscard]] OaI32 Width() const noexcept { return Width_; }
	[[nodiscard]] OaI32 Height() const noexcept { return Height_; }
	[[nodiscard]] OaI32 Fps() const noexcept { return Fps_; }
	[[nodiscard]] bool IsStreaming() const noexcept { return Streaming_; }
	[[nodiscard]] bool UsesDmaBuf() const noexcept;
	[[nodiscard]] OaU64 FormatGeneration() const noexcept;
	[[nodiscard]] OaU64 ReconnectCount() const noexcept;

private:
	struct Impl;
	void Abandon_() noexcept;
	static OaStatus CompleteRetired_(void* InPayload);
	static void ReleaseRetired_(void* InPayload);
	OaUniquePtr<Impl> Impl_;
	OaI32 Width_ = 0;
	OaI32 Height_ = 0;
	OaI32 Fps_ = 0;
	bool Streaming_ = false;
	OaU64 LatestTimestampUs_ = 0;
};
