// OA Vision — Wayland screen capture through xdg-desktop-portal + PipeWire.
//
// Wayland deliberately requires an interactive portal grant. open() may show
// the compositor's monitor/window picker and block until the user accepts or
// cancels. poll() exposes frames through the common VideoFrame contract.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/vision/videoDecoder.h>
#include <oa/runtime/event.h>

namespace oa {

class Engine;

enum class ScreenCaptureTarget : oa::U8 {
	MonitorOrWindow = 0,
	Monitor = 1,
	Window = 2,
};

enum class ScreenCaptureCursor : oa::U8 {
	Hidden = 0,
	Embedded = 1,
};

struct ScreenCaptureConfig {
	ScreenCaptureTarget target = ScreenCaptureTarget::MonitorOrWindow;
	ScreenCaptureCursor cursor = ScreenCaptureCursor::Embedded;
	oa::U32 preferredWidth = 1920;
	oa::U32 preferredHeight = 1080;
	oa::U32 preferredFps = 30;
	oa::U32 ringFrames = 4;
};

class ScreenCapture {
public:
	// Opaque implementation is public only so C ABI callbacks can name the
	// type; callers never receive or own it.
	struct Impl;

	ScreenCapture();
	ScreenCapture(ScreenCapture&& inOther) noexcept;
	ScreenCapture& operator=(ScreenCapture&& inOther) noexcept;
	ScreenCapture(const ScreenCapture&) = delete;
	ScreenCapture& operator=(const ScreenCapture&) = delete;
	~ScreenCapture();

	[[nodiscard]] static bool isSupported() noexcept;
	[[nodiscard]] static oa::Result<ScreenCapture> open(
		oa::Engine& inEngine,
		const ScreenCaptureConfig& inConfig = {});

	// Non-blocking. Returns true when a newer frame than the previous poll()
	// is published through the common video-frame contract.
	bool poll(VideoFrame& outFrame);
	// release a frame returned by poll(). The no-token overload declares that
	// no asynchronous consumer remains.
	void release(const VideoFrame& inFrame);
	// GPU-deferred release. DMA-BUF ownership returns to PipeWire only after
	// inConsumed completes. mapped fallback ring slots are likewise withheld
	// from producer reuse until the exact completion becomes ready.
	void release(const VideoFrame& inFrame, const oa::Event& inConsumed);
	// Stops the producer thread, completes exact frame-consumer dependencies,
	// and releases portal, PipeWire, and vulkan resources.
	[[nodiscard]] oa::Status close();

	[[nodiscard]] bool isStreaming() const noexcept;
	[[nodiscard]] oa::U32 width() const noexcept;
	[[nodiscard]] oa::U32 height() const noexcept;

private:
	void abandon_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
