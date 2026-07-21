// OA Vision — Wayland screen capture through xdg-desktop-portal + PipeWire.
//
// Wayland deliberately requires an interactive portal grant. Open() may show
// the compositor's monitor/window picker and block until the user accepts or
// cancels. Poll() exposes frames through the common OaVideoFrame contract.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Std/UniquePtr.h>
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/Sync.h>

class OaEngine;

enum class OaScreenCaptureTarget : OaU8 {
	MonitorOrWindow = 0,
	Monitor = 1,
	Window = 2,
};

enum class OaScreenCaptureCursor : OaU8 {
	Hidden = 0,
	Embedded = 1,
};

struct OaScreenCaptureConfig {
	OaScreenCaptureTarget Target = OaScreenCaptureTarget::MonitorOrWindow;
	OaScreenCaptureCursor Cursor = OaScreenCaptureCursor::Embedded;
	OaU32 PreferredWidth = 1920;
	OaU32 PreferredHeight = 1080;
	OaU32 PreferredFps = 30;
	OaU32 RingFrames = 4;
};

class OaScreenCapture {
public:
	// Opaque implementation is public only so C ABI callbacks can name the
	// type; callers never receive or own it.
	struct Impl;

	OaScreenCapture() = default;
	OaScreenCapture(OaScreenCapture&& InOther) noexcept;
	OaScreenCapture& operator=(OaScreenCapture&& InOther) noexcept;
	OaScreenCapture(const OaScreenCapture&) = delete;
	OaScreenCapture& operator=(const OaScreenCapture&) = delete;
	~OaScreenCapture();

	[[nodiscard]] static bool IsSupported() noexcept;
	[[nodiscard]] static OaResult<OaScreenCapture> Open(
		OaEngine& InEngine,
		const OaScreenCaptureConfig& InConfig = {});

	// Non-blocking. Returns true when a newer frame than the previous Poll()
	// is published through the common video-frame contract.
	bool Poll(OaVideoFrame& OutFrame);
	// Release a frame returned by Poll(). The no-token overload declares that
	// no asynchronous consumer remains.
	void Release(const OaVideoFrame& InFrame);
	// GPU-deferred release. DMA-BUF ownership returns to PipeWire only after
	// InConsumed completes. Mapped fallback ring slots are likewise withheld
	// from producer reuse until the exact completion becomes ready.
	void Release(const OaVideoFrame& InFrame, const OaCompletionToken& InConsumed);
	// Stops the producer thread, completes exact frame-consumer dependencies,
	// and releases portal, PipeWire, and Vulkan resources.
	[[nodiscard]] OaStatus Close();
	// Compatibility wrapper that logs Close() failures. Prefer Close() where
	// the shutdown result can be propagated.
	void Destroy();

	[[nodiscard]] bool IsStreaming() const noexcept;
	[[nodiscard]] OaU32 Width() const noexcept;
	[[nodiscard]] OaU32 Height() const noexcept;

private:
	void Abandon_() noexcept;
	static OaStatus CompleteRetired_(void* InPayload);
	static void ReleaseRetired_(void* InPayload);
	OaUniquePtr<Impl> Impl_;
};
