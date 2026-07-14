// OaViewport — Universal GPU-accelerated viewer application.
//
// Lives at <Oa/Core/Viewport.h> as of UnifiedExecutionArchitecture.md Step 3d: the viewer is the
// first user-facing consumer of the unified OaContext recorder, so it sits in
// Core alongside OaContext/OaTexture rather than in Ui/. It still depends on
// the Ui/ layer (it inherits OaDeviceUiApp), but its role is foundational.
//
// Today: a ready-to-use image viewer with minimal boilerplate, similar to
// OpenCV's cv::imshow() but Vulkan-native and bindless. Provides channel
// viewing (R/G/B/A/RGB), letterboxed display, and keyboard shortcuts out
// of the box.
//
// Roadmap (UnifiedExecutionArchitecture.md §9): one viewer for every kind of GPU resource —
// image stacks, video (OaVideoDecoder), 3D meshes/scenes (OaPass pipeline),
// camera streams (OaCameraCapture), matrix-as-heatmap, stable-diffusion live
// previews via the LiveTexture mailbox pattern. Wayland touchpad gestures
// (pinch-zoom, two-finger pan). Threaded UI so SD generation can update
// the texture while the user tweaks knobs.
//
// Usage:
//   OaViewport viewer;
//   viewer.SetPath("path/to/image.jpg");
//   return viewer.Run();
//
// Or with custom config:
//   OaViewportConfig config;
//   config.Path = "image.png";
//   config.Title = "My Viewer";
//   config.KeyRed = OuiKey::Num1;  // customize shortcuts
//   OaViewport viewer(config);
//   return viewer.Run();
//
// Batch / headless usage (no window, no swapchain):
//   OaViewport::Save(engine, tex, "/tmp/out.png");

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Constant.h>
#include <Oa/Core/Navigation.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Input.h>

class OaComputeEngine;

// ─── OaViewportConfig ──────────────────────────────────────────────────────

struct OaViewportConfig {
	// Image path to load
	OaString Path = "Asset/Image/Realm1024px.jpg";

	// Window configuration
	OaString Title  = OA_TITLE_VIEWPORT;
	OaU32    Width  = 1280;
	OaU32    Height = 720;

	// Keyboard shortcuts (customizable)
	OuiKey KeyQuit   = OuiKey::Escape;
	OuiKey KeyQuitQ  = OuiKey::Q;
	OuiKey KeyRed    = OuiKey::Num1;
	OuiKey KeyGreen  = OuiKey::Num2;
	OuiKey KeyBlue   = OuiKey::Num3;
	OuiKey KeyAlpha  = OuiKey::Num4;
	OuiKey KeyRGB    = OuiKey::Num5;
	OuiKey KeyZoomIn = OuiKey::Equals;   // + or =
	OuiKey KeyZoomOut= OuiKey::Minus;   // -
	OuiKey KeyZoomFit= OuiKey::Num0;    // 0
	OuiKey KeyZoom100= OuiKey::Num9;    // 9 (100%)
	OuiKey KeyPanUp  = OuiKey::Kp8;
	OuiKey KeyPanDown= OuiKey::Kp2;
	OuiKey KeyPanLeft= OuiKey::Kp4;
	OuiKey KeyPanRight= OuiKey::Kp6;

	// Display options
	bool ShowHelp = true;  // Show keyboard shortcuts on startup
};

// ─── OaViewport ────────────────────────────────────────────────────────────

class OaViewport : public OaDeviceUiApp {
public:
	OaViewport() = default;
	explicit OaViewport(const char* InPath) { Config_.Path = InPath; }
	explicit OaViewport(const OaString& InPath) { Config_.Path = InPath; }
	explicit OaViewport(const OaViewportConfig& InConfig) : Config_(InConfig) {}

	void SetPath(const OaString& InPath) { Config_.Path = InPath; }
	void SetPath(const char* InPath) { Config_.Path = InPath; }
	void SetConfig(const OaViewportConfig& InConfig) { Config_ = InConfig; }

	// Run the viewer (blocks until window closed)
	[[nodiscard]] OaStatus Run();

	// Batch sink (UnifiedExecutionArchitecture.md §3.5 SaveImage): readback InTex and write it to
	// disk via OaFnImage::SaveFile. No window, no swapchain. Sibling of Run().
	[[nodiscard]] static OaStatus Save(
		OaComputeEngine& InEngine,
		const OaTexture&   InTex,
		const char*        InPath);

	// OaDeviceUiApp overrides
	void OnInit(OaDeviceUi& InGpui) override;
	void OnUpdate(OaF32 InDeltaMs) override;
	void OnRender(OaUi& InOui) override;
	void OnEvent(const OaUiEvent& InEvent) override;
	void OnShutdown(OaDeviceUi& InGpui) override;

private:
	enum class ImageViewMode : OaU8 { RGB = 0, R = 1, G = 2, B = 3, A = 4 };

	OaViewportConfig Config_;
	OaTexture        Image_;
	OaImagePlanes    Planes_;
	ImageViewMode    ImageMode_ = ImageViewMode::RGB;
	OaNavigation     Nav_;
};
