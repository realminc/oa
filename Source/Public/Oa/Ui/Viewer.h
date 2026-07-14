// OaViewport — Universal 3D/2D viewport wrapper.
//
// High-level wrapper around presentation/swapchain/window that aligns with
// the rendering system. Supports both 3D (perspective camera) and 2D
// (orthographic camera) rendering modes.
//
// Lives at <Oa/Core/Viewport.h> as a foundational consumer of the unified
// OaContext recorder. It depends on the Ui/ layer (inherits OaDeviceUiApp)
// but provides a higher-level abstraction for viewport rendering.
//
// Capabilities:
// - 2D mode: Image/video viewing with orthographic camera (current functionality)
// - 3D mode: Scene rendering with perspective camera (new functionality)
// - Uses OaCamera for view/projection
// - Uses OaMesh for geometry
// - Uses OaPass for render pipelines
// - Animated pan/zoom via OaAnimControl
// - Wayland touchpad gestures (pinch-zoom, two-finger pan)
// - Threaded UI for live previews (LiveTexture mailbox pattern)
//
// Usage (2D image viewer):
//   OaViewport viewer(OaViewportMode::Image2D);
//   viewer.SetContent("path/to/image.jpg");
//   return viewer.Run();
//
// Usage (3D scene viewer):
//   OaViewport viewer(OaViewportMode::Scene3D);
//   viewer.SetScene(&scene);
//   viewer.SetCamera(&camera);
//   return viewer.Run();
//
// Batch / headless usage (no window, no swapchain):
//   OaViewport::Save(engine, tex, "/tmp/out.png");

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Constant.h>
#include <Oa/Core/Navigation.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Render/Scene.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Input.h>

class OaComputeEngine;

// ─── OaViewportMode ────────────────────────────────────────────────────────

enum class OaViewerMode : OaU8 {
	Image2D,   // 2D image/video viewer (orthographic camera)
	Scene3D,   // 3D scene viewer (perspective camera)
	Matrix,    // Matrix-as-heatmap viewer
	Video,     // Video player with timeline control
};

// ─── OaViewportConfig ──────────────────────────────────────────────────────

struct OaViewerConfig {
	// Viewport mode
	OaViewerMode Mode = OaViewerMode::Image2D;

	// Content (2D mode)
	OaString Path = "Asset/Image/Realm1024px.jpg";

	// Content (3D mode)
	OaScene*  Scene  = nullptr;
	OaCamera* Camera = nullptr;

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
	OuiKey KeyPanUp  = OuiKey::Up;
	OuiKey KeyPanDown= OuiKey::Down;
	OuiKey KeyPanLeft= OuiKey::Left;
	OuiKey KeyPanRight= OuiKey::Right;

	// Display options
	bool ShowHelp = true;  // Show keyboard shortcuts on startup
};

// ─── OaViewer ─────────────────────────────────────────────────────────────

class OaViewer : public OaDeviceUiApp {
public:
	OaViewer() = default;
	explicit OaViewer(OaViewerMode InMode) { Config_.Mode = InMode; }
	explicit OaViewer(const char* InPath) {
		Config_.Mode = OaViewerMode::Image2D;
		Config_.Path = InPath;
	}
	explicit OaViewer(const OaString& InPath) {
		Config_.Mode = OaViewerMode::Image2D;
		Config_.Path = InPath;
	}
	explicit OaViewer(const OaViewerConfig& InConfig) : Config_(InConfig) {}

	// Content setters
	void SetMode(OaViewerMode InMode) { Config_.Mode = InMode; }
	void SetPath(const OaString& InPath) { Config_.Path = InPath; }
	void SetPath(const char* InPath) { Config_.Path = InPath; }
	void SetScene(OaScene* InScene) { Config_.Scene = InScene; }
	void SetCamera(OaCamera* InCamera) { Config_.Camera = InCamera; }
	void SetConfig(const OaViewerConfig& InConfig) { Config_ = InConfig; }

	// Run the viewport (blocks until window closed)
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

	OaViewerConfig Config_;

	// 2D image/video state
	OaTexture     Image_;
	OaImagePlanes Planes_;
	ImageViewMode ImageMode_ = ImageViewMode::RGB;

	// 3D scene state
	OaCamera InternalCamera_;  // Default camera if none provided

	OaNavigation Nav_;

	// Mode-specific rendering
	void RenderImage2D(OaUi& InOui);
	void RenderScene3D(OaUi& InOui);
};
