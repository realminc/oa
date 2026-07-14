// OaDeviceUi — Main entry point.
//
// OaDeviceUi owns:
//   OaUiComposeImage — off-screen RGBA8 storage image (widgets write here)
//   OaTextAtlas      — MSDF font atlas
//   OaUi             — widget API
//   OaNodeCanvas     — pan/zoom canvas
//   OaNodeGraph      — node editor
//   OaInputSystem    — key actions
//
// The swapchain itself lives on the caller-supplied OaGraphicsEngine
// (engine.Swapchain()). OaDeviceUi calls engine.InitPresentation(surface)
// at Init time, then uses ctx.RecordAcquire / RecordBlit / RecordPresent
// for the per-frame compose→swap path. See UnifiedExecutionArchitecture.md §3.5.
//
// OaDeviceUiApp: convenience app loop for standalone tools.  Subclass and
// override OnUpdate() + OnRender() to use.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Sync.h>
#include <Oa/Render/Renderer.h>
#include <Oa/Ui/Canvas.h>
#include <Oa/Ui/Style.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Ui/Text.h>
#include <Oa/Ui/Node.h>
#include <Oa/Ui/Input.h>

#include <vulkan/vulkan.h>

class OaComputeEngine;
class OaGraphicsEngine;


// ─── OaUiComposeImage ────────────────────────────────────────────────────────
// A device-local RGBA8 storage image. BindlessIndex is registered in
// OaComputeEngine::Bindless so widget shaders can address it by slot.
// Used as the off-screen compose target before the ctx-mediated blit to the
// acquired swapchain image (UnifiedExecutionArchitecture.md §3.5).

struct OaUiComposeImage {
	VkImage     Image          = VK_NULL_HANDLE;
	VkImageView View           = VK_NULL_HANDLE;
	void*       Allocation     = nullptr;   // VmaAllocation
	OaU32       BindlessIndex  = UINT32_MAX;
	OaU32       Width          = 0;
	OaU32       Height         = 0;

	[[nodiscard]] bool IsValid() const noexcept { return Image != VK_NULL_HANDLE; }
};


// ─── OaUiConfig ─────────────────────────────────────────────────────────────

struct OaUiConfig {
	OaString Title       = "Realm";
	OaU32    Width       = 1280;
	OaU32    Height      = 720;
	OaUiStyle Style;                        // defaults = RealmDark
	bool     EnableNodeGraph = false;      // allocate OaNodeGraph
	bool     Vsync           = true;
	OaFilter PresentFilter   = OaFilter::Nearest;  // Nearest = sharp, no smoothing
};


// ─── OaDeviceUi ───────────────────────────────────────────────────────────────
// Attach to an OaGraphicsEngine (shared device + swapchain). The engine
// must have been Created with PresentationMode::Swapchain; InitPresentation
// is called by Init() below.

class OaDeviceUi {
public:
	OaDeviceUi() = default;
	OaDeviceUi(const OaDeviceUi&)            = delete;
	OaDeviceUi& operator=(const OaDeviceUi&) = delete;
	OaDeviceUi(OaDeviceUi&&) noexcept;
	OaDeviceUi& operator=(OaDeviceUi&&) noexcept;
	~OaDeviceUi();

	// InSurface — caller-owned VkSurfaceKHR (void* to avoid header transitivity).
	// Init builds the engine's swapchain (engine.InitPresentation) and the
	// off-screen compose image. The surface lifetime is the caller's; Destroy
	// detaches presentation but does not destroy the surface.
	[[nodiscard]] OaStatus Init(
		OaGraphicsEngine& InEngine,
		void*               InSurface,
		const OaUiConfig& InConfig = {});

	void Destroy();

	// ── Per-frame ─────────────────────────────────────────────────────────────

	void BeginFrame(OaF32 InDeltaMs);
	void RouteEvents(OaSpan<const OaUiEvent> InEvents);
	// Record all widget + node graph commands. Outputs to the compose image.
	void RecordRender(VkCommandBuffer InCmd);
	// Acquire the swap image, blit compose→swap, present. Handles OUT_OF_DATE
	// internally via engine.RecreateSwapchain on RecordAcquire.
	[[nodiscard]] OaStatus Present();
	void SetRenderCompletion(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) noexcept {
		RenderCompletionSemaphore_ = InSemaphore;
		RenderCompletionValue_ = InValue;
	}
	// Declare GPU work that produced an image sampled by this frame. The app
	// loop forwards this timeline edge into the graphics-batch submission.
	void SetRenderDependency(
		const OaVkTimelineSemaphore& InSemaphore,
		OaU64 InValue) noexcept {
		if (InSemaphore.Semaphore != nullptr && InValue > RenderDependencyValue_) {
			RenderDependencySemaphore_ = &InSemaphore;
			RenderDependencyValue_ = InValue;
		}
	}
	[[nodiscard]] const OaVkTimelineSemaphore* RenderDependencySemaphore() const noexcept {
		return RenderDependencySemaphore_;
	}
	[[nodiscard]] OaU64 RenderDependencyValue() const noexcept {
		return RenderDependencyValue_;
	}
	void EndFrame();
	// Resize engine swapchain + compose image and refresh widget blit binding.
	// Call from window resize events.
	[[nodiscard]] OaStatus Resize(OaU32 InWidth, OaU32 InHeight);

	// ── Accessors ─────────────────────────────────────────────────────────────

	[[nodiscard]] OaUi&              WidgetLayer()  noexcept { return Oui_; }
	[[nodiscard]] OaNodeGraph&      NodeGraph()    noexcept { return NodeGraph_; }
	[[nodiscard]] OaNodeCanvas&     Canvas()       noexcept { return Canvas_; }
	[[nodiscard]] OaInputSystem&    Input()        noexcept { return Input_; }
	[[nodiscard]] OaTextAtlas&      TextAtlas()    noexcept { return TextAtlas_; }
	[[nodiscard]] OaCanvasRenderer& CanvasRenderer() noexcept { return CanvasRenderer_; }
	[[nodiscard]] const OaUiComposeImage& ComposeImage() const noexcept { return Compose_; }
	// Non-owning RGBA8 view of the off-screen compose target. Use
	// OaFnVideo::FromTexture with OnRenderSubmitted's timeline dependency when
	// recording it.
	[[nodiscard]] OaTexture ComposeTexture() const noexcept;

	[[nodiscard]] OaU32  Width()  const noexcept { return Compose_.IsValid() ? Compose_.Width  : Config_.Width; }
	[[nodiscard]] OaU32  Height() const noexcept { return Compose_.IsValid() ? Compose_.Height : Config_.Height; }

private:
	[[nodiscard]] OaStatus BuildComposeImage(OaU32 InWidth, OaU32 InHeight);
	void                   DestroyComposeImage();

	OaGraphicsEngine* Gfx_     = nullptr;    // graphics engine (owns swapchain)
	OaComputeEngine*  Rt_      = nullptr;    // same pointer, compute-typed
	void*               Surface_ = nullptr;    // VkSurfaceKHR — caller-owned
	OaUiConfig          Config_;
	OaUiComposeImage    Compose_;
	OaTextAtlas         TextAtlas_;
	OaCanvasRenderer    CanvasRenderer_;
	OaUi                Oui_;
	OaNodeGraph         NodeGraph_;
	OaNodeCanvas        Canvas_;
	OaInputSystem       Input_;
	OaVkTimelineSemaphore RenderCompletionSemaphore_;
	OaU64 RenderCompletionValue_ = 0;
	const OaVkTimelineSemaphore* RenderDependencySemaphore_ = nullptr;
	OaU64 RenderDependencyValue_ = 0;
};


// ─── OaDeviceUiApp ────────────────────────────────────────────────────────────────
// Self-contained SDL3 app loop. Creates the engine and window internally.
//
// Typical use:
//   class MyApp : public OaDeviceUiApp {
//       void OnInit(OaDeviceUi& InGpui) override { /* load assets, register keys */ }
//       void OnRender(OaUi& InOui) override { /* declare widgets */ }
//       void OnShutdown(OaDeviceUi& InGpui) override { /* free GPU resources */ }
//   };
//   MyApp app;
//   app.Run({.Title = "My App"});

class OaDeviceUiApp {
public:
	OaDeviceUiApp() = default;
	virtual ~OaDeviceUiApp() = default;
	OaDeviceUiApp(const OaDeviceUiApp&)            = delete;
	OaDeviceUiApp& operator=(const OaDeviceUiApp&) = delete;

	// Blocks until the window is closed or Quit() is called.
	[[nodiscard]] OaStatus Run(const OaUiConfig& InConfig = {});

	// Override in subclass:
	// Called after the graphics-capable device, native window, and Vulkan
	// surface exist, but before the surface is attached to the engine and the
	// presentation resources are created. Use this only for device resources
	// whose driver initialization must precede swapchain setup (for example
	// Vulkan Video sessions).
	virtual OaStatus OnDeviceReady(OaGraphicsEngine&) { return OaStatus::Ok(); }
	// Called once after the engine + OaDeviceUi are ready, before the first frame.
	// Load GPU resources (images, models) and register key actions here.
	virtual void OnInit(OaDeviceUi& InGpui) {}
	virtual void OnUpdate(OaF32 InDeltaMs) {}
	virtual void OnRender(OaUi& InOui)      {}
	virtual void OnEvent(const OaUiEvent& InEvent) {}
	// Called after the widget compute batch is submitted. Resources sampled
	// by OnRender remain in use until this timeline value completes.
	virtual void OnRenderSubmitted(
		const OaVkTimelineSemaphore&,
		OaU64) {}
	// Called once after the loop exits, before OaDeviceUi::Destroy.
	// Free GPU resources (OaTexture::Destroy, OaImagePlanes::Destroy, etc.) here.
	virtual void OnShutdown(OaDeviceUi& InGpui) {}

protected:
	OaDeviceUi& Gpui() noexcept { return Gpui_; }
	// Signal the run loop to stop at the end of the current frame.
	void Quit() noexcept { Running_ = false; }
	// Resize the SDL window and swapchain. Call from OnInit after loading assets.
	void ResizeWindow(OaU32 InWidth, OaU32 InHeight) noexcept;
	// Pointer capture only (pan drags that continue outside the window).
	void CapturePointer(bool InEnabled) noexcept;
	// Relative + captured (FPS camera look).
	void CaptureRelativeMouse(bool InEnabled) noexcept;
	// Alias for CapturePointer — image viewers should not use relative mode.
	void CaptureMouse(bool InEnabled) noexcept;

private:
	OaDeviceUi Gpui_;
	bool   Running_ = false;
	void*  Window_  = nullptr;   // SDL_Window*
};
