// OaViewer::Run — SDL3-backed windowed application loop.
// Creates OaPresenter + SDL3 window + Vulkan surface, then drives
// OaUi per-frame for the single windowed viewer lifecycle.

// Engine first: Device.h → OaVk.h → VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Ui/Viewer.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Input.h>
#include <Oa/Core/Validation.h>

#include "ViewerPlatform.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>


static OaUiEvent ConvertSdlEvent(const SDL_Event& InSdl, SDL_Window* InWindow) {
	OaUiEvent e;
	OaF32 pixelScaleX = 1.0F;
	OaF32 pixelScaleY = 1.0F;
	if (InWindow != nullptr) {
		int logicalWidth = 0;
		int logicalHeight = 0;
		int pixelWidth = 0;
		int pixelHeight = 0;
		if (SDL_GetWindowSize(InWindow, &logicalWidth, &logicalHeight)
			and SDL_GetWindowSizeInPixels(InWindow, &pixelWidth, &pixelHeight)
			and logicalWidth > 0 and logicalHeight > 0) {
			pixelScaleX = static_cast<OaF32>(pixelWidth)
				/ static_cast<OaF32>(logicalWidth);
			pixelScaleY = static_cast<OaF32>(pixelHeight)
				/ static_cast<OaF32>(logicalHeight);
		}
	}
	const auto pixelX = [pixelScaleX](OaF32 InX) { return InX * pixelScaleX; };
	const auto pixelY = [pixelScaleY](OaF32 InY) { return InY * pixelScaleY; };
	const SDL_Keymod mods = SDL_GetModState();
	if ((mods & SDL_KMOD_SHIFT) != 0) { e.Modifiers |= OUI_MOD_SHIFT; }
	if ((mods & SDL_KMOD_CTRL)  != 0) { e.Modifiers |= OUI_MOD_CTRL; }
	if ((mods & SDL_KMOD_ALT)   != 0) { e.Modifiers |= OUI_MOD_ALT; }
	if ((mods & SDL_KMOD_GUI)   != 0) { e.Modifiers |= OUI_MOD_SUPER; }
	switch (InSdl.type) {
		case SDL_EVENT_MOUSE_MOTION:
			e.Type   = OuiEventType::MouseMove;
			e.MouseX = pixelX(InSdl.motion.x);
			e.MouseY = pixelY(InSdl.motion.y);
			e.MouseDX = pixelX(InSdl.motion.xrel);
			e.MouseDY = pixelY(InSdl.motion.yrel);
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
			e.Type   = OuiEventType::MouseDown;
			e.MouseX = pixelX(InSdl.button.x); e.MouseY = pixelY(InSdl.button.y);
			e.Button = static_cast<OaI32>(InSdl.button.button);
			break;
		case SDL_EVENT_MOUSE_BUTTON_UP:
			e.Type   = OuiEventType::MouseUp;
			e.MouseX = pixelX(InSdl.button.x); e.MouseY = pixelY(InSdl.button.y);
			e.Button = static_cast<OaI32>(InSdl.button.button);
			break;
		case SDL_EVENT_MOUSE_WHEEL:
			e.Type    = OuiEventType::MouseScroll;
			e.ScrollX = InSdl.wheel.x; e.ScrollY = InSdl.wheel.y;
			e.MouseX  = pixelX(InSdl.wheel.mouse_x); e.MouseY = pixelY(InSdl.wheel.mouse_y);
			e.IntegerScrollX = InSdl.wheel.integer_x;
			e.IntegerScrollY = InSdl.wheel.integer_y;
			e.TimestampNs    = InSdl.wheel.timestamp;
			e.ScrollGesture  = OaInput::ClassifyScroll(e);
			break;
		case SDL_EVENT_PINCH_BEGIN:
			e.Type = OuiEventType::Pinch;
			e.PinchPhase = OuiPinchPhase::Begin;
			e.GestureScale = 1.0F;
			break;
		case SDL_EVENT_PINCH_UPDATE:
			e.Type = OuiEventType::Pinch;
			e.PinchPhase = OuiPinchPhase::Update;
			e.GestureScale = InSdl.pinch.scale;
			break;
		case SDL_EVENT_PINCH_END:
			e.Type = OuiEventType::Pinch;
			e.PinchPhase = OuiPinchPhase::End;
			e.GestureScale = 1.0F;
			break;
		case SDL_EVENT_KEY_DOWN:
			e.Type = OuiEventType::KeyDown;
			e.Key  = static_cast<OuiKey>(InSdl.key.scancode);
			e.KeyRepeat = InSdl.key.repeat;
			break;
		case SDL_EVENT_KEY_UP:
			e.Type = OuiEventType::KeyUp;
			e.Key  = static_cast<OuiKey>(InSdl.key.scancode);
			e.KeyRepeat = InSdl.key.repeat;
			break;
		case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
		case SDL_EVENT_WINDOW_RESIZED: {
			e.Type = OuiEventType::WindowResize;
			int wPx = InSdl.window.data1;
			int hPx = InSdl.window.data2;
			if (SDL_Window* win = SDL_GetWindowFromID(InSdl.window.windowID)) {
				(void)SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
			}
			e.WindowW = wPx;
			e.WindowH = hPx;
			break;
		}
		default:
			break;
	}
	return e;
}


void OaViewer::ResizeWindow(OaU32 InWidth, OaU32 InHeight) noexcept {
	if (not Window_ or InWidth == 0 or InHeight == 0) return;
	SDL_Window* win = static_cast<SDL_Window*>(Window_);
	SDL_SetWindowSize(win, static_cast<int>(InWidth), static_cast<int>(InHeight));
	SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	(void)Resize(static_cast<OaU32>(wPx), static_cast<OaU32>(hPx));
}

void OaViewer::CapturePointer(bool InEnabled) noexcept {
	if (not Window_) { return; }
	// Capture only — NOT mouse-grab (grab confines cursor inside the window).
	(void)SDL_CaptureMouse(InEnabled);
}

void OaViewer::CaptureRelativeMouse(bool InEnabled) noexcept {
	if (not Window_) { return; }
	SDL_Window* win = static_cast<SDL_Window*>(Window_);
	(void)SDL_SetWindowRelativeMouseMode(win, InEnabled);
	(void)SDL_CaptureMouse(InEnabled);
}

OaStatus OaViewer::Run() {
	return RunApplication(nullptr);
}

OaStatus OaViewer::Run(OaEngine& InEngine) {
	return RunApplication(&InEngine);
}

OaStatus OaViewer::RunApplication(OaEngine* InBorrowedEngine) {
	if (Running_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewer::Run cannot enter an already-running viewer");
	}

	// Deterministic, graceful smoke-test boundary for windowed tutorials. Zero
	// or an invalid value preserves the normal interactive run-until-closed loop.
	OaU64 maxFrames = 0;
	if (const char* value = std::getenv("OA_UI_MAX_FRAMES"); value and *value) {
		char* end = nullptr;
		const unsigned long long parsed = std::strtoull(value, &end, 10);
		if (end != value and *end == '\0') {
			maxFrames = static_cast<OaU64>(parsed);
		}
	}
	OaU64 renderedFrames = 0;

	OaEngineConfig engineCfg;
	OaViewerPlatformLease platform;
	const OaStatus platformStatus = platform.Acquire(
		InBorrowedEngine == nullptr ? &engineCfg : nullptr);
	if (not platformStatus.IsOk()) return platformStatus;

	// Keep windowed applications inside the same validation contract as tests
	// and headless workloads. Tools/Diagnostics/run_validation.py sets this
	// process-local flag and verifies that OA actually enabled the requested
	// validation profile.
	if (InBorrowedEngine == nullptr
		and OaEnvFlag::IsSet("OA_VK_VALIDATION")) {
		engineCfg.EnableValidation = true;
	}

	// Respect OA_DEVICE env var (same semantics as gtest harness).
	if (InBorrowedEngine == nullptr) {
		if (const char* dev = std::getenv("OA_DEVICE"); dev and *dev) {
			if (std::strcmp(dev, "integrated") == 0
				or std::strcmp(dev, "igpu") == 0) {
				engineCfg.DevicePref = OaDevicePreference::Integrated;
			} else if (std::strcmp(dev, "discrete") == 0
				or std::strcmp(dev, "dgpu") == 0) {
				engineCfg.DevicePref = OaDevicePreference::Discrete;
			} else if (std::strcmp(dev, "cpu") == 0) {
				engineCfg.DevicePref = OaDevicePreference::Cpu;
			} else {
				char* end = nullptr;
				unsigned long idx = std::strtoul(dev, &end, 10);
				if (end != dev and *end == '\0' and idx <= 0xFFFFu) {
					engineCfg.DevicePref = OaDevicePreference::ByIndex;
					engineCfg.DeviceIndex = static_cast<OaU32>(idx);
				}
			}
		}
	}

	// Phase A — one engine with a graphics-capable queue, no surface yet.
	OaUniquePtr<OaEngine> ownedEngine;
	OaEngine* engine = InBorrowedEngine;
	if (engine == nullptr) {
		auto engineResult = OaEngine::Create(engineCfg);
		if (not engineResult.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::App, "Engine create failed: %s",
				engineResult.GetStatus().GetMessage().c_str());
			return engineResult.GetStatus();
		}
		ownedEngine = OaStdMove(*engineResult);
		engine = ownedEngine.get();
	}
	if (not engine->IsReady() or not engine->HasGraphics()) {
		if (ownedEngine != nullptr) (void)engine->Close();
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaViewer::Run requires a ready presentation-capable OaEngine");
	}
	OaEngine& rt = *engine;
	const auto closeOwnedEngine = [&]() -> OaStatus {
		return ownedEngine != nullptr ? rt.Close() : OaStatus::Ok();
	};

	OaInput::Initialize();
	OaPresenter presenter(rt);

	SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
	SDL_Window* win = SDL_CreateWindow(
		Config_.Title.c_str(),
		static_cast<int>(Config_.Width),
		static_cast<int>(Config_.Height),
		flags);
	if (win == nullptr) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_CreateWindow failed: %s", SDL_GetError());
		OaInput::Shutdown();
		(void)closeOwnedEngine();
		return OaStatus::Error("SDL_CreateWindow failed");
	}

	// Phase B — surface against the live VkInstance.
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (not SDL_Vulkan_CreateSurface(win, static_cast<VkInstance>(rt.Device.Instance), nullptr, &surface)) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
		SDL_DestroyWindow(win);
		OaInput::Shutdown();
		(void)closeOwnedEngine();
		return OaStatus::Error("Vulkan surface creation failed");
	}

	// Use actual pixel size for HiDPI displays.
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	Config_.Width  = static_cast<OaU32>(wPx);
	Config_.Height = static_cast<OaU32>(hPx);

	Window_ = static_cast<void*>(win);
	if (auto s = OpenSource(rt); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "Device-ready initialization failed: %s",
			s.GetMessage().c_str());
		const OaStatus closeStatus = CloseSource(rt);
		if (not closeStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::App,
				"Source cleanup after open failure failed: %s",
				closeStatus.ToString().c_str());
		}
		vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
		SDL_DestroyWindow(win);
		Window_ = nullptr;
		OaInput::Shutdown();
		(void)closeOwnedEngine();
		return s;
	}

	// Phase C — the presenter owns WSI; the viewer owns its UI render target.
	if (auto s = InitPresentation(presenter, static_cast<void*>(surface)); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaViewer initialization failed: %s", s.GetMessage().c_str());
		(void)CloseSource(rt);
		vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
		SDL_DestroyWindow(win);
		Window_ = nullptr;
		OaInput::Shutdown();
		(void)closeOwnedEngine();
		return s;
	}

	Running_ = true;
	if (auto s = InitView(); not s.IsOk()) {
		Running_ = false;
		(void)CloseSource(rt);
		const OaStatus presentationStatus = DestroyPresentation();
		if (not presentationStatus.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::App,
				"Presentation cleanup after view initialization failed: %s",
				presentationStatus.ToString().c_str());
		}
		vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
		SDL_DestroyWindow(win);
		Window_ = nullptr;
		OaInput::Shutdown();
		(void)closeOwnedEngine();
		return s;
	}
	OaVec<OaUiEvent> events;
	using Clock = std::chrono::steady_clock;
	auto tPrev = Clock::now();
	OaStatus runStatus = OaStatus::Ok();

	while (Running_) {
		events.Clear();
		SDL_Event sdlEvent;
		while (SDL_PollEvent(&sdlEvent)) {
			OaInput::ProcessEvent(&sdlEvent);
			if (sdlEvent.type == SDL_EVENT_QUIT
			 or sdlEvent.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
				Running_ = false;
			} else if (sdlEvent.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
				if (auto s = Resize(
					static_cast<OaU32>(sdlEvent.window.data1),
					static_cast<OaU32>(sdlEvent.window.data2)); not s.IsOk()) {
					OA_LOG_ERROR(OaLogComponent::App,
						"Presenter resize failed: %s", s.GetMessage().c_str());
					runStatus = s;
					Running_ = false;
				}
			}
			OaUiEvent e = ConvertSdlEvent(sdlEvent, win);
			RouteEvent(e);
			events.PushBack(e);
		}
		OaInput::Update();
		if (runStatus.IsError()) break;

		auto tNow = Clock::now();
		OaF32 deltaMs = static_cast<OaF32>(
			std::chrono::duration<double, std::milli>(tNow - tPrev).count());
		tPrev = tNow;

		Update(deltaMs);
		BeginFrame(deltaMs);
		RouteUiEvents(OaSpan<const OaUiEvent>(events.Data(), events.Size()));
		Render(Ui_);

		OaStatus frameStatus = OaStatus::Ok();
		OaVkTimelineWait renderDependency;
		if (RenderDependency_.IsValid()) {
			if (not rt.OwnsEvent(RenderDependency_)) {
				frameStatus = OaStatus::InvalidArgument(
					"OaViewer source readiness event belongs to another engine");
			} else {
				renderDependency = RenderDependency_.TimelineWait();
			}
		}
		if (frameStatus.IsOk()) frameStatus = presenter.BeginGraphicsBatch();
		if (frameStatus.IsOk()) {
			OaVkStream* stream = presenter.ActiveGraphicsBatchStream();
			RecordRender(static_cast<VkCommandBuffer>(stream->CommandBuffer));
			auto flush = presenter.FlushGraphicsBatch(
				renderDependency.Semaphore,
				renderDependency.Value);
			if (not flush.IsOk()) {
				frameStatus = flush.GetStatus();
			} else {
				const OaEvent completion = *flush;
				SetRenderCompletion(completion);
				frameStatus = MarkRenderSubmitted(completion);
			}
		}

		// OUT_OF_DATE handled internally; resize already applied above.
		if (frameStatus.IsOk()) frameStatus = Present();
		if (frameStatus.IsError()) {
			OA_LOG_ERROR(OaLogComponent::App,
				"OaViewer frame failed: %s", frameStatus.ToString().c_str());
			runStatus = frameStatus;
			Running_ = false;
		}

		EndFrame();
		OaInput::ClearForNextFrame();
		if (maxFrames != 0 and ++renderedFrames >= maxFrames) {
			Running_ = false;
		}
	}

	// The last frame no longer host-waits before present. Drain its compute
	// sampling work before user resources and the compose image are destroyed.
	if (const OaStatus syncStatus = presenter.SyncGraphicsBatch(); not syncStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"SyncGraphicsBatch failed during UI shutdown: %s",
			syncStatus.GetMessage().c_str());
		if (runStatus.IsOk()) runStatus = syncStatus;
	}
	// Presentation is submitted after the compose batch and is not covered by
	// the graphics-batch timeline. Drain WSI before application-owned images,
	// buffers, and the compose target are released below.
	if (const OaStatus presentStatus = presenter.WaitPresentationIdle();
		not presentStatus.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"presentation drain failed during UI shutdown: %s",
			presentStatus.ToString().c_str());
		if (runStatus.IsOk()) runStatus = presentStatus;
	}
	const OaStatus closeStatus = CloseSource(rt);
	const OaStatus presentationStatus = DestroyPresentation();
	Window_ = nullptr;
	vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
	SDL_DestroyWindow(win);
	const OaStatus engineStatus = closeOwnedEngine();
	OaInput::Shutdown();
	if (not runStatus.IsOk()) return runStatus;
	if (not closeStatus.IsOk()) return closeStatus;
	if (not presentationStatus.IsOk()) return presentationStatus;
	return engineStatus;
}
