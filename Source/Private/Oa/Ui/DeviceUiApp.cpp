// OaDeviceUiApp::Run — SDL3-backed windowed app loop.
// Creates OaGraphicsEngine + SDL3 window + Vulkan surface, then drives
// OaUi per-frame.  Subclass and override OnInit/OnRender/OnShutdown.

// Engine first: Device.h → OaVk.h → VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Core/Input.h>
#include <Oa/Core/Validation.h>

#include <chrono>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>


static OaUiEvent ConvertSdlEvent(const SDL_Event& InSdl) {
	OaUiEvent e;
	const SDL_Keymod mods = SDL_GetModState();
	if ((mods & SDL_KMOD_SHIFT) != 0) { e.Modifiers |= OUI_MOD_SHIFT; }
	if ((mods & SDL_KMOD_CTRL)  != 0) { e.Modifiers |= OUI_MOD_CTRL; }
	if ((mods & SDL_KMOD_ALT)   != 0) { e.Modifiers |= OUI_MOD_ALT; }
	if ((mods & SDL_KMOD_GUI)   != 0) { e.Modifiers |= OUI_MOD_SUPER; }
	switch (InSdl.type) {
		case SDL_EVENT_MOUSE_MOTION:
			e.Type   = OuiEventType::MouseMove;
			e.MouseX = InSdl.motion.x;
			e.MouseY = InSdl.motion.y;
			e.MouseDX = InSdl.motion.xrel;
			e.MouseDY = InSdl.motion.yrel;
			break;
		case SDL_EVENT_MOUSE_BUTTON_DOWN:
			e.Type   = OuiEventType::MouseDown;
			e.MouseX = InSdl.button.x; e.MouseY = InSdl.button.y;
			e.Button = static_cast<OaI32>(InSdl.button.button);
			break;
		case SDL_EVENT_MOUSE_BUTTON_UP:
			e.Type   = OuiEventType::MouseUp;
			e.MouseX = InSdl.button.x; e.MouseY = InSdl.button.y;
			e.Button = static_cast<OaI32>(InSdl.button.button);
			break;
		case SDL_EVENT_MOUSE_WHEEL:
			e.Type    = OuiEventType::MouseScroll;
			e.ScrollX = InSdl.wheel.x; e.ScrollY = InSdl.wheel.y;
			e.MouseX  = InSdl.wheel.mouse_x; e.MouseY = InSdl.wheel.mouse_y;
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
			break;
		case SDL_EVENT_KEY_UP:
			e.Type = OuiEventType::KeyUp;
			e.Key  = static_cast<OuiKey>(InSdl.key.scancode);
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


void OaDeviceUiApp::ResizeWindow(OaU32 InWidth, OaU32 InHeight) noexcept {
	if (!Window_ || InWidth == 0 || InHeight == 0) return;
	SDL_Window* win = static_cast<SDL_Window*>(Window_);
	SDL_SetWindowSize(win, static_cast<int>(InWidth), static_cast<int>(InHeight));
	SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	(void)Gpui_.Resize(static_cast<OaU32>(wPx), static_cast<OaU32>(hPx));
}

void OaDeviceUiApp::CapturePointer(bool InEnabled) noexcept {
	if (!Window_) { return; }
	// Capture only — NOT mouse-grab (grab confines cursor inside the window).
	(void)SDL_CaptureMouse(InEnabled);
}

void OaDeviceUiApp::CaptureRelativeMouse(bool InEnabled) noexcept {
	if (!Window_) { return; }
	SDL_Window* win = static_cast<SDL_Window*>(Window_);
	(void)SDL_SetWindowRelativeMouseMode(win, InEnabled);
	(void)SDL_CaptureMouse(InEnabled);
}

void OaDeviceUiApp::CaptureMouse(bool InEnabled) noexcept {
	CapturePointer(InEnabled);
}

OaStatus OaDeviceUiApp::Run(const OaUiConfig& InConfig) {
	// Deterministic, graceful smoke-test boundary for windowed tutorials. Zero
	// or an invalid value preserves the normal interactive run-until-closed loop.
	OaU64 maxFrames = 0;
	if (const char* value = std::getenv("OA_UI_MAX_FRAMES"); value && *value) {
		char* end = nullptr;
		const unsigned long long parsed = std::strtoull(value, &end, 10);
		if (end != value && *end == '\0') {
			maxFrames = static_cast<OaU64>(parsed);
		}
	}
	OaU64 renderedFrames = 0;

	// SDL uses this identity for Wayland app IDs and desktop integration.
	// Leaving it unset makes compositor extensions treat the window as an
	// anonymous client, which is particularly fragile on GNOME/Wayland.
	if (!SDL_SetAppMetadata("OA", nullptr, "com.empyrealm.oa")) {
		OA_LOG_WARN(OaLogComponent::App, "SDL_SetAppMetadata failed: %s", SDL_GetError());
	}

	// Diagnostic/workaround override for compositor-specific WSI failures.
	// SDL must receive the video backend hint before SDL_Init. An explicit
	// OA_UI_BACKEND always wins over the session-type heuristic below.
	const char* backendOverride = std::getenv("OA_UI_BACKEND");
	if (backendOverride && *backendOverride) {
		if (!SDL_SetHint(SDL_HINT_VIDEO_DRIVER, backendOverride)) {
			OA_LOG_WARN(OaLogComponent::App,
				"SDL video backend override '%s' was rejected", backendOverride);
		}
	} else if (const char* session = std::getenv("XDG_SESSION_TYPE");
	           session && std::strcmp(session, "wayland") == 0) {
		// Pure-Wayland session: pin the Wayland backend. This only works when SDL
		// was built with Wayland + Vulkan support (vcpkg sdl3 "vulkan" feature);
		// otherwise SDL_Vulkan_LoadLibrary below fails loudly instead of crashing.
		SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
	}

	// Set a default locale if none is set (prevents xkbcommon compose file errors)
	if (!std::getenv("LC_ALL") && !std::getenv("LANG")) {
		#if !defined(_WIN32)
		::setenv("LC_ALL", "C.UTF-8", 0);  // Don't override if already set
		#endif
	}

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_Init failed: %s", SDL_GetError());
		return OaStatus::Error("SDL_Init failed");
	}
	OaInput::Initialize();

	// SDL3 requires the Vulkan loader to be live before SDL_Vulkan_*; otherwise
	// the backend's instance-extension hook is null and the query jumps to 0x0.
	// Previously this worked only because the auto-selected backend happened to
	// load the loader implicitly; forcing the Wayland backend exposed the gap.
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_Vulkan_LoadLibrary failed: %s", SDL_GetError());
		OaInput::Shutdown();
		SDL_Quit();
		return OaStatus::Error("SDL_Vulkan_LoadLibrary failed");
	}

	// Collect WSI instance extensions (VK_KHR_surface + platform ext).
	OaEngineConfig engineCfg;
	engineCfg.PresentationMode = OaPresentationMode::Swapchain;
	OaU32 extCount = 0;
	const char* const* extNames = SDL_Vulkan_GetInstanceExtensions(&extCount);
	for (OaU32 i = 0; i < extCount; ++i) {
		engineCfg.InstanceExtraExtensions.PushBack(extNames[i]);
	}

	// Respect OA_DEVICE env var (same semantics as gtest harness).
	if (const char* dev = std::getenv("OA_DEVICE"); dev && *dev) {
		if (std::strcmp(dev, "integrated") == 0 || std::strcmp(dev, "igpu") == 0) {
			engineCfg.DevicePref = OaDevicePreference::Integrated;
		} else if (std::strcmp(dev, "discrete") == 0 || std::strcmp(dev, "dgpu") == 0) {
			engineCfg.DevicePref = OaDevicePreference::Discrete;
		} else if (std::strcmp(dev, "cpu") == 0) {
			engineCfg.DevicePref = OaDevicePreference::Cpu;
		} else {
			char* end = nullptr;
			unsigned long idx = std::strtoul(dev, &end, 10);
			if (end != dev && *end == '\0' && idx <= 0xFFFFu) {
				engineCfg.DevicePref = OaDevicePreference::ByIndex;
				engineCfg.DeviceIndex = static_cast<OaU32>(idx);
			}
		}
	}

	// Phase A — device with graphics + VK_KHR_swapchain, no surface yet.
	auto rtResult = OaGraphicsEngine::Create(engineCfg);
	if (!rtResult) {
		OA_LOG_ERROR(OaLogComponent::App, "Engine create failed: %s",
			rtResult.GetStatus().GetMessage().c_str());
		OaInput::Shutdown();
		SDL_Quit();
		return OaStatus::Error("Engine create failed");
	}
	// rtResult (a function-local) owns the pinned engine for this scope; bind a
	// reference so all downstream `rt.` usage stays unchanged.
	OaGraphicsEngine& rt = *rtResult.GetValue();

	SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
	SDL_Window* win = SDL_CreateWindow(
		InConfig.Title.c_str(),
		static_cast<int>(InConfig.Width),
		static_cast<int>(InConfig.Height),
		flags);
	if (win == nullptr) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_CreateWindow failed: %s", SDL_GetError());
		rt.Destroy();
		OaInput::Shutdown();
		SDL_Quit();
		return OaStatus::Error("SDL_CreateWindow failed");
	}

	// Phase B — surface against the live VkInstance.
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(win, static_cast<VkInstance>(rt.Device.Instance), nullptr, &surface)) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
		SDL_DestroyWindow(win);
		rt.Destroy();
		OaInput::Shutdown();
		SDL_Quit();
		return OaStatus::Error("Vulkan surface creation failed");
	}

	// Use actual pixel size for HiDPI displays.
	int wPx = 0;
	int hPx = 0;
	SDL_GetWindowSizeInPixels(win, &wPx, &hPx);
	OaUiConfig cfg = InConfig;
	cfg.Width  = static_cast<OaU32>(wPx);
	cfg.Height = static_cast<OaU32>(hPx);

	Window_ = static_cast<void*>(win);
	if (auto s = OnDeviceReady(rt); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "Device-ready initialization failed: %s",
			s.GetMessage().c_str());
		vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
		SDL_DestroyWindow(win);
		Window_ = nullptr;
		rt.Destroy();
		OaInput::Shutdown();
		SDL_Quit();
		return s;
	}

	// Phase C — swapchain + compose image + widget layer via OaUi.
	if (auto s = Gpui_.Init(rt, static_cast<void*>(surface), cfg); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaUi::Init failed: %s", s.GetMessage().c_str());
		vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
		SDL_DestroyWindow(win);
		rt.Destroy();
		OaInput::Shutdown();
		SDL_Quit();
		return s;
	}

	Running_ = true;
	OnInit(Gpui_);
	OaVec<OaUiEvent> events;
	using Clock = std::chrono::steady_clock;
	auto tPrev = Clock::now();

	while (Running_) {
		events.Clear();
		SDL_Event sdlEvent;
		while (SDL_PollEvent(&sdlEvent)) {
			OaInput::ProcessEvent(&sdlEvent);
			if (sdlEvent.type == SDL_EVENT_QUIT
			 or sdlEvent.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
				Running_ = false;
			} else if (sdlEvent.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
				if (auto s = Gpui_.Resize(
					static_cast<OaU32>(sdlEvent.window.data1),
					static_cast<OaU32>(sdlEvent.window.data2)); !s.IsOk()) {
					OA_LOG_ERROR(OaLogComponent::App,
						"Presenter resize failed: %s", s.GetMessage().c_str());
				}
			}
			OaUiEvent e = ConvertSdlEvent(sdlEvent);
			OnEvent(e);
			events.PushBack(e);
		}
		OaInput::Update();

		auto tNow = Clock::now();
		OaF32 deltaMs = static_cast<OaF32>(
			std::chrono::duration<double, std::milli>(tNow - tPrev).count());
		tPrev = tNow;

		OnUpdate(deltaMs);
		Gpui_.BeginFrame(deltaMs);
		Gpui_.RouteEvents(OaSpan<const OaUiEvent>(events.Data(), events.Size()));
		OnRender(Gpui_.WidgetLayer());

		if (auto s = rt.BeginGraphicsBatch(); s.IsOk()) {
			OaVkStream* stream = rt.ActiveGraphicsBatchStream();
			Gpui_.RecordRender(static_cast<VkCommandBuffer>(stream->CommandBuffer));
			if (auto fs = rt.FlushGraphicsBatch(
				Gpui_.RenderDependencySemaphore(),
				Gpui_.RenderDependencyValue()); !fs.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::App,
					"FlushGraphicsBatch failed: %s", fs.GetMessage().c_str());
			} else {
				Gpui_.SetRenderCompletion(stream->TimelineSem, stream->TimelineValue);
				OnRenderSubmitted(stream->TimelineSem, stream->TimelineValue);
			}
		}

		// OUT_OF_DATE handled internally; resize already applied above.
		(void)Gpui_.Present();

		Gpui_.EndFrame();
		OaInput::ClearForNextFrame();
		if (maxFrames != 0 && ++renderedFrames >= maxFrames) {
			Running_ = false;
		}
	}

	// The last frame no longer host-waits before present. Drain its compute
	// sampling work before user resources and the compose image are destroyed.
	(void)rt.SyncGraphicsBatch();
	OnShutdown(Gpui_);
	Gpui_.Destroy();
	Window_ = nullptr;
	vkDestroySurfaceKHR(static_cast<VkInstance>(rt.Device.Instance), surface, nullptr);
	SDL_DestroyWindow(win);
	rt.Destroy();
	OaInput::Shutdown();
	SDL_Quit();
	return OaStatus::Ok();
}
