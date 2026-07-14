// OaWindow — portable window + event pump (no SDL/GLFW in oa).
// OaVkWindow — adds presenter VkInstance extensions + surface creation (WSI).
// Concrete backends (SDL3, etc.) live in consumer repos.

#pragma once

#include <Oa/Core/Types.h>

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>


class OaWindowConfig {
public:
	OaString Title  = "Oa";
	int WidthPx     = 1280;
	int HeightPx    = 720;
	bool Resizable  = true;
	// Backends that need a Vulkan-capable window set this true.
	bool VulkanSurface = true;
	// High-DPI: request native pixel density from the compositor (SDL3: HIGH_PIXEL_DENSITY).
	// When true, SDL reports logical vs pixel sizes separately so ImGui, Vulkan swapchain,
	// and mouse coordinates all stay in sync on scaled displays (e.g. GNOME 150-200%).
	bool HighDpi = true;
	// Linux: set SDL_HINT_VIDEO_DRIVER=wayland before init; default leaves driver to session.
	bool PreferWaylandVideoDriver = false;
};


// Callback invoked when the window's pixel size changes (resize / DPI change).
using OaWindowResizeFn = void(*)(int InWidthPx, int InHeightPx, void* InUserData);

class OaWindow {
public:
	virtual ~OaWindow() = default;

	virtual bool PumpEvents(bool& out_should_quit) = 0;

	[[nodiscard]] virtual int DrawableWidthPx()  const = 0;
	[[nodiscard]] virtual int DrawableHeightPx() const = 0;

	void SetResizeCallback(OaWindowResizeFn InFn, void* InUserData) {
		ResizeFn_       = InFn;
		ResizeUserData_ = InUserData;
	}

protected:
	OaWindow() = default;

	OaWindowResizeFn ResizeFn_       = nullptr;
	void*            ResizeUserData_ = nullptr;
};


class OaVkWindow : public OaWindow {
public:

	// Destructor.
	~OaVkWindow() override = default;

	// Presenter path: returns the VK_KHR_surface + platform extension names
	// required for VkInstance creation (e.g. from SDL_Vulkan_GetInstanceExtensions).
	[[nodiscard]] virtual bool GetPresenterInstanceExtensions(std::vector<const char*>* out_extensions) const;

	// Creates a VkSurfaceKHR against the provided instance.
	// SDL3: SDL_Vulkan_CreateSurface(win, instance, nullptr, out_surface).
	[[nodiscard]] virtual bool CreatePresenterVkSurface(VkInstance instance, VkSurfaceKHR* out_surface) const;

	// Returns the platform-native window handle (e.g. SDL_Window*).
	// Used by ImGui backends (ImGui_ImplSDL3_InitForVulkan).
	// Returns nullptr in the base class; override in concrete backends.
	[[nodiscard]] virtual void* GetNativeWindowHandle() const { return nullptr; }

protected:

	// Constructor.
	OaVkWindow() = default;
};
