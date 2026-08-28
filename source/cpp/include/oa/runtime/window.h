// oa::Window — portable window + event pump (no SDL/GLFW in oa).
// oa::VulkanWindow — adds presenter VkInstance extensions + surface creation (WSI).
// Concrete backends (SDL3, etc.) live in consumer repos.

#pragma once

#include <oa/core/types.h>

#include <vulkan/vulkan_core.h>


namespace oa {

class Presenter;

class WindowConfig {
public:
	oa::String title  = "Oa";
	int widthPx     = 1280;
	int heightPx    = 720;
	bool resizable  = true;
	// Backends that need a vulkan-capable window set this true.
	bool vulkanSurface = true;
	// high-DPI: request native pixel density from the compositor (SDL3: HIGH_PIXEL_DENSITY).
	// When true, SDL reports logical vs pixel sizes separately so ImGui, vulkan swapchain,
	// and mouse coordinates all stay in sync on scaled displays (e.g. GNOME 150-200%).
	bool highDpi = true;
	// Linux: set SDL_HINT_VIDEO_DRIVER=wayland before init; default leaves driver to session.
	bool preferWaylandVideoDriver = false;
};


// Callback invoked when the window's pixel size changes (resize / DPI change).
using WindowResizeFn = void(*)(int inWidthPx, int inHeightPx, void* inUserData);

class Window {
public:
	virtual ~Window() = default;

	virtual bool pumpEvents(bool& outShouldQuit) = 0;

	[[nodiscard]] virtual int drawableWidthPx()  const = 0;
	[[nodiscard]] virtual int drawableHeightPx() const = 0;

	void setResizeCallback(WindowResizeFn inFn, void* inUserData) {
		resizeFn_       = inFn;
		resizeUserData_ = inUserData;
	}

protected:
	Window() = default;

	WindowResizeFn resizeFn_       = nullptr;
	void*            resizeUserData_ = nullptr;
};


class VulkanWindow : public Window {
public:
	friend class Presenter;

	// Destructor.
	~VulkanWindow() override = default;

	// Presenter path: returns the VK_KHR_surface + platform extension names
	// required for VkInstance creation (e.g. from SDL_Vulkan_GetInstanceExtensions).
	[[nodiscard]] virtual bool getPresenterInstanceExtensions(
		oa::Vector<const char*>* outExtensions) const;

	// Returns the platform-native window handle (e.g. SDL_Window*).
	// Used by ImGui backends (ImGui_ImplSDL3_InitForVulkan).
	// Returns nullptr in the base class; override in concrete backends.
	[[nodiscard]] virtual void* getNativeWindowHandle() const { return nullptr; }

protected:

	// Constructor.
	VulkanWindow() = default;

	// Runtime-only WSI bridge. Concrete backends override this, while callers
	// ask oa::Presenter to create an opaque surface without receiving VkInstance.
	// SDL3: SDL_Vulkan_CreateSurface(win, instance, nullptr, out_surface).
	[[nodiscard]] virtual bool createPresenterVkSurface(
		VkInstance instance,
		VkSurfaceKHR* outSurface) const;
};

} // namespace oa
