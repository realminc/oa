// OaSwapchain — WSI swapchain state for the Present render sink.
//
// See UnifiedExecutionArchitecture.md §3.5: one render sink-resource among peers. The renderer
// produces an OaTexture; the chosen sink decides what happens next.
//   Present     — this file (Vulkan WSI swapchain → vkQueuePresentKHR)
//   SaveImage   — filesystem
//   EncodeFrame — video encoder
//
// Lifecycle is owned by OaGraphicsEngine. Engine-internal Build/Destroy
// methods write into the held OaSwapchain; external callers read via
// OaGraphicsEngine::Swapchain(). Headless-mode engines never instantiate
// one.
//
// State sequence:
//   default construction       → zero-state (Handle == VK_NULL_HANDLE)
//   engine.InitPresentation()  → swapchain objects + per-frame sync built;
//                                PresentReady = true
//   engine.DetachPresentation()→ torn back down; zero-state again
//   engine destructor          → torn down if still attached
//
// Why a separate type: the Step 3a extraction (UnifiedExecutionArchitecture.md §13.3) untangles
// presentation state from the engine's core compute/graphics queues and
// caches. After this, the engine can host zero, one, or N swapchains
// (multi-window) the same way it hosts a graphics queue. Step 3b adds the
// OaContext record APIs (RecordAcquire / RecordPresent) that consume an
// OaSwapchain&.

#pragma once

#include <Oa/Core/Types.h>

#include <vector>
#include <vulkan/vulkan.h>


struct OaSwapchain {
	// Frames in flight: how many independent acquire/render/present cycles
	// can overlap. Two is the standard tradeoff for desktop apps (one being
	// presented, one being recorded). Per-frame sync arrays are sized by this.
	static constexpr int kFramesInFlight = 2;

	// ─── WSI handles ─────────────────────────────────────────────────────────
	// Surface is OWNED BY THE CALLER (typically OaVkWindow / SDL3). The engine
	// references it during the lifetime of the swapchain and clears the pointer
	// on DetachPresentation; it never destroys the surface.
	void*                     Surface       = nullptr;
	VkSwapchainKHR            Handle        = VK_NULL_HANDLE;
	VkFormat                  Format        = VK_FORMAT_UNDEFINED;
	VkExtent2D                Extent        = {};
	std::vector<VkImage>      Images;
	std::vector<VkImageView>  Views;
	bool                      PresentReady  = false;

	// ─── Per-frame-in-flight sync ────────────────────────────────────────────
	// Sized kFramesInFlight after BuildSyncObjects. FrameIndex_ cycles
	// 0..kFramesInFlight-1 on every successful present.
	int                       FrameIndex    = 0;
	std::vector<VkSemaphore>  ImageAvailSem;
	std::vector<VkSemaphore>  RenderDoneSem;
	std::vector<VkFence>      InFlightFence;

	// ─── Pending resize signal ───────────────────────────────────────────────
	// Set from the SDL window event handler (main thread). Consumed at the top
	// of the next DrawFrame, which calls RecreateSwapchain(DirtySize). This
	// avoids racing the render thread when the OS reports a pixel-size change.
	bool                      Dirty         = false;
	VkExtent2D                DirtySize     = {};

	// When true (default), BuildSwapchainObjects picks VK_PRESENT_MODE_FIFO_KHR
	// (vsync — one frame per refresh). When false, prefer MAILBOX, falling back
	// to IMMEDIATE, then FIFO. Set via OaUiConfig.Vsync before Init/Recreate.
	bool                      Vsync         = true;

	// ─── Convenience accessors ───────────────────────────────────────────────
	[[nodiscard]] OaU32 Width()      const noexcept { return Extent.width;  }
	[[nodiscard]] OaU32 Height()     const noexcept { return Extent.height; }
	[[nodiscard]] OaU32 ImageCount() const noexcept {
		return static_cast<OaU32>(Images.size());
	}
	[[nodiscard]] bool  IsValid()    const noexcept {
		return Handle != VK_NULL_HANDLE and PresentReady;
	}
};
