// oa::Swapchain — WSI swapchain state for the Present render sink.
//
// One render sink-resource among peers. The renderer produces an oa::Texture;
// the chosen sink decides what happens next.
//   Present     — this file (vulkan WSI swapchain → vkQueuePresentKHR)
//   SaveImage   — filesystem
//   encodeFrame — video encoder
//
// Lifecycle is owned by oa::Presenter. Engine-internal Build/destroy
// methods write into the held oa::Swapchain; external callers read via
// oa::Presenter::swapchain(). Headless-mode engines never instantiate
// one.
//
// State sequence:
//   default construction       → zero-state (handle == VK_NULL_HANDLE)
//   presenter.initPresentation()  → swapchain objects + per-frame sync built;
//                                presentReady = true
//   presenter.detachPresentation()→ torn back down; zero-state again
//   presenter destructor          → unsubmitted work cancelled; submitted
//                                work and attached WSI state retired by engine
//
// This separate type untangles
// presentation state from the engine's core compute/graphics queues and
// caches. oa::Presenter owns the acquire/present protocol as a composed service.

#pragma once

#include <oa/core/types.h>
#include <oa/runtime/oaVk.h>


namespace oa {

struct Swapchain {
	// frames in flight: how many independent acquire/render/present cycles
	// can overlap. Two is the standard tradeoff for desktop apps (one being
	// presented, one being recorded). Per-frame sync arrays are sized by this.
	static constexpr int kFramesInFlight = 2;

	// ─── WSI handles ─────────────────────────────────────────────────────────
	// surface is normally OWNED BY THE CALLER (typically oa::VulkanWindow / SDL3). The
	// engine references it during the swapchain lifetime and clears the pointer
	// on explicit Close/detach. If an attached presenter is abandoned, ownership
	// transfers to engine retirement so the surface outlives the swapchain and is
	// destroyed before the VkInstance; the caller must not destroy it afterward.
	void*                     surface       = nullptr;
	VkSwapchainKHR            handle        = VK_NULL_HANDLE;
	VkFormat                  format        = VK_FORMAT_UNDEFINED;
	VkExtent2D                extent        = {};
	oa::Vec<VkImage>      images;
	oa::Vec<VkImageView>  views;
	bool                      presentReady  = false;

	// ─── Per-frame-in-flight sync ────────────────────────────────────────────
	// Sized kFramesInFlight after BuildSyncObjects. frameIndex_ cycles
	// 0..kFramesInFlight-1 on every successful present.
	int                       frameIndex    = 0;
	oa::Vec<VkSemaphore>  imageAvailSem;
	oa::Vec<VkSemaphore>  renderDoneSem;
	oa::Vec<VkFence>      inFlightFence;
	// Optional VK_KHR/EXT_swapchain_maintenance1 fences. Unlike inFlightFence,
	// these retire presentation-engine access to the swapchain image and the
	// RenderDone semaphore. pending tracks fences currently owned by a present.
	oa::Vec<VkFence>      presentFence;
	oa::Vec<bool>         presentFencePending;
	// A failed vkQueuePresentKHR may leave completion ownership ambiguous. The
	// rare recovery path falls back to a queue drain before destroying WSI state.
	bool                      presentCompletionUncertain = false;

	// ─── pending resize signal ───────────────────────────────────────────────
	// set from the SDL window event handler (main thread). consumed at the top
	// of the next DrawFrame, which calls recreateSwapchain(dirtySize). This
	// avoids racing the render thread when the OS reports a pixel-size change.
	bool                      dirty         = false;
	VkExtent2D                dirtySize     = {};

	// When true (default), BuildSwapchainObjects picks VK_PRESENT_MODE_FIFO_KHR
	// (vsync — one frame per refresh). When false, prefer MAILBOX, falling back
	// to IMMEDIATE, then FIFO. set by the presenter owner before init/Recreate.
	bool                      vsync         = true;

	// ─── Convenience accessors ───────────────────────────────────────────────
	[[nodiscard]] oa::U32 width()      const noexcept { return extent.width;  }
	[[nodiscard]] oa::U32 height()     const noexcept { return extent.height; }
	[[nodiscard]] oa::U32 imageCount() const noexcept {
		return static_cast<oa::U32>(images.size());
	}
	[[nodiscard]] bool  isValid()    const noexcept {
		return handle != VK_NULL_HANDLE and presentReady;
	}
};

} // namespace oa
