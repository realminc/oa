// OaSwapchain — WSI swapchain state for the Present render sink.
//
// See Architecture/OaArchitecture.md §10: one render sink-resource among peers. The renderer
// produces an OaTexture; the chosen sink decides what happens next.
//   Present     — this file (Vulkan WSI swapchain → vkQueuePresentKHR)
//   SaveImage   — filesystem
//   EncodeFrame — video encoder
//
// Lifecycle is owned by OaPresenter. Engine-internal Build/Destroy
// methods write into the held OaSwapchain; external callers read via
// OaPresenter::Swapchain(). Headless-mode engines never instantiate
// one.
//
// State sequence:
//   default construction       → zero-state (Handle == VK_NULL_HANDLE)
//   presenter.InitPresentation()  → swapchain objects + per-frame sync built;
//                                PresentReady = true
//   presenter.DetachPresentation()→ torn back down; zero-state again
//   presenter destructor          → unsubmitted work cancelled; submitted
//                                work and attached WSI state retired by engine
//
// This separate type untangles
// presentation state from the engine's core compute/graphics queues and
// caches. OaPresenter owns the acquire/present protocol as a composed service.

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
	// Surface is normally OWNED BY THE CALLER (typically OaVkWindow / SDL3). The
	// engine references it during the swapchain lifetime and clears the pointer
	// on explicit Close/Detach. If an attached presenter is abandoned, ownership
	// transfers to engine retirement so the surface outlives the swapchain and is
	// destroyed before the VkInstance; the caller must not destroy it afterward.
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
	// Optional VK_KHR/EXT_swapchain_maintenance1 fences. Unlike InFlightFence,
	// these retire presentation-engine access to the swapchain image and the
	// RenderDone semaphore. Pending tracks fences currently owned by a present.
	std::vector<VkFence>      PresentFence;
	std::vector<bool>         PresentFencePending;
	// A failed vkQueuePresentKHR may leave completion ownership ambiguous. The
	// rare recovery path falls back to a queue drain before destroying WSI state.
	bool                      PresentCompletionUncertain = false;

	// ─── Pending resize signal ───────────────────────────────────────────────
	// Set from the SDL window event handler (main thread). Consumed at the top
	// of the next DrawFrame, which calls RecreateSwapchain(DirtySize). This
	// avoids racing the render thread when the OS reports a pixel-size change.
	bool                      Dirty         = false;
	VkExtent2D                DirtySize     = {};

	// When true (default), BuildSwapchainObjects picks VK_PRESENT_MODE_FIFO_KHR
	// (vsync — one frame per refresh). When false, prefer MAILBOX, falling back
	// to IMMEDIATE, then FIFO. Set by the presenter owner before Init/Recreate.
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
