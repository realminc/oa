// Optional presentation session borrowing one oa::Engine.
// include this header only for window-system integration; compute-only callers
// need only <oa/runtime/engine.h>.
//
// Two-phase SDL3 setup:
//   oa::EngineConfig config;
//   config.presentationMode = oa::PresentationMode::Swapchain;
//   std::vector<const char*> extensions;
//   window.getPresenterInstanceExtensions(&extensions);
//   for (const char* extension : extensions)
//       config.instanceExtraExtensions.pushBack(extension);
//   auto engine = oa::Engine::create(config).getValue();
//   oa::Presenter presenter(*engine);
//   auto surface = presenter.createSurface(window).getValue();
//   presenter.initPresentation(surface, extent);

#pragma once

#include <vector>

#include <oa/runtime/engine.h>
#include <oa/runtime/swapchain.h>


namespace oavk { class Stream; }

namespace oa {

class VulkanWindow;

// surface ownership normally stays with the caller. call close() or
// detachPresentation() before destroying the surface. as misuse containment,
// destroying an attached presenter transfers the surface and WSI state to the
// engine; the engine destroys them at close() and the caller must not do so.
class Presenter {
public:
	explicit Presenter(oa::Engine& inEngine) noexcept : engine_(inEngine) {}
	Presenter(Presenter&&)            = delete;
	Presenter(const Presenter&)       = delete;
	~Presenter();

	Presenter& operator=(Presenter&&)      = delete;
	Presenter& operator=(const Presenter&) = delete;

	[[nodiscard]] oa::Engine& engine() const noexcept { return engine_; }
	[[nodiscard]] bool hasGraphics() const noexcept { return engine_.hasGraphics(); }
	[[nodiscard]] bool hasPresent()  const noexcept { return swapchain_.presentReady; }

	// Create/destroy a caller-owned opaque WSI surface through a window backend.
	// The vulkan instance never leaves the runtime boundary. detach presentation
	// before destroying an attached surface.
	[[nodiscard]] oa::Result<void*> createSurface(const VulkanWindow& inWindow) const;
	[[nodiscard]] oa::Status destroySurface(void*& inOutSurface) const;

	// phase-C: attach surface, build swapchain + renderpass + sync.
	// Safe to call again after detachPresentation() with a new surface.
	[[nodiscard]] bool initPresentation(void* inSurface, VkExtent2D inExtent);

	// Tear down swapchain-dependent resources, clear surface_.
	// call BEFORE vkDestroySurfaceKHR on the old surface.
	void detachPresentation();

	// phase-D: ImGui SDL3 backend init. compile with -DOA_IMGUI to activate.
	// inNativeWindow — SDL_Window* from oa::VulkanWindow::getNativeWindowHandle().
	// Without OA_IMGUI: returns true immediately (no-op).
	[[nodiscard]] bool initImGui(void* inNativeWindow);
	void               shutdownImGui();

	// Per-frame ImGui — no-ops without OA_IMGUI.
	void beginImGuiFrame();   // processes SDL events + ImGui::NewFrame
	void endImGuiFrame();     // ImGui::Render

	// acquire → record (clear + optional ImGui) → submit → present.
	[[nodiscard]] bool drawFrame();

	// Explicit presentation primitives. oa::Viewer uses these directly; compute
	// graph recording has no swapchain or WSI state.
	//
	// acquire result: indices + handles the caller needs to address the
	// per-frame sync slot and the acquired image. out parameters keep the
	// header free of additional includes.
	struct AcquireResult {
		oa::U32       frameSlot   = 0;   // = the oa::Swapchain::frameIndex used for this acquire
		oa::U32       imageIndex  = 0;
		VkImage     image       = VK_NULL_HANDLE;
		VkImageView view        = VK_NULL_HANDLE;
		bool        recreated   = false; // true if swapchain was recreated; frame should be skipped
	};

	// acquire the next swapchain image. Waits the inFlightFence at the
	// current frameIndex, calls vkAcquireNextImageKHR signalling
	// imageAvailSem[frameIndex], populates outResult. Returns false on hard
	// error (surface lost, zero-size window).
	[[nodiscard]] bool acquireSwapchainImage(Swapchain& inSwap, AcquireResult& outResult);

	// Body of the graphics CB that presentSwapchainImage builds. All members
	// are optional; precedence (step 3b.5):
	//
	//   drawImGui = true        → render pass with loadOp=CLEAR + ImGui draw
	//                              → finalLayout PRESENT_SRC. ImGui-only
	//                              renders today; blit+ImGui composite needs
	//                              a render-pass with loadOp=LOAD (step 3c).
	//   blitSrcBuffer != nullptr → UNDEFINED → TRANSFER_DST →
	//                              vkCmdCopyBufferToImage → PRESENT_SRC
	//   clearRgba != nullptr     → UNDEFINED → TRANSFER_DST →
	//                              vkCmdClearColorImage → PRESENT_SRC
	//   none                    → UNDEFINED → PRESENT_SRC (no content;
	//                              chain smoke-test only)
	//
	// Buffer source is assumed to be packed 4-bytes-per-pixel and the same
	// extent as the swapchain image. format swizzle: today's swapchain is
	// VK_FORMAT_B8G8R8A8_SRGB but oa::Texture is RGBA8; channel-swap +
	// linear→sRGB conversion are tracked as known limitations until the
	// staging-image vkCmdBlitImage path lands.
	struct PresentArgs {
		const oa::F32* clearRgba       = nullptr;  // 4 floats in [0,1]
		void*        blitSrcBuffer   = nullptr;  // VkBuffer
		void*        blitSrcImage    = nullptr;  // vkImage (precedence over Buffer)
		oa::I32        blitSrcLayout   = 0;        // current VkImageLayout of blitSrcImage
		oa::U32        blitSrcWidth    = 0;
		oa::U32        blitSrcHeight   = 0;
		bool         drawImGui       = false;    // record ImGui draw via render pass
		oa::Filter     filter          = oa::Filter::Linear;
		void*        waitTimelineSemaphore = nullptr; // VkSemaphore
		oa::U64        waitTimelineValue = 0;
	};

	// Build a one-off graphics command buffer per the body described above,
	// then submit on the graphics queue waiting on imageAvailSem[inFrameSlot],
	// signalling renderDoneSem[inFrameSlot] with inFlightFence[inFrameSlot];
	// call vkQueuePresentKHR waiting on renderDoneSem[inFrameSlot]. Advances
	// Swapchain.frameIndex on success.
	[[nodiscard]] bool presentSwapchainImage(
		Swapchain&       inSwap,
		oa::U32              inImageIndex,
		oa::U32              inFrameSlot,
		const PresentArgs& inArgs);

	// True when graphicsQueue and computeQueue are the same vkQueue (common on laptops).
	[[nodiscard]] bool usesMergedGraphicsComputeQueue() const;

	// Compatibility spelling for presentation/ImGui callbacks that must surround
	// a raw vulkan queue operation. It selects the centralized mutex for the
	// exact compute, graphics, or present queue handle, including distinct queues.
	void lockSharedQueueSubmit(void* inQueue);
	void unlockSharedQueueSubmit(void* inQueue);

	// Explicit swapchain resize (DrawFrame handles OUT_OF_DATE automatically).
	[[nodiscard]] bool recreateSwapchain(VkExtent2D inNewExtent);
	// WSI completion is not implied by the render-submission timeline. Until a
	// present-id/fence path is enabled, this is the narrow explicit boundary for
	// destroying presentation-owned resources.
	[[nodiscard]] oa::Status waitPresentationIdle();

	// Unified graphics/compute frame batch recorded on the graphics queue.
	// Graphics queues also support compute, so Render canvas draws and oa::Ui
	// compute composition can share one command buffer and timeline edge. flush
	// returns the exact submission event consumed by presentation resources.
	[[nodiscard]] oa::Status beginGraphicsBatch();
	[[nodiscard]] oa::Result<oa::Event> flushGraphicsBatch(
		const oa::Event& inProducer = {});
	[[nodiscard]] oa::Status syncGraphicsBatch();
	[[nodiscard]] oavk::Stream* activeGraphicsBatchStream() const {
		return graphicsBatchStream_;
	}

	// call from the window's SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED handler.
	// DrawFrame will recreate the swapchain at the top of the next frame.
	void notifyPixelSizeChanged(int inWidthPx, int inHeightPx);

	[[nodiscard]] bool         isPresentationReady() const { return swapchain_.presentReady; }
	[[nodiscard]] bool         isImGuiReady()        const { return imGuiReady_; }
	[[nodiscard]] VkRenderPass getRenderPass()        const { return renderPass_; }
	[[nodiscard]] VkFormat     swapFormat()            const { return swapchain_.format; }
	[[nodiscard]] VkExtent2D   swapchainExtent()      const { return swapchain_.extent; }

	// Direct access to the swapchain (FinalGlue §3.5). Returns a reference to
	// the engine-owned oa::Swapchain; zero-state when no surface is attached.
	// Use isPresentationReady() / swapchain().isValid() to gate.
	[[nodiscard]] const Swapchain& swapchain() const noexcept { return swapchain_; }
	[[nodiscard]] Swapchain&       swapchain()       noexcept { return swapchain_; }

	[[nodiscard]] oa::Status close();

private:
	oa::Engine& engine_;

	[[nodiscard]] bool buildSwapchainObjects();
	[[nodiscard]] bool buildRenderPass();
	[[nodiscard]] bool buildFramebuffers();
	[[nodiscard]] bool buildCommandPool();
	[[nodiscard]] bool buildSyncObjects();
	[[nodiscard]] oa::Status preparePresentFence(Swapchain& inSwap, oa::U32 inFrameSlot);
	void finishPresent(Swapchain& inSwap, oa::U32 inFrameSlot, VkResult inResult);

	void destroySwapchainObjects();
	void destroySyncObjects();
	void destroyCommandPool();
	void shutdownImGuiResources_();
	void abandon_() noexcept;
	[[nodiscard]] bool hasOwnedState_() const noexcept;
	static void lockSharedQueueSubmitCallback_(VkQueue inQueue, void* inUser);
	static void unlockSharedQueueSubmitCallback_(VkQueue inQueue, void* inUser);

	// WSI swapchain state (handle, format, extent, images, views, per-frame
	// sync, dirty-resize signal). Extracted into a standalone type so other
	// render sinks (saveImage, encodeFrame) sit at the same ownership level.
	Swapchain swapchain_;

	// Render pass + framebuffers reference swapchain_.views directly; they go
	// away in step 5 when dynamic rendering replaces VkRenderPass.
	VkRenderPass               renderPass_  = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> framebuffers_;

	VkCommandPool                cmdPool_ = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> cmdBufs_;

	VkDescriptorPool imGuiPool_  = VK_NULL_HANDLE;
	bool             imGuiReady_ = false;

	static constexpr oa::U32 kGraphicsBatchRingSize = 4;
	oa::UniquePtr<oavk::Stream> graphicsBatchRing_[kGraphicsBatchRingSize];
	oavk::Stream* graphicsBatchStream_ = nullptr;
	oa::U32 graphicsBatchRingIndex_ = 0;
};

} // namespace oa
