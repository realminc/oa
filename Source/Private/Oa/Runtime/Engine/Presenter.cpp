// OaPresenter — WSI swapchain + optional Dear ImGui on top of OaEngine.
//
// SDL3 backend.  Build with -DOA_IMGUI to enable ImGui integration.
// Without OA_IMGUI all ImGui-related methods compile as no-ops so the codebase
// stays clean while the UI layer is being wired up.
//
// When OA_IMGUI is defined you also need to link:
//   imgui  imgui_impl_vulkan  imgui_impl_sdl3
// and provide SDL3/SDL.h + backends/imgui_impl_sdl3.h in your include path.

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Core/Log.h>

#include "../PresenterRetirement.h"

#include <vulkan/vulkan.h>

#ifdef OA_IMGUI
#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#endif

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>


// ─── Internal helpers ────────────────────────────────────────────────────────

struct OaVkSharedQueueSubmitScope {
	OaPresenter* Eng_ = nullptr;
	explicit OaVkSharedQueueSubmitScope(OaPresenter* InEng) : Eng_(InEng) {
		if (Eng_) {
			Eng_->LockSharedQueueSubmit(Eng_->Engine().Device.Queues.GraphicsQueue);
		}
	}
	~OaVkSharedQueueSubmitScope() {
		if (Eng_) {
			Eng_->UnlockSharedQueueSubmit(Eng_->Engine().Device.Queues.GraphicsQueue);
		}
	}
};

static inline VkDevice ReDev(const OaPresenter& E) {
	return static_cast<VkDevice>(E.Engine().Device.Device);
}
static inline VkPhysicalDevice RePhys(const OaPresenter& E) {
	return static_cast<VkPhysicalDevice>(E.Engine().Device.PhysicalDevice);
}


// ─── Move / copy ─────────────────────────────────────────────────────────────

// OaPresenter is pinned (move/copy = delete in the header) and borrows its
// engine. It never relocates,
// so the swapchain/render-pass/command-pool/ImGui handles it owns can't be
// aliased into a moved-from twin. (The old hand-written move ctor/assignment that
// reset the source's Swapchain_/RenderPass_/CmdPool_/ImGuiPool_ to dodge
// double-destroy is gone.)

OaPresenter::~OaPresenter() {
	Abandon_();
}

bool OaPresenter::HasOwnedState_() const noexcept {
	OaBool hasBatchState = GraphicsBatchStream_ != nullptr;
	for (const auto& stream : GraphicsBatchRing_) {
		hasBatchState = hasBatchState || stream != nullptr;
	}
	const OaBool hasPresentationState =
		Swapchain_.Surface != VK_NULL_HANDLE ||
		Swapchain_.Handle != VK_NULL_HANDLE ||
		RenderPass_ != VK_NULL_HANDLE ||
		CmdPool_ != VK_NULL_HANDLE ||
		ImGuiPool_ != VK_NULL_HANDLE ||
		!Framebuffers_.empty() || !CmdBufs_.empty();
	return hasBatchState || hasPresentationState;
}

OaStatus OaPresenter::Close() {
	if (!HasOwnedState_()) return OaStatus::Ok();
	if (!Engine_.Device.Device) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaPresenter::Close: borrowed engine is not live");
	}
	// A presenter is cheap to construct as a borrowed service. If it was never
	// initialized, close must not turn into an unrelated graphics-queue wait.
	OaStatus firstError = OaStatus::Ok();
	auto retainError = [&firstError](const OaStatus& InStatus) {
		if (firstError.IsOk() && !InStatus.IsOk()) firstError = InStatus;
	};
	if (GraphicsBatchStream_) {
		const auto resetStatus =
			GraphicsBatchStream_->ResetUnsubmitted(Engine_.Device);
		retainError(resetStatus);
		if (resetStatus.IsOk()) GraphicsBatchStream_ = nullptr;
	}
	for (OaU32 i = 0; i < kGraphicsBatchRingSize; ++i) {
		if (GraphicsBatchRing_[i] && GraphicsBatchRing_[i]->Submitted) {
			retainError(GraphicsBatchRing_[i]->Synchronize(Engine_.Device));
		}
	}
	// Render fences do not prove that presentation has retired. Drain only the
	// WSI queue; compute, transfer, and video queues remain independent.
	if (Swapchain_.PresentReady) retainError(WaitPresentationIdle());
	if (!firstError.IsOk()) return firstError;

	GraphicsBatchStream_ = nullptr;
	for (OaU32 i = 0; i < kGraphicsBatchRingSize; ++i) {
		if (GraphicsBatchRing_[i]) {
			GraphicsBatchRing_[i]->Destroy(Engine_.Device);
			GraphicsBatchRing_[i].reset();
		}
	}
	GraphicsBatchRingIndex_ = 0;
	ShutdownImGuiResources_();
	DestroySyncObjects();
	DestroyCommandPool();
	DestroySwapchainObjects();
	Engine_.Device.Queues.HasPresentation = false;
	Engine_.Device.Queues.PresentQueue = nullptr;
	Engine_.Device.Queues.PresentQueueFamily = OaVkEnumerationIndexUnset;
	Swapchain_ = {};
	// Note: Swapchain_.Surface is caller-owned — do NOT destroy it here.
	// Caller does: vkDestroySurfaceKHR(instance, surface, nullptr)
	return firstError;
}

void OaPresenter::Destroy() {
	if (auto status = Close(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaPresenter::Destroy: shutdown failed: %s",
			status.ToString().c_str());
	}
}

void OaPresenter::Abandon_() noexcept {
	if (!HasOwnedState_()) return;
	if (!Engine_.Device.Device) {
		for (auto& stream : GraphicsBatchRing_) stream.reset();
		GraphicsBatchStream_ = nullptr;
		Swapchain_ = {};
		RenderPass_ = VK_NULL_HANDLE;
		Framebuffers_.clear();
		CmdPool_ = VK_NULL_HANDLE;
		CmdBufs_.clear();
		ImGuiPool_ = VK_NULL_HANDLE;
		ImGuiReady_ = false;
		return;
	}

	if (GraphicsBatchStream_) {
		if (const auto status =
			GraphicsBatchStream_->ResetUnsubmitted(Engine_.Device);
			not status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaPresenter abandonment failed to cancel graphics batch: %s",
				status.GetMessage().c_str());
		}
		GraphicsBatchStream_ = nullptr;
	}

	auto retired = OaMakeUniquePtr<OaRetiredPresenter>();
	retired->Swapchain = std::move(Swapchain_);
	retired->RenderPass = RenderPass_;
	retired->Framebuffers = std::move(Framebuffers_);
	retired->CommandPool = CmdPool_;
	retired->CommandBuffers = std::move(CmdBufs_);
	retired->ImGuiPool = ImGuiPool_;
	retired->ImGuiReady = ImGuiReady_;
	retired->PresentQueue = Engine_.Device.Queues.PresentQueue != nullptr
		? Engine_.Device.Queues.PresentQueue
		: Engine_.Device.Queues.GraphicsQueue;
	retired->UsesMergedGraphicsComputeQueue = UsesMergedGraphicsComputeQueue();
	retired->HasSwapchainMaintenance1 =
		Engine_.Device.Info.Software.HasSwapchainMaintenance1;
	retired->OwnsAbandonedSurface = retired->Swapchain.Surface != nullptr;
	for (auto& stream : GraphicsBatchRing_) {
		if (stream) retired->GraphicsStreams.PushBack(OaStdMove(stream));
	}

	const OaBool retainedCallerSurface = retired->Swapchain.Surface != nullptr;
	Swapchain_ = {};
	RenderPass_ = VK_NULL_HANDLE;
	Framebuffers_.clear();
	CmdPool_ = VK_NULL_HANDLE;
	CmdBufs_.clear();
	ImGuiPool_ = VK_NULL_HANDLE;
	ImGuiReady_ = false;
	GraphicsBatchRingIndex_ = 0;
	Engine_.Device.Queues.HasPresentation = false;
	Engine_.Device.Queues.PresentQueue = nullptr;
	Engine_.Device.Queues.PresentQueueFamily = OaVkEnumerationIndexUnset;
	Engine_.RetirePresenter(OaStdMove(retired));

	if (retainedCallerSurface) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaPresenter was destroyed while attached; ownership of its surface "
			"transferred to OaEngine retirement and the caller must not destroy it");
	}
}

OaStatus OaPresenter::BeginGraphicsBatch() {
	if (GraphicsBatchStream_) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"BeginGraphicsBatch: already active");
	}
	const OaU32 index = GraphicsBatchRingIndex_ % kGraphicsBatchRingSize;
	if (!GraphicsBatchRing_[index]) {
		auto stream = OaVkStream::Create(
			Engine_.Device,
			Engine_.Device.Queues.GraphicsQueueFamily,
			Engine_.Device.Queues.GraphicsQueue);
		if (!stream.IsOk()) return stream.GetStatus();
		GraphicsBatchRing_[index] =
			OaMakeUniquePtr<OaVkStream>(OaStdMove(*stream));
	}
	OA_RETURN_IF_ERROR(GraphicsBatchRing_[index]->Begin(Engine_.Device));
	GraphicsBatchStream_ = GraphicsBatchRing_[index].get();
	return OaStatus::Ok();
}

OaStatus OaPresenter::FlushGraphicsBatch(
	const OaVkTimelineSemaphore* InProducerSemaphore,
	OaU64 InProducerValue) {
	if (!GraphicsBatchStream_) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"FlushGraphicsBatch: no active batch");
	}
	OaVkStream* current = GraphicsBatchStream_;
	GraphicsBatchStream_ = nullptr;
	OaVkTimelineWait waits[2] = {};
	OaU32 waitCount = 0;
	if (GraphicsBatchRingIndex_ > 0) {
		const OaU32 previous =
			(GraphicsBatchRingIndex_ - 1) % kGraphicsBatchRingSize;
		if (GraphicsBatchRing_[previous]
			&& GraphicsBatchRing_[previous]->Submitted) {
			waits[waitCount++] = {
				&GraphicsBatchRing_[previous]->TimelineSem,
				GraphicsBatchRing_[previous]->TimelineValue};
		}
	}
	if (InProducerSemaphore != nullptr && InProducerSemaphore->Semaphore != nullptr
		&& InProducerValue > 0) {
		waits[waitCount++] = {InProducerSemaphore, InProducerValue};
	}
	OaStatus status = waitCount > 0
		? current->SubmitWithDependencies(
			Engine_, OaSpan<const OaVkTimelineWait>(waits, waitCount))
		: current->Submit(Engine_);
	if (status.IsOk()) ++GraphicsBatchRingIndex_;
	return status;
}

OaStatus OaPresenter::SyncGraphicsBatch() {
	if (GraphicsBatchRingIndex_ == 0) return OaStatus::Ok();
	const OaU32 previous =
		(GraphicsBatchRingIndex_ - 1) % kGraphicsBatchRingSize;
	if (!GraphicsBatchRing_[previous]
		|| !GraphicsBatchRing_[previous]->Submitted) {
		return OaStatus::Ok();
	}
	return GraphicsBatchRing_[previous]->Synchronize(Engine_.Device);
}


// Phase-C: InitPresentation.
bool OaPresenter::InitPresentation(void* InSurface, VkExtent2D InExtent) {
	if (!InSurface) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaPresenter::InitPresentation: null surface");
		return false;
	}
	if (!Engine_.Device.Device) {
		OA_LOG_ERROR(OaLogComponent::Core, "OaPresenter::InitPresentation: device not ready");
		return false;
	}
	if (Engine_.Device.Queues.GraphicsQueueFamily == OaVkEnumerationIndexUnset) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaPresenter::InitPresentation: no graphics queue — "
			"was PresentationMode = Swapchain set in the config?"
		);
		return false;
	}

	// If already presenting, tear down the old resources first.
	if (Swapchain_.PresentReady) {
		DetachPresentation();
		if (Swapchain_.PresentReady) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaPresenter::InitPresentation: existing presentation could not be detached");
			return false;
		}
	}

	// Confirm the graphics queue family supports present on this surface.
	VkBool32 presentOk = VK_FALSE;
	vkGetPhysicalDeviceSurfaceSupportKHR(
		RePhys(*this),
		Engine_.Device.Queues.GraphicsQueueFamily,
		static_cast<VkSurfaceKHR>(InSurface),
		&presentOk);

	if (!presentOk) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaPresenter::InitPresentation: graphics queue family %u "
			"does not support present on this surface",
			Engine_.Device.Queues.GraphicsQueueFamily);
		return false;
	}

	// Promote queue state so everything downstream can use PresentQueue.
	Engine_.Device.Queues.PresentQueue       = Engine_.Device.Queues.GraphicsQueue;
	Engine_.Device.Queues.PresentQueueFamily = Engine_.Device.Queues.GraphicsQueueFamily;
	Engine_.Device.Queues.HasPresentation    = true;
	Swapchain_.Surface   = InSurface;
	Swapchain_.Extent = InExtent;

	const auto rollbackUnsubmittedInit = [this]() {
		DestroySyncObjects();
		DestroyCommandPool();
		DestroySwapchainObjects();
		Engine_.Device.Queues.HasPresentation = false;
		Engine_.Device.Queues.PresentQueue = nullptr;
		Engine_.Device.Queues.PresentQueueFamily = OaVkEnumerationIndexUnset;
		Swapchain_ = {};
	};
	if (!BuildSwapchainObjects() || !BuildRenderPass()
		|| !BuildFramebuffers() || !BuildCommandPool() || !BuildSyncObjects()) {
		rollbackUnsubmittedInit();
		return false;
	}

	Swapchain_.PresentReady = true;
	OA_LOG_INFO(OaLogComponent::Core,
		"OaPresenter: presentation ready (%ux%u, %zu swap images, "
		"retirement=%s)",
		Swapchain_.Extent.width, Swapchain_.Extent.height, Swapchain_.Images.size(),
		Engine_.Device.Info.Software.HasSwapchainMaintenance1
			? "present-fence" : "queue-idle-fallback");
	return true;
}

// DetachPresentation — tear down swapchain resources, leave device alive.
// Caller must vkDestroySurfaceKHR the old surface after this.
void OaPresenter::DetachPresentation() {
	if (!Swapchain_.PresentReady) return;
	if (auto status = WaitPresentationIdle(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"DetachPresentation: %s", status.ToString().c_str());
		return;
	}
	ShutdownImGuiResources_();
	DestroySyncObjects();
	DestroyCommandPool();
	DestroySwapchainObjects();
	Engine_.Device.Queues.HasPresentation    = false;
	Engine_.Device.Queues.PresentQueue       = nullptr;
	Engine_.Device.Queues.PresentQueueFamily = OaVkEnumerationIndexUnset;
	Swapchain_ = {};
}


// ─── RecreateSwapchain ───────────────────────────────────────────────────────

bool OaPresenter::RecreateSwapchain(VkExtent2D InNewExtent) {
	if (!Swapchain_.Surface) return false;
	if (auto status = WaitPresentationIdle(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"RecreateSwapchain: %s", status.ToString().c_str());
		return false;
	}

	// Keep sync objects alive — only rebuild swapchain-dependent things.
	DestroyCommandPool();
	DestroySwapchainObjects();

	Swapchain_.Extent = InNewExtent;
	if (!BuildSwapchainObjects()) return false;
	if (!BuildRenderPass())       return false;
	if (!BuildFramebuffers())     return false;
	if (!BuildCommandPool())      return false;

	// Re-allocate command buffers into the new pool.
	CmdBufs_.resize(OaSwapchain::kFramesInFlight);
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = CmdPool_;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = static_cast<uint32_t>(CmdBufs_.size());
	if (vkAllocateCommandBuffers(ReDev(*this), &ai, CmdBufs_.data()) != VK_SUCCESS)
		return false;

	// ImGui_ImplVulkan_SetMinImageCount after Init hits IM_ASSERT(0) in imgui_impl_vulkan
	// (unsupported); NDEBUG hid it, debug aborted when swap image count != OaSwapchain::kFramesInFlight.
	return true;
}


// ─── BuildSwapchainObjects ───────────────────────────────────────────────────

bool OaPresenter::BuildSwapchainObjects() {
	VkSurfaceKHR surf = static_cast<VkSurfaceKHR>(Swapchain_.Surface);

	VkSurfaceCapabilitiesKHR caps{};
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(RePhys(*this), surf, &caps);

	// SDL3 / Wayland: currentExtent may be (0,0) for hidden windows — skip frame.
	if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) {
		// Try hint extent; if that's also 0, silently succeed and let DrawFrame skip.
		if (Swapchain_.Extent.width == 0 || Swapchain_.Extent.height == 0) return true;
	}

	VkExtent2D extent = caps.currentExtent;
	if (extent.width == std::numeric_limits<uint32_t>::max()) {
		extent.width  = std::clamp(Swapchain_.Extent.width,
			caps.minImageExtent.width,  caps.maxImageExtent.width);
		extent.height = std::clamp(Swapchain_.Extent.height,
			caps.minImageExtent.height, caps.maxImageExtent.height);
	}
	Swapchain_.Extent = extent;

	// Pick surface format. The compose image is RGBA8 UNORM and widgets/video
	// already write what they think are sRGB-encoded bytes. Picking a SRGB
	// swapchain format makes the compose→swap blit apply OETF a second time,
	// which is the classic "washed out" symptom. Prefer UNORM so the blit is
	// a straight byte copy; fall back to SRGB if the surface only offers that.
	uint32_t fmtCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(RePhys(*this), surf, &fmtCount, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(fmtCount);
	if (fmtCount) vkGetPhysicalDeviceSurfaceFormatsKHR(RePhys(*this), surf, &fmtCount, formats.data());

	VkSurfaceFormatKHR chosenFmt = fmtCount
		? formats[0]
		: VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
	for (const auto& f : formats) {
		if ((f.format == VK_FORMAT_B8G8R8A8_UNORM
		  or f.format == VK_FORMAT_R8G8B8A8_UNORM)
		 and f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			chosenFmt = f;
			break;
		}
	}

	// Present mode. Default FIFO (vsync, one frame per monitor refresh, GPU
	// idles between). Override via Swapchain_.Vsync = false: prefer MAILBOX
	// (triple-buffered, never-tear, doesn't wait), fall back to IMMEDIATE
	// (tear-allowed, never-wait), then FIFO when neither is exposed.
	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if (not Swapchain_.Vsync) {
		uint32_t modeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(RePhys(*this), surf, &modeCount, nullptr);
		std::vector<VkPresentModeKHR> modes(modeCount);
		if (modeCount) vkGetPhysicalDeviceSurfacePresentModesKHR(RePhys(*this), surf, &modeCount, modes.data());
		bool hasMailbox   = false;
		bool hasImmediate = false;
		for (auto m : modes) {
			if (m == VK_PRESENT_MODE_MAILBOX_KHR)   hasMailbox   = true;
			if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) hasImmediate = true;
		}
		if (hasMailbox)        presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
		else if (hasImmediate) presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
	}

	uint32_t imgCount = caps.minImageCount + 1;
	if (caps.maxImageCount > 0) imgCount = std::min(imgCount, caps.maxImageCount);

	VkSwapchainCreateInfoKHR sci{};
	sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	sci.surface          = surf;
	sci.minImageCount    = imgCount;
	sci.imageFormat      = chosenFmt.format;
	sci.imageColorSpace  = chosenFmt.colorSpace;
	sci.imageExtent      = extent;
	sci.imageArrayLayers = 1;
	// COLOR_ATTACHMENT for ImGui/render-pass renders (legacy DrawFrame path).
	// TRANSFER_DST so explicit presentation can vkCmdClearColorImage and
	// vkCmdCopyBufferToImage into the acquired swap image (FinalGlue Step 3b.3/4).
	sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
	                     | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	sci.preTransform     = caps.currentTransform;
	sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	sci.presentMode      = presentMode;
	sci.clipped          = VK_TRUE;

	uint32_t families[] = {
		Engine_.Device.Queues.GraphicsQueueFamily,
		Engine_.Device.Queues.PresentQueueFamily
	};
	if (Engine_.Device.Queues.GraphicsQueueFamily != Engine_.Device.Queues.PresentQueueFamily) {
		sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		sci.queueFamilyIndexCount = 2;
		sci.pQueueFamilyIndices   = families;
	} else {
		sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	if (vkCreateSwapchainKHR(ReDev(*this), &sci, nullptr, &Swapchain_.Handle) != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkCreateSwapchainKHR failed");
		return false;
	}
	Swapchain_.Format = chosenFmt.format;

	uint32_t actualCount = 0;
	vkGetSwapchainImagesKHR(ReDev(*this), Swapchain_.Handle, &actualCount, nullptr);
	Swapchain_.Images.resize(actualCount);
	vkGetSwapchainImagesKHR(ReDev(*this), Swapchain_.Handle, &actualCount, Swapchain_.Images.data());

	Swapchain_.Views.resize(actualCount);
	for (uint32_t i = 0; i < actualCount; ++i) {
		VkImageViewCreateInfo vi{};
		vi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image                       = Swapchain_.Images[i];
		vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
		vi.format                      = Swapchain_.Format;
		vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		vi.subresourceRange.levelCount = 1;
		vi.subresourceRange.layerCount = 1;
		if (vkCreateImageView(ReDev(*this), &vi, nullptr, &Swapchain_.Views[i]) != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "vkCreateImageView failed (index %u)", i);
			return false;
		}
	}
	return true;
}


// ─── BuildRenderPass ─────────────────────────────────────────────────────────

bool OaPresenter::BuildRenderPass() {
	VkAttachmentDescription color{};
	color.format         = Swapchain_.Format;
	color.samples        = VK_SAMPLE_COUNT_1_BIT;
	color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorRef{};
	colorRef.attachment = 0;
	colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &colorRef;

	// Ensure image layout transition completes before writing colour attachment.
	VkSubpassDependency dep{};
	dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
	dep.dstSubpass    = 0;
	dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dep.srcAccessMask = 0;
	dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo rpi{};
	rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	rpi.attachmentCount = 1;
	rpi.pAttachments    = &color;
	rpi.subpassCount    = 1;
	rpi.pSubpasses      = &subpass;
	rpi.dependencyCount = 1;
	rpi.pDependencies   = &dep;

	if (vkCreateRenderPass(ReDev(*this), &rpi, nullptr, &RenderPass_) != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkCreateRenderPass failed");
		return false;
	}
	return true;
}


// ─── BuildFramebuffers ───────────────────────────────────────────────────────

bool OaPresenter::BuildFramebuffers() {
	Framebuffers_.resize(Swapchain_.Views.size());
	for (size_t i = 0; i < Swapchain_.Views.size(); ++i) {
		VkImageView att = Swapchain_.Views[i];
		VkFramebufferCreateInfo fbi{};
		fbi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbi.renderPass      = RenderPass_;
		fbi.attachmentCount = 1;
		fbi.pAttachments    = &att;
		fbi.width           = Swapchain_.Extent.width;
		fbi.height          = Swapchain_.Extent.height;
		fbi.layers          = 1;
		if (vkCreateFramebuffer(ReDev(*this), &fbi, nullptr, &Framebuffers_[i]) != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "vkCreateFramebuffer failed (index %zu)", i);
			return false;
		}
	}
	return true;
}


// ─── BuildCommandPool ────────────────────────────────────────────────────────

bool OaPresenter::BuildCommandPool() {
	VkCommandPoolCreateInfo pci{};
	pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pci.queueFamilyIndex = Engine_.Device.Queues.GraphicsQueueFamily;
	pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	if (vkCreateCommandPool(ReDev(*this), &pci, nullptr, &CmdPool_) != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkCreateCommandPool failed");
		return false;
	}

	CmdBufs_.resize(OaSwapchain::kFramesInFlight);
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = CmdPool_;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = static_cast<uint32_t>(CmdBufs_.size());
	if (vkAllocateCommandBuffers(ReDev(*this), &ai, CmdBufs_.data()) != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkAllocateCommandBuffers failed");
		return false;
	}
	return true;
}


// ─── BuildSyncObjects ────────────────────────────────────────────────────────

bool OaPresenter::BuildSyncObjects() {
	Swapchain_.ImageAvailSem.resize(OaSwapchain::kFramesInFlight);
	Swapchain_.RenderDoneSem.resize(OaSwapchain::kFramesInFlight);
	Swapchain_.InFlightFence.resize(OaSwapchain::kFramesInFlight);
	if (Engine_.Device.Info.Software.HasSwapchainMaintenance1) {
		Swapchain_.PresentFence.resize(OaSwapchain::kFramesInFlight);
		Swapchain_.PresentFencePending.assign(OaSwapchain::kFramesInFlight, false);
	}
	Swapchain_.PresentCompletionUncertain = false;

	VkSemaphoreCreateInfo si{};
	si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkFenceCreateInfo fi{};
	fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // pre-signalled so frame 0 doesn't stall
	VkFenceCreateInfo presentFi{};
	presentFi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	for (int i = 0; i < OaSwapchain::kFramesInFlight; ++i) {
		if (vkCreateSemaphore(ReDev(*this), &si, nullptr, &Swapchain_.ImageAvailSem[i]) != VK_SUCCESS ||
			vkCreateSemaphore(ReDev(*this), &si, nullptr, &Swapchain_.RenderDoneSem[i]) != VK_SUCCESS ||
			vkCreateFence    (ReDev(*this), &fi, nullptr, &Swapchain_.InFlightFence[i])  != VK_SUCCESS ||
			(Engine_.Device.Info.Software.HasSwapchainMaintenance1 &&
				vkCreateFence(ReDev(*this), &presentFi, nullptr,
					&Swapchain_.PresentFence[i]) != VK_SUCCESS))
		{
			OA_LOG_ERROR(OaLogComponent::Core, "sync object creation failed (slot %d)", i);
			return false;
		}
	}
	return true;
}


// ─── Destroy helpers ─────────────────────────────────────────────────────────

void OaPresenter::DestroySwapchainObjects() {
	if (!Engine_.Device.Device) return;
	VkDevice d = ReDev(*this);

	for (auto fb : Framebuffers_)   if (fb)  vkDestroyFramebuffer(d, fb, nullptr);
	Framebuffers_.clear();

	for (auto iv : Swapchain_.Views) if (iv)  vkDestroyImageView(d, iv, nullptr);
	Swapchain_.Views.clear();
	Swapchain_.Images.clear();

	if (RenderPass_ != VK_NULL_HANDLE) {
		vkDestroyRenderPass(d, RenderPass_, nullptr);
		RenderPass_ = VK_NULL_HANDLE;
	}
	if (Swapchain_.Handle != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(d, Swapchain_.Handle, nullptr);
		Swapchain_.Handle = VK_NULL_HANDLE;
	}
}

void OaPresenter::DestroyCommandPool() {
	if (!Engine_.Device.Device || CmdPool_ == VK_NULL_HANDLE) return;
	// Explicitly free command buffers before destroying pool (Vulkan validation requirement)
	if (!CmdBufs_.empty()) {
		vkFreeCommandBuffers(ReDev(*this), CmdPool_,
			static_cast<uint32_t>(CmdBufs_.size()), CmdBufs_.data());
		CmdBufs_.clear();
	}
	vkDestroyCommandPool(ReDev(*this), CmdPool_, nullptr);
	CmdPool_ = VK_NULL_HANDLE;
}

void OaPresenter::DestroySyncObjects() {
	if (!Engine_.Device.Device) return;
	VkDevice d = ReDev(*this);
	for (auto s : Swapchain_.ImageAvailSem) if (s) vkDestroySemaphore(d, s, nullptr);
	for (auto s : Swapchain_.RenderDoneSem) if (s) vkDestroySemaphore(d, s, nullptr);
	for (auto f : Swapchain_.InFlightFence) if (f) vkDestroyFence(d, f, nullptr);
	for (auto f : Swapchain_.PresentFence) if (f) vkDestroyFence(d, f, nullptr);
	Swapchain_.ImageAvailSem.clear();
	Swapchain_.RenderDoneSem.clear();
	Swapchain_.InFlightFence.clear();
	Swapchain_.PresentFence.clear();
	Swapchain_.PresentFencePending.clear();
	Swapchain_.PresentCompletionUncertain = false;
}


// ─── ImGui integration (OA_IMGUI) ────────────────────────────────────────────

#ifdef OA_IMGUI
void OaPresenter::LockSharedQueueSubmitCallback_(VkQueue InQueue, void* InUser) {
	auto* engine = static_cast<OaEngine*>(InUser);
	if (engine == nullptr || InQueue == VK_NULL_HANDLE) return;
	const void* queue = static_cast<void*>(InQueue);
	if (engine->Device.Queues.ComputeQueue != nullptr
		&& engine->Device.Queues.ComputeQueue == engine->Device.Queues.GraphicsQueue
		&& (queue == engine->Device.Queues.ComputeQueue
			|| queue == engine->Device.Queues.GraphicsQueue
			|| queue == engine->Device.Queues.PresentQueue)) {
		engine->ComputeQueueMutex_.lock();
	}
}
void OaPresenter::UnlockSharedQueueSubmitCallback_(VkQueue InQueue, void* InUser) {
	auto* engine = static_cast<OaEngine*>(InUser);
	if (engine == nullptr || InQueue == VK_NULL_HANDLE) return;
	const void* queue = static_cast<void*>(InQueue);
	if (engine->Device.Queues.ComputeQueue != nullptr
		&& engine->Device.Queues.ComputeQueue == engine->Device.Queues.GraphicsQueue
		&& (queue == engine->Device.Queues.ComputeQueue
			|| queue == engine->Device.Queues.GraphicsQueue
			|| queue == engine->Device.Queues.PresentQueue)) {
		engine->ComputeQueueMutex_.unlock();
	}
}
#endif

bool OaPresenter::InitImGui(void* InNativeWindow) {
#ifdef OA_IMGUI
	if (!Swapchain_.PresentReady) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaPresenter::InitImGui: call InitPresentation first");
		return false;
	}
	if (ImGuiReady_) return true;

	// Dedicated descriptor pool for ImGui textures + fonts.
	VkDescriptorPoolSize poolSizes[] = {
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,           64},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           64},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          64},
	};
	VkDescriptorPoolCreateInfo dpci{};
	dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	dpci.maxSets       = 1024;
	dpci.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
	dpci.pPoolSizes    = poolSizes;
	if (vkCreateDescriptorPool(ReDev(*this), &dpci, nullptr, &ImGuiPool_) != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "ImGui: descriptor pool creation failed");
		return false;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // needed for node editor layouts
	ImGui::StyleColorsDark();

	// SDL3 platform backend.
	if (!ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(InNativeWindow))) {
		OA_LOG_ERROR(OaLogComponent::Core, "ImGui_ImplSDL3_InitForVulkan failed");
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(ReDev(*this), ImGuiPool_, nullptr);
		ImGuiPool_ = VK_NULL_HANDLE;
		return false;
	}

	// Load Vulkan functions for ImGui via our volk-style loader (OaVk).
	// Required because imgui_impl_vulkan.cpp is compiled with VK_NO_PROTOTYPES.
	VkInstance inst = static_cast<VkInstance>(Engine_.Device.Instance);
	ImGui_ImplVulkan_LoadFunctions(
		VK_API_VERSION_1_4,
		[](const char* InName, void* InUserData) -> PFN_vkVoidFunction {
			return vkGetInstanceProcAddr(
				static_cast<VkInstance>(InUserData), InName);
		},
		inst);

	// Vulkan backend.
	ImGui_ImplVulkan_InitInfo vii{};
	vii.Instance        = inst;
	vii.PhysicalDevice  = RePhys(*this);
	vii.Device          = ReDev(*this);
	vii.QueueFamily     = Engine_.Device.Queues.GraphicsQueueFamily;
	vii.Queue           = static_cast<VkQueue>(Engine_.Device.Queues.GraphicsQueue);
	vii.DescriptorPool  = ImGuiPool_;
	vii.MinImageCount   = OaSwapchain::kFramesInFlight;
	vii.ImageCount      = static_cast<uint32_t>(Swapchain_.Images.size());
	vii.PipelineInfoMain.RenderPass  = RenderPass_;
	vii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	vii.LockQueueSubmitFn = LockSharedQueueSubmitCallback_;
	vii.UnlockQueueSubmitFn = UnlockSharedQueueSubmitCallback_;
	vii.LockQueueSubmitUserData = &Engine_;

	if (!ImGui_ImplVulkan_Init(&vii)) {
		OA_LOG_ERROR(OaLogComponent::Core, "ImGui_ImplVulkan_Init failed");
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(ReDev(*this), ImGuiPool_, nullptr);
		ImGuiPool_ = VK_NULL_HANDLE;
		return false;
	}

	ImGuiReady_ = true;
	OA_LOG_INFO(OaLogComponent::Core, "OaPresenter: ImGui (SDL3) ready");
	return true;
#else
	(void)InNativeWindow;
	return true;   // No-op — link with OA_IMGUI to enable
#endif
}

void OaPresenter::ShutdownImGui() {
#ifdef OA_IMGUI
	if (!ImGuiReady_) return;
	if (auto status = WaitPresentationIdle(); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"ShutdownImGui: %s", status.ToString().c_str());
		return;
	}
	ShutdownImGuiResources_();
#endif
}

void OaPresenter::ShutdownImGuiResources_() {
#ifdef OA_IMGUI
	if (ImGuiReady_) {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
		ImGuiReady_ = false;
	}
#endif
	if (ImGuiPool_ != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(ReDev(*this), ImGuiPool_, nullptr);
		ImGuiPool_ = VK_NULL_HANDLE;
	}
}

void OaPresenter::BeginImGuiFrame() {
#ifdef OA_IMGUI
	assert(ImGuiReady_ && "call InitImGui before BeginImGuiFrame");
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL3_NewFrame();
	ImGui::NewFrame();
	// Also call ImGui::DockSpaceOverViewport() here if you want a dockspace by default.
#endif
}

void OaPresenter::EndImGuiFrame() {
#ifdef OA_IMGUI
	assert(ImGuiReady_ && "call InitImGui before EndImGuiFrame");
	ImGui::Render();
#endif
}


// ─── NotifyPixelSizeChanged ──────────────────────────────────────────────────

void OaPresenter::NotifyPixelSizeChanged(int InWidthPx, int InHeightPx) {
	Swapchain_.Dirty     = true;
	Swapchain_.DirtySize = {
		static_cast<uint32_t>(InWidthPx > 0 ? InWidthPx : 0),
		static_cast<uint32_t>(InHeightPx > 0 ? InHeightPx : 0)
	};
}

bool OaPresenter::UsesMergedGraphicsComputeQueue() const {
	return Engine_.Device.Queues.ComputeQueue != nullptr
		and Engine_.Device.Queues.GraphicsQueue == Engine_.Device.Queues.ComputeQueue;
}

OaStatus OaPresenter::WaitPresentationIdle() {
	if (!Engine_.Device.Device) return OaStatus::Ok();
	// A present fence retires both presentation-engine access to the swapchain
	// resources and its consumption of RenderDoneSem. This is the exact WSI
	// lifetime boundary; render-submission fences alone cannot prove either.
	if (Engine_.Device.Info.Software.HasSwapchainMaintenance1
		&& !Swapchain_.PresentCompletionUncertain
		&& !Swapchain_.PresentFence.empty()) {
		std::vector<VkFence> pending;
		pending.reserve(Swapchain_.PresentFence.size());
		for (OaUsize i = 0; i < Swapchain_.PresentFence.size(); ++i) {
			if (i < Swapchain_.PresentFencePending.size()
				&& Swapchain_.PresentFencePending[i]) {
				pending.push_back(Swapchain_.PresentFence[i]);
			}
		}
		if (!pending.empty()) {
			const VkResult result = vkWaitForFences(
				ReDev(*this), static_cast<OaU32>(pending.size()), pending.data(),
				VK_TRUE, UINT64_MAX);
			if (result != VK_SUCCESS) {
				return OaStatus::Error(OaStatusCode::VulkanError,
					OaString("presentation fence wait failed: VkResult=")
						+ std::to_string(static_cast<int>(result)));
			}
			std::fill(Swapchain_.PresentFencePending.begin(),
				Swapchain_.PresentFencePending.end(), false);
		}
		return OaStatus::Ok();
	}

	// Compatibility and error-recovery fallback. This is reached only when the
	// optional maintenance extension is unavailable or vkQueuePresentKHR failed
	// before its fence ownership became certain.
	void* queueHandle = Engine_.Device.Queues.PresentQueue != nullptr
		? Engine_.Device.Queues.PresentQueue
		: Engine_.Device.Queues.GraphicsQueue;
	if (queueHandle == nullptr) return OaStatus::Ok();
	LockSharedQueueSubmit(queueHandle);
	const VkResult result = vkQueueWaitIdle(static_cast<VkQueue>(queueHandle));
	UnlockSharedQueueSubmit(queueHandle);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			OaString("presentation queue wait failed: VkResult=")
				+ std::to_string(static_cast<int>(result)));
	}
	std::fill(Swapchain_.PresentFencePending.begin(),
		Swapchain_.PresentFencePending.end(), false);
	Swapchain_.PresentCompletionUncertain = false;
	return OaStatus::Ok();
}

OaStatus OaPresenter::PreparePresentFence(
	OaSwapchain& InSwap, OaU32 InFrameSlot) {
	if (!Engine_.Device.Info.Software.HasSwapchainMaintenance1) return OaStatus::Ok();
	if (InSwap.PresentCompletionUncertain) {
		// A failed present does not establish the normal fence-retirement
		// guarantee. Resolve that exceptional state before reusing any fence or
		// binary semaphore from the presentation ring.
		if (const auto status = WaitPresentationIdle(); !status.IsOk()) {
			return status;
		}
	}
	if (InFrameSlot >= InSwap.PresentFence.size()
		|| InFrameSlot >= InSwap.PresentFencePending.size()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"presentation fence ring is not initialized");
	}
	VkFence fence = InSwap.PresentFence[InFrameSlot];
	if (InSwap.PresentFencePending[InFrameSlot]) {
		const VkResult wait = vkWaitForFences(
			ReDev(*this), 1, &fence, VK_TRUE, UINT64_MAX);
		if (wait != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				OaString("presentation fence reuse wait failed: VkResult=")
					+ std::to_string(static_cast<int>(wait)));
		}
	}
	const VkResult reset = vkResetFences(ReDev(*this), 1, &fence);
	if (reset != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			OaString("presentation fence reset failed: VkResult=")
				+ std::to_string(static_cast<int>(reset)));
	}
	InSwap.PresentFencePending[InFrameSlot] = false;
	return OaStatus::Ok();
}

void OaPresenter::FinishPresent(
	OaSwapchain& InSwap, OaU32 InFrameSlot, VkResult InResult) {
	if (!Engine_.Device.Info.Software.HasSwapchainMaintenance1
		|| InFrameSlot >= InSwap.PresentFencePending.size()) return;
	if (InResult == VK_SUCCESS || InResult == VK_SUBOPTIMAL_KHR) {
		InSwap.PresentFencePending[InFrameSlot] = true;
	} else {
		// The spec only gives the fence a useful lifetime guarantee for a queued
		// presentation. Preserve correctness on failure with the narrow fallback.
		InSwap.PresentCompletionUncertain = true;
	}
}

void OaPresenter::LockSharedQueueSubmit(void* InQueue) {
	if (not UsesMergedGraphicsComputeQueue() or InQueue == nullptr) {
		return;
	}
	const void* const qv = InQueue;
	if (qv != Engine_.Device.Queues.ComputeQueue
		and qv != Engine_.Device.Queues.GraphicsQueue
		and qv != Engine_.Device.Queues.PresentQueue) {
		return;
	}
	Engine_.ComputeQueueMutex_.lock();
}

void OaPresenter::UnlockSharedQueueSubmit(void* InQueue) {
	if (not UsesMergedGraphicsComputeQueue() or InQueue == nullptr) {
		return;
	}
	const void* const qv = InQueue;
	if (qv != Engine_.Device.Queues.ComputeQueue
		and qv != Engine_.Device.Queues.GraphicsQueue
		and qv != Engine_.Device.Queues.PresentQueue) {
		return;
	}
	Engine_.ComputeQueueMutex_.unlock();
}

void OaEngine::RetirePresenter(OaUniquePtr<OaRetiredPresenter>&& InPresenter) {
	if (not InPresenter) return;
	OaScopedLock<OaMutex> lock(RetiredPresenterMutex_);
	RetiredPresenters_.PushBack(OaStdMove(InPresenter));
}

OaStatus OaEngine::CompleteRetiredPresenters_() {
	OaStatus result = OaStatus::Ok();
	OaVec<OaUniquePtr<OaRetiredPresenter>> retired;
	{
		OaScopedLock<OaMutex> lock(RetiredPresenterMutex_);
		retired = OaStdMove(RetiredPresenters_);
	}

	OaVec<OaUniquePtr<OaRetiredPresenter>> pending;
	for (auto& presenter : retired) {
		OaStatus presenterStatus = OaStatus::Ok();
		const auto retainError = [&presenterStatus](const OaStatus& InStatus) {
			if (presenterStatus.IsOk() && !InStatus.IsOk()) {
				presenterStatus = InStatus;
			}
		};

		for (auto& stream : presenter->GraphicsStreams) {
			if (stream && stream->Submitted) {
				retainError(stream->Synchronize(Device));
			}
		}

		if (presenter->Swapchain.PresentReady) {
			if (presenter->HasSwapchainMaintenance1
				&& !presenter->Swapchain.PresentCompletionUncertain
				&& !presenter->Swapchain.PresentFence.empty()) {
				std::vector<VkFence> fences;
				for (OaUsize i = 0;
					i < presenter->Swapchain.PresentFence.size(); ++i) {
					if (i < presenter->Swapchain.PresentFencePending.size()
						&& presenter->Swapchain.PresentFencePending[i]) {
						fences.push_back(presenter->Swapchain.PresentFence[i]);
					}
				}
				if (!fences.empty()) {
					const VkResult wait = vkWaitForFences(
						static_cast<VkDevice>(Device.Device),
						static_cast<OaU32>(fences.size()), fences.data(),
						VK_TRUE, UINT64_MAX);
					if (wait != VK_SUCCESS) {
						retainError(OaStatus::Error(
							OaStatusCode::VulkanError,
							OaString("retired presentation fence wait failed: VkResult=")
								+ std::to_string(static_cast<int>(wait))));
					}
				}
			} else if (presenter->PresentQueue != nullptr) {
				if (presenter->UsesMergedGraphicsComputeQueue) {
					ComputeQueueMutex_.lock();
				}
				const VkResult wait = vkQueueWaitIdle(
					static_cast<VkQueue>(presenter->PresentQueue));
				if (presenter->UsesMergedGraphicsComputeQueue) {
					ComputeQueueMutex_.unlock();
				}
				if (wait != VK_SUCCESS) {
					retainError(OaStatus::Error(
						OaStatusCode::VulkanError,
						OaString("retired presentation queue wait failed: VkResult=")
							+ std::to_string(static_cast<int>(wait))));
				}
			}
		}

		if (!presenterStatus.IsOk()) {
			if (result.IsOk()) result = presenterStatus;
			pending.PushBack(OaStdMove(presenter));
			continue;
		}

#ifdef OA_IMGUI
		if (presenter->ImGuiReady) {
			ImGui_ImplVulkan_Shutdown();
			ImGui_ImplSDL3_Shutdown();
			ImGui::DestroyContext();
			presenter->ImGuiReady = false;
		}
#endif
		VkDevice device = static_cast<VkDevice>(Device.Device);
		if (presenter->ImGuiPool != VK_NULL_HANDLE) {
			vkDestroyDescriptorPool(device, presenter->ImGuiPool, nullptr);
			presenter->ImGuiPool = VK_NULL_HANDLE;
		}
		for (auto& stream : presenter->GraphicsStreams) {
			if (stream) stream->Destroy(Device);
		}
		for (auto semaphore : presenter->Swapchain.ImageAvailSem) {
			if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
		}
		for (auto semaphore : presenter->Swapchain.RenderDoneSem) {
			if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
		}
		for (auto fence : presenter->Swapchain.InFlightFence) {
			if (fence) vkDestroyFence(device, fence, nullptr);
		}
		for (auto fence : presenter->Swapchain.PresentFence) {
			if (fence) vkDestroyFence(device, fence, nullptr);
		}
		if (presenter->CommandPool != VK_NULL_HANDLE) {
			if (!presenter->CommandBuffers.empty()) {
				vkFreeCommandBuffers(
					device, presenter->CommandPool,
					static_cast<OaU32>(presenter->CommandBuffers.size()),
					presenter->CommandBuffers.data());
			}
			vkDestroyCommandPool(device, presenter->CommandPool, nullptr);
		}
		for (auto framebuffer : presenter->Framebuffers) {
			if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		for (auto view : presenter->Swapchain.Views) {
			if (view) vkDestroyImageView(device, view, nullptr);
		}
		if (presenter->RenderPass != VK_NULL_HANDLE) {
			vkDestroyRenderPass(device, presenter->RenderPass, nullptr);
		}
		if (presenter->Swapchain.Handle != VK_NULL_HANDLE) {
			vkDestroySwapchainKHR(device, presenter->Swapchain.Handle, nullptr);
		}
		if (presenter->OwnsAbandonedSurface
			&& presenter->Swapchain.Surface != nullptr) {
			vkDestroySurfaceKHR(
				static_cast<VkInstance>(Device.Instance),
				static_cast<VkSurfaceKHR>(presenter->Swapchain.Surface), nullptr);
		}
	}

	if (!pending.Empty()) {
		OaScopedLock<OaMutex> lock(RetiredPresenterMutex_);
		for (auto& presenter : pending) {
			RetiredPresenters_.PushBack(OaStdMove(presenter));
		}
	}
	return result;
}

// ─── DrawFrame ───────────────────────────────────────────────────────────────

bool OaPresenter::DrawFrame() {
	if (!Swapchain_.PresentReady || Swapchain_.Handle == VK_NULL_HANDLE) return false;

	// Proactive swapchain recreate when the window reported a pixel size change.
	if (Swapchain_.Dirty) {
		Swapchain_.Dirty = false;
		if (Swapchain_.DirtySize.width > 0 and Swapchain_.DirtySize.height > 0) {
			if (not RecreateSwapchain(Swapchain_.DirtySize)) return false;
		}
	}

	// Skip if window is minimised / zero-sized (SDL3 / Wayland may report 0).
	if (Swapchain_.Extent.width == 0 || Swapchain_.Extent.height == 0) return true;

	if (const auto status = PreparePresentFence(
		Swapchain_, static_cast<OaU32>(Swapchain_.FrameIndex)); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core, "DrawFrame: %s", status.ToString().c_str());
		return false;
	}

	vkWaitForFences(ReDev(*this), 1, &Swapchain_.InFlightFence[Swapchain_.FrameIndex], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex = 0;
	VkResult acq = vkAcquireNextImageKHR(
		ReDev(*this), Swapchain_.Handle, UINT64_MAX,
		Swapchain_.ImageAvailSem[Swapchain_.FrameIndex], VK_NULL_HANDLE,
		&imageIndex);

	if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
		return RecreateSwapchain(Swapchain_.Extent);
	}
	if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return false;

	vkResetFences(ReDev(*this), 1, &Swapchain_.InFlightFence[Swapchain_.FrameIndex]);

	VkCommandBuffer cmd = CmdBufs_[Swapchain_.FrameIndex];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(cmd, &bi);

	VkClearValue clearColor{};
	clearColor.color = {{0.12f, 0.12f, 0.12f, 1.0f}};   // dark editor grey

	VkRenderPassBeginInfo rpBi{};
	rpBi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpBi.renderPass        = RenderPass_;
	rpBi.framebuffer       = Framebuffers_[imageIndex];
	rpBi.renderArea.offset = {0, 0};
	rpBi.renderArea.extent = Swapchain_.Extent;
	rpBi.clearValueCount   = 1;
	rpBi.pClearValues      = &clearColor;

	vkCmdBeginRenderPass(cmd, &rpBi, VK_SUBPASS_CONTENTS_INLINE);

#ifdef OA_IMGUI
	if (ImGuiReady_) {
		ImDrawData* drawData = ImGui::GetDrawData();
		if (drawData) {
			ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
		}
	}
#endif

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);

	// Same VkQueue as compute: serialize with SubmitToQueue + Dear ImGui texture submits.
	OaVkSharedQueueSubmitScope sharedQueueSubmitScope(this);

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo si{};
	si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.waitSemaphoreCount   = 1;
	si.pWaitSemaphores      = &Swapchain_.ImageAvailSem[Swapchain_.FrameIndex];
	si.pWaitDstStageMask    = &waitStage;
	si.commandBufferCount   = 1;
	si.pCommandBuffers      = &cmd;
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores    = &Swapchain_.RenderDoneSem[Swapchain_.FrameIndex];

	const VkResult submitRes = vkQueueSubmit(
		static_cast<VkQueue>(Engine_.Device.Queues.GraphicsQueue),
		1, &si, Swapchain_.InFlightFence[Swapchain_.FrameIndex]);
	if (submitRes != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkQueueSubmit failed (VkResult=%d)",
			static_cast<int>(submitRes));
		return false;
	}

	VkPresentInfoKHR pi{};
	pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores    = &Swapchain_.RenderDoneSem[Swapchain_.FrameIndex];
	pi.swapchainCount     = 1;
	pi.pSwapchains        = &Swapchain_.Handle;
	pi.pImageIndices      = &imageIndex;
	VkSwapchainPresentFenceInfoKHR presentFenceInfo{};
	if (Engine_.Device.Info.Software.HasSwapchainMaintenance1) {
		presentFenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
		presentFenceInfo.swapchainCount = 1;
		presentFenceInfo.pFences =
			&Swapchain_.PresentFence[Swapchain_.FrameIndex];
		pi.pNext = &presentFenceInfo;
	}

	VkResult pres = vkQueuePresentKHR(
		static_cast<VkQueue>(Engine_.Device.Queues.PresentQueue), &pi);
	FinishPresent(Swapchain_, static_cast<OaU32>(Swapchain_.FrameIndex), pres);

	Swapchain_.FrameIndex = (Swapchain_.FrameIndex + 1) % OaSwapchain::kFramesInFlight;

	if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
		if (!RecreateSwapchain(Swapchain_.Extent)) {
			return false;
		}
		return true;
	}
	if (pres != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core, "vkQueuePresentKHR failed (VkResult=%d)",
			static_cast<int>(pres));
		return false;
	}
	return true;
}


// ─── Ctx-mediated present primitives (Step 3b.3) ────────────────────────────
//
// Split of DrawFrame's body into AcquireSwapchainImage + PresentSwapchainImage.
// OaViewer uses these directly; DrawFrame remains the compact one-call path.

bool OaPresenter::AcquireSwapchainImage(
	OaSwapchain& InSwap, AcquireResult& OutResult) {
	OutResult = AcquireResult{};

	if (not InSwap.PresentReady or InSwap.Handle == VK_NULL_HANDLE) {
		return false;
	}

	// Proactive recreate on signalled resize.
	if (InSwap.Dirty) {
		InSwap.Dirty = false;
		if (InSwap.DirtySize.width > 0 and InSwap.DirtySize.height > 0) {
			if (not RecreateSwapchain(InSwap.DirtySize)) return false;
			OutResult.Recreated = true;
			return true;
		}
	}

	if (InSwap.Extent.width == 0 or InSwap.Extent.height == 0) {
		// Zero-size window — nothing to acquire. Caller should skip this frame.
		return false;
	}

	const OaU32 frameSlot = static_cast<OaU32>(InSwap.FrameIndex);
	if (const auto status = PreparePresentFence(InSwap, frameSlot); !status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"AcquireSwapchainImage: %s", status.ToString().c_str());
		return false;
	}

	vkWaitForFences(ReDev(*this), 1, &InSwap.InFlightFence[frameSlot],
		VK_TRUE, UINT64_MAX);

	uint32_t imageIndex = 0;
	const VkResult acq = vkAcquireNextImageKHR(
		ReDev(*this), InSwap.Handle, UINT64_MAX,
		InSwap.ImageAvailSem[frameSlot], VK_NULL_HANDLE, &imageIndex);

	if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
		if (not RecreateSwapchain(InSwap.Extent)) return false;
		OutResult.Recreated = true;
		return true;
	}
	if (acq != VK_SUCCESS and acq != VK_SUBOPTIMAL_KHR) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"AcquireSwapchainImage: vkAcquireNextImageKHR failed (VkResult=%d)",
			static_cast<int>(acq));
		return false;
	}

	vkResetFences(ReDev(*this), 1, &InSwap.InFlightFence[frameSlot]);

	OutResult.FrameSlot  = frameSlot;
	OutResult.ImageIndex = imageIndex;
	OutResult.Image      = InSwap.Images[imageIndex];
	OutResult.View       = InSwap.Views[imageIndex];
	OutResult.Recreated  = false;
	return true;
}

bool OaPresenter::PresentSwapchainImage(
	OaSwapchain&       InSwap,
	OaU32              InImageIndex,
	OaU32              InFrameSlot,
	const PresentArgs& InArgs) {
	if (not InSwap.PresentReady or InSwap.Handle == VK_NULL_HANDLE) return false;
	if (InImageIndex >= InSwap.Images.size())                       return false;

	const bool hasImGui     = InArgs.DrawImGui;
	const bool hasBlitImage = (not hasImGui) and InArgs.BlitSrcImage  != nullptr;
	const bool hasBlitBuf   = (not hasImGui) and (not hasBlitImage) and InArgs.BlitSrcBuffer != nullptr;
	const bool hasClear     = (not hasImGui) and (not hasBlitImage) and (not hasBlitBuf)
		and InArgs.ClearRgba != nullptr;
	const bool toTransferDst = hasBlitImage or hasBlitBuf or hasClear;

	// ImGui-only path (Step 3b.5): use the engine's render pass +
	// framebuffer[ImageIndex], same machinery DrawFrame uses. The render
	// pass's loadOp=CLEAR provides the clear; finalLayout=PRESENT_SRC
	// transitions for us. Blit+ImGui composite needs a separate render pass
	// with loadOp=LOAD and lands in Step 3c when DrawFrame's body is
	// replaced wholesale by explicit present.
	if (hasImGui) {
#ifdef OA_IMGUI
		if (not ImGuiReady_) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"PresentSwapchainImage: DrawImGui set but ImGui not initialised");
			return false;
		}

		VkCommandBuffer cmd = CmdBufs_[InFrameSlot];
		vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkBeginCommandBuffer(cmd, &bi);

		VkClearValue clearColor{};
		clearColor.color = {{0.12F, 0.12F, 0.12F, 1.0F}};  // matches DrawFrame's dark grey
		VkRenderPassBeginInfo rpBi{};
		rpBi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpBi.renderPass        = RenderPass_;
		rpBi.framebuffer       = Framebuffers_[InImageIndex];
		rpBi.renderArea.offset = {0, 0};
		rpBi.renderArea.extent = InSwap.Extent;
		rpBi.clearValueCount   = 1;
		rpBi.pClearValues      = &clearColor;
		vkCmdBeginRenderPass(cmd, &rpBi, VK_SUBPASS_CONTENTS_INLINE);

		ImDrawData* drawData = ImGui::GetDrawData();
		if (drawData != nullptr) {
			ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);

		// Submit + present — same semaphore chain as the transfer paths below.
		OaVkSharedQueueSubmitScope sharedQueueSubmitScopeImgui(this);
		VkSemaphore waitSemaphoresImgui[2] = {
			InSwap.ImageAvailSem[InFrameSlot],
			static_cast<VkSemaphore>(InArgs.WaitTimelineSemaphore),
		};
		VkPipelineStageFlags waitStagesImgui[2] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		};
		OaU64 waitValuesImgui[2] = {0, InArgs.WaitTimelineValue};
		OaU64 signalValuesImgui[1] = {0};
		const OaU32 waitCountImgui = InArgs.WaitTimelineSemaphore != nullptr ? 2U : 1U;
		VkTimelineSemaphoreSubmitInfo timelineInfoImgui{};
		timelineInfoImgui.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timelineInfoImgui.waitSemaphoreValueCount = waitCountImgui;
		timelineInfoImgui.pWaitSemaphoreValues = waitValuesImgui;
		timelineInfoImgui.signalSemaphoreValueCount = 1;
		timelineInfoImgui.pSignalSemaphoreValues = signalValuesImgui;
		VkSubmitInfo siImgui{};
		siImgui.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		siImgui.pNext                = InArgs.WaitTimelineSemaphore != nullptr
			? &timelineInfoImgui : nullptr;
		siImgui.waitSemaphoreCount   = waitCountImgui;
		siImgui.pWaitSemaphores      = waitSemaphoresImgui;
		siImgui.pWaitDstStageMask    = waitStagesImgui;
		siImgui.commandBufferCount   = 1;
		siImgui.pCommandBuffers      = &cmd;
		siImgui.signalSemaphoreCount = 1;
		siImgui.pSignalSemaphores    = &InSwap.RenderDoneSem[InFrameSlot];
		const VkResult subImgui = vkQueueSubmit(
			static_cast<VkQueue>(Engine_.Device.Queues.GraphicsQueue),
			1, &siImgui, InSwap.InFlightFence[InFrameSlot]);
		if (subImgui != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"PresentSwapchainImage(ImGui): vkQueueSubmit failed (VkResult=%d)",
				static_cast<int>(subImgui));
			return false;
		}
		VkPresentInfoKHR piImgui{};
		piImgui.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		piImgui.waitSemaphoreCount = 1;
		piImgui.pWaitSemaphores    = &InSwap.RenderDoneSem[InFrameSlot];
		piImgui.swapchainCount     = 1;
		piImgui.pSwapchains        = &InSwap.Handle;
		piImgui.pImageIndices      = &InImageIndex;
		VkSwapchainPresentFenceInfoKHR presentFenceInfoImgui{};
		if (Engine_.Device.Info.Software.HasSwapchainMaintenance1) {
			presentFenceInfoImgui.sType =
				VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
			presentFenceInfoImgui.swapchainCount = 1;
			presentFenceInfoImgui.pFences = &InSwap.PresentFence[InFrameSlot];
			piImgui.pNext = &presentFenceInfoImgui;
		}
		const VkResult presImgui = vkQueuePresentKHR(
			static_cast<VkQueue>(Engine_.Device.Queues.PresentQueue), &piImgui);
		FinishPresent(InSwap, InFrameSlot, presImgui);
		InSwap.FrameIndex = (InSwap.FrameIndex + 1) % OaSwapchain::kFramesInFlight;
		if (presImgui == VK_ERROR_OUT_OF_DATE_KHR or presImgui == VK_SUBOPTIMAL_KHR) {
			return RecreateSwapchain(InSwap.Extent);
		}
		if (presImgui != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"PresentSwapchainImage(ImGui): vkQueuePresentKHR failed (VkResult=%d)",
				static_cast<int>(presImgui));
			return false;
		}
		return true;
#else
		OA_LOG_ERROR(OaLogComponent::Core,
			"PresentSwapchainImage: DrawImGui set but OA_IMGUI not compiled in");
		return false;
#endif
	}

	VkCommandBuffer cmd = CmdBufs_[InFrameSlot];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(cmd, &bi);

	VkImage swapImg = InSwap.Images[InImageIndex];

	// First barrier: UNDEFINED → (TRANSFER_DST if blit/clear, else PRESENT_SRC).
	{
		VkImageMemoryBarrier b{};
		b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		b.newLayout           = toTransferDst
			? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			: VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		b.srcAccessMask       = 0;
		b.dstAccessMask       = toTransferDst ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = swapImg;
		b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdPipelineBarrier(cmd,
			// The acquire semaphore is waited at the first stage that touches the
			// image. Include that same stage on the layout transition's source
			// side so synchronization validation can connect acquire to the write.
			toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
				: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
				: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b);
	}

	if (hasBlitImage) {
		// VkImage→VkImage path (Step 3c.1): vkCmdBlitImage handles format
		// conversion (compose RGBA8 → swap BGRA8_SRGB) and scaling. Source
		// image current layout supplied via InArgs.BlitSrcLayout; we
		// transition it to TRANSFER_SRC_OPTIMAL for the blit, then back.
		const VkImage srcImg = static_cast<VkImage>(InArgs.BlitSrcImage);
		const auto    srcLayoutIn  = static_cast<VkImageLayout>(InArgs.BlitSrcLayout);

		VkImageMemoryBarrier srcIn{};
		srcIn.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		srcIn.oldLayout           = srcLayoutIn;
		srcIn.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		srcIn.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
		srcIn.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		srcIn.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		srcIn.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		srcIn.image               = srcImg;
		srcIn.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &srcIn);

		VkImageBlit region{};
		region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.srcOffsets[0]  = { 0, 0, 0 };
		region.srcOffsets[1]  = {
			static_cast<int32_t>(InArgs.BlitSrcWidth),
			static_cast<int32_t>(InArgs.BlitSrcHeight),
			1
		};
		region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.dstOffsets[0]  = { 0, 0, 0 };
		region.dstOffsets[1]  = {
			static_cast<int32_t>(InSwap.Extent.width),
			static_cast<int32_t>(InSwap.Extent.height),
			1
		};
		const VkFilter vkFilter = (InArgs.Filter == OaFilter::Nearest)
			? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
		vkCmdBlitImage(cmd,
			srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			swapImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &region, vkFilter);

		// Restore source layout so the next frame's compute writes don't
		// race on a stale layout.
		VkImageMemoryBarrier srcOut{};
		srcOut.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		srcOut.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		srcOut.newLayout           = srcLayoutIn;
		srcOut.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
		srcOut.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
		srcOut.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		srcOut.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		srcOut.image               = srcImg;
		srcOut.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &srcOut);
	} else if (hasBlitBuf) {
		// vkCmdCopyBufferToImage: assumes the source buffer holds packed RGBA8
		// at the same extent as the swap image. Format mismatch (swap is
		// BGRA8_SRGB today) is a known limitation — channel swap and
		// linear→sRGB are tracked for the VkImage path (BlitSrcImage above).
		VkBufferImageCopy copy{};
		copy.bufferOffset      = 0;
		copy.bufferRowLength   = 0;  // tightly packed
		copy.bufferImageHeight = 0;
		copy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		copy.imageSubresource.mipLevel       = 0;
		copy.imageSubresource.baseArrayLayer = 0;
		copy.imageSubresource.layerCount     = 1;
		copy.imageOffset = { 0, 0, 0 };
		copy.imageExtent = { InArgs.BlitSrcWidth, InArgs.BlitSrcHeight, 1 };
		vkCmdCopyBufferToImage(cmd,
			static_cast<VkBuffer>(InArgs.BlitSrcBuffer),
			swapImg,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copy);
	} else if (hasClear) {
		VkClearColorValue clear{};
		clear.float32[0] = InArgs.ClearRgba[0];
		clear.float32[1] = InArgs.ClearRgba[1];
		clear.float32[2] = InArgs.ClearRgba[2];
		clear.float32[3] = InArgs.ClearRgba[3];
		VkImageSubresourceRange sub{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdClearColorImage(cmd, swapImg,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &sub);
	}

	if (toTransferDst) {
		// Second barrier: TRANSFER_DST → PRESENT_SRC.
		VkImageMemoryBarrier b{};
		b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		b.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
		b.dstAccessMask       = 0;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = swapImg;
		b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b);
	}

	vkEndCommandBuffer(cmd);

	// Submit on graphics queue. Wait on ImageAvail, signal RenderDone,
	// fence = InFlight[FrameSlot] — same model DrawFrame uses.
	OaVkSharedQueueSubmitScope sharedQueueSubmitScope(this);

	VkSemaphore waitSemaphores[2] = {
		InSwap.ImageAvailSem[InFrameSlot],
		static_cast<VkSemaphore>(InArgs.WaitTimelineSemaphore),
	};
	VkPipelineStageFlags waitStages[2] = {
		toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
			: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	};
	OaU64 waitValues[2] = {0, InArgs.WaitTimelineValue};
	OaU64 signalValues[1] = {0};
	const OaU32 waitCount = InArgs.WaitTimelineSemaphore != nullptr ? 2U : 1U;
	VkTimelineSemaphoreSubmitInfo timelineInfo{};
	timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timelineInfo.waitSemaphoreValueCount = waitCount;
	timelineInfo.pWaitSemaphoreValues = waitValues;
	timelineInfo.signalSemaphoreValueCount = 1;
	timelineInfo.pSignalSemaphoreValues = signalValues;
	VkSubmitInfo si{};
	si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.pNext                = InArgs.WaitTimelineSemaphore != nullptr
		? &timelineInfo : nullptr;
	si.waitSemaphoreCount   = waitCount;
	si.pWaitSemaphores      = waitSemaphores;
	si.pWaitDstStageMask    = waitStages;
	si.commandBufferCount   = 1;
	si.pCommandBuffers      = &cmd;
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores    = &InSwap.RenderDoneSem[InFrameSlot];

	const VkResult submitRes = vkQueueSubmit(
		static_cast<VkQueue>(Engine_.Device.Queues.GraphicsQueue),
		1, &si, InSwap.InFlightFence[InFrameSlot]);
	if (submitRes != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"PresentSwapchainImage: vkQueueSubmit failed (VkResult=%d)",
			static_cast<int>(submitRes));
		return false;
	}

	VkPresentInfoKHR pi{};
	pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores    = &InSwap.RenderDoneSem[InFrameSlot];
	pi.swapchainCount     = 1;
	pi.pSwapchains        = &InSwap.Handle;
	pi.pImageIndices      = &InImageIndex;
	VkSwapchainPresentFenceInfoKHR presentFenceInfo{};
	if (Engine_.Device.Info.Software.HasSwapchainMaintenance1) {
		presentFenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
		presentFenceInfo.swapchainCount = 1;
		presentFenceInfo.pFences = &InSwap.PresentFence[InFrameSlot];
		pi.pNext = &presentFenceInfo;
	}

	const VkResult pres = vkQueuePresentKHR(
		static_cast<VkQueue>(Engine_.Device.Queues.PresentQueue), &pi);
	FinishPresent(InSwap, InFrameSlot, pres);

	InSwap.FrameIndex = (InSwap.FrameIndex + 1) % OaSwapchain::kFramesInFlight;

	if (pres == VK_ERROR_OUT_OF_DATE_KHR or pres == VK_SUBOPTIMAL_KHR) {
		return RecreateSwapchain(InSwap.Extent);
	}
	if (pres != VK_SUCCESS) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"PresentSwapchainImage: vkQueuePresentKHR failed (VkResult=%d)",
			static_cast<int>(pres));
		return false;
	}
	return true;
}
