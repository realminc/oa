// oa::Presenter — WSI swapchain + optional Dear ImGui on top of oa::Engine.
//
// SDL3 backend.  Build with -DOA_IMGUI to enable ImGui integration.
// Without OA_IMGUI all ImGui-related methods compile as no-ops so the codebase
// stays clean while the UI layer is being wired up.
//
// When OA_IMGUI is defined you also need to link:
//   imgui  imgui_impl_vulkan  imgui_impl_sdl3
// and provide SDL3/SDL.h + backends/imgui_impl_sdl3.h in your include path.

#include <oa/runtime/presenter.h>
#include <oa/runtime/eventAccess.h>
#include "engineAccess.h"
#include "deviceAccess.h"
#include <oa/runtime/window.h>
#include <oa/runtime/oaVk.h>
#include <oa/runtime/device.h>
#include <oa/core/log.h>

#include "../presenterRetirement.h"

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

struct QueueSubmitScope {
	oa::Presenter* eng_ = nullptr;
	void* queue_ = nullptr;
	QueueSubmitScope(oa::Presenter* inEng, void* inQueue)
		: eng_(inEng), queue_(inQueue) {
		if (eng_ and queue_) eng_->lockSharedQueueSubmit(queue_);
	}
	~QueueSubmitScope() {
		if (eng_ and queue_) eng_->unlockSharedQueueSubmit(queue_);
	}
};

static inline VkDevice reDev(const oa::Presenter& E) {
	return static_cast<VkDevice>(
		oa::EngineDeviceAccess::get(E.engine()).device);
}
static inline const OaVkDeviceTable& reVk(const oa::Presenter& E) {
	return oa::EngineDeviceAccess::get(E.engine()).deviceDispatch;
}
static inline const OaVkInstanceTable& reInstanceVk(const oa::Presenter& E) {
	return oa::EngineDeviceAccess::get(E.engine()).instanceDispatch;
}
static inline VkPhysicalDevice rePhys(const oa::Presenter& E) {
	return static_cast<VkPhysicalDevice>(
		oa::EngineDeviceAccess::get(E.engine()).physicalDevice);
}

oa::Result<void*> oa::Presenter::createSurface(const oa::VulkanWindow& inWindow) const {
	if (not engine_.isReady() or oa::EngineDeviceAccess::get(engine_).instance == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Presenter::createSurface requires a live engine");
	}
	if (not hasGraphics()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Presenter::createSurface requires a graphics-capable engine");
	}
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (not inWindow.createPresenterVkSurface(
		static_cast<VkInstance>(oa::EngineDeviceAccess::get(engine_).instance), &surface)
		or surface == VK_NULL_HANDLE)
	{
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"oa::Presenter::createSurface window backend failed");
	}
	return static_cast<void*>(surface);
}

oa::Status oa::Presenter::destroySurface(void*& inOutSurface) const {
	if (inOutSurface == nullptr) return oa::Status::ok();
	if (swapchain_.surface == inOutSurface) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Presenter::destroySurface requires detached presentation");
	}
	if (not engine_.isReady() or oa::EngineDeviceAccess::get(engine_).instance == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::Presenter::destroySurface requires a live engine");
	}
	oa::EngineDeviceAccess::get(engine_).instanceDispatch.vkDestroySurfaceKHR(
		static_cast<VkInstance>(oa::EngineDeviceAccess::get(engine_).instance),
		static_cast<VkSurfaceKHR>(inOutSurface),
		nullptr);
	inOutSurface = nullptr;
	return oa::Status::ok();
}


// ─── move / copy ─────────────────────────────────────────────────────────────

// oa::Presenter is pinned (move/copy = delete in the header) and borrows its
// engine. It never relocates,
// so the swapchain/render-pass/command-pool/ImGui handles it owns can't be
// aliased into a moved-from twin. (The old hand-written move ctor/assignment that
// reset the source's swapchain_/renderPass_/cmdPool_/imGuiPool_ to dodge
// double-destroy is gone.)

oa::Presenter::~Presenter() {
	abandon_();
}

bool oa::Presenter::hasOwnedState_() const noexcept {
	oa::Bool hasBatchState = graphicsBatchStream_ != nullptr;
	for (const auto& stream : graphicsBatchRing_) {
		hasBatchState = hasBatchState || stream != nullptr;
	}
	const oa::Bool hasPresentationState =
		swapchain_.surface != VK_NULL_HANDLE ||
		swapchain_.handle != VK_NULL_HANDLE ||
		renderPass_ != VK_NULL_HANDLE ||
		cmdPool_ != VK_NULL_HANDLE ||
		imGuiPool_ != VK_NULL_HANDLE ||
		!framebuffers_.empty() || !cmdBufs_.empty();
	return hasBatchState || hasPresentationState;
}

oa::Status oa::Presenter::close() {
	if (!hasOwnedState_()) return oa::Status::ok();
	if (!oa::EngineDeviceAccess::get(engine_).device) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Presenter::close: borrowed engine is not live");
	}
	// A presenter is cheap to construct as a borrowed service. If it was never
	// initialized, close must not turn into an unrelated graphics-queue wait.
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() && !inStatus.isOk()) firstError = inStatus;
	};
	if (graphicsBatchStream_) {
		const auto resetStatus =
			graphicsBatchStream_->resetUnsubmitted(oa::EngineDeviceAccess::get(engine_));
		retainError(resetStatus);
		if (resetStatus.isOk()) graphicsBatchStream_ = nullptr;
	}
	for (oa::U32 i = 0; i < kGraphicsBatchRingSize; ++i) {
		if (graphicsBatchRing_[i] && graphicsBatchRing_[i]->submitted) {
			retainError(graphicsBatchRing_[i]->synchronize(oa::EngineDeviceAccess::get(engine_)));
		}
	}
	// Render fences do not prove that presentation has retired. drain only the
	// WSI queue; compute, transfer, and video queues remain independent.
	if (swapchain_.presentReady) retainError(waitPresentationIdle());
	if (!firstError.isOk()) return firstError;

	graphicsBatchStream_ = nullptr;
	for (oa::U32 i = 0; i < kGraphicsBatchRingSize; ++i) {
		if (graphicsBatchRing_[i]) {
			graphicsBatchRing_[i]->destroy(oa::EngineDeviceAccess::get(engine_));
			graphicsBatchRing_[i].reset();
		}
	}
	graphicsBatchRingIndex_ = 0;
	shutdownImGuiResources_();
	destroySyncObjects();
	destroyCommandPool();
	destroySwapchainObjects();
	oa::EngineDeviceAccess::get(engine_).queues.hasPresentation = false;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueue = nullptr;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily = oavk::EnumerationIndexUnset;
	swapchain_ = {};
	// Note: swapchain_.surface is caller-owned — do NOT destroy it here.
	// Caller does: vkDestroySurfaceKHR(instance, surface, nullptr)
	return firstError;
}

void oa::Presenter::abandon_() noexcept {
	if (!hasOwnedState_()) return;
	if (!oa::EngineDeviceAccess::get(engine_).device) {
		for (auto& stream : graphicsBatchRing_) stream.reset();
		graphicsBatchStream_ = nullptr;
		swapchain_ = {};
		renderPass_ = VK_NULL_HANDLE;
		framebuffers_.clear();
		cmdPool_ = VK_NULL_HANDLE;
		cmdBufs_.clear();
		imGuiPool_ = VK_NULL_HANDLE;
		imGuiReady_ = false;
		return;
	}

	if (graphicsBatchStream_) {
		if (const auto status =
			graphicsBatchStream_->resetUnsubmitted(oa::EngineDeviceAccess::get(engine_));
			not status.isOk()) {
			OaLogError(oa::LogComponent::Engine,
				"oa::Presenter abandonment failed to cancel graphics batch: %s",
				status.getMessage().cStr());
		}
		graphicsBatchStream_ = nullptr;
	}

	auto retired = oa::makeUnique<oa::RetiredPresenter>();
	retired->swapchain = std::move(swapchain_);
	retired->renderPass = renderPass_;
	retired->framebuffers = std::move(framebuffers_);
	retired->commandPool = cmdPool_;
	retired->commandBuffers = std::move(cmdBufs_);
	retired->imGuiPool = imGuiPool_;
	retired->imGuiReady = imGuiReady_;
	retired->presentQueue = oa::EngineDeviceAccess::get(engine_).queues.presentQueue != nullptr
		? oa::EngineDeviceAccess::get(engine_).queues.presentQueue
		: oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue;
	retired->presentQueueRoute = oavk::classifyQueueSubmitRoute(
		oa::EngineDeviceAccess::get(engine_).queues, retired->presentQueue);
	retired->hasSwapchainMaintenance1 =
		oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1;
	retired->ownsAbandonedSurface = retired->swapchain.surface != nullptr;
	for (auto& stream : graphicsBatchRing_) {
		if (stream) retired->graphicsStreams.pushBack(oa::move(stream));
	}

	const oa::Bool retainedCallerSurface = retired->swapchain.surface != nullptr;
	swapchain_ = {};
	renderPass_ = VK_NULL_HANDLE;
	framebuffers_.clear();
	cmdPool_ = VK_NULL_HANDLE;
	cmdBufs_.clear();
	imGuiPool_ = VK_NULL_HANDLE;
	imGuiReady_ = false;
	graphicsBatchRingIndex_ = 0;
	oa::EngineDeviceAccess::get(engine_).queues.hasPresentation = false;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueue = nullptr;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily = oavk::EnumerationIndexUnset;
	oa::EngineAccess(engine_).retirePresenter(oa::move(retired));

	if (retainedCallerSurface) {
		OaLogWarn(oa::LogComponent::Engine,
			"oa::Presenter was destroyed while attached; ownership of its surface "
			"transferred to oa::Engine retirement and the caller must not destroy it");
	}
}

oa::Status oa::Presenter::beginGraphicsBatch() {
	if (graphicsBatchStream_) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"beginGraphicsBatch: already active");
	}
	const oa::U32 index = graphicsBatchRingIndex_ % kGraphicsBatchRingSize;
	if (!graphicsBatchRing_[index]) {
		auto stream = oavk::Stream::create(
			oa::EngineDeviceAccess::get(engine_),
			oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily,
			oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue);
		if (!stream.isOk()) return stream.getStatus();
		graphicsBatchRing_[index] =
			oa::makeUnique<oavk::Stream>(oa::move(*stream));
	}
	OA_RETURN_IF_ERROR(graphicsBatchRing_[index]->begin(oa::EngineDeviceAccess::get(engine_)));
	graphicsBatchStream_ = graphicsBatchRing_[index].get();
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::Presenter::flushGraphicsBatch(
	const oa::Event& inProducer) {
	if (!graphicsBatchStream_) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"flushGraphicsBatch: no active batch");
	}
	oavk::Stream* current = graphicsBatchStream_;
	graphicsBatchStream_ = nullptr;
	oavk::TimelineWait waits[2] = {};
	oa::U32 waitCount = 0;
	if (graphicsBatchRingIndex_ > 0) {
		const oa::U32 previous =
			(graphicsBatchRingIndex_ - 1) % kGraphicsBatchRingSize;
		if (graphicsBatchRing_[previous]
			&& graphicsBatchRing_[previous]->submitted) {
			waits[waitCount++] = {
				graphicsBatchRing_[previous]->timelineSem.semaphore,
				graphicsBatchRing_[previous]->timelineValue};
		}
	}
	if (inProducer.isValid()) {
		waits[waitCount++] = oa::EventAccess::timelineWait(inProducer);
	}
	oa::Status status = waitCount > 0
		? current->submitWithDependencies(
			engine_, oa::Span<const oavk::TimelineWait>(waits, waitCount))
		: current->submit(engine_);
	if (not status.isOk()) return status;
	++graphicsBatchRingIndex_;
	oa::Event completion = current->completion(oa::EngineDeviceAccess::get(engine_));
	if (not completion.isValid()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"flushGraphicsBatch: submission returned no completion event");
	}
	return completion;
}

oa::Status oa::Presenter::syncGraphicsBatch() {
	if (graphicsBatchRingIndex_ == 0) return oa::Status::ok();
	const oa::U32 previous =
		(graphicsBatchRingIndex_ - 1) % kGraphicsBatchRingSize;
	if (!graphicsBatchRing_[previous]
		|| !graphicsBatchRing_[previous]->submitted) {
		return oa::Status::ok();
	}
	return graphicsBatchRing_[previous]->synchronize(oa::EngineDeviceAccess::get(engine_));
}


// phase-C: initPresentation.
bool oa::Presenter::initPresentation(void* inSurface, VkExtent2D inExtent) {
	if (!inSurface) {
		OaLogError(oa::LogComponent::Engine, "oa::Presenter::initPresentation: null surface");
		return false;
	}
	if (!oa::EngineDeviceAccess::get(engine_).device) {
		OaLogError(oa::LogComponent::Engine, "oa::Presenter::initPresentation: device not ready");
		return false;
	}
	if (oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily == oavk::EnumerationIndexUnset) {
		OaLogError(oa::LogComponent::Engine,
			"oa::Presenter::initPresentation: no graphics queue — "
			"was PresentationMode = Swapchain set in the config?"
		);
		return false;
	}

	// If already presenting, tear down the old resources first.
	if (swapchain_.presentReady) {
		detachPresentation();
		if (swapchain_.presentReady) {
			OaLogError(oa::LogComponent::Engine,
				"oa::Presenter::initPresentation: existing presentation could not be detached");
			return false;
		}
	}

	// Confirm the graphics queue family supports present on this surface.
	VkBool32 presentOk = VK_FALSE;
	reInstanceVk(*this).vkGetPhysicalDeviceSurfaceSupportKHR(
		rePhys(*this),
		oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily,
		static_cast<VkSurfaceKHR>(inSurface),
		&presentOk);

	if (!presentOk) {
		OaLogError(oa::LogComponent::Engine,
			"oa::Presenter::initPresentation: graphics queue family %u "
			"does not support present on this surface",
			oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily);
		return false;
	}

	// Promote queue state so everything downstream can use presentQueue.
	oa::EngineDeviceAccess::get(engine_).queues.presentQueue       = oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily = oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily;
	oa::EngineDeviceAccess::get(engine_).queues.hasPresentation    = true;
	swapchain_.surface   = inSurface;
	swapchain_.extent = inExtent;

	const auto rollbackUnsubmittedInit = [this]() {
		destroySyncObjects();
		destroyCommandPool();
		destroySwapchainObjects();
		oa::EngineDeviceAccess::get(engine_).queues.hasPresentation = false;
		oa::EngineDeviceAccess::get(engine_).queues.presentQueue = nullptr;
		oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily = oavk::EnumerationIndexUnset;
		swapchain_ = {};
	};
	if (!buildSwapchainObjects() || !buildRenderPass()
		|| !buildFramebuffers() || !buildCommandPool() || !buildSyncObjects()) {
		rollbackUnsubmittedInit();
		return false;
	}

	swapchain_.presentReady = true;
	OaLogInfo(oa::LogComponent::Engine,
		"oa::Presenter: presentation ready (%ux%u, %zu swap images, "
		"retirement=%s)",
		swapchain_.extent.width, swapchain_.extent.height, swapchain_.images.size(),
		oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1
			? "present-fence" : "queue-idle-fallback");
	return true;
}

// detachPresentation — tear down swapchain resources, leave device alive.
// Caller must vkDestroySurfaceKHR the old surface after this.
void oa::Presenter::detachPresentation() {
	if (!swapchain_.presentReady) return;
	if (auto status = waitPresentationIdle(); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine,
			"detachPresentation: %s", status.toString().cStr());
		return;
	}
	shutdownImGuiResources_();
	destroySyncObjects();
	destroyCommandPool();
	destroySwapchainObjects();
	oa::EngineDeviceAccess::get(engine_).queues.hasPresentation    = false;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueue       = nullptr;
	oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily = oavk::EnumerationIndexUnset;
	swapchain_ = {};
}


// ─── recreateSwapchain ───────────────────────────────────────────────────────

bool oa::Presenter::recreateSwapchain(VkExtent2D inNewExtent) {
	if (!swapchain_.surface) return false;
	if (auto status = waitPresentationIdle(); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine,
			"recreateSwapchain: %s", status.toString().cStr());
		return false;
	}

	// Keep sync objects alive — only rebuild swapchain-dependent things.
	destroyCommandPool();
	destroySwapchainObjects();

	swapchain_.extent = inNewExtent;
	if (!buildSwapchainObjects()) return false;
	if (!buildRenderPass())       return false;
	if (!buildFramebuffers())     return false;
	if (!buildCommandPool())      return false;

	// Re-allocate command buffers into the new pool.
	cmdBufs_.resize(oa::Swapchain::kFramesInFlight);
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = cmdPool_;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = static_cast<uint32_t>(cmdBufs_.size());
	if (reVk(*this).vkAllocateCommandBuffers(reDev(*this), &ai, cmdBufs_.data()) != VK_SUCCESS)
		return false;

	// ImGui_ImplVulkan_SetMinImageCount after init hits IM_ASSERT(0) in imgui_impl_vulkan
	// (unsupported); NDEBUG hid it, debug aborted when swap image count != oa::Swapchain::kFramesInFlight.
	return true;
}


// ─── BuildSwapchainObjects ───────────────────────────────────────────────────

bool oa::Presenter::buildSwapchainObjects() {
	VkSurfaceKHR surf = static_cast<VkSurfaceKHR>(swapchain_.surface);

	VkSurfaceCapabilitiesKHR caps{};
	reInstanceVk(*this).vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
		rePhys(*this), surf, &caps);

	// SDL3 / Wayland: currentExtent may be (0,0) for hidden windows — skip frame.
	if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) {
		// Try hint extent; if that's also 0, silently succeed and let DrawFrame skip.
		if (swapchain_.extent.width == 0 || swapchain_.extent.height == 0) return true;
	}

	VkExtent2D extent = caps.currentExtent;
	if (extent.width == std::numeric_limits<uint32_t>::max()) {
		extent.width  = std::clamp(swapchain_.extent.width,
			caps.minImageExtent.width,  caps.maxImageExtent.width);
		extent.height = std::clamp(swapchain_.extent.height,
			caps.minImageExtent.height, caps.maxImageExtent.height);
	}
	swapchain_.extent = extent;

	// Pick surface format. The compose image is RGBA8 UNORM and widgets/video
	// already write what they think are sRGB-encoded bytes. Picking a SRGB
	// swapchain format makes the compose→swap blit apply OETF a second time,
	// which is the classic "washed out" symptom. Prefer UNORM so the blit is
	// a straight byte copy; fall back to SRGB if the surface only offers that.
	uint32_t fmtCount = 0;
	reInstanceVk(*this).vkGetPhysicalDeviceSurfaceFormatsKHR(
		rePhys(*this), surf, &fmtCount, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(fmtCount);
	if (fmtCount) {
		reInstanceVk(*this).vkGetPhysicalDeviceSurfaceFormatsKHR(
			rePhys(*this), surf, &fmtCount, formats.data());
	}

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
	// idles between). Override via swapchain_.vsync = false: prefer MAILBOX
	// (triple-buffered, never-tear, doesn't wait), fall back to IMMEDIATE
	// (tear-allowed, never-wait), then FIFO when neither is exposed.
	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if (not swapchain_.vsync) {
		uint32_t modeCount = 0;
		reInstanceVk(*this).vkGetPhysicalDeviceSurfacePresentModesKHR(
			rePhys(*this), surf, &modeCount, nullptr);
		std::vector<VkPresentModeKHR> modes(modeCount);
		if (modeCount) {
			reInstanceVk(*this).vkGetPhysicalDeviceSurfacePresentModesKHR(
				rePhys(*this), surf, &modeCount, modes.data());
		}
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
	// vkCmdCopyBufferToImage into the acquired swap image (FinalGlue step 3b.3/4).
	sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
	                     | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	sci.preTransform     = caps.currentTransform;
	sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	sci.presentMode      = presentMode;
	sci.clipped          = VK_TRUE;

	uint32_t families[] = {
		oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily,
		oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily
	};
	if (oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily != oa::EngineDeviceAccess::get(engine_).queues.presentQueueFamily) {
		sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		sci.queueFamilyIndexCount = 2;
		sci.pQueueFamilyIndices   = families;
	} else {
		sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	if (reVk(*this).vkCreateSwapchainKHR(reDev(*this), &sci, nullptr, &swapchain_.handle) != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkCreateSwapchainKHR failed");
		return false;
	}
	swapchain_.format = chosenFmt.format;

	uint32_t actualCount = 0;
	reVk(*this).vkGetSwapchainImagesKHR(reDev(*this), swapchain_.handle, &actualCount, nullptr);
	swapchain_.images.resize(actualCount);
	reVk(*this).vkGetSwapchainImagesKHR(reDev(*this), swapchain_.handle, &actualCount, swapchain_.images.data());

	swapchain_.views.resize(actualCount);
	for (uint32_t i = 0; i < actualCount; ++i) {
		VkImageViewCreateInfo vi{};
		vi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image                       = swapchain_.images[i];
		vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
		vi.format                      = swapchain_.format;
		vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		vi.subresourceRange.levelCount = 1;
		vi.subresourceRange.layerCount = 1;
		if (reVk(*this).vkCreateImageView(reDev(*this), &vi, nullptr, &swapchain_.views[i]) != VK_SUCCESS) {
			OaLogError(oa::LogComponent::Engine, "vkCreateImageView failed (index %u)", i);
			return false;
		}
	}
	return true;
}


// ─── BuildRenderPass ─────────────────────────────────────────────────────────

bool oa::Presenter::buildRenderPass() {
	VkAttachmentDescription color{};
	color.format         = swapchain_.format;
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

	if (reVk(*this).vkCreateRenderPass(reDev(*this), &rpi, nullptr, &renderPass_) != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkCreateRenderPass failed");
		return false;
	}
	return true;
}


// ─── BuildFramebuffers ───────────────────────────────────────────────────────

bool oa::Presenter::buildFramebuffers() {
	framebuffers_.resize(swapchain_.views.size());
	for (size_t i = 0; i < swapchain_.views.size(); ++i) {
		VkImageView att = swapchain_.views[i];
		VkFramebufferCreateInfo fbi{};
		fbi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbi.renderPass      = renderPass_;
		fbi.attachmentCount = 1;
		fbi.pAttachments    = &att;
		fbi.width           = swapchain_.extent.width;
		fbi.height          = swapchain_.extent.height;
		fbi.layers          = 1;
		if (reVk(*this).vkCreateFramebuffer(reDev(*this), &fbi, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
			OaLogError(oa::LogComponent::Engine, "vkCreateFramebuffer failed (index %zu)", i);
			return false;
		}
	}
	return true;
}


// ─── BuildCommandPool ────────────────────────────────────────────────────────

bool oa::Presenter::buildCommandPool() {
	VkCommandPoolCreateInfo pci{};
	pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pci.queueFamilyIndex = oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily;
	pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	if (reVk(*this).vkCreateCommandPool(reDev(*this), &pci, nullptr, &cmdPool_) != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkCreateCommandPool failed");
		return false;
	}

	cmdBufs_.resize(oa::Swapchain::kFramesInFlight);
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = cmdPool_;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = static_cast<uint32_t>(cmdBufs_.size());
	if (reVk(*this).vkAllocateCommandBuffers(reDev(*this), &ai, cmdBufs_.data()) != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkAllocateCommandBuffers failed");
		return false;
	}
	return true;
}


// ─── BuildSyncObjects ────────────────────────────────────────────────────────

bool oa::Presenter::buildSyncObjects() {
	swapchain_.imageAvailSem.resize(oa::Swapchain::kFramesInFlight);
	swapchain_.renderDoneSem.resize(oa::Swapchain::kFramesInFlight);
	swapchain_.inFlightFence.resize(oa::Swapchain::kFramesInFlight);
	if (oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1) {
		swapchain_.presentFence.resize(oa::Swapchain::kFramesInFlight);
		swapchain_.presentFencePending.assign(oa::Swapchain::kFramesInFlight, false);
	}
	swapchain_.presentCompletionUncertain = false;

	VkSemaphoreCreateInfo si{};
	si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkFenceCreateInfo fi{};
	fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // pre-signalled so frame 0 doesn't stall
	VkFenceCreateInfo presentFi{};
	presentFi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	for (int i = 0; i < oa::Swapchain::kFramesInFlight; ++i) {
		if (reVk(*this).vkCreateSemaphore(reDev(*this), &si, nullptr, &swapchain_.imageAvailSem[i]) != VK_SUCCESS ||
			reVk(*this).vkCreateSemaphore(reDev(*this), &si, nullptr, &swapchain_.renderDoneSem[i]) != VK_SUCCESS ||
			reVk(*this).vkCreateFence(reDev(*this), &fi, nullptr, &swapchain_.inFlightFence[i])  != VK_SUCCESS ||
			(oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1 &&
				reVk(*this).vkCreateFence(reDev(*this), &presentFi, nullptr,
					&swapchain_.presentFence[i]) != VK_SUCCESS))
		{
			OaLogError(oa::LogComponent::Engine, "sync object creation failed (slot %d)", i);
			return false;
		}
	}
	return true;
}


// ─── destroy helpers ─────────────────────────────────────────────────────────

void oa::Presenter::destroySwapchainObjects() {
	if (!oa::EngineDeviceAccess::get(engine_).device) return;
	VkDevice d = reDev(*this);

	for (auto fb : framebuffers_)   if (fb)  reVk(*this).vkDestroyFramebuffer(d, fb, nullptr);
	framebuffers_.clear();

	for (auto iv : swapchain_.views) if (iv)  reVk(*this).vkDestroyImageView(d, iv, nullptr);
	swapchain_.views.clear();
	swapchain_.images.clear();

	if (renderPass_ != VK_NULL_HANDLE) {
		reVk(*this).vkDestroyRenderPass(d, renderPass_, nullptr);
		renderPass_ = VK_NULL_HANDLE;
	}
	if (swapchain_.handle != VK_NULL_HANDLE) {
		reVk(*this).vkDestroySwapchainKHR(d, swapchain_.handle, nullptr);
		swapchain_.handle = VK_NULL_HANDLE;
	}
}

void oa::Presenter::destroyCommandPool() {
	if (!oa::EngineDeviceAccess::get(engine_).device || cmdPool_ == VK_NULL_HANDLE) return;
	// Explicitly free command buffers before destroying pool (vulkan validation requirement)
	if (!cmdBufs_.empty()) {
		reVk(*this).vkFreeCommandBuffers(reDev(*this), cmdPool_,
			static_cast<uint32_t>(cmdBufs_.size()), cmdBufs_.data());
		cmdBufs_.clear();
	}
	reVk(*this).vkDestroyCommandPool(reDev(*this), cmdPool_, nullptr);
	cmdPool_ = VK_NULL_HANDLE;
}

void oa::Presenter::destroySyncObjects() {
	if (!oa::EngineDeviceAccess::get(engine_).device) return;
	VkDevice d = reDev(*this);
	for (auto s : swapchain_.imageAvailSem) if (s) reVk(*this).vkDestroySemaphore(d, s, nullptr);
	for (auto s : swapchain_.renderDoneSem) if (s) reVk(*this).vkDestroySemaphore(d, s, nullptr);
	for (auto f : swapchain_.inFlightFence) if (f) reVk(*this).vkDestroyFence(d, f, nullptr);
	for (auto f : swapchain_.presentFence) if (f) reVk(*this).vkDestroyFence(d, f, nullptr);
	swapchain_.imageAvailSem.clear();
	swapchain_.renderDoneSem.clear();
	swapchain_.inFlightFence.clear();
	swapchain_.presentFence.clear();
	swapchain_.presentFencePending.clear();
	swapchain_.presentCompletionUncertain = false;
}


// ─── ImGui integration (OA_IMGUI) ────────────────────────────────────────────

#ifdef OA_IMGUI
void oa::Presenter::lockSharedQueueSubmitCallback_(VkQueue inQueue, void* inUser) {
	auto* engine = static_cast<oa::Engine*>(inUser);
	if (engine == nullptr || inQueue == VK_NULL_HANDLE) return;
	oa::EngineAccess(*engine).lockQueueSubmit(static_cast<void*>(inQueue));
}
void oa::Presenter::unlockSharedQueueSubmitCallback_(VkQueue inQueue, void* inUser) {
	auto* engine = static_cast<oa::Engine*>(inUser);
	if (engine == nullptr || inQueue == VK_NULL_HANDLE) return;
	oa::EngineAccess(*engine).unlockQueueSubmit(static_cast<void*>(inQueue));
}
#endif

bool oa::Presenter::initImGui(void* inNativeWindow) {
#ifdef OA_IMGUI
	if (!swapchain_.presentReady) {
		OaLogError(oa::LogComponent::Engine,
			"oa::Presenter::initImGui: call initPresentation first");
		return false;
	}
	if (imGuiReady_) return true;

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
	if (reVk(*this).vkCreateDescriptorPool(reDev(*this), &dpci, nullptr, &imGuiPool_) != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "ImGui: descriptor pool creation failed");
		return false;
	}

	IMGUI_CHECKVERSION();
	ImGui::createContext();
	ImGuiIO& io = ImGui::getIO();
	io.configFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.configFlags |= ImGuiConfigFlags_DockingEnable;   // needed for node editor layouts
	ImGui::styleColorsDark();

	// SDL3 platform backend.
	if (!imGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(inNativeWindow))) {
		OaLogError(oa::LogComponent::Engine, "ImGui_ImplSDL3_InitForVulkan failed");
		ImGui::destroyContext();
		reVk(*this).vkDestroyDescriptorPool(reDev(*this), imGuiPool_, nullptr);
		imGuiPool_ = VK_NULL_HANDLE;
		return false;
	}

	// load vulkan functions for ImGui via our volk-style loader (OaVk).
	// Required because imgui_impl_vulkan.cpp is compiled with VK_NO_PROTOTYPES.
	oavk::Device& engineDevice = oa::EngineDeviceAccess::get(engine_);
	VkInstance inst = static_cast<VkInstance>(engineDevice.instance);
	imGui_ImplVulkan_LoadFunctions(
		VK_API_VERSION_1_4,
		[](const char* inName, void* inUserData) -> PFN_vkVoidFunction {
			const auto& device = *static_cast<const oavk::Device*>(inUserData);
			const auto getInstanceProcAddr = oaVkGetInstanceProcAddr();
			return getInstanceProcAddr != nullptr
				? getInstanceProcAddr(static_cast<VkInstance>(device.instance), inName)
				: nullptr;
		},
		&engineDevice);

	// vulkan backend.
	ImGui_ImplVulkan_InitInfo vii{};
	vii.instance        = inst;
	vii.physicalDevice  = rePhys(*this);
	vii.device          = reDev(*this);
	vii.queueFamily     = oa::EngineDeviceAccess::get(engine_).queues.graphicsQueueFamily;
	vii.queue           = static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue);
	vii.descriptorPool  = imGuiPool_;
	vii.minImageCount   = oa::Swapchain::kFramesInFlight;
	vii.imageCount      = static_cast<uint32_t>(swapchain_.images.size());
	vii.pipelineInfoMain.renderPass  = renderPass_;
	vii.pipelineInfoMain.mSAASamples = VK_SAMPLE_COUNT_1_BIT;
	vii.lockQueueSubmitFn = lockSharedQueueSubmitCallback_;
	vii.unlockQueueSubmitFn = unlockSharedQueueSubmitCallback_;
	vii.lockQueueSubmitUserData = &engine_;

	if (!imGui_ImplVulkan_Init(&vii)) {
		OaLogError(oa::LogComponent::Engine, "ImGui_ImplVulkan_Init failed");
		imGui_ImplSDL3_Shutdown();
		ImGui::destroyContext();
		reVk(*this).vkDestroyDescriptorPool(reDev(*this), imGuiPool_, nullptr);
		imGuiPool_ = VK_NULL_HANDLE;
		return false;
	}

	imGuiReady_ = true;
	OaLogInfo(oa::LogComponent::Engine, "oa::Presenter: imGui (SDL3) ready");
	return true;
#else
	(void)inNativeWindow;
	return true;   // No-op — link with OA_IMGUI to enable
#endif
}

void oa::Presenter::shutdownImGui() {
#ifdef OA_IMGUI
	if (!imGuiReady_) return;
	if (auto status = waitPresentationIdle(); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine,
			"ShutdownImGui: %s", status.toString().cStr());
		return;
	}
	shutdownImGuiResources_();
#endif
}

void oa::Presenter::shutdownImGuiResources_() {
#ifdef OA_IMGUI
	if (imGuiReady_) {
		imGui_ImplVulkan_Shutdown();
		imGui_ImplSDL3_Shutdown();
		ImGui::destroyContext();
		imGuiReady_ = false;
	}
#endif
	if (imGuiPool_ != VK_NULL_HANDLE) {
		reVk(*this).vkDestroyDescriptorPool(reDev(*this), imGuiPool_, nullptr);
		imGuiPool_ = VK_NULL_HANDLE;
	}
}

void oa::Presenter::beginImGuiFrame() {
#ifdef OA_IMGUI
	assert(imGuiReady_ && "call InitImGui before BeginImGuiFrame");
	imGui_ImplVulkan_NewFrame();
	imGui_ImplSDL3_NewFrame();
	ImGui::newFrame();
	// Also call ImGui::dockSpaceOverViewport() here if you want a dockspace by default.
#endif
}

void oa::Presenter::endImGuiFrame() {
#ifdef OA_IMGUI
	assert(imGuiReady_ && "call InitImGui before EndImGuiFrame");
	ImGui::render();
#endif
}


// ─── NotifyPixelSizeChanged ──────────────────────────────────────────────────

void oa::Presenter::notifyPixelSizeChanged(int inWidthPx, int inHeightPx) {
	swapchain_.dirty     = true;
	swapchain_.dirtySize = {
		static_cast<uint32_t>(inWidthPx > 0 ? inWidthPx : 0),
		static_cast<uint32_t>(inHeightPx > 0 ? inHeightPx : 0)
	};
}

bool oa::Presenter::usesMergedGraphicsComputeQueue() const {
	return oa::EngineDeviceAccess::get(engine_).queues.computeQueue != nullptr
		and oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue == oa::EngineDeviceAccess::get(engine_).queues.computeQueue;
}

oa::Status oa::Presenter::waitPresentationIdle() {
	if (!oa::EngineDeviceAccess::get(engine_).device) return oa::Status::ok();
	// A present fence retires both presentation-engine access to the swapchain
	// resources and its consumption of renderDoneSem. This is the exact WSI
	// lifetime boundary; render-submission fences alone cannot prove either.
	if (oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1
		&& !swapchain_.presentCompletionUncertain
		&& !swapchain_.presentFence.empty()) {
		std::vector<VkFence> pending;
		pending.reserve(swapchain_.presentFence.size());
		for (oa::Usize i = 0; i < swapchain_.presentFence.size(); ++i) {
			if (i < swapchain_.presentFencePending.size()
				&& swapchain_.presentFencePending[i]) {
				pending.push_back(swapchain_.presentFence[i]);
			}
		}
		if (!pending.empty()) {
			const VkResult result = reVk(*this).vkWaitForFences(
				reDev(*this), static_cast<oa::U32>(pending.size()), pending.data(),
				VK_TRUE, UINT64_MAX);
			if (result != VK_SUCCESS) {
				return oa::Status::error(oa::StatusCode::VulkanError,
					oa::String("presentation fence wait failed: VkResult=")
						+ std::to_string(static_cast<int>(result)));
			}
			std::fill(swapchain_.presentFencePending.begin(),
				swapchain_.presentFencePending.end(), false);
		}
		return oa::Status::ok();
	}

	// Compatibility and error-recovery fallback. This is reached only when the
	// optional maintenance extension is unavailable or vkQueuePresentKHR failed
	// before its fence ownership became certain.
	void* queueHandle = oa::EngineDeviceAccess::get(engine_).queues.presentQueue != nullptr
		? oa::EngineDeviceAccess::get(engine_).queues.presentQueue
		: oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue;
	if (queueHandle == nullptr) return oa::Status::ok();
	lockSharedQueueSubmit(queueHandle);
	const VkResult result = reVk(*this).vkQueueWaitIdle(static_cast<VkQueue>(queueHandle));
	unlockSharedQueueSubmit(queueHandle);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			oa::String("presentation queue wait failed: VkResult=")
				+ std::to_string(static_cast<int>(result)));
	}
	std::fill(swapchain_.presentFencePending.begin(),
		swapchain_.presentFencePending.end(), false);
	swapchain_.presentCompletionUncertain = false;
	return oa::Status::ok();
}

oa::Status oa::Presenter::preparePresentFence(
	oa::Swapchain& inSwap, oa::U32 inFrameSlot) {
	if (!oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1) return oa::Status::ok();
	if (inSwap.presentCompletionUncertain) {
		// A failed present does not establish the normal fence-retirement
		// guarantee. Resolve that exceptional state before reusing any fence or
		// binary semaphore from the presentation ring.
		if (const auto status = waitPresentationIdle(); !status.isOk()) {
			return status;
		}
	}
	if (inFrameSlot >= inSwap.presentFence.size()
		|| inFrameSlot >= inSwap.presentFencePending.size()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"presentation fence ring is not initialized");
	}
	VkFence fence = inSwap.presentFence[inFrameSlot];
	if (inSwap.presentFencePending[inFrameSlot]) {
		const VkResult wait = reVk(*this).vkWaitForFences(
			reDev(*this), 1, &fence, VK_TRUE, UINT64_MAX);
		if (wait != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				oa::String("presentation fence reuse wait failed: VkResult=")
					+ std::to_string(static_cast<int>(wait)));
		}
	}
	const VkResult reset = reVk(*this).vkResetFences(reDev(*this), 1, &fence);
	if (reset != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			oa::String("presentation fence reset failed: VkResult=")
				+ std::to_string(static_cast<int>(reset)));
	}
	inSwap.presentFencePending[inFrameSlot] = false;
	return oa::Status::ok();
}

void oa::Presenter::finishPresent(
	oa::Swapchain& inSwap, oa::U32 inFrameSlot, VkResult inResult) {
	if (!oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1
		|| inFrameSlot >= inSwap.presentFencePending.size()) return;
	if (inResult == VK_SUCCESS || inResult == VK_SUBOPTIMAL_KHR) {
		inSwap.presentFencePending[inFrameSlot] = true;
	} else {
		// The spec only gives the fence a useful lifetime guarantee for a queued
		// presentation. Preserve correctness on failure with the narrow fallback.
		inSwap.presentCompletionUncertain = true;
	}
}

void oa::Presenter::lockSharedQueueSubmit(void* inQueue) {
	oa::EngineAccess(engine_).lockQueueSubmit(inQueue);
}

void oa::Presenter::unlockSharedQueueSubmit(void* inQueue) {
	oa::EngineAccess(engine_).unlockQueueSubmit(inQueue);
}

void oa::EngineAccess::retirePresenter(
	oa::UniquePtr<oa::RetiredPresenter>&& inPresenter)
{
	if (not inPresenter) return;
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredPresenterMutex_);
	impl_->retiredPresenters_.pushBack(oa::move(inPresenter));
}

oa::Status oa::EngineAccess::completeRetiredPresenters() {
	oa::Status result = oa::Status::ok();
	oa::Vec<oa::UniquePtr<oa::RetiredPresenter>> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredPresenterMutex_);
		retired = oa::move(impl_->retiredPresenters_);
	}

	oa::Vec<oa::UniquePtr<oa::RetiredPresenter>> pending;
	for (auto& presenter : retired) {
		oa::Status presenterStatus = oa::Status::ok();
		const auto retainError = [&presenterStatus](const oa::Status& inStatus) {
			if (presenterStatus.isOk() && !inStatus.isOk()) {
				presenterStatus = inStatus;
			}
		};

		for (auto& stream : presenter->graphicsStreams) {
			if (stream && stream->submitted) {
				retainError(stream->synchronize(impl_->device_));
			}
		}

		if (presenter->swapchain.presentReady) {
			if (presenter->hasSwapchainMaintenance1
				&& !presenter->swapchain.presentCompletionUncertain
				&& !presenter->swapchain.presentFence.empty()) {
				std::vector<VkFence> fences;
				for (oa::Usize i = 0;
					i < presenter->swapchain.presentFence.size(); ++i) {
					if (i < presenter->swapchain.presentFencePending.size()
						&& presenter->swapchain.presentFencePending[i]) {
						fences.push_back(presenter->swapchain.presentFence[i]);
					}
				}
				if (!fences.empty()) {
					const VkResult wait = impl_->device_.deviceDispatch.vkWaitForFences(
						static_cast<VkDevice>(impl_->device_.device),
						static_cast<oa::U32>(fences.size()), fences.data(),
						VK_TRUE, UINT64_MAX);
					if (wait != VK_SUCCESS) {
						retainError(oa::Status::error(
							oa::StatusCode::VulkanError,
							oa::String("retired presentation fence wait failed: VkResult=")
								+ std::to_string(static_cast<int>(wait))));
					}
				}
			} else if (presenter->presentQueue != nullptr) {
				std::mutex* queueMutex = nullptr;
				switch (presenter->presentQueueRoute) {
				case oavk::QueueSubmitRoute::Compute:
					queueMutex = &impl_->computeQueueMutex_;
					break;
				case oavk::QueueSubmitRoute::AsyncCompute:
					queueMutex = &impl_->asyncComputeQueueMutex_;
					break;
				case oavk::QueueSubmitRoute::Transfer:
					queueMutex = &impl_->transferQueueMutex_;
					break;
				case oavk::QueueSubmitRoute::Graphics:
					queueMutex = &impl_->graphicsQueueMutex_;
					break;
				case oavk::QueueSubmitRoute::Present:
					queueMutex = &impl_->presentQueueMutex_;
					break;
				case oavk::QueueSubmitRoute::Unknown:
					break;
				}
				if (queueMutex == nullptr) {
					retainError(oa::Status::error(
						oa::StatusCode::Internal,
						"retired presentation queue has no synchronization route"));
				} else {
					queueMutex->lock();
					const VkResult wait = impl_->device_.deviceDispatch.vkQueueWaitIdle(
						static_cast<VkQueue>(presenter->presentQueue));
					queueMutex->unlock();
					if (wait != VK_SUCCESS) {
						retainError(oa::Status::error(
							oa::StatusCode::VulkanError,
							oa::String("retired presentation queue wait failed: VkResult=")
								+ std::to_string(static_cast<int>(wait))));
					}
				}
			}
		}

		if (!presenterStatus.isOk()) {
			if (result.isOk()) result = presenterStatus;
			pending.pushBack(oa::move(presenter));
			continue;
		}

#ifdef OA_IMGUI
		if (presenter->imGuiReady) {
			imGui_ImplVulkan_Shutdown();
			imGui_ImplSDL3_Shutdown();
			ImGui::destroyContext();
			presenter->imGuiReady = false;
		}
#endif
		VkDevice device = static_cast<VkDevice>(impl_->device_.device);
		if (presenter->imGuiPool != VK_NULL_HANDLE) {
			impl_->device_.deviceDispatch.vkDestroyDescriptorPool(device, presenter->imGuiPool, nullptr);
			presenter->imGuiPool = VK_NULL_HANDLE;
		}
		for (auto& stream : presenter->graphicsStreams) {
			if (stream) stream->destroy(impl_->device_);
		}
		for (auto semaphore : presenter->swapchain.imageAvailSem) {
			if (semaphore) impl_->device_.deviceDispatch.vkDestroySemaphore(device, semaphore, nullptr);
		}
		for (auto semaphore : presenter->swapchain.renderDoneSem) {
			if (semaphore) impl_->device_.deviceDispatch.vkDestroySemaphore(device, semaphore, nullptr);
		}
		for (auto fence : presenter->swapchain.inFlightFence) {
			if (fence) impl_->device_.deviceDispatch.vkDestroyFence(device, fence, nullptr);
		}
		for (auto fence : presenter->swapchain.presentFence) {
			if (fence) impl_->device_.deviceDispatch.vkDestroyFence(device, fence, nullptr);
		}
		if (presenter->commandPool != VK_NULL_HANDLE) {
			if (!presenter->commandBuffers.empty()) {
				impl_->device_.deviceDispatch.vkFreeCommandBuffers(
					device, presenter->commandPool,
					static_cast<oa::U32>(presenter->commandBuffers.size()),
					presenter->commandBuffers.data());
			}
			impl_->device_.deviceDispatch.vkDestroyCommandPool(device, presenter->commandPool, nullptr);
		}
		for (auto framebuffer : presenter->framebuffers) {
			if (framebuffer) impl_->device_.deviceDispatch.vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		for (auto view : presenter->swapchain.views) {
			if (view) impl_->device_.deviceDispatch.vkDestroyImageView(device, view, nullptr);
		}
		if (presenter->renderPass != VK_NULL_HANDLE) {
			impl_->device_.deviceDispatch.vkDestroyRenderPass(device, presenter->renderPass, nullptr);
		}
		if (presenter->swapchain.handle != VK_NULL_HANDLE) {
			impl_->device_.deviceDispatch.vkDestroySwapchainKHR(device, presenter->swapchain.handle, nullptr);
		}
		if (presenter->ownsAbandonedSurface
			&& presenter->swapchain.surface != nullptr) {
			impl_->device_.instanceDispatch.vkDestroySurfaceKHR(
				static_cast<VkInstance>(impl_->device_.instance),
				static_cast<VkSurfaceKHR>(presenter->swapchain.surface), nullptr);
		}
	}

	if (!pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredPresenterMutex_);
		for (auto& presenter : pending) {
			impl_->retiredPresenters_.pushBack(oa::move(presenter));
		}
	}
	return result;
}

// ─── DrawFrame ───────────────────────────────────────────────────────────────

bool oa::Presenter::drawFrame() {
	if (!swapchain_.presentReady || swapchain_.handle == VK_NULL_HANDLE) return false;

	// Proactive swapchain recreate when the window reported a pixel size change.
	if (swapchain_.dirty) {
		swapchain_.dirty = false;
		if (swapchain_.dirtySize.width > 0 and swapchain_.dirtySize.height > 0) {
			if (not recreateSwapchain(swapchain_.dirtySize)) return false;
		}
	}

	// Skip if window is minimised / zero-sized (SDL3 / Wayland may report 0).
	if (swapchain_.extent.width == 0 || swapchain_.extent.height == 0) return true;

	if (const auto status = preparePresentFence(
		swapchain_, static_cast<oa::U32>(swapchain_.frameIndex)); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine, "DrawFrame: %s", status.toString().cStr());
		return false;
	}

	reVk(*this).vkWaitForFences(reDev(*this), 1, &swapchain_.inFlightFence[swapchain_.frameIndex], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex = 0;
	VkResult acq = reVk(*this).vkAcquireNextImageKHR(
		reDev(*this), swapchain_.handle, UINT64_MAX,
		swapchain_.imageAvailSem[swapchain_.frameIndex], VK_NULL_HANDLE,
		&imageIndex);

	if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
		return recreateSwapchain(swapchain_.extent);
	}
	if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return false;

	reVk(*this).vkResetFences(reDev(*this), 1, &swapchain_.inFlightFence[swapchain_.frameIndex]);

	VkCommandBuffer cmd = cmdBufs_[swapchain_.frameIndex];
	reVk(*this).vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	reVk(*this).vkBeginCommandBuffer(cmd, &bi);

	VkClearValue clearColor{};
	clearColor.color = {{0.12f, 0.12f, 0.12f, 1.0f}};   // dark editor grey

	VkRenderPassBeginInfo rpBi{};
	rpBi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpBi.renderPass        = renderPass_;
	rpBi.framebuffer       = framebuffers_[imageIndex];
	rpBi.renderArea.offset = {0, 0};
	rpBi.renderArea.extent = swapchain_.extent;
	rpBi.clearValueCount   = 1;
	rpBi.pClearValues      = &clearColor;

	reVk(*this).vkCmdBeginRenderPass(cmd, &rpBi, VK_SUBPASS_CONTENTS_INLINE);

#ifdef OA_IMGUI
	if (imGuiReady_) {
		ImDrawData* drawData = ImGui::getDrawData();
		if (drawData) {
			imGui_ImplVulkan_RenderDrawData(drawData, cmd);
		}
	}
#endif

	reVk(*this).vkCmdEndRenderPass(cmd);
	reVk(*this).vkEndCommandBuffer(cmd);

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo si{};
	si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.waitSemaphoreCount   = 1;
	si.pWaitSemaphores      = &swapchain_.imageAvailSem[swapchain_.frameIndex];
	si.pWaitDstStageMask    = &waitStage;
	si.commandBufferCount   = 1;
	si.pCommandBuffers      = &cmd;
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores    = &swapchain_.renderDoneSem[swapchain_.frameIndex];

	VkResult submitRes = VK_ERROR_UNKNOWN;
	{
		QueueSubmitScope queueScope(
			this, oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue);
		submitRes = reVk(*this).vkQueueSubmit(
			static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue),
			1, &si, swapchain_.inFlightFence[swapchain_.frameIndex]);
	}
	if (submitRes != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkQueueSubmit failed (VkResult=%d)",
			static_cast<int>(submitRes));
		return false;
	}

	VkPresentInfoKHR pi{};
	pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores    = &swapchain_.renderDoneSem[swapchain_.frameIndex];
	pi.swapchainCount     = 1;
	pi.pSwapchains        = &swapchain_.handle;
	pi.pImageIndices      = &imageIndex;
	VkSwapchainPresentFenceInfoKHR presentFenceInfo{};
	if (oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1) {
		presentFenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
		presentFenceInfo.swapchainCount = 1;
		presentFenceInfo.pFences =
			&swapchain_.presentFence[swapchain_.frameIndex];
		pi.pNext = &presentFenceInfo;
	}

	VkResult pres = VK_ERROR_UNKNOWN;
	{
		QueueSubmitScope queueScope(
			this, oa::EngineDeviceAccess::get(engine_).queues.presentQueue);
		pres = reVk(*this).vkQueuePresentKHR(
			static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.presentQueue), &pi);
	}
	finishPresent(swapchain_, static_cast<oa::U32>(swapchain_.frameIndex), pres);

	swapchain_.frameIndex = (swapchain_.frameIndex + 1) % oa::Swapchain::kFramesInFlight;

	if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
		if (!recreateSwapchain(swapchain_.extent)) {
			return false;
		}
		return true;
	}
	if (pres != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine, "vkQueuePresentKHR failed (VkResult=%d)",
			static_cast<int>(pres));
		return false;
	}
	return true;
}


// ─── ctx-mediated present primitives (step 3b.3) ────────────────────────────
//
// split of DrawFrame's body into acquireSwapchainImage + presentSwapchainImage.
// oa::Viewer uses these directly; DrawFrame remains the compact one-call path.

bool oa::Presenter::acquireSwapchainImage(
	oa::Swapchain& inSwap, AcquireResult& outResult) {
	outResult = AcquireResult{};

	if (not inSwap.presentReady or inSwap.handle == VK_NULL_HANDLE) {
		return false;
	}

	// Proactive recreate on signalled resize.
	if (inSwap.dirty) {
		inSwap.dirty = false;
		if (inSwap.dirtySize.width > 0 and inSwap.dirtySize.height > 0) {
			if (not recreateSwapchain(inSwap.dirtySize)) return false;
			outResult.recreated = true;
			return true;
		}
	}

	if (inSwap.extent.width == 0 or inSwap.extent.height == 0) {
		// Zero-size window — nothing to acquire. Caller should skip this frame.
		return false;
	}

	const oa::U32 frameSlot = static_cast<oa::U32>(inSwap.frameIndex);
	if (const auto status = preparePresentFence(inSwap, frameSlot); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine,
			"acquireSwapchainImage: %s", status.toString().cStr());
		return false;
	}

	reVk(*this).vkWaitForFences(reDev(*this), 1, &inSwap.inFlightFence[frameSlot],
		VK_TRUE, UINT64_MAX);

	uint32_t imageIndex = 0;
	const VkResult acq = reVk(*this).vkAcquireNextImageKHR(
		reDev(*this), inSwap.handle, UINT64_MAX,
		inSwap.imageAvailSem[frameSlot], VK_NULL_HANDLE, &imageIndex);

	if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
		if (not recreateSwapchain(inSwap.extent)) return false;
		outResult.recreated = true;
		return true;
	}
	if (acq != VK_SUCCESS and acq != VK_SUBOPTIMAL_KHR) {
		OaLogError(oa::LogComponent::Engine,
			"acquireSwapchainImage: vkAcquireNextImageKHR failed (VkResult=%d)",
			static_cast<int>(acq));
		return false;
	}

	reVk(*this).vkResetFences(reDev(*this), 1, &inSwap.inFlightFence[frameSlot]);

	outResult.frameSlot  = frameSlot;
	outResult.imageIndex = imageIndex;
	outResult.image      = inSwap.images[imageIndex];
	outResult.view       = inSwap.views[imageIndex];
	outResult.recreated  = false;
	return true;
}

bool oa::Presenter::presentSwapchainImage(
	oa::Swapchain&       inSwap,
	oa::U32              inImageIndex,
	oa::U32              inFrameSlot,
	const PresentArgs& inArgs) {
	if (not inSwap.presentReady or inSwap.handle == VK_NULL_HANDLE) return false;
	if (inImageIndex >= inSwap.images.size())                       return false;

	const bool hasImGui     = inArgs.drawImGui;
	const bool hasBlitImage = (not hasImGui) and inArgs.blitSrcImage  != nullptr;
	const bool hasBlitBuf   = (not hasImGui) and (not hasBlitImage) and inArgs.blitSrcBuffer != nullptr;
	const bool hasClear     = (not hasImGui) and (not hasBlitImage) and (not hasBlitBuf)
		and inArgs.clearRgba != nullptr;
	const bool toTransferDst = hasBlitImage or hasBlitBuf or hasClear;

	// ImGui-only path (step 3b.5): use the engine's render pass +
	// framebuffer[imageIndex], same machinery DrawFrame uses. The render
	// pass's loadOp=CLEAR provides the clear; finalLayout=PRESENT_SRC
	// transitions for us. Blit+ImGui composite needs a separate render pass
	// with loadOp=LOAD and lands in step 3c when DrawFrame's body is
	// replaced wholesale by explicit present.
	if (hasImGui) {
#ifdef OA_IMGUI
		if (not imGuiReady_) {
			OaLogError(oa::LogComponent::Engine,
				"presentSwapchainImage: drawImGui set but ImGui not initialised");
			return false;
		}

		VkCommandBuffer cmd = cmdBufs_[inFrameSlot];
		reVk(*this).vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		reVk(*this).vkBeginCommandBuffer(cmd, &bi);

		VkClearValue clearColor{};
		clearColor.color = {{0.12F, 0.12F, 0.12F, 1.0F}};  // matches DrawFrame's dark grey
		VkRenderPassBeginInfo rpBi{};
		rpBi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpBi.renderPass        = renderPass_;
		rpBi.framebuffer       = framebuffers_[inImageIndex];
		rpBi.renderArea.offset = {0, 0};
		rpBi.renderArea.extent = inSwap.extent;
		rpBi.clearValueCount   = 1;
		rpBi.pClearValues      = &clearColor;
		reVk(*this).vkCmdBeginRenderPass(cmd, &rpBi, VK_SUBPASS_CONTENTS_INLINE);

		ImDrawData* drawData = ImGui::getDrawData();
		if (drawData != nullptr) {
			imGui_ImplVulkan_RenderDrawData(drawData, cmd);
		}

		reVk(*this).vkCmdEndRenderPass(cmd);
		reVk(*this).vkEndCommandBuffer(cmd);

		// submit + present — same semaphore chain as the transfer paths below.
		VkSemaphore waitSemaphoresImgui[2] = {
			inSwap.imageAvailSem[inFrameSlot],
			static_cast<VkSemaphore>(inArgs.waitTimelineSemaphore),
		};
		VkPipelineStageFlags waitStagesImgui[2] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		};
		oa::U64 waitValuesImgui[2] = {0, inArgs.waitTimelineValue};
		oa::U64 signalValuesImgui[1] = {0};
		const oa::U32 waitCountImgui = inArgs.waitTimelineSemaphore != nullptr ? 2U : 1U;
		VkTimelineSemaphoreSubmitInfo timelineInfoImgui{};
		timelineInfoImgui.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timelineInfoImgui.waitSemaphoreValueCount = waitCountImgui;
		timelineInfoImgui.pWaitSemaphoreValues = waitValuesImgui;
		timelineInfoImgui.signalSemaphoreValueCount = 1;
		timelineInfoImgui.pSignalSemaphoreValues = signalValuesImgui;
		VkSubmitInfo siImgui{};
		siImgui.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		siImgui.pNext                = inArgs.waitTimelineSemaphore != nullptr
			? &timelineInfoImgui : nullptr;
		siImgui.waitSemaphoreCount   = waitCountImgui;
		siImgui.pWaitSemaphores      = waitSemaphoresImgui;
		siImgui.pWaitDstStageMask    = waitStagesImgui;
		siImgui.commandBufferCount   = 1;
		siImgui.pCommandBuffers      = &cmd;
		siImgui.signalSemaphoreCount = 1;
		siImgui.pSignalSemaphores    = &inSwap.renderDoneSem[inFrameSlot];
		VkResult subImgui = VK_ERROR_UNKNOWN;
		{
			QueueSubmitScope queueScope(
				this, oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue);
			subImgui = reVk(*this).vkQueueSubmit(
				static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue),
				1, &siImgui, inSwap.inFlightFence[inFrameSlot]);
		}
		if (subImgui != VK_SUCCESS) {
			OaLogError(oa::LogComponent::Engine,
				"presentSwapchainImage(ImGui): vkQueueSubmit failed (VkResult=%d)",
				static_cast<int>(subImgui));
			return false;
		}
		VkPresentInfoKHR piImgui{};
		piImgui.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		piImgui.waitSemaphoreCount = 1;
		piImgui.pWaitSemaphores    = &inSwap.renderDoneSem[inFrameSlot];
		piImgui.swapchainCount     = 1;
		piImgui.pSwapchains        = &inSwap.handle;
		piImgui.pImageIndices      = &inImageIndex;
		VkSwapchainPresentFenceInfoKHR presentFenceInfoImgui{};
		if (oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1) {
			presentFenceInfoImgui.sType =
				VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
			presentFenceInfoImgui.swapchainCount = 1;
			presentFenceInfoImgui.pFences = &inSwap.presentFence[inFrameSlot];
			piImgui.pNext = &presentFenceInfoImgui;
		}
		VkResult presImgui = VK_ERROR_UNKNOWN;
		{
			QueueSubmitScope queueScope(
				this, oa::EngineDeviceAccess::get(engine_).queues.presentQueue);
			presImgui = reVk(*this).vkQueuePresentKHR(
				static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.presentQueue),
				&piImgui);
		}
		finishPresent(inSwap, inFrameSlot, presImgui);
		inSwap.frameIndex = (inSwap.frameIndex + 1) % oa::Swapchain::kFramesInFlight;
		if (presImgui == VK_ERROR_OUT_OF_DATE_KHR or presImgui == VK_SUBOPTIMAL_KHR) {
			return recreateSwapchain(inSwap.extent);
		}
		if (presImgui != VK_SUCCESS) {
			OaLogError(oa::LogComponent::Engine,
				"presentSwapchainImage(ImGui): vkQueuePresentKHR failed (VkResult=%d)",
				static_cast<int>(presImgui));
			return false;
		}
		return true;
#else
		OaLogError(oa::LogComponent::Engine,
			"presentSwapchainImage: drawImGui set but OA_IMGUI not compiled in");
		return false;
#endif
	}

	VkCommandBuffer cmd = cmdBufs_[inFrameSlot];
	reVk(*this).vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	reVk(*this).vkBeginCommandBuffer(cmd, &bi);

	VkImage swapImg = inSwap.images[inImageIndex];

	// first barrier: UNDEFINED → (TRANSFER_DST if blit/clear, else PRESENT_SRC).
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
		reVk(*this).vkCmdPipelineBarrier(cmd,
			// The acquire semaphore is waited at the first stage that touches the
			// image. include that same stage on the layout transition's source
			// side so synchronization validation can connect acquire to the write.
			toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
				: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
				: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b);
	}

	if (hasBlitImage) {
		// VkImage→VkImage path (step 3c.1): vkCmdBlitImage handles format
		// conversion (compose RGBA8 → swap BGRA8_SRGB) and scaling. source
		// image current layout supplied via inArgs.blitSrcLayout; we
		// transition it to TRANSFER_SRC_OPTIMAL for the blit, then back.
		const VkImage srcImg = static_cast<VkImage>(inArgs.blitSrcImage);
		const auto    srcLayoutIn  = static_cast<VkImageLayout>(inArgs.blitSrcLayout);

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
		reVk(*this).vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &srcIn);

		VkImageBlit region{};
		region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.srcOffsets[0]  = { 0, 0, 0 };
		region.srcOffsets[1]  = {
			static_cast<int32_t>(inArgs.blitSrcWidth),
			static_cast<int32_t>(inArgs.blitSrcHeight),
			1
		};
		region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.dstOffsets[0]  = { 0, 0, 0 };
		region.dstOffsets[1]  = {
			static_cast<int32_t>(inSwap.extent.width),
			static_cast<int32_t>(inSwap.extent.height),
			1
		};
		const VkFilter vkFilter = (inArgs.filter == oa::Filter::Nearest)
			? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
		reVk(*this).vkCmdBlitImage(cmd,
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
		reVk(*this).vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &srcOut);
	} else if (hasBlitBuf) {
		// vkCmdCopyBufferToImage: assumes the source buffer holds packed RGBA8
		// at the same extent as the swap image. format mismatch (swap is
		// BGRA8_SRGB today) is a known limitation — channel swap and
		// linear→sRGB are tracked for the VkImage path (blitSrcImage above).
		VkBufferImageCopy copy{};
		copy.bufferOffset      = 0;
		copy.bufferRowLength   = 0;  // tightly packed
		copy.bufferImageHeight = 0;
		copy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		copy.imageSubresource.mipLevel       = 0;
		copy.imageSubresource.baseArrayLayer = 0;
		copy.imageSubresource.layerCount     = 1;
		copy.imageOffset = { 0, 0, 0 };
		copy.imageExtent = { inArgs.blitSrcWidth, inArgs.blitSrcHeight, 1 };
		reVk(*this).vkCmdCopyBufferToImage(cmd,
			static_cast<::VkBuffer>(inArgs.blitSrcBuffer),
			swapImg,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copy);
	} else if (hasClear) {
		VkClearColorValue clear{};
		clear.float32[0] = inArgs.clearRgba[0];
		clear.float32[1] = inArgs.clearRgba[1];
		clear.float32[2] = inArgs.clearRgba[2];
		clear.float32[3] = inArgs.clearRgba[3];
		VkImageSubresourceRange sub{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		reVk(*this).vkCmdClearColorImage(cmd, swapImg,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &sub);
	}

	if (toTransferDst) {
		// second barrier: TRANSFER_DST → PRESENT_SRC.
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
		reVk(*this).vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b);
	}

	reVk(*this).vkEndCommandBuffer(cmd);

	// submit on graphics queue. wait on ImageAvail, signal RenderDone,
	// fence = inFlight[frameSlot] — same model DrawFrame uses.
	VkSemaphore waitSemaphores[2] = {
		inSwap.imageAvailSem[inFrameSlot],
		static_cast<VkSemaphore>(inArgs.waitTimelineSemaphore),
	};
	VkPipelineStageFlags waitStages[2] = {
		toTransferDst ? VK_PIPELINE_STAGE_TRANSFER_BIT
			: VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	};
	oa::U64 waitValues[2] = {0, inArgs.waitTimelineValue};
	oa::U64 signalValues[1] = {0};
	const oa::U32 waitCount = inArgs.waitTimelineSemaphore != nullptr ? 2U : 1U;
	VkTimelineSemaphoreSubmitInfo timelineInfo{};
	timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timelineInfo.waitSemaphoreValueCount = waitCount;
	timelineInfo.pWaitSemaphoreValues = waitValues;
	timelineInfo.signalSemaphoreValueCount = 1;
	timelineInfo.pSignalSemaphoreValues = signalValues;
	VkSubmitInfo si{};
	si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.pNext                = inArgs.waitTimelineSemaphore != nullptr
		? &timelineInfo : nullptr;
	si.waitSemaphoreCount   = waitCount;
	si.pWaitSemaphores      = waitSemaphores;
	si.pWaitDstStageMask    = waitStages;
	si.commandBufferCount   = 1;
	si.pCommandBuffers      = &cmd;
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores    = &inSwap.renderDoneSem[inFrameSlot];

	VkResult submitRes = VK_ERROR_UNKNOWN;
	{
		QueueSubmitScope queueScope(
			this, oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue);
		submitRes = reVk(*this).vkQueueSubmit(
			static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.graphicsQueue),
			1, &si, inSwap.inFlightFence[inFrameSlot]);
	}
	if (submitRes != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine,
			"presentSwapchainImage: vkQueueSubmit failed (VkResult=%d)",
			static_cast<int>(submitRes));
		return false;
	}

	VkPresentInfoKHR pi{};
	pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores    = &inSwap.renderDoneSem[inFrameSlot];
	pi.swapchainCount     = 1;
	pi.pSwapchains        = &inSwap.handle;
	pi.pImageIndices      = &inImageIndex;
	VkSwapchainPresentFenceInfoKHR presentFenceInfo{};
	if (oa::EngineDeviceAccess::get(engine_).info.software.hasSwapchainMaintenance1) {
		presentFenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
		presentFenceInfo.swapchainCount = 1;
		presentFenceInfo.pFences = &inSwap.presentFence[inFrameSlot];
		pi.pNext = &presentFenceInfo;
	}

	VkResult pres = VK_ERROR_UNKNOWN;
	{
		QueueSubmitScope queueScope(
			this, oa::EngineDeviceAccess::get(engine_).queues.presentQueue);
		pres = reVk(*this).vkQueuePresentKHR(
			static_cast<VkQueue>(oa::EngineDeviceAccess::get(engine_).queues.presentQueue), &pi);
	}
	finishPresent(inSwap, inFrameSlot, pres);

	inSwap.frameIndex = (inSwap.frameIndex + 1) % oa::Swapchain::kFramesInFlight;

	if (pres == VK_ERROR_OUT_OF_DATE_KHR or pres == VK_SUBOPTIMAL_KHR) {
		return recreateSwapchain(inSwap.extent);
	}
	if (pres != VK_SUCCESS) {
		OaLogError(oa::LogComponent::Engine,
			"presentSwapchainImage: vkQueuePresentKHR failed (VkResult=%d)",
			static_cast<int>(pres));
		return false;
	}
	return true;
}
