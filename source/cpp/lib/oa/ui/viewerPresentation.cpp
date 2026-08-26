// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/presenter.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/stream.h>
#include <oa/runtime/oaVma.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/ui/viewer.h>
#include <oa/ui/image.h>
#include <oa/core/log.h>



oa::Status oa::Viewer::initPresentation(
	oa::Presenter& inPresenter,
	void* inSurface)
{
	presenter_ = &inPresenter;
	engine_    = &inPresenter.engine();

	// Build the presenter-owned swapchain on the caller's surface.
	// vsync preference must be set before
	// initPresentation calls buildSwapchainObjects (which reads it).
	inPresenter.swapchain().vsync = config_.vsync;
	if (not inPresenter.initPresentation(
			inSurface,
			VkExtent2D{ config_.width, config_.height })) {
		OaLogError(oa::LogComponent::Ui,
			"oa::Viewer: presenter initialization failed");
		return oa::Status::error("oa::Viewer: presenter initialization failed");
	}

	if (auto s = buildComposeImage(config_.width, config_.height); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer: compose image build failed: %s",
			s.getMessage().cStr());
		inPresenter.detachPresentation();
		presenter_ = nullptr;
		engine_ = nullptr;
		return s;
	}

	if (auto s = ui_.init(*engine_, config_.style); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer: UI initialization failed: %s",
			s.getMessage().cStr());
		destroyComposeImage();
		inPresenter.detachPresentation();
		presenter_ = nullptr;
		engine_ = nullptr;
		return s;
	}

	if (auto s = ui_.initBlit(composeView_); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer: UI compositor initialization failed: %s",
			s.getMessage().cStr());
		const oa::Status cleanupStatus = ui_.close();
		if (not cleanupStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer: UI cleanup after initialization failure also failed: %s",
				cleanupStatus.getMessage().cStr());
		}
		destroyComposeImage();
		inPresenter.detachPresentation();
		presenter_ = nullptr;
		engine_ = nullptr;
		return s;
	}

	if (auto s = textAtlas_.init(*engine_); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer: text atlas initialization failed: %s",
			s.getMessage().cStr());
		const oa::Status cleanupStatus = ui_.close();
		if (not cleanupStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer: UI cleanup after atlas initialization failure also failed: %s",
				cleanupStatus.getMessage().cStr());
		}
		destroyComposeImage();
		inPresenter.detachPresentation();
		presenter_ = nullptr;
		engine_ = nullptr;
		return s;
	}
	if (auto s = ui_.bindTextAtlas(textAtlas_); not s.isOk()) {
		OaLogError(oa::LogComponent::Ui, "oa::Viewer: UI text binding failed: %s",
			s.getMessage().cStr());
		const oa::Status cleanupStatus = ui_.close();
		if (not cleanupStatus.isOk()) {
			OaLogError(oa::LogComponent::Ui,
				"oa::Viewer: UI cleanup after text binding failure also failed: %s",
				cleanupStatus.getMessage().cStr());
		}
		textAtlas_ = {};
		destroyComposeImage();
		inPresenter.detachPresentation();
		presenter_ = nullptr;
		engine_ = nullptr;
		return s;
	}
	return oa::Status::ok();
}

oa::Status oa::Viewer::destroyPresentation() {
	if (presenter_ == nullptr) return oa::Status::ok();
	// UI uploads and title glyphs retain exact events from the Presenter's
	// graphics-stream ring. Complete and release those consumers while their
	// timeline semaphores are still alive; Presenter::Close destroys the ring.
	// Reversing this order leaves copied completion events pointing at destroyed
	// vulkan semaphores during oa::Ui/oa::GlyphBuffer cleanup.
	const oa::Status uiStatus = ui_.close();
	destroyWindowDecoration();
	textAtlas_ = {};
	// A renderer frame remains submitted for the whole blocking viewer session.
	// Return only the final graphics completion after ui_.close has completed
	// every sample, and before Presenter::Close destroys its timeline object.
	const oa::Status frameStatus = finalizeBorrowedFrame();
	// Closing the Presenter also completes and destroys its graphics-stream
	// ring. Detaching only the swapchain would leave those device children live
	// until the borrowed Presenter wrapper dies. A borrowed engine must remain
	// valid after oa::Viewer::run(inEngine) returns.
	const oa::Status presenterStatus = presenter_->close();
	destroyComposeImage();
	renderCompletion_ = {};
	renderDependency_ = {};
	presenter_ = nullptr;
	engine_    = nullptr;
	if (not uiStatus.isOk()) return uiStatus;
	if (not frameStatus.isOk()) return frameStatus;
	return presenterStatus;
}


// ─── BuildComposeImage / DestroyComposeImage ─────────────────────────────────

oa::Status oa::Viewer::buildComposeImage(oa::U32 inWidth, oa::U32 inHeight) {
	if (engine_ == nullptr) return oa::Status::error("oa::Viewer: no engine");

	VkDevice dev = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine_).device);

	VkImageCreateInfo ici{};
	ici.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ici.imageType         = VK_IMAGE_TYPE_2D;
	ici.format            = VK_FORMAT_R8G8B8A8_UNORM;
	ici.extent.width      = inWidth;
	ici.extent.height     = inHeight;
	ici.extent.depth      = 1;
	ici.mipLevels         = 1;
	ici.arrayLayers       = 1;
	ici.samples           = VK_SAMPLE_COUNT_1_BIT;
	ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
	ici.usage             = VK_IMAGE_USAGE_STORAGE_BIT
	                      | VK_IMAGE_USAGE_SAMPLED_BIT
	                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
	                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

	const oa::U32 computeFamily  = oa::EngineDeviceAccess::get(*engine_).queues.computeQueueFamily;
	const oa::U32 graphicsFamily = oa::EngineDeviceAccess::get(*engine_).queues.graphicsQueueFamily;
	oa::U32 families[2] = { computeFamily, graphicsFamily };
	if (computeFamily != graphicsFamily) {
		ici.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		ici.queueFamilyIndexCount = 2;
		ici.pQueueFamilyIndices   = families;
	} else {
		ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	OaVmaAllocationCreateInfo allocCI{};
	allocCI.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;

	VkImage          img   = VK_NULL_HANDLE;
	OaVmaAllocation  alloc = VK_NULL_HANDLE;
	if (OaVmaCreateImage(
		static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator),
		&ici, &allocCI, &img, &alloc, nullptr) != VK_SUCCESS)
	{
		return oa::Status::error(oa::StatusCode::OutOfMemory, "oa::Viewer: compose image allocation failed");
	}

	VkImageViewCreateInfo vi{};
	vi.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vi.image                       = img;
	vi.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
	vi.format                      = VK_FORMAT_R8G8B8A8_UNORM;
	vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vi.subresourceRange.levelCount = 1;
	vi.subresourceRange.layerCount = 1;
	VkImageView view = VK_NULL_HANDLE;
	if (oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCreateImageView(dev, &vi, nullptr, &view) != VK_SUCCESS) {
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator), img, alloc);
		return oa::Status::error(oa::StatusCode::VulkanError, "oa::Viewer: compose image view creation failed");
	}

	// Transition UNDEFINED → GENERAL via one-shot compute stream.
	{
		oavk::Stream* s = oa::EngineSubmissionAccess::acquireStream(*engine_);
		if (s == nullptr) {
			oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator), img, alloc);
			return oa::Status::error(oa::StatusCode::VulkanError, "oa::Viewer: stream acquisition failed");
		}
		if (auto st = s->begin(oa::EngineDeviceAccess::get(*engine_)); not st.isOk()) {
			oa::EngineSubmissionAccess::releaseStream(*engine_, s);
			oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator), img, alloc);
			return st;
		}
		VkCommandBuffer cmd = static_cast<VkCommandBuffer>(s->commandBuffer);
		VkImageMemoryBarrier barrier{};
		barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = img;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask       = 0;
		barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
		oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
		if (auto st = s->submitAndWait(*engine_); not st.isOk()) {
			oa::EngineSubmissionAccess::releaseStream(*engine_, s);
			oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator), img, alloc);
			return st;
		}
		oa::EngineSubmissionAccess::releaseStream(*engine_, s);
	}

	oa::U32 bindlessIndex = oa::EngineBindlessAccess::get(*engine_)
		.registerStorageImage(oa::EngineDeviceAccess::get(*engine_), view, VK_IMAGE_LAYOUT_GENERAL);
	if (bindlessIndex == OA_BINDLESS_INVALID) {
		oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkDestroyImageView(dev, view, nullptr);
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator), img, alloc);
		return oa::Status::error(oa::StatusCode::ResourceExhausted, "oa::Viewer: compose bindless registration failed");
	}

	composeImage_ = img;
	composeView_ = view;
	composeAllocation_ = alloc;
	composeBindlessIndex_ = bindlessIndex;
	composeWidth_ = inWidth;
	composeHeight_ = inHeight;
	return oa::Status::ok();
}

void oa::Viewer::destroyComposeImage() {
	if (engine_ == nullptr) return;
	VkDevice dev = static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine_).device);
	if (composeBindlessIndex_ != OA_BINDLESS_INVALID) {
		oa::EngineBindlessAccess::get(*engine_)
			.deregisterStorageImage(composeBindlessIndex_);
		composeBindlessIndex_ = OA_BINDLESS_INVALID;
	}
	if (composeView_ != nullptr) {
		oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkDestroyImageView(dev, static_cast<VkImageView>(composeView_), nullptr);
		composeView_ = nullptr;
	}
	if (composeImage_ != nullptr) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(oa::EngineAllocatorAccess::get(*engine_).allocator),
			static_cast<VkImage>(composeImage_),
			static_cast<OaVmaAllocation>(composeAllocation_));
		composeImage_ = nullptr;
		composeAllocation_ = nullptr;
	}
	composeWidth_ = 0;
	composeHeight_ = 0;
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

void oa::Viewer::beginFrame(oa::F32 inDeltaMs) {
	renderDependency_ = {};
	ui_.beginFrame(inDeltaMs, {
		0,
		0,
		static_cast<oa::I32>(composeWidth_),
		static_cast<oa::I32>(composeHeight_),
	}, oa::max(0.01F,
		(windowPixelScaleX_ + windowPixelScaleY_) * 0.5F));
}

oa::Status oa::Viewer::routeUiEvents(oa::Span<const oa::UiEvent> inEvents) {
	for (const oa::UiEvent& e : inEvents) {
		// Popups and focused controls own their events before viewport/live-source
		// navigation and application shortcuts. Unrelated events preserve the
		// former navigation-then-action ordering.
		if (ui_.routeEvent(e)) continue;
		OA_RETURN_IF_ERROR(routeEvent(e));
		(void)input_.dispatch(e);
	}
	return oa::Status::ok();
}

oa::Status oa::Viewer::recordRender(VkCommandBuffer inCmd) {
	if (presenter_ == nullptr or composeImage_ == nullptr
		or composeView_ == nullptr or composeBindlessIndex_ == OA_BINDLESS_INVALID) {
		return oa::Status::ok();
	}

	// The prior presentation submission reads this persistent compose image as
	// a transfer source and restores GENERAL for the next compute pass. queue
	// order is an execution dependency, not a memory dependency: make all prior
	// reads/writes available before this frame overwrites the image with clear.
	VkImageSubresourceRange fullRange{};
	fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	fullRange.levelCount = 1;
	fullRange.layerCount = 1;
	VkImageMemoryBarrier reuseBarrier{};
	reuseBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	reuseBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	reuseBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	reuseBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	reuseBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	reuseBarrier.image               = static_cast<VkImage>(composeImage_);
	reuseBarrier.subresourceRange    = fullRange;
	reuseBarrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT
		| VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	reuseBarrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCmdPipelineBarrier(inCmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &reuseBarrier);

	// clear compose image to dark background each frame so letterbox bars and
	// any uncovered regions don't contain stale data from prior frames.
	VkClearColorValue bg{};
	bg.float32[0] = config_.style.background.r;
	bg.float32[1] = config_.style.background.g;
	bg.float32[2] = config_.style.background.b;
	bg.float32[3] = config_.style.background.a;
	oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCmdClearColorImage(inCmd, static_cast<VkImage>(composeImage_),
		VK_IMAGE_LAYOUT_GENERAL, &bg, 1, &fullRange);

	// Barrier: clear write → compute shader write.
	VkImageMemoryBarrier clearBarrier{};
	clearBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	clearBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.image               = static_cast<VkImage>(composeImage_);
	clearBarrier.subresourceRange    = fullRange;
	clearBarrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	clearBarrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	oa::EngineDeviceAccess::get(*engine_).deviceDispatch.vkCmdPipelineBarrier(inCmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &clearBarrier);

	return ui_.recordRender(inCmd, composeBindlessIndex_);
}

oa::Status oa::Viewer::resize(oa::U32 inWidth, oa::U32 inHeight) {
	if (presenter_ == nullptr) return oa::Status::ok();
	if (inWidth == 0 or inHeight == 0) return oa::Status::ok();
	OA_RETURN_IF_ERROR(presenter_->syncGraphicsBatch());

	// Presenter recreates its swapchain to match the new pixel size.
	if (not presenter_->recreateSwapchain(VkExtent2D{ inWidth, inHeight })) {
		return oa::Status::error("oa::Viewer::resize: swapchain recreation failed");
	}

	// Rebuild the compose image to match the new extent.
	destroyComposeImage();
	if (auto s = buildComposeImage(inWidth, inHeight); not s.isOk()) return s;

	ui_.updateBlitImage(composeView_);
	refreshWindowDecorationScale();
	return oa::Status::ok();
}

oa::Status oa::Viewer::present() {
	if (presenter_ == nullptr or not presenter_->isPresentationReady()) return oa::Status::ok();

	auto& swap = presenter_->swapchain();
	oa::Presenter::AcquireResult acquired;
	if (not presenter_->acquireSwapchainImage(swap, acquired)
		or acquired.recreated) {
		// acquire failed (zero-size window / surface lost / swapchain
		// recreated). Nothing to present this frame.
		return oa::Status::ok();
	}

	oa::Presenter::PresentArgs args;
	args.blitSrcImage  = static_cast<VkImage>(composeImage_);
	args.blitSrcLayout = static_cast<oa::I32>(VK_IMAGE_LAYOUT_GENERAL);
	args.blitSrcWidth  = composeWidth_;
	args.blitSrcHeight = composeHeight_;
	args.filter        = config_.presentFilter;
	if (renderCompletion_.isValid()) {
		const oavk::TimelineWait wait =
			oa::EventAccess::timelineWait(renderCompletion_);
		args.waitTimelineSemaphore = wait.semaphore;
		args.waitTimelineValue = wait.value;
	}
	if (not presenter_->presentSwapchainImage(
		swap, acquired.imageIndex, acquired.frameSlot, args)) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"oa::Viewer: presentation failed");
	}
	return oa::Status::ok();
}

void oa::Viewer::setRenderDependency(const oa::Event& inEvent) {
	if (not inEvent.isValid()) return;
	renderDependency_ = inEvent;
}

oa::Status oa::Viewer::setRenderCompletion(const oa::Event& inCompletion) {
	renderCompletion_ = inCompletion;
	return ui_.markFrameSubmitted(inCompletion);
}

oa::U32 oa::Viewer::width() const noexcept {
	return composeWidth_ != 0 ? composeWidth_ : config_.width;
}

oa::U32 oa::Viewer::height() const noexcept {
	return composeHeight_ != 0 ? composeHeight_ : config_.height;
}

void oa::Viewer::endFrame() {
	ui_.endFrame();
}
