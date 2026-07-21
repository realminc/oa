// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Ui/Viewer.h>
#include <Oa/Ui/Image.h>
#include <Oa/Core/Log.h>


OaStatus OaViewer::InitPresentation(
	OaPresenter& InPresenter,
	void* InSurface)
{
	Presenter_ = &InPresenter;
	Engine_    = &InPresenter.Engine();

	// Build the presenter-owned swapchain on the caller's surface.
	// Vsync preference must be set before
	// InitPresentation calls BuildSwapchainObjects (which reads it).
	InPresenter.Swapchain().Vsync = Config_.Vsync;
	if (not InPresenter.InitPresentation(
			InSurface,
			VkExtent2D{ Config_.Width, Config_.Height })) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaViewer: presenter initialization failed");
		return OaStatus::Error("OaViewer: presenter initialization failed");
	}

	if (auto s = BuildComposeImage(Config_.Width, Config_.Height); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaViewer: compose image build failed: %s",
			s.GetMessage().c_str());
		InPresenter.DetachPresentation();
		Presenter_ = nullptr;
		Engine_ = nullptr;
		return s;
	}

	if (auto s = Ui_.Init(*Engine_, Config_.Style); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaViewer: UI initialization failed: %s",
			s.GetMessage().c_str());
		DestroyComposeImage();
		InPresenter.DetachPresentation();
		Presenter_ = nullptr;
		Engine_ = nullptr;
		return s;
	}

	if (auto s = Ui_.InitBlit(ComposeView_); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaViewer: UI renderer initialization failed: %s",
			s.GetMessage().c_str());
		Ui_.Destroy();
		DestroyComposeImage();
		InPresenter.DetachPresentation();
		Presenter_ = nullptr;
		Engine_ = nullptr;
		return s;
	}

	if (auto s = TextAtlas_.Init(*Engine_); not s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaViewer: text atlas initialization failed: %s",
			s.GetMessage().c_str());
		Ui_.Destroy();
		DestroyComposeImage();
		InPresenter.DetachPresentation();
		Presenter_ = nullptr;
		Engine_ = nullptr;
		return s;
	}
	return OaStatus::Ok();
}

OaStatus OaViewer::DestroyPresentation() {
	if (Presenter_ == nullptr) return OaStatus::Ok();
	// Closing the Presenter also completes and destroys its graphics-stream
	// ring. Detaching only the swapchain would leave those device children live
	// until the borrowed Presenter wrapper dies, which is after the explicit
	// engine-close call in OaViewer::Run().
	const OaStatus presenterStatus = Presenter_->Close();
	if (not presenterStatus.IsOk()) return presenterStatus;
	Ui_.Destroy();
	TextAtlas_.Destroy();
	DestroyComposeImage();
	Presenter_ = nullptr;
	Engine_    = nullptr;
	return OaStatus::Ok();
}


// ─── BuildComposeImage / DestroyComposeImage ─────────────────────────────────

OaStatus OaViewer::BuildComposeImage(OaU32 InWidth, OaU32 InHeight) {
	if (Engine_ == nullptr) return OaStatus::Error("OaViewer: no engine");

	VkDevice dev = static_cast<VkDevice>(Engine_->Device.Device);

	VkImageCreateInfo ici{};
	ici.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ici.imageType         = VK_IMAGE_TYPE_2D;
	ici.format            = VK_FORMAT_R8G8B8A8_UNORM;
	ici.extent.width      = InWidth;
	ici.extent.height     = InHeight;
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

	const OaU32 computeFamily  = Engine_->Device.Queues.ComputeQueueFamily;
	const OaU32 graphicsFamily = Engine_->Device.Queues.GraphicsQueueFamily;
	OaU32 families[2] = { computeFamily, graphicsFamily };
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
		static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator),
		&ici, &allocCI, &img, &alloc, nullptr) != VK_SUCCESS)
	{
		return OaStatus::Error(OaStatusCode::OutOfMemory, "OaViewer: compose image allocation failed");
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
	if (vkCreateImageView(dev, &vi, nullptr, &view) != VK_SUCCESS) {
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator), img, alloc);
		return OaStatus::Error(OaStatusCode::VulkanError, "OaViewer: compose image view creation failed");
	}

	// Transition UNDEFINED → GENERAL via one-shot compute stream.
	{
		OaVkStream* s = Engine_->AcquireStream();
		if (s == nullptr) {
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator), img, alloc);
			return OaStatus::Error(OaStatusCode::VulkanError, "OaViewer: stream acquisition failed");
		}
		if (auto st = s->Begin(Engine_->Device); not st.IsOk()) {
			Engine_->ReleaseStream(s);
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator), img, alloc);
			return st;
		}
		VkCommandBuffer cmd = static_cast<VkCommandBuffer>(s->CommandBuffer);
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
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
		if (auto st = s->SubmitAndWait(*Engine_); not st.IsOk()) {
			Engine_->ReleaseStream(s);
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator), img, alloc);
			return st;
		}
		Engine_->ReleaseStream(s);
	}

	OaU32 bindlessIndex = Engine_->Bindless.RegisterStorageImage(Engine_->Device, view, VK_IMAGE_LAYOUT_GENERAL);
	if (bindlessIndex == OA_BINDLESS_INVALID) {
		vkDestroyImageView(dev, view, nullptr);
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator), img, alloc);
		return OaStatus::Error(OaStatusCode::ResourceExhausted, "OaViewer: compose bindless registration failed");
	}

	ComposeImage_ = img;
	ComposeView_ = view;
	ComposeAllocation_ = alloc;
	ComposeBindlessIndex_ = bindlessIndex;
	ComposeWidth_ = InWidth;
	ComposeHeight_ = InHeight;
	return OaStatus::Ok();
}

void OaViewer::DestroyComposeImage() {
	if (Engine_ == nullptr) return;
	VkDevice dev = static_cast<VkDevice>(Engine_->Device.Device);
	if (ComposeBindlessIndex_ != OA_BINDLESS_INVALID) {
		Engine_->Bindless.DeregisterStorageImage(ComposeBindlessIndex_);
		ComposeBindlessIndex_ = OA_BINDLESS_INVALID;
	}
	if (ComposeView_ != nullptr) {
		vkDestroyImageView(dev, static_cast<VkImageView>(ComposeView_), nullptr);
		ComposeView_ = nullptr;
	}
	if (ComposeImage_ != nullptr) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(Engine_->Allocator.Allocator),
			static_cast<VkImage>(ComposeImage_),
			static_cast<OaVmaAllocation>(ComposeAllocation_));
		ComposeImage_ = nullptr;
		ComposeAllocation_ = nullptr;
	}
	ComposeWidth_ = 0;
	ComposeHeight_ = 0;
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

void OaViewer::BeginFrame(OaF32 InDeltaMs) {
	RenderDependencySemaphore_ = nullptr;
	RenderDependencyValue_ = 0;
	Ui_.BeginFrame(InDeltaMs);
}

void OaViewer::RouteUiEvents(OaSpan<const OaUiEvent> InEvents) {
	for (const OaUiEvent& e : InEvents) {
		if (Input_.Dispatch(e)) continue;
		(void)Ui_.RouteEvent(e);
	}
}

void OaViewer::RecordRender(VkCommandBuffer InCmd) {
	if (Presenter_ == nullptr or ComposeImage_ == nullptr
		or ComposeView_ == nullptr or ComposeBindlessIndex_ == OA_BINDLESS_INVALID) {
		return;
	}

	// The prior presentation submission reads this persistent compose image as
	// a transfer source and restores GENERAL for the next compute pass. Queue
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
	reuseBarrier.image               = static_cast<VkImage>(ComposeImage_);
	reuseBarrier.subresourceRange    = fullRange;
	reuseBarrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT
		| VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	reuseBarrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	vkCmdPipelineBarrier(InCmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &reuseBarrier);

	// Clear compose image to dark background each frame so letterbox bars and
	// any uncovered regions don't contain stale data from prior frames.
	VkClearColorValue bg{};
	bg.float32[0] = Config_.Style.Background.R;
	bg.float32[1] = Config_.Style.Background.G;
	bg.float32[2] = Config_.Style.Background.B;
	bg.float32[3] = Config_.Style.Background.A;
	vkCmdClearColorImage(InCmd, static_cast<VkImage>(ComposeImage_),
		VK_IMAGE_LAYOUT_GENERAL, &bg, 1, &fullRange);

	// Barrier: clear write → compute shader write.
	VkImageMemoryBarrier clearBarrier{};
	clearBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	clearBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.image               = static_cast<VkImage>(ComposeImage_);
	clearBarrier.subresourceRange    = fullRange;
	clearBarrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	clearBarrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(InCmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &clearBarrier);

	Ui_.RecordRender(InCmd, ComposeBindlessIndex_);
}

OaStatus OaViewer::Resize(OaU32 InWidth, OaU32 InHeight) {
	if (Presenter_ == nullptr) return OaStatus::Ok();
	if (InWidth == 0 or InHeight == 0) return OaStatus::Ok();
	OA_RETURN_IF_ERROR(Presenter_->SyncGraphicsBatch());

	// Presenter recreates its swapchain to match the new pixel size.
	if (not Presenter_->RecreateSwapchain(VkExtent2D{ InWidth, InHeight })) {
		return OaStatus::Error("OaViewer::Resize: swapchain recreation failed");
	}

	// Rebuild the compose image to match the new extent.
	DestroyComposeImage();
	if (auto s = BuildComposeImage(InWidth, InHeight); not s.IsOk()) return s;

	Ui_.UpdateBlitImage(ComposeView_);
	return OaStatus::Ok();
}

OaStatus OaViewer::Present() {
	if (Presenter_ == nullptr or not Presenter_->IsPresentationReady()) return OaStatus::Ok();

	auto& swap = Presenter_->Swapchain();
	OaPresenter::AcquireResult acquired;
	if (not Presenter_->AcquireSwapchainImage(swap, acquired)
		or acquired.Recreated) {
		// Acquire failed (zero-size window / surface lost / swapchain
		// recreated). Nothing to present this frame.
		return OaStatus::Ok();
	}

	OaPresenter::PresentArgs args;
	args.BlitSrcImage  = static_cast<VkImage>(ComposeImage_);
	args.BlitSrcLayout = static_cast<OaI32>(VK_IMAGE_LAYOUT_GENERAL);
	args.BlitSrcWidth  = ComposeWidth_;
	args.BlitSrcHeight = ComposeHeight_;
	args.Filter        = Config_.PresentFilter;
	if (RenderCompletionSemaphore_.Semaphore != nullptr and RenderCompletionValue_ > 0) {
		args.WaitTimelineSemaphore = RenderCompletionSemaphore_.Semaphore;
		args.WaitTimelineValue = RenderCompletionValue_;
	}
	if (not Presenter_->PresentSwapchainImage(
		swap, acquired.ImageIndex, acquired.FrameSlot, args)) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"OaViewer: presentation failed");
	}
	return OaStatus::Ok();
}

void OaViewer::SetRenderDependency(const OaEvent& InEvent) {
	if (not InEvent.IsValid()) return;
	const OaVkTimelineWait wait = InEvent.TimelineWait();
	if (wait.Semaphore != nullptr and wait.Value > RenderDependencyValue_) {
		RenderDependencySemaphore_ = wait.Semaphore;
		RenderDependencyValue_ = wait.Value;
	}
}

void OaViewer::SetRenderCompletion(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	RenderCompletionSemaphore_ = InSemaphore;
	RenderCompletionValue_ = InValue;
	Ui_.MarkFrameSubmitted(InSemaphore, InValue);
}

OaU32 OaViewer::Width() const noexcept {
	return ComposeWidth_ != 0 ? ComposeWidth_ : Config_.Width;
}

OaU32 OaViewer::Height() const noexcept {
	return ComposeHeight_ != 0 ? ComposeHeight_ : Config_.Height;
}

void OaViewer::EndFrame() {
	Ui_.EndFrame();
}
