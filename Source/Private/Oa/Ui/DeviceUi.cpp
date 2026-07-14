// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Image.h>
#include <Oa/Core/Log.h>


// ─── OaDeviceUi move/dtor ────────────────────────────────────────────────────────

OaDeviceUi::OaDeviceUi(OaDeviceUi&& InOther) noexcept
	: Gfx_(InOther.Gfx_)
	, Rt_(InOther.Rt_)
	, Surface_(InOther.Surface_)
	, Config_(OaStdMove(InOther.Config_))
	, Compose_(InOther.Compose_)
	, TextAtlas_(OaStdMove(InOther.TextAtlas_))
	, CanvasRenderer_(OaStdMove(InOther.CanvasRenderer_))
	, Oui_(OaStdMove(InOther.Oui_))
	, NodeGraph_(OaStdMove(InOther.NodeGraph_))
	, Canvas_(InOther.Canvas_)
	, Input_(OaStdMove(InOther.Input_))
{
	InOther.Gfx_     = nullptr;
	InOther.Rt_      = nullptr;
	InOther.Surface_ = nullptr;
	InOther.Compose_ = {};
}

OaDeviceUi& OaDeviceUi::operator=(OaDeviceUi&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		new (this) OaDeviceUi(OaStdMove(InOther));
	}
	return *this;
}

OaDeviceUi::~OaDeviceUi() { Destroy(); }


// ─── Init / Destroy ───────────────────────────────────────────────────────────

OaStatus OaDeviceUi::Init(
	OaGraphicsEngine& InEngine,
	void*               InSurface,
	const OaUiConfig& InConfig)
{
	Gfx_     = &InEngine;
	Rt_      = &InEngine;  // OaGraphicsEngine inherits OaComputeEngine
	Surface_ = InSurface;
	Config_  = InConfig;

	// Build the engine-owned swapchain on the caller's surface (FinalGlue §3.5).
	// Vsync preference must be set on the engine's swapchain BEFORE
	// InitPresentation calls BuildSwapchainObjects (which reads it).
	InEngine.Swapchain().Vsync = InConfig.Vsync;
	if (not InEngine.InitPresentation(
			InSurface,
			VkExtent2D{ InConfig.Width, InConfig.Height })) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaDeviceUi: engine.InitPresentation failed");
		return OaStatus::Error("OaDeviceUi: InitPresentation failed");
	}

	if (auto s = BuildComposeImage(InConfig.Width, InConfig.Height); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaDeviceUi: compose image build failed: %s",
			s.GetMessage().c_str());
		InEngine.DetachPresentation();
		return s;
	}

	if (auto s = Oui_.Init(*Rt_, InConfig.Style); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaDeviceUi: OaDeviceUi init failed: %s",
			s.GetMessage().c_str());
		DestroyComposeImage();
		InEngine.DetachPresentation();
		return s;
	}

	if (auto s = Oui_.InitBlit(Compose_.View); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaDeviceUi: OaDeviceUi InitBlit failed: %s",
			s.GetMessage().c_str());
		Oui_.Destroy();
		DestroyComposeImage();
		InEngine.DetachPresentation();
		return s;
	}

	if (auto s = TextAtlas_.Init(*Rt_); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App, "OaDeviceUi: TextAtlas init failed: %s",
			s.GetMessage().c_str());
		Oui_.Destroy();
		DestroyComposeImage();
		InEngine.DetachPresentation();
		return s;
	}
	if (auto s = CanvasRenderer_.Init(
		InEngine, Compose_.Width, Compose_.Height); !s.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaDeviceUi: canvas renderer init failed: %s",
			s.GetMessage().c_str());
		TextAtlas_.Destroy();
		Oui_.Destroy();
		DestroyComposeImage();
		InEngine.DetachPresentation();
		return s;
	}
	return OaStatus::Ok();
}

void OaDeviceUi::Destroy() {
	if (Gfx_ == nullptr) return;
	CanvasRenderer_.Destroy();
	Oui_.Destroy();
	TextAtlas_.Destroy();
	DestroyComposeImage();
	Gfx_->DetachPresentation();
	Gfx_     = nullptr;
	Rt_      = nullptr;
	Surface_ = nullptr;
}


// ─── BuildComposeImage / DestroyComposeImage ─────────────────────────────────

OaStatus OaDeviceUi::BuildComposeImage(OaU32 InWidth, OaU32 InHeight) {
	if (Rt_ == nullptr) return OaStatus::Error("OaDeviceUi: no engine");

	VkDevice dev = static_cast<VkDevice>(Rt_->Device.Device);

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

	const OaU32 computeFamily  = Rt_->Device.Queues.ComputeQueueFamily;
	const OaU32 graphicsFamily = Rt_->Device.Queues.GraphicsQueueFamily;
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
		static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator),
		&ici, &allocCI, &img, &alloc, nullptr) != VK_SUCCESS)
	{
		return OaStatus::Error(OaStatusCode::OutOfMemory, "OaDeviceUi: compose image alloc failed");
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
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), img, alloc);
		return OaStatus::Error(OaStatusCode::VulkanError, "OaDeviceUi: compose image view failed");
	}

	// Transition UNDEFINED → GENERAL via one-shot compute stream.
	{
		OaVkStream* s = Rt_->AcquireStream();
		if (s == nullptr) {
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), img, alloc);
			return OaStatus::Error(OaStatusCode::VulkanError, "OaDeviceUi: stream acquire failed");
		}
		if (auto st = s->Begin(Rt_->Device); !st.IsOk()) {
			Rt_->ReleaseStream(s);
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), img, alloc);
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
		if (auto st = s->SubmitAndWait(*Rt_); !st.IsOk()) {
			Rt_->ReleaseStream(s);
			vkDestroyImageView(dev, view, nullptr);
			OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), img, alloc);
			return st;
		}
		Rt_->ReleaseStream(s);
	}

	OaU32 bindlessIndex = Rt_->Bindless.RegisterStorageImage(Rt_->Device, view, VK_IMAGE_LAYOUT_GENERAL);
	if (bindlessIndex == OA_BINDLESS_INVALID) {
		vkDestroyImageView(dev, view, nullptr);
		OaVmaDestroyImage(static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator), img, alloc);
		return OaStatus::Error(OaStatusCode::ResourceExhausted, "OaDeviceUi: compose bindless registration failed");
	}

	Compose_.Image         = img;
	Compose_.View          = view;
	Compose_.Allocation    = alloc;
	Compose_.BindlessIndex = bindlessIndex;
	Compose_.Width         = InWidth;
	Compose_.Height        = InHeight;
	return OaStatus::Ok();
}

void OaDeviceUi::DestroyComposeImage() {
	if (Rt_ == nullptr) return;
	VkDevice dev = static_cast<VkDevice>(Rt_->Device.Device);
	if (Compose_.BindlessIndex != OA_BINDLESS_INVALID) {
		Rt_->Bindless.DeregisterStorageImage(Compose_.BindlessIndex);
		Compose_.BindlessIndex = OA_BINDLESS_INVALID;
	}
	if (Compose_.View != VK_NULL_HANDLE) {
		vkDestroyImageView(dev, Compose_.View, nullptr);
		Compose_.View = VK_NULL_HANDLE;
	}
	if (Compose_.Image != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(Rt_->Allocator.Allocator),
			Compose_.Image,
			static_cast<OaVmaAllocation>(Compose_.Allocation));
		Compose_.Image      = VK_NULL_HANDLE;
		Compose_.Allocation = nullptr;
	}
	Compose_.Width  = 0;
	Compose_.Height = 0;
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

void OaDeviceUi::BeginFrame(OaF32 InDeltaMs) {
	RenderDependencySemaphore_ = nullptr;
	RenderDependencyValue_ = 0;
	Canvas_.StepAnimation(InDeltaMs);
	CanvasRenderer_.BeginFrame();
	Oui_.BeginFrame(InDeltaMs);
}

void OaDeviceUi::RouteEvents(OaSpan<const OaUiEvent> InEvents) {
	for (const OaUiEvent& e : InEvents) {
		if (Input_.Dispatch(e)) continue;
		if (Oui_.RouteEvent(e)) continue;
		// Node graph / canvas events could be dispatched here in future.
	}
}

void OaDeviceUi::RecordRender(VkCommandBuffer InCmd) {
	if (Gfx_ == nullptr or not Compose_.IsValid()) return;

	// Clear compose image to dark background each frame so letterbox bars and
	// any uncovered regions don't contain stale data from prior frames.
	VkImageSubresourceRange fullRange{};
	fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	fullRange.levelCount = 1;
	fullRange.layerCount = 1;
	VkClearColorValue bg{};
	bg.float32[0] = 0.07F;
	bg.float32[1] = 0.07F;
	bg.float32[2] = 0.07F;
	bg.float32[3] = 1.0F;
	vkCmdClearColorImage(InCmd, Compose_.Image,
		VK_IMAGE_LAYOUT_GENERAL, &bg, 1, &fullRange);

	if (CanvasRenderer_.HasDraws()) {
		CanvasRenderer_.Record(InCmd, Compose_.Image);
		Oui_.RecordRender(InCmd, Compose_.BindlessIndex);
		return;
	}

	// Barrier: clear write → compute shader write.
	VkImageMemoryBarrier clearBarrier{};
	clearBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	clearBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
	clearBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	clearBarrier.image               = Compose_.Image;
	clearBarrier.subresourceRange    = fullRange;
	clearBarrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
	clearBarrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(InCmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &clearBarrier);

	Oui_.RecordRender(InCmd, Compose_.BindlessIndex);
}

OaStatus OaDeviceUi::Resize(OaU32 InWidth, OaU32 InHeight) {
	if (Gfx_ == nullptr) return OaStatus::Ok();
	if (InWidth == 0 or InHeight == 0) return OaStatus::Ok();
	OA_RETURN_IF_ERROR(Gfx_->SyncGraphicsBatch());

	// Engine recreates its swapchain to match the new pixel size.
	if (not Gfx_->RecreateSwapchain(VkExtent2D{ InWidth, InHeight })) {
		return OaStatus::Error("OaDeviceUi::Resize: RecreateSwapchain failed");
	}

	// Rebuild the compose image to match the new extent.
	DestroyComposeImage();
	if (auto s = BuildComposeImage(InWidth, InHeight); !s.IsOk()) return s;

	CanvasRenderer_.SetTarget(Compose_.Width, Compose_.Height);
	Oui_.UpdateBlitImage(Compose_.View);
	return OaStatus::Ok();
}

OaStatus OaDeviceUi::Present() {
	if (Gfx_ == nullptr or not Gfx_->IsPresentationReady()) return OaStatus::Ok();

	auto& swap = Gfx_->Swapchain();

	auto& ctx = OaContext::GetDefault();
	OaTexture target = ctx.RecordAcquire(swap);
	if (not target.IsImageBacked()) {
		// Acquire failed (zero-size window / surface lost / swapchain
		// recreated). Nothing to present this frame.
		return OaStatus::Ok();
	}

	// Wrap the compose image as a non-owning image-backed OaTexture so ctx's
	// VkImage→VkImage blit path (Step 3c.1) can handle the compose→swap copy.
	OaTexture composeAsTex;
	composeAsTex.Image  = Compose_.Image;
	composeAsTex.View   = Compose_.View;
	composeAsTex.Format = static_cast<OaI32>(VK_FORMAT_R8G8B8A8_UNORM);
	composeAsTex.Layout = static_cast<OaI32>(VK_IMAGE_LAYOUT_GENERAL);
	composeAsTex.Width  = static_cast<OaI32>(Compose_.Width);
	composeAsTex.Height = static_cast<OaI32>(Compose_.Height);

	OaBlitDesc desc;
	desc.Src    = &composeAsTex;
	desc.Dst    = &target;
	desc.Filter = Config_.PresentFilter;
	ctx.RecordBlit(desc);
	if (RenderCompletionSemaphore_.Semaphore != nullptr && RenderCompletionValue_ > 0) {
		ctx.RecordPresent(
			swap,
			target,
			RenderCompletionSemaphore_,
			RenderCompletionValue_);
	} else {
		ctx.RecordPresent(swap, target);
	}

	ctx.SubmitPresent();
	return OaStatus::Ok();
}

OaTexture OaDeviceUi::ComposeTexture() const noexcept {
	OaTexture texture;
	texture.Image = Compose_.Image;
	texture.View = Compose_.View;
	texture.Format = static_cast<OaI32>(VK_FORMAT_R8G8B8A8_UNORM);
	texture.Layout = static_cast<OaI32>(VK_IMAGE_LAYOUT_GENERAL);
	texture.Width = static_cast<OaI32>(Compose_.Width);
	texture.Height = static_cast<OaI32>(Compose_.Height);
	return texture;
}

void OaDeviceUi::EndFrame() {
	Oui_.EndFrame();
	NodeGraph_.Tick(0.0F);
}
