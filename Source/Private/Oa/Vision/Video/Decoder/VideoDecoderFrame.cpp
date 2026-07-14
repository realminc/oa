// OA Vision — video frame pool and frame-level bridges.

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVma.h>

// Video frame pool implementation
OaVideoFramePool::OaVideoFramePool(OaVideoFramePool&& InOther) noexcept
{
	MoveFrom(OaStdMove(InOther));
}

OaVideoFramePool& OaVideoFramePool::operator=(OaVideoFramePool&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		MoveFrom(OaStdMove(InOther));
	}
	return *this;
}

OaVideoFramePool::~OaVideoFramePool()
{
	Destroy();
}

void OaVideoFramePool::MoveFrom(OaVideoFramePool&& InOther) noexcept
{
	Frames_ = OaStdMove(InOther.Frames_);
	InUse_ = OaStdMove(InOther.InUse_);
	Allocations_ = OaStdMove(InOther.Allocations_);
	Rt_ = InOther.Rt_;
	InOther.Rt_ = nullptr;
}

OaResult<OaVideoFramePool> OaVideoFramePool::Create(
	OaEngine& InRt,
	OaU32 InWidth,
	OaU32 InHeight,
	OaU32 InPoolSize)
{
	OaVideoFramePool pool;
	pool.Rt_ = &InRt;

	if (InWidth == 0 || InHeight == 0 || InPoolSize == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Video frame pool dimensions and size must be non-zero");
	}

	auto& vkEngine = static_cast<OaComputeEngine&>(InRt);
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	pool.Frames_.Reserve(InPoolSize);
	pool.InUse_.Reserve(InPoolSize);
	pool.Allocations_.Reserve(InPoolSize);

	for (OaU32 i = 0; i < InPoolSize; ++i) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		imageInfo.extent.width = InWidth;
		imageInfo.extent.height = InHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkImage image = VK_NULL_HANDLE;
		OaVmaAllocation allocation = VK_NULL_HANDLE;
		OaVmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
		VkResult result = OaVmaCreateImage(
			static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
			&imageInfo,
			&allocInfo,
			&image,
			&allocation,
			nullptr);
		if (result != VK_SUCCESS) {
			pool.Destroy();
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create video frame pool image");
		}

		VkImageView imageView = VK_NULL_HANDLE;
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = imageInfo.format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		result = vkCreateImageView(device, &viewInfo, nullptr, &imageView);
		if (result != VK_SUCCESS) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
				image,
				allocation);
			pool.Destroy();
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create video frame pool image view");
		}

		OaVideoFrame frame = {};
		frame.Image = image;
		frame.ImageView = imageView;
		frame.Format = imageInfo.format;
		frame.Width = InWidth;
		frame.Height = InHeight;
		frame.PresentationTimestamp = 0;
		frame.IsRgb = false;
		pool.Frames_.PushBack(frame);
		pool.InUse_.PushBack(false);
		pool.Allocations_.PushBack(allocation);
	}

	return pool;
}

OaVideoFrame OaVideoFramePool::Acquire()
{
	for (OaUsize i = 0; i < Frames_.Size(); ++i) {
		if (!InUse_[i]) {
			InUse_[i] = true;
			return Frames_[i];
		}
	}
	OaVideoFrame frame = {};
	return frame;
}

void OaVideoFramePool::Release(const OaVideoFrame& InFrame)
{
	for (OaUsize i = 0; i < Frames_.Size(); ++i) {
		if (Frames_[i].Image == InFrame.Image) {
			InUse_[i] = false;
			return;
		}
	}
}

void OaVideoFramePool::Destroy()
{
	if (!Rt_) {
		return;
	}

	auto& vkEngine = static_cast<OaComputeEngine&>(*Rt_);
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	for (OaUsize i = 0; i < Frames_.Size(); ++i) {
		if (Frames_[i].ImageView) {
			vkDestroyImageView(device, Frames_[i].ImageView, nullptr);
			Frames_[i].ImageView = VK_NULL_HANDLE;
		}
		void* allocation = i < Allocations_.Size() ? Allocations_[i] : nullptr;
		if (Frames_[i].Image && allocation) {
			OaVmaDestroyImage(
				static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
				Frames_[i].Image,
				static_cast<OaVmaAllocation>(allocation));
			Frames_[i].Image = VK_NULL_HANDLE;
		}
	}
	Frames_.Clear();
	InUse_.Clear();
	Allocations_.Clear();
	Rt_ = nullptr;
}

