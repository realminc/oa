// OA Vision — video frame pool and frame-level bridges.

#include <oa/vision/videoDecoder.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <vma/vma.hpp>

// Video frame pool implementation
oa::VideoFramePool::VideoFramePool(oa::VideoFramePool&& inOther) noexcept
{
	moveFrom(oa::move(inOther));
}

oa::VideoFramePool& oa::VideoFramePool::operator=(oa::VideoFramePool&& inOther) noexcept
{
	if (this != &inOther) {
		reset_();
		moveFrom(oa::move(inOther));
	}
	return *this;
}

oa::VideoFramePool::~VideoFramePool()
{
	reset_();
}

void oa::VideoFramePool::moveFrom(oa::VideoFramePool&& inOther) noexcept
{
	frames_ = oa::move(inOther.frames_);
	inUse_ = oa::move(inOther.inUse_);
	allocations_ = oa::move(inOther.allocations_);
	rt_ = inOther.rt_;
	inOther.rt_ = nullptr;
}

oa::Result<oa::VideoFramePool> oa::VideoFramePool::create(
	oa::Engine& inRt,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U32 inPoolSize)
{
	oa::VideoFramePool pool;
	pool.rt_ = &inRt;

	if (inWidth == 0 || inHeight == 0 || inPoolSize == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Video frame pool dimensions and size must be non-zero");
	}

	auto& vkEngine = inRt;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	pool.frames_.reserve(inPoolSize);
	pool.inUse_.reserve(inPoolSize);
	pool.allocations_.reserve(inPoolSize);

	for (oa::U32 i = 0; i < inPoolSize; ++i) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		imageInfo.extent.width = inWidth;
		imageInfo.extent.height = inHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VkImage image = VK_NULL_HANDLE;
		vma::Allocation allocation = VK_NULL_HANDLE;
		vma::AllocationCreateInfo allocInfo = {};
		allocInfo.usage = vma::memoryUsageGpuOnly;
		VkResult result = vma::createImage(
			static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
			&imageInfo,
			&allocInfo,
			&image,
			&allocation,
			nullptr);
		if (result != VK_SUCCESS) {
			pool.reset_();
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create video frame pool image");
		}

		VkImageView imageView = VK_NULL_HANDLE;
		// A full multiplane view is a transfer/identity handle only. Sampling a
		// COLOR-aspect NV12 view requires a matched YCbCr conversion on both the
		// view and sampler; conversion code creates that qualified view itself.
		VkImageViewUsageCreateInfo viewUsage = {};
		viewUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
		viewUsage.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.pNext = &viewUsage;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = imageInfo.format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &imageView);
		if (result != VK_SUCCESS) {
			vma::destroyImage(
				static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
				image,
				allocation);
			pool.reset_();
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create video frame pool image view");
		}

		oa::VideoFrame frame = {};
		frame.image = image;
		frame.imageView = imageView;
		frame.format = imageInfo.format;
		frame.width = inWidth;
		frame.height = inHeight;
		frame.presentationTimestamp = 0;
		frame.isRgb = false;
		pool.frames_.pushBack(frame);
		pool.inUse_.pushBack(false);
		pool.allocations_.pushBack(allocation);
	}

	return pool;
}

oa::VideoFrame oa::VideoFramePool::acquire()
{
	for (oa::Usize i = 0; i < frames_.size(); ++i) {
		if (!inUse_[i]) {
			inUse_[i] = true;
			return frames_[i];
		}
	}
	oa::VideoFrame frame = {};
	return frame;
}

void oa::VideoFramePool::release(const oa::VideoFrame& inFrame)
{
	for (oa::Usize i = 0; i < frames_.size(); ++i) {
		if (frames_[i].image == inFrame.image) {
			inUse_[i] = false;
			return;
		}
	}
}

void oa::VideoFramePool::reset_() noexcept
{
	if (!rt_) {
		return;
	}

	auto& vkEngine = *rt_;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	for (oa::Usize i = 0; i < frames_.size(); ++i) {
		if (frames_[i].imageView) {
			oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyImageView(device, frames_[i].imageView, nullptr);
			frames_[i].imageView = VK_NULL_HANDLE;
		}
		void* allocation = i < allocations_.size() ? allocations_[i] : nullptr;
		if (frames_[i].image && allocation) {
			vma::destroyImage(
				static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
				frames_[i].image,
				static_cast<vma::Allocation>(allocation));
			frames_[i].image = VK_NULL_HANDLE;
		}
	}
	frames_.clear();
	inUse_.clear();
	allocations_.clear();
	rt_ = nullptr;
}
