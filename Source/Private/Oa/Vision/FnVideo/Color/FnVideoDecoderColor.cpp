// OA Vision — decoded-frame readback, NV12→RGB conversion, DPB layout restore.
// FnVideo/Color impl for OaVideoDecoder; public API remains on VideoDecoder.h.

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Core/FnMatrix.h>

static OaF32 ClampUnit(OaF32 InValue)
{
	if (InValue < 0.0F) return 0.0F;
	if (InValue > 1.0F) return 1.0F;
	return InValue;
}

static VkSamplerYcbcrModelConversion ToVkYcbcrModel(OaYCbCrModel InColorSpace, OaU32 InWidth, OaU32 InHeight) {
	if (InColorSpace == OaYCbCrModel::BT2020) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
	}
	if (InColorSpace == OaYCbCrModel::BT709 || InWidth >= 1280 || InHeight >= 720) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
	return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
}

static OaU32 ToVisionColorSpace(OaYCbCrModel InColorSpace, OaU32 InWidth, OaU32 InHeight)
{
	if (InColorSpace == OaYCbCrModel::BT2020) {
		return 2;
	}
	if (InColorSpace == OaYCbCrModel::BT709 || InWidth >= 1280 || InHeight >= 720) {
		return 1;
	}
	return 0;
}

OaResult<OaVec<OaU8>> OaVideoDecoder::ReadbackLuma(const OaVideoFrame& InFrame)
{
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0]) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InFrame.Image || InFrame.Width == 0 || InFrame.Height == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video frame for luma readback");
	}
	OA_RETURN_IF_ERROR(InFrame.Ready.Wait());

	auto& vkEngine = *Rt_;
	auto allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);
	const OaU64 byteSize = static_cast<OaU64>(InFrame.Width) * InFrame.Height;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = OA_VMA_MEMORY_USAGE_CPU_ONLY;
	allocInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

	VkBuffer readbackBuffer = VK_NULL_HANDLE;
	OaVmaAllocation readbackAllocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo readbackInfo = {};
	VkResult result = OaVmaCreateBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "Failed to allocate video luma readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			OaVmaDestroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	bool isOutput = false;
	OaU32 imageIndex = 0;
	for (OaUsize i = 0; i < OutputImages_.Size(); ++i) {
		if (OutputImages_[i] == InFrame.Image) {
			oldLayout = OutputImageLayouts_[i];
			isOutput = true;
			imageIndex = static_cast<OaU32>(i);
			break;
		}
	}
	if (!isOutput && Dpb_.GetImage() == InFrame.Image) {
		oldLayout = DpbImageLayouts_[InFrame.ArrayLayer];
	}
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		cleanup();
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unknown video frame image layout");
	}

	OaVkStream* stream = vkEngine.AcquireStream();
	if (stream == nullptr) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to acquire compute stream for luma readback");
	}
	OaStatus beginStatus = stream->Begin(vkEngine.Device);
	if (!beginStatus.IsOk()) {
		vkEngine.ReleaseStream(stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);

	VkImageMemoryBarrier2 toTransfer = {};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	const bool sameQueueFamily =
		vkEngine.Device.Queues.VideoDecodeQueueFamily == vkEngine.Device.Queues.ComputeQueueFamily;
	toTransfer.srcStageMask = sameQueueFamily
		? (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
			: VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR)
		: VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
	toTransfer.srcAccessMask = sameQueueFamily
		? (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
			: VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR)
		: 0;
	toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toTransfer.oldLayout = oldLayout;
	toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.image = InFrame.Image;
	toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	toTransfer.subresourceRange.baseMipLevel = 0;
	toTransfer.subresourceRange.levelCount = 1;
	toTransfer.subresourceRange.baseArrayLayer = InFrame.ArrayLayer;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copy = {};
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copy.imageSubresource.mipLevel = 0;
	copy.imageSubresource.baseArrayLayer = InFrame.ArrayLayer;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = {InFrame.Width, InFrame.Height, 1};
	vkCmdCopyImageToBuffer(
		cb,
		InFrame.Image,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		readbackBuffer,
		1,
		&copy);

	VkBufferMemoryBarrier2 hostBarrier = {};
	hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	hostBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.buffer = readbackBuffer;
	hostBarrier.offset = 0;
	hostBarrier.size = byteSize;

	VkDependencyInfo hostDep = {};
	hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDep.bufferMemoryBarrierCount = 1;
	hostDep.pBufferMemoryBarriers = &hostBarrier;
	vkCmdPipelineBarrier2(cb, &hostDep);

	OaStatus submitStatus = stream->Submit(vkEngine);
	if (submitStatus.IsOk()) {
		submitStatus = stream->Synchronize(vkEngine.Device);
	}
	vkEngine.ReleaseStream(stream);
	if (!submitStatus.IsOk()) {
		cleanup();
		return submitStatus;
	}

	result = OaVmaInvalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to invalidate video luma readback allocation");
	}

	OaVec<OaU8> data(static_cast<OaUsize>(byteSize));
	OaMemcpy(data.Data(), readbackInfo.pMappedData, static_cast<OaUsize>(byteSize));
	cleanup();

	if (isOutput) {
		OutputImageLayouts_[imageIndex] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else {
		DpbImageLayouts_[InFrame.ArrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	return data;
}

OaResult<OaVec<OaU8>> OaVideoDecoder::ReadbackNv12(const OaVideoFrame& InFrame) {
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0]) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InFrame.Image || InFrame.Width == 0 || InFrame.Height == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video frame for NV12 readback");
	}
	if (InFrame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ReadbackNv12 requires VK_FORMAT_G8_B8R8_2PLANE_420_UNORM");
	}
	OA_RETURN_IF_ERROR(InFrame.Ready.Wait());

	auto& vkEngine = *Rt_;
	VkImage readbackImage = InFrame.Image;
	OaU32 readbackLayer = InFrame.ArrayLayer;
	bool isSampleStaging = false;
	if (UseSampleStaging_ && InFrame.Image == Dpb_.GetImage()) {
		OA_RETURN_IF_ERROR(CopyDpbLayerToSampleImage(InFrame));
		if (InFrame.ArrayLayer >= SampleImages_.Size()) {
			return OaStatus::Error(
				OaStatusCode::InvalidArgument,
				"NV12 readback staging layer is unavailable");
		}
		readbackImage = SampleImages_[InFrame.ArrayLayer];
		readbackLayer = 0;
		isSampleStaging = true;
	}
	auto allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);
	const OaU64 lumaBytes = static_cast<OaU64>(InFrame.Width) * InFrame.Height;
	const OaU64 chromaBytes = lumaBytes / 2;
	const OaU64 byteSize = lumaBytes + chromaBytes;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = OA_VMA_MEMORY_USAGE_CPU_ONLY;
	allocInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

	VkBuffer readbackBuffer = VK_NULL_HANDLE;
	OaVmaAllocation readbackAllocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo readbackInfo = {};
	VkResult result = OaVmaCreateBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "Failed to allocate video NV12 readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			OaVmaDestroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	bool isOutput = false;
	OaU32 imageIndex = 0;
	for (OaUsize i = 0; i < OutputImages_.Size(); ++i) {
		if (OutputImages_[i] == readbackImage) {
			oldLayout = OutputImageLayouts_[i];
			isOutput = true;
			imageIndex = static_cast<OaU32>(i);
			break;
		}
	}
	if (isSampleStaging) {
		oldLayout = SampleImageLayouts_[InFrame.ArrayLayer];
	} else if (!isOutput && Dpb_.GetImage() == readbackImage) {
		oldLayout = DpbImageLayouts_[InFrame.ArrayLayer];
	}
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		cleanup();
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unknown video frame image layout");
	}

	OaVkStream* stream = vkEngine.AcquireStream();
	if (stream == nullptr) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to acquire compute stream for NV12 readback");
	}
	OaStatus beginStatus = stream->Begin(vkEngine.Device);
	if (!beginStatus.IsOk()) {
		vkEngine.ReleaseStream(stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);

	VkImageMemoryBarrier2 toTransfer = {};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	const bool sameQueueFamily =
		vkEngine.Device.Queues.VideoDecodeQueueFamily == vkEngine.Device.Queues.ComputeQueueFamily;
	if (!sameQueueFamily) {
		toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
		toTransfer.srcAccessMask = 0;
	} else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		toTransfer.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	} else {
		toTransfer.srcStageMask =
			VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		toTransfer.srcAccessMask =
			VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR |
			VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
			VK_ACCESS_2_TRANSFER_READ_BIT;
	}
	toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toTransfer.oldLayout = oldLayout;
	toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.image = readbackImage;
	toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	toTransfer.subresourceRange.baseMipLevel = 0;
	toTransfer.subresourceRange.levelCount = 1;
	const OaU32 frameLayer = readbackLayer;
	toTransfer.subresourceRange.baseArrayLayer = frameLayer;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copies[2] = {};
	copies[0].bufferOffset = 0;
	copies[0].bufferRowLength = 0;
	copies[0].bufferImageHeight = 0;
	copies[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copies[0].imageSubresource.mipLevel = 0;
	copies[0].imageSubresource.baseArrayLayer = frameLayer;
	copies[0].imageSubresource.layerCount = 1;
	copies[0].imageExtent = {InFrame.Width, InFrame.Height, 1};

	copies[1].bufferOffset = lumaBytes;
	copies[1].bufferRowLength = 0;
	copies[1].bufferImageHeight = 0;
	copies[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	copies[1].imageSubresource.mipLevel = 0;
	copies[1].imageSubresource.baseArrayLayer = frameLayer;
	copies[1].imageSubresource.layerCount = 1;
	copies[1].imageExtent = {InFrame.Width / 2, InFrame.Height / 2, 1};

	vkCmdCopyImageToBuffer(
		cb,
		readbackImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		readbackBuffer,
		2,
		copies);

	VkBufferMemoryBarrier2 hostBarrier = {};
	hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	hostBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.buffer = readbackBuffer;
	hostBarrier.offset = 0;
	hostBarrier.size = byteSize;

	VkDependencyInfo hostDep = {};
	hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDep.bufferMemoryBarrierCount = 1;
	hostDep.pBufferMemoryBarriers = &hostBarrier;
	vkCmdPipelineBarrier2(cb, &hostDep);

	OaStatus submitStatus = stream->Submit(vkEngine);
	if (submitStatus.IsOk()) {
		submitStatus = stream->Synchronize(vkEngine.Device);
	}
	vkEngine.ReleaseStream(stream);
	if (!submitStatus.IsOk()) {
		cleanup();
		return submitStatus;
	}

	result = OaVmaInvalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to invalidate video NV12 readback allocation");
	}

	OaVec<OaU8> data(static_cast<OaUsize>(byteSize));
	OaMemcpy(data.Data(), readbackInfo.pMappedData, static_cast<OaUsize>(byteSize));
	cleanup();

	if (isOutput) {
		OutputImageLayouts_[imageIndex] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else if (isSampleStaging) {
		SampleImageLayouts_[InFrame.ArrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else {
		DpbImageLayouts_[InFrame.ArrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	return data;
}

OaResult<OaVec<OaU8>> OaVideoDecoder::ReadbackRgba(const OaVideoFrame& InFrame)
{
	if (!Rt_ || !CmdBuffers_[0]) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InFrame.Image || InFrame.Width == 0 || InFrame.Height == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video frame for RGBA readback");
	}
	if (InFrame.Format != VK_FORMAT_R8G8B8A8_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ReadbackRgba requires VK_FORMAT_R8G8B8A8_UNORM");
	}

	auto& vkEngine = *Rt_;
	auto allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);
	const OaU64 byteSize = static_cast<OaU64>(InFrame.Width) * InFrame.Height * 4;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = OA_VMA_MEMORY_USAGE_CPU_ONLY;
	allocInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT | OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

	VkBuffer readbackBuffer = VK_NULL_HANDLE;
	OaVmaAllocation readbackAllocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo readbackInfo = {};
	VkResult result = OaVmaCreateBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "Failed to allocate video RGBA readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			OaVmaDestroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	OaU32 imageIndex = 0;
	bool foundImage = false;
	for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
		if (RgbImages_[i] == InFrame.Image) {
			imageIndex = static_cast<OaU32>(i);
			oldLayout = i < RgbImageLayouts_.Size() ? RgbImageLayouts_[i] : VK_IMAGE_LAYOUT_GENERAL;
			foundImage = true;
			break;
		}
	}
	if (!foundImage) {
		cleanup();
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unknown RGBA video frame image");
	}

	OaVkStream* stream = vkEngine.AcquireStream();
	if (stream == nullptr) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to acquire compute stream for RGBA readback");
	}
	OaStatus beginStatus = stream->Begin(vkEngine.Device);
	if (!beginStatus.IsOk()) {
		vkEngine.ReleaseStream(stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);

	VkImageMemoryBarrier2 toTransfer = {};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	toTransfer.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
	toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toTransfer.oldLayout = oldLayout;
	toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.image = InFrame.Image;
	toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	toTransfer.subresourceRange.baseMipLevel = 0;
	toTransfer.subresourceRange.levelCount = 1;
	toTransfer.subresourceRange.baseArrayLayer = 0;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copy = {};
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.mipLevel = 0;
	copy.imageSubresource.baseArrayLayer = 0;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = {InFrame.Width, InFrame.Height, 1};

	vkCmdCopyImageToBuffer(
		cb,
		InFrame.Image,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		readbackBuffer,
		1,
		&copy);

	VkImageMemoryBarrier2 restoreLayout = {};
	restoreLayout.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	restoreLayout.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	restoreLayout.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	restoreLayout.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	restoreLayout.dstAccessMask =
		VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
	restoreLayout.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	restoreLayout.newLayout = oldLayout;
	restoreLayout.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restoreLayout.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	restoreLayout.image = InFrame.Image;
	restoreLayout.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	restoreLayout.subresourceRange.baseMipLevel = 0;
	restoreLayout.subresourceRange.levelCount = 1;
	restoreLayout.subresourceRange.baseArrayLayer = 0;
	restoreLayout.subresourceRange.layerCount = 1;

	VkBufferMemoryBarrier2 hostBarrier = {};
	hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
	hostBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	hostBarrier.buffer = readbackBuffer;
	hostBarrier.offset = 0;
	hostBarrier.size = byteSize;

	VkDependencyInfo hostDep = {};
	hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDep.imageMemoryBarrierCount = 1;
	hostDep.pImageMemoryBarriers = &restoreLayout;
	hostDep.bufferMemoryBarrierCount = 1;
	hostDep.pBufferMemoryBarriers = &hostBarrier;
	vkCmdPipelineBarrier2(cb, &hostDep);

	OaStatus submitStatus = stream->Submit(vkEngine);
	if (submitStatus.IsOk()) {
		submitStatus = stream->Synchronize(vkEngine.Device);
	}
	vkEngine.ReleaseStream(stream);
	if (!submitStatus.IsOk()) {
		cleanup();
		return submitStatus;
	}

	result = OaVmaInvalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to invalidate video RGBA readback allocation");
	}

	OaVec<OaU8> data(static_cast<OaUsize>(byteSize));
	OaMemcpy(data.Data(), readbackInfo.pMappedData, static_cast<OaUsize>(byteSize));
	cleanup();

	if (imageIndex < RgbImageLayouts_.Size()) {
		RgbImageLayouts_[imageIndex] = oldLayout;
	}
	return data;
}

VkImageLayout OaVideoDecoder::GetFrameLayout(const OaVideoFrame& InFrame, bool& OutIsOutput, OaU32& OutImageIndex) const
{
	OutIsOutput = false;
	OutImageIndex = 0;
	for (OaUsize i = 0; i < OutputImages_.Size(); ++i) {
		if (OutputImages_[i] == InFrame.Image) {
			OutIsOutput = true;
			OutImageIndex = static_cast<OaU32>(i);
			return OutputImageLayouts_[i];
		}
	}
	if (Dpb_.GetImage() == InFrame.Image) {
		OutImageIndex = InFrame.ArrayLayer;
		return DpbImageLayouts_[InFrame.ArrayLayer];
	}
	return VK_IMAGE_LAYOUT_UNDEFINED;
}

void OaVideoDecoder::SetFrameLayout(bool InIsOutput, OaU32 InIndex, VkImageLayout InLayout)
{
	if (InIsOutput && InIndex < OutputImageLayouts_.Size()) {
		OutputImageLayouts_[InIndex] = InLayout;
		return;
	}
	if (InIndex < DpbImageLayouts_.Size()) {
		DpbImageLayouts_[InIndex] = InLayout;
	}
}

OaStatus OaVideoDecoder::RestoreDpbLayerToDecodeLayout(const OaVideoFrame& InFrame)
{
	return RestoreDpbLayerToDecodeLayoutAfter(InFrame, nullptr, 0);
}

OaStatus OaVideoDecoder::ReleaseDpbLayerForComputeCopy(const OaVideoFrame& InFrame)
{
	if (!Rt_ || !CmdBuffers_[0] || InFrame.Image != Dpb_.GetImage()) {
		return OaStatus::Ok();
	}
	const OaU32 layer = InFrame.ArrayLayer;
	if (layer >= DpbImageLayouts_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"DPB release requires a valid array layer");
	}
	if (DpbImageLayouts_[layer] == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		return OaStatus::Ok();
	}
	if (DpbImageLayouts_[layer] != VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"DPB release requires VIDEO_DECODE_DPB layout");
	}

	auto& vkEngine = *Rt_;
	auto slot = AcquireVideoCmdSlot();
	if (!slot.Status.IsOk()) {
		return slot.Status;
	}
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = vkBeginCommandBuffer(slot.cb, &beginInfo);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkBeginCommandBuffer failed for DPB release");
	}

	// Producer-side layout transition. The following timeline signal makes the
	// decode write and transition available to the compute-queue copy; therefore
	// the destination synchronization scope is intentionally NONE.
	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
	barrier.dstAccessMask = VK_ACCESS_2_NONE;
	barrier.oldLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = Dpb_.GetImage();
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = layer;
	barrier.subresourceRange.layerCount = 1;
	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(slot.cb, &dependency);

	result = vkEndCommandBuffer(slot.cb);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkEndCommandBuffer failed for DPB release");
	}
	const OaU64 signalValue = TimelineValue_ + 1;
	VkSemaphoreSubmitInfo signalInfo = {};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(TimelineSem_.Semaphore);
	signalInfo.value = signalValue;
	signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	VkCommandBufferSubmitInfo commandInfo = {};
	commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
	commandInfo.commandBuffer = slot.cb;
	VkSubmitInfo2 submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &commandInfo;
	submitInfo.signalSemaphoreInfoCount = 1;
	submitInfo.pSignalSemaphoreInfos = &signalInfo;
	result = vkQueueSubmit2(
		static_cast<VkQueue>(vkEngine.Device.Queues.VideoDecodeQueue),
		1,
		&submitInfo,
		slot.fence);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkQueueSubmit2 failed for DPB release");
	}
	TimelineValue_ = signalValue;
	DpbImageLayouts_[layer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	ReleaseVideoCmdSlot();
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::RestoreDpbLayerToDecodeLayoutAfter(
	const OaVideoFrame& InFrame,
	const OaVkTimelineSemaphore* InWaitSemaphore,
	OaU64 InWaitValue)
{
	// Only meaningful for DPB-backed NV12 frames (the shader-convert path).
	// Output-image frames don't need to flip back.
	if (!Rt_ || !CmdBuffers_[0] || !InFrame.Image) {
		return OaStatus::Ok();
	}
	if (InFrame.IsRgb || InFrame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Ok();
	}
	bool isOutput = false;
	OaU32 imageIndex = 0;
	VkImageLayout oldLayout = GetFrameLayout(InFrame, isOutput, imageIndex);
	// Distinct output images restore to DST; DPB-backed frames restore to DPB.
	const VkImageLayout decodeLayout = isOutput
		? VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
		: VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	if (oldLayout == decodeLayout) {
		return OaStatus::Ok();
	}
	if (oldLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		&& oldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"NV12 restore requires SHADER_READ_ONLY_OPTIMAL or TRANSFER_SRC_OPTIMAL layout");
	}
	const OaU32 barrierLayer = isOutput ? 0u : InFrame.ArrayLayer;

	auto& vkEngine = *Rt_;
	auto slot = AcquireVideoCmdSlot();
	if (!slot.Status.IsOk()) {
		return slot.Status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkBeginCommandBuffer failed for DPB restore");
	}

	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	auto& vkEngineRestore = *Rt_;
	const bool sameFamilyRestore = vkEngineRestore.Device.Queues.VideoDecodeQueueFamily == vkEngineRestore.Device.Queues.ComputeQueueFamily;
	if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		if (sameFamilyRestore) {
			barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
		} else {
			barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
			barrier.srcAccessMask = 0;
		}
	} else if (sameFamilyRestore) {
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	} else {
		// Concurrent sharing across different queue families. We already
		// waited on the compute dispatch's timeline semaphore, so the
		// shader read is complete. No src access sync needed on decode queue.
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
		barrier.srcAccessMask = 0;
	}
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = decodeLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = InFrame.Image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = barrierLayer;
	barrier.subresourceRange.layerCount = 1;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(cb, &dependency);

	result = vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkEndCommandBuffer failed for DPB restore");
	}
	const OaU64 signalValue = TimelineValue_ + 1;
	VkSemaphore signalSemaphore = static_cast<VkSemaphore>(TimelineSem_.Semaphore);
	VkSemaphore waitSemaphore = InWaitSemaphore
		? static_cast<VkSemaphore>(InWaitSemaphore->Semaphore)
		: VK_NULL_HANDLE;
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkTimelineSemaphoreSubmitInfo timelineInfo = {};
	timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timelineInfo.waitSemaphoreValueCount = waitSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	timelineInfo.pWaitSemaphoreValues = waitSemaphore != VK_NULL_HANDLE ? &InWaitValue : nullptr;
	timelineInfo.signalSemaphoreValueCount = 1;
	timelineInfo.pSignalSemaphoreValues = &signalValue;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = &timelineInfo;
	submitInfo.waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	submitInfo.pWaitSemaphores = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr;
	submitInfo.pWaitDstStageMask = waitSemaphore != VK_NULL_HANDLE ? &waitStage : nullptr;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cb;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &signalSemaphore;
	if (vkEngine.Device.Queues.VideoDecodeQueue == nullptr) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = vkQueueSubmit(static_cast<VkQueue>(vkEngine.Device.Queues.VideoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		// Don't reset the fence on error: it was already reset before recording and
		// the failed submit never gave it to the GPU, so resetting would leave it
		// permanently unsignaled and deadlock the next AcquireVideoCmdSlot.
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit failed for DPB restore");
	}
	TimelineValue_ = signalValue;
	ReleaseVideoCmdSlot();

	// Do not host-wait here. Conversion has completed before this submit, and
	// subsequent decode submissions use the same video queue, so Vulkan queue
	// order guarantees that this restore finishes before a decode can reuse
	// the layer. The command-buffer fence is waited only when its ring slot is
	// acquired again.
	SetFrameLayout(isOutput, imageIndex, decodeLayout);
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::TransitionFrameForSampledRead(const OaVideoFrame& InFrame)
{
	if (!Rt_ || !CmdBuffers_[0]) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InFrame.Image) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid video frame for sampled read transition");
	}

	bool isOutput = false;
	OaU32 imageIndex = 0;
	VkImageLayout oldLayout = GetFrameLayout(InFrame, isOutput, imageIndex);
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Unknown video frame image layout");
	}
	
	// Already in correct layout for sampling
	if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		return OaStatus::Ok();
	}

	// Distinct decode-output VkImages are single-layer (baseArrayLayer 0).
	// DPB layers use InFrame.ArrayLayer. Using the DPB slot as arrayLayer on
	// an output image trips validation and eventually DEVICE_LOST on submit.
	const OaU32 barrierLayer = isOutput ? 0u : InFrame.ArrayLayer;

	// DPB images with SAMPLED_BIT can be transitioned from VIDEO_DECODE_DST/DPB to SHADER_READ_ONLY_OPTIMAL
	// Output images with SAMPLED_BIT can also be transitioned

	auto& vkEngine = *Rt_;
	auto slot = AcquireVideoCmdSlot();
	if (!slot.Status.IsOk()) {
		return slot.Status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkBeginCommandBuffer failed for sampled read transition");
	}

	VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	VkAccessFlags2 srcAccess = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
	if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		srcStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
		srcAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
	}

	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = srcStage;
	barrier.srcAccessMask = srcAccess;
	auto& vkEngineSample = *Rt_;
	const bool sameFamilySample = vkEngineSample.Device.Queues.VideoDecodeQueueFamily == vkEngineSample.Device.Queues.ComputeQueueFamily;
	if (sameFamilySample) {
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	} else {
		// Concurrent sharing across different queue families. The compute
		// dispatch waits on our timeline semaphore, so we only need to
		// finish the layout transition on the decode queue.
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
		barrier.dstAccessMask = 0;
	}
	barrier.oldLayout = oldLayout;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = InFrame.Image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	// MUST transition the specific layer we just decoded into. The DPB image
	// is a layered VkImage (one slot per array layer); transitioning layer 0
	// while sampling layer N≠0 leaves that layer in VIDEO_DECODE_DPB_KHR and
	// the shader reads garbage on most slots — visible as macroblock glitches
	// and "almost static" output because some reads land on stale layer 0.
	barrier.subresourceRange.baseArrayLayer = barrierLayer;
	barrier.subresourceRange.layerCount = 1;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(cb, &dependency);

	result = vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		return OaStatus::Error(OaStatusCode::VulkanError, "vkEndCommandBuffer failed for sampled read transition");
	}

	const OaU64 signalValue = TimelineValue_ + 1;
	VkSemaphore sem = static_cast<VkSemaphore>(TimelineSem_.Semaphore);
	VkTimelineSemaphoreSubmitInfo tsInfo = {};
	tsInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	tsInfo.signalSemaphoreValueCount = 1;
	tsInfo.pSignalSemaphoreValues = &signalValue;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = &tsInfo;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cb;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &sem;
	if (vkEngine.Device.Queues.VideoDecodeQueue == nullptr) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = vkQueueSubmit(static_cast<VkQueue>(vkEngine.Device.Queues.VideoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		// Don't reset the fence on error: it was already reset before recording and
		// the failed submit never gave it to the GPU, so resetting would leave it
		// permanently unsignaled and deadlock the next AcquireVideoCmdSlot.
		ReleaseVideoCmdSlot();
		OA_LOG_ERROR(OaLogComponent::Core, "vkQueueSubmit failed for sampled read transition, VkResult=%d", (int)result);
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkQueueSubmit failed for sampled read transition");
	}
	TimelineValue_ = signalValue;
	ReleaseVideoCmdSlot();
	SetFrameLayout(isOutput, imageIndex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::EnsureYcbcrSampler(OaYCbCrModel InColorSpace, OaFilter InFilter)
{
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	auto& vkEngine = *Rt_;
	if (!vkEngine.Device.Info.Software.HasSamplerYcbcrConversion) {
		return OaStatus::Error(OaStatusCode::Unavailable, "VK_KHR_sampler_ycbcr_conversion is not supported");
	}

	VkSampler* targetSampler = (InFilter == OaFilter::Nearest)
		? &YcbcrSamplerNearest_ : &YcbcrSampler_;
	if (*targetSampler && YcbcrConversion_) {
		return OaStatus::Ok();
	}

	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	if (!YcbcrConversion_) {
		VkSamplerYcbcrConversionCreateInfo conversionInfo = {};
		conversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
		conversionInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		conversionInfo.ycbcrModel = ToVkYcbcrModel(InColorSpace, Profile_.Width, Profile_.Height);
		conversionInfo.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
		conversionInfo.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY};
		conversionInfo.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
		conversionInfo.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
		conversionInfo.chromaFilter = VK_FILTER_LINEAR;
		conversionInfo.forceExplicitReconstruction = VK_FALSE;

		VkResult result = vkCreateSamplerYcbcrConversion(device, &conversionInfo, nullptr, &YcbcrConversion_);
		if (result != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create YCbCr sampler conversion");
		}
	}

	VkSamplerYcbcrConversionInfo samplerConversion = {};
	samplerConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	samplerConversion.conversion = YcbcrConversion_;

	const VkFilter vkFilter = (InFilter == OaFilter::Nearest)
		? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.pNext = &samplerConversion;
	samplerInfo.magFilter = vkFilter;
	samplerInfo.minFilter = vkFilter;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 0.0f;

	VkResult result = vkCreateSampler(device, &samplerInfo, nullptr, targetSampler);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create YCbCr sampler");
	}

	return OaStatus::Ok();
}

OaResult<OaVideoFrame> OaVideoDecoder::AllocateRgbaFrame(OaU32 InWidth, OaU32 InHeight, OaU64 InPts)
{
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (InWidth == 0 || InHeight == 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "RGBA frame dimensions must be non-zero");
	}

	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	auto allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent = {InWidth, InHeight, 1};
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage =
		VK_IMAGE_USAGE_STORAGE_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	OaU32 queueFamilies[3] = {
		vkEngine.Device.Queues.ComputeQueueFamily,
		vkEngine.Device.Queues.VideoDecodeQueueFamily,
		vkEngine.Device.Queues.GraphicsQueueFamily,
	};
	OaU32 queueFamilyCount = 0;
	for (OaU32 family : queueFamilies) {
		if (family == OaVkEnumerationIndexUnset) {
			continue;
		}
		bool found = false;
		for (OaU32 index = 0; index < queueFamilyCount; ++index) {
			found = found or queueFamilies[index] == family;
		}
		if (not found) {
			queueFamilies[queueFamilyCount++] = family;
		}
	}
	if (queueFamilyCount > 1) {
		imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
		imageInfo.queueFamilyIndexCount = queueFamilyCount;
		imageInfo.pQueueFamilyIndices = queueFamilies;
	} else {
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkImage image = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	VkResult result = OaVmaCreateImage(
		allocator,
		&imageInfo,
		&allocInfo,
		&image,
		&allocation,
		nullptr);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::OutOfMemory, "Failed to allocate RGBA video frame");
	}

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

	VkImageView imageView = VK_NULL_HANDLE;
	result = vkCreateImageView(device, &viewInfo, nullptr, &imageView);
	if (result != VK_SUCCESS) {
		OaVmaDestroyImage(allocator, image, allocation);
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create RGBA video frame image view");
	}

	auto slot = AcquireVideoCmdSlot();
	if (!slot.Status.IsOk()) {
		return slot.Status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	result = vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		vkDestroyImageView(device, imageView, nullptr);
		OaVmaDestroyImage(allocator, image, allocation);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkBeginCommandBuffer failed for RGBA frame transition");
	}

	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
	barrier.srcAccessMask = 0;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(cb, &dependency);

	result = vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		ReleaseVideoCmdSlot();
		vkDestroyImageView(device, imageView, nullptr);
		OaVmaDestroyImage(allocator, image, allocation);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkEndCommandBuffer failed for RGBA frame transition");
	}

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cb;
	if (vkEngine.Device.Queues.VideoDecodeQueue == nullptr) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = vkQueueSubmit(static_cast<VkQueue>(vkEngine.Device.Queues.VideoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		vkResetFences(device, 1, &fence);
		ReleaseVideoCmdSlot();
		vkDestroyImageView(device, imageView, nullptr);
		OaVmaDestroyImage(allocator, image, allocation);
		return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit failed for RGBA frame transition");
	}
	vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
	ReleaseVideoCmdSlot();

	RgbImages_.PushBack(image);
	RgbViews_.PushBack(imageView);
	RgbAllocations_.PushBack(allocation);
	RgbImageLayouts_.PushBack(VK_IMAGE_LAYOUT_GENERAL);

	OaVideoFrame frame = {};
	frame.Image = image;
	frame.ImageView = imageView;
	frame.Layout = VK_IMAGE_LAYOUT_GENERAL;
	frame.Format = VK_FORMAT_R8G8B8A8_UNORM;
	frame.Width = InWidth;
	frame.Height = InHeight;
	frame.PresentationTimestamp = InPts;
	frame.IsRgb = true;
	return frame;
}

OaResult<OaMatrix> OaVideoDecoder::ConvertFrameToBf16Hardware(
	const OaVideoFrame& InFrame,
	bool InNormalizeImageNet)
{
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InFrame.Image || InFrame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertFrameToBf16Hardware requires an NV12 frame");
	}
	OA_RETURN_IF_ERROR(EnsureYcbcrSampler(OaYCbCrModel::Auto));
	OA_RETURN_IF_ERROR(TransitionFrameForSampledRead(InFrame));

	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

	VkSamplerYcbcrConversionInfo viewConversion = {};
	viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	viewConversion.conversion = YcbcrConversion_;

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &viewConversion;
	viewInfo.image = InFrame.Image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	viewInfo.components = {
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY};
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView ycbcrView = VK_NULL_HANDLE;
	VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create YCbCr video frame image view");
	}

	auto output = OaFnMatrix::Empty(OaMatrixShape{1, 3, InFrame.Height, InFrame.Width}, OaScalarType::BFloat16);
	struct Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
		OaU32 Normalize;
	};
	Push push = {
		.Width = InFrame.Width,
		.Height = InFrame.Height,
		.CodedWidth = CodedWidth_,
		.CodedHeight = CodedHeight_,
		.Normalize = InNormalizeImageNet ? 1U : 0U
	};

	OaVkImageDispatchBinding bindings[3] = {};
	bindings[0].Kind = OaVkDescriptorKind::StorageBuffer;
	bindings[0].Binding = 0;
	bindings[0].Buffer = output.GetVkBuffer();
	bindings[1].Kind = OaVkDescriptorKind::SampledImage;
	bindings[1].Binding = 1;
	bindings[1].ImageView = ycbcrView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].Kind = OaVkDescriptorKind::Sampler;
	bindings[2].Binding = 2;
	// EnsureYcbcrSampler defaults to nearest for exact video texel sampling.
	// Bind the sampler it actually created; YcbcrSampler_ is the separate
	// linear-filter cache and remains null on this path.
	bindings[2].Sampler = YcbcrSamplerNearest_;

	OaStatus status = OaVkImageDispatch::Run(
		vkEngine,
		"CvtNv12YcbcrToBf16",
		OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
		&push,
		sizeof(push),
		OaDivCeil(InFrame.Width, 16),
		OaDivCeil(InFrame.Height, 16),
		1);
	vkDestroyImageView(device, ycbcrView, nullptr);
	if (!status.IsOk()) {
		return status;
	}
	return output;
}

OaResult<OaMatrix> OaVideoDecoder::ConvertFrameToBf16(const OaVideoFrame& InFrame, bool InNormalizeImageNet) {
	auto hardwareResult = ConvertFrameToBf16Hardware(InFrame, InNormalizeImageNet);
	if (hardwareResult.IsOk()) {
		return hardwareResult;
	}

	auto nv12Result = ReadbackNv12(InFrame);
	if (!nv12Result.IsOk()) {
		return nv12Result.GetStatus();
	}

	const OaU32 width = InFrame.Width;
	const OaU32 height = InFrame.Height;
	const OaU64 lumaBytes = static_cast<OaU64>(width) * height;
	const OaVec<OaU8>& nv12 = *nv12Result;
	if (nv12.Size() < static_cast<OaUsize>(lumaBytes + lumaBytes / 2)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "NV12 readback was smaller than expected");
	}

	auto tensor = OaFnMatrix::Empty(OaMatrixShape{1, 3, height, width}, OaScalarType::BFloat16);
	const OaF32 mean[3] = {0.485f, 0.456f, 0.406f};
	const OaF32 stdv[3] = {0.229f, 0.224f, 0.225f};
	const OaU64 spatial = lumaBytes;

	for (OaU32 y = 0; y < height; ++y) {
		for (OaU32 x = 0; x < width; ++x) {
			const OaU8 yy = nv12[y * width + x];
			const OaU64 uvOffset = lumaBytes + static_cast<OaU64>(y / 2) * width + static_cast<OaU64>(x / 2) * 2;
			const OaU8 uu = nv12[static_cast<OaUsize>(uvOffset + 0)];
			const OaU8 vv = nv12[static_cast<OaUsize>(uvOffset + 1)];

			const OaF32 Y = 1.164f * (static_cast<OaF32>(yy) - 16.0f) / 255.0f;
			const OaF32 U = (static_cast<OaF32>(uu) - 128.0f) / 255.0f;
			const OaF32 V = (static_cast<OaF32>(vv) - 128.0f) / 255.0f;
			OaF32 rgb[3] = {
				ClampUnit(Y + 1.596f * V),
				ClampUnit(Y - 0.391f * U - 0.813f * V),
				ClampUnit(Y + 2.018f * U),
			};
			if (InNormalizeImageNet) {
				for (OaU32 c = 0; c < 3; ++c) {
					rgb[c] = (rgb[c] - mean[c]) / stdv[c];
				}
			}

			const OaI64 pixel = static_cast<OaI64>(y) * width + x;
			tensor.Set(static_cast<OaI64>(0 * spatial + pixel), rgb[0]);
			tensor.Set(static_cast<OaI64>(1 * spatial + pixel), rgb[1]);
			tensor.Set(static_cast<OaI64>(2 * spatial + pixel), rgb[2]);
		}
	}
	return tensor;
}

OaResult<OaMatrix> OaVideoDecoder::DecodeFrameToBf16(
	const OaSpan<const OaU8>& InBitstream,
	bool InNormalizeImageNet)
{
	OaVideoFrame frame = {};
	OaStatus status = DecodeFrame(InBitstream, frame);
	if (!status.IsOk()) {
		return status;
	}
	return ConvertFrameToBf16(frame, InNormalizeImageNet);
}

// ============================================================================
// Phase 2.5: NV12 to RGB/BF16 Conversion
// ============================================================================

// Convert NV12 frame to RGB using hardware or compute shader
OaStatus OaVideoDecoder::ConvertNv12ToRgb(
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutRgbFrame)
{
	// A coincident video DPB is not an ordinary sampled image. When Create()
	// selected the staging path, preserve that decision at this older
	// synchronous entry point instead of falling through to the direct YCbCr
	// sampler merely because the device supports sampler conversion.
	if (UseSampleStaging_ && InNv12Frame.Image == Dpb_.GetImage()) {
		auto rgbaResult = AcquireConvertedRgbaTarget(
			InNv12Frame.Width,
			InNv12Frame.Height,
			InNv12Frame.PresentationTimestamp);
		if (!rgbaResult.IsOk()) {
			return rgbaResult.GetStatus();
		}
		OaVideoFrame rgbaFrame = *rgbaResult;
		OA_RETURN_IF_ERROR(ConvertNv12ToRgbInto(InNv12Frame, InOptions, rgbaFrame));
		OutRgbFrame = rgbaFrame;
		return OaStatus::Ok();
	}

	// Route to hardware or compute path based on options
	if (InOptions.PreferHardwareYCbCr && HasHardwareYCbCrConversion(*Rt_))
	{
		return ConvertNv12ToRgbHardware(InNv12Frame, InOptions.ColorSpace, OutRgbFrame, InOptions.Filter);
	}
	else
	{
		return ConvertNv12ToRgbCompute(InNv12Frame, InOptions.ColorSpace, OutRgbFrame, InOptions.Filter);
	}
}

OaStatus OaVideoDecoder::ConvertFrameToRgba(
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutRgbFrame)
{
	return ConvertNv12ToRgb(InNv12Frame, InOptions, OutRgbFrame);
}

OaStatus OaVideoDecoder::ConvertNv12ToRgbInto(
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& InOutRgbTarget)
{
	auto ticketResult = ConvertNv12ToRgbIntoAsync(
		InNv12Frame,
		InOptions,
		InOutRgbTarget);
	if (!ticketResult.IsOk()) {
		return ticketResult.GetStatus();
	}
	// A host wait establishes completion for the CPU, but it is not a
	// device-to-device memory dependency for a later graphics-queue sampler.
	// Preserve the producing stream's timeline token on the RGBA frame so the
	// presenter can wait on it even after this synchronous convenience call.
	const OaEvent ready = ticketResult->Completion();
	OA_RETURN_IF_ERROR(ticketResult->Wait());
	InOutRgbTarget.Ready = ready;
	return OaStatus::Ok();
}

OaResult<OaVkImageDispatchTicket> OaVideoDecoder::ConvertNv12ToRgbIntoAsync(
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	const OaVideoFrame& InRgbTarget)
{
	// Shader-path conversion writing to a caller-owned RGBA target. This is
	// the variant the OaVideo reorder buffer uses: each decoded NV12 frame
	// gets converted into its own pool slot, so the DPB layer is free to be
	// overwritten by the next decode without trashing any frame we've yet
	// to display.
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InNv12Frame.Image || InNv12Frame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertNv12ToRgbInto requires an NV12 frame");
	}
	if (!InRgbTarget.Image || !InRgbTarget.ImageView || InRgbTarget.Format != VK_FORMAT_R8G8B8A8_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertNv12ToRgbInto: invalid RGBA target");
	}

	// Direct video-profile images must use the multiplane YCbCr sampler path.
	// PreferHardwareYCbCr remains a preference for ordinary/staging NV12
	// images, but mutable R8/R8G8 plane views of a coincident video DPB are
	// not a portable fallback (ANV loses the device when they are sampled).
	if (not UseSampleStaging_
		and HasHardwareYCbCrConversion(*Rt_)) {
		OA_RETURN_IF_ERROR(EnsureYcbcrSampler(InOptions.ColorSpace, InOptions.Filter));
		OA_RETURN_IF_ERROR(TransitionFrameForSampledRead(InNv12Frame));

		auto& vkEngine = *Rt_;
		VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

		VkImageViewUsageCreateInfo usageInfo = {};
		usageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
		usageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

		VkSamplerYcbcrConversionInfo viewConversion = {};
		viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
		viewConversion.pNext = &usageInfo;
		viewConversion.conversion = YcbcrConversion_;

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.pNext = &viewConversion;
		viewInfo.image = InNv12Frame.Image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		viewInfo.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY};
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = GetNv12PlaneArrayLayer(InNv12Frame);
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView ycbcrView = VK_NULL_HANDLE;
		VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
		if (result != VK_SUCCESS) {
			return OaStatus::Error(
				OaStatusCode::VulkanError,
				"ConvertNv12ToRgbInto: failed to create YCbCr image view");
		}

		struct Push {
			OaU32 Width;
			OaU32 Height;
			OaU32 CodedWidth;
			OaU32 CodedHeight;
		};
		Push push = {
			.Width = InNv12Frame.Width,
			.Height = InNv12Frame.Height,
			.CodedWidth = CodedWidth_,
			.CodedHeight = CodedHeight_};

		OaVkImageDispatchBinding bindings[3] = {};
		bindings[0].Kind = OaVkDescriptorKind::StorageImage;
		bindings[0].Binding = 1;
		bindings[0].ImageView = InRgbTarget.ImageView;
		bindings[0].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
		bindings[1].Kind = OaVkDescriptorKind::SampledImage;
		bindings[1].Binding = 2;
		bindings[1].ImageView = ycbcrView;
		bindings[1].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		bindings[2].Kind = OaVkDescriptorKind::Sampler;
		bindings[2].Binding = 3;
		bindings[2].Sampler = InOptions.Filter == OaFilter::Nearest
			? YcbcrSamplerNearest_ : YcbcrSampler_;

		auto ticketResult = OaVkImageDispatch::RunWithDependencyAsync(
			vkEngine,
			"CvtNv12YcbcrToRgba",
			OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
			&push,
			sizeof(push),
			OaDivCeil(InNv12Frame.Width, 16),
			OaDivCeil(InNv12Frame.Height, 16),
			1,
			TimelineSem_,
			TimelineValue_);
		if (not ticketResult.IsOk()) {
			vkDestroyImageView(device, ycbcrView, nullptr);
			return ticketResult.GetStatus();
		}
		OaVkImageDispatchTicket ticket = OaStdMove(*ticketResult);
		ticket.AdoptImageView(ycbcrView);
		for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
			if (RgbImages_[i] == InRgbTarget.Image && i < RgbImageLayouts_.Size()) {
				RgbImageLayouts_[i] = VK_IMAGE_LAYOUT_GENERAL;
				break;
			}
		}
		OA_RETURN_IF_ERROR(RestoreDpbLayerToDecodeLayoutAfter(
			InNv12Frame, &ticket.Semaphore(), ticket.Value()));
		return ticket;
	}

	// Coincident staging: copy DPB to a plain NV12 image, then sample its planes.
	if (UseSampleStaging_ && InNv12Frame.Image == Dpb_.GetImage()) {
		OA_RETURN_IF_ERROR(CopyDpbLayerToSampleImage(InNv12Frame));
		const OaU32 layer = InNv12Frame.ArrayLayer;
		if (layer >= SampleYViews_.Size() || layer >= SampleUvViews_.Size()) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertNv12ToRgbInto: invalid staging layer");
		}
		VkImageView yView = SampleYViews_[layer];
		VkImageView uvView = SampleUvViews_[layer];
		VkSampler sampler = GetCachedNv12Sampler(InOptions.Filter);
		const VkImageLayout sampleLayout = (layer < SampleImageLayouts_.Size())
			? SampleImageLayouts_[layer]
			: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
			return OaStatus::Error(OaStatusCode::VulkanError, "ConvertNv12ToRgbInto: missing staging plane views");
		}

		struct Push {
			OaU32 Width;
			OaU32 Height;
			OaU32 CodedWidth;
			OaU32 CodedHeight;
			OaU32 ColorSpace;
			OaU32 FullRange;
		};

		OaVkImageDispatchBinding bindings[4] = {};
		bindings[0].Kind = OaVkDescriptorKind::SampledImage;
		bindings[0].Binding = 0;
		bindings[0].ImageView = yView;
		bindings[0].ImageLayout = sampleLayout;
		bindings[1].Kind = OaVkDescriptorKind::SampledImage;
		bindings[1].Binding = 1;
		bindings[1].ImageView = uvView;
		bindings[1].ImageLayout = sampleLayout;
		bindings[2].Kind = OaVkDescriptorKind::StorageImage;
		bindings[2].Binding = 2;
		bindings[2].ImageView = InRgbTarget.ImageView;
		bindings[2].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
		bindings[3].Kind = OaVkDescriptorKind::Sampler;
		bindings[3].Binding = 3;
		bindings[3].Sampler = sampler;

		Push push = {
			.Width = InNv12Frame.Width,
			.Height = InNv12Frame.Height,
			.CodedWidth = CodedWidth_,
			.CodedHeight = CodedHeight_,
			.ColorSpace = ToVisionColorSpace(OaYCbCrModel::Auto, InNv12Frame.Width, InNv12Frame.Height),
			.FullRange = 0U};

		auto& vkEngine = *Rt_;
		const OaU64 convertWaitValue = TimelineValue_;
		auto ticketResult = OaVkImageDispatch::RunWithDependencyAsync(
			vkEngine,
			"CvtNv12ToRgb",
			OaSpan<const OaVkImageDispatchBinding>(bindings, 4),
			&push,
			sizeof(push),
			OaDivCeil(InNv12Frame.Width, 16),
			OaDivCeil(InNv12Frame.Height, 16),
			1,
			TimelineSem_,
			convertWaitValue);
		if (!ticketResult.IsOk()) {
			return ticketResult.GetStatus();
		}
		OaVkImageDispatchTicket ticket = OaStdMove(*ticketResult);
		for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
			if (RgbImages_[i] == InRgbTarget.Image && i < RgbImageLayouts_.Size()) {
				RgbImageLayouts_[i] = VK_IMAGE_LAYOUT_GENERAL;
				break;
			}
		}
		return ticket;
	}

	bool isDpbStaging = UseSampleStaging_ && InNv12Frame.Image == Dpb_.GetImage();
	if (!isDpbStaging) {
		OA_RETURN_IF_ERROR(TransitionFrameForSampledRead(InNv12Frame));
	}

	const OaU32 planeLayer = GetNv12PlaneArrayLayer(InNv12Frame);
	VkImageView yView  = GetCachedNv12PlaneView(InNv12Frame.Image, planeLayer, VK_FORMAT_R8_UNORM,    VK_IMAGE_ASPECT_PLANE_0_BIT);
	VkImageView uvView = GetCachedNv12PlaneView(InNv12Frame.Image, planeLayer, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT);
	VkSampler   sampler = GetCachedNv12Sampler(InOptions.Filter);
	if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
		return OaStatus::Error(OaStatusCode::VulkanError, "ConvertNv12ToRgbInto: failed to cache views/sampler");
	}

	if (isDpbStaging) {
		// Make sure the side staging copy exists (safe to call; it will copy from current DPB layout via its logic).
		(void)CopyDpbLayerToSampleImage(InNv12Frame);
		const OaU32 layer = InNv12Frame.ArrayLayer;  // or planeLayer
		if (layer < SampleYViews_.Size()) yView = SampleYViews_[layer];
		if (layer < SampleUvViews_.Size()) uvView = SampleUvViews_[layer];
	}

	struct Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
		OaU32 ColorSpace;
		OaU32 FullRange;
	};

	OaVkImageDispatchBinding bindings[4] = {};
	bindings[0].Kind = OaVkDescriptorKind::SampledImage;
	bindings[0].Binding = 0;
	bindings[0].ImageView = yView;
	bindings[0].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[1].Kind = OaVkDescriptorKind::SampledImage;
	bindings[1].Binding = 1;
	bindings[1].ImageView = uvView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].Kind = OaVkDescriptorKind::StorageImage;
	bindings[2].Binding = 2;
	bindings[2].ImageView = InRgbTarget.ImageView;
	bindings[2].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[3].Kind = OaVkDescriptorKind::Sampler;
	bindings[3].Binding = 3;
	bindings[3].Sampler = sampler;

	Push push = {
		.Width = InNv12Frame.Width,
		.Height = InNv12Frame.Height,
		.CodedWidth = CodedWidth_,
		.CodedHeight = CodedHeight_,
		.ColorSpace = ToVisionColorSpace(OaYCbCrModel::Auto, InNv12Frame.Width, InNv12Frame.Height),
		.FullRange = 0U};

	auto& vkEngine = *Rt_;
	const OaU64 convertWaitValue = TimelineValue_;
	auto ticketResult = OaVkImageDispatch::RunWithDependencyAsync(
		vkEngine,
		"CvtNv12ToRgb",
		OaSpan<const OaVkImageDispatchBinding>(bindings, 4),
		&push,
		sizeof(push),
		OaDivCeil(InNv12Frame.Width, 16),
		OaDivCeil(InNv12Frame.Height, 16),
		1,
		TimelineSem_,
		convertWaitValue);
	if (!ticketResult.IsOk()) {
		return ticketResult.GetStatus();
	}
	OaVkImageDispatchTicket ticket = OaStdMove(*ticketResult);
	for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
		if (RgbImages_[i] == InRgbTarget.Image && i < RgbImageLayouts_.Size()) {
			RgbImageLayouts_[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	OA_RETURN_IF_ERROR(RestoreDpbLayerToDecodeLayoutAfter(
		InNv12Frame, &ticket.Semaphore(), ticket.Value()));
	return ticket;
}

// Hardware YCbCr conversion path (VK_KHR_sampler_ycbcr_conversion)
OaStatus OaVideoDecoder::ConvertNv12ToRgbHardware(
	const OaVideoFrame& InNv12Frame,
	OaYCbCrModel InColorSpace,
	OaVideoFrame& OutRgbFrame,
	OaFilter InFilter)
{
	if (!Rt_) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (!InNv12Frame.Image || InNv12Frame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertNv12ToRgbHardware requires an NV12 frame");
	}

	OA_RETURN_IF_ERROR(EnsureYcbcrSampler(InColorSpace, InFilter));
	OA_RETURN_IF_ERROR(TransitionFrameForSampledRead(InNv12Frame));

	auto rgbaResult = AcquireConvertedRgbaTarget(
		InNv12Frame.Width,
		InNv12Frame.Height,
		InNv12Frame.PresentationTimestamp);
	if (!rgbaResult.IsOk()) {
		return rgbaResult.GetStatus();
	}
	OaVideoFrame rgbaFrame = *rgbaResult;

	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

	VkImageViewUsageCreateInfo usageInfo = {};
	usageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	usageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

	VkSamplerYcbcrConversionInfo viewConversion = {};
	viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	viewConversion.pNext = &usageInfo;
	viewConversion.conversion = YcbcrConversion_;

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &viewConversion;
	viewInfo.image = InNv12Frame.Image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	viewInfo.components = {
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY,
		VK_COMPONENT_SWIZZLE_IDENTITY};
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = GetNv12PlaneArrayLayer(InNv12Frame);
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView ycbcrView = VK_NULL_HANDLE;
	VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create YCbCr video frame image view");
	}

	struct Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
	};
	Push push = {
		.Width = InNv12Frame.Width,
		.Height = InNv12Frame.Height,
		.CodedWidth = CodedWidth_,
		.CodedHeight = CodedHeight_};

	OaVkImageDispatchBinding bindings[3] = {};
	bindings[0].Kind = OaVkDescriptorKind::StorageImage;
	bindings[0].Binding = 1;
	bindings[0].ImageView = rgbaFrame.ImageView;
	bindings[0].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].Kind = OaVkDescriptorKind::SampledImage;
	bindings[1].Binding = 2;
	bindings[1].ImageView = ycbcrView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].Kind = OaVkDescriptorKind::Sampler;
	bindings[2].Binding = 3;
	bindings[2].Sampler = (InFilter == OaFilter::Nearest)
		? YcbcrSamplerNearest_ : YcbcrSampler_;

	OaStatus status = OaVkImageDispatch::RunWithDependency(
		vkEngine,
		"CvtNv12YcbcrToRgba",
		OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
		&push,
		sizeof(push),
		OaDivCeil(InNv12Frame.Width, 16),
		OaDivCeil(InNv12Frame.Height, 16),
		1,
		TimelineSem_,
		TimelineValue_);
	vkDestroyImageView(device, ycbcrView, nullptr);
	if (!status.IsOk()) {
		return status;
	}

	for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
		if (RgbImages_[i] == rgbaFrame.Image && i < RgbImageLayouts_.Size()) {
			RgbImageLayouts_[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	OutRgbFrame = rgbaFrame;
	return OaStatus::Ok();
}

VkSampler OaVideoDecoder::GetCachedNv12Sampler(OaFilter InFilter) {
	VkSampler* target = (InFilter == OaFilter::Nearest)
		? &CachedNv12SamplerNearest_ : &CachedNv12Sampler_;
	if (*target || !Rt_) {
		return *target;
	}
	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	const VkFilter vkFilter = (InFilter == OaFilter::Nearest)	? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = vkFilter;
	samplerInfo.minFilter = vkFilter;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.minLod = 0.0F;
	samplerInfo.maxLod = 0.0F;
	(void)vkCreateSampler(device, &samplerInfo, nullptr, target);
	return *target;
}

VkImageView OaVideoDecoder::GetCachedNv12PlaneView(
	VkImage InImage,
	OaU32 InLayer,
	VkFormat InFormat,
	VkImageAspectFlagBits InPlane) {
	if (!Rt_ || !InImage || InLayer >= CachedNv12YViews_.Size()) {
		return VK_NULL_HANDLE;
	}
	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

	// Invalidate the cache if the DPB image changed (e.g. session recreate).
	if (CachedNv12Image_ != InImage) {
		for (VkImageView& view : CachedNv12YViews_) {
			if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
		}
		for (VkImageView& view : CachedNv12UvViews_) {
			if (view) { vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
		}
		CachedNv12Image_ = InImage;
	}

	auto& slot = (InPlane == VK_IMAGE_ASPECT_PLANE_0_BIT)
		? CachedNv12YViews_[InLayer]
		: CachedNv12UvViews_[InLayer];
	if (slot) {
		return slot;
	}

	// These per-plane R8/R8G8 views are only ever sampled by the NV12→RGBA
	// conversion shader. The DPB image carries VIDEO_DECODE_DST usage, which a
	// single-plane R8/R8G8 view cannot satisfy (R8 lacks VIDEO_DECODE_OUTPUT
	// format features) — so without restricting usage, vkCreateImageView fails
	// (VUID-08333) and leaves a NULL view, and the shader samples nothing →
	// grey output. Override the view usage to SAMPLED so the format-feature
	// check is against sampling, which R8/R8G8 support.
	VkImageViewUsageCreateInfo usageInfo = {};
	usageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	usageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &usageInfo;
	viewInfo.image = InImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = InFormat;
	viewInfo.subresourceRange.aspectMask = InPlane;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = InLayer;
	viewInfo.subresourceRange.layerCount = 1;
	VkResult viewResult = vkCreateImageView(device, &viewInfo, nullptr, &slot);
	if (viewResult != VK_SUCCESS) {
		slot = VK_NULL_HANDLE;
	}
	return slot;
}

OaResult<OaVideoFrame> OaVideoDecoder::AcquireConvertedRgbaTarget(OaU32 InWidth, OaU32 InHeight, OaU64 InPts)
{
	if (ReusedRgbaIndex_ >= 0
		&& InWidth == ReusedRgbaWidth_
		&& InHeight == ReusedRgbaHeight_
		&& static_cast<OaUsize>(ReusedRgbaIndex_) < RgbImages_.Size())
	{
		OaVideoFrame frame = {};
		frame.Image  = RgbImages_[static_cast<OaUsize>(ReusedRgbaIndex_)];
		frame.ImageView = RgbViews_[static_cast<OaUsize>(ReusedRgbaIndex_)];
		frame.Layout = RgbImageLayouts_[static_cast<OaUsize>(ReusedRgbaIndex_)];
		frame.Format = VK_FORMAT_R8G8B8A8_UNORM;
		frame.Width  = InWidth;
		frame.Height = InHeight;
		frame.PresentationTimestamp = InPts;
		frame.IsRgb  = true;
		return frame;
	}
	auto allocResult = AllocateRgbaFrame(InWidth, InHeight, InPts);
	if (!allocResult.IsOk()) {
		return allocResult.GetStatus();
	}
	OaVideoFrame frame = *allocResult;
	ReusedRgbaIndex_ = static_cast<OaI32>(RgbImages_.Size()) - 1;
	ReusedRgbaWidth_ = InWidth;
	ReusedRgbaHeight_ = InHeight;
	return frame;
}

// Software YCbCr conversion path (compute shader fallback)
OaStatus OaVideoDecoder::ConvertNv12ToRgbCompute(
	const OaVideoFrame& InNv12Frame,
	OaYCbCrModel InColorSpace,
	OaVideoFrame& OutRgbFrame,
	OaFilter InFilter)
{
	if (!Rt_) {
		return OaStatus::Error("Runtime not initialized");
	}
	if (!InNv12Frame.Image || InNv12Frame.Format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "ConvertNv12ToRgbCompute requires an NV12 frame");
	}

	OA_RETURN_IF_ERROR(TransitionFrameForSampledRead(InNv12Frame));

	auto rgbaResult = AcquireConvertedRgbaTarget(
		InNv12Frame.Width,
		InNv12Frame.Height,
		InNv12Frame.PresentationTimestamp);
	if (!rgbaResult.IsOk()) {
		return rgbaResult.GetStatus();
	}
	OaVideoFrame rgbaFrame = *rgbaResult;

	const OaU32 planeLayer = GetNv12PlaneArrayLayer(InNv12Frame);
	VkImageView yView  = GetCachedNv12PlaneView(InNv12Frame.Image, planeLayer, VK_FORMAT_R8_UNORM,    VK_IMAGE_ASPECT_PLANE_0_BIT);
	VkImageView uvView = GetCachedNv12PlaneView(InNv12Frame.Image, planeLayer, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT);
	VkSampler   sampler = GetCachedNv12Sampler(InFilter);
	if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
		return OaStatus::Error(OaStatusCode::VulkanError, "ConvertNv12ToRgbCompute: failed to cache views/sampler");
	}

	// Note: OaVkImageDispatch::Run automatically prepends bindless descriptor indices
	// to the push constants, so we only provide the user data here.
	struct Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
		OaU32 ColorSpace;
		OaU32 FullRange;
	};

	OaVkImageDispatchBinding bindings[4] = {};
	bindings[0].Kind = OaVkDescriptorKind::SampledImage;
	bindings[0].Binding = 0;
	bindings[0].ImageView = yView;
	bindings[0].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[1].Kind = OaVkDescriptorKind::SampledImage;
	bindings[1].Binding = 1;
	bindings[1].ImageView = uvView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].Kind = OaVkDescriptorKind::StorageImage;
	bindings[2].Binding = 2;
	bindings[2].ImageView = rgbaFrame.ImageView;
	bindings[2].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[3].Kind = OaVkDescriptorKind::Sampler;
	bindings[3].Binding = 3;
	bindings[3].Sampler = sampler;

	Push push = {
		.Width = InNv12Frame.Width,
		.Height = InNv12Frame.Height,
		.CodedWidth = CodedWidth_,
		.CodedHeight = CodedHeight_,
		.ColorSpace = ToVisionColorSpace(InColorSpace, InNv12Frame.Width, InNv12Frame.Height),
		.FullRange = 0U};

	auto& vkEngine = *Rt_;
	OaStatus status = OaVkImageDispatch::Run(
		vkEngine,
		"CvtNv12ToRgb",
		OaSpan<const OaVkImageDispatchBinding>(bindings, 4),
		&push,
		sizeof(push),
		OaDivCeil(InNv12Frame.Width, 16),
		OaDivCeil(InNv12Frame.Height, 16),
		1);
	if (!status.IsOk()) {
		return status;
	}

	for (OaUsize i = 0; i < RgbImages_.Size(); ++i) {
		if (RgbImages_[i] == rgbaFrame.Image && i < RgbImageLayouts_.Size()) {
			RgbImageLayouts_[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	OutRgbFrame = rgbaFrame;
	return OaStatus::Ok();
}

// Decode with YCbCr→RGB conversion
OaStatus OaVideoDecoder::DecodeFrameWithConversion(
	const OaSpan<const OaU8>& InBitstream,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutFrame)
{
	// First decode to NV12
	OaVideoFrame nv12Frame;
	OaStatus status = DecodeFrame(InBitstream, nv12Frame);
	if (!status.IsOk())
	{
		return status;
	}
	
	// Then convert NV12 to RGB if requested
	if (InOptions.ConvertToRgb)
	{
		OaStatus convStatus = ConvertNv12ToRgb(nv12Frame, InOptions, OutFrame);
		if (!convStatus.IsOk()) {
			return convStatus;
		}
		// ConvertNv12ToRgb transitions the NV12 DPB layer to SHADER_READ_ONLY_OPTIMAL
		// for the compute/hardware sampler. The next decode expects VIDEO_DECODE_DPB_KHR.
		// Restore the layout now so the decoder state stays consistent.
		OaStatus restoreStatus = RestoreDpbLayerToDecodeLayout(nv12Frame);
		if (!restoreStatus.IsOk()) {
			return restoreStatus;
		}
		return OaStatus::Ok();
	}
	else
	{
		// Return NV12 frame as-is
		OutFrame = nv12Frame;
		return OaStatus::Ok();
	}
}

// Query hardware YCbCr conversion support
bool OaVideoDecoder::HasHardwareYCbCrConversion(OaEngine& InRt)
{
	auto& vkEngine = InRt;
	return vkEngine.Device.Info.Software.HasSamplerYcbcrConversion;
}

OaU32 OaVideoDecoder::GetNv12PlaneArrayLayer(const OaVideoFrame& InFrame) const
{
	for (OaUsize i = 0; i < OutputImages_.Size(); ++i) {
		if (OutputImages_[i] == InFrame.Image) {
			return 0;
		}
	}
	return InFrame.ArrayLayer;
}

OaStatus OaVideoDecoder::CreateOutputImages(
	OaEngine& InRt,
	const VkVideoProfileInfoKHR& InProfile,
	VkFormat InFormat,
	VkExtent2D InCodedExtent,
	OaU32 InSlotCount)
{
	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);
	auto allocator = static_cast<OaVmaAllocator>(InRt.Allocator.Allocator);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &InProfile;

	OaU32 sharedFamilies[3] = {
		InRt.Device.Queues.VideoDecodeQueueFamily,
		InRt.Device.Queues.ComputeQueueFamily,
		InRt.Device.Queues.GraphicsQueueFamily,
	};
	OaU32 sharedFamilyCount = 0;
	for (OaU32 family : sharedFamilies) {
		if (family == OaVkEnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (OaU32 i = 0; i < sharedFamilyCount; ++i) {
			dup = dup or sharedFamilies[i] == family;
		}
		if (not dup) {
			sharedFamilies[sharedFamilyCount++] = family;
		}
	}

	OutputImages_.Reserve(InSlotCount);
	OutputViews_.Reserve(InSlotCount);
	OutputAllocations_.Reserve(InSlotCount);
	OutputImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);

	for (OaU32 slot = 0; slot < InSlotCount; ++slot) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.pNext = &profileList;
		imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = InFormat;
		imageInfo.extent = {InCodedExtent.width, InCodedExtent.height, 1};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage =
			VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
			VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		if (sharedFamilyCount > 1) {
			imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			imageInfo.queueFamilyIndexCount = sharedFamilyCount;
			imageInfo.pQueueFamilyIndices = sharedFamilies;
		}
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		OaVmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		OaVmaAllocation allocation = VK_NULL_HANDLE;
		VkResult result = OaVmaCreateImage(allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
		if (result != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create decode output image");
		}

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = InFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView view = VK_NULL_HANDLE;
		result = vkCreateImageView(device, &viewInfo, nullptr, &view);
		if (result != VK_SUCCESS) {
			OaVmaDestroyImage(allocator, image, allocation);
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create decode output image view");
		}

		OutputImages_.PushBack(image);
		OutputViews_.PushBack(view);
		OutputAllocations_.PushBack(allocation);
		if (slot < OutputImageLayouts_.Size()) {
			OutputImageLayouts_[slot] = VK_IMAGE_LAYOUT_UNDEFINED;
		}
	}
	OutputFrameCapacity_ = InSlotCount;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::CreateSampleStagingImages(
	OaEngine& InRt,
	const VkVideoProfileInfoKHR& InProfile,
	VkExtent2D InCodedExtent,
	OaU32 InSlotCount
) {
	VkDevice device = static_cast<VkDevice>(InRt.Device.Device);
	auto allocator = static_cast<OaVmaAllocator>(InRt.Allocator.Allocator);

	OaU32 sharedFamilies[3] = {
		InRt.Device.Queues.VideoDecodeQueueFamily,
		InRt.Device.Queues.ComputeQueueFamily,
		InRt.Device.Queues.GraphicsQueueFamily,
	};
	OaU32 sharedFamilyCount = 0;
	for (OaU32 family : sharedFamilies) {
		if (family == OaVkEnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (OaU32 i = 0; i < sharedFamilyCount; ++i) {
			dup = dup or sharedFamilies[i] == family;
		}
		if (not dup) {
			sharedFamilies[sharedFamilyCount++] = family;
		}
	}

	SampleImages_.Reserve(InSlotCount);
	SampleYViews_.Reserve(InSlotCount);
	SampleUvViews_.Reserve(InSlotCount);
	SampleAllocations_.Reserve(InSlotCount);
	SampleImageLayouts_.Fill(VK_IMAGE_LAYOUT_UNDEFINED);

	for (OaU32 slot = 0; slot < InSlotCount; ++slot) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		imageInfo.extent = {InCodedExtent.width, InCodedExtent.height, 1};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage =
			VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT;
		if (sharedFamilyCount > 1) {
			imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			imageInfo.queueFamilyIndexCount = sharedFamilyCount;
			imageInfo.pQueueFamilyIndices = sharedFamilies;
		}
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		OaVmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		OaVmaAllocation allocation = VK_NULL_HANDLE;
		VkResult result = OaVmaCreateImage(allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
		if (result != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create NV12 staging image");
		}

		auto createPlaneView = [&](VkFormat InFormat, VkImageAspectFlagBits InPlane, VkImageView& OutView) -> OaStatus {
			VkImageViewCreateInfo viewInfo = {};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = InFormat;
			viewInfo.subresourceRange.aspectMask = InPlane;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			result = vkCreateImageView(device, &viewInfo, nullptr, &OutView);
			if (result != VK_SUCCESS) {
				return OaStatus::Error(OaStatusCode::VulkanError, "Failed to create NV12 staging plane view");
			}
			return OaStatus::Ok();
		};

		VkImageView yView = VK_NULL_HANDLE;
		VkImageView uvView = VK_NULL_HANDLE;
		OaStatus yStatus = createPlaneView(VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_PLANE_0_BIT, yView);
		if (!yStatus.IsOk()) {
			OaVmaDestroyImage(allocator, image, allocation);
			return yStatus;
		}
		OaStatus uvStatus = createPlaneView(VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT, uvView);
		if (!uvStatus.IsOk()) {
			vkDestroyImageView(device, yView, nullptr);
			OaVmaDestroyImage(allocator, image, allocation);
			return uvStatus;
		}

		SampleImages_.PushBack(image);
		SampleYViews_.PushBack(yView);
		SampleUvViews_.PushBack(uvView);
		SampleAllocations_.PushBack(allocation);
		if (slot < SampleImageLayouts_.Size()) {
			SampleImageLayouts_[slot] = VK_IMAGE_LAYOUT_UNDEFINED;
		}
	}
	(void)InProfile;
	return OaStatus::Ok();
}

OaStatus OaVideoDecoder::CopyDpbLayerToSampleImage(const OaVideoFrame& InDpbFrame)
{
	if (!UseSampleStaging_ || !Rt_ || !Dpb_.GetImage()) {
		return OaStatus::Ok();
	}
	if (CopySampleStagingOnVideoQueue_) {
		return OaStatus::Ok();
	}
	if (InDpbFrame.Image != Dpb_.GetImage()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyDpbLayerToSampleImage: not a DPB frame");
	}
	const OaU32 layer = InDpbFrame.ArrayLayer;
	if (layer >= SampleImages_.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "CopyDpbLayerToSampleImage: invalid layer");
	}

	OA_RETURN_IF_ERROR(ReleaseDpbLayerForComputeCopy(InDpbFrame));
	auto& vkEngine = *Rt_;
	const OaU64 decodeWaitValue = TimelineValue_;

	OaVkStream* stream = vkEngine.AcquireStream();
	if (stream == nullptr) {
		return OaStatus::Error(OaStatusCode::VulkanError, "CopyDpbLayerToSampleImage: stream acquire failed");
	}
	OaStatus beginStatus = stream->Begin(vkEngine.Device);
	if (!beginStatus.IsOk()) {
		vkEngine.ReleaseStream(stream);
		return beginStatus;
	}

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->CommandBuffer);
	VkImage srcImage = Dpb_.GetImage();
	VkImage dstImage = SampleImages_[layer];
	VkImageLayout srcLayout = layer < DpbImageLayouts_.Size()
		? DpbImageLayouts_[layer]
		: VK_IMAGE_LAYOUT_UNDEFINED;
	if (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		srcLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	}
	VkImageLayout dstLayout = layer < SampleImageLayouts_.Size()
		? SampleImageLayouts_[layer]
		: VK_IMAGE_LAYOUT_UNDEFINED;

	auto barrier = [&](VkImage image, VkImageLayout& InOutLayout, VkImageLayout newLayout,
		VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
		OaU32 arrayLayer) {
		if (InOutLayout == newLayout) {
			return;
		}
		VkImageMemoryBarrier2 b = {};
		b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		b.srcStageMask = srcStage;
		b.srcAccessMask = srcAccess;
		b.dstStageMask = dstStage;
		b.dstAccessMask = dstAccess;
		b.oldLayout = InOutLayout;
		b.newLayout = newLayout;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image = image;
		b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		b.subresourceRange.baseMipLevel = 0;
		b.subresourceRange.levelCount = 1;
		b.subresourceRange.baseArrayLayer = arrayLayer;
		b.subresourceRange.layerCount = 1;
		VkDependencyInfo dep = {};
		dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers = &b;
		vkCmdPipelineBarrier2(cb, &dep);
		InOutLayout = newLayout;
	};

	VkImageLayout srcTransfer = srcLayout;
	VkImageLayout dstTransfer = dstLayout;

	const bool sameFamily = vkEngine.Device.Queues.VideoDecodeQueueFamily
		== vkEngine.Device.Queues.ComputeQueueFamily;
	VkPipelineStageFlags2 srcStage = sameFamily
		? VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR
		: VK_PIPELINE_STAGE_2_NONE;
	VkAccessFlags2 srcAccess = sameFamily
		? (VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR)
		: 0;

	barrier(srcImage, srcTransfer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		srcStage, srcAccess,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT,
		layer);

	VkPipelineStageFlags2 dstSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
	VkAccessFlags2 dstSrcAccess = 0;
	if (dstTransfer == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		dstSrcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		dstSrcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	}
	barrier(dstImage, dstTransfer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		dstSrcStage, dstSrcAccess,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		0);

	const OaU32 chromaWidth = CodedWidth_ / 2;
	const OaU32 chromaHeight = CodedHeight_ / 2;

	VkImageCopy2 copies[2] = {};
	copies[0].sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
	copies[0].srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copies[0].srcSubresource.mipLevel = 0;
	copies[0].srcSubresource.baseArrayLayer = layer;
	copies[0].srcSubresource.layerCount = 1;
	copies[0].dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copies[0].dstSubresource.mipLevel = 0;
	copies[0].dstSubresource.baseArrayLayer = 0;
	copies[0].dstSubresource.layerCount = 1;
	copies[0].extent = {CodedWidth_, CodedHeight_, 1};

	copies[1].sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
	copies[1].srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	copies[1].srcSubresource.mipLevel = 0;
	copies[1].srcSubresource.baseArrayLayer = layer;
	copies[1].srcSubresource.layerCount = 1;
	copies[1].dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	copies[1].dstSubresource.mipLevel = 0;
	copies[1].dstSubresource.baseArrayLayer = 0;
	copies[1].dstSubresource.layerCount = 1;
	copies[1].extent = {chromaWidth, chromaHeight, 1};

	if (vkCmdCopyImage2) {
		VkCopyImageInfo2 copyInfo = {};
		copyInfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
		copyInfo.srcImage = srcImage;
		copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		copyInfo.dstImage = dstImage;
		copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		copyInfo.regionCount = 2;
		copyInfo.pRegions = copies;
		vkCmdCopyImage2(cb, &copyInfo);
	} else {
		VkImageCopy legacy[2] = {};
		legacy[0].srcSubresource = copies[0].srcSubresource;
		legacy[0].dstSubresource = copies[0].dstSubresource;
		legacy[0].extent = copies[0].extent;
		legacy[1].srcSubresource = copies[1].srcSubresource;
		legacy[1].dstSubresource = copies[1].dstSubresource;
		legacy[1].extent = copies[1].extent;
		vkCmdCopyImage(cb,
			srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			2, legacy);
	}

	barrier(dstImage, dstTransfer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		0);

	OaStatus submitStatus = stream->SubmitWithDependency(vkEngine, TimelineSem_, decodeWaitValue);
	if (submitStatus.IsOk()) {
		submitStatus = stream->Synchronize(vkEngine.Device);
	}
	vkEngine.ReleaseStream(stream);
	if (!submitStatus.IsOk()) {
		return submitStatus;
	}

	if (layer < DpbImageLayouts_.Size()) {
		DpbImageLayouts_[layer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	if (layer < SampleImageLayouts_.Size()) {
		SampleImageLayouts_[layer] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	OA_RETURN_IF_ERROR(RestoreDpbLayerToDecodeLayout(InDpbFrame));
	return OaStatus::Ok();
}

void OaVideoDecoder::RecordDpbLayerToSampleImage(
	VkCommandBuffer InCommandBuffer,
	OaI32 InDpbSlot)
{
	if (!CopySampleStagingOnVideoQueue_
		or InCommandBuffer == VK_NULL_HANDLE
		or InDpbSlot < 0
		or static_cast<OaUsize>(InDpbSlot) >= SampleImages_.Size()) {
		return;
	}

	const OaU32 layer = static_cast<OaU32>(InDpbSlot);
	VkImageLayout& srcLayout = DpbImageLayouts_[layer];
	VkImageLayout& dstLayout = SampleImageLayouts_[layer];

	auto barrier = [&](VkImage InImage,
		VkImageLayout InOldLayout,
		VkImageLayout InNewLayout,
		VkPipelineStageFlags2 InSrcStage,
		VkAccessFlags2 InSrcAccess,
		VkPipelineStageFlags2 InDstStage,
		VkAccessFlags2 InDstAccess,
		OaU32 InArrayLayer) {
		VkImageMemoryBarrier2 imageBarrier = {};
		imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		imageBarrier.srcStageMask = InSrcStage;
		imageBarrier.srcAccessMask = InSrcAccess;
		imageBarrier.dstStageMask = InDstStage;
		imageBarrier.dstAccessMask = InDstAccess;
		imageBarrier.oldLayout = InOldLayout;
		imageBarrier.newLayout = InNewLayout;
		imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.image = InImage;
		imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBarrier.subresourceRange.baseMipLevel = 0;
		imageBarrier.subresourceRange.levelCount = 1;
		imageBarrier.subresourceRange.baseArrayLayer = InArrayLayer;
		imageBarrier.subresourceRange.layerCount = 1;
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &imageBarrier;
		vkCmdPipelineBarrier2(InCommandBuffer, &dependency);
	};

	barrier(
		Dpb_.GetImage(),
		srcLayout,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
		VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT,
		layer
	);

	VkPipelineStageFlags2 dstSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
	VkAccessFlags2 dstSrcAccess = 0;
	if (dstLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		dstSrcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		dstSrcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	}
	barrier(
		SampleImages_[layer],
		dstLayout,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		dstSrcStage,
		dstSrcAccess,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		0);

	VkImageCopy regions[2] = {};
	regions[0].srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	regions[0].srcSubresource.layerCount = 1;
	regions[0].srcSubresource.baseArrayLayer = layer;
	regions[0].dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	regions[0].dstSubresource.layerCount = 1;
	regions[0].extent = {CodedWidth_, CodedHeight_, 1};
	regions[1].srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	regions[1].srcSubresource.layerCount = 1;
	regions[1].srcSubresource.baseArrayLayer = layer;
	regions[1].dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	regions[1].dstSubresource.layerCount = 1;
	regions[1].extent = {CodedWidth_ / 2, CodedHeight_ / 2, 1};
	vkCmdCopyImage(
		InCommandBuffer,
		Dpb_.GetImage(),
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		SampleImages_[layer],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		2,
		regions);

	barrier(
		SampleImages_[layer],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
		0,
		0);
	barrier(
		Dpb_.GetImage(),
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_READ_BIT,
		VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
		VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR
			| VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
		layer);

	srcLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	dstLayout = VK_IMAGE_LAYOUT_GENERAL;
}
