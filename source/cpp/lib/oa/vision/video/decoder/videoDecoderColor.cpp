// OA Vision — decoded-frame readback, NV12→RGB conversion, DPB layout restore.
// These are stateful oa::VideoDecoder session commands, not oa::FnVideo operations.

#include <oa/vision/videoDecoder.h>
#include "videoDecoderImpl.h"
#include <oa/core/log.h>
#include <oa/vision/fnImage.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/engine/submissionAccess.h>
#include <oa/runtime/imageDispatch.h>
#include <oa/runtime/stream.h>
#include <vma/vma.hpp>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>

static oa::F32 clampUnit(oa::F32 inValue)
{
	if (inValue < 0.0F) return 0.0F;
	if (inValue > 1.0F) return 1.0F;
	return inValue;
}

static VkSamplerYcbcrModelConversion toVkYcbcrModel(oa::YCbCrModel inColorSpace, oa::U32 inWidth, oa::U32 inHeight) {
	if (inColorSpace == oa::YCbCrModel::BT2020) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
	}
	if (inColorSpace == oa::YCbCrModel::BT709 || inWidth >= 1280 || inHeight >= 720) {
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
	return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
}

static oa::U32 toVisionColorSpace(oa::YCbCrModel inColorSpace, oa::U32 inWidth, oa::U32 inHeight)
{
	if (inColorSpace == oa::YCbCrModel::BT2020) {
		return 2;
	}
	if (inColorSpace == oa::YCbCrModel::BT709 || inWidth >= 1280 || inHeight >= 720) {
		return 1;
	}
	return 0;
}

oa::Result<oa::Vector<oa::U8>> oa::VideoDecoder::readbackLuma(const oa::VideoFrame& inFrame)
{
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0]) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inFrame.image || inFrame.width == 0 || inFrame.height == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video frame for luma readback");
	}
	OA_RETURN_IF_ERROR(inFrame.ready.wait());

	auto& vkEngine = *impl_->engine;
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);
	const oa::U64 byteSize = static_cast<oa::U64>(inFrame.width) * inFrame.height;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	vma::AllocationCreateInfo allocInfo = {};
	allocInfo.usage = vma::memoryUsageCpuOnly;
	allocInfo.flags = vma::allocationCreateMappedBit | vma::allocationCreateHostAccessRandomBit;

	::VkBuffer readbackBuffer = VK_NULL_HANDLE;
	vma::Allocation readbackAllocation = VK_NULL_HANDLE;
	vma::AllocationInfo readbackInfo = {};
	VkResult result = vma::createBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "Failed to allocate video luma readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			vma::destroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	bool isOutput = false;
	oa::U32 imageIndex = 0;
	for (oa::Usize i = 0; i < impl_->outputImages.size(); ++i) {
		if (impl_->outputImages[i] == inFrame.image) {
			oldLayout = impl_->outputImageLayouts[i];
			isOutput = true;
			imageIndex = static_cast<oa::U32>(i);
			break;
		}
	}
	if (!isOutput && impl_->dpb.getImage() == inFrame.image) {
		oldLayout = impl_->dpbImageLayouts[inFrame.arrayLayer];
	}
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		cleanup();
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unknown video frame image layout");
	}

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(vkEngine);
	if (stream == nullptr) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to acquire compute stream for luma readback");
	}
	oa::Status beginStatus = stream->begin(oa::EngineDeviceAccess::get(vkEngine));
	if (!beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->commandBuffer);

	VkImageMemoryBarrier2 toTransfer = {};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	const bool sameQueueFamily =
		oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueueFamily == oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily;
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
	toTransfer.image = inFrame.image;
	toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	toTransfer.subresourceRange.baseMipLevel = 0;
	toTransfer.subresourceRange.levelCount = 1;
	toTransfer.subresourceRange.baseArrayLayer = inFrame.arrayLayer;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copy = {};
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copy.imageSubresource.mipLevel = 0;
	copy.imageSubresource.baseArrayLayer = inFrame.arrayLayer;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = {inFrame.width, inFrame.height, 1};
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImageToBuffer(
		cb,
		inFrame.image,
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
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &hostDep);

	oa::Status submitStatus = stream->submit(vkEngine);
	if (submitStatus.isOk()) {
		submitStatus = stream->synchronize(oa::EngineDeviceAccess::get(vkEngine));
	}
	oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
	if (!submitStatus.isOk()) {
		cleanup();
		return submitStatus;
	}

	result = vma::invalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to invalidate video luma readback allocation");
	}

	oa::Vector<oa::U8> data(static_cast<oa::Usize>(byteSize));
	oa::memcpy(data.data(), readbackInfo.pMappedData, static_cast<oa::Usize>(byteSize));
	cleanup();

	if (isOutput) {
		impl_->outputImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else {
		impl_->dpbImageLayouts[inFrame.arrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	return data;
}

oa::Result<oa::Vector<oa::U8>> oa::VideoDecoder::readbackNv12(const oa::VideoFrame& inFrame) {
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0]) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inFrame.image || inFrame.width == 0 || inFrame.height == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video frame for NV12 readback");
	}
	if (inFrame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "readbackNv12 requires VK_FORMAT_G8_B8R8_2PLANE_420_UNORM");
	}
	OA_RETURN_IF_ERROR(inFrame.ready.wait());

	auto& vkEngine = *impl_->engine;
	VkImage readbackImage = inFrame.image;
	oa::U32 readbackLayer = inFrame.arrayLayer;
	bool isSampleStaging = false;
	if (impl_->useSampleStaging && inFrame.image == impl_->dpb.getImage()) {
		OA_RETURN_IF_ERROR(copyDpbLayerToSampleImage(inFrame));
		if (inFrame.arrayLayer >= impl_->sampleImages.size()) {
			return oa::Status::error(
				oa::StatusCode::InvalidArgument,
				"NV12 readback staging layer is unavailable");
		}
		readbackImage = impl_->sampleImages[inFrame.arrayLayer];
		readbackLayer = 0;
		isSampleStaging = true;
	}
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);
	const oa::U64 lumaBytes = static_cast<oa::U64>(inFrame.width) * inFrame.height;
	const oa::U64 chromaBytes = lumaBytes / 2;
	const oa::U64 byteSize = lumaBytes + chromaBytes;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	vma::AllocationCreateInfo allocInfo = {};
	allocInfo.usage = vma::memoryUsageCpuOnly;
	allocInfo.flags = vma::allocationCreateMappedBit | vma::allocationCreateHostAccessRandomBit;

	::VkBuffer readbackBuffer = VK_NULL_HANDLE;
	vma::Allocation readbackAllocation = VK_NULL_HANDLE;
	vma::AllocationInfo readbackInfo = {};
	VkResult result = vma::createBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "Failed to allocate video NV12 readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			vma::destroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	bool isOutput = false;
	oa::U32 imageIndex = 0;
	for (oa::Usize i = 0; i < impl_->outputImages.size(); ++i) {
		if (impl_->outputImages[i] == readbackImage) {
			oldLayout = impl_->outputImageLayouts[i];
			isOutput = true;
			imageIndex = static_cast<oa::U32>(i);
			break;
		}
	}
	if (isSampleStaging) {
		oldLayout = impl_->sampleImageLayouts[inFrame.arrayLayer];
	} else if (!isOutput && impl_->dpb.getImage() == readbackImage) {
		oldLayout = impl_->dpbImageLayouts[inFrame.arrayLayer];
	}
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		cleanup();
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unknown video frame image layout");
	}

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(vkEngine);
	if (stream == nullptr) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to acquire compute stream for NV12 readback");
	}
	oa::Status beginStatus = stream->begin(oa::EngineDeviceAccess::get(vkEngine));
	if (!beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->commandBuffer);

	VkImageMemoryBarrier2 toTransfer = {};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	const bool sameQueueFamily =
		oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueueFamily == oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily;
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
	const oa::U32 frameLayer = readbackLayer;
	toTransfer.subresourceRange.baseArrayLayer = frameLayer;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copies[2] = {};
	copies[0].bufferOffset = 0;
	copies[0].bufferRowLength = 0;
	copies[0].bufferImageHeight = 0;
	copies[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	copies[0].imageSubresource.mipLevel = 0;
	copies[0].imageSubresource.baseArrayLayer = frameLayer;
	copies[0].imageSubresource.layerCount = 1;
	copies[0].imageExtent = {inFrame.width, inFrame.height, 1};

	copies[1].bufferOffset = lumaBytes;
	copies[1].bufferRowLength = 0;
	copies[1].bufferImageHeight = 0;
	copies[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	copies[1].imageSubresource.mipLevel = 0;
	copies[1].imageSubresource.baseArrayLayer = frameLayer;
	copies[1].imageSubresource.layerCount = 1;
	copies[1].imageExtent = {inFrame.width / 2, inFrame.height / 2, 1};

	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImageToBuffer(
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
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &hostDep);

	oa::Status submitStatus = stream->submit(vkEngine);
	if (submitStatus.isOk()) {
		submitStatus = stream->synchronize(oa::EngineDeviceAccess::get(vkEngine));
	}
	oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
	if (!submitStatus.isOk()) {
		cleanup();
		return submitStatus;
	}

	result = vma::invalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to invalidate video NV12 readback allocation");
	}

	oa::Vector<oa::U8> data(static_cast<oa::Usize>(byteSize));
	oa::memcpy(data.data(), readbackInfo.pMappedData, static_cast<oa::Usize>(byteSize));
	cleanup();

	if (isOutput) {
		impl_->outputImageLayouts[imageIndex] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else if (isSampleStaging) {
		impl_->sampleImageLayouts[inFrame.arrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	} else {
		impl_->dpbImageLayouts[inFrame.arrayLayer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	return data;
}

oa::Result<oa::Vector<oa::U8>> oa::VideoDecoder::readbackRgba(const oa::VideoFrame& inFrame)
{
	if (!impl_->engine || !impl_->commandBuffers[0]) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inFrame.image || inFrame.width == 0 || inFrame.height == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video frame for RGBA readback");
	}
	if (inFrame.format != VK_FORMAT_R8G8B8A8_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "readbackRgba requires VK_FORMAT_R8G8B8A8_UNORM");
	}

	auto& vkEngine = *impl_->engine;
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);
	const oa::U64 byteSize = static_cast<oa::U64>(inFrame.width) * inFrame.height * 4;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = byteSize;
	bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	vma::AllocationCreateInfo allocInfo = {};
	allocInfo.usage = vma::memoryUsageCpuOnly;
	allocInfo.flags = vma::allocationCreateMappedBit | vma::allocationCreateHostAccessRandomBit;

	::VkBuffer readbackBuffer = VK_NULL_HANDLE;
	vma::Allocation readbackAllocation = VK_NULL_HANDLE;
	vma::AllocationInfo readbackInfo = {};
	VkResult result = vma::createBuffer(
		allocator,
		&bufferInfo,
		&allocInfo,
		&readbackBuffer,
		&readbackAllocation,
		&readbackInfo);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "Failed to allocate video RGBA readback buffer");
	}

	auto cleanup = [&]() {
		if (readbackBuffer || readbackAllocation) {
			vma::destroyBuffer(allocator, readbackBuffer, readbackAllocation);
		}
	};

	VkImageLayout oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	oa::U32 imageIndex = 0;
	bool foundImage = false;
	for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
		if (impl_->rgbImages[i] == inFrame.image) {
			imageIndex = static_cast<oa::U32>(i);
			oldLayout = i < impl_->rgbImageLayouts.size() ? impl_->rgbImageLayouts[i] : VK_IMAGE_LAYOUT_GENERAL;
			foundImage = true;
			break;
		}
	}
	if (!foundImage) {
		cleanup();
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unknown RGBA video frame image");
	}

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(vkEngine);
	if (stream == nullptr) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to acquire compute stream for RGBA readback");
	}
	oa::Status beginStatus = stream->begin(oa::EngineDeviceAccess::get(vkEngine));
	if (!beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
		cleanup();
		return beginStatus;
	}
	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->commandBuffer);

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
	toTransfer.image = inFrame.image;
	toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	toTransfer.subresourceRange.baseMipLevel = 0;
	toTransfer.subresourceRange.levelCount = 1;
	toTransfer.subresourceRange.baseArrayLayer = 0;
	toTransfer.subresourceRange.layerCount = 1;

	VkDependencyInfo toTransferDep = {};
	toTransferDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	toTransferDep.imageMemoryBarrierCount = 1;
	toTransferDep.pImageMemoryBarriers = &toTransfer;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &toTransferDep);

	VkBufferImageCopy copy = {};
	copy.bufferOffset = 0;
	copy.bufferRowLength = 0;
	copy.bufferImageHeight = 0;
	copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.imageSubresource.mipLevel = 0;
	copy.imageSubresource.baseArrayLayer = 0;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = {inFrame.width, inFrame.height, 1};

	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImageToBuffer(
		cb,
		inFrame.image,
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
	restoreLayout.image = inFrame.image;
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
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &hostDep);

	oa::Status submitStatus = stream->submit(vkEngine);
	if (submitStatus.isOk()) {
		submitStatus = stream->synchronize(oa::EngineDeviceAccess::get(vkEngine));
	}
	oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
	if (!submitStatus.isOk()) {
		cleanup();
		return submitStatus;
	}

	result = vma::invalidateAllocation(allocator, readbackAllocation, 0, byteSize);
	if (result != VK_SUCCESS) {
		cleanup();
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to invalidate video RGBA readback allocation");
	}

	oa::Vector<oa::U8> data(static_cast<oa::Usize>(byteSize));
	oa::memcpy(data.data(), readbackInfo.pMappedData, static_cast<oa::Usize>(byteSize));
	cleanup();

	if (imageIndex < impl_->rgbImageLayouts.size()) {
		impl_->rgbImageLayouts[imageIndex] = oldLayout;
	}
	return data;
}

VkImageLayout oa::VideoDecoder::getFrameLayout(const oa::VideoFrame& inFrame, bool& outIsOutput, oa::U32& outImageIndex) const
{
	outIsOutput = false;
	outImageIndex = 0;
	for (oa::Usize i = 0; i < impl_->outputImages.size(); ++i) {
		if (impl_->outputImages[i] == inFrame.image) {
			outIsOutput = true;
			outImageIndex = static_cast<oa::U32>(i);
			return impl_->outputImageLayouts[i];
		}
	}
	if (impl_->dpb.getImage() == inFrame.image) {
		outImageIndex = inFrame.arrayLayer;
		return impl_->dpbImageLayouts[inFrame.arrayLayer];
	}
	return VK_IMAGE_LAYOUT_UNDEFINED;
}

void oa::VideoDecoder::setFrameLayout(bool inIsOutput, oa::U32 inIndex, VkImageLayout inLayout)
{
	if (inIsOutput && inIndex < impl_->outputImageLayouts.size()) {
		impl_->outputImageLayouts[inIndex] = inLayout;
		return;
	}
	if (inIndex < impl_->dpbImageLayouts.size()) {
		impl_->dpbImageLayouts[inIndex] = inLayout;
	}
}

oa::Status oa::VideoDecoder::restoreDpbLayerToDecodeLayout(const oa::VideoFrame& inFrame)
{
	return restoreDpbLayerToDecodeLayoutAfter(inFrame, {});
}

oa::Status oa::VideoDecoder::releaseDpbLayerForComputeCopy(const oa::VideoFrame& inFrame)
{
	if (!impl_->engine || !impl_->commandBuffers[0] || inFrame.image != impl_->dpb.getImage()) {
		return oa::Status::ok();
	}
	const oa::U32 layer = inFrame.arrayLayer;
	if (layer >= impl_->dpbImageLayouts.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"DPB release requires a valid array layer");
	}
	if (impl_->dpbImageLayouts[layer] == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		return oa::Status::ok();
	}
	if (impl_->dpbImageLayouts[layer] != VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"DPB release requires VIDEO_DECODE_DPB layout");
	}

	auto& vkEngine = *impl_->engine;
	auto slot = acquireVideoCmdSlot();
	if (!slot.status.isOk()) {
		return slot.status;
	}
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkBeginCommandBuffer(slot.cb, &beginInfo);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkBeginCommandBuffer failed for DPB release");
	}

	// producer-side layout transition. The following timeline signal makes the
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
	barrier.image = impl_->dpb.getImage();
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = layer;
	barrier.subresourceRange.layerCount = 1;
	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(slot.cb, &dependency);

	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkEndCommandBuffer(slot.cb);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkEndCommandBuffer failed for DPB release");
	}
	const oa::U64 signalValue = impl_->timelineValue + 1;
	VkSemaphoreSubmitInfo signalInfo = {};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(impl_->timelineSemaphore.semaphore);
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
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkQueueSubmit2(
		static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue),
		1,
		&submitInfo,
		slot.fence);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkQueueSubmit2 failed for DPB release");
	}
	impl_->timelineValue = signalValue;
	impl_->dpbImageLayouts[layer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	releaseVideoCmdSlot();
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::restoreDpbLayerToDecodeLayoutAfter(
	const oa::VideoFrame& inFrame,
	const oa::Event& inWait)
{
	// Only meaningful for DPB-backed NV12 frames (the shader-convert path).
	// output-image frames don't need to flip back.
	if (!impl_->engine || !impl_->commandBuffers[0] || !inFrame.image) {
		return oa::Status::ok();
	}
	if (inFrame.isRgb || inFrame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::ok();
	}
	bool isOutput = false;
	oa::U32 imageIndex = 0;
	VkImageLayout oldLayout = getFrameLayout(inFrame, isOutput, imageIndex);
	// Distinct output images restore to DST; DPB-backed frames restore to DPB.
	const VkImageLayout decodeLayout = isOutput
		? VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
		: VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	if (oldLayout == decodeLayout) {
		return oa::Status::ok();
	}
	if (oldLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		&& oldLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"NV12 restore requires SHADER_READ_ONLY_OPTIMAL or TRANSFER_SRC_OPTIMAL layout");
	}
	const oa::U32 barrierLayer = isOutput ? 0u : inFrame.arrayLayer;

	auto& vkEngine = *impl_->engine;
	auto slot = acquireVideoCmdSlot();
	if (!slot.status.isOk()) {
		return slot.status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkBeginCommandBuffer failed for DPB restore");
	}

	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	auto& vkEngineRestore = *impl_->engine;
	const bool sameFamilyRestore = oa::EngineDeviceAccess::get(vkEngineRestore).queues.videoDecodeQueueFamily == oa::EngineDeviceAccess::get(vkEngineRestore).queues.computeQueueFamily;
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
	barrier.image = inFrame.image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = barrierLayer;
	barrier.subresourceRange.layerCount = 1;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &dependency);

	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkEndCommandBuffer failed for DPB restore");
	}
	const oa::U64 signalValue = impl_->timelineValue + 1;
	VkSemaphore signalSemaphore = static_cast<VkSemaphore>(impl_->timelineSemaphore.semaphore);
	const oavk::TimelineWait wait = oa::EventAccess::timelineWait(inWait);
	VkSemaphore waitSemaphore = static_cast<VkSemaphore>(wait.semaphore);
	const oa::U64 waitValue = wait.value;
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkTimelineSemaphoreSubmitInfo timelineInfo = {};
	timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timelineInfo.waitSemaphoreValueCount = waitSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	timelineInfo.pWaitSemaphoreValues = waitSemaphore != VK_NULL_HANDLE ? &waitValue : nullptr;
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
	if (oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue == nullptr) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkQueueSubmit(static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		// Don't reset the fence on error: it was already reset before recording and
		// the failed submit never gave it to the GPU, so resetting would leave it
		// permanently unsignaled and deadlock the next acquireVideoCmdSlot.
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkQueueSubmit failed for DPB restore");
	}
	impl_->timelineValue = signalValue;
	releaseVideoCmdSlot();

	// Do not host-wait here. conversion has completed before this submit, and
	// subsequent decode submissions use the same video queue, so vulkan queue
	// order guarantees that this restore finishes before a decode can reuse
	// the layer. The command-buffer fence is waited only when its ring slot is
	// acquired again.
	setFrameLayout(isOutput, imageIndex, decodeLayout);
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::transitionFrameForSampledRead(const oa::VideoFrame& inFrame)
{
	if (!impl_->engine || !impl_->commandBuffers[0]) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inFrame.image) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid video frame for sampled read transition");
	}

	bool isOutput = false;
	oa::U32 imageIndex = 0;
	VkImageLayout oldLayout = getFrameLayout(inFrame, isOutput, imageIndex);
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Unknown video frame image layout");
	}
	
	// Already in correct layout for sampling
	if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		return oa::Status::ok();
	}

	// Distinct decode-output VkImages are single-layer (baseArrayLayer 0).
	// DPB layers use inFrame.arrayLayer. Using the DPB slot as arrayLayer on
	// an output image trips validation and eventually DEVICE_LOST on submit.
	const oa::U32 barrierLayer = isOutput ? 0u : inFrame.arrayLayer;

	// DPB images with SAMPLED_BIT can be transitioned from VIDEO_DECODE_DST/DPB to SHADER_READ_ONLY_OPTIMAL
	// output images with SAMPLED_BIT can also be transitioned

	auto& vkEngine = *impl_->engine;
	auto slot = acquireVideoCmdSlot();
	if (!slot.status.isOk()) {
		return slot.status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkBeginCommandBuffer failed for sampled read transition");
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
	auto& vkEngineSample = *impl_->engine;
	const bool sameFamilySample = oa::EngineDeviceAccess::get(vkEngineSample).queues.videoDecodeQueueFamily == oa::EngineDeviceAccess::get(vkEngineSample).queues.computeQueueFamily;
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
	barrier.image = inFrame.image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	// MUST transition the specific layer we just decoded into. The DPB image
	// is a layered vkImage (one slot per array layer); transitioning layer 0
	// while sampling layer N≠0 leaves that layer in VIDEO_DECODE_DPB_KHR and
	// the shader reads garbage on most slots — visible as macroblock glitches
	// and "almost static" output because some reads land on stale layer 0.
	barrier.subresourceRange.baseArrayLayer = barrierLayer;
	barrier.subresourceRange.layerCount = 1;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &dependency);

	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		return oa::Status::error(oa::StatusCode::VulkanError, "vkEndCommandBuffer failed for sampled read transition");
	}

	const oa::U64 signalValue = impl_->timelineValue + 1;
	VkSemaphore sem = static_cast<VkSemaphore>(impl_->timelineSemaphore.semaphore);
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
	if (oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue == nullptr) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkQueueSubmit(static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		// Don't reset the fence on error: it was already reset before recording and
		// the failed submit never gave it to the GPU, so resetting would leave it
		// permanently unsignaled and deadlock the next acquireVideoCmdSlot.
		releaseVideoCmdSlot();
		OaLogError(oa::LogComponent::Video, "vkQueueSubmit failed for sampled read transition, VkResult=%d", (int)result);
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkQueueSubmit failed for sampled read transition");
	}
	impl_->timelineValue = signalValue;
	releaseVideoCmdSlot();
	setFrameLayout(isOutput, imageIndex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::ensureYcbcrSampler(oa::YCbCrModel inColorSpace, oa::Filter inFilter)
{
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	auto& vkEngine = *impl_->engine;
	if (!oa::EngineDeviceAccess::get(vkEngine).info.software.hasSamplerYcbcrConversion) {
		return oa::Status::error(oa::StatusCode::Unavailable, "VK_KHR_sampler_ycbcr_conversion is not supported");
	}

	VkSampler* targetSampler = (inFilter == oa::Filter::Nearest)
		? &impl_->ycbcrSamplerNearest : &impl_->ycbcrSampler;
	if (*targetSampler && impl_->ycbcrConversion) {
		return oa::Status::ok();
	}

	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	if (!impl_->ycbcrConversion) {
		VkSamplerYcbcrConversionCreateInfo conversionInfo = {};
		conversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
		conversionInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		conversionInfo.ycbcrModel = toVkYcbcrModel(inColorSpace, impl_->profile.width, impl_->profile.height);
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

		VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateSamplerYcbcrConversion(device, &conversionInfo, nullptr, &impl_->ycbcrConversion);
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create YCbCr sampler conversion");
		}
	}

	VkSamplerYcbcrConversionInfo samplerConversion = {};
	samplerConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	samplerConversion.conversion = impl_->ycbcrConversion;

	const VkFilter vkFilter = (inFilter == oa::Filter::Nearest)
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

	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateSampler(device, &samplerInfo, nullptr, targetSampler);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create YCbCr sampler");
	}

	return oa::Status::ok();
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::allocateRgbaFrame_(oa::U32 inWidth, oa::U32 inHeight, oa::U64 inPts)
{
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (inWidth == 0 || inHeight == 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "RGBA frame dimensions must be non-zero");
	}

	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent = {inWidth, inHeight, 1};
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage =
		VK_IMAGE_USAGE_STORAGE_BIT |
		VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
		VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	oa::U32 queueFamilies[3] = {
		oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily,
		oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueueFamily,
		oa::EngineDeviceAccess::get(vkEngine).queues.graphicsQueueFamily,
	};
	oa::U32 queueFamilyCount = 0;
	for (oa::U32 family : queueFamilies) {
		if (family == oavk::EnumerationIndexUnset) {
			continue;
		}
		bool found = false;
		for (oa::U32 index = 0; index < queueFamilyCount; ++index) {
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
	vma::Allocation allocation = VK_NULL_HANDLE;
	vma::AllocationCreateInfo allocInfo = {};
	allocInfo.usage = vma::memoryUsageGpuOnly;
	VkResult result = vma::createImage(
		allocator,
		&imageInfo,
		&allocInfo,
		&image,
		&allocation,
		nullptr);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::OutOfMemory, "Failed to allocate RGBA video frame");
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
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &imageView);
	if (result != VK_SUCCESS) {
		vma::destroyImage(allocator, image, allocation);
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create RGBA video frame image view");
	}

	auto slot = acquireVideoCmdSlot();
	if (!slot.status.isOk()) {
		return slot.status;
	}
	VkCommandBuffer cb = slot.cb;
	VkFence fence = slot.fence;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkBeginCommandBuffer(cb, &beginInfo);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, imageView, nullptr);
		vma::destroyImage(allocator, image, allocation);
		return oa::Status::error(oa::StatusCode::VulkanError, "vkBeginCommandBuffer failed for RGBA frame transition");
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
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &dependency);

	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkEndCommandBuffer(cb);
	if (result != VK_SUCCESS) {
		releaseVideoCmdSlot();
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, imageView, nullptr);
		vma::destroyImage(allocator, image, allocation);
		return oa::Status::error(oa::StatusCode::VulkanError, "vkEndCommandBuffer failed for RGBA frame transition");
	}

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cb;
	if (oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue == nullptr) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"video decode queue unavailable at submit — decoder reached without a video queue");
	}
	result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkQueueSubmit(static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue), 1, &submitInfo, fence);
	if (result != VK_SUCCESS) {
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkResetFences(device, 1, &fence);
		releaseVideoCmdSlot();
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, imageView, nullptr);
		vma::destroyImage(allocator, image, allocation);
		return oa::Status::error(oa::StatusCode::VulkanError, "vkQueueSubmit failed for RGBA frame transition");
	}
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
	releaseVideoCmdSlot();

	impl_->rgbImages.pushBack(image);
	impl_->rgbViews.pushBack(imageView);
	impl_->rgbAllocations.pushBack(allocation);
	impl_->rgbImageLayouts.pushBack(VK_IMAGE_LAYOUT_GENERAL);

	oa::VideoFrame frame = {};
	frame.image = image;
	frame.imageView = imageView;
	frame.layout = VK_IMAGE_LAYOUT_GENERAL;
	frame.format = VK_FORMAT_R8G8B8A8_UNORM;
	frame.width = inWidth;
	frame.height = inHeight;
	frame.presentationTimestamp = inPts;
	frame.isRgb = true;
	return frame;
}

oa::Result<oa::Matrix> oa::VideoDecoder::convertFrameToBf16Hardware(
	const oa::VideoFrame& inFrame,
	bool inNormalizeImageNet)
{
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inFrame.image || inFrame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "convertFrameToBf16Hardware requires an NV12 frame");
	}
	OA_RETURN_IF_ERROR(ensureYcbcrSampler(oa::YCbCrModel::Auto));
	OA_RETURN_IF_ERROR(transitionFrameForSampledRead(inFrame));

	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

	VkSamplerYcbcrConversionInfo viewConversion = {};
	viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	viewConversion.conversion = impl_->ycbcrConversion;

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &viewConversion;
	viewInfo.image = inFrame.image;
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
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create YCbCr video frame image view");
	}

	auto output = oa::FnMatrix::empty(oa::MatrixShape{1, 3, inFrame.height, inFrame.width}, oa::ScalarType::BFloat16);
	struct Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
		oa::U32 normalize;
	};
	Push push = {
		.width = inFrame.width,
		.height = inFrame.height,
		.codedWidth = impl_->codedWidth,
		.codedHeight = impl_->codedHeight,
		.normalize = inNormalizeImageNet ? 1U : 0U
	};

	oavk::ImageDispatchBinding bindings[3] = {};
	bindings[0].kind = oavk::DescriptorKind::StorageBuffer;
	bindings[0].binding = 0;
	bindings[0].buffer = oa::MatrixAccess::descriptor(output);
	bindings[1].kind = oavk::DescriptorKind::SampledImage;
	bindings[1].binding = 1;
	bindings[1].imageView = ycbcrView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].kind = oavk::DescriptorKind::Sampler;
	bindings[2].binding = 2;
	// ensureYcbcrSampler defaults to nearest for exact video texel sampling.
	// Bind the sampler it actually created; impl_->ycbcrSampler is the separate
	// linear-filter cache and remains null on this path.
	bindings[2].sampler = impl_->ycbcrSamplerNearest;

	oa::Status status = oavk::ImageDispatch::run(
		vkEngine,
		"CvtNv12YcbcrToBf16",
		oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
		&push,
		sizeof(push),
		oa::ScalarType::BFloat16,
		oa::divCeil(inFrame.width, 16),
		oa::divCeil(inFrame.height, 16),
		1);
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, ycbcrView, nullptr);
	if (!status.isOk()) {
		return status;
	}
	return output;
}

oa::Result<oa::Matrix> oa::VideoDecoder::convertFrameToBf16(const oa::VideoFrame& inFrame, bool inNormalizeImageNet) {
	auto hardwareResult = convertFrameToBf16Hardware(inFrame, inNormalizeImageNet);
	if (hardwareResult.isOk()) {
		return hardwareResult;
	}

	auto nv12Result = readbackNv12(inFrame);
	if (!nv12Result.isOk()) {
		return nv12Result.getStatus();
	}

	const oa::U32 width = inFrame.width;
	const oa::U32 height = inFrame.height;
	const oa::U64 lumaBytes = static_cast<oa::U64>(width) * height;
	const oa::Vector<oa::U8>& nv12 = *nv12Result;
	if (nv12.size() < static_cast<oa::Usize>(lumaBytes + lumaBytes / 2)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "NV12 readback was smaller than expected");
	}

	auto tensor = oa::FnMatrix::empty(oa::MatrixShape{1, 3, height, width}, oa::ScalarType::BFloat16);
	const oa::F32 mean[3] = {0.485f, 0.456f, 0.406f};
	const oa::F32 stdv[3] = {0.229f, 0.224f, 0.225f};
	const oa::U64 spatial = lumaBytes;

	for (oa::U32 y = 0; y < height; ++y) {
		for (oa::U32 x = 0; x < width; ++x) {
			const oa::U8 yy = nv12[y * width + x];
			const oa::U64 uvOffset = lumaBytes + static_cast<oa::U64>(y / 2) * width + static_cast<oa::U64>(x / 2) * 2;
			const oa::U8 uu = nv12[static_cast<oa::Usize>(uvOffset + 0)];
			const oa::U8 vv = nv12[static_cast<oa::Usize>(uvOffset + 1)];

			const oa::F32 Y = 1.164f * (static_cast<oa::F32>(yy) - 16.0f) / 255.0f;
			const oa::F32 U = (static_cast<oa::F32>(uu) - 128.0f) / 255.0f;
			const oa::F32 V = (static_cast<oa::F32>(vv) - 128.0f) / 255.0f;
			oa::F32 rgb[3] = {
				clampUnit(Y + 1.596f * V),
				clampUnit(Y - 0.391f * U - 0.813f * V),
				clampUnit(Y + 2.018f * U),
			};
			if (inNormalizeImageNet) {
				for (oa::U32 c = 0; c < 3; ++c) {
					rgb[c] = (rgb[c] - mean[c]) / stdv[c];
				}
			}

			const oa::I64 pixel = static_cast<oa::I64>(y) * width + x;
			tensor.set(static_cast<oa::I64>(0 * spatial + pixel), rgb[0]);
			tensor.set(static_cast<oa::I64>(1 * spatial + pixel), rgb[1]);
			tensor.set(static_cast<oa::I64>(2 * spatial + pixel), rgb[2]);
		}
	}
	return tensor;
}

oa::Result<oa::Matrix> oa::VideoDecoder::decodeFrameToBf16(
	const oa::Span<const oa::U8>& inBitstream,
	bool inNormalizeImageNet)
{
	oa::VideoFrame frame = {};
	oa::Status status = decodeFrame(inBitstream, frame);
	if (!status.isOk()) {
		return status;
	}
	return convertFrameToBf16(frame, inNormalizeImageNet);
}

// ============================================================================
// phase 2.5: NV12 to RGB/BF16 conversion
// ============================================================================

// convert NV12 frame to RGB using hardware or compute shader
oa::Status oa::VideoDecoder::convertNv12ToRgb(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outRgbFrame)
{
	// A coincident video DPB is not an ordinary sampled image. When create()
	// selected the staging path, preserve that decision at this older
	// synchronous entry point instead of falling through to the direct YCbCr
	// sampler merely because the device supports sampler conversion.
	if (impl_->useSampleStaging && inNv12Frame.image == impl_->dpb.getImage()) {
		auto rgbaResult = acquireConvertedRgbaTarget(
			inNv12Frame.width,
			inNv12Frame.height,
			inNv12Frame.presentationTimestamp);
		if (!rgbaResult.isOk()) {
			return rgbaResult.getStatus();
		}
		oa::VideoFrame rgbaFrame = *rgbaResult;
		OA_RETURN_IF_ERROR(convertNv12ToRgbInto(inNv12Frame, inOptions, rgbaFrame));
		outRgbFrame = rgbaFrame;
		return oa::Status::ok();
	}

	// route to hardware or compute path based on options
	if (inOptions.preferHardwareYCbCr && hasHardwareYCbCrConversion(*impl_->engine))
	{
		return convertNv12ToRgbHardware(inNv12Frame, inOptions.colorSpace, outRgbFrame, inOptions.filter);
	}
	else
	{
		return convertNv12ToRgbCompute(inNv12Frame, inOptions.colorSpace, outRgbFrame, inOptions.filter);
	}
}

oa::Status oa::VideoDecoder::convertFrameToRgba(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outRgbFrame)
{
	return convertNv12ToRgb(inNv12Frame, inOptions, outRgbFrame);
}

oa::Status oa::VideoDecoder::convertNv12ToRgbInto(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& inOutRgbTarget)
{
	auto readyResult = convertNv12ToRgbIntoAsync(
		inNv12Frame,
		inOptions,
		inOutRgbTarget);
	if (!readyResult.isOk()) {
		return readyResult.getStatus();
	}
	// A host wait establishes completion for the CPU, but it is not a
	// device-to-device memory dependency for a later graphics-queue sampler.
	// Preserve the producing stream's timeline token on the RGBA frame so the
	// presenter can wait on it even after this synchronous convenience call.
	const oa::Event ready = *readyResult;
	OA_RETURN_IF_ERROR(ready.wait());
	inOutRgbTarget.ready = ready;
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::VideoDecoder::convertNv12ToRgbIntoAsync(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	const oa::VideoFrame& inRgbTarget)
{
	// shader-path conversion writing to a caller-owned RGBA target. This is
	// the variant the oa::VideoPlayer reorder buffer uses: each decoded NV12 frame
	// gets converted into its own pool slot, so the DPB layer is free to be
	// overwritten by the next decode without trashing any frame we've yet
	// to display.
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inNv12Frame.image || inNv12Frame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "ConvertNv12ToRgbInto requires an NV12 frame");
	}
	if (!inRgbTarget.image || !inRgbTarget.imageView || inRgbTarget.format != VK_FORMAT_R8G8B8A8_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "ConvertNv12ToRgbInto: invalid RGBA target");
	}

	// Direct video-profile images must use the multiplane YCbCr sampler path.
	// preferHardwareYCbCr remains a preference for ordinary/staging NV12
	// images, but mutable R8/R8G8 plane views of a coincident video DPB are
	// not a portable fallback (ANV loses the device when they are sampled).
	if (not impl_->useSampleStaging
		and hasHardwareYCbCrConversion(*impl_->engine)) {
		OA_RETURN_IF_ERROR(ensureYcbcrSampler(inOptions.colorSpace, inOptions.filter));
		OA_RETURN_IF_ERROR(transitionFrameForSampledRead(inNv12Frame));

		auto& vkEngine = *impl_->engine;
		VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

		VkImageViewUsageCreateInfo usageInfo = {};
		usageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
		usageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

		VkSamplerYcbcrConversionInfo viewConversion = {};
		viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
		viewConversion.pNext = &usageInfo;
		viewConversion.conversion = impl_->ycbcrConversion;

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.pNext = &viewConversion;
		viewInfo.image = inNv12Frame.image;
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
		viewInfo.subresourceRange.baseArrayLayer = getNv12PlaneArrayLayer(inNv12Frame);
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView ycbcrView = VK_NULL_HANDLE;
		VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
		if (result != VK_SUCCESS) {
			return oa::Status::error(
				oa::StatusCode::VulkanError,
				"ConvertNv12ToRgbInto: failed to create YCbCr image view");
		}

		struct Push {
			oa::U32 width;
			oa::U32 height;
			oa::U32 codedWidth;
			oa::U32 codedHeight;
		};
		Push push = {
			.width = inNv12Frame.width,
			.height = inNv12Frame.height,
			.codedWidth = impl_->codedWidth,
			.codedHeight = impl_->codedHeight};

		oavk::ImageDispatchBinding bindings[3] = {};
		bindings[0].kind = oavk::DescriptorKind::StorageImage;
		bindings[0].binding = 1;
		bindings[0].imageView = inRgbTarget.imageView;
		bindings[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		bindings[1].kind = oavk::DescriptorKind::SampledImage;
		bindings[1].binding = 2;
		bindings[1].imageView = ycbcrView;
		bindings[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		bindings[2].kind = oavk::DescriptorKind::Sampler;
		bindings[2].binding = 3;
		bindings[2].sampler = inOptions.filter == oa::Filter::Nearest
			? impl_->ycbcrSamplerNearest : impl_->ycbcrSampler;

		auto ticketResult = oavk::ImageDispatch::runWithDependencyAsync(
			vkEngine,
			"CvtNv12YcbcrToRgba",
			oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
			&push,
			sizeof(push),
			oa::ScalarType::Float32,
			oa::divCeil(inNv12Frame.width, 16),
			oa::divCeil(inNv12Frame.height, 16),
			1,
			impl_->timelineSemaphore,
			impl_->timelineValue);
		if (not ticketResult.isOk()) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, ycbcrView, nullptr);
			return ticketResult.getStatus();
		}
		oavk::ImageDispatchTicket ticket = oa::move(*ticketResult);
		ticket.adoptImageView(ycbcrView);
		for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
			if (impl_->rgbImages[i] == inRgbTarget.image && i < impl_->rgbImageLayouts.size()) {
				impl_->rgbImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
				break;
			}
		}
		OA_RETURN_IF_ERROR(restoreDpbLayerToDecodeLayoutAfter(
			inNv12Frame, ticket.completion()));
		return ticket.completion();
	}

	// Coincident staging: copy DPB to a plain NV12 image, then sample its planes.
	if (impl_->useSampleStaging && inNv12Frame.image == impl_->dpb.getImage()) {
		OA_RETURN_IF_ERROR(copyDpbLayerToSampleImage(inNv12Frame));
		const oa::U32 layer = inNv12Frame.arrayLayer;
		if (layer >= impl_->sampleYViews.size() || layer >= impl_->sampleUvViews.size()) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "ConvertNv12ToRgbInto: invalid staging layer");
		}
		VkImageView yView = impl_->sampleYViews[layer];
		VkImageView uvView = impl_->sampleUvViews[layer];
		VkSampler sampler = getCachedNv12Sampler(inOptions.filter);
		const VkImageLayout sampleLayout = (layer < impl_->sampleImageLayouts.size())
			? impl_->sampleImageLayouts[layer]
			: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
			return oa::Status::error(oa::StatusCode::VulkanError, "ConvertNv12ToRgbInto: missing staging plane views");
		}

		struct Push {
			oa::U32 width;
			oa::U32 height;
			oa::U32 codedWidth;
			oa::U32 codedHeight;
			oa::U32 colorSpace;
			oa::U32 fullRange;
		};

		oavk::ImageDispatchBinding bindings[4] = {};
		bindings[0].kind = oavk::DescriptorKind::SampledImage;
		bindings[0].binding = 0;
		bindings[0].imageView = yView;
		bindings[0].imageLayout = sampleLayout;
		bindings[1].kind = oavk::DescriptorKind::SampledImage;
		bindings[1].binding = 1;
		bindings[1].imageView = uvView;
		bindings[1].imageLayout = sampleLayout;
		bindings[2].kind = oavk::DescriptorKind::StorageImage;
		bindings[2].binding = 2;
		bindings[2].imageView = inRgbTarget.imageView;
		bindings[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		bindings[3].kind = oavk::DescriptorKind::Sampler;
		bindings[3].binding = 3;
		bindings[3].sampler = sampler;

		Push push = {
			.width = inNv12Frame.width,
			.height = inNv12Frame.height,
			.codedWidth = impl_->codedWidth,
			.codedHeight = impl_->codedHeight,
			.colorSpace = toVisionColorSpace(oa::YCbCrModel::Auto, inNv12Frame.width, inNv12Frame.height),
			.fullRange = 0U};

		auto& vkEngine = *impl_->engine;
		const oa::U64 convertWaitValue = impl_->timelineValue;
		auto ticketResult = oavk::ImageDispatch::runWithDependencyAsync(
			vkEngine,
			"CvtNv12ToRgb",
			oa::Span<const oavk::ImageDispatchBinding>(bindings, 4),
			&push,
			sizeof(push),
			oa::ScalarType::Float32,
			oa::divCeil(inNv12Frame.width, 16),
			oa::divCeil(inNv12Frame.height, 16),
			1,
			impl_->timelineSemaphore,
			convertWaitValue);
		if (!ticketResult.isOk()) {
			return ticketResult.getStatus();
		}
		oavk::ImageDispatchTicket ticket = oa::move(*ticketResult);
		for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
			if (impl_->rgbImages[i] == inRgbTarget.image && i < impl_->rgbImageLayouts.size()) {
				impl_->rgbImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
				break;
			}
		}
		return ticket.completion();
	}

	bool isDpbStaging = impl_->useSampleStaging && inNv12Frame.image == impl_->dpb.getImage();
	if (!isDpbStaging) {
		OA_RETURN_IF_ERROR(transitionFrameForSampledRead(inNv12Frame));
	}

	const oa::U32 planeLayer = getNv12PlaneArrayLayer(inNv12Frame);
	VkImageView yView  = getCachedNv12PlaneView(inNv12Frame.image, planeLayer, VK_FORMAT_R8_UNORM,    VK_IMAGE_ASPECT_PLANE_0_BIT);
	VkImageView uvView = getCachedNv12PlaneView(inNv12Frame.image, planeLayer, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT);
	VkSampler   sampler = getCachedNv12Sampler(inOptions.filter);
	if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
		return oa::Status::error(oa::StatusCode::VulkanError, "ConvertNv12ToRgbInto: failed to cache views/sampler");
	}

	if (isDpbStaging) {
		// Make sure the side staging copy exists (safe to call; it will copy from current DPB layout via its logic).
		(void)copyDpbLayerToSampleImage(inNv12Frame);
		const oa::U32 layer = inNv12Frame.arrayLayer;  // or planeLayer
		if (layer < impl_->sampleYViews.size()) yView = impl_->sampleYViews[layer];
		if (layer < impl_->sampleUvViews.size()) uvView = impl_->sampleUvViews[layer];
	}

	struct Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
		oa::U32 colorSpace;
		oa::U32 fullRange;
	};

	oavk::ImageDispatchBinding bindings[4] = {};
	bindings[0].kind = oavk::DescriptorKind::SampledImage;
	bindings[0].binding = 0;
	bindings[0].imageView = yView;
	bindings[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[1].kind = oavk::DescriptorKind::SampledImage;
	bindings[1].binding = 1;
	bindings[1].imageView = uvView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].kind = oavk::DescriptorKind::StorageImage;
	bindings[2].binding = 2;
	bindings[2].imageView = inRgbTarget.imageView;
	bindings[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[3].kind = oavk::DescriptorKind::Sampler;
	bindings[3].binding = 3;
	bindings[3].sampler = sampler;

	Push push = {
		.width = inNv12Frame.width,
		.height = inNv12Frame.height,
		.codedWidth = impl_->codedWidth,
		.codedHeight = impl_->codedHeight,
		.colorSpace = toVisionColorSpace(oa::YCbCrModel::Auto, inNv12Frame.width, inNv12Frame.height),
		.fullRange = 0U};

	auto& vkEngine = *impl_->engine;
	const oa::U64 convertWaitValue = impl_->timelineValue;
	auto ticketResult = oavk::ImageDispatch::runWithDependencyAsync(
		vkEngine,
		"CvtNv12ToRgb",
		oa::Span<const oavk::ImageDispatchBinding>(bindings, 4),
		&push,
		sizeof(push),
		oa::ScalarType::Float32,
		oa::divCeil(inNv12Frame.width, 16),
		oa::divCeil(inNv12Frame.height, 16),
		1,
		impl_->timelineSemaphore,
		convertWaitValue);
	if (!ticketResult.isOk()) {
		return ticketResult.getStatus();
	}
	oavk::ImageDispatchTicket ticket = oa::move(*ticketResult);
	for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
		if (impl_->rgbImages[i] == inRgbTarget.image && i < impl_->rgbImageLayouts.size()) {
			impl_->rgbImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	OA_RETURN_IF_ERROR(restoreDpbLayerToDecodeLayoutAfter(
		inNv12Frame, ticket.completion()));
	return ticket.completion();
}

// hardware YCbCr conversion path (VK_KHR_sampler_ycbcr_conversion)
oa::Status oa::VideoDecoder::convertNv12ToRgbHardware(
	const oa::VideoFrame& inNv12Frame,
	oa::YCbCrModel inColorSpace,
	oa::VideoFrame& outRgbFrame,
	oa::Filter inFilter)
{
	if (!impl_->engine) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (!inNv12Frame.image || inNv12Frame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "ConvertNv12ToRgbHardware requires an NV12 frame");
	}

	OA_RETURN_IF_ERROR(ensureYcbcrSampler(inColorSpace, inFilter));
	OA_RETURN_IF_ERROR(transitionFrameForSampledRead(inNv12Frame));

	auto rgbaResult = acquireConvertedRgbaTarget(
		inNv12Frame.width,
		inNv12Frame.height,
		inNv12Frame.presentationTimestamp);
	if (!rgbaResult.isOk()) {
		return rgbaResult.getStatus();
	}
	oa::VideoFrame rgbaFrame = *rgbaResult;

	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

	VkImageViewUsageCreateInfo usageInfo = {};
	usageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	usageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

	VkSamplerYcbcrConversionInfo viewConversion = {};
	viewConversion.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
	viewConversion.pNext = &usageInfo;
	viewConversion.conversion = impl_->ycbcrConversion;

	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.pNext = &viewConversion;
	viewInfo.image = inNv12Frame.image;
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
	viewInfo.subresourceRange.baseArrayLayer = getNv12PlaneArrayLayer(inNv12Frame);
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView ycbcrView = VK_NULL_HANDLE;
	VkResult result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &ycbcrView);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create YCbCr video frame image view");
	}

	struct Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
	};
	Push push = {
		.width = inNv12Frame.width,
		.height = inNv12Frame.height,
		.codedWidth = impl_->codedWidth,
		.codedHeight = impl_->codedHeight};

	oavk::ImageDispatchBinding bindings[3] = {};
	bindings[0].kind = oavk::DescriptorKind::StorageImage;
	bindings[0].binding = 1;
	bindings[0].imageView = rgbaFrame.imageView;
	bindings[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].kind = oavk::DescriptorKind::SampledImage;
	bindings[1].binding = 2;
	bindings[1].imageView = ycbcrView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].kind = oavk::DescriptorKind::Sampler;
	bindings[2].binding = 3;
	bindings[2].sampler = (inFilter == oa::Filter::Nearest)
		? impl_->ycbcrSamplerNearest : impl_->ycbcrSampler;

	oa::Status status = oavk::ImageDispatch::runWithDependency(
		vkEngine,
		"CvtNv12YcbcrToRgba",
		oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
		&push,
		sizeof(push),
		oa::ScalarType::Float32,
		oa::divCeil(inNv12Frame.width, 16),
		oa::divCeil(inNv12Frame.height, 16),
		1,
		impl_->timelineSemaphore,
		impl_->timelineValue);
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, ycbcrView, nullptr);
	if (!status.isOk()) {
		return status;
	}

	for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
		if (impl_->rgbImages[i] == rgbaFrame.image && i < impl_->rgbImageLayouts.size()) {
			impl_->rgbImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	outRgbFrame = rgbaFrame;
	return oa::Status::ok();
}

VkSampler oa::VideoDecoder::getCachedNv12Sampler(oa::Filter inFilter) {
	VkSampler* target = (inFilter == oa::Filter::Nearest)
		? &impl_->cachedNv12SamplerNearest : &impl_->cachedNv12Sampler;
	if (*target || !impl_->engine) {
		return *target;
	}
	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	const VkFilter vkFilter = (inFilter == oa::Filter::Nearest)	? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
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
	(void)oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateSampler(device, &samplerInfo, nullptr, target);
	return *target;
}

VkImageView oa::VideoDecoder::getCachedNv12PlaneView(
	VkImage inImage,
	oa::U32 inLayer,
	VkFormat inFormat,
	VkImageAspectFlagBits inPlane) {
	if (!impl_->engine || !inImage || inLayer >= impl_->cachedNv12YViews.size()) {
		return VK_NULL_HANDLE;
	}
	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

	// Invalidate the cache if the DPB image changed (e.g. session recreate).
	if (impl_->cachedNv12Image != inImage) {
		for (VkImageView& view : impl_->cachedNv12YViews) {
			if (view) { oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
		}
		for (VkImageView& view : impl_->cachedNv12UvViews) {
			if (view) { oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
		}
		impl_->cachedNv12Image = inImage;
	}

	auto& slot = (inPlane == VK_IMAGE_ASPECT_PLANE_0_BIT)
		? impl_->cachedNv12YViews[inLayer]
		: impl_->cachedNv12UvViews[inLayer];
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
	viewInfo.image = inImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = inFormat;
	viewInfo.subresourceRange.aspectMask = inPlane;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = inLayer;
	viewInfo.subresourceRange.layerCount = 1;
	VkResult viewResult = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &slot);
	if (viewResult != VK_SUCCESS) {
		slot = VK_NULL_HANDLE;
	}
	return slot;
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::acquireConvertedRgbaTarget(oa::U32 inWidth, oa::U32 inHeight, oa::U64 inPts)
{
	if (impl_->reusedRgbaIndex >= 0
		&& inWidth == impl_->reusedRgbaWidth
		&& inHeight == impl_->reusedRgbaHeight
		&& static_cast<oa::Usize>(impl_->reusedRgbaIndex) < impl_->rgbImages.size())
	{
		oa::VideoFrame frame = {};
		frame.image  = impl_->rgbImages[static_cast<oa::Usize>(impl_->reusedRgbaIndex)];
		frame.imageView = impl_->rgbViews[static_cast<oa::Usize>(impl_->reusedRgbaIndex)];
		frame.layout = impl_->rgbImageLayouts[static_cast<oa::Usize>(impl_->reusedRgbaIndex)];
		frame.format = VK_FORMAT_R8G8B8A8_UNORM;
		frame.width  = inWidth;
		frame.height = inHeight;
		frame.presentationTimestamp = inPts;
		frame.isRgb  = true;
		return frame;
	}
	auto allocResult = allocateRgbaFrame_(inWidth, inHeight, inPts);
	if (!allocResult.isOk()) {
		return allocResult.getStatus();
	}
	oa::VideoFrame frame = *allocResult;
	impl_->reusedRgbaIndex = static_cast<oa::I32>(impl_->rgbImages.size()) - 1;
	impl_->reusedRgbaWidth = inWidth;
	impl_->reusedRgbaHeight = inHeight;
	return frame;
}

// software YCbCr conversion path (compute shader fallback)
oa::Status oa::VideoDecoder::convertNv12ToRgbCompute(
	const oa::VideoFrame& inNv12Frame,
	oa::YCbCrModel inColorSpace,
	oa::VideoFrame& outRgbFrame,
	oa::Filter inFilter)
{
	if (!impl_->engine) {
		return oa::Status::error("Runtime not initialized");
	}
	if (!inNv12Frame.image || inNv12Frame.format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "ConvertNv12ToRgbCompute requires an NV12 frame");
	}

	OA_RETURN_IF_ERROR(transitionFrameForSampledRead(inNv12Frame));

	auto rgbaResult = acquireConvertedRgbaTarget(
		inNv12Frame.width,
		inNv12Frame.height,
		inNv12Frame.presentationTimestamp);
	if (!rgbaResult.isOk()) {
		return rgbaResult.getStatus();
	}
	oa::VideoFrame rgbaFrame = *rgbaResult;

	const oa::U32 planeLayer = getNv12PlaneArrayLayer(inNv12Frame);
	VkImageView yView  = getCachedNv12PlaneView(inNv12Frame.image, planeLayer, VK_FORMAT_R8_UNORM,    VK_IMAGE_ASPECT_PLANE_0_BIT);
	VkImageView uvView = getCachedNv12PlaneView(inNv12Frame.image, planeLayer, VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT);
	VkSampler   sampler = getCachedNv12Sampler(inFilter);
	if (yView == VK_NULL_HANDLE || uvView == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) {
		return oa::Status::error(oa::StatusCode::VulkanError, "ConvertNv12ToRgbCompute: failed to cache views/sampler");
	}

	// Note: oavk::ImageDispatch::run automatically prepends bindless descriptor indices
	// to the push constants, so we only provide the user data here.
	struct Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
		oa::U32 colorSpace;
		oa::U32 fullRange;
	};

	oavk::ImageDispatchBinding bindings[4] = {};
	bindings[0].kind = oavk::DescriptorKind::SampledImage;
	bindings[0].binding = 0;
	bindings[0].imageView = yView;
	bindings[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[1].kind = oavk::DescriptorKind::SampledImage;
	bindings[1].binding = 1;
	bindings[1].imageView = uvView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[2].kind = oavk::DescriptorKind::StorageImage;
	bindings[2].binding = 2;
	bindings[2].imageView = rgbaFrame.imageView;
	bindings[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[3].kind = oavk::DescriptorKind::Sampler;
	bindings[3].binding = 3;
	bindings[3].sampler = sampler;

	Push push = {
		.width = inNv12Frame.width,
		.height = inNv12Frame.height,
		.codedWidth = impl_->codedWidth,
		.codedHeight = impl_->codedHeight,
		.colorSpace = toVisionColorSpace(inColorSpace, inNv12Frame.width, inNv12Frame.height),
		.fullRange = 0U};

	auto& vkEngine = *impl_->engine;
	oa::Status status = oavk::ImageDispatch::run(
		vkEngine,
		"CvtNv12ToRgb",
		oa::Span<const oavk::ImageDispatchBinding>(bindings, 4),
		&push,
		sizeof(push),
		oa::ScalarType::Float32,
		oa::divCeil(inNv12Frame.width, 16),
		oa::divCeil(inNv12Frame.height, 16),
		1);
	if (!status.isOk()) {
		return status;
	}

	for (oa::Usize i = 0; i < impl_->rgbImages.size(); ++i) {
		if (impl_->rgbImages[i] == rgbaFrame.image && i < impl_->rgbImageLayouts.size()) {
			impl_->rgbImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
			break;
		}
	}
	outRgbFrame = rgbaFrame;
	return oa::Status::ok();
}

// Decode with YCbCr→RGB conversion
oa::Status oa::VideoDecoder::decodeFrameWithConversion(
	const oa::Span<const oa::U8>& inBitstream,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outFrame)
{
	// first decode to NV12
	oa::VideoFrame nv12Frame;
	oa::Status status = decodeFrame(inBitstream, nv12Frame);
	if (!status.isOk())
	{
		return status;
	}
	
	// then convert NV12 to RGB if requested
	if (inOptions.convertToRgb)
	{
		oa::Status convStatus = convertNv12ToRgb(nv12Frame, inOptions, outFrame);
		if (!convStatus.isOk()) {
			return convStatus;
		}
		// ConvertNv12ToRgb transitions the NV12 DPB layer to SHADER_READ_ONLY_OPTIMAL
		// for the compute/hardware sampler. The next decode expects VIDEO_DECODE_DPB_KHR.
		// Restore the layout now so the decoder state stays consistent.
		oa::Status restoreStatus = restoreDpbLayerToDecodeLayout(nv12Frame);
		if (!restoreStatus.isOk()) {
			return restoreStatus;
		}
		return oa::Status::ok();
	}
	else
	{
		// Return NV12 frame as-is
		outFrame = nv12Frame;
		return oa::Status::ok();
	}
}

// query hardware YCbCr conversion support
bool oa::VideoDecoder::hasHardwareYCbCrConversion(oa::Engine& inRt)
{
	auto& vkEngine = inRt;
	return oa::EngineDeviceAccess::get(vkEngine).info.software.hasSamplerYcbcrConversion;
}

oa::U32 oa::VideoDecoder::getNv12PlaneArrayLayer(const oa::VideoFrame& inFrame) const
{
	for (oa::Usize i = 0; i < impl_->outputImages.size(); ++i) {
		if (impl_->outputImages[i] == inFrame.image) {
			return 0;
		}
	}
	return inFrame.arrayLayer;
}

oa::Status oa::VideoDecoder::createOutputImages(
	oa::Engine& inRt,
	const VkVideoProfileInfoKHR& inProfile,
	VkFormat inFormat,
	VkExtent2D inCodedExtent,
	oa::U32 inSlotCount)
{
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(inRt).allocator);

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles = &inProfile;

	oa::U32 sharedFamilies[3] = {
		oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.graphicsQueueFamily,
	};
	oa::U32 sharedFamilyCount = 0;
	for (oa::U32 family : sharedFamilies) {
		if (family == oavk::EnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (oa::U32 i = 0; i < sharedFamilyCount; ++i) {
			dup = dup or sharedFamilies[i] == family;
		}
		if (not dup) {
			sharedFamilies[sharedFamilyCount++] = family;
		}
	}

	impl_->outputImages.reserve(inSlotCount);
	impl_->outputViews.reserve(inSlotCount);
	impl_->outputAllocations.reserve(inSlotCount);
	impl_->outputImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);

	for (oa::U32 slot = 0; slot < inSlotCount; ++slot) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.pNext = &profileList;
		imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = inFormat;
		imageInfo.extent = {inCodedExtent.width, inCodedExtent.height, 1};
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

		vma::AllocationCreateInfo allocInfo = {};
		allocInfo.usage = vma::memoryUsageGpuOnly;

		VkImage image = VK_NULL_HANDLE;
		vma::Allocation allocation = VK_NULL_HANDLE;
		VkResult result = vma::createImage(allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create decode output image");
		}

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = inFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView view = VK_NULL_HANDLE;
		result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &view);
		if (result != VK_SUCCESS) {
			vma::destroyImage(allocator, image, allocation);
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create decode output image view");
		}

		impl_->outputImages.pushBack(image);
		impl_->outputViews.pushBack(view);
		impl_->outputAllocations.pushBack(allocation);
		if (slot < impl_->outputImageLayouts.size()) {
			impl_->outputImageLayouts[slot] = VK_IMAGE_LAYOUT_UNDEFINED;
		}
	}
	impl_->outputFrameCapacity = inSlotCount;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::createSampleStagingImages(
	oa::Engine& inRt,
	const VkVideoProfileInfoKHR& inProfile,
	VkExtent2D inCodedExtent,
	oa::U32 inSlotCount
) {
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(inRt).device);
	auto allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(inRt).allocator);

	oa::U32 sharedFamilies[3] = {
		oa::EngineDeviceAccess::get(inRt).queues.videoDecodeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily,
		oa::EngineDeviceAccess::get(inRt).queues.graphicsQueueFamily,
	};
	oa::U32 sharedFamilyCount = 0;
	for (oa::U32 family : sharedFamilies) {
		if (family == oavk::EnumerationIndexUnset) {
			continue;
		}
		bool dup = false;
		for (oa::U32 i = 0; i < sharedFamilyCount; ++i) {
			dup = dup or sharedFamilies[i] == family;
		}
		if (not dup) {
			sharedFamilies[sharedFamilyCount++] = family;
		}
	}

	impl_->sampleImages.reserve(inSlotCount);
	impl_->sampleYViews.reserve(inSlotCount);
	impl_->sampleUvViews.reserve(inSlotCount);
	impl_->sampleAllocations.reserve(inSlotCount);
	impl_->sampleImageLayouts.fill(VK_IMAGE_LAYOUT_UNDEFINED);

	for (oa::U32 slot = 0; slot < inSlotCount; ++slot) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		imageInfo.extent = {inCodedExtent.width, inCodedExtent.height, 1};
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

		vma::AllocationCreateInfo allocInfo = {};
		allocInfo.usage = vma::memoryUsageGpuOnly;

		VkImage image = VK_NULL_HANDLE;
		vma::Allocation allocation = VK_NULL_HANDLE;
		VkResult result = vma::createImage(allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create NV12 staging image");
		}

		auto createPlaneView = [&](VkFormat inFormat, VkImageAspectFlagBits inPlane, VkImageView& outView) -> oa::Status {
			VkImageViewCreateInfo viewInfo = {};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = inFormat;
			viewInfo.subresourceRange.aspectMask = inPlane;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			result = oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCreateImageView(device, &viewInfo, nullptr, &outView);
			if (result != VK_SUCCESS) {
				return oa::Status::error(oa::StatusCode::VulkanError, "Failed to create NV12 staging plane view");
			}
			return oa::Status::ok();
		};

		VkImageView yView = VK_NULL_HANDLE;
		VkImageView uvView = VK_NULL_HANDLE;
		oa::Status yStatus = createPlaneView(VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_PLANE_0_BIT, yView);
		if (!yStatus.isOk()) {
			vma::destroyImage(allocator, image, allocation);
			return yStatus;
		}
		oa::Status uvStatus = createPlaneView(VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT, uvView);
		if (!uvStatus.isOk()) {
			oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkDestroyImageView(device, yView, nullptr);
			vma::destroyImage(allocator, image, allocation);
			return uvStatus;
		}

		impl_->sampleImages.pushBack(image);
		impl_->sampleYViews.pushBack(yView);
		impl_->sampleUvViews.pushBack(uvView);
		impl_->sampleAllocations.pushBack(allocation);
		if (slot < impl_->sampleImageLayouts.size()) {
			impl_->sampleImageLayouts[slot] = VK_IMAGE_LAYOUT_UNDEFINED;
		}
	}
	(void)inProfile;
	return oa::Status::ok();
}

oa::Status oa::VideoDecoder::copyDpbLayerToSampleImage(const oa::VideoFrame& inDpbFrame)
{
	if (!impl_->useSampleStaging || !impl_->engine || !impl_->dpb.getImage()) {
		return oa::Status::ok();
	}
	if (impl_->copySampleStagingOnVideoQueue) {
		return oa::Status::ok();
	}
	if (inDpbFrame.image != impl_->dpb.getImage()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "copyDpbLayerToSampleImage: not a DPB frame");
	}
	const oa::U32 layer = inDpbFrame.arrayLayer;
	if (layer >= impl_->sampleImages.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "copyDpbLayerToSampleImage: invalid layer");
	}

	OA_RETURN_IF_ERROR(releaseDpbLayerForComputeCopy(inDpbFrame));
	auto& vkEngine = *impl_->engine;
	const oa::U64 decodeWaitValue = impl_->timelineValue;

	oavk::Stream* stream = oa::EngineSubmissionAccess::acquireStream(vkEngine);
	if (stream == nullptr) {
		return oa::Status::error(oa::StatusCode::VulkanError, "copyDpbLayerToSampleImage: stream acquire failed");
	}
	oa::Status beginStatus = stream->begin(oa::EngineDeviceAccess::get(vkEngine));
	if (!beginStatus.isOk()) {
		oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
		return beginStatus;
	}

	VkCommandBuffer cb = static_cast<VkCommandBuffer>(stream->commandBuffer);
	VkImage srcImage = impl_->dpb.getImage();
	VkImage dstImage = impl_->sampleImages[layer];
	VkImageLayout srcLayout = layer < impl_->dpbImageLayouts.size()
		? impl_->dpbImageLayouts[layer]
		: VK_IMAGE_LAYOUT_UNDEFINED;
	if (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		srcLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	}
	VkImageLayout dstLayout = layer < impl_->sampleImageLayouts.size()
		? impl_->sampleImageLayouts[layer]
		: VK_IMAGE_LAYOUT_UNDEFINED;

	auto barrier = [&](VkImage image, VkImageLayout& inOutLayout, VkImageLayout newLayout,
		VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
		VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
		oa::U32 arrayLayer) {
		if (inOutLayout == newLayout) {
			return;
		}
		VkImageMemoryBarrier2 b = {};
		b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		b.srcStageMask = srcStage;
		b.srcAccessMask = srcAccess;
		b.dstStageMask = dstStage;
		b.dstAccessMask = dstAccess;
		b.oldLayout = inOutLayout;
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
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(cb, &dep);
		inOutLayout = newLayout;
	};

	VkImageLayout srcTransfer = srcLayout;
	VkImageLayout dstTransfer = dstLayout;

	const bool sameFamily = oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueueFamily
		== oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily;
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

	const oa::U32 chromaWidth = impl_->codedWidth / 2;
	const oa::U32 chromaHeight = impl_->codedHeight / 2;

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
	copies[0].extent = {impl_->codedWidth, impl_->codedHeight, 1};

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

	if (oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImage2) {
		VkCopyImageInfo2 copyInfo = {};
		copyInfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
		copyInfo.srcImage = srcImage;
		copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		copyInfo.dstImage = dstImage;
		copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		copyInfo.regionCount = 2;
		copyInfo.pRegions = copies;
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImage2(cb, &copyInfo);
	} else {
		VkImageCopy legacy[2] = {};
		legacy[0].srcSubresource = copies[0].srcSubresource;
		legacy[0].dstSubresource = copies[0].dstSubresource;
		legacy[0].extent = copies[0].extent;
		legacy[1].srcSubresource = copies[1].srcSubresource;
		legacy[1].dstSubresource = copies[1].dstSubresource;
		legacy[1].extent = copies[1].extent;
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImage(cb,
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

	oa::Status submitStatus = stream->submitWithDependency(vkEngine, impl_->timelineSemaphore, decodeWaitValue);
	if (submitStatus.isOk()) {
		submitStatus = stream->synchronize(oa::EngineDeviceAccess::get(vkEngine));
	}
	oa::EngineSubmissionAccess::releaseStream(vkEngine, stream);
	if (!submitStatus.isOk()) {
		return submitStatus;
	}

	if (layer < impl_->dpbImageLayouts.size()) {
		impl_->dpbImageLayouts[layer] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}
	if (layer < impl_->sampleImageLayouts.size()) {
		impl_->sampleImageLayouts[layer] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	OA_RETURN_IF_ERROR(restoreDpbLayerToDecodeLayout(inDpbFrame));
	return oa::Status::ok();
}

void oa::VideoDecoder::recordDpbLayerToSampleImage(
	VkCommandBuffer inCommandBuffer,
	oa::I32 inDpbSlot)
{
	if (!impl_->copySampleStagingOnVideoQueue
		or inCommandBuffer == VK_NULL_HANDLE
		or inDpbSlot < 0
		or static_cast<oa::Usize>(inDpbSlot) >= impl_->sampleImages.size()) {
		return;
	}

	const oa::U32 layer = static_cast<oa::U32>(inDpbSlot);
	VkImageLayout& srcLayout = impl_->dpbImageLayouts[layer];
	VkImageLayout& dstLayout = impl_->sampleImageLayouts[layer];

	auto barrier = [&](VkImage inImage,
		VkImageLayout inOldLayout,
		VkImageLayout inNewLayout,
		VkPipelineStageFlags2 inSrcStage,
		VkAccessFlags2 inSrcAccess,
		VkPipelineStageFlags2 inDstStage,
		VkAccessFlags2 inDstAccess,
		oa::U32 inArrayLayer) {
		VkImageMemoryBarrier2 imageBarrier = {};
		imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		imageBarrier.srcStageMask = inSrcStage;
		imageBarrier.srcAccessMask = inSrcAccess;
		imageBarrier.dstStageMask = inDstStage;
		imageBarrier.dstAccessMask = inDstAccess;
		imageBarrier.oldLayout = inOldLayout;
		imageBarrier.newLayout = inNewLayout;
		imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageBarrier.image = inImage;
		imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBarrier.subresourceRange.baseMipLevel = 0;
		imageBarrier.subresourceRange.levelCount = 1;
		imageBarrier.subresourceRange.baseArrayLayer = inArrayLayer;
		imageBarrier.subresourceRange.layerCount = 1;
		VkDependencyInfo dependency = {};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &imageBarrier;
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdPipelineBarrier2(inCommandBuffer, &dependency);
	};

	barrier(
		impl_->dpb.getImage(),
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
		impl_->sampleImages[layer],
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
	regions[0].extent = {impl_->codedWidth, impl_->codedHeight, 1};
	regions[1].srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	regions[1].srcSubresource.layerCount = 1;
	regions[1].srcSubresource.baseArrayLayer = layer;
	regions[1].dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	regions[1].dstSubresource.layerCount = 1;
	regions[1].extent = {impl_->codedWidth / 2, impl_->codedHeight / 2, 1};
	oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch.vkCmdCopyImage(
		inCommandBuffer,
		impl_->dpb.getImage(),
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		impl_->sampleImages[layer],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		2,
		regions);

	barrier(
		impl_->sampleImages[layer],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_ACCESS_2_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
		0,
		0);
	barrier(
		impl_->dpb.getImage(),
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
