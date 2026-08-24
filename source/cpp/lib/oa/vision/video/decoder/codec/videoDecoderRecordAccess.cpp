// Shared vulkan record helpers for oa::VideoDecoder codec bodies.

#include "videoDecoderRecordAccess.h"
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/deviceAccess.h>

#include <algorithm>

oa::Result<oa::VideoDecoderRecordAccess::ActiveCmd> oa::VideoDecoderRecordAccess::begin(
	oa::VideoDecoder& inDecoder,
	const char* inLabel)
{
	if (!inDecoder.impl_->engine || !inDecoder.impl_->commandBuffers[0]) {
		return oa::Status::error("Video decoder command resources are not initialized");
	}
	auto slot = inDecoder.acquireVideoCmdSlot();
	if (!slot.status.isOk()) {
		return slot.status;
	}
	VkCommandBufferBeginInfo cmdBegin = {};
	cmdBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	const auto& device = oa::EngineDeviceAccess::get(*inDecoder.impl_->engine);
	const VkResult result = device.deviceDispatch.vkBeginCommandBuffer(
		slot.cb, &cmdBegin);
	if (result != VK_SUCCESS) {
		inDecoder.releaseVideoCmdSlot();
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			oa::String("vkBeginCommandBuffer failed for ") + inLabel);
	}
	return ActiveCmd{slot.cb, slot.fence, &device.deviceDispatch};
}

void oa::VideoDecoderRecordAccess::releaseSlot(oa::VideoDecoder& inDecoder)
{
	inDecoder.releaseVideoCmdSlot();
}

bool oa::VideoDecoderRecordAccess::getDpbView(
	oa::VideoDecoder& inDecoder,
	oa::I32 inSlot,
	VkImageView& outView,
	oa::U32& outBaseLayer)
{
	if (inSlot < 0 || static_cast<oa::U32>(inSlot) >= inDecoder.impl_->dpbSlotCapacity) {
		return false;
	}
	outView = inDecoder.impl_->dpb.getView();
	outBaseLayer = static_cast<oa::U32>(inSlot);
	return outView != VK_NULL_HANDLE;
}

oa::Status oa::VideoDecoderRecordAccess::resolveOutputView(
	oa::VideoDecoder& inDecoder,
	oa::I32 inDpbSlot,
	VkImageView inSetupDpbView,
	VkImageView& outDstView,
	bool& outHasDistinctOutput)
{
	outHasDistinctOutput = !inDecoder.impl_->outputImages.empty()
		&& static_cast<oa::Usize>(inDpbSlot) < inDecoder.impl_->outputImages.size();
	if (outHasDistinctOutput) {
		outDstView = inDecoder.impl_->outputViews[inDpbSlot];
	} else {
		outDstView = inSetupDpbView;
	}
	if (outDstView == VK_NULL_HANDLE) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			"Video decode requires a profile-compatible output image view");
	}
	return oa::Status::ok();
}

void oa::VideoDecoderRecordAccess::transitionDecodeImage(
	const ActiveCmd& inCmd,
	VkImage inImage,
	VkImageLayout& inOutLayout,
	VkImageLayout inNewLayout,
	oa::U32 inBaseLayer,
	oa::U32 inLayerCount)
{
	if (inOutLayout == inNewLayout) {
		return;
	}
	const VkImageLayout oldLayout = inOutLayout;
	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
		? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
		: VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.srcAccessMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
		? 0
		: VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.dstAccessMask = inNewLayout == VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
		? VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR
		: VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = inNewLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = inImage;
	// Decoder images are non-disjoint multi-planar images. vulkan requires
	// whole-image layout transitions to use COLOR_BIT; plane aspects are only
	// valid here for images created with VK_IMAGE_CREATE_DISJOINT_BIT.
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = inBaseLayer;
	barrier.subresourceRange.layerCount = inLayerCount;
	VkDependencyInfo dependencyInfo = {};
	dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependencyInfo.imageMemoryBarrierCount = 1;
	dependencyInfo.pImageMemoryBarriers = &barrier;
	inCmd.deviceDispatch->vkCmdPipelineBarrier2(inCmd.cb, &dependencyInfo);
	inOutLayout = inNewLayout;
}

void oa::VideoDecoderRecordAccess::ensureDpbLayer(
	ActiveCmd& inCmd,
	oa::VideoDecoder& inDecoder,
	oa::I32 inSlot)
{
	if (inSlot < 0 || static_cast<oa::U32>(inSlot) >= inDecoder.impl_->dpbSlotCapacity) {
		return;
	}
	const oa::U32 layer = static_cast<oa::U32>(inSlot);
	if (inDecoder.impl_->dpbImageLayouts[layer] == VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
		return;
	}
	bool wholeDpbUndefined = true;
	for (VkImageLayout layout : inDecoder.impl_->dpbImageLayouts) {
		if (layout != VK_IMAGE_LAYOUT_UNDEFINED) {
			wholeDpbUndefined = false;
			break;
		}
	}
	if (wholeDpbUndefined) {
		// The DPB is one array image. initialize every layer in one transition
		// before the first decode instead of lazily transitioning a new layer
		// after another layer has already been written by the codec engine.
		transitionDecodeImage(
			inCmd,
			inDecoder.impl_->dpb.getImage(),
			inDecoder.impl_->dpbImageLayouts[0],
			VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
			0,
			inDecoder.impl_->dpbSlotCapacity);
		inDecoder.impl_->dpbImageLayouts.fill(VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
		return;
	}
	transitionDecodeImage(
		inCmd,
		inDecoder.impl_->dpb.getImage(),
		inDecoder.impl_->dpbImageLayouts[layer],
		VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
		layer,
		1);
}

void oa::VideoDecoderRecordAccess::ensureDistinctOutput(
	ActiveCmd& inCmd,
	oa::VideoDecoder& inDecoder,
	oa::I32 inDpbSlot,
	bool inHasDistinctOutput)
{
	if (!inHasDistinctOutput) {
		return;
	}
	transitionDecodeImage(
		inCmd,
		inDecoder.impl_->outputImages[inDpbSlot],
		inDecoder.impl_->outputImageLayouts[inDpbSlot],
		VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
		0,
		1);
}

void oa::VideoDecoderRecordAccess::resetSessionIfNeeded(
	const ActiveCmd& inCmd,
	oa::VideoDecoder& inDecoder)
{
	if (inDecoder.impl_->videoSessionInitialized) {
		return;
	}
	VkVideoCodingControlInfoKHR controlInfo = {};
	controlInfo.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
	controlInfo.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
	inCmd.deviceDispatch->vkCmdControlVideoCodingKHR(inCmd.cb, &controlInfo);
	inDecoder.impl_->videoSessionInitialized = true;
}

void oa::VideoDecoderRecordAccess::emitBitstreamDecodeBarrier(
	const ActiveCmd& inCmd,
	VkBuffer inBuffer,
	VkDeviceSize inOffset,
	VkDeviceSize inSize)
{
	VkBufferMemoryBarrier2 bitstreamBarrier = {};
	bitstreamBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	bitstreamBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	bitstreamBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
	bitstreamBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	bitstreamBarrier.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
	bitstreamBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bitstreamBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bitstreamBarrier.buffer = inBuffer;
	bitstreamBarrier.offset = inOffset;
	bitstreamBarrier.size = inSize;
	VkDependencyInfo bitstreamDep = {};
	bitstreamDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	bitstreamDep.bufferMemoryBarrierCount = 1;
	bitstreamDep.pBufferMemoryBarriers = &bitstreamBarrier;
	inCmd.deviceDispatch->vkCmdPipelineBarrier2(inCmd.cb, &bitstreamDep);
}

oa::Status oa::VideoDecoderRecordAccess::finishAndSubmit(
	oa::VideoDecoder& inDecoder,
	const ActiveCmd& inCmd,
	FinishParams inParams)
{
	auto& vkEngine = *inDecoder.impl_->engine;

	if (inDecoder.impl_->copySampleStagingOnVideoQueue) {
		inDecoder.recordDpbLayerToSampleImage(inCmd.cb, inParams.dpbSlot);
	}

	VkResult result = inCmd.deviceDispatch->vkEndCommandBuffer(inCmd.cb);
	if (result != VK_SUCCESS) {
		releaseSlot(inDecoder);
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			oa::String("vkEndCommandBuffer failed for ") + inParams.errorContext);
	}

	const oa::U64 signalValue = inDecoder.impl_->timelineValue + 1;
	VkSemaphore sem = static_cast<VkSemaphore>(inDecoder.impl_->timelineSemaphore.semaphore);
	VkSemaphore reuseSemaphore = inParams.hasDistinctOutput
		? inDecoder.impl_->outputReuseSemaphores[inParams.dpbSlot]
		: VK_NULL_HANDLE;
	const oa::U64 reuseValue = inParams.hasDistinctOutput
		? inDecoder.impl_->outputReuseValues[inParams.dpbSlot]
		: 0;
	VkSemaphore waitSemaphores[2] = {};
	oa::U64 waitValues[2] = {};
	VkPipelineStageFlags2 waitStages[2] = {};
	oa::U32 waitCount = 0;
	// queue submission order is not a memory dependency. Chain decoder jobs on
	// their own timeline so DPB writes and following layout transitions are
	// ordered without a host wait or a queue-wide idle.
	if (inDecoder.impl_->timelineValue > 0) {
		waitSemaphores[waitCount] = sem;
		waitValues[waitCount] = inDecoder.impl_->timelineValue;
		waitStages[waitCount] = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		++waitCount;
	}
	if (reuseSemaphore != VK_NULL_HANDLE && reuseValue > 0) {
		if (waitCount > 0 && waitSemaphores[0] == reuseSemaphore) {
			waitValues[0] = std::max(waitValues[0], reuseValue);
			waitStages[0] |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
		} else {
			waitSemaphores[waitCount] = reuseSemaphore;
			waitValues[waitCount] = reuseValue;
			waitStages[waitCount] = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
			++waitCount;
		}
	}
	VkSemaphoreSubmitInfo waitInfos[2] = {};
	for (oa::U32 waitIdx = 0; waitIdx < waitCount; ++waitIdx) {
		waitInfos[waitIdx].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
		waitInfos[waitIdx].semaphore = waitSemaphores[waitIdx];
		waitInfos[waitIdx].value = waitValues[waitIdx];
		waitInfos[waitIdx].stageMask = waitStages[waitIdx];
	}
	VkSemaphoreSubmitInfo signalInfo = {};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	signalInfo.semaphore = sem;
	signalInfo.value = signalValue;
	signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	VkCommandBufferSubmitInfo commandInfo = {};
	commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
	commandInfo.commandBuffer = inCmd.cb;
	VkSubmitInfo2 submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submitInfo.waitSemaphoreInfoCount = waitCount;
	submitInfo.pWaitSemaphoreInfos = waitCount > 0 ? waitInfos : nullptr;
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &commandInfo;
	submitInfo.signalSemaphoreInfoCount = 1;
	submitInfo.pSignalSemaphoreInfos = &signalInfo;
	result = inCmd.deviceDispatch->vkQueueSubmit2(
		static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoDecodeQueue),
		1,
		&submitInfo,
		inCmd.fence);
	if (result != VK_SUCCESS) {
		// Don't reset fence on error - it was never submitted, so it's still signaled.
		// Resetting it would cause acquireVideoCmdSlot() to wait forever on next use.
		releaseSlot(inDecoder);
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			oa::String("vkQueueSubmit failed for ") + inParams.errorContext);
	}

	inDecoder.impl_->timelineValue = signalValue;
	inDecoder.impl_->bitstreamRing[inDecoder.impl_->currentBitstreamIndex].useValue = inDecoder.impl_->timelineValue;
	inDecoder.impl_->outputReuseSemaphores[inParams.dpbSlot] = VK_NULL_HANDLE;
	inDecoder.impl_->outputReuseValues[inParams.dpbSlot] = 0;
	// Submission order makes this the slot state observed by the next decode.
	// A codec can invalidate an already-active slot by reconstructing a
	// non-reference picture into it, so this must assign both true and false.
	inDecoder.impl_->slotDeviceActivated[inParams.dpbSlot] =
		inParams.markSlotDeviceActivated;
	releaseSlot(inDecoder);
	if (inParams.hasDistinctOutput) {
		inDecoder.impl_->dpbImageLayouts[inParams.dpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
		inDecoder.impl_->outputImageLayouts[inParams.dpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;
	} else {
		inDecoder.impl_->dpbImageLayouts[inParams.dpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	}
	return oa::Status::ok();
}
