// Shared Vulkan record helpers for FnVideoDecoder* Record* bodies.

#include "FnVideoDecoderRecordShared.h"
#include <Oa/Runtime/Engine.h>

#include <algorithm>

OaResult<OaFnVideoDecoderRecord::ActiveCmd> OaFnVideoDecoderRecord::Begin(
	OaVideoDecoder& InDecoder,
	const char* InLabel)
{
	if (!InDecoder.Rt_ || !InDecoder.CmdBuffers_[0]) {
		return OaStatus::Error("Video decoder command resources are not initialized");
	}
	auto slot = InDecoder.AcquireVideoCmdSlot();
	if (!slot.Status.IsOk()) {
		return slot.Status;
	}
	VkCommandBufferBeginInfo cmdBegin = {};
	cmdBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	const VkResult result = vkBeginCommandBuffer(slot.cb, &cmdBegin);
	if (result != VK_SUCCESS) {
		InDecoder.ReleaseVideoCmdSlot();
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			OaString("vkBeginCommandBuffer failed for ") + InLabel);
	}
	return ActiveCmd{slot.cb, slot.fence};
}

void OaFnVideoDecoderRecord::ReleaseSlot(OaVideoDecoder& InDecoder)
{
	InDecoder.ReleaseVideoCmdSlot();
}

bool OaFnVideoDecoderRecord::GetDpbView(
	OaVideoDecoder& InDecoder,
	OaI32 InSlot,
	VkImageView& OutView,
	OaU32& OutBaseLayer)
{
	if (InSlot < 0 || static_cast<OaU32>(InSlot) >= InDecoder.DpbSlotCapacity_) {
		return false;
	}
	OutView = InDecoder.Dpb_.GetView();
	OutBaseLayer = static_cast<OaU32>(InSlot);
	return OutView != VK_NULL_HANDLE;
}

OaStatus OaFnVideoDecoderRecord::ResolveOutputView(
	OaVideoDecoder& InDecoder,
	OaI32 InDpbSlot,
	VkImageView InSetupDpbView,
	VkImageView& OutDstView,
	bool& OutHasDistinctOutput)
{
	OutHasDistinctOutput = !InDecoder.OutputImages_.Empty()
		&& static_cast<OaUsize>(InDpbSlot) < InDecoder.OutputImages_.Size();
	if (OutHasDistinctOutput) {
		OutDstView = InDecoder.OutputViews_[InDpbSlot];
	} else {
		OutDstView = InSetupDpbView;
	}
	if (OutDstView == VK_NULL_HANDLE) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"Video decode requires a profile-compatible output image view");
	}
	return OaStatus::Ok();
}

void OaFnVideoDecoderRecord::TransitionDecodeImage(
	const ActiveCmd& InCmd,
	VkImage InImage,
	VkImageLayout& InOutLayout,
	VkImageLayout InNewLayout,
	OaU32 InBaseLayer,
	OaU32 InLayerCount)
{
	if (InOutLayout == InNewLayout) {
		return;
	}
	const VkImageLayout oldLayout = InOutLayout;
	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
		? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
		: VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.srcAccessMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
		? 0
		: VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
	barrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	barrier.dstAccessMask = InNewLayout == VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
		? VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR
		: VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = InNewLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = InImage;
	// Decoder images are non-disjoint multi-planar images. Vulkan requires
	// whole-image layout transitions to use COLOR_BIT; plane aspects are only
	// valid here for images created with VK_IMAGE_CREATE_DISJOINT_BIT.
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = InBaseLayer;
	barrier.subresourceRange.layerCount = InLayerCount;
	VkDependencyInfo dependencyInfo = {};
	dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependencyInfo.imageMemoryBarrierCount = 1;
	dependencyInfo.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(InCmd.Cb, &dependencyInfo);
	InOutLayout = InNewLayout;
}

void OaFnVideoDecoderRecord::EnsureDpbLayer(
	ActiveCmd& InCmd,
	OaVideoDecoder& InDecoder,
	OaI32 InSlot)
{
	if (InSlot < 0 || static_cast<OaU32>(InSlot) >= InDecoder.DpbSlotCapacity_) {
		return;
	}
	const OaU32 layer = static_cast<OaU32>(InSlot);
	if (InDecoder.DpbImageLayouts_[layer] == VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR) {
		return;
	}
	bool wholeDpbUndefined = true;
	for (VkImageLayout layout : InDecoder.DpbImageLayouts_) {
		if (layout != VK_IMAGE_LAYOUT_UNDEFINED) {
			wholeDpbUndefined = false;
			break;
		}
	}
	if (wholeDpbUndefined) {
		// The DPB is one array image. Initialize every layer in one transition
		// before the first decode instead of lazily transitioning a new layer
		// after another layer has already been written by the codec engine.
		TransitionDecodeImage(
			InCmd,
			InDecoder.Dpb_.GetImage(),
			InDecoder.DpbImageLayouts_[0],
			VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
			0,
			InDecoder.DpbSlotCapacity_);
		InDecoder.DpbImageLayouts_.Fill(VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
		return;
	}
	TransitionDecodeImage(
		InCmd,
		InDecoder.Dpb_.GetImage(),
		InDecoder.DpbImageLayouts_[layer],
		VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
		layer,
		1);
}

void OaFnVideoDecoderRecord::EnsureDistinctOutput(
	ActiveCmd& InCmd,
	OaVideoDecoder& InDecoder,
	OaI32 InDpbSlot,
	bool InHasDistinctOutput)
{
	if (!InHasDistinctOutput) {
		return;
	}
	TransitionDecodeImage(
		InCmd,
		InDecoder.OutputImages_[InDpbSlot],
		InDecoder.OutputImageLayouts_[InDpbSlot],
		VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
		0,
		1);
}

void OaFnVideoDecoderRecord::ResetSessionIfNeeded(
	const ActiveCmd& InCmd,
	OaVideoDecoder& InDecoder)
{
	if (InDecoder.VideoSessionInitialized_) {
		return;
	}
	VkVideoCodingControlInfoKHR controlInfo = {};
	controlInfo.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
	controlInfo.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
	vkCmdControlVideoCodingKHR(InCmd.Cb, &controlInfo);
	InDecoder.VideoSessionInitialized_ = true;
}

void OaFnVideoDecoderRecord::EmitBitstreamDecodeBarrier(
	const ActiveCmd& InCmd,
	VkBuffer InBuffer,
	VkDeviceSize InOffset,
	VkDeviceSize InSize)
{
	VkBufferMemoryBarrier2 bitstreamBarrier = {};
	bitstreamBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	bitstreamBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	bitstreamBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
	bitstreamBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
	bitstreamBarrier.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
	bitstreamBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bitstreamBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bitstreamBarrier.buffer = InBuffer;
	bitstreamBarrier.offset = InOffset;
	bitstreamBarrier.size = InSize;
	VkDependencyInfo bitstreamDep = {};
	bitstreamDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	bitstreamDep.bufferMemoryBarrierCount = 1;
	bitstreamDep.pBufferMemoryBarriers = &bitstreamBarrier;
	vkCmdPipelineBarrier2(InCmd.Cb, &bitstreamDep);
}

OaStatus OaFnVideoDecoderRecord::FinishAndSubmit(
	OaVideoDecoder& InDecoder,
	const ActiveCmd& InCmd,
	FinishParams InParams)
{
	auto& vkEngine = *InDecoder.Rt_;

	if (InDecoder.CopySampleStagingOnVideoQueue_) {
		InDecoder.RecordDpbLayerToSampleImage(InCmd.Cb, InParams.DpbSlot);
	}

	VkResult result = vkEndCommandBuffer(InCmd.Cb);
	if (result != VK_SUCCESS) {
		ReleaseSlot(InDecoder);
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			OaString("vkEndCommandBuffer failed for ") + InParams.ErrorContext);
	}

	const OaU64 signalValue = InDecoder.TimelineValue_ + 1;
	VkSemaphore sem = static_cast<VkSemaphore>(InDecoder.TimelineSem_.Semaphore);
	VkSemaphore reuseSemaphore = InParams.HasDistinctOutput
		? InDecoder.OutputReuseSemaphores_[InParams.DpbSlot]
		: VK_NULL_HANDLE;
	const OaU64 reuseValue = InParams.HasDistinctOutput
		? InDecoder.OutputReuseValues_[InParams.DpbSlot]
		: 0;
	VkSemaphore waitSemaphores[2] = {};
	OaU64 waitValues[2] = {};
	VkPipelineStageFlags2 waitStages[2] = {};
	OaU32 waitCount = 0;
	// Queue submission order is not a memory dependency. Chain decoder jobs on
	// their own timeline so DPB writes and following layout transitions are
	// ordered without a host wait or a queue-wide idle.
	if (InDecoder.TimelineValue_ > 0) {
		waitSemaphores[waitCount] = sem;
		waitValues[waitCount] = InDecoder.TimelineValue_;
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
	for (OaU32 waitIdx = 0; waitIdx < waitCount; ++waitIdx) {
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
	commandInfo.commandBuffer = InCmd.Cb;
	VkSubmitInfo2 submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submitInfo.waitSemaphoreInfoCount = waitCount;
	submitInfo.pWaitSemaphoreInfos = waitCount > 0 ? waitInfos : nullptr;
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &commandInfo;
	submitInfo.signalSemaphoreInfoCount = 1;
	submitInfo.pSignalSemaphoreInfos = &signalInfo;
	result = vkQueueSubmit2(
		static_cast<VkQueue>(vkEngine.Device.Queues.VideoDecodeQueue),
		1,
		&submitInfo,
		InCmd.Fence);
	if (result != VK_SUCCESS) {
		// Don't reset fence on error - it was never submitted, so it's still signaled.
		// Resetting it would cause AcquireVideoCmdSlot() to wait forever on next use.
		ReleaseSlot(InDecoder);
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			OaString("vkQueueSubmit failed for ") + InParams.ErrorContext);
	}

	InDecoder.TimelineValue_ = signalValue;
	InDecoder.BitstreamRing_[InDecoder.CurrentBitstreamIndex_].UseValue = InDecoder.TimelineValue_;
	InDecoder.OutputReuseSemaphores_[InParams.DpbSlot] = VK_NULL_HANDLE;
	InDecoder.OutputReuseValues_[InParams.DpbSlot] = 0;
	// Submission order makes this the slot state observed by the next decode.
	// A codec can invalidate an already-active slot by reconstructing a
	// non-reference picture into it, so this must assign both true and false.
	InDecoder.SlotDeviceActivated_[InParams.DpbSlot] =
		InParams.MarkSlotDeviceActivated;
	ReleaseSlot(InDecoder);
	if (InParams.HasDistinctOutput) {
		InDecoder.DpbImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
		InDecoder.OutputImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;
	} else {
		InDecoder.DpbImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	}
	return OaStatus::Ok();
}
