// Shared Vulkan record helpers for FnVideoDecoder* Record* bodies.

#include "FnVideoDecoderRecordShared.h"
#include <Oa/Runtime/Engine.h>

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
	auto& vkEngine = static_cast<OaComputeEngine&>(*InDecoder.Rt_);
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

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
	VkPipelineStageFlags reuseStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkTimelineSemaphoreSubmitInfo tsInfo = {};
	tsInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	tsInfo.waitSemaphoreValueCount = reuseSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	tsInfo.pWaitSemaphoreValues = reuseSemaphore != VK_NULL_HANDLE ? &reuseValue : nullptr;
	tsInfo.signalSemaphoreValueCount = 1;
	tsInfo.pSignalSemaphoreValues = &signalValue;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = &tsInfo;
	submitInfo.waitSemaphoreCount = reuseSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	submitInfo.pWaitSemaphores = reuseSemaphore != VK_NULL_HANDLE ? &reuseSemaphore : nullptr;
	submitInfo.pWaitDstStageMask = reuseSemaphore != VK_NULL_HANDLE ? &reuseStage : nullptr;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &InCmd.Cb;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &sem;
	result = vkQueueSubmit(
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
	if (InParams.MarkSlotDeviceActivated) {
		InDecoder.SlotDeviceActivated_[InParams.DpbSlot] = true;
	}
	ReleaseSlot(InDecoder);
	if (InParams.HasDistinctOutput) {
		InDecoder.DpbImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
		InDecoder.OutputImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;
	} else {
		InDecoder.DpbImageLayouts_[InParams.DpbSlot] = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR;
	}
	return OaStatus::Ok();
}
