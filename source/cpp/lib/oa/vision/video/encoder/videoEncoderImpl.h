#pragma once

#include <oa/vision/videoEncoder.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/imageDispatch.h>
#include <oa/runtime/oaVkVideo.h>

struct oa::VideoEncoder::EncodeSlot {
	// Stable packed-RGBA snapshot. submitRgba owns the caller data before
	// returning so the producer may immediately recycle its buffer.
	oavk::Buffer rgbaSnapshot;
	VkImage inputImage = VK_NULL_HANDLE;
	VkImageView inputView = VK_NULL_HANDLE;
	VkImageView inputYView = VK_NULL_HANDLE;
	VkImageView inputUvView = VK_NULL_HANDLE;
	oa::U32 inputYBindless = 0U;
	oa::U32 inputUvBindless = 0U;
	bool inputBindlessRegistered = false;
	void* inputAllocation = nullptr;
	oavk::ImageDispatchTicket inputTicket;
	bool inputInitialized = false;
	oavk::VideoBitstream bitstream;
	oa::U64 bitstreamDirtyEnd = 0U;
	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	VkQueryPool queryPool = VK_NULL_HANDLE;
	oa::U64 presentationTimestamp = 0U;
	bool isKeyframe = false;
	bool pending = false;

	EncodeSlot() = default;
	EncodeSlot(EncodeSlot&&) noexcept = default;
	EncodeSlot& operator=(EncodeSlot&&) noexcept = default;
	EncodeSlot(const EncodeSlot&) = delete;
	EncodeSlot& operator=(const EncodeSlot&) = delete;
};

class oa::VideoEncoder::Impl {
public:
	oa::VideoEncodeProfile profile = {};
	oa::Engine* engine = nullptr;

	oavk::VideoSession session;
	oavk::VideoParameters sessionParameters;
	oavk::VideoQueue queue;
	oavk::VideoDpb dpb;
	oa::U32 dpbSlotCapacity = 0U;

	oa::Vec<EncodeSlot> slots;
	oa::U32 submitSlot = 0U;
	oa::U32 harvestSlot = 0U;
	oa::U32 pendingSlots = 0U;
	bool compatibilityUploadReady = false;
	oa::Vec<oa::U8> cachedHeaders;

	bool rateControlReset = false;
	bool queryResultStatusSupported = false;
	oa::U64 zeroFeedbackRecoveryCount = 0U;
	VkVideoEncodeRateControlModeFlagBitsKHR rateControlMode =
		VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;

	oa::U32 codedWidth = 0U;
	oa::U32 codedHeight = 0U;
	oa::U64 minBitstreamBufferOffsetAlignment = 1U;
	oa::U64 minBitstreamBufferSizeAlignment = 1U;
	oa::U32 frameCount = 0U;
	oa::U32 lastKeyframeIndex = 0U;
	oa::U32 gopSize = 30U;
	oa::U32 currentGopFrame = 0U;
};
