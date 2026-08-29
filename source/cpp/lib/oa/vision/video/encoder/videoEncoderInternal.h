// Internal access to physical oa::VideoEncoder buffer/image routes.

#pragma once

#include <oa/vision/videoEncoder.h>

namespace oa {

struct VideoEncoderAccess {
	static oa::Status uploadInputRgba(
		VideoEncoder& inEncoder,
		const oavk::Buffer& inRgba,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		oa::YCbCrModel inColorSpace = oa::YCbCrModel::BT709,
		bool inFullRange = false);
	static oa::Status encodeFrame(
		VideoEncoder& inEncoder,
		VkImage inImage,
		oa::U64 inPts,
		EncodedVideoPacket& outFrame);
	static oa::Status submitRgba(
		VideoEncoder& inEncoder,
		const oavk::Buffer& inRgba,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		oa::U64 inPts,
		oa::Vector<EncodedVideoPacket>& outReady,
		YCbCrModel inColorSpace = YCbCrModel::BT709,
		bool inFullRange = false);
	static oa::Status submitRgbaImage(
		VideoEncoder& inEncoder,
		VkImage inImage,
		VkImageView inImageView,
		VkFormat inFormat,
		VkImageLayout inLayout,
		oa::U32 inVisibleWidth,
		oa::U32 inVisibleHeight,
		oa::U64 inPts,
		oa::Vector<EncodedVideoPacket>& outReady,
		YCbCrModel inColorSpace = YCbCrModel::BT709,
		bool inFullRange = false,
		oa::U32 inArrayLayer = 0U,
		oa::Event inReady = {},
		oa::U32 inExternalQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		oa::Event* outInputConsumed = nullptr);
	[[nodiscard]] static oa::U64 zeroFeedbackRecoveryCount(
		const VideoEncoder& inEncoder) noexcept;
};

} // namespace oa
