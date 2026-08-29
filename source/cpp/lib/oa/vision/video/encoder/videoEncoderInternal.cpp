#include "videoEncoderInternal.h"
#include "videoEncoderImpl.h"

oa::Status oa::VideoEncoderAccess::uploadInputRgba(
	VideoEncoder& inEncoder,
	const oavk::Buffer& inRgba,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::YCbCrModel inColorSpace,
	bool inFullRange
) {
	return inEncoder.uploadInputRgba(inRgba, inVisibleWidth, inVisibleHeight, inColorSpace, inFullRange);
}

oa::Status oa::VideoEncoderAccess::encodeFrame(
	VideoEncoder& inEncoder,
	VkImage inImage,
	oa::U64 inPts,
	EncodedVideoPacket& outFrame
) {
	return inEncoder.encodeFrame(inImage, inPts, outFrame);
}

oa::Status oa::VideoEncoderAccess::submitRgba(
	VideoEncoder& inEncoder,
	const oavk::Buffer& inRgba,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::U64 inPts,
	oa::Vector<EncodedVideoPacket>& outReady,
	YCbCrModel inColorSpace,
	bool inFullRange
) {
	return inEncoder.submitRgba(
		inRgba, inVisibleWidth, inVisibleHeight, inPts, outReady,
		inColorSpace, inFullRange
	);
}

oa::Status oa::VideoEncoderAccess::submitRgbaImage(
	VideoEncoder& inEncoder,
	VkImage inImage,
	VkImageView inImageView,
	VkFormat inFormat,
	VkImageLayout inLayout,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::U64 inPts,
	oa::Vector<EncodedVideoPacket>& outReady,
	YCbCrModel inColorSpace,
	bool inFullRange,
	oa::U32 inArrayLayer,
	oa::Event inReady,
	oa::U32 inExternalQueueFamilyIndex,
	oa::Event* outInputConsumed
) {
	return inEncoder.submitRgbaImage(
		inImage, inImageView, inFormat, inLayout,
		inVisibleWidth, inVisibleHeight, inPts, outReady,
		inColorSpace, inFullRange, inArrayLayer, inReady,
		inExternalQueueFamilyIndex, outInputConsumed
	);
}

oa::U64 oa::VideoEncoderAccess::zeroFeedbackRecoveryCount(
	const VideoEncoder& inEncoder) noexcept
{
	return inEncoder.impl_ ? inEncoder.impl_->zeroFeedbackRecoveryCount : 0U;
}
