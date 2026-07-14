// Shared VkVideoProfileInfoKHR builder for OaVideoDecoder.

#include "VideoDecoderProfile.h"

namespace OaVideoDecoderProfile {

VkVideoProfileInfoKHR BuildDecodeProfile(
	OaVideoCodec InCodec,
	VkVideoDecodeH264ProfileInfoKHR& OutH264,
	VkVideoDecodeH265ProfileInfoKHR& OutH265,
	VkVideoDecodeAV1ProfileInfoKHR& OutAV1,
	VkVideoDecodeVP9ProfileInfoKHR& OutVp9)
{
	VkVideoProfileInfoKHR profile = {};
	profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	switch (InCodec) {
		case OaVideoCodec::H264:
			OutH264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
			OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
			// Picture layout must match the bitstream. Modern H.264 content
			// is overwhelmingly progressive; declaring INTERLACED here makes
			// the driver expect field pictures and treat each P-frame's
			// references as half-frames, which breaks motion compensation
			// in proportion to GOP length — exactly the "looks fine at the
			// cut, gets worse as the shot continues" symptom. PROGRESSIVE
			// also matches what shibuya uses (field_pic_flag = 0 in every
			// slice header). If we ever want interlaced support, sniff
			// frame_mbs_only_flag from the SPS and pick at session create.
			OutH264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR;
			profile.pNext = &OutH264;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
			break;
		case OaVideoCodec::H265:
			OutH265.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR;
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
			profile.pNext = &OutH265;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
			break;
		case OaVideoCodec::AV1:
			OutAV1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
			OutAV1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
			OutAV1.filmGrainSupport = VK_FALSE;
			profile.pNext = &OutAV1;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
			break;
		case OaVideoCodec::VP9:
			OutVp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR;
			OutVp9.stdProfile = STD_VIDEO_VP9_PROFILE_0;
			profile.pNext = &OutVp9;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
			break;
	}
	profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	profile.lumaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	profile.chromaBitDepth    = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	return profile;
}

} // namespace OaVideoDecoderProfile
