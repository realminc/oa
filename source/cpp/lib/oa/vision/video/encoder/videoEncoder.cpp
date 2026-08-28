// OA Vision — hardware Video encoder Implementation
// VK_KHR_video_encode_queue + VK_KHR_video_encode_h264 / h265
//
// This file currently lands:
//   - queryEncodeCapabilities (mirrors oa::VideoDecoder::queryDecodeCapabilities).
//   - move ctor / move assign / dtor / reset_ over the expanded state members.
//   - create() — full session bring-up: VkVideoSessionKHR + memory binding
//     + vkVideoSessionParametersKHR (manufactured H.264 SPS/PPS — the
//     encoder writes these, the decoder parses them) + DPB image array
//     + NV12 input image + bitstream output buffer + command pool /
//     command buffer on the video-encode queue family.
//   - encodeFrame / flush — still return unavailable (3g.3 — the actual
//     vkCmdEncodeVideoKHR + rate control + GOP / IDR insertion +
//     bitstream readback).
//   - Transcoder pipeline — vulkan decode -> compute conversion -> vulkan encode.

#include <oa/vision/videoEncoder.h>
#include "videoEncoderImpl.h"
#include <oa/core/std/format.h>
#include <oa/core/log.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/imageDispatch.h>
#include <vma/vma.hpp>
#include "oa/runtime/engine/borrowedServiceRetirement.h"
#include "videoEncoderInternal.h"
#include "../decoder/videoDecoderInternal.h"


// ──────────────────────────────────────────────────────────────────────
//                         file-local helpers
// ──────────────────────────────────────────────────────────────────────

namespace {

template<typename session>
oa::Status closeSessionAfterCreateFailure(session& inSession,	const oa::Status& inCreateFailure, const char* inSessionName) {
	const oa::Status closeStatus = inSession.close();
	if (closeStatus.isOk()) {
		return inCreateFailure;
	}

	oa::String message = inSessionName;
	message += " creation failed: ";
	message += inCreateFailure.toString();
	message += "; cleanup also failed: ";
	message += closeStatus.toString();
	return oa::Status::error(closeStatus.getCode(), oa::move(message));
}

// Builds the VkVideoProfileInfoKHR chain for an encode profile.
// Same shape as the decoder's GetVideoProfileInfo helper but with encode
// codec-operation flags + encode-side profile-info structs.
VkVideoProfileInfoKHR buildEncodeProfileInfo(
	oa::VideoCodec inCodec,
	VkVideoEncodeH264ProfileInfoKHR& outH264,
	VkVideoEncodeH265ProfileInfoKHR& outH265)
{
	VkVideoProfileInfoKHR profile = {};
	profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	switch (inCodec) {
		case oa::VideoCodec::H264:
			outH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR;
			outH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
			profile.pNext = &outH264;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
			break;
		case oa::VideoCodec::H265:
			outH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR;
			outH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
			profile.pNext = &outH265;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
			break;
		case oa::VideoCodec::AV1:
			// AV1 encode is not yet covered by a finalized KHR extension on
			// the desktop platforms we target; gated out at the cap-query
			// level rather than emitting an undefined profile chain here.
			break;
		case oa::VideoCodec::VP9:
			// Vulkan exposes VP9 decode, not a matching KHR encode operation.
			break;
	}
	profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	profile.lumaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	profile.chromaBitDepth    = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	return profile;
}


// Attaches encode-side capability structs onto the VkVideoCapabilitiesKHR
// pNext chain so vkGetPhysicalDeviceVideoCapabilitiesKHR populates them.
void attachEncodeCapabilityStructs(
	oa::VideoCodec inCodec,
	VkVideoCapabilitiesKHR& inOutCaps,
	VkVideoEncodeCapabilitiesKHR& outEncode,
	VkVideoEncodeH264CapabilitiesKHR& outH264,
	VkVideoEncodeH265CapabilitiesKHR& outH265)
{
	outEncode.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
	inOutCaps.pNext = &outEncode;
	switch (inCodec) {
		case oa::VideoCodec::H264:
			outH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR;
			outEncode.pNext = &outH264;
			break;
		case oa::VideoCodec::H265:
			outH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR;
			outEncode.pNext = &outH265;
			break;
		case oa::VideoCodec::AV1:
		case oa::VideoCodec::VP9:
			break;
	}
}


bool hasFormatWithUsage(
	const oa::Vector<VkVideoFormatPropertiesKHR>& inFormats,
	VkFormat inFormat,
	VkImageUsageFlags inUsage)
{
	for (const auto& format : inFormats) {
		if (format.format == inFormat && (format.imageUsageFlags & inUsage) == inUsage) {
			return true;
		}
	}
	return false;
}


// Manufactures a minimal valid H.264 SPS for an encode session.
// The encoder side WRITES this (vs the decoder which parses it from the
// bitstream); we only fill the fields a baseline encoder needs.
//
// level is picked conservatively at 4.2 (covers 1080p60) — the cap query
// has already bounded width/height so the level is always over-provisioned
// rather than under-provisioned. Refine when we wire user-facing level
// selection.
StdVideoH264SequenceParameterSet buildSpsForH264Encode(const oa::VideoEncodeProfile& inProfile)
{
	StdVideoH264SequenceParameterSet sps = {};
	sps.flags.constraint_set0_flag           = 0;
	sps.flags.constraint_set1_flag           = 0;
	sps.flags.constraint_set2_flag           = 0;
	sps.flags.constraint_set3_flag           = 0;
	sps.flags.constraint_set4_flag           = 0;
	sps.flags.constraint_set5_flag           = 0;
	sps.flags.direct_8x8_inference_flag      = 1;
	sps.flags.mb_adaptive_frame_field_flag   = 0;
	sps.flags.frame_mbs_only_flag            = 1;
	sps.flags.delta_pic_order_always_zero_flag = 0;
	sps.flags.separate_colour_plane_flag     = 0;
	sps.flags.gaps_in_frame_num_value_allowed_flag = 0;
	sps.flags.qpprime_y_zero_transform_bypass_flag = 0;
	sps.flags.frame_cropping_flag            = 0;
	sps.profile_idc                          = STD_VIDEO_H264_PROFILE_IDC_HIGH;
	sps.level_idc                            = STD_VIDEO_H264_LEVEL_IDC_4_2;
	sps.chroma_format_idc                    = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
	sps.seq_parameter_set_id                 = 0;
	sps.bit_depth_luma_minus8                = 0;
	sps.bit_depth_chroma_minus8              = 0;
	sps.log2_max_frame_num_minus4            = 4;   // frame_num cycles every 256
	sps.pic_order_cnt_type                   = STD_VIDEO_H264_POC_TYPE_0;
	sps.offset_for_non_ref_pic               = 0;
	sps.offset_for_top_to_bottom_field       = 0;
	sps.log2_max_pic_order_cnt_lsb_minus4    = 4;
	sps.num_ref_frames_in_pic_order_cnt_cycle = 0;
	sps.max_num_ref_frames                   = 1;
	const oa::U32 widthMbs                     = (inProfile.width  + 15U) / 16U;
	const oa::U32 heightMbs                    = (inProfile.height + 15U) / 16U;
	sps.pic_width_in_mbs_minus1              = widthMbs  > 0U ? widthMbs  - 1U : 0U;
	sps.pic_height_in_map_units_minus1       = heightMbs > 0U ? heightMbs - 1U : 0U;
	sps.frame_crop_left_offset               = 0;
	sps.frame_crop_right_offset              = 0;
	sps.frame_crop_top_offset                = 0;
	sps.frame_crop_bottom_offset             = 0;
	sps.pOffsetForRefFrame                   = nullptr;
	return sps;
}


// Manufactures a minimal valid H.264 PPS for an encode session.
// CABAC + QP 26 baseline; deblocking filter control present so we can
// emit deblocking filter idc per slice header later.
StdVideoH264PictureParameterSet buildPpsForH264Encode(const oa::VideoEncodeProfile&)
{
	StdVideoH264PictureParameterSet pps = {};
	pps.flags.transform_8x8_mode_flag                          = 1;
	pps.flags.redundant_pic_cnt_present_flag                   = 0;
	pps.flags.constrained_intra_pred_flag                      = 0;
	pps.flags.deblocking_filter_control_present_flag           = 1;
	pps.flags.weighted_pred_flag                               = 0;
	pps.flags.bottom_field_pic_order_in_frame_present_flag     = 0;
	pps.flags.entropy_coding_mode_flag                         = 1;   // CABAC
	pps.seq_parameter_set_id                  = 0;
	pps.pic_parameter_set_id                  = 0;
	pps.num_ref_idx_l0_default_active_minus1  = 0;
	pps.num_ref_idx_l1_default_active_minus1  = 0;
	pps.weighted_bipred_idc                   = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT;
	pps.pic_init_qp_minus26                   = 0;
	pps.pic_init_qs_minus26                   = 0;
	pps.chroma_qp_index_offset                = 0;
	pps.second_chroma_qp_index_offset         = 0;
	return pps;
}


// storage for the pointer-bearing HEVC standard structures. vulkan consumes
// these during vkCreateVideoSessionParametersKHR; keeping them together makes
// the ownership/lifetime relationship explicit and prevents dangling pNext-
// style payload pointers while the create call is assembled.
struct H265EncodeParameters {
	StdVideoH265ProfileTierLevel profileTierLevel = {};
	StdVideoH265DecPicBufMgr dpbManager = {};
	StdVideoH265ShortTermRefPicSet shortTermRefPicSet = {};
	StdVideoH265VideoParameterSet vps = {};
	StdVideoH265SequenceParameterSet sps = {};
	StdVideoH265PictureParameterSet pps = {};
};


H265EncodeParameters buildParametersForH265Encode(
	const oa::VideoEncodeProfile& inProfile,
	oa::U32 inCodedWidth,
	oa::U32 inCodedHeight,
	oa::U32 inDpbSlots,
	VkVideoEncodeH265CtbSizeFlagsKHR inCtbSizes,
	VkVideoEncodeH265TransformBlockSizeFlagsKHR inTransformBlockSizes)
{
	H265EncodeParameters out;
	const oa::U32 ctbLog2 = (inCtbSizes & VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_KHR) != 0U
		? 4U : ((inCtbSizes & VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_KHR) != 0U ? 5U : 6U);
	const oa::U32 minTransformLog2 =
		(inTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_KHR) != 0U
			? 2U
			: ((inTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_KHR) != 0U
				? 3U : 4U);
	oa::U32 maxTransformLog2 = minTransformLog2;
	if (ctbLog2 >= 5U
		and (inTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_KHR) != 0U) {
		maxTransformLog2 = 5U;
	} else if ((inTransformBlockSizes
		& VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_KHR) != 0U) {
		maxTransformLog2 = 4U;
	} else if ((inTransformBlockSizes
		& VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_KHR) != 0U) {
		maxTransformLog2 = 3U;
	}

	out.profileTierLevel.flags.general_progressive_source_flag = 1;
	out.profileTierLevel.flags.general_frame_only_constraint_flag = 1;
	out.profileTierLevel.general_profile_idc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
	// main-tier level 4.1 covers the presentation and capture profiles OA
	// currently exposes (up to 1080p60 / 4K30 at practical bitrates).
	out.profileTierLevel.general_level_idc = STD_VIDEO_H265_LEVEL_IDC_4_1;

	out.dpbManager.max_dec_pic_buffering_minus1[0] = static_cast<oa::U8>(
		inDpbSlots > 0U ? inDpbSlots - 1U : 0U);
	out.dpbManager.max_num_reorder_pics[0] = 0;
	out.dpbManager.max_latency_increase_plus1[0] = 0;

	// One flat L0 reference: the immediately preceding I/P picture.
	out.shortTermRefPicSet.num_negative_pics = 1;
	out.shortTermRefPicSet.num_positive_pics = 0;
	out.shortTermRefPicSet.used_by_curr_pic_s0_flag = 1;
	out.shortTermRefPicSet.delta_poc_s0_minus1[0] = 0;

	out.vps.flags.vps_temporal_id_nesting_flag = 1;
	out.vps.flags.vps_sub_layer_ordering_info_present_flag = 1;
	out.vps.vps_video_parameter_set_id = 0;
	out.vps.vps_max_sub_layers_minus1 = 0;
	out.vps.pDecPicBufMgr = &out.dpbManager;
	out.vps.pProfileTierLevel = &out.profileTierLevel;

	out.sps.flags.sps_temporal_id_nesting_flag = 1;
	out.sps.flags.sps_sub_layer_ordering_info_present_flag = 1;
	out.sps.flags.amp_enabled_flag = 1;
	out.sps.flags.sample_adaptive_offset_enabled_flag = 1;
	out.sps.flags.conformance_window_flag =
		inCodedWidth != inProfile.width || inCodedHeight != inProfile.height;
	out.sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
	out.sps.pic_width_in_luma_samples = inCodedWidth;
	out.sps.pic_height_in_luma_samples = inCodedHeight;
	out.sps.sps_video_parameter_set_id = 0;
	out.sps.sps_max_sub_layers_minus1 = 0;
	out.sps.sps_seq_parameter_set_id = 0;
	out.sps.bit_depth_luma_minus8 = 0;
	out.sps.bit_depth_chroma_minus8 = 0;
	out.sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	// MinCb stays 16 while CTB/transform geometry follows the device's HEVC
	// capability masks. Intel TGL, for example, exposes CTB 32/64 but not 16;
	// hard-coding 16 produces a valid-looking command that hangs the engine.
	constexpr oa::U32 minCodingBlockLog2 = 4U;
	out.sps.log2_min_luma_coding_block_size_minus3 =
		static_cast<oa::U8>(minCodingBlockLog2 - 3U);
	out.sps.log2_diff_max_min_luma_coding_block_size =
		static_cast<oa::U8>(ctbLog2 - minCodingBlockLog2);
	out.sps.log2_min_luma_transform_block_size_minus2 =
		static_cast<oa::U8>(minTransformLog2 - 2U);
	out.sps.log2_diff_max_min_luma_transform_block_size =
		static_cast<oa::U8>(maxTransformLog2 - minTransformLog2);
	out.sps.max_transform_hierarchy_depth_inter = static_cast<oa::U8>(
		ctbLog2 - minTransformLog2 > 1U ? ctbLog2 - minTransformLog2 : 1U);
	out.sps.max_transform_hierarchy_depth_intra = 3U;
	out.sps.pcm_sample_bit_depth_luma_minus1 = 7U;
	out.sps.pcm_sample_bit_depth_chroma_minus1 = 7U;
	out.sps.log2_min_pcm_luma_coding_block_size_minus3 =
		static_cast<oa::U8>(minCodingBlockLog2 - 3U);
	out.sps.log2_diff_max_min_pcm_luma_coding_block_size =
		static_cast<oa::U8>(ctbLog2 - minCodingBlockLog2);
	out.sps.num_short_term_ref_pic_sets = 1;
	out.sps.conf_win_right_offset = (inCodedWidth - inProfile.width) / 2U;
	out.sps.conf_win_bottom_offset = (inCodedHeight - inProfile.height) / 2U;
	out.sps.pProfileTierLevel = &out.profileTierLevel;
	out.sps.pDecPicBufMgr = &out.dpbManager;
	out.sps.pShortTermRefPicSet = &out.shortTermRefPicSet;

	out.pps.flags.cabac_init_present_flag = 1;
	out.pps.flags.transform_skip_enabled_flag = 1;
	out.pps.flags.cu_qp_delta_enabled_flag = 1;
	out.pps.flags.pps_loop_filter_across_slices_enabled_flag = 1;
	out.pps.flags.deblocking_filter_control_present_flag = 1;
	out.pps.pps_pic_parameter_set_id = 0;
	out.pps.pps_seq_parameter_set_id = 0;
	out.pps.sps_video_parameter_set_id = 0;
	out.pps.num_ref_idx_l0_default_active_minus1 = 0;
	out.pps.num_ref_idx_l1_default_active_minus1 = 0;
	out.pps.init_qp_minus26 = 0;
	out.pps.log2_parallel_merge_level_minus2 = 0;
	return out;
}


oa::Status queryVideoFormats(
	const VklInstanceTable& inDispatch,
	VkPhysicalDevice inPhys,
	const VkVideoProfileInfoKHR& inProfile,
	VkImageUsageFlags inUsage,
	oa::Vector<VkVideoFormatPropertiesKHR>& outFormats)
{
	if (inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR == nullptr) {
		return oa::Status::error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType        = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles    = &inProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext      = &profileList;
	formatInfo.imageUsage = inUsage;

	oa::U32 formatCount = 0;
	VkResult result = inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR(
		inPhys, &formatInfo, &formatCount, nullptr);
	if (result != VK_SUCCESS) {
		return oa::Status::error("Failed to query vulkan Video encode format count");
	}
	outFormats.resize(formatCount);
	for (auto& format : outFormats) {
		format = {};
		format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
	}
	if (formatCount == 0) {
		return oa::Status::ok();
	}
	result = inDispatch.vkGetPhysicalDeviceVideoFormatPropertiesKHR(
		inPhys, &formatInfo, &formatCount, outFormats.data());
	if (result != VK_SUCCESS) {
		outFormats.resize(0);
		return oa::Status::error("Failed to query vulkan Video encode formats");
	}
	outFormats.resize(formatCount);
	return oa::Status::ok();
}

}  // namespace


// ──────────────────────────────────────────────────────────────────────
//                       Capability querying
// ──────────────────────────────────────────────────────────────────────

oa::Result<oa::VideoEncodeCapabilities> oa::VideoEncoder::queryEncodeCapabilities(oa::Engine& inRt,	oa::VideoCodec inCodec) {
	if (inCodec != oa::VideoCodec::H264 and inCodec != oa::VideoCodec::H265) {
		return oa::Status::error(oa::StatusCode::Unimplemented,	"Requested vulkan Video encode codec is not implemented in OA");
	}
	auto& vkEngine = inRt;
	const auto& deviceState = oa::EngineDeviceAccess::get(vkEngine);
	const auto& instanceDispatch = deviceState.instanceDispatch;
	const auto& hw = deviceState.info.hardware;
	const auto& sw = deviceState.info.software;
	// Mesa 26.1.4, 26.1.5, and 26.1.7 advertise HEVC encode on Tiger Lake GT2,
	// but OA loses the device on all three. Khronos vulkan-Video-samples independently
	// reproduces the 26.1.4 failure. Never submit either evidenced-broken path;
	// keep the list version-specific so a later Mesa release is retested.
	const bool brokenTglMesaVersion =
		sw.driverInfo.find("Mesa 26.1.4") != oa::String::Npos
		or sw.driverInfo.find("Mesa 26.1.5") != oa::String::Npos
		or sw.driverInfo.find("Mesa 26.1.7") != oa::String::Npos;
	const bool brokenTglH265Encode = inCodec == oa::VideoCodec::H265
		and hw.vendorId == oavk::VendorIdIntel
		and hw.deviceId == 0x9A49U
		and sw.driverId == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA_KHR
		and brokenTglMesaVersion;
	if (brokenTglH265Encode) {
		return oa::Status::error(oa::StatusCode::Unavailable,	"H.265 encode is disabled on Intel TGL GT2 with Mesa 26.1.4/26.1.5/26.1.7: all three evidenced driver versions lose the device");
	}

	if (not sw.hasVideoQueue or not sw.hasVideoEncodeQueue or not oa::EngineDeviceAccess::get(vkEngine).queues.hasVideoEncodeQueue) {
		return oa::Status::error(oa::StatusCode::Unavailable,	"Selected vulkan device does not expose a video encode queue");
	}
	if (inCodec == oa::VideoCodec::H264 and not sw.hasVideoEncodeH264) {
		return oa::Status::error(oa::StatusCode::Unavailable,	"VK_KHR_video_encode_h264 is not enabled");
	}
	if (inCodec == oa::VideoCodec::H265 and not sw.hasVideoEncodeH265) {
		return oa::Status::error(oa::StatusCode::Unavailable,	"VK_KHR_video_encode_h265 is not enabled");
	}
	if (inCodec == oa::VideoCodec::AV1) {
		return oa::Status::error(oa::StatusCode::Unavailable,	"AV1 video encode is not supported in this build");
	}
	if (instanceDispatch.vkGetPhysicalDeviceVideoCapabilitiesKHR == nullptr) {
		return oa::Status::error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(vkEngine).physicalDevice);
	VkVideoEncodeH264ProfileInfoKHR h264 = {};
	VkVideoEncodeH265ProfileInfoKHR h265 = {};
	VkVideoProfileInfoKHR profile = buildEncodeProfileInfo(inCodec, h264, h265);

	VkVideoCapabilitiesKHR caps           = {};
	caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	VkVideoEncodeCapabilitiesKHR encCaps  = {};
	VkVideoEncodeH264CapabilitiesKHR h264Caps = {};
	VkVideoEncodeH265CapabilitiesKHR h265Caps = {};
	attachEncodeCapabilityStructs(inCodec, caps, encCaps, h264Caps, h265Caps);

	VkResult result = instanceDispatch.vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &profile, &caps);
	if (result != VK_SUCCESS) {
		return oa::Status::error("vulkan Video encode profile is not supported");
	}

	oa::VideoEncodeCapabilities out;
	out.supported                          = true;
	out.maxWidth                           = caps.maxCodedExtent.width;
	out.maxHeight                          = caps.maxCodedExtent.height;
	out.minWidth                           = caps.minCodedExtent.width;
	out.minHeight                          = caps.minCodedExtent.height;
	out.pictureAccessGranularityWidth      = caps.pictureAccessGranularity.width  == 0U ? 1U : caps.pictureAccessGranularity.width;
	out.pictureAccessGranularityHeight     = caps.pictureAccessGranularity.height == 0U ? 1U : caps.pictureAccessGranularity.height;
	out.maxDpbSlots                        = caps.maxDpbSlots;
	out.maxActiveReferencePictures         = caps.maxActiveReferencePictures;
	out.minBitstreamBufferOffsetAlignment  = caps.minBitstreamBufferOffsetAlignment;
	out.minBitstreamBufferSizeAlignment    = caps.minBitstreamBufferSizeAlignment;
	out.stdHeaderVersion                   = caps.stdHeaderVersion;
	out.encodeFlags                        = encCaps.flags;
	out.rateControlModes                   = encCaps.rateControlModes;
	out.maxQualityLevels                   = encCaps.maxQualityLevels;
	out.maxBitrate                         = encCaps.maxBitrate > oa::U64{0xFFFFFFFFULL} ? oa::U32{0xFFFFFFFFU} : static_cast<oa::U32>(encCaps.maxBitrate);

	if (inCodec == oa::VideoCodec::H264) {
		out.maxH264SliceCount                = h264Caps.maxSliceCount;
		out.maxH264PPictureL0ReferenceCount  = h264Caps.maxPPictureL0ReferenceCount;
		out.maxH264BPictureL0ReferenceCount  = h264Caps.maxBPictureL0ReferenceCount;
		out.maxH264L1ReferenceCount          = h264Caps.maxL1ReferenceCount;
	} else if (inCodec == oa::VideoCodec::H265) {
		out.maxH265SliceSegmentCount         = h265Caps.maxSliceSegmentCount;
		out.maxH265PPictureL0ReferenceCount  = h265Caps.maxPPictureL0ReferenceCount;
		out.maxH265BPictureL0ReferenceCount  = h265Caps.maxBPictureL0ReferenceCount;
		out.maxH265L1ReferenceCount          = h265Caps.maxL1ReferenceCount;
		out.h265CtbSizes                     = h265Caps.ctbSizes;
		out.h265TransformBlockSizes          = h265Caps.transformBlockSizes;
		out.h265StdSyntaxFlags               = h265Caps.stdSyntaxFlags;
		out.minH265Qp                        = h265Caps.minQp;
		out.maxH265Qp                        = h265Caps.maxQp;
	}

	// input-image format support — encoder consumes NV12 (the standard
	// 4:2:0 planar YCbCr format) and writes the DPB in the same format.
	const VkImageUsageFlags inputUsage = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
	const VkImageUsageFlags dpbUsage   = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR;
	oa::Status inputStatus = queryVideoFormats(
		instanceDispatch, phys, profile, inputUsage, out.inputFormats);
	if (not inputStatus.isOk()) { return inputStatus; }
	oa::Status dpbStatus = queryVideoFormats(
		instanceDispatch, phys, profile, dpbUsage, out.dpbFormats);
	if (not dpbStatus.isOk()) { return dpbStatus; }

	if (not hasFormatWithUsage(out.inputFormats, out.pictureFormat, inputUsage)) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"vulkan Video encoder does not expose NV12 input format support");
	}
	if (not hasFormatWithUsage(out.dpbFormats, out.referencePictureFormat, dpbUsage)) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"vulkan Video encoder does not expose NV12 DPB format support");
	}

	return out;
}


// ──────────────────────────────────────────────────────────────────────
//                       lifecycle (ctor/dtor/move)
// ──────────────────────────────────────────────────────────────────────

oa::VideoEncoder::VideoEncoder()
	: impl_(oa::makeUnique<Impl>()) {}


void oa::VideoEncoder::reset_() noexcept
{
	if (not impl_) return;
	impl_->session = {};
	impl_->sessionParameters = {};
	impl_->queue = {};
	impl_->dpb = {};
	impl_->dpbSlotCapacity = 0;
	impl_->slots.clear();
	impl_->submitSlot = 0U;
	impl_->harvestSlot = 0U;
	impl_->pendingSlots = 0U;
	impl_->compatibilityUploadReady = false;
	impl_->cachedHeaders.clear();
	impl_->rateControlReset = false;
	impl_->queryResultStatusSupported = false;
	impl_->zeroFeedbackRecoveryCount = 0U;
	impl_->rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
	impl_->codedWidth = 0;
	impl_->codedHeight = 0;
	impl_->minBitstreamBufferOffsetAlignment = 1;
	impl_->minBitstreamBufferSizeAlignment = 1;
	impl_->frameCount = 0;
	impl_->lastKeyframeIndex = 0;
	impl_->gopSize = 30;
	impl_->currentGopFrame = 0;
	impl_->engine = nullptr;
}


oa::VideoEncoder::VideoEncoder(oa::VideoEncoder&& inOther) noexcept = default;


oa::VideoEncoder& oa::VideoEncoder::operator=(oa::VideoEncoder&& inOther) noexcept
{
	if (this != &inOther) {
		abandon_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}


oa::VideoEncoder::~VideoEncoder()
{
	abandon_();
}


void oa::VideoEncoder::abandon_() noexcept
{
	if (not impl_ or impl_->engine == nullptr) return;
	oa::Engine* engine = impl_->engine;
	auto retired = oa::makeUnique<oa::VideoEncoder>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::VideoEncoder::completeRetired_,
		&oa::VideoEncoder::releaseRetired_);
}


oa::Engine* oa::VideoEncoder::getEngine() const noexcept
{
	return impl_ ? impl_->engine : nullptr;
}


const oa::VideoEncodeProfile& oa::VideoEncoder::getProfile() const noexcept
{
	static const oa::VideoEncodeProfile Empty = {};
	return impl_ ? impl_->profile : Empty;
}


oa::U32 oa::VideoEncoder::getCodedWidth() const noexcept
{
	return impl_ ? impl_->codedWidth : 0U;
}


oa::U32 oa::VideoEncoder::getCodedHeight() const noexcept
{
	return impl_ ? impl_->codedHeight : 0U;
}


oa::Status oa::VideoEncoder::completeRetired_(void* inPayload)
{
	auto* encoder = static_cast<oa::VideoEncoder*>(inPayload);
	return encoder ? encoder->close() : oa::Status::ok();
}


void oa::VideoEncoder::releaseRetired_(void* inPayload)
{
	oa::UniquePtr<oa::VideoEncoder> encoder(
		static_cast<oa::VideoEncoder*>(inPayload));
}


// ──────────────────────────────────────────────────────────────────────
//                       Create / encodeFrame / flush / Close
// ──────────────────────────────────────────────────────────────────────

oa::Result<oa::VideoEncoder> oa::VideoEncoder::create(
	oa::Engine& inRt,
	const oa::VideoEncodeProfile& inProfile)
{
	oa::VideoEncoder encoder;
	encoder.impl_->engine      = &inRt;
	encoder.impl_->profile = inProfile;

	auto capsResult = queryEncodeCapabilities(inRt, inProfile.codec);
	if (not capsResult.isOk()) {
		return capsResult.getStatus();
	}
	const oa::VideoEncodeCapabilities& caps = *capsResult;
	if (inProfile.codec == oa::VideoCodec::H265) {
		OaLogInfo(oa::LogComponent::Video,
			"H.265 encoder caps: CTB=0x%x transform=0x%x syntax=0x%x P-L0=%u DPB=%u activeRefs=%u",
			static_cast<unsigned>(caps.h265CtbSizes),
			static_cast<unsigned>(caps.h265TransformBlockSizes),
			static_cast<unsigned>(caps.h265StdSyntaxFlags),
			static_cast<unsigned>(caps.maxH265PPictureL0ReferenceCount),
			static_cast<unsigned>(caps.maxDpbSlots),
			static_cast<unsigned>(caps.maxActiveReferencePictures));
	}

	if (inProfile.width  == 0U or inProfile.height == 0U or
	    inProfile.width  >  caps.maxWidth or
	    inProfile.height >  caps.maxHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video encode extent is unsupported");
	}
	if (inProfile.frameRate == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video encode frame rate must be > 0");
	}
	if (inProfile.gopSize == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video encode GOP size must be > 0");
	}
	if (inProfile.constantQp > 51U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"H.264/H.265 constant QP must be in the range 0..51");
	}
	if (inProfile.codec == oa::VideoCodec::H265
		and (static_cast<oa::I32>(inProfile.constantQp) < caps.minH265Qp
			or static_cast<oa::I32>(inProfile.constantQp) > caps.maxH265Qp)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"H.265 constant QP is outside the device-supported range");
	}
	if (inProfile.qualityLevel >= caps.maxQualityLevels) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Video encode quality level exceeds device capabilities");
	}
	if (inProfile.rateControl != oa::VideoRateControl::ConstantQp and inProfile.bitrate == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"CBR/VBR video encode requires a non-zero target bitrate");
	}
	if (caps.maxBitrate > 0U and inProfile.bitrate > caps.maxBitrate) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"target video bitrate exceeds device capabilities");
	}
	if (inProfile.maxBitrate > 0U and inProfile.maxBitrate < inProfile.bitrate) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"VBR maximum bitrate must be greater than or equal to target bitrate");
	}
	if (caps.maxBitrate > 0U and inProfile.maxBitrate > caps.maxBitrate) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"maximum video bitrate exceeds device capabilities");
	}

	encoder.impl_->minBitstreamBufferOffsetAlignment =
		caps.minBitstreamBufferOffsetAlignment == 0U ? 1U : caps.minBitstreamBufferOffsetAlignment;
	encoder.impl_->minBitstreamBufferSizeAlignment   =
		caps.minBitstreamBufferSizeAlignment   == 0U ? 1U : caps.minBitstreamBufferSizeAlignment;

	switch (inProfile.rateControl) {
		case oa::VideoRateControl::ConstantQp:
			encoder.impl_->rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
			break;
		case oa::VideoRateControl::Cbr:
			encoder.impl_->rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;
			break;
		case oa::VideoRateControl::Vbr:
			encoder.impl_->rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR;
			break;
	}
	if (encoder.impl_->rateControlMode != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
		and (caps.rateControlModes & encoder.impl_->rateControlMode) == 0U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"Requested video rate-control mode is not supported by this device");
	}
	if (inProfile.rateControl == oa::VideoRateControl::Cbr) {
		encoder.impl_->profile.maxBitrate = inProfile.bitrate;
	} else if (inProfile.rateControl == oa::VideoRateControl::Vbr and inProfile.maxBitrate == 0U) {
		oa::U64 resolvedMaximum = static_cast<oa::U64>(inProfile.bitrate) * 2ULL;
		if (caps.maxBitrate > 0U and resolvedMaximum > caps.maxBitrate) {
			resolvedMaximum = caps.maxBitrate;
		}
		encoder.impl_->profile.maxBitrate = static_cast<oa::U32>(resolvedMaximum);
	}

	// Align coded extent up to picture-access granularity + 16-MB granularity
	// (H.264 macroblock = 16x16). Same dance as VideoDecoder::Create — the
	// driver fails session create if coded extent isn't aligned.
	const oa::U32 widthGranularity  = caps.pictureAccessGranularityWidth  > 16U
		? caps.pictureAccessGranularityWidth  : 16U;
	const oa::U32 heightGranularity = caps.pictureAccessGranularityHeight > 16U
		? caps.pictureAccessGranularityHeight : 16U;
	const oa::U32 alignedWidth      = ((inProfile.width  + widthGranularity  - 1U) / widthGranularity)  * widthGranularity;
	const oa::U32 alignedHeight     = ((inProfile.height + heightGranularity - 1U) / heightGranularity) * heightGranularity;
	if (alignedWidth > caps.maxWidth or alignedHeight > caps.maxHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"Aligned video encode extent exceeds caps");
	}
	encoder.impl_->codedWidth   = alignedWidth;
	encoder.impl_->codedHeight  = alignedHeight;

	auto& vkEngine = inRt;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);

	VkVideoEncodeH264ProfileInfoKHR h264 = {};
	VkVideoEncodeH265ProfileInfoKHR h265 = {};
	VkVideoProfileInfoKHR profile = buildEncodeProfileInfo(inProfile.codec, h264, h265);

	// ── 1. Create video session using oavk::VideoSession ─────────────────
	VkExtent2D codedExtent = { encoder.impl_->codedWidth, encoder.impl_->codedHeight };

	// IPP cadence: 1 reference frame (the previous P/I) is enough for the
	// minimum useful encoder. Bump if/when we expose multi-ref or B-frames.
	const oa::U32 requestedDpbSlots = inProfile.maxDpbSlots == 0U ? 2U : inProfile.maxDpbSlots;
	const oa::U32 maxDpbSlots = requestedDpbSlots < caps.maxDpbSlots
		? requestedDpbSlots : caps.maxDpbSlots;
	if (maxDpbSlots == 0U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"vulkan Video encoder reports zero DPB slots");
	}
	if (inProfile.gopSize > 1U and maxDpbSlots < 2U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"P-frame encoding requires at least two DPB slots");
	}
	const oa::U32 maxActiveReferences = maxDpbSlots < caps.maxActiveReferencePictures
		? maxDpbSlots : caps.maxActiveReferencePictures;
	const oa::U32 finalMaxActiveReferences = maxActiveReferences == 0U and caps.maxActiveReferencePictures > 0U ? 1U : maxActiveReferences;

	auto sessionResult = oavk::VideoSession::create(
		vkEngine,
		profile,
		codedExtent,
		caps.pictureFormat,
		caps.referencePictureFormat,
		maxDpbSlots,
		finalMaxActiveReferences,
		inProfile.qualityLevel);
	if (!sessionResult.isOk()) {
		return sessionResult.getStatus();
	}
	encoder.impl_->session = oa::move(sessionResult.getValue());
	encoder.impl_->dpbSlotCapacity = maxDpbSlots;

	// ── 2. Create video queue ───────────────────────────────────────────
	auto queueResult = oavk::VideoQueue::create(vkEngine, oavk::VideoQueue::QueueType::Encode);
	if (!queueResult.isOk()) {
		return queueResult.getStatus();
	}
	encoder.impl_->queue = oa::move(queueResult.getValue());

	// ── 3. Create shared DPB using oavk::VideoDpb ─────────────────────────
	// The encoder never samples its DPB. Requesting SAMPLED here makes the
	// profile/format combination invalid on drivers that expose encode-DPB but
	// not YCbCr sampling for the same video profile.
	const VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR;

	oavk::VideoDpb::CreateInfo dpbInfo = {};
	dpbInfo.profile = profile;
	dpbInfo.format = caps.referencePictureFormat;
	dpbInfo.codedExtent = codedExtent;
	dpbInfo.maxDpbSlots = maxDpbSlots;
	dpbInfo.usage = dpbUsage;

	auto dpbResult = oavk::VideoDpb::create(vkEngine, dpbInfo);
	if (!dpbResult.isOk()) {
		return dpbResult.getStatus();
	}
	encoder.impl_->dpb = oa::move(dpbResult.getValue());

	// ── 4. Create codec session parameters ─────────────────────────────
	StdVideoH264SequenceParameterSet h264Sps = buildSpsForH264Encode(inProfile);
	StdVideoH264PictureParameterSet  h264Pps = buildPpsForH264Encode(inProfile);
	VkVideoEncodeH264SessionParametersAddInfoKHR h264AddInfo = {};
	h264AddInfo.sType        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
	h264AddInfo.stdSPSCount  = 1;
	h264AddInfo.pStdSPSs     = &h264Sps;
	h264AddInfo.stdPPSCount  = 1;
	h264AddInfo.pStdPPSs     = &h264Pps;
	VkVideoEncodeH264SessionParametersCreateInfoKHR h264ParamsCreate = {};
	h264ParamsCreate.sType                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR;
	h264ParamsCreate.maxStdSPSCount       = 1;
	h264ParamsCreate.maxStdPPSCount       = 1;
	h264ParamsCreate.pParametersAddInfo   = &h264AddInfo;

	H265EncodeParameters h265Parameters = buildParametersForH265Encode(
		inProfile, encoder.impl_->codedWidth, encoder.impl_->codedHeight, maxDpbSlots,
		caps.h265CtbSizes, caps.h265TransformBlockSizes);
	VkVideoEncodeH265SessionParametersAddInfoKHR h265AddInfo = {};
	h265AddInfo.sType       = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	h265AddInfo.stdVPSCount = 1;
	h265AddInfo.pStdVPSs    = &h265Parameters.vps;
	h265AddInfo.stdSPSCount = 1;
	h265AddInfo.pStdSPSs    = &h265Parameters.sps;
	h265AddInfo.stdPPSCount = 1;
	h265AddInfo.pStdPPSs    = &h265Parameters.pps;
	VkVideoEncodeH265SessionParametersCreateInfoKHR h265ParamsCreate = {};
	h265ParamsCreate.sType              = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR;
	h265ParamsCreate.maxStdVPSCount     = 1;
	h265ParamsCreate.maxStdSPSCount     = 1;
	h265ParamsCreate.maxStdPPSCount     = 1;
	h265ParamsCreate.pParametersAddInfo = &h265AddInfo;
	VkVideoEncodeQualityLevelInfoKHR qualityInfo = {};
	qualityInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR;
	qualityInfo.qualityLevel = inProfile.qualityLevel;
	h264ParamsCreate.pNext = &qualityInfo;
	h265ParamsCreate.pNext = &qualityInfo;

	VkVideoSessionParametersCreateInfoKHR paramsCreate = {};
	paramsCreate.sType              = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
	paramsCreate.pNext              = inProfile.codec == oa::VideoCodec::H264
		? static_cast<const void*>(&h264ParamsCreate)
		: static_cast<const void*>(&h265ParamsCreate);
	paramsCreate.videoSession       = encoder.impl_->session.handle();
	paramsCreate.videoSessionParametersTemplate = VK_NULL_HANDLE;
	VkVideoSessionParametersKHR paramsHandle = VK_NULL_HANDLE;
	VkResult result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateVideoSessionParametersKHR(device, &paramsCreate, nullptr, &paramsHandle);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			inProfile.codec == oa::VideoCodec::H264
				? "vkCreateVideoSessionParametersKHR (encode H.264) failed"
				: "vkCreateVideoSessionParametersKHR (encode H.265) failed"),
			"video encoder");
	}
	encoder.impl_->sessionParameters.attach(vkEngine, paramsHandle);

	// ── 5. Per-job resources ───────────────────────────────────────────
	// Each slot owns the source image, command buffer, feedback query,
	// bitstream target, compute ticket and video fence. The video session and
	// DPB above stay shared; queue submission order preserves reference order.
	const oa::U32 slotCount = inProfile.asyncDepth == 0U ? 1U : inProfile.asyncDepth;
	encoder.impl_->slots.resize(slotCount);

	// NV12 input image (encoder source picture).
	// MUTABLE_FORMAT + EXTENDED_USAGE_BIT so we can create per-plane
	// R8 / R8G8 storage views into the multi-plane NV12 image — the
	// CvtRgbaToNv12 compute shader writes through those single-plane
	// views to fill Y and UV separately. The format list spells out
	// the three compatible formats (NV12 itself + the two plane
	// formats per vulkan plane-format compatibility table).
	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType        = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles    = &profile;

	const VkFormat kInputFormatList[3] = {
		caps.pictureFormat,            // VK_FORMAT_G8_B8R8_2PLANE_420_UNORM
		VK_FORMAT_R8_UNORM,            // plane 0 (Y)
		VK_FORMAT_R8G8_UNORM,          // plane 1 (UV interleaved)
	};
	VkImageFormatListCreateInfo formatList = {};
	formatList.sType           = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
	formatList.pNext           = &profileList;
	formatList.viewFormatCount = 3;
	formatList.pViewFormats    = kInputFormatList;

	VkImageCreateInfo inputInfo = {};
	inputInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	inputInfo.pNext         = &formatList;
	inputInfo.flags         = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
	                      | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
	inputInfo.imageType     = VK_IMAGE_TYPE_2D;
	inputInfo.format        = caps.pictureFormat;
	inputInfo.extent        = { encoder.impl_->codedWidth, encoder.impl_->codedHeight, 1U };
	inputInfo.mipLevels     = 1;
	inputInfo.arrayLayers   = 1;
	inputInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
	inputInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
	inputInfo.usage         = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR
	                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT
	                      | VK_IMAGE_USAGE_STORAGE_BIT;
	const oa::U32 queueFamilies[2] = {
		oa::EngineDeviceAccess::get(vkEngine).queues.computeQueueFamily,
		oa::EngineDeviceAccess::get(vkEngine).queues.videoEncodeQueueFamily,
	};
	const bool separateQueueFamilies = queueFamilies[0] != queueFamilies[1];
	inputInfo.sharingMode = separateQueueFamilies
		? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
	inputInfo.queueFamilyIndexCount = separateQueueFamilies ? 2U : 0U;
	inputInfo.pQueueFamilyIndices = separateQueueFamilies ? queueFamilies : nullptr;
	inputInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	vma::AllocationCreateInfo imageAllocInfo = {};
	imageAllocInfo.usage = vma::memoryUsageGpuOnly;
	for (auto& slot : encoder.impl_->slots) {
		const oa::U64 bitstreamSize = 4ULL * 1024ULL * 1024ULL;
		auto bitstreamResult = oavk::VideoBitstream::create(
			vkEngine, bitstreamSize, oavk::VideoBitstream::Direction::Encoder,
			caps.minBitstreamBufferOffsetAlignment == 0U ? 1U : caps.minBitstreamBufferOffsetAlignment,
			caps.minBitstreamBufferSizeAlignment == 0U ? 1U : caps.minBitstreamBufferSizeAlignment,
			&profile);
		if (not bitstreamResult.isOk()) {
			return closeSessionAfterCreateFailure(
				encoder, bitstreamResult.getStatus(), "video encoder");
		}
		slot.bitstream = oa::move(*bitstreamResult);
		oa::memset(slot.bitstream.getMappedPtr(), 0,
			static_cast<oa::Usize>(slot.bitstream.getCapacity()));
		result = vma::flushAllocation(
			static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
			static_cast<vma::Allocation>(slot.bitstream.getAllocation()),
			0, slot.bitstream.getCapacity());
		if (result != VK_SUCCESS) {
			return closeSessionAfterCreateFailure(encoder, oa::Status::error(
				oa::StatusCode::VulkanError,
				"Failed to initialize video encode bitstream buffer"),
				"video encoder");
		}

		auto commandResult = encoder.impl_->queue.allocateCommandBuffer();
		if (not commandResult.isOk()) {
			return closeSessionAfterCreateFailure(
				encoder, commandResult.getStatus(), "video encoder");
		}
		slot.commandBuffer = *commandResult;
		auto rgbaResult = oa::EngineResourceAccess::allocBuffer(vkEngine,
			static_cast<oa::U64>(encoder.impl_->codedWidth) * encoder.impl_->codedHeight * 4ULL);
		if (not rgbaResult.isOk()) {
			return closeSessionAfterCreateFailure(
				encoder, rgbaResult.getStatus(), "video encoder");
		}
		slot.rgbaSnapshot = oa::move(*rgbaResult);

	vma::Allocation inputAllocation = VK_NULL_HANDLE;
	result = vma::createImage(
		static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
		&inputInfo, &imageAllocInfo, &slot.inputImage, &inputAllocation, nullptr);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"Failed to create vulkan Video encode input image"),
			"video encoder");
	}
	slot.inputAllocation = inputAllocation;

	// Combined NV12 view — used by vkCmdEncodeVideoKHR.
	VkImageViewCreateInfo inputViewInfo = {};
	inputViewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	VkImageViewUsageCreateInfo inputVideoUsage = {};
	inputVideoUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	inputVideoUsage.usage = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
	inputViewInfo.pNext = &inputVideoUsage;
	inputViewInfo.image    = slot.inputImage;
	inputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	inputViewInfo.format   = caps.pictureFormat;
	inputViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	inputViewInfo.subresourceRange.baseMipLevel   = 0;
	inputViewInfo.subresourceRange.levelCount     = 1;
	inputViewInfo.subresourceRange.baseArrayLayer = 0;
	inputViewInfo.subresourceRange.layerCount     = 1;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateImageView(device, &inputViewInfo, nullptr, &slot.inputView);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"Failed to create vulkan Video encode input image view"),
			"video encoder");
	}

	// Y-plane view (R8_UNORM, aspect = PLANE_0) — storage-write target.
	VkImageViewUsageCreateInfo planeUsage = {};
	planeUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	planeUsage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
	VkImageViewCreateInfo yViewInfo = {};
	yViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	yViewInfo.pNext                           = &planeUsage;
	yViewInfo.image                           = slot.inputImage;
	yViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	yViewInfo.format                          = VK_FORMAT_R8_UNORM;
	yViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_PLANE_0_BIT;
	yViewInfo.subresourceRange.baseMipLevel   = 0;
	yViewInfo.subresourceRange.levelCount     = 1;
	yViewInfo.subresourceRange.baseArrayLayer = 0;
	yViewInfo.subresourceRange.layerCount     = 1;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateImageView(device, &yViewInfo, nullptr, &slot.inputYView);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"Failed to create vulkan Video encode Y-plane storage view"),
			"video encoder");
	}

	// UV-plane view (R8G8_UNORM, aspect = PLANE_1) — storage-write target.
	VkImageViewCreateInfo uvViewInfo = yViewInfo;
	uvViewInfo.format                      = VK_FORMAT_R8G8_UNORM;
	uvViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateImageView(device, &uvViewInfo, nullptr, &slot.inputUvView);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"Failed to create vulkan Video encode UV-plane storage view"),
			"video encoder");
	}

	// register both plane views in the bindless storage-image heap so
	// CvtRgbaToNv12 can address them via push-constant indices.
	slot.inputYBindless = oa::EngineBindlessAccess::get(vkEngine).registerStorageImage(
		oa::EngineDeviceAccess::get(vkEngine), slot.inputYView, VK_IMAGE_LAYOUT_GENERAL);
	slot.inputUvBindless = oa::EngineBindlessAccess::get(vkEngine).registerStorageImage(
		oa::EngineDeviceAccess::get(vkEngine), slot.inputUvView, VK_IMAGE_LAYOUT_GENERAL);
	slot.inputBindlessRegistered = true;

	// ── 8. fence + feedback query pool ──────────────────────────────
	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateFence(device, &fenceInfo, nullptr, &slot.fence);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"vkCreateFence for video encode failed"),
			"video encoder");
	}

	// Feedback query — returns bitstream start offset + bytes written
	// per encoded frame. Both fields are required by VK_KHR_video_encode_queue.
	VkQueryPoolVideoEncodeFeedbackCreateInfoKHR feedbackInfo = {};
	feedbackInfo.sType                = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR;
	feedbackInfo.pNext                = &profile;
	feedbackInfo.encodeFeedbackFlags  =
		  VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR
		| VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
	VkQueryPoolCreateInfo qpInfo = {};
	qpInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	qpInfo.pNext      = &feedbackInfo;
	qpInfo.queryType  = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR;
	qpInfo.queryCount = 1;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCreateQueryPool(device, &qpInfo, nullptr, &slot.queryPool);
	if (result != VK_SUCCESS) {
		return closeSessionAfterCreateFailure(encoder, oa::Status::error(
			oa::StatusCode::VulkanError,
			"vkCreateQueryPool for video encode feedback failed"),
			"video encoder");
	}
	}

	// ── 6. Extract encoded codec parameter NAL bytes ─────────────────
	// cache SPS/PPS for AVC or VPS/SPS/PPS for HEVC and prepend them to every
	// IDR. This gives both Annex-B output and the MP4 recorder a single,
	// self-describing keyframe contract.
	if (oa::EngineDeviceAccess::get(vkEngine).deviceDispatch
		.vkGetEncodedVideoSessionParametersKHR != nullptr) {
		VkVideoEncodeH264SessionParametersGetInfoKHR h264GetInfo = {};
		h264GetInfo.sType        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR;
		h264GetInfo.writeStdSPS  = VK_TRUE;
		h264GetInfo.writeStdPPS  = VK_TRUE;
		h264GetInfo.stdSPSId     = 0;
		h264GetInfo.stdPPSId     = 0;
		VkVideoEncodeH265SessionParametersGetInfoKHR h265GetInfo = {};
		h265GetInfo.sType        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR;
		h265GetInfo.writeStdVPS  = VK_TRUE;
		h265GetInfo.writeStdSPS  = VK_TRUE;
		h265GetInfo.writeStdPPS  = VK_TRUE;
		h265GetInfo.stdVPSId     = 0;
		h265GetInfo.stdSPSId     = 0;
		h265GetInfo.stdPPSId     = 0;
		VkVideoEncodeSessionParametersGetInfoKHR getInfo = {};
		getInfo.sType                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR;
		getInfo.pNext                   = inProfile.codec == oa::VideoCodec::H264
			? static_cast<const void*>(&h264GetInfo)
			: static_cast<const void*>(&h265GetInfo);
		getInfo.videoSessionParameters  = encoder.impl_->sessionParameters.handle();
		VkVideoEncodeSessionParametersFeedbackInfoKHR feedback = {};
		feedback.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR;
		oa::Usize headerSize = 0;
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkGetEncodedVideoSessionParametersKHR(
			device, &getInfo, &feedback, &headerSize, nullptr);
		if (result == VK_SUCCESS and headerSize > 0U) {
			encoder.impl_->cachedHeaders.resize(headerSize);
			result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkGetEncodedVideoSessionParametersKHR(
				device, &getInfo, &feedback, &headerSize,
				encoder.impl_->cachedHeaders.data());
			if (result != VK_SUCCESS) {
				encoder.impl_->cachedHeaders.clear();
			} else {
				encoder.impl_->cachedHeaders.resize(headerSize);
			}
		}
		// A driver that returns 0 bytes here (some Mesa/RADV builds) is
		// not fatal — encodeFrame just won't prepend headers. bitstream
		// will need a sidecar SPS+PPS to decode, which is fine for the
		// MP4-muxer follow-up.
	}

	encoder.impl_->frameCount        = 0;
	encoder.impl_->lastKeyframeIndex = 0;
	encoder.impl_->rateControlReset  = false;
	encoder.impl_->queryResultStatusSupported = false;
	encoder.impl_->zeroFeedbackRecoveryCount = 0U;
	const auto& instanceDispatch = oa::EngineDeviceAccess::get(vkEngine).instanceDispatch;
	if (instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties2 != nullptr) {
		oa::U32 familyCount = 0U;
		auto physicalDevice = static_cast<VkPhysicalDevice>(oa::EngineDeviceAccess::get(vkEngine).physicalDevice);
		instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties2(
			physicalDevice, &familyCount, nullptr);
		oa::Vector<VkQueueFamilyProperties2> familyProps(familyCount);
		oa::Vector<VkQueueFamilyQueryResultStatusPropertiesKHR> statusProps(familyCount);
		for (oa::U32 idx = 0U; idx < familyCount; ++idx) {
			statusProps[idx].sType =
				VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR;
			familyProps[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
			familyProps[idx].pNext = &statusProps[idx];
		}
		instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties2(
			physicalDevice, &familyCount, familyProps.data());
		const oa::U32 family = oa::EngineDeviceAccess::get(vkEngine).queues.videoEncodeQueueFamily;
		if (family < familyCount) {
			encoder.impl_->queryResultStatusSupported =
				statusProps[family].queryResultStatusSupport == VK_TRUE;
		}
	}
	encoder.impl_->currentGopFrame   = 0;
	encoder.impl_->gopSize           = inProfile.gopSize;
	return oa::Result<oa::VideoEncoder>(oa::move(encoder));
}


oa::Status oa::VideoEncoder::uploadInputRgba(
	const oavk::Buffer& inRgba,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::YCbCrModel inColorSpace,
	bool inFullRange)
{
	if (impl_->slots.empty()) {
		return oa::Status::error("oa::VideoEncoder::uploadInputRgba: encoder has no job slots");
	}
	if (impl_->pendingSlots != 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Cannot mix synchronous encode calls with pending asynchronous jobs");
	}
	OA_RETURN_IF_ERROR(uploadInputRgba_(
		impl_->slots[0], inRgba, inVisibleWidth, inVisibleHeight, inColorSpace, inFullRange));
	impl_->compatibilityUploadReady = true;
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::uploadInputRgba_(
	EncodeSlot& inSlot,
	const oavk::Buffer& inRgba,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::YCbCrModel inColorSpace,
	bool inFullRange)
{
	if (impl_->engine == nullptr or impl_->session.handle() == VK_NULL_HANDLE) {
		return oa::Status::error("oa::VideoEncoder::uploadInputRgba: encoder not initialized");
	}
	if (inRgba.buffer == VK_NULL_HANDLE or inRgba.bindlessIndex == OA_BINDLESS_INVALID) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::uploadInputRgba: source buffer is not bindless-registered");
	}
	if (inVisibleWidth == 0U or inVisibleHeight == 0U
	    or inVisibleWidth > impl_->codedWidth or inVisibleHeight > impl_->codedHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::uploadInputRgba: visible extent out of range");
	}
	const oa::U64 visibleBytes = static_cast<oa::U64>(inVisibleWidth) * inVisibleHeight * 4ULL;
	auto& vkEngine = *impl_->engine;
	if (inRgba.allocation == nullptr or inRgba.aliasIdentity != nullptr
		or inRgba.allocatorIdentity != oa::EngineAllocatorAccess::get(vkEngine).allocator) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"RGBA source must be a non-aliased allocation owned by the encoder engine");
	}
	if (inRgba.size < visibleBytes or inSlot.rgbaSnapshot.size < visibleBytes) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"RGBA source buffer is smaller than the visible frame");
	}
	oa::Event copyReady;
	if (inRgba.mappedPtr != nullptr and inSlot.rgbaSnapshot.mappedPtr != nullptr) {
		OA_RETURN_IF_ERROR(oa::EngineResourceAccess::readbackBuffer(vkEngine,
			inRgba, 0U, inSlot.rgbaSnapshot.mappedPtr, visibleBytes));
		if (not oa::EngineAllocatorAccess::get(vkEngine).flushHostBuffer(inSlot.rgbaSnapshot, 0U, visibleBytes)) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"Failed to flush asynchronous RGBA snapshot buffer");
		}
	} else {
		// Device-only producers need a stable copy before returning ownership to
		// the caller. This exact event wait is intentionally limited to that legacy
		// buffer path; mapped capture frames take the fully asynchronous route.
		auto copy = oa::EngineResourceAccess::copyBufferAsync(vkEngine,
			inRgba, inSlot.rgbaSnapshot, visibleBytes);
		if (not copy.isOk()) return copy.getStatus();
		OA_RETURN_IF_ERROR(copy->wait());
		copyReady = *copy;
	}
	oa::U32 colorSpace = 1U;  // BT.709 default
	switch (inColorSpace) {
		case oa::YCbCrModel::BT709:  colorSpace = 1U; break;
		case oa::YCbCrModel::BT2020: colorSpace = 2U; break;
		case oa::YCbCrModel::Auto:
			colorSpace = (inVisibleWidth >= 1280U or inVisibleHeight >= 720U) ? 1U : 0U;
			break;
	}

	// Push struct only carries the user-data fields. oavk::ImageDispatch::run
	// prepends the resource indices (rgba_idx / y_idx / uv_idx) from the
	// binding[] array so the shader sees the full PushConstants struct as
	// declared in CvtRgbaToNv12.slang.
	struct alignas(4) Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
		oa::U32 colorSpace;
		oa::U32 fullRange;
		oa::U32 pad;
	};
	Push push = {
		.width       = inVisibleWidth,
		.height      = inVisibleHeight,
		.codedWidth  = impl_->codedWidth,
		.codedHeight = impl_->codedHeight,
		.colorSpace  = colorSpace,
		.fullRange   = inFullRange ? 1U : 0U,
		.pad         = 0U,
	};

	oavk::ImageDispatchBinding bindings[3] = {};
	// bindings[0] — RGBA8 packed source buffer, indexed via heap[] in the
	// shader. run() picks up its bindlessIndex from the oavk::Buffer.
	bindings[0].kind        = oavk::DescriptorKind::StorageBuffer;
	bindings[0].binding     = 0;
	bindings[0].buffer      = inSlot.rgbaSnapshot;
	// bindings[1] — Y-plane storage image view (R8_UNORM).
	bindings[1].kind        = oavk::DescriptorKind::StorageImage;
	bindings[1].binding     = 1;
	bindings[1].imageView   = inSlot.inputYView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].image       = inSlot.inputImage;
	bindings[1].initialLayout = inSlot.inputInitialized
		? VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	bindings[1].finalLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	bindings[1].aspectMask  = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
	// bindings[2] — UV-plane storage image view (R8G8_UNORM).
	bindings[2].kind        = oavk::DescriptorKind::StorageImage;
	bindings[2].binding     = 2;
	bindings[2].imageView   = inSlot.inputUvView;
	bindings[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	const oa::U32 groupsX = (impl_->codedWidth  + 15U) / 16U;
	const oa::U32 groupsY = (impl_->codedHeight + 15U) / 16U;
	const auto submitConversion = [&]() -> oa::Result<oavk::ImageDispatchTicket> {
		if (copyReady.isValid()) {
			const oavk::TimelineWait wait =
				oa::EventAccess::timelineWait(copyReady);
			const oavk::TimelineSemaphore waitSemaphore =
				oa::EventAccess::timelineSemaphore(copyReady);
			return oavk::ImageDispatch::runWithDependencyAsync(
				vkEngine,
				"CvtRgbaToNv12",
				oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
				&push,
				static_cast<oa::U32>(sizeof(push)),
				oa::ScalarType::Float32,
				groupsX,
				groupsY,
				1U,
				waitSemaphore,
				wait.value);
		}
		return oavk::ImageDispatch::runAsync(
			vkEngine,
			"CvtRgbaToNv12",
			oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
			&push,
			static_cast<oa::U32>(sizeof(push)),
			oa::ScalarType::Float32,
			groupsX,
			groupsY,
			1U);
	};
	auto ticketResult = submitConversion();
	if (not ticketResult.isOk()) return ticketResult.getStatus();
	inSlot.inputTicket = oa::move(*ticketResult);
	inSlot.inputInitialized = true;
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::uploadInputRgbaImage_(
	EncodeSlot& inSlot,
	VkImage inImage,
	VkImageView inImageView,
	VkFormat inFormat,
	VkImageLayout inLayout,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::YCbCrModel inColorSpace,
	bool inFullRange,
	oa::U32 inArrayLayer,
	oa::Event inReady,
	oa::U32 inExternalQueueFamilyIndex)
{
	if (impl_->engine == nullptr or impl_->session.handle() == VK_NULL_HANDLE) {
		return oa::Status::error("oa::VideoEncoder::submitRgbaImage: encoder not initialized");
	}
	if (inImage == VK_NULL_HANDLE or inImageView == VK_NULL_HANDLE) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::submitRgbaImage requires a valid image and view");
	}
	if (inFormat != VK_FORMAT_R8G8B8A8_UNORM and inFormat != VK_FORMAT_B8G8R8A8_UNORM) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::submitRgbaImage supports RGBA8/BGRA8 UNORM images");
	}
	if (inLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::submitRgbaImage requires the producer's current image layout");
	}
	if (inVisibleWidth == 0U or inVisibleHeight == 0U
		or inVisibleWidth > impl_->codedWidth or inVisibleHeight > impl_->codedHeight) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::submitRgbaImage visible extent is out of range");
	}
	oa::U32 colorSpace = 1U;
	switch (inColorSpace) {
		case oa::YCbCrModel::BT709:  colorSpace = 1U; break;
		case oa::YCbCrModel::BT2020: colorSpace = 2U; break;
		case oa::YCbCrModel::Auto:
			colorSpace = (inVisibleWidth >= 1280U or inVisibleHeight >= 720U) ? 1U : 0U;
			break;
	}

	struct alignas(4) Push {
		oa::U32 width;
		oa::U32 height;
		oa::U32 codedWidth;
		oa::U32 codedHeight;
		oa::U32 colorSpace;
		oa::U32 fullRange;
		oa::U32 pad;
	};
	const Push push = {
		.width = inVisibleWidth,
		.height = inVisibleHeight,
		.codedWidth = impl_->codedWidth,
		.codedHeight = impl_->codedHeight,
		.colorSpace = colorSpace,
		.fullRange = inFullRange ? 1U : 0U,
		.pad = 0U,
	};

	oavk::ImageDispatchBinding bindings[3] = {};
	bindings[0].kind = oavk::DescriptorKind::SampledImage;
	bindings[0].imageView = inImageView;
	bindings[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[0].image = inImage;
	bindings[0].initialLayout = inLayout;
	bindings[0].finalLayout = inLayout;
	bindings[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bindings[0].baseArrayLayer = inArrayLayer;
	bindings[0].initialQueueFamilyIndex = inExternalQueueFamilyIndex;
	bindings[0].finalQueueFamilyIndex = inExternalQueueFamilyIndex;

	bindings[1].kind = oavk::DescriptorKind::StorageImage;
	bindings[1].imageView = inSlot.inputYView;
	bindings[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].image = inSlot.inputImage;
	bindings[1].initialLayout = inSlot.inputInitialized
		? VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	bindings[1].finalLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	bindings[1].aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;

	bindings[2].kind = oavk::DescriptorKind::StorageImage;
	bindings[2].imageView = inSlot.inputUvView;
	bindings[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	auto& vkEngine = *impl_->engine;
	const oa::U32 groupsX = (impl_->codedWidth + 15U) / 16U;
	const oa::U32 groupsY = (impl_->codedHeight + 15U) / 16U;
	const oavk::TimelineWait ready = oa::EventAccess::timelineWait(inReady);
	const oavk::TimelineSemaphore readySemaphore =
		oa::EventAccess::timelineSemaphore(inReady);
	oa::Result<oavk::ImageDispatchTicket> ticketResult = inReady.isValid()
		? oavk::ImageDispatch::runWithDependencyAsync(
			vkEngine, "CvtRgbaImageToNv12",
			oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
			&push, static_cast<oa::U32>(sizeof(push)), oa::ScalarType::Float32,
			groupsX, groupsY, 1U,
			readySemaphore, ready.value)
		: oavk::ImageDispatch::runAsync(
			vkEngine, "CvtRgbaImageToNv12",
			oa::Span<const oavk::ImageDispatchBinding>(bindings, 3),
			&push, static_cast<oa::U32>(sizeof(push)), oa::ScalarType::Float32,
			groupsX, groupsY, 1U);
	if (not ticketResult.isOk()) return ticketResult.getStatus();
	inSlot.inputTicket = oa::move(*ticketResult);
	inSlot.inputInitialized = true;
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::encodeFrame(
	VkImage inImage,
	oa::U64 inPts,
	oa::EncodedVideoPacket& outFrame)
{
	(void)inImage;
	if (impl_->slots.empty() or not impl_->compatibilityUploadReady) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"encodeFrame requires a preceding uploadInputRgba call");
	}
	OA_RETURN_IF_ERROR(submitEncode_(impl_->slots[0], inPts));
	impl_->compatibilityUploadReady = false;
	bool ready = false;
	OA_RETURN_IF_ERROR(harvest_(impl_->slots[0], true, outFrame, ready));
	if (not ready) {
		return oa::Status::error("Synchronous video encode did not produce a completed frame");
	}
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::submitEncode_(EncodeSlot& inSlot, oa::U64 inPts)
{
	// phase D: P-frame support with GOP structure
	// - Every impl_->gopSize frames, emit an IDR (keyframe)
	// - Otherwise, emit P-frames referencing the previous frame
	// - DPB slot 0 is used for the reference frame
	if (impl_->engine == nullptr or impl_->session.handle() == VK_NULL_HANDLE) {
		return oa::Status::error("oa::VideoEncoder::encodeFrame: encoder not initialized");
	}
	const auto& deviceDispatch =
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch;
	if (deviceDispatch.vkCmdBeginVideoCodingKHR == nullptr
		or deviceDispatch.vkCmdEncodeVideoKHR == nullptr
		or deviceDispatch.vkCmdEndVideoCodingKHR == nullptr
		or deviceDispatch.vkCmdControlVideoCodingKHR == nullptr) {
		return oa::Status::error("vulkan Video encode commands are not loaded");
	}

	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	VkQueue  queue  = static_cast<VkQueue>(oa::EngineDeviceAccess::get(vkEngine).queues.videoEncodeQueue);
	if (queue == VK_NULL_HANDLE) {
		return oa::Status::error("Video encode queue is null");
	}

	// clear only the range written by this slot's preceding job. This makes a
	// zero-byte feedback fallback unambiguous without paying to clear the full
	// 4 MiB buffer every frame.
	if (inSlot.bitstreamDirtyEnd > 0U) {
		oa::memset(inSlot.bitstream.getMappedPtr(), 0,
			static_cast<oa::Usize>(inSlot.bitstreamDirtyEnd));
		VkResult flushResult = vma::flushAllocation(
			static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
			static_cast<vma::Allocation>(inSlot.bitstream.getAllocation()),
			0, inSlot.bitstreamDirtyEnd);
		if (flushResult != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"Failed to clear reused video bitstream range");
		}
		inSlot.bitstreamDirtyEnd = 0U;
	}

	// ── Begin command buffer ────────────────────────────────────────
	VkResult result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkResetCommandBuffer(inSlot.commandBuffer, 0);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkResetCommandBuffer (encode) failed");
	}
	VkCommandBufferBeginInfo cbBegin = {};
	cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cbBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkBeginCommandBuffer(inSlot.commandBuffer, &cbBegin);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkBeginCommandBuffer (encode) failed");
	}

	// ── reset feedback query ────────────────────────────────────────
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdResetQueryPool(inSlot.commandBuffer, inSlot.queryPool, 0, 1);

	// ── Image layout transitions (synchronization2) ──────────────────
	// The KHR video-encode access/stage bits are only defined on the
	// synchronization2 path, so we use VkImageMemoryBarrier2 / vkCmd
	// PipelineBarrier2 here regardless of the rest of the codebase.
	// input image: GENERAL (where CvtRgbaToNv12 left it) → VIDEO_ENCODE_SRC_KHR.
	// DPB image:   UNDEFINED on first use → VIDEO_ENCODE_DPB_KHR.
	VkImageMemoryBarrier2 inputBarrier = {};
	inputBarrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	inputBarrier.srcStageMask                    = VK_PIPELINE_STAGE_2_NONE;
	inputBarrier.srcAccessMask                   = VK_ACCESS_2_NONE;
	inputBarrier.dstStageMask                    = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
	inputBarrier.dstAccessMask                   = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
	inputBarrier.oldLayout                       = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	inputBarrier.newLayout                       = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	inputBarrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	inputBarrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	inputBarrier.image                           = inSlot.inputImage;
	inputBarrier.subresourceRange.aspectMask     =
		  VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
	inputBarrier.subresourceRange.baseMipLevel   = 0;
	inputBarrier.subresourceRange.levelCount     = 1;
	inputBarrier.subresourceRange.baseArrayLayer = 0;
	inputBarrier.subresourceRange.layerCount     = 1;

	VkImageMemoryBarrier2 dpbBarrier = inputBarrier;
	// queue submission order alone is not a memory dependency. Once the DPB
	// has been initialized, make the preceding encode's reference write
	// available to the next P-frame read/write. Without this dependency Intel
	// Xe intermittently completed feedback queries with zero output bytes.
	dpbBarrier.srcStageMask                    = impl_->rateControlReset
		? VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR : VK_PIPELINE_STAGE_2_NONE;
	dpbBarrier.srcAccessMask                   = impl_->rateControlReset
		? VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR
		: VK_ACCESS_2_NONE;
	dpbBarrier.dstStageMask                    = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
	dpbBarrier.dstAccessMask                   = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
	dpbBarrier.oldLayout                       = impl_->rateControlReset ? VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	dpbBarrier.newLayout                       = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
	dpbBarrier.image                           = impl_->dpb.getImage();
	dpbBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	dpbBarrier.subresourceRange.layerCount     = impl_->dpbSlotCapacity;

	const VkImageMemoryBarrier2 barriers[2] = { inputBarrier, dpbBarrier };
	VkDependencyInfo depInfo = {};
	depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	depInfo.imageMemoryBarrierCount  = 2;
	depInfo.pImageMemoryBarriers     = barriers;
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdPipelineBarrier2(inSlot.commandBuffer, &depInfo);

	// ── Begin video coding ──────────────────────────────────────────
	const bool isIdr = (impl_->currentGopFrame == 0);
	const oa::U32 referenceDepth = impl_->dpbSlotCapacity > 0U ? impl_->dpbSlotCapacity : 1U;
	const oa::U32 currentDpbSlot = impl_->frameCount % referenceDepth;
	const oa::U32 previousDpbSlot = impl_->frameCount == 0U
		? currentDpbSlot : (impl_->frameCount - 1U) % referenceDepth;

	// Bind active DPB resources into the coding scope. Listing a slot only in
	// VkVideoEncodeInfoKHR is insufficient: vkCmdBeginVideoCodingKHR defines
	// the slot-to-picture association visible to commands in this scope.
	StdVideoEncodeH264ReferenceInfo beginPreviousRefInfo = {};
	VkVideoEncodeH264DpbSlotInfoKHR beginPreviousH264 = {};
	StdVideoEncodeH264ReferenceInfo beginCurrentH264RefInfo = {};
	VkVideoEncodeH264DpbSlotInfoKHR beginCurrentH264 = {};
	StdVideoEncodeH265ReferenceInfo beginPreviousH265RefInfo = {};
	VkVideoEncodeH265DpbSlotInfoKHR beginPreviousH265 = {};
	VkVideoPictureResourceInfoKHR beginPreviousResource = {};
	VkVideoReferenceSlotInfoKHR beginReferenceSlot = {};
	StdVideoEncodeH265ReferenceInfo beginCurrentH265RefInfo = {};
	VkVideoEncodeH265DpbSlotInfoKHR beginCurrentH265 = {};
	VkVideoPictureResourceInfoKHR beginCurrentResource = {};
	VkVideoReferenceSlotInfoKHR beginH264Slots[2] = {};
	VkVideoReferenceSlotInfoKHR beginH265Slots[2] = {};
	if (not isIdr) {
		if (impl_->profile.codec == oa::VideoCodec::H264) {
			beginPreviousRefInfo.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_P;
			beginPreviousRefInfo.FrameNum = impl_->frameCount - impl_->lastKeyframeIndex - 1U;
			beginPreviousRefInfo.PicOrderCnt = static_cast<oa::I32>(
				impl_->frameCount - impl_->lastKeyframeIndex - 1U);
			beginPreviousH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
			beginPreviousH264.pStdReferenceInfo = &beginPreviousRefInfo;
			beginReferenceSlot.pNext = &beginPreviousH264;
		} else {
			beginPreviousH265RefInfo.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
			beginPreviousH265RefInfo.PicOrderCntVal =
				static_cast<oa::I32>(impl_->frameCount - impl_->lastKeyframeIndex - 1U);
			beginPreviousH265RefInfo.TemporalId = 0;
			beginPreviousH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
			beginPreviousH265.pStdReferenceInfo = &beginPreviousH265RefInfo;
			beginReferenceSlot.pNext = &beginPreviousH265;
		}
		beginPreviousResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginPreviousResource.codedExtent = { impl_->codedWidth, impl_->codedHeight };
		beginPreviousResource.baseArrayLayer = previousDpbSlot;
		beginPreviousResource.imageViewBinding = impl_->dpb.getView();
		beginReferenceSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		beginReferenceSlot.slotIndex = static_cast<oa::I32>(previousDpbSlot);
		beginReferenceSlot.pPictureResource = &beginPreviousResource;
	}
	if (impl_->profile.codec == oa::VideoCodec::H264) {
		// The picture named by pSetupReferenceSlot must be associated with the
		// coding scope too. Bind the target layer as an inactive resource; the
		// encode command activates the same resource at currentDpbSlot.
		beginCurrentH264RefInfo.primary_pic_type = isIdr
			? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
		beginCurrentH264RefInfo.FrameNum = isIdr
			? 0U : impl_->frameCount - impl_->lastKeyframeIndex;
		beginCurrentH264RefInfo.PicOrderCnt = isIdr
			? 0 : static_cast<oa::I32>(impl_->frameCount - impl_->lastKeyframeIndex);
		beginCurrentH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
		beginCurrentH264.pStdReferenceInfo = &beginCurrentH264RefInfo;
		beginCurrentResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginCurrentResource.codedExtent = { impl_->codedWidth, impl_->codedHeight };
		beginCurrentResource.baseArrayLayer = currentDpbSlot;
		beginCurrentResource.imageViewBinding = impl_->dpb.getView();
		beginH264Slots[0].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		beginH264Slots[0].pNext = &beginCurrentH264;
		beginH264Slots[0].slotIndex = -1;
		beginH264Slots[0].pPictureResource = &beginCurrentResource;
		if (not isIdr) beginH264Slots[1] = beginReferenceSlot;
	} else {
		// An encode begin scope must include the target DPB resource as an
		// inactive association (slotIndex=-1). This explicitly releases the
		// layer's previous slot association before a later IDR reuses it. The
		// Khronos encoder follows the same setup-resource + active-references
		// convention; omitting it caused Intel Xe to hang on the second IDR.
		// The setup slot describes the DPB reference state, not the NAL picture
		// type. The Khronos reference encoder leaves this at P for newly stored
		// pictures (including the first IDR), which is also what Intel expects.
		beginCurrentH265RefInfo.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
		beginCurrentH265RefInfo.PicOrderCntVal = isIdr
			? 0 : static_cast<oa::I32>(impl_->frameCount - impl_->lastKeyframeIndex);
		beginCurrentH265RefInfo.TemporalId = 0;
		beginCurrentH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
		beginCurrentH265.pStdReferenceInfo = &beginCurrentH265RefInfo;
		beginCurrentResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginCurrentResource.codedExtent = { impl_->codedWidth, impl_->codedHeight };
		beginCurrentResource.baseArrayLayer = currentDpbSlot;
		beginCurrentResource.imageViewBinding = impl_->dpb.getView();
		beginH265Slots[0].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		beginH265Slots[0].pNext = &beginCurrentH265;
		beginH265Slots[0].slotIndex = -1;
		beginH265Slots[0].pPictureResource = &beginCurrentResource;
		if (not isIdr) beginH265Slots[1] = beginReferenceSlot;
	}

	// The coding scope must declare the active rate-control state. in
	// particular, constant-QP uses DISABLED rather than the session's DEFAULT;
	// omitting this chain makes non-zero slice QP invalid after the first scope.
	VkVideoEncodeRateControlInfoKHR beginRcInfo = {};
	VkVideoEncodeRateControlLayerInfoKHR beginRcLayer = {};
	VkVideoEncodeH264RateControlInfoKHR beginH264Rc = {};
	VkVideoEncodeH264RateControlLayerInfoKHR beginH264Layer = {};
	VkVideoEncodeH265RateControlInfoKHR beginH265Rc = {};
	VkVideoEncodeH265RateControlLayerInfoKHR beginH265Layer = {};
	const bool beginRcEnabled =
		impl_->rateControlMode != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
		and impl_->profile.bitrate > 0U;
	beginRcInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR;
	beginRcInfo.rateControlMode = not impl_->rateControlReset
		? VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR
		: (beginRcEnabled ? impl_->rateControlMode
			: VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR);
	if (beginRcEnabled and impl_->rateControlReset) {
		beginRcLayer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR;
		beginRcLayer.averageBitrate = impl_->profile.bitrate;
		beginRcLayer.maxBitrate = impl_->rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR
			? impl_->profile.bitrate : impl_->profile.maxBitrate;
		beginRcLayer.frameRateNumerator = impl_->profile.frameRate;
		beginRcLayer.frameRateDenominator = 1U;
		if (impl_->profile.codec == oa::VideoCodec::H264) {
			beginH264Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR;
			beginRcLayer.pNext = &beginH264Layer;
			beginH264Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR;
			beginH264Rc.gopFrameCount = impl_->profile.gopSize;
			beginH264Rc.idrPeriod = impl_->profile.gopSize;
			beginH264Rc.consecutiveBFrameCount = impl_->profile.maxBFrames;
			beginH264Rc.temporalLayerCount = 1U;
			beginRcInfo.pNext = &beginH264Rc;
		} else {
			beginH265Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR;
			beginRcLayer.pNext = &beginH265Layer;
			beginH265Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR;
			beginH265Rc.gopFrameCount = impl_->profile.gopSize;
			beginH265Rc.idrPeriod = impl_->profile.gopSize;
			beginH265Rc.consecutiveBFrameCount = impl_->profile.maxBFrames;
			beginH265Rc.subLayerCount = 1U;
			beginRcInfo.pNext = &beginH265Rc;
		}
		beginRcInfo.layerCount = 1U;
		beginRcInfo.pLayers = &beginRcLayer;
		beginRcInfo.virtualBufferSizeInMs = 1000U;
		beginRcInfo.initialVirtualBufferSizeInMs = 500U;
	}

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType                  = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.pNext                  = &beginRcInfo;
	beginCoding.videoSession           = impl_->session.handle();
	beginCoding.videoSessionParameters = impl_->sessionParameters.handle();
	beginCoding.referenceSlotCount = isIdr ? 1U : 2U;
	beginCoding.pReferenceSlots = impl_->profile.codec == oa::VideoCodec::H265
		? beginH265Slots : beginH264Slots;
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdBeginVideoCodingKHR(inSlot.commandBuffer, &beginCoding);

	// ── first-time: RESET the encoder + program rate control ───────────
	// constantQp on the slice header is ignored once we attach a non-DISABLED
	// rate-control info chain; we keep it as a sensible fallback for drivers
	// that downgrade us to DISABLED mid-session.
	if (not impl_->rateControlReset) {
		VkVideoCodingControlInfoKHR controlInfo = {};
		controlInfo.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
		controlInfo.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;

		VkVideoEncodeRateControlInfoKHR        rcInfo       = {};
		VkVideoEncodeRateControlLayerInfoKHR   layer        = {};
		VkVideoEncodeH264RateControlInfoKHR    h264Rc       = {};
		VkVideoEncodeH264RateControlLayerInfoKHR h264Layer  = {};
		VkVideoEncodeH265RateControlInfoKHR    h265Rc       = {};
		VkVideoEncodeH265RateControlLayerInfoKHR h265Layer  = {};

		const bool rcEnabled = impl_->rateControlMode != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
			and impl_->profile.bitrate > 0U;
		// RESET returns rate control to DEFAULT, not DISABLED. program the
		// selected mode explicitly even for constant-QP so slice QP is legal.
		rcInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR;
		rcInfo.rateControlMode = rcEnabled
			? impl_->rateControlMode : VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
		if (rcEnabled) {
			if (impl_->profile.codec == oa::VideoCodec::H264) {
				h264Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR;
				h264Layer.useMinQp = VK_FALSE;
				h264Layer.useMaxQp = VK_FALSE;
				h264Layer.useMaxFrameSize = VK_FALSE;
				layer.pNext = &h264Layer;
			} else {
				h265Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR;
				h265Layer.useMinQp = VK_FALSE;
				h265Layer.useMaxQp = VK_FALSE;
				h265Layer.useMaxFrameSize = VK_FALSE;
				layer.pNext = &h265Layer;
			}

			layer.sType                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR;
			layer.averageBitrate         = impl_->profile.bitrate;
			layer.maxBitrate = impl_->rateControlMode == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR
				? impl_->profile.bitrate : impl_->profile.maxBitrate;
			layer.frameRateNumerator     = impl_->profile.frameRate;
			layer.frameRateDenominator   = 1U;

			if (impl_->profile.codec == oa::VideoCodec::H264) {
				h264Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR;
				h264Rc.gopFrameCount = impl_->profile.gopSize;
				h264Rc.idrPeriod = impl_->profile.gopSize;
				h264Rc.consecutiveBFrameCount = impl_->profile.maxBFrames;
				h264Rc.temporalLayerCount = 1U;
				rcInfo.pNext = &h264Rc;
			} else {
				h265Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR;
				h265Rc.gopFrameCount = impl_->profile.gopSize;
				h265Rc.idrPeriod = impl_->profile.gopSize;
				h265Rc.consecutiveBFrameCount = impl_->profile.maxBFrames;
				h265Rc.subLayerCount = 1U;
				rcInfo.pNext = &h265Rc;
			}

			rcInfo.layerCount       = 1U;
			rcInfo.pLayers          = &layer;
			rcInfo.virtualBufferSizeInMs   = 1000U;
			rcInfo.initialVirtualBufferSizeInMs = 500U;

		}
		controlInfo.pNext = &rcInfo;
		controlInfo.flags |= VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR;

		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdControlVideoCodingKHR(inSlot.commandBuffer, &controlInfo);
		impl_->rateControlReset = true;
	}

	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdBeginQuery(inSlot.commandBuffer, inSlot.queryPool, 0, 0);

	if (impl_->profile.codec == oa::VideoCodec::H264) {
	// ── Build the H.264 picture / slice info ────────────────────────
	// phase D — IDR + P cadence. impl_->profile.maxBFrames is propagated into
	// rate-control's consecutiveBFrameCount above so the driver budgets
	// for them, but actual B-frame *emission* needs a display-order
	// reorder buffer + a 2nd DPB slot for the future reference. That
	// architecture is tracked separately; emitting STD_VIDEO_H264_SLICE_TYPE_B
	// here without it would produce a malformed bitstream.
	StdVideoEncodeH264ReferenceListsInfo refListsInfo = {};
	refListsInfo.num_ref_idx_l0_active_minus1 = 0;
	refListsInfo.num_ref_idx_l1_active_minus1 = 0;
	for (auto& v : refListsInfo.RefPicList0) { v = STD_VIDEO_H264_NO_REFERENCE_PICTURE; }
	for (auto& v : refListsInfo.RefPicList1) { v = STD_VIDEO_H264_NO_REFERENCE_PICTURE; }
	if (not isIdr) {
		refListsInfo.RefPicList0[0] = static_cast<oa::U8>(previousDpbSlot);
	}

	StdVideoEncodeH264PictureInfo h264PicInfo = {};
	h264PicInfo.flags.IdrPicFlag                       = isIdr ? 1 : 0;
	h264PicInfo.flags.is_reference                     = 1;
	h264PicInfo.flags.no_output_of_prior_pics_flag     = isIdr ? 1 : 0;
	h264PicInfo.flags.long_term_reference_flag         = 0;
	h264PicInfo.flags.adaptive_ref_pic_marking_mode_flag = 0;
	h264PicInfo.seq_parameter_set_id                   = 0;
	h264PicInfo.pic_parameter_set_id                   = 0;
	h264PicInfo.idr_pic_id                             = static_cast<uint16_t>(impl_->frameCount & 0xFFFFU);
	h264PicInfo.primary_pic_type                       = isIdr ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
	h264PicInfo.frame_num                              = isIdr ? 0 : (impl_->frameCount - impl_->lastKeyframeIndex);
	h264PicInfo.PicOrderCnt = isIdr ? 0 : static_cast<oa::I32>(
		impl_->frameCount - impl_->lastKeyframeIndex);
	h264PicInfo.pRefLists                              = &refListsInfo;

	StdVideoEncodeH264SliceHeader sliceHeader = {};
	sliceHeader.flags.direct_spatial_mv_pred_flag      = 0;
	sliceHeader.flags.num_ref_idx_active_override_flag = 0;
	sliceHeader.first_mb_in_slice                      = 0;
	sliceHeader.slice_type                             = isIdr ? STD_VIDEO_H264_SLICE_TYPE_I : STD_VIDEO_H264_SLICE_TYPE_P;
	sliceHeader.cabac_init_idc                         = STD_VIDEO_H264_CABAC_INIT_IDC_0;
	sliceHeader.disable_deblocking_filter_idc          = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED;
	sliceHeader.slice_alpha_c0_offset_div2             = 0;
	sliceHeader.slice_beta_offset_div2                 = 0;

	VkVideoEncodeH264NaluSliceInfoKHR naluSlice = {};
	naluSlice.sType            = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR;
	naluSlice.constantQp       = static_cast<oa::I32>(impl_->profile.constantQp);
	naluSlice.pStdSliceHeader  = &sliceHeader;

	VkVideoEncodeH264PictureInfoKHR h264Picture = {};
	h264Picture.sType                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR;
	h264Picture.naluSliceEntryCount    = 1;
	h264Picture.pNaluSliceEntries      = &naluSlice;
	h264Picture.pStdPictureInfo        = &h264PicInfo;
	h264Picture.generatePrefixNalu     = VK_FALSE;

	// Ping-pong two DPB layers: the current picture is written to one while a
	// P-frame reads the previous layer. Reading and overwriting slot 0 in the
	// same command is invalid and caused intermittent zero-byte Intel outputs.
	StdVideoEncodeH264ReferenceInfo currentRefInfo = {};
	currentRefInfo.primary_pic_type = isIdr
		? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
	currentRefInfo.FrameNum = h264PicInfo.frame_num;
	currentRefInfo.PicOrderCnt = h264PicInfo.PicOrderCnt;
	VkVideoEncodeH264DpbSlotInfoKHR currentSlotH264 = {};
	currentSlotH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
	currentSlotH264.pStdReferenceInfo = &currentRefInfo;
	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType                = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedOffset          = { 0, 0 };
	setupResource.codedExtent          = { impl_->codedWidth, impl_->codedHeight };
	setupResource.baseArrayLayer       = currentDpbSlot;
	setupResource.imageViewBinding     = impl_->dpb.getView();
	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType            = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext            = &currentSlotH264;
	setupSlot.slotIndex        = static_cast<oa::I32>(currentDpbSlot);
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR srcResource = {};
	srcResource.sType            = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	srcResource.codedOffset      = { 0, 0 };
	srcResource.codedExtent      = { impl_->codedWidth, impl_->codedHeight };
	srcResource.baseArrayLayer   = 0;
	srcResource.imageViewBinding = inSlot.inputView;

	VkVideoEncodeInfoKHR encodeInfo = {};
	encodeInfo.sType                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR;
	encodeInfo.pNext                = &h264Picture;
	encodeInfo.dstBuffer            = inSlot.bitstream.getBuffer();
	encodeInfo.dstBufferOffset      = 0;
	encodeInfo.dstBufferRange       = static_cast<oa::U32>(inSlot.bitstream.getCapacity());
	encodeInfo.srcPictureResource   = srcResource;
	encodeInfo.pSetupReferenceSlot  = &setupSlot;

	// phase D: P-frames need reference slots
	StdVideoEncodeH264ReferenceInfo previousRefInfo = {};
	previousRefInfo.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_P;
	previousRefInfo.FrameNum = impl_->frameCount - impl_->lastKeyframeIndex - 1U;
	previousRefInfo.PicOrderCnt = static_cast<oa::I32>(
		impl_->frameCount - impl_->lastKeyframeIndex - 1U);
	VkVideoEncodeH264DpbSlotInfoKHR previousSlotH264 = {};
	previousSlotH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
	previousSlotH264.pStdReferenceInfo = &previousRefInfo;
	VkVideoPictureResourceInfoKHR previousResource = setupResource;
	previousResource.baseArrayLayer = previousDpbSlot;
	VkVideoReferenceSlotInfoKHR refSlots[1] = {};
	if (!isIdr) {
		// P-frame: read the previous layer, write the current layer.
		refSlots[0].sType            = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		refSlots[0].pNext            = &previousSlotH264;
		refSlots[0].slotIndex        = static_cast<oa::I32>(previousDpbSlot);
		refSlots[0].pPictureResource = &previousResource;
		encodeInfo.referenceSlotCount = 1;
		encodeInfo.pReferenceSlots    = refSlots;
	} else {
		// IDR: no reference slots
		encodeInfo.referenceSlotCount = 0;
		encodeInfo.pReferenceSlots    = nullptr;
	}

	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdEncodeVideoKHR(inSlot.commandBuffer, &encodeInfo);
	} else {
		// ── Build the H.265 picture / slice info ─────────────────────
		StdVideoEncodeH265ReferenceListsInfo refListsInfo = {};
		for (auto& value : refListsInfo.RefPicList0) {
			value = STD_VIDEO_H265_NO_REFERENCE_PICTURE;
		}
		for (auto& value : refListsInfo.RefPicList1) {
			value = STD_VIDEO_H265_NO_REFERENCE_PICTURE;
		}
		for (auto& value : refListsInfo.list_entry_l0) {
			value = STD_VIDEO_H265_NO_REFERENCE_PICTURE;
		}
		for (auto& value : refListsInfo.list_entry_l1) {
			value = STD_VIDEO_H265_NO_REFERENCE_PICTURE;
		}
		if (not isIdr) {
			refListsInfo.num_ref_idx_l0_active_minus1 = 0;
			refListsInfo.RefPicList0[0] = static_cast<oa::U8>(previousDpbSlot);
		}

		const oa::I32 picturePoc = isIdr
			? 0 : static_cast<oa::I32>(impl_->frameCount - impl_->lastKeyframeIndex);
		StdVideoEncodeH265PictureInfo h265PicInfo = {};
		h265PicInfo.flags.is_reference = 1;
		h265PicInfo.flags.IrapPicFlag = isIdr ? 1 : 0;
		h265PicInfo.flags.pic_output_flag = 1;
		h265PicInfo.flags.no_output_of_prior_pics_flag =
			isIdr and impl_->frameCount != 0U ? 1 : 0;
		h265PicInfo.flags.short_term_ref_pic_set_sps_flag = 1;
		h265PicInfo.pic_type = isIdr
			? STD_VIDEO_H265_PICTURE_TYPE_IDR : STD_VIDEO_H265_PICTURE_TYPE_P;
		h265PicInfo.sps_video_parameter_set_id = 0;
		h265PicInfo.pps_seq_parameter_set_id = 0;
		h265PicInfo.pps_pic_parameter_set_id = 0;
		h265PicInfo.short_term_ref_pic_set_idx = 0;
		h265PicInfo.PicOrderCntVal = picturePoc;
		h265PicInfo.TemporalId = 0;
		h265PicInfo.pRefLists = isIdr ? nullptr : &refListsInfo;

		StdVideoEncodeH265SliceSegmentHeader sliceHeader = {};
		sliceHeader.flags.first_slice_segment_in_pic_flag = 1;
		sliceHeader.flags.slice_sao_luma_flag = 1;
		sliceHeader.flags.slice_sao_chroma_flag = 1;
		sliceHeader.slice_type = isIdr
			? STD_VIDEO_H265_SLICE_TYPE_I : STD_VIDEO_H265_SLICE_TYPE_P;
		sliceHeader.MaxNumMergeCand = 5;

		VkVideoEncodeH265NaluSliceSegmentInfoKHR naluSlice = {};
		naluSlice.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_KHR;
		naluSlice.constantQp = static_cast<oa::I32>(impl_->profile.constantQp);
		naluSlice.pStdSliceSegmentHeader = &sliceHeader;

		VkVideoEncodeH265PictureInfoKHR h265Picture = {};
		h265Picture.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PICTURE_INFO_KHR;
		h265Picture.naluSliceSegmentEntryCount = 1;
		h265Picture.pNaluSliceSegmentEntries = &naluSlice;
		h265Picture.pStdPictureInfo = &h265PicInfo;

		StdVideoEncodeH265ReferenceInfo currentRefInfo = {};
		currentRefInfo.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
		currentRefInfo.PicOrderCntVal = picturePoc;
		currentRefInfo.TemporalId = 0;
		VkVideoEncodeH265DpbSlotInfoKHR currentSlotH265 = {};
		currentSlotH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
		currentSlotH265.pStdReferenceInfo = &currentRefInfo;
		VkVideoPictureResourceInfoKHR setupResource = {};
		setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		setupResource.codedExtent = { impl_->codedWidth, impl_->codedHeight };
		setupResource.baseArrayLayer = currentDpbSlot;
		setupResource.imageViewBinding = impl_->dpb.getView();
		VkVideoReferenceSlotInfoKHR setupSlot = {};
		setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		setupSlot.pNext = &currentSlotH265;
		setupSlot.slotIndex = static_cast<oa::I32>(currentDpbSlot);
		setupSlot.pPictureResource = &setupResource;

		VkVideoPictureResourceInfoKHR srcResource = {};
		srcResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		srcResource.codedExtent = { impl_->codedWidth, impl_->codedHeight };
		srcResource.imageViewBinding = inSlot.inputView;

		VkVideoEncodeInfoKHR encodeInfo = {};
		encodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR;
		encodeInfo.pNext = &h265Picture;
		encodeInfo.dstBuffer = inSlot.bitstream.getBuffer();
		encodeInfo.dstBufferOffset = 0;
		encodeInfo.dstBufferRange = inSlot.bitstream.getCapacity();
		encodeInfo.srcPictureResource = srcResource;
		encodeInfo.pSetupReferenceSlot = &setupSlot;

		StdVideoEncodeH265ReferenceInfo previousRefInfo = {};
		previousRefInfo.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
		previousRefInfo.PicOrderCntVal = picturePoc - 1;
		previousRefInfo.TemporalId = 0;
		VkVideoEncodeH265DpbSlotInfoKHR previousSlotH265 = {};
		previousSlotH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
		previousSlotH265.pStdReferenceInfo = &previousRefInfo;
		VkVideoPictureResourceInfoKHR previousResource = setupResource;
		previousResource.baseArrayLayer = previousDpbSlot;
		VkVideoReferenceSlotInfoKHR referenceSlot = {};
		if (not isIdr) {
			referenceSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
			referenceSlot.pNext = &previousSlotH265;
			referenceSlot.slotIndex = static_cast<oa::I32>(previousDpbSlot);
			referenceSlot.pPictureResource = &previousResource;
			encodeInfo.referenceSlotCount = 1;
			encodeInfo.pReferenceSlots = &referenceSlot;
		}
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdEncodeVideoKHR(inSlot.commandBuffer, &encodeInfo);
	}
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdEndQuery(inSlot.commandBuffer, inSlot.queryPool, 0);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkCmdEndVideoCodingKHR(inSlot.commandBuffer, &endCoding);

	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkEndCommandBuffer(inSlot.commandBuffer);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkEndCommandBuffer (encode) failed");
	}

	// ── submit without a host wait ──────────────────────────────────
	oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkResetFences(device, 1, &inSlot.fence);
	VkTimelineSemaphoreSubmitInfo timeline = {};
	VkSemaphore waitSemaphore = VK_NULL_HANDLE;
	oa::U64 waitValue = 0U;
	// VkSubmitInfo uses the legacy 32-bit stage mask; not every vulkan header
	// exposes a legacy VIDEO_ENCODE enumerant. ALL_COMMANDS safely gates the
	// video command buffer while the timeline dependency resolves.
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	if (inSlot.inputTicket.isValid()) {
		waitSemaphore = static_cast<VkSemaphore>(
			oa::EventAccess::semaphoreHandle(inSlot.inputTicket.completion()));
		waitValue = inSlot.inputTicket.value();
		timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timeline.waitSemaphoreValueCount = 1;
		timeline.pWaitSemaphoreValues = &waitValue;
	}
	VkSubmitInfo submit = {};
	submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext              = waitSemaphore != VK_NULL_HANDLE ? &timeline : nullptr;
	submit.waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1U : 0U;
	submit.pWaitSemaphores    = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr;
	submit.pWaitDstStageMask  = waitSemaphore != VK_NULL_HANDLE ? &waitStage : nullptr;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &inSlot.commandBuffer;
	result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkQueueSubmit(queue, 1, &submit, inSlot.fence);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkQueueSubmit (encode) failed");
	}
	inSlot.presentationTimestamp = inPts;
	inSlot.isKeyframe = isIdr;
	inSlot.pending = true;

	// GOP state advances at queue submission time. vulkan queue order makes the
	// shared reference slot deterministic even while older jobs are unharvested.
	if (isIdr) {
		impl_->lastKeyframeIndex = impl_->frameCount;
		impl_->currentGopFrame = impl_->gopSize > 1U ? 1U : 0U;
	} else {
		++impl_->currentGopFrame;
		if (impl_->currentGopFrame >= impl_->gopSize) impl_->currentGopFrame = 0U;
	}
	++impl_->frameCount;
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::harvest_(
	EncodeSlot& inSlot,
	bool inWait,
	oa::EncodedVideoPacket& outFrame,
	bool& outReady)
{
	outReady = false;
	if (not inSlot.pending) return oa::Status::ok();
	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	VkResult result = VK_SUCCESS;
	if (inWait) {
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkWaitForFences(
			device, 1, &inSlot.fence, VK_TRUE, oa::U64{1'000'000'000ULL} * 5ULL);
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				oa::String("vkWaitForFences (encode job) failed: VkResult=")
					+ oa::toString(static_cast<oa::I64>(result)));
		}
	} else {
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkGetFenceStatus(device, inSlot.fence);
		if (result == VK_NOT_READY) return oa::Status::ok();
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				oa::String("vkGetFenceStatus (encode job) failed: VkResult=")
					+ oa::toString(static_cast<oa::I64>(result)));
		}
	}

	// The video fence implies its timeline wait was satisfied, so the compute
	// conversion ticket can now safely release its stream and bindless slots.
	OA_RETURN_IF_ERROR(inSlot.inputTicket.wait());

	struct Feedback {
		oa::U32 bitstreamStartOffset;
		oa::U32 bitstreamBytesWritten;
	};
	Feedback fb = {};
	VkQueryResultStatusKHR queryStatus = VK_QUERY_RESULT_STATUS_COMPLETE_KHR;
	if (impl_->queryResultStatusSupported) {
		struct FeedbackWithStatus {
			Feedback values;
			VkQueryResultStatusKHR status;
		};
		FeedbackWithStatus resultWithStatus = {};
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkGetQueryPoolResults(
			device, inSlot.queryPool, 0, 1,
			sizeof(resultWithStatus), &resultWithStatus,
			sizeof(resultWithStatus),
			VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
		fb = resultWithStatus.values;
		queryStatus = resultWithStatus.status;
	} else {
		result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkGetQueryPoolResults(
			device, inSlot.queryPool, 0, 1,
			sizeof(fb), &fb,
			sizeof(fb),
			VK_QUERY_RESULT_WAIT_BIT);
	}
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkGetQueryPoolResults (encode feedback) failed");
	}
	if (queryStatus != VK_QUERY_RESULT_STATUS_COMPLETE_KHR) {
		return oa::Status::error(
			"vulkan video encode query failed for PTS "
			+ oa::toString(static_cast<oa::I64>(inSlot.presentationTimestamp))
			+ " (status=" + oa::toString(static_cast<oa::I64>(queryStatus)) + ")");
	}
	if (fb.bitstreamBytesWritten == 0U) {
		(void)vma::invalidateAllocation(
			static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
			static_cast<vma::Allocation>(inSlot.bitstream.getAllocation()),
			0, inSlot.bitstream.getCapacity());
		const auto* data = static_cast<const oa::U8*>(inSlot.bitstream.getMappedPtr());
		oa::U64 nonzeroExtent = 0U;
		for (oa::U64 idx = inSlot.bitstream.getCapacity(); idx > 0U; --idx) {
			if (data[idx - 1U] != 0U) { nonzeroExtent = idx; break; }
		}
		oa::U64 annexBStart = nonzeroExtent;
		for (oa::U64 idx = 0U; idx + 3U < nonzeroExtent; ++idx) {
			if (data[idx] == 0U and data[idx + 1U] == 0U
				and ((data[idx + 2U] == 1U)
					or (data[idx + 2U] == 0U and data[idx + 3U] == 1U))) {
				annexBStart = idx;
				break;
			}
		}
		if (annexBStart >= nonzeroExtent) {
			return oa::Status::error(
				"vulkan video encoder produced neither feedback nor Annex-B payload for PTS "
				+ oa::toString(static_cast<oa::I64>(inSlot.presentationTimestamp)));
		}
		fb.bitstreamStartOffset = static_cast<oa::U32>(annexBStart);
		fb.bitstreamBytesWritten = static_cast<oa::U32>(nonzeroExtent - annexBStart);
		++impl_->zeroFeedbackRecoveryCount;
		if (impl_->zeroFeedbackRecoveryCount == 1U) {
			OaLogWarn(oa::LogComponent::Video,
				"Driver returned COMPLETE video encode feedback with zero bytes; "
				"recovered a validated %u-byte Annex-B payload (later recoveries suppressed)",
				static_cast<unsigned>(fb.bitstreamBytesWritten));
		} else {
			OaLogDebug(oa::LogComponent::Video,
				"Recovered zero-byte video encode feedback for PTS %llu (%u Annex-B bytes)",
				static_cast<unsigned long long>(inSlot.presentationTimestamp),
				static_cast<unsigned>(fb.bitstreamBytesWritten));
		}
	}

	// The encoder writes this GPU_TO_CPU allocation. Invalidate non-coherent
	// memory before reading its persistent mapping; coherent heaps make this a
	// no-op.
	result = vma::invalidateAllocation(
		static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator),
		static_cast<vma::Allocation>(inSlot.bitstream.getAllocation()),
		fb.bitstreamStartOffset, fb.bitstreamBytesWritten);
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"Failed to invalidate encoded video bitstream memory");
	}
	inSlot.bitstreamDirtyEnd = static_cast<oa::U64>(fb.bitstreamStartOffset)
		+ fb.bitstreamBytesWritten;

	// ── Pack oa::EncodedVideoPacket ─────────────────────────────────────────
	const oa::U8* slicePtr   = static_cast<const oa::U8*>(inSlot.bitstream.getMappedPtr()) + fb.bitstreamStartOffset;
	const oa::Usize sliceLen = fb.bitstreamBytesWritten;
	const oa::Usize hdrLen   = inSlot.isKeyframe ? impl_->cachedHeaders.size() : 0U;
	outFrame.bitstream.resize(hdrLen + sliceLen);
	if (hdrLen > 0U) {
		oa::memcpy(outFrame.bitstream.data(), impl_->cachedHeaders.data(), hdrLen);
	}
	oa::memcpy(outFrame.bitstream.data() + hdrLen, slicePtr, sliceLen);
	outFrame.presentationTimestamp = inSlot.presentationTimestamp;
	outFrame.isKeyframe            = inSlot.isKeyframe;
	outFrame.frameSize             = static_cast<oa::U32>(hdrLen + sliceLen);
	inSlot.pending = false;
	outReady = true;
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::submitRgba(
	const oavk::Buffer& inRgba,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::U64 inPts,
	oa::Vector<oa::EncodedVideoPacket>& outReady,
	oa::YCbCrModel inColorSpace,
	bool inFullRange)
{
	if (impl_->slots.empty()) return oa::Status::error("Video encoder has no asynchronous slots");
	if (impl_->compatibilityUploadReady) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Cannot submit an asynchronous job after an unmatched uploadInputRgba call");
	}

	// ring wrap: the selected slot is the oldest submitted job.
	EncodeSlot& slot = impl_->slots[impl_->submitSlot];
	if (slot.pending) {
		oa::EncodedVideoPacket completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(harvest_(slot, true, completed, ready));
		if (ready) {
			outReady.pushBack(oa::move(completed));
			--impl_->pendingSlots;
			impl_->harvestSlot = (impl_->harvestSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());
		}
	}
	OA_RETURN_IF_ERROR(uploadInputRgba_(
		slot, inRgba, inVisibleWidth, inVisibleHeight, inColorSpace, inFullRange));
	OA_RETURN_IF_ERROR(submitEncode_(slot, inPts));
	++impl_->pendingSlots;
	impl_->submitSlot = (impl_->submitSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());

	// Harvest any already-completed prefix without waiting. Never return later
	// frames ahead of an unfinished earlier submission.
	while (impl_->pendingSlots > 0U) {
		EncodeSlot& oldest = impl_->slots[impl_->harvestSlot];
		oa::EncodedVideoPacket completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(harvest_(oldest, false, completed, ready));
		if (not ready) break;
		outReady.pushBack(oa::move(completed));
		--impl_->pendingSlots;
		impl_->harvestSlot = (impl_->harvestSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());
	}
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::submitRgbaImage(
	VkImage inImage,
	VkImageView inImageView,
	VkFormat inFormat,
	VkImageLayout inLayout,
	oa::U32 inVisibleWidth,
	oa::U32 inVisibleHeight,
	oa::U64 inPts,
	oa::Vector<oa::EncodedVideoPacket>& outReady,
	oa::YCbCrModel inColorSpace,
	bool inFullRange,
	oa::U32 inArrayLayer,
	oa::Event inReady,
	oa::U32 inExternalQueueFamilyIndex,
	oa::Event* outInputConsumed)
{
	if (impl_->slots.empty()) return oa::Status::error("Video encoder has no asynchronous slots");
	if (impl_->compatibilityUploadReady) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Cannot submit an asynchronous job after an unmatched uploadInputRgba call");
	}

	EncodeSlot& slot = impl_->slots[impl_->submitSlot];
	if (slot.pending) {
		oa::EncodedVideoPacket completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(harvest_(slot, true, completed, ready));
		if (ready) {
			outReady.pushBack(oa::move(completed));
			--impl_->pendingSlots;
			impl_->harvestSlot = (impl_->harvestSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());
		}
	}
	OA_RETURN_IF_ERROR(uploadInputRgbaImage_(
		slot, inImage, inImageView, inFormat, inLayout,
		inVisibleWidth, inVisibleHeight, inColorSpace, inFullRange,
		inArrayLayer, inReady,
		inExternalQueueFamilyIndex));
	OA_RETURN_IF_ERROR(submitEncode_(slot, inPts));
	++impl_->pendingSlots;
	impl_->submitSlot = (impl_->submitSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());

	// The source image is consumed by conversion, not by the video queue. New
	// graph/capture callers receive that GPU completion and defer recycling;
	// compatibility callers retain the historical safe synchronous contract.
	if (outInputConsumed != nullptr) {
		*outInputConsumed = slot.inputTicket.completion();
	} else {
		OA_RETURN_IF_ERROR(slot.inputTicket.waitForSignal());
	}

	while (impl_->pendingSlots > 0U) {
		EncodeSlot& oldest = impl_->slots[impl_->harvestSlot];
		oa::EncodedVideoPacket completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(harvest_(oldest, false, completed, ready));
		if (not ready) break;
		outReady.pushBack(oa::move(completed));
		--impl_->pendingSlots;
		impl_->harvestSlot = (impl_->harvestSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());
	}
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::flush(oa::Vector<oa::EncodedVideoPacket>& outFrames)
{
	if (not impl_ or impl_->session.handle() == VK_NULL_HANDLE) {
		return oa::Status::ok();
	}
	while (impl_->pendingSlots > 0U) {
		EncodeSlot& oldest = impl_->slots[impl_->harvestSlot];
		oa::EncodedVideoPacket completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(harvest_(oldest, true, completed, ready));
		if (not ready) return oa::Status::error("pending encode job was not harvestable");
		outFrames.pushBack(oa::move(completed));
		--impl_->pendingSlots;
		impl_->harvestSlot = (impl_->harvestSlot + 1U) % static_cast<oa::U32>(impl_->slots.size());
	}
	return oa::Status::ok();
}


oa::Status oa::VideoEncoder::destroySlot_(EncodeSlot& inSlot)
{
	if (impl_->engine == nullptr) return oa::Status::ok();
	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	auto* allocator = static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(vkEngine).allocator);
	const oa::Status inputStatus = inSlot.inputTicket.wait();
	if (inSlot.rgbaSnapshot.buffer != VK_NULL_HANDLE) {
		oa::EngineResourceAccess::freeBuffer(vkEngine, inSlot.rgbaSnapshot);
	}
	if (inSlot.queryPool != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyQueryPool(device, inSlot.queryPool, nullptr);
	}
	if (inSlot.fence != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyFence(device, inSlot.fence, nullptr);
	}
	inSlot.bitstream.destroy();
	if (inSlot.inputBindlessRegistered) {
		oa::EngineBindlessAccess::get(vkEngine)
			.deregisterStorageImage(inSlot.inputYBindless);
		oa::EngineBindlessAccess::get(vkEngine)
			.deregisterStorageImage(inSlot.inputUvBindless);
	}
	if (inSlot.inputUvView != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyImageView(device, inSlot.inputUvView, nullptr);
	}
	if (inSlot.inputYView != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyImageView(device, inSlot.inputYView, nullptr);
	}
	if (inSlot.inputView != VK_NULL_HANDLE) {
		oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkDestroyImageView(device, inSlot.inputView, nullptr);
	}
	if (inSlot.inputImage != VK_NULL_HANDLE) {
		vma::destroyImage(allocator, inSlot.inputImage,
			static_cast<vma::Allocation>(inSlot.inputAllocation));
	}
	inSlot = {};
	return inputStatus;
}


oa::Status oa::VideoEncoder::close()
{
	if (not impl_ or impl_->engine == nullptr) {
		reset_();
		return oa::Status::ok();
	}
	auto& vkEngine = *impl_->engine;
	VkDevice device = static_cast<VkDevice>(oa::EngineDeviceAccess::get(vkEngine).device);
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() and not inStatus.isOk()) firstError = inStatus;
	};
	// Each encode submission owns a fence. Waiting only those fences preserves
	// unrelated work on a shared video queue and makes the lifetime edge exact.
	for (auto& slot : impl_->slots) {
		if (slot.pending and slot.fence != VK_NULL_HANDLE) {
			const VkResult result = oa::EngineDeviceAccess::get(vkEngine).deviceDispatch.vkWaitForFences(
				device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
			if (result != VK_SUCCESS) {
				retainError(oa::Status::error(
					oa::StatusCode::VulkanError,
					"video encoder fence completion failed"));
			}
			slot.pending = false;
		}
	}
	for (auto& slot : impl_->slots) retainError(destroySlot_(slot));

	impl_->session.destroy();
	impl_->sessionParameters.destroy();
	impl_->queue.destroy();
	impl_->dpb.destroy();
	reset_();
	return firstError;
}

// ──────────────────────────────────────────────────────────────────────
//                       Transcoder
// ──────────────────────────────────────────────────────────────────────

oa::Result<oa::VideoTranscoder> oa::VideoTranscoder::create(
	oa::Engine& inRt,
	const oa::VideoProfile& inDecodeProfile,
	const oa::VideoEncodeProfile& inEncodeProfile)
{
	if (inDecodeProfile.width != inEncodeProfile.width
		or inDecodeProfile.height != inEncodeProfile.height) {
		return oa::Status::error(oa::StatusCode::Unimplemented,
			"oa::VideoTranscoder currently requires matching decode/encode extents");
	}
	auto decoder = oa::VideoDecoder::create(inRt, inDecodeProfile);
	if (not decoder.isOk()) return decoder.getStatus();
	auto encoder = oa::VideoEncoder::create(inRt, inEncodeProfile);
	if (not encoder.isOk()) {
		return closeSessionAfterCreateFailure(
			*decoder, encoder.getStatus(), "video transcoder");
	}
	oa::VideoTranscoder transcoder;
	transcoder.rt_ = &inRt;
	transcoder.decoder_ = oa::move(*decoder);
	transcoder.encoder_ = oa::move(*encoder);
	transcoder.frameDurationUs_ = inEncodeProfile.frameRate > 0U
		? 1'000'000ULL / inEncodeProfile.frameRate : 0U;
	return oa::move(transcoder);
}


oa::Status oa::VideoTranscoder::transcodeFrame(
	const oa::Span<const oa::U8>& inBitstream,
	oa::EncodedVideoPacket& outFrame)
{
	if (rt_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::VideoTranscoder::transcodeFrame called on an uninitialized transcoder");
	}
	oa::VideoConversionOptions conversion;
	conversion.convertToRgb = true;
	conversion.preferHardwareYCbCr = true;
	conversion.filter = oa::Filter::Nearest;
	oa::VideoFrame rgba;
	OA_RETURN_IF_ERROR(oa::VideoDecoderInternal::decodeFrameWithConversion(
		decoder_, inBitstream, conversion, rgba));
	if (not rgba.shown) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"Decoded packet is a hidden reference frame and has no transcode output");
	}
	if (rgba.resource != oa::VideoFrameResource::Image or not rgba.isRgb
		or rgba.image == VK_NULL_HANDLE or rgba.imageView == VK_NULL_HANDLE) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Decoder did not produce an image-backed RGBA frame");
	}

	oa::Vector<oa::EncodedVideoPacket> ready;
	const oa::U64 pts = rgba.presentationTimestamp != 0U
		? rgba.presentationTimestamp : nextPtsUs_;
	OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::submitRgbaImage(encoder_,
		rgba.image, rgba.imageView, rgba.format, rgba.layout,
		rgba.width, rgba.height, pts, ready, rgba.colorSpace, rgba.fullRange,
		rgba.arrayLayer, rgba.ready,
		rgba.externalQueueFamilyIndex));
	OA_RETURN_IF_ERROR(encoder_.flush(ready));
	if (ready.size() != 1U) {
		return oa::Status::error(oa::StatusCode::Internal,
			"Synchronous transcoder expected exactly one encoded frame");
	}
	outFrame = oa::move(ready[0]);
	nextPtsUs_ = pts + frameDurationUs_;
	return oa::Status::ok();
}


oa::Status oa::VideoTranscoder::close()
{
	// The encoder may still consume decoder-owned images. Complete it first,
	// then release the decoder resources that back those images.
	oa::Status firstError = encoder_.close();
	const oa::Status decoderStatus = decoder_.close();
	if (firstError.isOk() and not decoderStatus.isOk()) {
		firstError = decoderStatus;
	}
	rt_ = nullptr;
	nextPtsUs_ = 0U;
	frameDurationUs_ = 0U;
	return firstError;
}
