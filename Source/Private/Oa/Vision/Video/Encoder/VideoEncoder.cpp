// OA Vision — Hardware Video Encoder Implementation
// VK_KHR_video_encode_queue + VK_KHR_video_encode_h264 / h265
//
// This file currently lands:
//   - QueryEncodeCapabilities (mirrors OaVideoDecoder::QueryDecodeCapabilities).
//   - IsCodecSupported / GetMaxBitrate / GetMaxWidth / GetMaxHeight (caps wrappers).
//   - Move ctor / move assign / dtor / Reset_ over the expanded state members.
//   - Create() — full session bring-up: VkVideoSessionKHR + memory binding
//     + VkVideoSessionParametersKHR (manufactured H.264 SPS/PPS — the
//     encoder writes these, the decoder parses them) + DPB image array
//     + NV12 input image + bitstream output buffer + command pool /
//     command buffer on the video-encode queue family.
//   - EncodeFrame / Flush — still return Unavailable (3g.3 — the actual
//     vkCmdEncodeVideoKHR + rate control + GOP / IDR insertion +
//     bitstream readback).
//   - Transcoder pipeline — Vulkan decode -> compute conversion -> Vulkan encode.

#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Runtime/OaVma.h>
#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"


// ──────────────────────────────────────────────────────────────────────
//                         File-local helpers
// ──────────────────────────────────────────────────────────────────────

namespace {

// Builds the VkVideoProfileInfoKHR chain for an encode profile.
// Same shape as the decoder's GetVideoProfileInfo helper but with encode
// codec-operation flags + encode-side profile-info structs.
VkVideoProfileInfoKHR BuildEncodeProfileInfo(
	OaVideoCodec InCodec,
	VkVideoEncodeH264ProfileInfoKHR& OutH264,
	VkVideoEncodeH265ProfileInfoKHR& OutH265)
{
	VkVideoProfileInfoKHR profile = {};
	profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
	switch (InCodec) {
		case OaVideoCodec::H264:
			OutH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR;
			OutH264.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
			profile.pNext = &OutH264;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
			break;
		case OaVideoCodec::H265:
			OutH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR;
			OutH265.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
			profile.pNext = &OutH265;
			profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
			break;
		case OaVideoCodec::AV1:
			// AV1 encode is not yet covered by a finalized KHR extension on
			// the desktop platforms we target; gated out at the cap-query
			// level rather than emitting an undefined profile chain here.
			break;
	}
	profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
	profile.lumaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	profile.chromaBitDepth    = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
	return profile;
}


// Attaches encode-side capability structs onto the VkVideoCapabilitiesKHR
// pNext chain so vkGetPhysicalDeviceVideoCapabilitiesKHR populates them.
void AttachEncodeCapabilityStructs(
	OaVideoCodec InCodec,
	VkVideoCapabilitiesKHR& InOutCaps,
	VkVideoEncodeCapabilitiesKHR& OutEncode,
	VkVideoEncodeH264CapabilitiesKHR& OutH264,
	VkVideoEncodeH265CapabilitiesKHR& OutH265)
{
	OutEncode.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
	InOutCaps.pNext = &OutEncode;
	switch (InCodec) {
		case OaVideoCodec::H264:
			OutH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR;
			OutEncode.pNext = &OutH264;
			break;
		case OaVideoCodec::H265:
			OutH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR;
			OutEncode.pNext = &OutH265;
			break;
		case OaVideoCodec::AV1:
			break;
	}
}


bool HasFormatWithUsage(
	const OaVec<VkVideoFormatPropertiesKHR>& InFormats,
	VkFormat InFormat,
	VkImageUsageFlags InUsage)
{
	for (const auto& format : InFormats) {
		if (format.format == InFormat && (format.imageUsageFlags & InUsage) == InUsage) {
			return true;
		}
	}
	return false;
}


// Manufactures a minimal valid H.264 SPS for an encode session.
// The encoder side WRITES this (vs the decoder which parses it from the
// bitstream); we only fill the fields a baseline encoder needs.
//
// Level is picked conservatively at 4.2 (covers 1080p60) — the cap query
// has already bounded width/height so the level is always over-provisioned
// rather than under-provisioned. Refine when we wire user-facing level
// selection.
StdVideoH264SequenceParameterSet BuildSpsForH264Encode(const OaVideoEncodeProfile& InProfile)
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
	const OaU32 widthMbs                     = (InProfile.Width  + 15U) / 16U;
	const OaU32 heightMbs                    = (InProfile.Height + 15U) / 16U;
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
StdVideoH264PictureParameterSet BuildPpsForH264Encode(const OaVideoEncodeProfile&)
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


// Storage for the pointer-bearing HEVC standard structures. Vulkan consumes
// these during vkCreateVideoSessionParametersKHR; keeping them together makes
// the ownership/lifetime relationship explicit and prevents dangling pNext-
// style payload pointers while the create call is assembled.
struct H265EncodeParameters {
	StdVideoH265ProfileTierLevel ProfileTierLevel = {};
	StdVideoH265DecPicBufMgr DpbManager = {};
	StdVideoH265ShortTermRefPicSet ShortTermRefPicSet = {};
	StdVideoH265VideoParameterSet Vps = {};
	StdVideoH265SequenceParameterSet Sps = {};
	StdVideoH265PictureParameterSet Pps = {};
};


H265EncodeParameters BuildParametersForH265Encode(
	const OaVideoEncodeProfile& InProfile,
	OaU32 InCodedWidth,
	OaU32 InCodedHeight,
	OaU32 InDpbSlots,
	VkVideoEncodeH265CtbSizeFlagsKHR InCtbSizes,
	VkVideoEncodeH265TransformBlockSizeFlagsKHR InTransformBlockSizes)
{
	H265EncodeParameters out;
	const OaU32 ctbLog2 = (InCtbSizes & VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_KHR) != 0U
		? 4U : ((InCtbSizes & VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_KHR) != 0U ? 5U : 6U);
	const OaU32 minTransformLog2 =
		(InTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_KHR) != 0U
			? 2U
			: ((InTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_KHR) != 0U
				? 3U : 4U);
	OaU32 maxTransformLog2 = minTransformLog2;
	if (ctbLog2 >= 5U
		and (InTransformBlockSizes & VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_KHR) != 0U) {
		maxTransformLog2 = 5U;
	} else if ((InTransformBlockSizes
		& VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_KHR) != 0U) {
		maxTransformLog2 = 4U;
	} else if ((InTransformBlockSizes
		& VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_KHR) != 0U) {
		maxTransformLog2 = 3U;
	}

	out.ProfileTierLevel.flags.general_progressive_source_flag = 1;
	out.ProfileTierLevel.flags.general_frame_only_constraint_flag = 1;
	out.ProfileTierLevel.general_profile_idc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
	// Main-tier level 4.1 covers the presentation and capture profiles OA
	// currently exposes (up to 1080p60 / 4K30 at practical bitrates).
	out.ProfileTierLevel.general_level_idc = STD_VIDEO_H265_LEVEL_IDC_4_1;

	out.DpbManager.max_dec_pic_buffering_minus1[0] = static_cast<OaU8>(
		InDpbSlots > 0U ? InDpbSlots - 1U : 0U);
	out.DpbManager.max_num_reorder_pics[0] = 0;
	out.DpbManager.max_latency_increase_plus1[0] = 0;

	// One flat L0 reference: the immediately preceding I/P picture.
	out.ShortTermRefPicSet.num_negative_pics = 1;
	out.ShortTermRefPicSet.num_positive_pics = 0;
	out.ShortTermRefPicSet.used_by_curr_pic_s0_flag = 1;
	out.ShortTermRefPicSet.delta_poc_s0_minus1[0] = 0;

	out.Vps.flags.vps_temporal_id_nesting_flag = 1;
	out.Vps.flags.vps_sub_layer_ordering_info_present_flag = 1;
	out.Vps.vps_video_parameter_set_id = 0;
	out.Vps.vps_max_sub_layers_minus1 = 0;
	out.Vps.pDecPicBufMgr = &out.DpbManager;
	out.Vps.pProfileTierLevel = &out.ProfileTierLevel;

	out.Sps.flags.sps_temporal_id_nesting_flag = 1;
	out.Sps.flags.sps_sub_layer_ordering_info_present_flag = 1;
	out.Sps.flags.amp_enabled_flag = 1;
	out.Sps.flags.sample_adaptive_offset_enabled_flag = 1;
	out.Sps.flags.conformance_window_flag =
		InCodedWidth != InProfile.Width || InCodedHeight != InProfile.Height;
	out.Sps.chroma_format_idc = STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
	out.Sps.pic_width_in_luma_samples = InCodedWidth;
	out.Sps.pic_height_in_luma_samples = InCodedHeight;
	out.Sps.sps_video_parameter_set_id = 0;
	out.Sps.sps_max_sub_layers_minus1 = 0;
	out.Sps.sps_seq_parameter_set_id = 0;
	out.Sps.bit_depth_luma_minus8 = 0;
	out.Sps.bit_depth_chroma_minus8 = 0;
	out.Sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	// MinCb stays 16 while CTB/transform geometry follows the device's HEVC
	// capability masks. Intel TGL, for example, exposes CTB 32/64 but not 16;
	// hard-coding 16 produces a valid-looking command that hangs the engine.
	constexpr OaU32 minCodingBlockLog2 = 4U;
	out.Sps.log2_min_luma_coding_block_size_minus3 =
		static_cast<OaU8>(minCodingBlockLog2 - 3U);
	out.Sps.log2_diff_max_min_luma_coding_block_size =
		static_cast<OaU8>(ctbLog2 - minCodingBlockLog2);
	out.Sps.log2_min_luma_transform_block_size_minus2 =
		static_cast<OaU8>(minTransformLog2 - 2U);
	out.Sps.log2_diff_max_min_luma_transform_block_size =
		static_cast<OaU8>(maxTransformLog2 - minTransformLog2);
	out.Sps.max_transform_hierarchy_depth_inter = static_cast<OaU8>(
		ctbLog2 - minTransformLog2 > 1U ? ctbLog2 - minTransformLog2 : 1U);
	out.Sps.max_transform_hierarchy_depth_intra = 3U;
	out.Sps.pcm_sample_bit_depth_luma_minus1 = 7U;
	out.Sps.pcm_sample_bit_depth_chroma_minus1 = 7U;
	out.Sps.log2_min_pcm_luma_coding_block_size_minus3 =
		static_cast<OaU8>(minCodingBlockLog2 - 3U);
	out.Sps.log2_diff_max_min_pcm_luma_coding_block_size =
		static_cast<OaU8>(ctbLog2 - minCodingBlockLog2);
	out.Sps.num_short_term_ref_pic_sets = 1;
	out.Sps.conf_win_right_offset = (InCodedWidth - InProfile.Width) / 2U;
	out.Sps.conf_win_bottom_offset = (InCodedHeight - InProfile.Height) / 2U;
	out.Sps.pProfileTierLevel = &out.ProfileTierLevel;
	out.Sps.pDecPicBufMgr = &out.DpbManager;
	out.Sps.pShortTermRefPicSet = &out.ShortTermRefPicSet;

	out.Pps.flags.cabac_init_present_flag = 1;
	out.Pps.flags.transform_skip_enabled_flag = 1;
	out.Pps.flags.cu_qp_delta_enabled_flag = 1;
	out.Pps.flags.pps_loop_filter_across_slices_enabled_flag = 1;
	out.Pps.flags.deblocking_filter_control_present_flag = 1;
	out.Pps.pps_pic_parameter_set_id = 0;
	out.Pps.pps_seq_parameter_set_id = 0;
	out.Pps.sps_video_parameter_set_id = 0;
	out.Pps.num_ref_idx_l0_default_active_minus1 = 0;
	out.Pps.num_ref_idx_l1_default_active_minus1 = 0;
	out.Pps.init_qp_minus26 = 0;
	out.Pps.log2_parallel_merge_level_minus2 = 0;
	return out;
}


OaStatus QueryVideoFormats(
	VkPhysicalDevice InPhys,
	const VkVideoProfileInfoKHR& InProfile,
	VkImageUsageFlags InUsage,
	OaVec<VkVideoFormatPropertiesKHR>& OutFormats)
{
	if (vkGetPhysicalDeviceVideoFormatPropertiesKHR == nullptr) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoFormatPropertiesKHR is not loaded");
	}

	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType        = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles    = &InProfile;

	VkPhysicalDeviceVideoFormatInfoKHR formatInfo = {};
	formatInfo.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
	formatInfo.pNext      = &profileList;
	formatInfo.imageUsage = InUsage;

	OaU32 formatCount = 0;
	VkResult result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(InPhys, &formatInfo, &formatCount, nullptr);
	if (result != VK_SUCCESS) {
		return OaStatus::Error("Failed to query Vulkan Video encode format count");
	}
	OutFormats.Resize(formatCount);
	for (auto& format : OutFormats) {
		format = {};
		format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
	}
	if (formatCount == 0) {
		return OaStatus::Ok();
	}
	result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(InPhys, &formatInfo, &formatCount, OutFormats.Data());
	if (result != VK_SUCCESS) {
		OutFormats.Resize(0);
		return OaStatus::Error("Failed to query Vulkan Video encode formats");
	}
	OutFormats.Resize(formatCount);
	return OaStatus::Ok();
}

}  // namespace


// ──────────────────────────────────────────────────────────────────────
//                       Capability querying
// ──────────────────────────────────────────────────────────────────────

OaResult<OaVideoEncodeCapabilities> OaVideoEncoder::QueryEncodeCapabilities(
	OaEngine& InRt,
	OaVideoCodec InCodec)
{
	if (InCodec != OaVideoCodec::H264 and InCodec != OaVideoCodec::H265) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"Requested Vulkan Video encode codec is not implemented in OA");
	}
	auto& vkEngine = InRt;
	const auto& hw = vkEngine.Device.Info.Hardware;
	const auto& sw = vkEngine.Device.Info.Software;
	// Mesa 26.1.4 advertises HEVC encode on Tiger Lake GT2, but both OA and
	// Khronos Vulkan-Video-Samples hang on the first vkCmdEncodeVideoKHR and
	// eventually lose the device. Never submit that known-broken path. Keep
	// this version-specific so a fixed Mesa release is automatically retested.
	const bool brokenTglH265Encode = InCodec == OaVideoCodec::H265
		and hw.VendorId == OaVkVendorIdIntel
		and hw.DeviceId == 0x9A49U
		and sw.DriverId == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA_KHR
		and sw.DriverInfo.find("Mesa 26.1.4") != OaString::npos;
	if (brokenTglH265Encode) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"H.265 encode is disabled on Intel TGL GT2 with Mesa 26.1.4: "
			"the Khronos reference encoder also hangs and loses the device");
	}

	if (not sw.HasVideoQueue or not sw.HasVideoEncodeQueue or not vkEngine.Device.Queues.HasVideoEncodeQueue) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Selected Vulkan device does not expose a video encode queue");
	}
	if (InCodec == OaVideoCodec::H264 and not sw.HasVideoEncodeH264) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"VK_KHR_video_encode_h264 is not enabled");
	}
	if (InCodec == OaVideoCodec::H265 and not sw.HasVideoEncodeH265) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"VK_KHR_video_encode_h265 is not enabled");
	}
	if (InCodec == OaVideoCodec::AV1) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"AV1 video encode is not supported in this build");
	}
	if (vkGetPhysicalDeviceVideoCapabilitiesKHR == nullptr) {
		return OaStatus::Error("vkGetPhysicalDeviceVideoCapabilitiesKHR is not loaded");
	}

	VkPhysicalDevice phys = static_cast<VkPhysicalDevice>(vkEngine.Device.PhysicalDevice);
	VkVideoEncodeH264ProfileInfoKHR h264 = {};
	VkVideoEncodeH265ProfileInfoKHR h265 = {};
	VkVideoProfileInfoKHR profile = BuildEncodeProfileInfo(InCodec, h264, h265);

	VkVideoCapabilitiesKHR caps           = {};
	caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
	VkVideoEncodeCapabilitiesKHR encCaps  = {};
	VkVideoEncodeH264CapabilitiesKHR h264Caps = {};
	VkVideoEncodeH265CapabilitiesKHR h265Caps = {};
	AttachEncodeCapabilityStructs(InCodec, caps, encCaps, h264Caps, h265Caps);

	VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &profile, &caps);
	if (result != VK_SUCCESS) {
		return OaStatus::Error("Vulkan Video encode profile is not supported");
	}

	OaVideoEncodeCapabilities out;
	out.Supported                          = true;
	out.MaxWidth                           = caps.maxCodedExtent.width;
	out.MaxHeight                          = caps.maxCodedExtent.height;
	out.MinWidth                           = caps.minCodedExtent.width;
	out.MinHeight                          = caps.minCodedExtent.height;
	out.PictureAccessGranularityWidth      = caps.pictureAccessGranularity.width  == 0U ? 1U : caps.pictureAccessGranularity.width;
	out.PictureAccessGranularityHeight     = caps.pictureAccessGranularity.height == 0U ? 1U : caps.pictureAccessGranularity.height;
	out.MaxDpbSlots                        = caps.maxDpbSlots;
	out.MaxActiveReferencePictures         = caps.maxActiveReferencePictures;
	out.MinBitstreamBufferOffsetAlignment  = caps.minBitstreamBufferOffsetAlignment;
	out.MinBitstreamBufferSizeAlignment    = caps.minBitstreamBufferSizeAlignment;
	out.StdHeaderVersion                   = caps.stdHeaderVersion;
	out.EncodeFlags                        = encCaps.flags;
	out.RateControlModes                   = encCaps.rateControlModes;
	out.MaxQualityLevels                   = encCaps.maxQualityLevels;
	out.MaxBitrate                         = encCaps.maxBitrate > OaU64{0xFFFFFFFFULL}
	                                            ? OaU32{0xFFFFFFFFU}
	                                            : static_cast<OaU32>(encCaps.maxBitrate);

	if (InCodec == OaVideoCodec::H264) {
		out.MaxH264SliceCount                = h264Caps.maxSliceCount;
		out.MaxH264PPictureL0ReferenceCount  = h264Caps.maxPPictureL0ReferenceCount;
		out.MaxH264BPictureL0ReferenceCount  = h264Caps.maxBPictureL0ReferenceCount;
		out.MaxH264L1ReferenceCount          = h264Caps.maxL1ReferenceCount;
	} else if (InCodec == OaVideoCodec::H265) {
		out.MaxH265SliceSegmentCount         = h265Caps.maxSliceSegmentCount;
		out.MaxH265PPictureL0ReferenceCount  = h265Caps.maxPPictureL0ReferenceCount;
		out.MaxH265BPictureL0ReferenceCount  = h265Caps.maxBPictureL0ReferenceCount;
		out.MaxH265L1ReferenceCount          = h265Caps.maxL1ReferenceCount;
		out.H265CtbSizes                     = h265Caps.ctbSizes;
		out.H265TransformBlockSizes          = h265Caps.transformBlockSizes;
		out.H265StdSyntaxFlags               = h265Caps.stdSyntaxFlags;
		out.MinH265Qp                        = h265Caps.minQp;
		out.MaxH265Qp                        = h265Caps.maxQp;
	}

	// Input-image format support — encoder consumes NV12 (the standard
	// 4:2:0 planar YCbCr format) and writes the DPB in the same format.
	const VkImageUsageFlags inputUsage = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
	const VkImageUsageFlags dpbUsage   = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR;
	OaStatus inputStatus = QueryVideoFormats(phys, profile, inputUsage, out.InputFormats);
	if (not inputStatus.IsOk()) { return inputStatus; }
	OaStatus dpbStatus = QueryVideoFormats(phys, profile, dpbUsage, out.DpbFormats);
	if (not dpbStatus.IsOk()) { return dpbStatus; }

	if (not HasFormatWithUsage(out.InputFormats, out.PictureFormat, inputUsage)) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Vulkan Video encoder does not expose NV12 input format support");
	}
	if (not HasFormatWithUsage(out.DpbFormats, out.ReferencePictureFormat, dpbUsage)) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Vulkan Video encoder does not expose NV12 DPB format support");
	}

	return out;
}


bool OaVideoEncoder::IsCodecSupported(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryEncodeCapabilities(InRt, InCodec);
	return caps.IsOk() and caps->Supported;
}


OaU32 OaVideoEncoder::GetMaxBitrate(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryEncodeCapabilities(InRt, InCodec);
	return caps.IsOk() ? caps->MaxBitrate : 0U;
}


OaU32 OaVideoEncoder::GetMaxWidth(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryEncodeCapabilities(InRt, InCodec);
	return caps.IsOk() ? caps->MaxWidth : 0U;
}


OaU32 OaVideoEncoder::GetMaxHeight(OaEngine& InRt, OaVideoCodec InCodec)
{
	auto caps = QueryEncodeCapabilities(InRt, InCodec);
	return caps.IsOk() ? caps->MaxHeight : 0U;
}


// ──────────────────────────────────────────────────────────────────────
//                       Lifecycle (ctor/dtor/move)
// ──────────────────────────────────────────────────────────────────────

void OaVideoEncoder::Reset_() noexcept
{
	Session_ = {};
	SessionParams_ = {};
	Queue_ = {};
	Dpb_ = {};
	DpbSlotCapacity_ = 0;
	Slots_.Clear();
	SubmitSlot_ = 0U;
	HarvestSlot_ = 0U;
	PendingSlots_ = 0U;
	CompatibilityUploadReady_ = false;
	CachedHeaders_.Clear();
	RateControlReset_ = false;
	QueryResultStatusSupported_ = false;
	RateControlMode_ = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
	CodedWidth_ = 0;
	CodedHeight_ = 0;
	MinBitstreamBufferOffsetAlignment_ = 1;
	MinBitstreamBufferSizeAlignment_ = 1;
	FrameCount_ = 0;
	LastKeyframeIndex_ = 0;
	GopSize_ = 30;
	CurrentGopFrame_ = 0;
	Rt_ = nullptr;
}


OaVideoEncoder::OaVideoEncoder(OaVideoEncoder&& InOther) noexcept
	: Profile_(InOther.Profile_)
	, Rt_(InOther.Rt_)
	, Session_(std::move(InOther.Session_))
	, SessionParams_(std::move(InOther.SessionParams_))
	, Queue_(std::move(InOther.Queue_))
	, Dpb_(std::move(InOther.Dpb_))
	, DpbSlotCapacity_(InOther.DpbSlotCapacity_)
	, Slots_(std::move(InOther.Slots_))
	, SubmitSlot_(InOther.SubmitSlot_)
	, HarvestSlot_(InOther.HarvestSlot_)
	, PendingSlots_(InOther.PendingSlots_)
	, CompatibilityUploadReady_(InOther.CompatibilityUploadReady_)
	, CachedHeaders_(std::move(InOther.CachedHeaders_))
	, RateControlReset_(InOther.RateControlReset_)
	, QueryResultStatusSupported_(InOther.QueryResultStatusSupported_)
	, RateControlMode_(InOther.RateControlMode_)
	, CodedWidth_(InOther.CodedWidth_)
	, CodedHeight_(InOther.CodedHeight_)
	, MinBitstreamBufferOffsetAlignment_(InOther.MinBitstreamBufferOffsetAlignment_)
	, MinBitstreamBufferSizeAlignment_(InOther.MinBitstreamBufferSizeAlignment_)
	, FrameCount_(InOther.FrameCount_)
	, LastKeyframeIndex_(InOther.LastKeyframeIndex_)
	, GopSize_(InOther.GopSize_)
	, CurrentGopFrame_(InOther.CurrentGopFrame_)
{
	InOther.Reset_();
}


OaVideoEncoder& OaVideoEncoder::operator=(OaVideoEncoder&& InOther) noexcept
{
	if (this != &InOther) {
		Destroy();
		Profile_ = InOther.Profile_;
		Rt_ = InOther.Rt_;
		Session_ = std::move(InOther.Session_);
		SessionParams_ = std::move(InOther.SessionParams_);
		Queue_ = std::move(InOther.Queue_);
		Dpb_ = std::move(InOther.Dpb_);
		DpbSlotCapacity_ = InOther.DpbSlotCapacity_;
		Slots_ = std::move(InOther.Slots_);
		SubmitSlot_ = InOther.SubmitSlot_;
		HarvestSlot_ = InOther.HarvestSlot_;
		PendingSlots_ = InOther.PendingSlots_;
		CompatibilityUploadReady_ = InOther.CompatibilityUploadReady_;
		CachedHeaders_ = std::move(InOther.CachedHeaders_);
		RateControlReset_ = InOther.RateControlReset_;
		QueryResultStatusSupported_ = InOther.QueryResultStatusSupported_;
		RateControlMode_ = InOther.RateControlMode_;
		CodedWidth_ = InOther.CodedWidth_;
		CodedHeight_ = InOther.CodedHeight_;
		MinBitstreamBufferOffsetAlignment_ = InOther.MinBitstreamBufferOffsetAlignment_;
		MinBitstreamBufferSizeAlignment_ = InOther.MinBitstreamBufferSizeAlignment_;
		FrameCount_ = InOther.FrameCount_;
		LastKeyframeIndex_ = InOther.LastKeyframeIndex_;
		GopSize_ = InOther.GopSize_;
		CurrentGopFrame_ = InOther.CurrentGopFrame_;
		InOther.Reset_();
	}
	return *this;
}


OaVideoEncoder::~OaVideoEncoder()
{
	Abandon_();
}


void OaVideoEncoder::Abandon_() noexcept
{
	if (Rt_ == nullptr) return;
	OaEngine* engine = Rt_;
	auto retired = OaMakeUniquePtr<OaVideoEncoder>(OaStdMove(*this));
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retired.Release(),
		&OaVideoEncoder::CompleteRetired_,
		&OaVideoEncoder::ReleaseRetired_);
}


OaStatus OaVideoEncoder::CompleteRetired_(void* InPayload)
{
	auto* encoder = static_cast<OaVideoEncoder*>(InPayload);
	return encoder ? encoder->Close() : OaStatus::Ok();
}


void OaVideoEncoder::ReleaseRetired_(void* InPayload)
{
	OaUniquePtr<OaVideoEncoder> encoder(
		static_cast<OaVideoEncoder*>(InPayload));
}


// ──────────────────────────────────────────────────────────────────────
//                       Create / EncodeFrame / Flush / Destroy
// ──────────────────────────────────────────────────────────────────────

OaResult<OaVideoEncoder> OaVideoEncoder::Create(
	OaEngine& InRt,
	const OaVideoEncodeProfile& InProfile)
{
	OaVideoEncoder encoder;
	encoder.Rt_      = &InRt;
	encoder.Profile_ = InProfile;

	auto capsResult = QueryEncodeCapabilities(InRt, InProfile.Codec);
	if (not capsResult.IsOk()) {
		return capsResult.GetStatus();
	}
	const OaVideoEncodeCapabilities& caps = *capsResult;
	if (InProfile.Codec == OaVideoCodec::H265) {
		OA_LOG_INFO(OaLogComponent::Core,
			"H.265 encoder caps: CTB=0x%x transform=0x%x syntax=0x%x P-L0=%u DPB=%u activeRefs=%u",
			static_cast<unsigned>(caps.H265CtbSizes),
			static_cast<unsigned>(caps.H265TransformBlockSizes),
			static_cast<unsigned>(caps.H265StdSyntaxFlags),
			static_cast<unsigned>(caps.MaxH265PPictureL0ReferenceCount),
			static_cast<unsigned>(caps.MaxDpbSlots),
			static_cast<unsigned>(caps.MaxActiveReferencePictures));
	}

	if (InProfile.Width  == 0U or InProfile.Height == 0U or
	    InProfile.Width  >  caps.MaxWidth or
	    InProfile.Height >  caps.MaxHeight) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video encode extent is unsupported");
	}
	if (InProfile.FrameRate == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video encode frame rate must be > 0");
	}
	if (InProfile.GopSize == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video encode GOP size must be > 0");
	}
	if (InProfile.ConstantQp > 51U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"H.264/H.265 constant QP must be in the range 0..51");
	}
	if (InProfile.Codec == OaVideoCodec::H265
		and (static_cast<OaI32>(InProfile.ConstantQp) < caps.MinH265Qp
			or static_cast<OaI32>(InProfile.ConstantQp) > caps.MaxH265Qp)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"H.265 constant QP is outside the device-supported range");
	}
	if (InProfile.QualityLevel >= caps.MaxQualityLevels) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Video encode quality level exceeds device capabilities");
	}
	if (InProfile.RateControl != OaVideoRateControl::ConstantQp and InProfile.Bitrate == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"CBR/VBR video encode requires a non-zero target bitrate");
	}
	if (caps.MaxBitrate > 0U and InProfile.Bitrate > caps.MaxBitrate) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Target video bitrate exceeds device capabilities");
	}
	if (InProfile.MaxBitrate > 0U and InProfile.MaxBitrate < InProfile.Bitrate) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"VBR maximum bitrate must be greater than or equal to target bitrate");
	}
	if (caps.MaxBitrate > 0U and InProfile.MaxBitrate > caps.MaxBitrate) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Maximum video bitrate exceeds device capabilities");
	}

	encoder.MinBitstreamBufferOffsetAlignment_ =
		caps.MinBitstreamBufferOffsetAlignment == 0U ? 1U : caps.MinBitstreamBufferOffsetAlignment;
	encoder.MinBitstreamBufferSizeAlignment_   =
		caps.MinBitstreamBufferSizeAlignment   == 0U ? 1U : caps.MinBitstreamBufferSizeAlignment;

	switch (InProfile.RateControl) {
		case OaVideoRateControl::ConstantQp:
			encoder.RateControlMode_ = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
			break;
		case OaVideoRateControl::Cbr:
			encoder.RateControlMode_ = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR;
			break;
		case OaVideoRateControl::Vbr:
			encoder.RateControlMode_ = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR;
			break;
	}
	if (encoder.RateControlMode_ != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
		and (caps.RateControlModes & encoder.RateControlMode_) == 0U) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Requested video rate-control mode is not supported by this device");
	}
	if (InProfile.RateControl == OaVideoRateControl::Cbr) {
		encoder.Profile_.MaxBitrate = InProfile.Bitrate;
	} else if (InProfile.RateControl == OaVideoRateControl::Vbr and InProfile.MaxBitrate == 0U) {
		OaU64 resolvedMaximum = static_cast<OaU64>(InProfile.Bitrate) * 2ULL;
		if (caps.MaxBitrate > 0U and resolvedMaximum > caps.MaxBitrate) {
			resolvedMaximum = caps.MaxBitrate;
		}
		encoder.Profile_.MaxBitrate = static_cast<OaU32>(resolvedMaximum);
	}

	// Align coded extent up to picture-access granularity + 16-MB granularity
	// (H.264 macroblock = 16x16). Same dance as VideoDecoder::Create — the
	// driver fails session create if coded extent isn't aligned.
	const OaU32 widthGranularity  = caps.PictureAccessGranularityWidth  > 16U
		? caps.PictureAccessGranularityWidth  : 16U;
	const OaU32 heightGranularity = caps.PictureAccessGranularityHeight > 16U
		? caps.PictureAccessGranularityHeight : 16U;
	const OaU32 alignedWidth      = ((InProfile.Width  + widthGranularity  - 1U) / widthGranularity)  * widthGranularity;
	const OaU32 alignedHeight     = ((InProfile.Height + heightGranularity - 1U) / heightGranularity) * heightGranularity;
	if (alignedWidth > caps.MaxWidth or alignedHeight > caps.MaxHeight) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"Aligned video encode extent exceeds caps");
	}
	encoder.CodedWidth_   = alignedWidth;
	encoder.CodedHeight_  = alignedHeight;

	auto& vkEngine = InRt;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);

	VkVideoEncodeH264ProfileInfoKHR h264 = {};
	VkVideoEncodeH265ProfileInfoKHR h265 = {};
	VkVideoProfileInfoKHR profile = BuildEncodeProfileInfo(InProfile.Codec, h264, h265);

	// ── 1. Create video session using OaVkVideoSession ─────────────────
	VkExtent2D codedExtent = { encoder.CodedWidth_, encoder.CodedHeight_ };

	// IPP cadence: 1 reference frame (the previous P/I) is enough for the
	// minimum useful encoder. Bump if/when we expose multi-ref or B-frames.
	const OaU32 requestedDpbSlots = InProfile.MaxDpbSlots == 0U ? 2U : InProfile.MaxDpbSlots;
	const OaU32 maxDpbSlots = requestedDpbSlots < caps.MaxDpbSlots
		? requestedDpbSlots : caps.MaxDpbSlots;
	if (maxDpbSlots == 0U) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Vulkan Video encoder reports zero DPB slots");
	}
	if (InProfile.GopSize > 1U and maxDpbSlots < 2U) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"P-frame encoding requires at least two DPB slots");
	}
	const OaU32 maxActiveReferences = maxDpbSlots < caps.MaxActiveReferencePictures
		? maxDpbSlots : caps.MaxActiveReferencePictures;
	const OaU32 finalMaxActiveReferences = maxActiveReferences == 0U and caps.MaxActiveReferencePictures > 0U ? 1U : maxActiveReferences;

	auto sessionResult = OaVkVideoSession::Create(
		vkEngine,
		profile,
		codedExtent,
		caps.PictureFormat,
		caps.ReferencePictureFormat,
		maxDpbSlots,
		finalMaxActiveReferences,
		InProfile.QualityLevel);
	if (!sessionResult.IsOk()) {
		return sessionResult.GetStatus();
	}
	encoder.Session_ = std::move(sessionResult.GetValue());
	encoder.DpbSlotCapacity_ = maxDpbSlots;

	// ── 2. Create video queue ───────────────────────────────────────────
	auto queueResult = OaVkVideoQueue::Create(vkEngine, OaVkVideoQueue::QueueType::Encode);
	if (!queueResult.IsOk()) {
		return queueResult.GetStatus();
	}
	encoder.Queue_ = std::move(queueResult.GetValue());

	// ── 3. Create shared DPB using OaVkVideoDpb ─────────────────────────
	// The encoder never samples its DPB. Requesting SAMPLED here makes the
	// profile/format combination invalid on drivers that expose encode-DPB but
	// not YCbCr sampling for the same video profile.
	const VkImageUsageFlags dpbUsage = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR;

	OaVkVideoDpb::CreateInfo dpbInfo = {};
	dpbInfo.Profile = profile;
	dpbInfo.Format = caps.ReferencePictureFormat;
	dpbInfo.CodedExtent = codedExtent;
	dpbInfo.MaxDpbSlots = maxDpbSlots;
	dpbInfo.Usage = dpbUsage;

	auto dpbResult = OaVkVideoDpb::Create(vkEngine, dpbInfo);
	if (!dpbResult.IsOk()) {
		return dpbResult.GetStatus();
	}
	encoder.Dpb_ = std::move(dpbResult.GetValue());

	// ── 4. Create codec session parameters ─────────────────────────────
	StdVideoH264SequenceParameterSet h264Sps = BuildSpsForH264Encode(InProfile);
	StdVideoH264PictureParameterSet  h264Pps = BuildPpsForH264Encode(InProfile);
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

	H265EncodeParameters h265Parameters = BuildParametersForH265Encode(
		InProfile, encoder.CodedWidth_, encoder.CodedHeight_, maxDpbSlots,
		caps.H265CtbSizes, caps.H265TransformBlockSizes);
	VkVideoEncodeH265SessionParametersAddInfoKHR h265AddInfo = {};
	h265AddInfo.sType       = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR;
	h265AddInfo.stdVPSCount = 1;
	h265AddInfo.pStdVPSs    = &h265Parameters.Vps;
	h265AddInfo.stdSPSCount = 1;
	h265AddInfo.pStdSPSs    = &h265Parameters.Sps;
	h265AddInfo.stdPPSCount = 1;
	h265AddInfo.pStdPPSs    = &h265Parameters.Pps;
	VkVideoEncodeH265SessionParametersCreateInfoKHR h265ParamsCreate = {};
	h265ParamsCreate.sType              = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR;
	h265ParamsCreate.maxStdVPSCount     = 1;
	h265ParamsCreate.maxStdSPSCount     = 1;
	h265ParamsCreate.maxStdPPSCount     = 1;
	h265ParamsCreate.pParametersAddInfo = &h265AddInfo;
	VkVideoEncodeQualityLevelInfoKHR qualityInfo = {};
	qualityInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR;
	qualityInfo.qualityLevel = InProfile.QualityLevel;
	h264ParamsCreate.pNext = &qualityInfo;
	h265ParamsCreate.pNext = &qualityInfo;

	VkVideoSessionParametersCreateInfoKHR paramsCreate = {};
	paramsCreate.sType              = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
	paramsCreate.pNext              = InProfile.Codec == OaVideoCodec::H264
		? static_cast<const void*>(&h264ParamsCreate)
		: static_cast<const void*>(&h265ParamsCreate);
	paramsCreate.videoSession       = encoder.Session_.Handle();
	paramsCreate.videoSessionParametersTemplate = VK_NULL_HANDLE;
	VkVideoSessionParametersKHR paramsHandle = VK_NULL_HANDLE;
	VkResult result = vkCreateVideoSessionParametersKHR(device, &paramsCreate, nullptr, &paramsHandle);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			InProfile.Codec == OaVideoCodec::H264
				? "vkCreateVideoSessionParametersKHR (encode H.264) failed"
				: "vkCreateVideoSessionParametersKHR (encode H.265) failed");
	}
	encoder.SessionParams_.Attach(vkEngine, paramsHandle);

	// ── 5. Per-job resources ───────────────────────────────────────────
	// Each slot owns the source image, command buffer, feedback query,
	// bitstream target, compute ticket and video fence. The video session and
	// DPB above stay shared; queue submission order preserves reference order.
	const OaU32 slotCount = InProfile.AsyncDepth == 0U ? 1U : InProfile.AsyncDepth;
	encoder.Slots_.Resize(slotCount);

	// NV12 input image (encoder source picture).
	// MUTABLE_FORMAT + EXTENDED_USAGE_BIT so we can create per-plane
	// R8 / R8G8 storage views into the multi-plane NV12 image — the
	// CvtRgbaToNv12 compute shader writes through those single-plane
	// views to fill Y and UV separately. The format list spells out
	// the three compatible formats (NV12 itself + the two plane
	// formats per Vulkan plane-format compatibility table).
	VkVideoProfileListInfoKHR profileList = {};
	profileList.sType        = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
	profileList.profileCount = 1;
	profileList.pProfiles    = &profile;

	const VkFormat kInputFormatList[3] = {
		caps.PictureFormat,            // VK_FORMAT_G8_B8R8_2PLANE_420_UNORM
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
	inputInfo.format        = caps.PictureFormat;
	inputInfo.extent        = { encoder.CodedWidth_, encoder.CodedHeight_, 1U };
	inputInfo.mipLevels     = 1;
	inputInfo.arrayLayers   = 1;
	inputInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
	inputInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
	inputInfo.usage         = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR
	                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT
	                      | VK_IMAGE_USAGE_STORAGE_BIT;
	const OaU32 queueFamilies[2] = {
		vkEngine.Device.Queues.ComputeQueueFamily,
		vkEngine.Device.Queues.VideoEncodeQueueFamily,
	};
	const bool separateQueueFamilies = queueFamilies[0] != queueFamilies[1];
	inputInfo.sharingMode = separateQueueFamilies
		? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
	inputInfo.queueFamilyIndexCount = separateQueueFamilies ? 2U : 0U;
	inputInfo.pQueueFamilyIndices = separateQueueFamilies ? queueFamilies : nullptr;
	inputInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	OaVmaAllocationCreateInfo imageAllocInfo = {};
	imageAllocInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	for (auto& slot : encoder.Slots_) {
		const OaU64 bitstreamSize = 4ULL * 1024ULL * 1024ULL;
		auto bitstreamResult = OaVkVideoBitstream::Create(
			vkEngine, bitstreamSize, OaVkVideoBitstream::Direction::Encoder,
			caps.MinBitstreamBufferOffsetAlignment == 0U ? 1U : caps.MinBitstreamBufferOffsetAlignment,
			caps.MinBitstreamBufferSizeAlignment == 0U ? 1U : caps.MinBitstreamBufferSizeAlignment,
			&profile);
		if (not bitstreamResult.IsOk()) {
			encoder.Destroy();
			return bitstreamResult.GetStatus();
		}
		slot.Bitstream = OaStdMove(*bitstreamResult);
		std::memset(slot.Bitstream.GetMappedPtr(), 0,
			static_cast<OaUsize>(slot.Bitstream.GetCapacity()));
		result = OaVmaFlushAllocation(
			static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
			static_cast<OaVmaAllocation>(slot.Bitstream.GetAllocation()),
			0, slot.Bitstream.GetCapacity());
		if (result != VK_SUCCESS) {
			encoder.Destroy();
			return OaStatus::Error(OaStatusCode::VulkanError,
				"Failed to initialize video encode bitstream buffer");
		}

		auto commandResult = encoder.Queue_.AllocateCommandBuffer();
		if (not commandResult.IsOk()) {
			encoder.Destroy();
			return commandResult.GetStatus();
		}
		slot.CommandBuffer = *commandResult;
		auto rgbaResult = vkEngine.AllocBuffer(
			static_cast<OaU64>(encoder.CodedWidth_) * encoder.CodedHeight_ * 4ULL);
		if (not rgbaResult.IsOk()) {
			encoder.Destroy();
			return rgbaResult.GetStatus();
		}
		slot.RgbaSnapshot = OaStdMove(*rgbaResult);

	OaVmaAllocation inputAllocation = VK_NULL_HANDLE;
	result = OaVmaCreateImage(
		static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
		&inputInfo, &imageAllocInfo, &slot.InputImage, &inputAllocation, nullptr);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"Failed to create Vulkan Video encode input image");
	}
	slot.InputAllocation = inputAllocation;

	// Combined NV12 view — used by vkCmdEncodeVideoKHR.
	VkImageViewCreateInfo inputViewInfo = {};
	inputViewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	VkImageViewUsageCreateInfo inputVideoUsage = {};
	inputVideoUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	inputVideoUsage.usage = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
	inputViewInfo.pNext = &inputVideoUsage;
	inputViewInfo.image    = slot.InputImage;
	inputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	inputViewInfo.format   = caps.PictureFormat;
	inputViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	inputViewInfo.subresourceRange.baseMipLevel   = 0;
	inputViewInfo.subresourceRange.levelCount     = 1;
	inputViewInfo.subresourceRange.baseArrayLayer = 0;
	inputViewInfo.subresourceRange.layerCount     = 1;
	result = vkCreateImageView(device, &inputViewInfo, nullptr, &slot.InputView);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"Failed to create Vulkan Video encode input image view");
	}

	// Y-plane view (R8_UNORM, aspect = PLANE_0) — storage-write target.
	VkImageViewUsageCreateInfo planeUsage = {};
	planeUsage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
	planeUsage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
	VkImageViewCreateInfo yViewInfo = {};
	yViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	yViewInfo.pNext                           = &planeUsage;
	yViewInfo.image                           = slot.InputImage;
	yViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	yViewInfo.format                          = VK_FORMAT_R8_UNORM;
	yViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_PLANE_0_BIT;
	yViewInfo.subresourceRange.baseMipLevel   = 0;
	yViewInfo.subresourceRange.levelCount     = 1;
	yViewInfo.subresourceRange.baseArrayLayer = 0;
	yViewInfo.subresourceRange.layerCount     = 1;
	result = vkCreateImageView(device, &yViewInfo, nullptr, &slot.InputYView);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"Failed to create Vulkan Video encode Y-plane storage view");
	}

	// UV-plane view (R8G8_UNORM, aspect = PLANE_1) — storage-write target.
	VkImageViewCreateInfo uvViewInfo = yViewInfo;
	uvViewInfo.format                      = VK_FORMAT_R8G8_UNORM;
	uvViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	result = vkCreateImageView(device, &uvViewInfo, nullptr, &slot.InputUvView);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"Failed to create Vulkan Video encode UV-plane storage view");
	}

	// Register both plane views in the bindless storage-image heap so
	// CvtRgbaToNv12 can address them via push-constant indices.
	slot.InputYBindless = vkEngine.Bindless.RegisterStorageImage(
		vkEngine.Device, slot.InputYView, VK_IMAGE_LAYOUT_GENERAL);
	slot.InputUvBindless = vkEngine.Bindless.RegisterStorageImage(
		vkEngine.Device, slot.InputUvView, VK_IMAGE_LAYOUT_GENERAL);
	slot.InputBindlessRegistered = true;

	// ── 8. Fence + feedback query pool ──────────────────────────────
	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	result = vkCreateFence(device, &fenceInfo, nullptr, &slot.Fence);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateFence for video encode failed");
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
	result = vkCreateQueryPool(device, &qpInfo, nullptr, &slot.QueryPool);
	if (result != VK_SUCCESS) {
		encoder.Destroy();
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateQueryPool for video encode feedback failed");
	}
	}

	// ── 6. Extract encoded codec parameter NAL bytes ─────────────────
	// Cache SPS/PPS for AVC or VPS/SPS/PPS for HEVC and prepend them to every
	// IDR. This gives both Annex-B output and the MP4 recorder a single,
	// self-describing keyframe contract.
	if (vkGetEncodedVideoSessionParametersKHR != nullptr) {
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
		getInfo.pNext                   = InProfile.Codec == OaVideoCodec::H264
			? static_cast<const void*>(&h264GetInfo)
			: static_cast<const void*>(&h265GetInfo);
		getInfo.videoSessionParameters  = encoder.SessionParams_.Handle();
		VkVideoEncodeSessionParametersFeedbackInfoKHR feedback = {};
		feedback.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR;
		OaUsize headerSize = 0;
		result = vkGetEncodedVideoSessionParametersKHR(
			device, &getInfo, &feedback, &headerSize, nullptr);
		if (result == VK_SUCCESS and headerSize > 0U) {
			encoder.CachedHeaders_.Resize(headerSize);
			result = vkGetEncodedVideoSessionParametersKHR(
				device, &getInfo, &feedback, &headerSize,
				encoder.CachedHeaders_.Data());
			if (result != VK_SUCCESS) {
				encoder.CachedHeaders_.Clear();
			} else {
				encoder.CachedHeaders_.Resize(headerSize);
			}
		}
		// A driver that returns 0 bytes here (some Mesa/RADV builds) is
		// not fatal — EncodeFrame just won't prepend headers. Bitstream
		// will need a sidecar SPS+PPS to decode, which is fine for the
		// MP4-muxer follow-up.
	}

	encoder.FrameCount_        = 0;
	encoder.LastKeyframeIndex_ = 0;
	encoder.RateControlReset_  = false;
	encoder.QueryResultStatusSupported_ = false;
	if (vkGetPhysicalDeviceQueueFamilyProperties2 != nullptr) {
		OaU32 familyCount = 0U;
		auto physicalDevice = static_cast<VkPhysicalDevice>(vkEngine.Device.PhysicalDevice);
		vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &familyCount, nullptr);
		OaVec<VkQueueFamilyProperties2> familyProps(familyCount);
		OaVec<VkQueueFamilyQueryResultStatusPropertiesKHR> statusProps(familyCount);
		for (OaU32 idx = 0U; idx < familyCount; ++idx) {
			statusProps[idx].sType =
				VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR;
			familyProps[idx].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
			familyProps[idx].pNext = &statusProps[idx];
		}
		vkGetPhysicalDeviceQueueFamilyProperties2(
			physicalDevice, &familyCount, familyProps.Data());
		const OaU32 family = vkEngine.Device.Queues.VideoEncodeQueueFamily;
		if (family < familyCount) {
			encoder.QueryResultStatusSupported_ =
				statusProps[family].queryResultStatusSupport == VK_TRUE;
		}
	}
	encoder.CurrentGopFrame_   = 0;
	encoder.GopSize_           = InProfile.GopSize;
	return OaResult<OaVideoEncoder>(std::move(encoder));
}


OaStatus OaVideoEncoder::UploadInputRgba(
	const OaVkBuffer& InRgba,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaYCbCrModel InColorSpace,
	bool InFullRange)
{
	if (Slots_.Empty()) {
		return OaStatus::Error("OaVideoEncoder::UploadInputRgba: encoder has no job slots");
	}
	if (PendingSlots_ != 0U) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Cannot mix synchronous encode calls with pending asynchronous jobs");
	}
	OA_RETURN_IF_ERROR(UploadInputRgba_(
		Slots_[0], InRgba, InVisibleWidth, InVisibleHeight, InColorSpace, InFullRange));
	CompatibilityUploadReady_ = true;
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::UploadInputRgba_(
	EncodeSlot& InSlot,
	const OaVkBuffer& InRgba,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaYCbCrModel InColorSpace,
	bool InFullRange)
{
	if (Rt_ == nullptr or Session_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Error("OaVideoEncoder::UploadInputRgba: encoder not initialized");
	}
	if (InRgba.Buffer == VK_NULL_HANDLE or InRgba.BindlessIndex == OA_BINDLESS_INVALID) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::UploadInputRgba: source buffer is not bindless-registered");
	}
	if (InVisibleWidth == 0U or InVisibleHeight == 0U
	    or InVisibleWidth > CodedWidth_ or InVisibleHeight > CodedHeight_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::UploadInputRgba: visible extent out of range");
	}
	const OaU64 visibleBytes = static_cast<OaU64>(InVisibleWidth) * InVisibleHeight * 4ULL;
	auto& vkEngine = *Rt_;
	if (InRgba.Allocation == nullptr or InRgba.AliasIdentity != nullptr
		or InRgba.IsImported()
		or InRgba.NodeIndex != 0U
		or InRgba.AllocatorIdentity != vkEngine.Allocator.Allocator) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"RGBA source must be a non-aliased allocation owned by the encoder engine");
	}
	if (InRgba.Size < visibleBytes or InSlot.RgbaSnapshot.Size < visibleBytes) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"RGBA source buffer is smaller than the visible frame");
	}
	OaEvent copyReady;
	if (InRgba.MappedPtr != nullptr and InSlot.RgbaSnapshot.MappedPtr != nullptr) {
		OA_RETURN_IF_ERROR(vkEngine.ReadbackBuffer(
			InRgba, 0U, InSlot.RgbaSnapshot.MappedPtr, visibleBytes));
		if (not vkEngine.Allocator.FlushHostBuffer(InSlot.RgbaSnapshot, 0U, visibleBytes)) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"Failed to flush asynchronous RGBA snapshot buffer");
		}
	} else {
		// Device-only producers need a stable copy before returning ownership to
		// the caller. This exact event wait is intentionally limited to that legacy
		// buffer path; mapped capture frames take the fully asynchronous route.
		auto copy = vkEngine.CopyBufferAsync(
			InRgba, InSlot.RgbaSnapshot, visibleBytes);
		if (not copy.IsOk()) return copy.GetStatus();
		OA_RETURN_IF_ERROR(copy->Wait());
		copyReady = *copy;
	}
	OaU32 colorSpace = 1U;  // BT.709 default
	switch (InColorSpace) {
		case OaYCbCrModel::BT709:  colorSpace = 1U; break;
		case OaYCbCrModel::BT2020: colorSpace = 2U; break;
		case OaYCbCrModel::Auto:
			colorSpace = (InVisibleWidth >= 1280U or InVisibleHeight >= 720U) ? 1U : 0U;
			break;
	}

	// Push struct only carries the user-data fields. OaVkImageDispatch::Run
	// prepends the resource indices (rgba_idx / y_idx / uv_idx) from the
	// binding[] array so the shader sees the full PushConstants struct as
	// declared in CvtRgbaToNv12.slang.
	struct alignas(4) Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
		OaU32 ColorSpace;
		OaU32 FullRange;
		OaU32 Pad;
	};
	Push push = {
		.Width       = InVisibleWidth,
		.Height      = InVisibleHeight,
		.CodedWidth  = CodedWidth_,
		.CodedHeight = CodedHeight_,
		.ColorSpace  = colorSpace,
		.FullRange   = InFullRange ? 1U : 0U,
		.Pad         = 0U,
	};

	OaVkImageDispatchBinding bindings[3] = {};
	// bindings[0] — RGBA8 packed source buffer, indexed via heap[] in the
	// shader. Run() picks up its BindlessIndex from the OaVkBuffer.
	bindings[0].Kind        = OaVkDescriptorKind::StorageBuffer;
	bindings[0].Binding     = 0;
	bindings[0].Buffer      = InSlot.RgbaSnapshot;
	// bindings[1] — Y-plane storage image view (R8_UNORM).
	bindings[1].Kind        = OaVkDescriptorKind::StorageImage;
	bindings[1].Binding     = 1;
	bindings[1].ImageView   = InSlot.InputYView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].Image       = InSlot.InputImage;
	bindings[1].InitialLayout = InSlot.InputInitialized
		? VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	bindings[1].FinalLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	bindings[1].AspectMask  = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
	// bindings[2] — UV-plane storage image view (R8G8_UNORM).
	bindings[2].Kind        = OaVkDescriptorKind::StorageImage;
	bindings[2].Binding     = 2;
	bindings[2].ImageView   = InSlot.InputUvView;
	bindings[2].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;

	const OaU32 groupsX = (CodedWidth_  + 15U) / 16U;
	const OaU32 groupsY = (CodedHeight_ + 15U) / 16U;
	const auto submitConversion = [&]() -> OaResult<OaVkImageDispatchTicket> {
		if (copyReady.IsValid()) {
			const OaVkTimelineWait wait = copyReady.TimelineWait();
			return OaVkImageDispatch::RunWithDependencyAsync(
				vkEngine,
				"CvtRgbaToNv12",
				OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
				&push,
				static_cast<OaU32>(sizeof(push)),
				groupsX,
				groupsY,
				1U,
				*wait.Semaphore,
				wait.Value);
		}
		return OaVkImageDispatch::RunAsync(
			vkEngine,
			"CvtRgbaToNv12",
			OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
			&push,
			static_cast<OaU32>(sizeof(push)),
			groupsX,
			groupsY,
			1U);
	};
	auto ticketResult = submitConversion();
	if (not ticketResult.IsOk()) return ticketResult.GetStatus();
	InSlot.InputTicket = OaStdMove(*ticketResult);
	InSlot.InputInitialized = true;
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::UploadInputRgbaImage_(
	EncodeSlot& InSlot,
	VkImage InImage,
	VkImageView InImageView,
	VkFormat InFormat,
	VkImageLayout InLayout,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaYCbCrModel InColorSpace,
	bool InFullRange,
	OaU32 InArrayLayer,
	OaEvent InReady,
	OaU32 InExternalQueueFamilyIndex)
{
	if (Rt_ == nullptr or Session_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Error("OaVideoEncoder::SubmitRgbaImage: encoder not initialized");
	}
	if (InImage == VK_NULL_HANDLE or InImageView == VK_NULL_HANDLE) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::SubmitRgbaImage requires a valid image and view");
	}
	if (InFormat != VK_FORMAT_R8G8B8A8_UNORM and InFormat != VK_FORMAT_B8G8R8A8_UNORM) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::SubmitRgbaImage supports RGBA8/BGRA8 UNORM images");
	}
	if (InLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::SubmitRgbaImage requires the producer's current image layout");
	}
	if (InVisibleWidth == 0U or InVisibleHeight == 0U
		or InVisibleWidth > CodedWidth_ or InVisibleHeight > CodedHeight_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaVideoEncoder::SubmitRgbaImage visible extent is out of range");
	}
	OaU32 colorSpace = 1U;
	switch (InColorSpace) {
		case OaYCbCrModel::BT709:  colorSpace = 1U; break;
		case OaYCbCrModel::BT2020: colorSpace = 2U; break;
		case OaYCbCrModel::Auto:
			colorSpace = (InVisibleWidth >= 1280U or InVisibleHeight >= 720U) ? 1U : 0U;
			break;
	}

	struct alignas(4) Push {
		OaU32 Width;
		OaU32 Height;
		OaU32 CodedWidth;
		OaU32 CodedHeight;
		OaU32 ColorSpace;
		OaU32 FullRange;
		OaU32 Pad;
	};
	const Push push = {
		.Width = InVisibleWidth,
		.Height = InVisibleHeight,
		.CodedWidth = CodedWidth_,
		.CodedHeight = CodedHeight_,
		.ColorSpace = colorSpace,
		.FullRange = InFullRange ? 1U : 0U,
		.Pad = 0U,
	};

	OaVkImageDispatchBinding bindings[3] = {};
	bindings[0].Kind = OaVkDescriptorKind::SampledImage;
	bindings[0].ImageView = InImageView;
	bindings[0].ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bindings[0].Image = InImage;
	bindings[0].InitialLayout = InLayout;
	bindings[0].FinalLayout = InLayout;
	bindings[0].AspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	bindings[0].BaseArrayLayer = InArrayLayer;
	bindings[0].InitialQueueFamilyIndex = InExternalQueueFamilyIndex;
	bindings[0].FinalQueueFamilyIndex = InExternalQueueFamilyIndex;

	bindings[1].Kind = OaVkDescriptorKind::StorageImage;
	bindings[1].ImageView = InSlot.InputYView;
	bindings[1].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;
	bindings[1].Image = InSlot.InputImage;
	bindings[1].InitialLayout = InSlot.InputInitialized
		? VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	bindings[1].FinalLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
	bindings[1].AspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;

	bindings[2].Kind = OaVkDescriptorKind::StorageImage;
	bindings[2].ImageView = InSlot.InputUvView;
	bindings[2].ImageLayout = VK_IMAGE_LAYOUT_GENERAL;

	auto& vkEngine = *Rt_;
	const OaU32 groupsX = (CodedWidth_ + 15U) / 16U;
	const OaU32 groupsY = (CodedHeight_ + 15U) / 16U;
	const OaVkTimelineWait ready = InReady.TimelineWait();
	OaResult<OaVkImageDispatchTicket> ticketResult = InReady.IsValid()
		? OaVkImageDispatch::RunWithDependencyAsync(
			vkEngine, "CvtRgbaImageToNv12",
			OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
			&push, static_cast<OaU32>(sizeof(push)), groupsX, groupsY, 1U,
			*ready.Semaphore, ready.Value)
		: OaVkImageDispatch::RunAsync(
			vkEngine, "CvtRgbaImageToNv12",
			OaSpan<const OaVkImageDispatchBinding>(bindings, 3),
			&push, static_cast<OaU32>(sizeof(push)), groupsX, groupsY, 1U);
	if (not ticketResult.IsOk()) return ticketResult.GetStatus();
	InSlot.InputTicket = OaStdMove(*ticketResult);
	InSlot.InputInitialized = true;
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::EncodeFrame(
	VkImage InImage,
	OaU64 InPts,
	OaEncodedFrame& OutFrame)
{
	(void)InImage;
	if (Slots_.Empty() or not CompatibilityUploadReady_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"EncodeFrame requires a preceding UploadInputRgba call");
	}
	OA_RETURN_IF_ERROR(SubmitEncode_(Slots_[0], InPts));
	CompatibilityUploadReady_ = false;
	bool ready = false;
	OA_RETURN_IF_ERROR(Harvest_(Slots_[0], true, OutFrame, ready));
	if (not ready) {
		return OaStatus::Error("Synchronous video encode did not produce a completed frame");
	}
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::SubmitEncode_(EncodeSlot& InSlot, OaU64 InPts)
{
	// Phase D: P-frame support with GOP structure
	// - Every GopSize_ frames, emit an IDR (keyframe)
	// - Otherwise, emit P-frames referencing the previous frame
	// - DPB slot 0 is used for the reference frame
	if (Rt_ == nullptr or Session_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Error("OaVideoEncoder::EncodeFrame: encoder not initialized");
	}
	if (vkCmdBeginVideoCodingKHR == nullptr or vkCmdEncodeVideoKHR == nullptr
	    or vkCmdEndVideoCodingKHR == nullptr or vkCmdControlVideoCodingKHR == nullptr) {
		return OaStatus::Error("Vulkan Video encode commands are not loaded");
	}

	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	VkQueue  queue  = static_cast<VkQueue>(vkEngine.Device.Queues.VideoEncodeQueue);
	if (queue == VK_NULL_HANDLE) {
		return OaStatus::Error("Video encode queue is null");
	}

	// Clear only the range written by this slot's preceding job. This makes a
	// zero-byte feedback fallback unambiguous without paying to clear the full
	// 4 MiB buffer every frame.
	if (InSlot.BitstreamDirtyEnd > 0U) {
		std::memset(InSlot.Bitstream.GetMappedPtr(), 0,
			static_cast<OaUsize>(InSlot.BitstreamDirtyEnd));
		VkResult flushResult = OaVmaFlushAllocation(
			static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
			static_cast<OaVmaAllocation>(InSlot.Bitstream.GetAllocation()),
			0, InSlot.BitstreamDirtyEnd);
		if (flushResult != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"Failed to clear reused video bitstream range");
		}
		InSlot.BitstreamDirtyEnd = 0U;
	}

	// ── Begin command buffer ────────────────────────────────────────
	VkResult result = vkResetCommandBuffer(InSlot.CommandBuffer, 0);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkResetCommandBuffer (encode) failed");
	}
	VkCommandBufferBeginInfo cbBegin = {};
	cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cbBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	result = vkBeginCommandBuffer(InSlot.CommandBuffer, &cbBegin);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkBeginCommandBuffer (encode) failed");
	}

	// ── Reset feedback query ────────────────────────────────────────
	vkCmdResetQueryPool(InSlot.CommandBuffer, InSlot.QueryPool, 0, 1);

	// ── Image layout transitions (synchronization2) ──────────────────
	// The KHR video-encode access/stage bits are only defined on the
	// synchronization2 path, so we use VkImageMemoryBarrier2 / vkCmd
	// PipelineBarrier2 here regardless of the rest of the codebase.
	// Input image: GENERAL (where CvtRgbaToNv12 left it) → VIDEO_ENCODE_SRC_KHR.
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
	inputBarrier.image                           = InSlot.InputImage;
	inputBarrier.subresourceRange.aspectMask     =
		  VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
	inputBarrier.subresourceRange.baseMipLevel   = 0;
	inputBarrier.subresourceRange.levelCount     = 1;
	inputBarrier.subresourceRange.baseArrayLayer = 0;
	inputBarrier.subresourceRange.layerCount     = 1;

	VkImageMemoryBarrier2 dpbBarrier = inputBarrier;
	// Queue submission order alone is not a memory dependency. Once the DPB
	// has been initialized, make the preceding encode's reference write
	// available to the next P-frame read/write. Without this dependency Intel
	// Xe intermittently completed feedback queries with zero output bytes.
	dpbBarrier.srcStageMask                    = RateControlReset_
		? VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR : VK_PIPELINE_STAGE_2_NONE;
	dpbBarrier.srcAccessMask                   = RateControlReset_
		? VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR
		: VK_ACCESS_2_NONE;
	dpbBarrier.dstStageMask                    = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
	dpbBarrier.dstAccessMask                   = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;
	dpbBarrier.oldLayout                       = RateControlReset_ ? VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
	dpbBarrier.newLayout                       = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR;
	dpbBarrier.image                           = Dpb_.GetImage();
	dpbBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	dpbBarrier.subresourceRange.layerCount     = DpbSlotCapacity_;

	const VkImageMemoryBarrier2 barriers[2] = { inputBarrier, dpbBarrier };
	VkDependencyInfo depInfo = {};
	depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	depInfo.imageMemoryBarrierCount  = 2;
	depInfo.pImageMemoryBarriers     = barriers;
	vkCmdPipelineBarrier2(InSlot.CommandBuffer, &depInfo);

	// ── Begin video coding ──────────────────────────────────────────
	const bool isIdr = (CurrentGopFrame_ == 0);
	const OaU32 referenceDepth = DpbSlotCapacity_ > 0U ? DpbSlotCapacity_ : 1U;
	const OaU32 currentDpbSlot = FrameCount_ % referenceDepth;
	const OaU32 previousDpbSlot = FrameCount_ == 0U
		? currentDpbSlot : (FrameCount_ - 1U) % referenceDepth;

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
		if (Profile_.Codec == OaVideoCodec::H264) {
			beginPreviousRefInfo.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_P;
			beginPreviousRefInfo.FrameNum = FrameCount_ - LastKeyframeIndex_ - 1U;
			beginPreviousRefInfo.PicOrderCnt = FrameCount_ - LastKeyframeIndex_ - 1U;
			beginPreviousH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
			beginPreviousH264.pStdReferenceInfo = &beginPreviousRefInfo;
			beginReferenceSlot.pNext = &beginPreviousH264;
		} else {
			beginPreviousH265RefInfo.pic_type = STD_VIDEO_H265_PICTURE_TYPE_P;
			beginPreviousH265RefInfo.PicOrderCntVal =
				static_cast<OaI32>(FrameCount_ - LastKeyframeIndex_ - 1U);
			beginPreviousH265RefInfo.TemporalId = 0;
			beginPreviousH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
			beginPreviousH265.pStdReferenceInfo = &beginPreviousH265RefInfo;
			beginReferenceSlot.pNext = &beginPreviousH265;
		}
		beginPreviousResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginPreviousResource.codedExtent = { CodedWidth_, CodedHeight_ };
		beginPreviousResource.baseArrayLayer = previousDpbSlot;
		beginPreviousResource.imageViewBinding = Dpb_.GetView();
		beginReferenceSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		beginReferenceSlot.slotIndex = static_cast<OaI32>(previousDpbSlot);
		beginReferenceSlot.pPictureResource = &beginPreviousResource;
	}
	if (Profile_.Codec == OaVideoCodec::H264) {
		// The picture named by pSetupReferenceSlot must be associated with the
		// coding scope too. Bind the target layer as an inactive resource; the
		// encode command activates the same resource at currentDpbSlot.
		beginCurrentH264RefInfo.primary_pic_type = isIdr
			? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
		beginCurrentH264RefInfo.FrameNum = isIdr
			? 0U : FrameCount_ - LastKeyframeIndex_;
		beginCurrentH264RefInfo.PicOrderCnt = isIdr
			? 0 : static_cast<OaI32>(FrameCount_ - LastKeyframeIndex_);
		beginCurrentH264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR;
		beginCurrentH264.pStdReferenceInfo = &beginCurrentH264RefInfo;
		beginCurrentResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginCurrentResource.codedExtent = { CodedWidth_, CodedHeight_ };
		beginCurrentResource.baseArrayLayer = currentDpbSlot;
		beginCurrentResource.imageViewBinding = Dpb_.GetView();
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
			? 0 : static_cast<OaI32>(FrameCount_ - LastKeyframeIndex_);
		beginCurrentH265RefInfo.TemporalId = 0;
		beginCurrentH265.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR;
		beginCurrentH265.pStdReferenceInfo = &beginCurrentH265RefInfo;
		beginCurrentResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		beginCurrentResource.codedExtent = { CodedWidth_, CodedHeight_ };
		beginCurrentResource.baseArrayLayer = currentDpbSlot;
		beginCurrentResource.imageViewBinding = Dpb_.GetView();
		beginH265Slots[0].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		beginH265Slots[0].pNext = &beginCurrentH265;
		beginH265Slots[0].slotIndex = -1;
		beginH265Slots[0].pPictureResource = &beginCurrentResource;
		if (not isIdr) beginH265Slots[1] = beginReferenceSlot;
	}

	// The coding scope must declare the active rate-control state. In
	// particular, constant-QP uses DISABLED rather than the session's DEFAULT;
	// omitting this chain makes non-zero slice QP invalid after the first scope.
	VkVideoEncodeRateControlInfoKHR beginRcInfo = {};
	VkVideoEncodeRateControlLayerInfoKHR beginRcLayer = {};
	VkVideoEncodeH264RateControlInfoKHR beginH264Rc = {};
	VkVideoEncodeH264RateControlLayerInfoKHR beginH264Layer = {};
	VkVideoEncodeH265RateControlInfoKHR beginH265Rc = {};
	VkVideoEncodeH265RateControlLayerInfoKHR beginH265Layer = {};
	const bool beginRcEnabled =
		RateControlMode_ != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
		and Profile_.Bitrate > 0U;
	beginRcInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR;
	beginRcInfo.rateControlMode = not RateControlReset_
		? VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR
		: (beginRcEnabled ? RateControlMode_
			: VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR);
	if (beginRcEnabled and RateControlReset_) {
		beginRcLayer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR;
		beginRcLayer.averageBitrate = Profile_.Bitrate;
		beginRcLayer.maxBitrate = RateControlMode_ == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR
			? Profile_.Bitrate : Profile_.MaxBitrate;
		beginRcLayer.frameRateNumerator = Profile_.FrameRate;
		beginRcLayer.frameRateDenominator = 1U;
		if (Profile_.Codec == OaVideoCodec::H264) {
			beginH264Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR;
			beginRcLayer.pNext = &beginH264Layer;
			beginH264Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR;
			beginH264Rc.gopFrameCount = Profile_.GopSize;
			beginH264Rc.idrPeriod = Profile_.GopSize;
			beginH264Rc.consecutiveBFrameCount = Profile_.MaxBFrames;
			beginH264Rc.temporalLayerCount = 1U;
			beginRcInfo.pNext = &beginH264Rc;
		} else {
			beginH265Layer.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR;
			beginRcLayer.pNext = &beginH265Layer;
			beginH265Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR;
			beginH265Rc.gopFrameCount = Profile_.GopSize;
			beginH265Rc.idrPeriod = Profile_.GopSize;
			beginH265Rc.consecutiveBFrameCount = Profile_.MaxBFrames;
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
	beginCoding.videoSession           = Session_.Handle();
	beginCoding.videoSessionParameters = SessionParams_.Handle();
	beginCoding.referenceSlotCount = isIdr ? 1U : 2U;
	beginCoding.pReferenceSlots = Profile_.Codec == OaVideoCodec::H265
		? beginH265Slots : beginH264Slots;
	vkCmdBeginVideoCodingKHR(InSlot.CommandBuffer, &beginCoding);

	// ── First-time: RESET the encoder + program rate control ───────────
	// constantQp on the slice header is ignored once we attach a non-DISABLED
	// rate-control info chain; we keep it as a sensible fallback for drivers
	// that downgrade us to DISABLED mid-session.
	if (not RateControlReset_) {
		VkVideoCodingControlInfoKHR controlInfo = {};
		controlInfo.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
		controlInfo.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;

		VkVideoEncodeRateControlInfoKHR        rcInfo       = {};
		VkVideoEncodeRateControlLayerInfoKHR   layer        = {};
		VkVideoEncodeH264RateControlInfoKHR    h264Rc       = {};
		VkVideoEncodeH264RateControlLayerInfoKHR h264Layer  = {};
		VkVideoEncodeH265RateControlInfoKHR    h265Rc       = {};
		VkVideoEncodeH265RateControlLayerInfoKHR h265Layer  = {};

		const bool rcEnabled = RateControlMode_ != VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR
			and Profile_.Bitrate > 0U;
		// RESET returns rate control to DEFAULT, not DISABLED. Program the
		// selected mode explicitly even for constant-QP so slice QP is legal.
		rcInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR;
		rcInfo.rateControlMode = rcEnabled
			? RateControlMode_ : VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
		if (rcEnabled) {
			if (Profile_.Codec == OaVideoCodec::H264) {
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
			layer.averageBitrate         = Profile_.Bitrate;
			layer.maxBitrate = RateControlMode_ == VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR
				? Profile_.Bitrate : Profile_.MaxBitrate;
			layer.frameRateNumerator     = Profile_.FrameRate;
			layer.frameRateDenominator   = 1U;

			if (Profile_.Codec == OaVideoCodec::H264) {
				h264Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR;
				h264Rc.gopFrameCount = Profile_.GopSize;
				h264Rc.idrPeriod = Profile_.GopSize;
				h264Rc.consecutiveBFrameCount = Profile_.MaxBFrames;
				h264Rc.temporalLayerCount = 1U;
				rcInfo.pNext = &h264Rc;
			} else {
				h265Rc.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR;
				h265Rc.gopFrameCount = Profile_.GopSize;
				h265Rc.idrPeriod = Profile_.GopSize;
				h265Rc.consecutiveBFrameCount = Profile_.MaxBFrames;
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

		vkCmdControlVideoCodingKHR(InSlot.CommandBuffer, &controlInfo);
		RateControlReset_ = true;
	}

	vkCmdBeginQuery(InSlot.CommandBuffer, InSlot.QueryPool, 0, 0);

	if (Profile_.Codec == OaVideoCodec::H264) {
	// ── Build the H.264 picture / slice info ────────────────────────
	// Phase D — IDR + P cadence. Profile_.MaxBFrames is propagated into
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
		refListsInfo.RefPicList0[0] = static_cast<OaU8>(previousDpbSlot);
	}

	StdVideoEncodeH264PictureInfo h264PicInfo = {};
	h264PicInfo.flags.IdrPicFlag                       = isIdr ? 1 : 0;
	h264PicInfo.flags.is_reference                     = 1;
	h264PicInfo.flags.no_output_of_prior_pics_flag     = isIdr ? 1 : 0;
	h264PicInfo.flags.long_term_reference_flag         = 0;
	h264PicInfo.flags.adaptive_ref_pic_marking_mode_flag = 0;
	h264PicInfo.seq_parameter_set_id                   = 0;
	h264PicInfo.pic_parameter_set_id                   = 0;
	h264PicInfo.idr_pic_id                             = static_cast<uint16_t>(FrameCount_ & 0xFFFFU);
	h264PicInfo.primary_pic_type                       = isIdr ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
	h264PicInfo.frame_num                              = isIdr ? 0 : (FrameCount_ - LastKeyframeIndex_);
	h264PicInfo.PicOrderCnt                            = isIdr ? 0 : (FrameCount_ - LastKeyframeIndex_);
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
	naluSlice.constantQp       = static_cast<OaI32>(Profile_.ConstantQp);
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
	setupResource.codedExtent          = { CodedWidth_, CodedHeight_ };
	setupResource.baseArrayLayer       = currentDpbSlot;
	setupResource.imageViewBinding     = Dpb_.GetView();
	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType            = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext            = &currentSlotH264;
	setupSlot.slotIndex        = static_cast<OaI32>(currentDpbSlot);
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR srcResource = {};
	srcResource.sType            = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	srcResource.codedOffset      = { 0, 0 };
	srcResource.codedExtent      = { CodedWidth_, CodedHeight_ };
	srcResource.baseArrayLayer   = 0;
	srcResource.imageViewBinding = InSlot.InputView;

	VkVideoEncodeInfoKHR encodeInfo = {};
	encodeInfo.sType                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR;
	encodeInfo.pNext                = &h264Picture;
	encodeInfo.dstBuffer            = InSlot.Bitstream.GetBuffer();
	encodeInfo.dstBufferOffset      = 0;
	encodeInfo.dstBufferRange       = static_cast<OaU32>(InSlot.Bitstream.GetCapacity());
	encodeInfo.srcPictureResource   = srcResource;
	encodeInfo.pSetupReferenceSlot  = &setupSlot;
	
	// Phase D: P-frames need reference slots
	StdVideoEncodeH264ReferenceInfo previousRefInfo = {};
	previousRefInfo.primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_P;
	previousRefInfo.FrameNum = FrameCount_ - LastKeyframeIndex_ - 1U;
	previousRefInfo.PicOrderCnt = FrameCount_ - LastKeyframeIndex_ - 1U;
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
		refSlots[0].slotIndex        = static_cast<OaI32>(previousDpbSlot);
		refSlots[0].pPictureResource = &previousResource;
		encodeInfo.referenceSlotCount = 1;
		encodeInfo.pReferenceSlots    = refSlots;
	} else {
		// IDR: no reference slots
		encodeInfo.referenceSlotCount = 0;
		encodeInfo.pReferenceSlots    = nullptr;
	}

	vkCmdEncodeVideoKHR(InSlot.CommandBuffer, &encodeInfo);
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
			refListsInfo.RefPicList0[0] = static_cast<OaU8>(previousDpbSlot);
		}

		const OaI32 picturePoc = isIdr
			? 0 : static_cast<OaI32>(FrameCount_ - LastKeyframeIndex_);
		StdVideoEncodeH265PictureInfo h265PicInfo = {};
		h265PicInfo.flags.is_reference = 1;
		h265PicInfo.flags.IrapPicFlag = isIdr ? 1 : 0;
		h265PicInfo.flags.pic_output_flag = 1;
		h265PicInfo.flags.no_output_of_prior_pics_flag =
			isIdr and FrameCount_ != 0U ? 1 : 0;
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
		naluSlice.constantQp = static_cast<OaI32>(Profile_.ConstantQp);
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
		setupResource.codedExtent = { CodedWidth_, CodedHeight_ };
		setupResource.baseArrayLayer = currentDpbSlot;
		setupResource.imageViewBinding = Dpb_.GetView();
		VkVideoReferenceSlotInfoKHR setupSlot = {};
		setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
		setupSlot.pNext = &currentSlotH265;
		setupSlot.slotIndex = static_cast<OaI32>(currentDpbSlot);
		setupSlot.pPictureResource = &setupResource;

		VkVideoPictureResourceInfoKHR srcResource = {};
		srcResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
		srcResource.codedExtent = { CodedWidth_, CodedHeight_ };
		srcResource.imageViewBinding = InSlot.InputView;

		VkVideoEncodeInfoKHR encodeInfo = {};
		encodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR;
		encodeInfo.pNext = &h265Picture;
		encodeInfo.dstBuffer = InSlot.Bitstream.GetBuffer();
		encodeInfo.dstBufferOffset = 0;
		encodeInfo.dstBufferRange = InSlot.Bitstream.GetCapacity();
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
			referenceSlot.slotIndex = static_cast<OaI32>(previousDpbSlot);
			referenceSlot.pPictureResource = &previousResource;
			encodeInfo.referenceSlotCount = 1;
			encodeInfo.pReferenceSlots = &referenceSlot;
		}
		vkCmdEncodeVideoKHR(InSlot.CommandBuffer, &encodeInfo);
	}
	vkCmdEndQuery(InSlot.CommandBuffer, InSlot.QueryPool, 0);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	vkCmdEndVideoCodingKHR(InSlot.CommandBuffer, &endCoding);

	result = vkEndCommandBuffer(InSlot.CommandBuffer);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkEndCommandBuffer (encode) failed");
	}

	// ── Submit without a host wait ──────────────────────────────────
	vkResetFences(device, 1, &InSlot.Fence);
	VkTimelineSemaphoreSubmitInfo timeline = {};
	VkSemaphore waitSemaphore = VK_NULL_HANDLE;
	OaU64 waitValue = 0U;
	// VkSubmitInfo uses the legacy 32-bit stage mask; not every Vulkan header
	// exposes a legacy VIDEO_ENCODE enumerant. ALL_COMMANDS safely gates the
	// video command buffer while the timeline dependency resolves.
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	if (InSlot.InputTicket.IsValid()) {
		waitSemaphore = static_cast<VkSemaphore>(InSlot.InputTicket.Semaphore().Semaphore);
		waitValue = InSlot.InputTicket.Value();
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
	submit.pCommandBuffers    = &InSlot.CommandBuffer;
	result = vkQueueSubmit(queue, 1, &submit, InSlot.Fence);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (encode) failed");
	}
	InSlot.PresentationTimestamp = InPts;
	InSlot.IsKeyframe = isIdr;
	InSlot.Pending = true;

	// GOP state advances at queue submission time. Vulkan queue order makes the
	// shared reference slot deterministic even while older jobs are unharvested.
	if (isIdr) {
		LastKeyframeIndex_ = FrameCount_;
		CurrentGopFrame_ = GopSize_ > 1U ? 1U : 0U;
	} else {
		++CurrentGopFrame_;
		if (CurrentGopFrame_ >= GopSize_) CurrentGopFrame_ = 0U;
	}
	++FrameCount_;
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::Harvest_(
	EncodeSlot& InSlot,
	bool InWait,
	OaEncodedFrame& OutFrame,
	bool& OutReady)
{
	OutReady = false;
	if (not InSlot.Pending) return OaStatus::Ok();
	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	VkResult result = VK_SUCCESS;
	if (InWait) {
		result = vkWaitForFences(
			device, 1, &InSlot.Fence, VK_TRUE, OaU64{1'000'000'000ULL} * 5ULL);
		if (result != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				OaString("vkWaitForFences (encode job) failed: VkResult=")
					+ std::to_string(static_cast<int>(result)));
		}
	} else {
		result = vkGetFenceStatus(device, InSlot.Fence);
		if (result == VK_NOT_READY) return OaStatus::Ok();
		if (result != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				OaString("vkGetFenceStatus (encode job) failed: VkResult=")
					+ std::to_string(static_cast<int>(result)));
		}
	}

	// The video fence implies its timeline wait was satisfied, so the compute
	// conversion ticket can now safely release its stream and bindless slots.
	OA_RETURN_IF_ERROR(InSlot.InputTicket.Wait());

	struct Feedback {
		OaU32 BitstreamStartOffset;
		OaU32 BitstreamBytesWritten;
	};
	Feedback fb = {};
	VkQueryResultStatusKHR queryStatus = VK_QUERY_RESULT_STATUS_COMPLETE_KHR;
	if (QueryResultStatusSupported_) {
		struct FeedbackWithStatus {
			Feedback Values;
			VkQueryResultStatusKHR Status;
		};
		FeedbackWithStatus resultWithStatus = {};
		result = vkGetQueryPoolResults(
			device, InSlot.QueryPool, 0, 1,
			sizeof(resultWithStatus), &resultWithStatus,
			sizeof(resultWithStatus),
			VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
		fb = resultWithStatus.Values;
		queryStatus = resultWithStatus.Status;
	} else {
		result = vkGetQueryPoolResults(
			device, InSlot.QueryPool, 0, 1,
			sizeof(fb), &fb,
			sizeof(fb),
			VK_QUERY_RESULT_WAIT_BIT);
	}
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkGetQueryPoolResults (encode feedback) failed");
	}
	if (queryStatus != VK_QUERY_RESULT_STATUS_COMPLETE_KHR) {
		return OaStatus::Error(
			"Vulkan video encode query failed for PTS "
			+ OaToString(static_cast<OaI64>(InSlot.PresentationTimestamp))
			+ " (status=" + OaToString(static_cast<OaI64>(queryStatus)) + ")");
	}
	if (fb.BitstreamBytesWritten == 0U) {
		(void)OaVmaInvalidateAllocation(
			static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
			static_cast<OaVmaAllocation>(InSlot.Bitstream.GetAllocation()),
			0, InSlot.Bitstream.GetCapacity());
		const auto* data = static_cast<const OaU8*>(InSlot.Bitstream.GetMappedPtr());
		OaU64 nonzeroExtent = 0U;
		for (OaU64 idx = InSlot.Bitstream.GetCapacity(); idx > 0U; --idx) {
			if (data[idx - 1U] != 0U) { nonzeroExtent = idx; break; }
		}
		OaU64 annexBStart = nonzeroExtent;
		for (OaU64 idx = 0U; idx + 3U < nonzeroExtent; ++idx) {
			if (data[idx] == 0U and data[idx + 1U] == 0U
				and ((data[idx + 2U] == 1U)
					or (data[idx + 2U] == 0U and data[idx + 3U] == 1U))) {
				annexBStart = idx;
				break;
			}
		}
		if (annexBStart >= nonzeroExtent) {
			return OaStatus::Error(
				"Vulkan video encoder produced neither feedback nor Annex-B payload for PTS "
				+ OaToString(static_cast<OaI64>(InSlot.PresentationTimestamp)));
		}
		fb.BitstreamStartOffset = static_cast<OaU32>(annexBStart);
		fb.BitstreamBytesWritten = static_cast<OaU32>(nonzeroExtent - annexBStart);
		OA_LOG_WARN(OaLogComponent::Core,
			"Video encode feedback returned zero bytes for PTS %llu; recovered %u Annex-B bytes",
			static_cast<unsigned long long>(InSlot.PresentationTimestamp),
			static_cast<unsigned>(fb.BitstreamBytesWritten));
	}

	// The encoder writes this GPU_TO_CPU allocation. Invalidate non-coherent
	// memory before reading its persistent mapping; coherent heaps make this a
	// no-op.
	result = OaVmaInvalidateAllocation(
		static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator),
		static_cast<OaVmaAllocation>(InSlot.Bitstream.GetAllocation()),
		fb.BitstreamStartOffset, fb.BitstreamBytesWritten);
	if (result != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"Failed to invalidate encoded video bitstream memory");
	}
	InSlot.BitstreamDirtyEnd = static_cast<OaU64>(fb.BitstreamStartOffset)
		+ fb.BitstreamBytesWritten;

	// ── Pack OaEncodedFrame ─────────────────────────────────────────
	const OaU8* slicePtr   = static_cast<const OaU8*>(InSlot.Bitstream.GetMappedPtr()) + fb.BitstreamStartOffset;
	const OaUsize sliceLen = fb.BitstreamBytesWritten;
	const OaUsize hdrLen   = InSlot.IsKeyframe ? CachedHeaders_.Size() : 0U;
	OutFrame.Bitstream.Resize(hdrLen + sliceLen);
	if (hdrLen > 0U) {
		std::memcpy(OutFrame.Bitstream.Data(), CachedHeaders_.Data(), hdrLen);
	}
	std::memcpy(OutFrame.Bitstream.Data() + hdrLen, slicePtr, sliceLen);
	OutFrame.PresentationTimestamp = InSlot.PresentationTimestamp;
	OutFrame.IsKeyframe            = InSlot.IsKeyframe;
	OutFrame.FrameSize             = static_cast<OaU32>(hdrLen + sliceLen);
	InSlot.Pending = false;
	OutReady = true;
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::SubmitRgba(
	const OaVkBuffer& InRgba,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaU64 InPts,
	OaVec<OaEncodedFrame>& OutReady,
	OaYCbCrModel InColorSpace,
	bool InFullRange)
{
	if (Slots_.Empty()) return OaStatus::Error("Video encoder has no asynchronous slots");
	if (CompatibilityUploadReady_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Cannot submit an asynchronous job after an unmatched UploadInputRgba call");
	}

	// Ring wrap: the selected slot is the oldest submitted job.
	EncodeSlot& slot = Slots_[SubmitSlot_];
	if (slot.Pending) {
		OaEncodedFrame completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(Harvest_(slot, true, completed, ready));
		if (ready) {
			OutReady.PushBack(OaStdMove(completed));
			--PendingSlots_;
			HarvestSlot_ = (HarvestSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());
		}
	}
	OA_RETURN_IF_ERROR(UploadInputRgba_(
		slot, InRgba, InVisibleWidth, InVisibleHeight, InColorSpace, InFullRange));
	OA_RETURN_IF_ERROR(SubmitEncode_(slot, InPts));
	++PendingSlots_;
	SubmitSlot_ = (SubmitSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());

	// Harvest any already-completed prefix without waiting. Never return later
	// frames ahead of an unfinished earlier submission.
	while (PendingSlots_ > 0U) {
		EncodeSlot& oldest = Slots_[HarvestSlot_];
		OaEncodedFrame completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(Harvest_(oldest, false, completed, ready));
		if (not ready) break;
		OutReady.PushBack(OaStdMove(completed));
		--PendingSlots_;
		HarvestSlot_ = (HarvestSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());
	}
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::SubmitRgbaImage(
	VkImage InImage,
	VkImageView InImageView,
	VkFormat InFormat,
	VkImageLayout InLayout,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaU64 InPts,
	OaVec<OaEncodedFrame>& OutReady,
	OaYCbCrModel InColorSpace,
	bool InFullRange,
	OaU32 InArrayLayer,
	OaEvent InReady,
	OaU32 InExternalQueueFamilyIndex,
	OaCompletionToken* OutInputConsumed)
{
	if (Slots_.Empty()) return OaStatus::Error("Video encoder has no asynchronous slots");
	if (CompatibilityUploadReady_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"Cannot submit an asynchronous job after an unmatched UploadInputRgba call");
	}

	EncodeSlot& slot = Slots_[SubmitSlot_];
	if (slot.Pending) {
		OaEncodedFrame completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(Harvest_(slot, true, completed, ready));
		if (ready) {
			OutReady.PushBack(OaStdMove(completed));
			--PendingSlots_;
			HarvestSlot_ = (HarvestSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());
		}
	}
	OA_RETURN_IF_ERROR(UploadInputRgbaImage_(
		slot, InImage, InImageView, InFormat, InLayout,
		InVisibleWidth, InVisibleHeight, InColorSpace, InFullRange,
		InArrayLayer, InReady,
		InExternalQueueFamilyIndex));
	OA_RETURN_IF_ERROR(SubmitEncode_(slot, InPts));
	++PendingSlots_;
	SubmitSlot_ = (SubmitSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());

	// The source image is consumed by conversion, not by the video queue. New
	// graph/capture callers receive that GPU completion and defer recycling;
	// compatibility callers retain the historical safe synchronous contract.
	if (OutInputConsumed != nullptr) {
		*OutInputConsumed = slot.InputTicket.Completion();
	} else {
		OA_RETURN_IF_ERROR(slot.InputTicket.WaitForSignal());
	}

	while (PendingSlots_ > 0U) {
		EncodeSlot& oldest = Slots_[HarvestSlot_];
		OaEncodedFrame completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(Harvest_(oldest, false, completed, ready));
		if (not ready) break;
		OutReady.PushBack(OaStdMove(completed));
		--PendingSlots_;
		HarvestSlot_ = (HarvestSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());
	}
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::Flush(OaVec<OaEncodedFrame>& OutFrames)
{
	if (Session_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Ok();
	}
	while (PendingSlots_ > 0U) {
		EncodeSlot& oldest = Slots_[HarvestSlot_];
		OaEncodedFrame completed;
		bool ready = false;
		OA_RETURN_IF_ERROR(Harvest_(oldest, true, completed, ready));
		if (not ready) return OaStatus::Error("Pending encode job was not harvestable");
		OutFrames.PushBack(OaStdMove(completed));
		--PendingSlots_;
		HarvestSlot_ = (HarvestSlot_ + 1U) % static_cast<OaU32>(Slots_.Size());
	}
	return OaStatus::Ok();
}


OaStatus OaVideoEncoder::DestroySlot_(EncodeSlot& InSlot)
{
	if (Rt_ == nullptr) return OaStatus::Ok();
	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	auto* allocator = static_cast<OaVmaAllocator>(vkEngine.Allocator.Allocator);
	const OaStatus inputStatus = InSlot.InputTicket.Wait();
	if (InSlot.RgbaSnapshot.Buffer != VK_NULL_HANDLE) {
		vkEngine.FreeBuffer(InSlot.RgbaSnapshot);
	}
	if (InSlot.QueryPool != VK_NULL_HANDLE) {
		vkDestroyQueryPool(device, InSlot.QueryPool, nullptr);
	}
	if (InSlot.Fence != VK_NULL_HANDLE) {
		vkDestroyFence(device, InSlot.Fence, nullptr);
	}
	InSlot.Bitstream.Destroy();
	if (InSlot.InputBindlessRegistered) {
		vkEngine.Bindless.DeregisterStorageImage(InSlot.InputYBindless);
		vkEngine.Bindless.DeregisterStorageImage(InSlot.InputUvBindless);
	}
	if (InSlot.InputUvView != VK_NULL_HANDLE) {
		vkDestroyImageView(device, InSlot.InputUvView, nullptr);
	}
	if (InSlot.InputYView != VK_NULL_HANDLE) {
		vkDestroyImageView(device, InSlot.InputYView, nullptr);
	}
	if (InSlot.InputView != VK_NULL_HANDLE) {
		vkDestroyImageView(device, InSlot.InputView, nullptr);
	}
	if (InSlot.InputImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(allocator, InSlot.InputImage,
			static_cast<OaVmaAllocation>(InSlot.InputAllocation));
	}
	InSlot = {};
	return inputStatus;
}


OaStatus OaVideoEncoder::Close()
{
	if (Rt_ == nullptr) {
		Reset_();
		return OaStatus::Ok();
	}
	auto& vkEngine = *Rt_;
	VkDevice device = static_cast<VkDevice>(vkEngine.Device.Device);
	OaStatus firstError = OaStatus::Ok();
	auto retainError = [&firstError](const OaStatus& InStatus) {
		if (firstError.IsOk() and not InStatus.IsOk()) firstError = InStatus;
	};
	// Each encode submission owns a fence. Waiting only those fences preserves
	// unrelated work on a shared video queue and makes the lifetime edge exact.
	for (auto& slot : Slots_) {
		if (slot.Pending and slot.Fence != VK_NULL_HANDLE) {
			const VkResult result = vkWaitForFences(
				device, 1, &slot.Fence, VK_TRUE, UINT64_MAX);
			if (result != VK_SUCCESS) {
				retainError(OaStatus::Error(
					OaStatusCode::VulkanError,
					"video encoder fence completion failed"));
			}
			slot.Pending = false;
		}
	}
	for (auto& slot : Slots_) retainError(DestroySlot_(slot));

	Session_.Destroy();
	SessionParams_.Destroy();
	Queue_.Destroy();
	Dpb_.Destroy();
	Reset_();
	return firstError;
}


void OaVideoEncoder::Destroy()
{
	if (const auto status = Close(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaVideoEncoder::Destroy: shutdown failed: %s",
			status.ToString().c_str());
	}
}


// ──────────────────────────────────────────────────────────────────────
//                       Transcoder
// ──────────────────────────────────────────────────────────────────────

OaResult<OaVideoTranscoder> OaVideoTranscoder::Create(
	OaEngine& InRt,
	const OaVideoProfile& InDecodeProfile,
	const OaVideoEncodeProfile& InEncodeProfile)
{
	if (InDecodeProfile.Width != InEncodeProfile.Width
		or InDecodeProfile.Height != InEncodeProfile.Height) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"OaVideoTranscoder currently requires matching decode/encode extents");
	}
	auto decoder = OaVideoDecoder::Create(InRt, InDecodeProfile);
	if (not decoder.IsOk()) return decoder.GetStatus();
	auto encoder = OaVideoEncoder::Create(InRt, InEncodeProfile);
	if (not encoder.IsOk()) {
		decoder->Destroy();
		return encoder.GetStatus();
	}
	OaVideoTranscoder transcoder;
	transcoder.Rt_ = &InRt;
	transcoder.Decoder_ = OaStdMove(*decoder);
	transcoder.Encoder_ = OaStdMove(*encoder);
	transcoder.FrameDurationUs_ = InEncodeProfile.FrameRate > 0U
		? 1'000'000ULL / InEncodeProfile.FrameRate : 0U;
	return OaStdMove(transcoder);
}


OaStatus OaVideoTranscoder::TranscodeFrame(
	const OaSpan<const OaU8>& InBitstream,
	OaEncodedFrame& OutFrame)
{
	if (Rt_ == nullptr) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaVideoTranscoder::TranscodeFrame called on an uninitialized transcoder");
	}
	OaVideoConversionOptions conversion;
	conversion.ConvertToRgb = true;
	conversion.PreferHardwareYCbCr = true;
	conversion.Filter = OaFilter::Nearest;
	OaVideoFrame rgba;
	OA_RETURN_IF_ERROR(Decoder_.DecodeFrameWithConversion(
		InBitstream, conversion, rgba));
	if (not rgba.Shown) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Decoded packet is a hidden reference frame and has no transcode output");
	}
	if (rgba.Resource != OaVideoFrameResource::Image or not rgba.IsRgb
		or rgba.Image == VK_NULL_HANDLE or rgba.ImageView == VK_NULL_HANDLE) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Decoder did not produce an image-backed RGBA frame");
	}

	OaVec<OaEncodedFrame> ready;
	const OaU64 pts = rgba.PresentationTimestamp != 0U
		? rgba.PresentationTimestamp : NextPtsUs_;
	OA_RETURN_IF_ERROR(Encoder_.SubmitRgbaImage(
		rgba.Image, rgba.ImageView, rgba.Format, rgba.Layout,
		rgba.Width, rgba.Height, pts, ready, rgba.ColorSpace, rgba.FullRange,
		rgba.ArrayLayer, rgba.Ready,
		rgba.ExternalQueueFamilyIndex));
	OA_RETURN_IF_ERROR(Encoder_.Flush(ready));
	if (ready.Size() != 1U) {
		return OaStatus::Error(OaStatusCode::Internal,
			"Synchronous transcoder expected exactly one encoded frame");
	}
	OutFrame = OaStdMove(ready[0]);
	NextPtsUs_ = pts + FrameDurationUs_;
	return OaStatus::Ok();
}


void OaVideoTranscoder::Destroy()
{
	Decoder_.Destroy();
	Encoder_.Destroy();
	Rt_ = nullptr;
	NextPtsUs_ = 0U;
	FrameDurationUs_ = 0U;
}
