// OA Vision — H.265 codec parser Implementation
// Extracts and converts H.265 parameter sets

#include "vcpH265.h"
#include "codecRegistry.h"
#include "nalParser.h"
#include "bitstreamReader.h"
#include <oa/vision/videoDecoder.h>

// ============================================================================
// level conversion
// ============================================================================

StdVideoH265LevelIdc oa::VcpH265::toStdH265Level(oa::U32 inLevelIdc) {
	switch (inLevelIdc) {
		case 30: return STD_VIDEO_H265_LEVEL_IDC_1_0;
		case 60: return STD_VIDEO_H265_LEVEL_IDC_2_0;
		case 63: return STD_VIDEO_H265_LEVEL_IDC_2_1;
		case 90: return STD_VIDEO_H265_LEVEL_IDC_3_0;
		case 93: return STD_VIDEO_H265_LEVEL_IDC_3_1;
		case 120: return STD_VIDEO_H265_LEVEL_IDC_4_0;
		case 123: return STD_VIDEO_H265_LEVEL_IDC_4_1;
		case 150: return STD_VIDEO_H265_LEVEL_IDC_5_0;
		case 153: return STD_VIDEO_H265_LEVEL_IDC_5_1;
		case 156: return STD_VIDEO_H265_LEVEL_IDC_5_2;
		case 180: return STD_VIDEO_H265_LEVEL_IDC_6_0;
		case 183: return STD_VIDEO_H265_LEVEL_IDC_6_1;
		case 186: return STD_VIDEO_H265_LEVEL_IDC_6_2;
		default: return STD_VIDEO_H265_LEVEL_IDC_INVALID;
	}
}

// ============================================================================
// profile Tier level conversion
// ============================================================================

StdVideoH265ProfileTierLevel oa::VcpH265::toStdH265ProfileTierLevel(const oa::H265VpsData& inVps) {
	StdVideoH265ProfileTierLevel ptl = {};
	ptl.flags.general_tier_flag = inVps.generalTierFlag;
	ptl.flags.general_progressive_source_flag = inVps.generalProgressiveSourceFlag;
	ptl.flags.general_interlaced_source_flag = inVps.generalInterlacedSourceFlag;
	ptl.flags.general_non_packed_constraint_flag = inVps.generalNonPackedConstraintFlag;
	ptl.flags.general_frame_only_constraint_flag = inVps.generalFrameOnlyConstraintFlag;
	ptl.general_profile_idc = static_cast<StdVideoH265ProfileIdc>(inVps.generalProfileIdc);
	ptl.general_level_idc = toStdH265Level(inVps.generalLevelIdc);
	return ptl;
}

// ============================================================================
// Decoded Picture Buffer Manager conversion
// ============================================================================

StdVideoH265DecPicBufMgr oa::VcpH265::toStdH265DecPicBufMgr(const oa::H265SpsData& inSps)
{
	StdVideoH265DecPicBufMgr dpb = {};
	for (oa::U32 i = 0; i < inSps.maxDecPicBufferingMinus1.size() && i < STD_VIDEO_H265_SUBLAYERS_LIST_SIZE; ++i) {
		dpb.max_dec_pic_buffering_minus1[i] = static_cast<uint8_t>(inSps.maxDecPicBufferingMinus1[i]);
		dpb.max_num_reorder_pics[i] = static_cast<uint8_t>(inSps.maxNumReorderPics[i]);
		dpb.max_latency_increase_plus1[i] = inSps.maxLatencyIncreasePlus1[i];
	}
	return dpb;
}

// ============================================================================
// VPS/SPS/PPS conversion (Stubs)
// ============================================================================

StdVideoH265VideoParameterSet oa::VcpH265::toStdH265Vps(const oa::H265VpsData& inVps, const StdVideoH265ProfileTierLevel& inPtl) {
	StdVideoH265VideoParameterSet vps = {};
	vps.vps_video_parameter_set_id = static_cast<uint8_t>(inVps.vpsId);
	vps.vps_max_sub_layers_minus1 = static_cast<uint8_t>(inVps.maxSubLayersMinus1);
	vps.flags.vps_temporal_id_nesting_flag = inVps.temporalIdNesting;
	vps.pProfileTierLevel = &inPtl;
	return vps;
}

StdVideoH265SequenceParameterSet oa::VcpH265::toStdH265Sps(const oa::H265SpsData& inSps, const StdVideoH265ProfileTierLevel& inPtl, const StdVideoH265DecPicBufMgr& inDpb) {
	StdVideoH265SequenceParameterSet sps = {};
	sps.flags.sps_temporal_id_nesting_flag = inSps.temporalIdNesting;
	sps.flags.separate_colour_plane_flag = inSps.separateColourPlane;
	sps.flags.conformance_window_flag = inSps.conformanceWindowLeft != 0 ||
		inSps.conformanceWindowRight != 0 ||
		inSps.conformanceWindowTop != 0 ||
		inSps.conformanceWindowBottom != 0;
	sps.flags.sps_sub_layer_ordering_info_present_flag = inSps.spsSubLayerOrderingInfoPresent;
	sps.flags.scaling_list_enabled_flag = inSps.scalingListEnabled;
	sps.flags.sps_scaling_list_data_present_flag = inSps.spsScalingListDataPresent;
	sps.flags.amp_enabled_flag = inSps.ampEnabled;
	sps.flags.sample_adaptive_offset_enabled_flag = inSps.sampleAdaptiveOffsetEnabled;
	sps.flags.pcm_enabled_flag = inSps.pcmEnabled;
	sps.flags.long_term_ref_pics_present_flag = inSps.longTermRefPicsPresent;
	sps.flags.sps_temporal_mvp_enabled_flag = inSps.temporalMvpEnabled;
	sps.flags.strong_intra_smoothing_enabled_flag = inSps.strongIntraSmoothingEnabled;
	sps.chroma_format_idc = static_cast<StdVideoH265ChromaFormatIdc>(inSps.chromaFormatIdc);
	sps.pic_width_in_luma_samples = inSps.codedWidth != 0 ? inSps.codedWidth : inSps.width;
	sps.pic_height_in_luma_samples = inSps.codedHeight != 0 ? inSps.codedHeight : inSps.height;
	sps.sps_video_parameter_set_id = static_cast<uint8_t>(inSps.vpsId);
	sps.sps_max_sub_layers_minus1 = static_cast<uint8_t>(inSps.maxSubLayersMinus1);
	sps.sps_seq_parameter_set_id = static_cast<uint8_t>(inSps.spsId);
	sps.bit_depth_luma_minus8 = static_cast<uint8_t>(inSps.bitDepthLumaMinus8);
	sps.bit_depth_chroma_minus8 = static_cast<uint8_t>(inSps.bitDepthChromaMinus8);
	sps.log2_max_pic_order_cnt_lsb_minus4 = static_cast<uint8_t>(inSps.log2MaxPicOrderCntLsbMinus4);
	sps.log2_min_luma_coding_block_size_minus3 = static_cast<uint8_t>(inSps.log2MinLumaCodingBlockSizeMinus3);
	sps.log2_diff_max_min_luma_coding_block_size = static_cast<uint8_t>(inSps.log2DiffMaxMinLumaCodingBlockSize);
	sps.log2_min_luma_transform_block_size_minus2 = static_cast<uint8_t>(inSps.log2MinLumaTransformBlockSizeMinus2);
	sps.log2_diff_max_min_luma_transform_block_size = static_cast<uint8_t>(inSps.log2DiffMaxMinLumaTransformBlockSize);
	sps.max_transform_hierarchy_depth_inter = static_cast<uint8_t>(inSps.maxTransformHierarchyDepthInter);
	sps.max_transform_hierarchy_depth_intra = static_cast<uint8_t>(inSps.maxTransformHierarchyDepthIntra);
	sps.num_short_term_ref_pic_sets = static_cast<uint8_t>(inSps.numShortTermRefPicSets);
	sps.conf_win_left_offset = inSps.conformanceWindowLeft;
	sps.conf_win_right_offset = inSps.conformanceWindowRight;
	sps.conf_win_top_offset = inSps.conformanceWindowTop;
	sps.conf_win_bottom_offset = inSps.conformanceWindowBottom;
	sps.pProfileTierLevel = &inPtl;
	sps.pDecPicBufMgr = &inDpb;
	return sps;
}

StdVideoH265PictureParameterSet oa::VcpH265::toStdH265Pps(const oa::H265PpsData& inPps, const oa::H265SpsData& inSps) {
	StdVideoH265PictureParameterSet pps = {};
	pps.flags.dependent_slice_segments_enabled_flag = inPps.dependentSliceSegmentsEnabled;
	pps.flags.output_flag_present_flag = inPps.outputFlagPresent;
	pps.flags.sign_data_hiding_enabled_flag = inPps.signDataHidingEnabled;
	pps.flags.cabac_init_present_flag = inPps.cabacInitPresent;
	pps.flags.constrained_intra_pred_flag = inPps.constrainedIntraPred;
	pps.flags.transform_skip_enabled_flag = inPps.transformSkipEnabled;
	pps.flags.cu_qp_delta_enabled_flag = inPps.cuQpDeltaEnabled;
	pps.flags.pps_slice_chroma_qp_offsets_present_flag = inPps.ppsSliceChromaQpOffsetsPresent;
	pps.flags.weighted_pred_flag = inPps.weightedPred;
	pps.flags.weighted_bipred_flag = inPps.weightedBipred;
	pps.flags.transquant_bypass_enabled_flag = inPps.transquantBypassEnabled;
	pps.flags.tiles_enabled_flag = inPps.tilesEnabled;
	pps.flags.entropy_coding_sync_enabled_flag = inPps.entropyCodingSyncEnabled;
	pps.flags.uniform_spacing_flag = inPps.uniformSpacing;
	pps.flags.loop_filter_across_tiles_enabled_flag = inPps.loopFilterAcrossTilesEnabled;
	pps.flags.pps_loop_filter_across_slices_enabled_flag = inPps.ppsLoopFilterAcrossSlicesEnabled;
	pps.flags.deblocking_filter_control_present_flag = inPps.deblockingFilterControlPresent;
	pps.flags.deblocking_filter_override_enabled_flag = inPps.deblockingFilterOverrideEnabled;
	pps.flags.pps_deblocking_filter_disabled_flag = inPps.ppsDeblockingFilterDisabled;
	pps.flags.pps_scaling_list_data_present_flag = inPps.ppsScalingListDataPresent;
	pps.flags.lists_modification_present_flag = inPps.listsModificationPresent;
	pps.flags.slice_segment_header_extension_present_flag = inPps.sliceSegmentHeaderExtensionPresent;
	pps.flags.pps_extension_present_flag = inPps.ppsExtensionPresent;
	pps.pps_pic_parameter_set_id = static_cast<uint8_t>(inPps.ppsId);
	pps.pps_seq_parameter_set_id = static_cast<uint8_t>(inPps.spsId);
	pps.sps_video_parameter_set_id = static_cast<uint8_t>(inSps.vpsId);
	pps.num_extra_slice_header_bits = static_cast<uint8_t>(inPps.numExtraSliceHeaderBits);
	pps.num_ref_idx_l0_default_active_minus1 = static_cast<uint8_t>(inPps.numRefIdxL0DefaultActiveMinus1);
	pps.num_ref_idx_l1_default_active_minus1 = static_cast<uint8_t>(inPps.numRefIdxL1DefaultActiveMinus1);
	pps.init_qp_minus26 = static_cast<int8_t>(inPps.initQpMinus26);
	pps.diff_cu_qp_delta_depth = static_cast<uint8_t>(inPps.diffCuQpDeltaDepth);
	pps.pps_cb_qp_offset = static_cast<int8_t>(inPps.cbQpOffset);
	pps.pps_cr_qp_offset = static_cast<int8_t>(inPps.crQpOffset);
	pps.pps_beta_offset_div2 = static_cast<int8_t>(inPps.betaOffsetDiv2);
	pps.pps_tc_offset_div2 = static_cast<int8_t>(inPps.tcOffsetDiv2);
	pps.log2_parallel_merge_level_minus2 = static_cast<uint8_t>(inPps.log2ParallelMergeLevelMinus2);
	pps.num_tile_columns_minus1 = static_cast<uint8_t>(inPps.numTileColumnsMinus1);
	pps.num_tile_rows_minus1 = static_cast<uint8_t>(inPps.numTileRowsMinus1);
	return pps;
}

// ============================================================================
// parser Implementation
// ============================================================================

oa::Status oa::VcpH265::parseVps(const oa::Span<const oa::U8>& inNal)
{
	oa::H265VpsData vpsData;
	if (!oa::NalParser::parseH265Vps(inNal.data(), inNal.size(), vpsData)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Failed to parse H.265 VPS");
	}

	oaVpsCache_.insert({vpsData.vpsId, vpsData});
	
	StdVideoH265ProfileTierLevel ptl = toStdH265ProfileTierLevel(vpsData);
	profileTierLevelStorage_.insert({vpsData.vpsId, ptl});
	
	auto ptlIt = profileTierLevelStorage_.find(vpsData.vpsId);
	if (ptlIt != profileTierLevelStorage_.end()) {
		StdVideoH265VideoParameterSet stdVps = toStdH265Vps(vpsData, ptlIt->second);
		stdVpsCache_.insert({vpsData.vpsId, stdVps});
	}
	
	return oa::Status::ok();
}

oa::Status oa::VcpH265::parseSps(const oa::Span<const oa::U8>& inNal)
{
	oa::H265SpsData spsData;
	if (!oa::NalParser::parseH265Sps(inNal.data(), inNal.size(), spsData)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Failed to parse H.265 SPS");
	}

	oaSpsCache_.insert({spsData.spsId, spsData});
	
	// get VPS for profile tier level
	auto vpsIt = oaVpsCache_.find(spsData.vpsId);
	if (vpsIt != oaVpsCache_.end()) {
		StdVideoH265ProfileTierLevel ptl = toStdH265ProfileTierLevel(vpsIt->second);
		profileTierLevelStorage_.insert({spsData.spsId, ptl});
	}
	
	StdVideoH265DecPicBufMgr dpb = toStdH265DecPicBufMgr(spsData);
	decPicBufMgrStorage_.insert({spsData.spsId, dpb});
	
	auto ptlIt = profileTierLevelStorage_.find(spsData.spsId);
	auto dpbIt = decPicBufMgrStorage_.find(spsData.spsId);
	if (ptlIt != profileTierLevelStorage_.end() && dpbIt != decPicBufMgrStorage_.end()) {
		StdVideoH265SequenceParameterSet stdSps = toStdH265Sps(spsData, ptlIt->second, dpbIt->second);
		stdSpsCache_.insert({spsData.spsId, stdSps});
	}
	
	return oa::Status::ok();
}

oa::Status oa::VcpH265::parsePps(const oa::Span<const oa::U8>& inNal)
{
	oa::H265PpsData ppsData;
	if (!oa::NalParser::parseH265Pps(inNal.data(), inNal.size(), ppsData)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Failed to parse H.265 PPS");
	}

	oaPpsCache_.insert({ppsData.ppsId, ppsData});
	
	auto spsIt = oaSpsCache_.find(ppsData.spsId);
	if (spsIt != oaSpsCache_.end()) {
		StdVideoH265PictureParameterSet stdPps = toStdH265Pps(ppsData, spsIt->second);
		stdPpsCache_.insert({ppsData.ppsId, stdPps});
	}
	
	return oa::Status::ok();
}

const StdVideoH265VideoParameterSet* oa::VcpH265::getH265Vps(oa::U32 inVpsId) const
{
	auto it = stdVpsCache_.find(inVpsId);
	return (it != stdVpsCache_.end()) ? &it->second : nullptr;
}

const StdVideoH265SequenceParameterSet* oa::VcpH265::getH265Sps(oa::U32 inSpsId) const
{
	auto it = stdSpsCache_.find(inSpsId);
	return (it != stdSpsCache_.end()) ? &it->second : nullptr;
}

const StdVideoH265PictureParameterSet* oa::VcpH265::getH265Pps(oa::U32 inPpsId) const
{
	auto it = stdPpsCache_.find(inPpsId);
	return (it != stdPpsCache_.end()) ? &it->second : nullptr;
}

const oa::H265VpsData* oa::VcpH265::getVpsData(oa::U32 inVpsId) const
{
	auto it = oaVpsCache_.find(inVpsId);
	return (it != oaVpsCache_.end()) ? &it->second : nullptr;
}

const oa::H265SpsData* oa::VcpH265::getSpsData(oa::U32 inSpsId) const
{
	auto it = oaSpsCache_.find(inSpsId);
	return (it != oaSpsCache_.end()) ? &it->second : nullptr;
}

const oa::H265PpsData* oa::VcpH265::getPpsData(oa::U32 inPpsId) const
{
	auto it = oaPpsCache_.find(inPpsId);
	return (it != oaPpsCache_.end()) ? &it->second : nullptr;
}

void oa::VcpH265::clearParameterSets()
{
	oaVpsCache_.clear();
	oaSpsCache_.clear();
	oaPpsCache_.clear();
	stdVpsCache_.clear();
	stdSpsCache_.clear();
	stdPpsCache_.clear();
	profileTierLevelStorage_.clear();
	decPicBufMgrStorage_.clear();
}

// ============================================================================
// access Unit Parsing
// ============================================================================

namespace {
	struct AnnexBNalUnit {
		oa::Usize startCodeOffset = 0;
		oa::Usize startCodeSize = 0;
		oa::Usize offset = 0;
		oa::Usize size = 0;
	};

	bool isAnnexBStartCodeAt(const oa::U8* inData, oa::Usize inSize, oa::Usize inOffset, oa::Usize& outStartCodeSize)
	{
		if (inOffset + 3 <= inSize && inData[inOffset] == 0 && inData[inOffset + 1] == 0 && inData[inOffset + 2] == 1) {
			outStartCodeSize = 3;
			return true;
		}
		if (inOffset + 4 <= inSize && inData[inOffset] == 0 && inData[inOffset + 1] == 0 && inData[inOffset + 2] == 0 && inData[inOffset + 3] == 1) {
			outStartCodeSize = 4;
			return true;
		}
		return false;
	}

	oa::Status findAnnexBNalUnits(const oa::Span<const oa::U8>& inBitstream, oa::Vec<AnnexBNalUnit>& outNalUnits)	{
		outNalUnits.clear();
		const oa::U8* data = inBitstream.data();
		const oa::Usize size = inBitstream.size();
		oa::Usize offset = 0;
		while (offset < size) {
			oa::Usize startCodeSize = 0;
			while (offset < size && !isAnnexBStartCodeAt(data, size, offset, startCodeSize)) {
				++offset;
			}
			if (offset >= size) {
				break;
			}
			const oa::Usize startCodeOffset = offset;
			const oa::Usize nalOffset = offset + startCodeSize;
			offset = nalOffset;
			while (offset < size) {
				oa::Usize nextStartCodeSize = 0;
				if (isAnnexBStartCodeAt(data, size, offset, nextStartCodeSize)) {
					break;
				}
				++offset;
			}
			if (nalOffset < offset) {
				outNalUnits.pushBack({startCodeOffset, startCodeSize, nalOffset, offset - nalOffset});
			}
		}
		if (outNalUnits.empty()) {
			return oa::Status::error("Invalid H.265 Annex-B bitstream - no NAL start code found");
		}
		return oa::Status::ok();
	}
} // namespace

oa::Vec<oa::U32> oa::VcpH265::getCachedVpsIds() const {
	oa::Vec<oa::U32> ids;
	ids.reserve(oaVpsCache_.size());
	for (const auto& kv : oaVpsCache_) ids.pushBack(kv.first);
	return ids;
}

oa::Vec<oa::U32> oa::VcpH265::getCachedSpsIds() const {
	oa::Vec<oa::U32> ids;
	ids.reserve(oaSpsCache_.size());
	for (const auto& kv : oaSpsCache_) ids.pushBack(kv.first);
	return ids;
}

oa::Vec<oa::U32> oa::VcpH265::getCachedPpsIds() const {
	oa::Vec<oa::U32> ids;
	ids.reserve(oaPpsCache_.size());
	for (const auto& kv : oaPpsCache_) ids.pushBack(kv.first);
	return ids;
}

oa::Status oa::VcpH265::parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, oa::H265PictureDesc& outDesc) {
	outDesc.hasPicture = false;
	outDesc.sliceOffsets.clear();
	outDesc.isReference = false;

	oa::Vec<AnnexBNalUnit> nalUnits;
	oa::Status findStatus = findAnnexBNalUnits(inBitstream, nalUnits);
	if (!findStatus.isOk()) {
		return findStatus;
	}

	const oa::U8* data = inBitstream.data();
	bool sawPicture = false;
	bool haveFirstSliceHeader = false;
	oa::H265SliceHeader firstSliceHeader = {};
	oa::H265SpsData firstSliceSps = {};
	oa::H265PpsData firstSlicePps = {};

	// Pass 1: cache all VPS/SPS/PPS parameter sets
	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.size < 2) {
			continue;
		}
		oa::U8 temporalId = 0;
		oa::H265NalType nalType = oa::NalParser::parseH265NalHeader(data + nalUnit.offset, nalUnit.size, temporalId);
		(void)temporalId;

		if (nalType == oa::H265NalType::Vps) {
			oa::H265VpsData vps;
			if (!oa::NalParser::parseH265Vps(data + nalUnit.offset, nalUnit.size, vps)) {
				return oa::Status::error("Failed to parse H.265 VPS");
			}
			oaVpsCache_.insert({vps.vpsId, vps});
			StdVideoH265ProfileTierLevel ptl = toStdH265ProfileTierLevel(vps);
			profileTierLevelStorage_.insert({vps.vpsId, ptl});
			StdVideoH265VideoParameterSet stdVps = toStdH265Vps(vps, ptl);
			stdVpsCache_.insert({vps.vpsId, stdVps});
			outDesc.vpsInAu.pushBack(vps);
		}
		else if (nalType == oa::H265NalType::Sps) {
			oa::H265SpsData sps;
			if (!oa::NalParser::parseH265Sps(data + nalUnit.offset, nalUnit.size, sps)) {
				return oa::Status::error("Failed to parse H.265 SPS");
			}
			oaSpsCache_.insert({sps.spsId, sps});
			auto vpsIt = oaVpsCache_.find(sps.vpsId);
			if (vpsIt != oaVpsCache_.end()) {
				StdVideoH265ProfileTierLevel ptl = toStdH265ProfileTierLevel(vpsIt->second);
				profileTierLevelStorage_.insert({sps.spsId, ptl});
				StdVideoH265DecPicBufMgr dpb = toStdH265DecPicBufMgr(sps);
				decPicBufMgrStorage_.insert({sps.spsId, dpb});
				auto ptlIt = profileTierLevelStorage_.find(sps.spsId);
				auto dpbIt = decPicBufMgrStorage_.find(sps.spsId);
				if (ptlIt != profileTierLevelStorage_.end() && dpbIt != decPicBufMgrStorage_.end()) {
					StdVideoH265SequenceParameterSet stdSps = toStdH265Sps(sps, ptlIt->second, dpbIt->second);
					stdSpsCache_.insert({sps.spsId, stdSps});
				}
			}
			outDesc.spsInAu.pushBack(sps);
		}
		else if (nalType == oa::H265NalType::Pps) {
			oa::H265PpsData pps;
			if (!oa::NalParser::parseH265Pps(data + nalUnit.offset, nalUnit.size, pps)) {
				return oa::Status::error("Failed to parse H.265 PPS");
			}
			oaPpsCache_.insert({pps.ppsId, pps});
			auto spsIt = oaSpsCache_.find(pps.spsId);
			if (spsIt != oaSpsCache_.end()) {
				StdVideoH265PictureParameterSet stdPps = toStdH265Pps(pps, spsIt->second);
				stdPpsCache_.insert({pps.ppsId, stdPps});
			}
			outDesc.ppsInAu.pushBack(pps);
		}
	}

	// Pass 2: process slices now that all parameter sets are cached
	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.size < 2) {
			continue;
		}
		oa::U8 temporalId = 0;
		oa::H265NalType nalType = oa::NalParser::parseH265NalHeader(data + nalUnit.offset, nalUnit.size, temporalId);
		(void)temporalId;

		if (static_cast<oa::U8>(nalType) < 32u) {
			sawPicture = true;

			// Manually read PPS ID from the first few bytes of the slice header
			oa::U32 ppsId = 0;
			{
				oa::Vec<oa::U8> rbsp = oa::NalParser::makeRbsp(data + nalUnit.offset, nalUnit.size);
				oa::BitstreamReader reader(rbsp.data(), rbsp.size());
				reader.skipBits(16); // NAL header
				const bool irap = static_cast<oa::U8>(nalType) >= 16 && static_cast<oa::U8>(nalType) <= 23;
				(void)reader.readBit(); // first_slice_segment_in_pic_flag
				if (irap) {
					reader.skipBits(1);
				}
				ppsId = reader.readUE();
			}

			auto ppsIt = oaPpsCache_.find(ppsId);
			if (ppsIt == oaPpsCache_.end()) {
				return oa::Status::error("H.265 PPS not found in cache");
			}
			auto spsIt = oaSpsCache_.find(ppsIt->second.spsId);
			if (spsIt == oaSpsCache_.end()) {
				return oa::Status::error("H.265 SPS not found in cache");
			}

			oa::H265SliceHeader sliceHeader;
			if (!oa::NalParser::parseH265SliceHeader(
				data + nalUnit.offset,
				nalUnit.size,
				nalType,
				spsIt->second,
				ppsIt->second,
				sliceHeader)) {
				return oa::Status::error("Failed to parse H.265 slice header");
			}
			if (!haveFirstSliceHeader) {
				firstSliceHeader = sliceHeader;
				firstSliceSps = spsIt->second;
				firstSlicePps = ppsIt->second;
				haveFirstSliceHeader = true;
			} else if (sliceHeader.ppsId != firstSliceHeader.ppsId ||
				sliceHeader.spsId != firstSliceHeader.spsId ||
				sliceHeader.picOrderCntVal != firstSliceHeader.picOrderCntVal ||
				sliceHeader.sliceType != firstSliceHeader.sliceType) {
				return oa::Status::error(oa::StatusCode::Unavailable, "H.265 mixed-picture slice access units are not implemented");
			}
			outDesc.sliceOffsets.pushBack(static_cast<oa::U32>(nalUnit.startCodeOffset));
		}
	}

	if (!sawPicture) {
		return oa::Status::ok(); // Parameter-set-only access unit
	}
	if (!haveFirstSliceHeader || outDesc.sliceOffsets.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.265 picture has no decodable slices");
	}

	outDesc.hasPicture = true;
	outDesc.sliceHeader = firstSliceHeader;
	outDesc.sps = firstSliceSps;
	outDesc.pps = firstSlicePps;
	outDesc.isReference = firstSliceHeader.isReference;
	return oa::Status::ok();
}

namespace {
struct H265CodecRegistrar {
	H265CodecRegistrar() {
		auto parser = oa::makeUnique<oa::VcpH265>();
		oa::VideoCodecRegistry::getInstance().registerParser(
			oa::VideoCodec::H265,
			oa::move(parser));
	}
};
static H265CodecRegistrar g_H265Registrar __attribute__((used));
} // namespace

