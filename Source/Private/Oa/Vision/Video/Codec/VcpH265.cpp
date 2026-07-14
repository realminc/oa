// OA Vision — H.265 Codec Parser Implementation
// Extracts and converts H.265 parameter sets

#include "VcpH265.h"
#include "CodecRegistry.h"
#include "NalParser.h"
#include "BitstreamReader.h"
#include <Oa/Vision/VideoDecoder.h>

// ============================================================================
// Level Conversion
// ============================================================================

StdVideoH265LevelIdc OaVcpH265::ToStdH265Level(OaU32 InLevelIdc) {
	switch (InLevelIdc) {
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
// Profile Tier Level Conversion
// ============================================================================

StdVideoH265ProfileTierLevel OaVcpH265::ToStdH265ProfileTierLevel(const OaH265VpsData& InVps) {
	StdVideoH265ProfileTierLevel ptl = {};
	ptl.flags.general_tier_flag = InVps.GeneralTierFlag;
	ptl.flags.general_progressive_source_flag = InVps.GeneralProgressiveSourceFlag;
	ptl.flags.general_interlaced_source_flag = InVps.GeneralInterlacedSourceFlag;
	ptl.flags.general_non_packed_constraint_flag = InVps.GeneralNonPackedConstraintFlag;
	ptl.flags.general_frame_only_constraint_flag = InVps.GeneralFrameOnlyConstraintFlag;
	ptl.general_profile_idc = static_cast<StdVideoH265ProfileIdc>(InVps.GeneralProfileIdc);
	ptl.general_level_idc = ToStdH265Level(InVps.GeneralLevelIdc);
	return ptl;
}

// ============================================================================
// Decoded Picture Buffer Manager Conversion
// ============================================================================

StdVideoH265DecPicBufMgr OaVcpH265::ToStdH265DecPicBufMgr(const OaH265SpsData& InSps)
{
	StdVideoH265DecPicBufMgr dpb = {};
	for (OaU32 i = 0; i < InSps.MaxDecPicBufferingMinus1.Size() && i < STD_VIDEO_H265_SUBLAYERS_LIST_SIZE; ++i) {
		dpb.max_dec_pic_buffering_minus1[i] = static_cast<uint8_t>(InSps.MaxDecPicBufferingMinus1[i]);
		dpb.max_num_reorder_pics[i] = static_cast<uint8_t>(InSps.MaxNumReorderPics[i]);
		dpb.max_latency_increase_plus1[i] = InSps.MaxLatencyIncreasePlus1[i];
	}
	return dpb;
}

// ============================================================================
// VPS/SPS/PPS Conversion (Stubs)
// ============================================================================

StdVideoH265VideoParameterSet OaVcpH265::ToStdH265Vps(const OaH265VpsData& InVps, const StdVideoH265ProfileTierLevel& InPtl) {
	StdVideoH265VideoParameterSet vps = {};
	vps.vps_video_parameter_set_id = static_cast<uint8_t>(InVps.VpsId);
	vps.vps_max_sub_layers_minus1 = static_cast<uint8_t>(InVps.MaxSubLayersMinus1);
	vps.flags.vps_temporal_id_nesting_flag = InVps.TemporalIdNesting;
	vps.pProfileTierLevel = &InPtl;
	return vps;
}

StdVideoH265SequenceParameterSet OaVcpH265::ToStdH265Sps(const OaH265SpsData& InSps, const StdVideoH265ProfileTierLevel& InPtl, const StdVideoH265DecPicBufMgr& InDpb) {
	StdVideoH265SequenceParameterSet sps = {};
	sps.flags.sps_temporal_id_nesting_flag = InSps.TemporalIdNesting;
	sps.flags.separate_colour_plane_flag = InSps.SeparateColourPlane;
	sps.flags.conformance_window_flag = InSps.ConformanceWindowLeft != 0 ||
		InSps.ConformanceWindowRight != 0 ||
		InSps.ConformanceWindowTop != 0 ||
		InSps.ConformanceWindowBottom != 0;
	sps.flags.sps_sub_layer_ordering_info_present_flag = InSps.SpsSubLayerOrderingInfoPresent;
	sps.flags.scaling_list_enabled_flag = InSps.ScalingListEnabled;
	sps.flags.sps_scaling_list_data_present_flag = InSps.SpsScalingListDataPresent;
	sps.flags.amp_enabled_flag = InSps.AmpEnabled;
	sps.flags.sample_adaptive_offset_enabled_flag = InSps.SampleAdaptiveOffsetEnabled;
	sps.flags.pcm_enabled_flag = InSps.PcmEnabled;
	sps.flags.long_term_ref_pics_present_flag = InSps.LongTermRefPicsPresent;
	sps.flags.sps_temporal_mvp_enabled_flag = InSps.TemporalMvpEnabled;
	sps.flags.strong_intra_smoothing_enabled_flag = InSps.StrongIntraSmoothingEnabled;
	sps.chroma_format_idc = static_cast<StdVideoH265ChromaFormatIdc>(InSps.ChromaFormatIdc);
	sps.pic_width_in_luma_samples = InSps.CodedWidth != 0 ? InSps.CodedWidth : InSps.Width;
	sps.pic_height_in_luma_samples = InSps.CodedHeight != 0 ? InSps.CodedHeight : InSps.Height;
	sps.sps_video_parameter_set_id = static_cast<uint8_t>(InSps.VpsId);
	sps.sps_max_sub_layers_minus1 = static_cast<uint8_t>(InSps.MaxSubLayersMinus1);
	sps.sps_seq_parameter_set_id = static_cast<uint8_t>(InSps.SpsId);
	sps.bit_depth_luma_minus8 = static_cast<uint8_t>(InSps.BitDepthLumaMinus8);
	sps.bit_depth_chroma_minus8 = static_cast<uint8_t>(InSps.BitDepthChromaMinus8);
	sps.log2_max_pic_order_cnt_lsb_minus4 = static_cast<uint8_t>(InSps.Log2MaxPicOrderCntLsbMinus4);
	sps.log2_min_luma_coding_block_size_minus3 = static_cast<uint8_t>(InSps.Log2MinLumaCodingBlockSizeMinus3);
	sps.log2_diff_max_min_luma_coding_block_size = static_cast<uint8_t>(InSps.Log2DiffMaxMinLumaCodingBlockSize);
	sps.log2_min_luma_transform_block_size_minus2 = static_cast<uint8_t>(InSps.Log2MinLumaTransformBlockSizeMinus2);
	sps.log2_diff_max_min_luma_transform_block_size = static_cast<uint8_t>(InSps.Log2DiffMaxMinLumaTransformBlockSize);
	sps.max_transform_hierarchy_depth_inter = static_cast<uint8_t>(InSps.MaxTransformHierarchyDepthInter);
	sps.max_transform_hierarchy_depth_intra = static_cast<uint8_t>(InSps.MaxTransformHierarchyDepthIntra);
	sps.num_short_term_ref_pic_sets = static_cast<uint8_t>(InSps.NumShortTermRefPicSets);
	sps.conf_win_left_offset = InSps.ConformanceWindowLeft;
	sps.conf_win_right_offset = InSps.ConformanceWindowRight;
	sps.conf_win_top_offset = InSps.ConformanceWindowTop;
	sps.conf_win_bottom_offset = InSps.ConformanceWindowBottom;
	sps.pProfileTierLevel = &InPtl;
	sps.pDecPicBufMgr = &InDpb;
	return sps;
}

StdVideoH265PictureParameterSet OaVcpH265::ToStdH265Pps(const OaH265PpsData& InPps, const OaH265SpsData& InSps) {
	StdVideoH265PictureParameterSet pps = {};
	pps.flags.dependent_slice_segments_enabled_flag = InPps.DependentSliceSegmentsEnabled;
	pps.flags.output_flag_present_flag = InPps.OutputFlagPresent;
	pps.flags.sign_data_hiding_enabled_flag = InPps.SignDataHidingEnabled;
	pps.flags.cabac_init_present_flag = InPps.CabacInitPresent;
	pps.flags.constrained_intra_pred_flag = InPps.ConstrainedIntraPred;
	pps.flags.transform_skip_enabled_flag = InPps.TransformSkipEnabled;
	pps.flags.cu_qp_delta_enabled_flag = InPps.CuQpDeltaEnabled;
	pps.flags.pps_slice_chroma_qp_offsets_present_flag = InPps.PpsSliceChromaQpOffsetsPresent;
	pps.flags.weighted_pred_flag = InPps.WeightedPred;
	pps.flags.weighted_bipred_flag = InPps.WeightedBipred;
	pps.flags.transquant_bypass_enabled_flag = InPps.TransquantBypassEnabled;
	pps.flags.tiles_enabled_flag = InPps.TilesEnabled;
	pps.flags.entropy_coding_sync_enabled_flag = InPps.EntropyCodingSyncEnabled;
	pps.flags.uniform_spacing_flag = InPps.UniformSpacing;
	pps.flags.loop_filter_across_tiles_enabled_flag = InPps.LoopFilterAcrossTilesEnabled;
	pps.flags.pps_loop_filter_across_slices_enabled_flag = InPps.PpsLoopFilterAcrossSlicesEnabled;
	pps.flags.deblocking_filter_control_present_flag = InPps.DeblockingFilterControlPresent;
	pps.flags.deblocking_filter_override_enabled_flag = InPps.DeblockingFilterOverrideEnabled;
	pps.flags.pps_deblocking_filter_disabled_flag = InPps.PpsDeblockingFilterDisabled;
	pps.flags.pps_scaling_list_data_present_flag = InPps.PpsScalingListDataPresent;
	pps.flags.lists_modification_present_flag = InPps.ListsModificationPresent;
	pps.flags.slice_segment_header_extension_present_flag = InPps.SliceSegmentHeaderExtensionPresent;
	pps.flags.pps_extension_present_flag = InPps.PpsExtensionPresent;
	pps.pps_pic_parameter_set_id = static_cast<uint8_t>(InPps.PpsId);
	pps.pps_seq_parameter_set_id = static_cast<uint8_t>(InPps.SpsId);
	pps.sps_video_parameter_set_id = static_cast<uint8_t>(InSps.VpsId);
	pps.num_extra_slice_header_bits = static_cast<uint8_t>(InPps.NumExtraSliceHeaderBits);
	pps.num_ref_idx_l0_default_active_minus1 = static_cast<uint8_t>(InPps.NumRefIdxL0DefaultActiveMinus1);
	pps.num_ref_idx_l1_default_active_minus1 = static_cast<uint8_t>(InPps.NumRefIdxL1DefaultActiveMinus1);
	pps.init_qp_minus26 = static_cast<int8_t>(InPps.InitQpMinus26);
	pps.diff_cu_qp_delta_depth = static_cast<uint8_t>(InPps.DiffCuQpDeltaDepth);
	pps.pps_cb_qp_offset = static_cast<int8_t>(InPps.CbQpOffset);
	pps.pps_cr_qp_offset = static_cast<int8_t>(InPps.CrQpOffset);
	pps.pps_beta_offset_div2 = static_cast<int8_t>(InPps.BetaOffsetDiv2);
	pps.pps_tc_offset_div2 = static_cast<int8_t>(InPps.TcOffsetDiv2);
	pps.log2_parallel_merge_level_minus2 = static_cast<uint8_t>(InPps.Log2ParallelMergeLevelMinus2);
	pps.num_tile_columns_minus1 = static_cast<uint8_t>(InPps.NumTileColumnsMinus1);
	pps.num_tile_rows_minus1 = static_cast<uint8_t>(InPps.NumTileRowsMinus1);
	return pps;
}

// ============================================================================
// Parser Implementation
// ============================================================================

OaStatus OaVcpH265::ParseVps(const OaSpan<const OaU8>& InNal)
{
	OaH265VpsData vpsData;
	if (!OaNalParser::ParseH265Vps(InNal.Data(), InNal.Size(), vpsData)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Failed to parse H.265 VPS");
	}

	OaVpsCache_.Insert({vpsData.VpsId, vpsData});
	
	StdVideoH265ProfileTierLevel ptl = ToStdH265ProfileTierLevel(vpsData);
	ProfileTierLevelStorage_.Insert({vpsData.VpsId, ptl});
	
	auto ptlIt = ProfileTierLevelStorage_.Find(vpsData.VpsId);
	if (ptlIt != ProfileTierLevelStorage_.End()) {
		StdVideoH265VideoParameterSet stdVps = ToStdH265Vps(vpsData, ptlIt->second);
		StdVpsCache_.Insert({vpsData.VpsId, stdVps});
	}
	
	return OaStatus::Ok();
}

OaStatus OaVcpH265::ParseSps(const OaSpan<const OaU8>& InNal)
{
	OaH265SpsData spsData;
	if (!OaNalParser::ParseH265Sps(InNal.Data(), InNal.Size(), spsData)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Failed to parse H.265 SPS");
	}

	OaSpsCache_.Insert({spsData.SpsId, spsData});
	
	// Get VPS for profile tier level
	auto vpsIt = OaVpsCache_.Find(spsData.VpsId);
	if (vpsIt != OaVpsCache_.End()) {
		StdVideoH265ProfileTierLevel ptl = ToStdH265ProfileTierLevel(vpsIt->second);
		ProfileTierLevelStorage_.Insert({spsData.SpsId, ptl});
	}
	
	StdVideoH265DecPicBufMgr dpb = ToStdH265DecPicBufMgr(spsData);
	DecPicBufMgrStorage_.Insert({spsData.SpsId, dpb});
	
	auto ptlIt = ProfileTierLevelStorage_.Find(spsData.SpsId);
	auto dpbIt = DecPicBufMgrStorage_.Find(spsData.SpsId);
	if (ptlIt != ProfileTierLevelStorage_.End() && dpbIt != DecPicBufMgrStorage_.End()) {
		StdVideoH265SequenceParameterSet stdSps = ToStdH265Sps(spsData, ptlIt->second, dpbIt->second);
		StdSpsCache_.Insert({spsData.SpsId, stdSps});
	}
	
	return OaStatus::Ok();
}

OaStatus OaVcpH265::ParsePps(const OaSpan<const OaU8>& InNal)
{
	OaH265PpsData ppsData;
	if (!OaNalParser::ParseH265Pps(InNal.Data(), InNal.Size(), ppsData)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Failed to parse H.265 PPS");
	}

	OaPpsCache_.Insert({ppsData.PpsId, ppsData});
	
	auto spsIt = OaSpsCache_.Find(ppsData.SpsId);
	if (spsIt != OaSpsCache_.End()) {
		StdVideoH265PictureParameterSet stdPps = ToStdH265Pps(ppsData, spsIt->second);
		StdPpsCache_.Insert({ppsData.PpsId, stdPps});
	}
	
	return OaStatus::Ok();
}

const StdVideoH265VideoParameterSet* OaVcpH265::GetH265Vps(OaU32 InVpsId) const
{
	auto it = StdVpsCache_.Find(InVpsId);
	return (it != StdVpsCache_.End()) ? &it->second : nullptr;
}

const StdVideoH265SequenceParameterSet* OaVcpH265::GetH265Sps(OaU32 InSpsId) const
{
	auto it = StdSpsCache_.Find(InSpsId);
	return (it != StdSpsCache_.End()) ? &it->second : nullptr;
}

const StdVideoH265PictureParameterSet* OaVcpH265::GetH265Pps(OaU32 InPpsId) const
{
	auto it = StdPpsCache_.Find(InPpsId);
	return (it != StdPpsCache_.End()) ? &it->second : nullptr;
}

const OaH265VpsData* OaVcpH265::GetOaVps(OaU32 InVpsId) const
{
	auto it = OaVpsCache_.Find(InVpsId);
	return (it != OaVpsCache_.End()) ? &it->second : nullptr;
}

const OaH265SpsData* OaVcpH265::GetOaSps(OaU32 InSpsId) const
{
	auto it = OaSpsCache_.Find(InSpsId);
	return (it != OaSpsCache_.End()) ? &it->second : nullptr;
}

const OaH265PpsData* OaVcpH265::GetOaPps(OaU32 InPpsId) const
{
	auto it = OaPpsCache_.Find(InPpsId);
	return (it != OaPpsCache_.End()) ? &it->second : nullptr;
}

void OaVcpH265::ClearParameterSets()
{
	OaVpsCache_.Clear();
	OaSpsCache_.Clear();
	OaPpsCache_.Clear();
	StdVpsCache_.Clear();
	StdSpsCache_.Clear();
	StdPpsCache_.Clear();
	ProfileTierLevelStorage_.Clear();
	DecPicBufMgrStorage_.Clear();
}

// ============================================================================
// Access Unit Parsing
// ============================================================================

namespace {
	struct AnnexBNalUnit {
		OaUsize StartCodeOffset = 0;
		OaUsize StartCodeSize = 0;
		OaUsize Offset = 0;
		OaUsize Size = 0;
	};

	bool IsAnnexBStartCodeAt(const OaU8* InData, OaUsize InSize, OaUsize InOffset, OaUsize& OutStartCodeSize)
	{
		if (InOffset + 3 <= InSize && InData[InOffset] == 0 && InData[InOffset + 1] == 0 && InData[InOffset + 2] == 1) {
			OutStartCodeSize = 3;
			return true;
		}
		if (InOffset + 4 <= InSize && InData[InOffset] == 0 && InData[InOffset + 1] == 0 && InData[InOffset + 2] == 0 && InData[InOffset + 3] == 1) {
			OutStartCodeSize = 4;
			return true;
		}
		return false;
	}

	OaStatus FindAnnexBNalUnits(const OaSpan<const OaU8>& InBitstream, OaVec<AnnexBNalUnit>& OutNalUnits)	{
		OutNalUnits.Clear();
		const OaU8* data = InBitstream.Data();
		const OaUsize size = InBitstream.Size();
		OaUsize offset = 0;
		while (offset < size) {
			OaUsize startCodeSize = 0;
			while (offset < size && !IsAnnexBStartCodeAt(data, size, offset, startCodeSize)) {
				++offset;
			}
			if (offset >= size) {
				break;
			}
			const OaUsize startCodeOffset = offset;
			const OaUsize nalOffset = offset + startCodeSize;
			offset = nalOffset;
			while (offset < size) {
				OaUsize nextStartCodeSize = 0;
				if (IsAnnexBStartCodeAt(data, size, offset, nextStartCodeSize)) {
					break;
				}
				++offset;
			}
			if (nalOffset < offset) {
				OutNalUnits.PushBack({startCodeOffset, startCodeSize, nalOffset, offset - nalOffset});
			}
		}
		if (OutNalUnits.Empty()) {
			return OaStatus::Error("Invalid H.265 Annex-B bitstream - no NAL start code found");
		}
		return OaStatus::Ok();
	}
} // namespace

OaVec<OaU32> OaVcpH265::GetCachedVpsIds() const {
	OaVec<OaU32> ids;
	ids.Reserve(OaVpsCache_.Size());
	for (const auto& kv : OaVpsCache_) ids.PushBack(kv.first);
	return ids;
}

OaVec<OaU32> OaVcpH265::GetCachedSpsIds() const {
	OaVec<OaU32> ids;
	ids.Reserve(OaSpsCache_.Size());
	for (const auto& kv : OaSpsCache_) ids.PushBack(kv.first);
	return ids;
}

OaVec<OaU32> OaVcpH265::GetCachedPpsIds() const {
	OaVec<OaU32> ids;
	ids.Reserve(OaPpsCache_.Size());
	for (const auto& kv : OaPpsCache_) ids.PushBack(kv.first);
	return ids;
}

OaStatus OaVcpH265::ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaH265PictureDesc& OutDesc) {
	OutDesc.HasPicture = false;
	OutDesc.SliceOffsets.Clear();
	OutDesc.IsReference = false;

	OaVec<AnnexBNalUnit> nalUnits;
	OaStatus findStatus = FindAnnexBNalUnits(InBitstream, nalUnits);
	if (!findStatus.IsOk()) {
		return findStatus;
	}

	const OaU8* data = InBitstream.Data();
	bool sawPicture = false;
	bool haveFirstSliceHeader = false;
	OaH265SliceHeader firstSliceHeader = {};
	OaH265SpsData firstSliceSps = {};
	OaH265PpsData firstSlicePps = {};

	// Pass 1: Cache all VPS/SPS/PPS parameter sets
	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.Size < 2) {
			continue;
		}
		OaU8 temporalId = 0;
		OaH265NalType nalType = OaNalParser::ParseH265NalHeader(data + nalUnit.Offset, nalUnit.Size, temporalId);
		(void)temporalId;

		if (nalType == OaH265NalType::Vps) {
			OaH265VpsData vps;
			if (!OaNalParser::ParseH265Vps(data + nalUnit.Offset, nalUnit.Size, vps)) {
				return OaStatus::Error("Failed to parse H.265 VPS");
			}
			OaVpsCache_.Insert({vps.VpsId, vps});
			StdVideoH265ProfileTierLevel ptl = ToStdH265ProfileTierLevel(vps);
			ProfileTierLevelStorage_.Insert({vps.VpsId, ptl});
			StdVideoH265VideoParameterSet stdVps = ToStdH265Vps(vps, ptl);
			StdVpsCache_.Insert({vps.VpsId, stdVps});
			OutDesc.VpsInAu.PushBack(vps);
		}
		else if (nalType == OaH265NalType::Sps) {
			OaH265SpsData sps;
			if (!OaNalParser::ParseH265Sps(data + nalUnit.Offset, nalUnit.Size, sps)) {
				return OaStatus::Error("Failed to parse H.265 SPS");
			}
			OaSpsCache_.Insert({sps.SpsId, sps});
			auto vpsIt = OaVpsCache_.Find(sps.VpsId);
			if (vpsIt != OaVpsCache_.End()) {
				StdVideoH265ProfileTierLevel ptl = ToStdH265ProfileTierLevel(vpsIt->second);
				ProfileTierLevelStorage_.Insert({sps.SpsId, ptl});
				StdVideoH265DecPicBufMgr dpb = ToStdH265DecPicBufMgr(sps);
				DecPicBufMgrStorage_.Insert({sps.SpsId, dpb});
				auto ptlIt = ProfileTierLevelStorage_.Find(sps.SpsId);
				auto dpbIt = DecPicBufMgrStorage_.Find(sps.SpsId);
				if (ptlIt != ProfileTierLevelStorage_.End() && dpbIt != DecPicBufMgrStorage_.End()) {
					StdVideoH265SequenceParameterSet stdSps = ToStdH265Sps(sps, ptlIt->second, dpbIt->second);
					StdSpsCache_.Insert({sps.SpsId, stdSps});
				}
			}
			OutDesc.SpsInAu.PushBack(sps);
		}
		else if (nalType == OaH265NalType::Pps) {
			OaH265PpsData pps;
			if (!OaNalParser::ParseH265Pps(data + nalUnit.Offset, nalUnit.Size, pps)) {
				return OaStatus::Error("Failed to parse H.265 PPS");
			}
			OaPpsCache_.Insert({pps.PpsId, pps});
			auto spsIt = OaSpsCache_.Find(pps.SpsId);
			if (spsIt != OaSpsCache_.End()) {
				StdVideoH265PictureParameterSet stdPps = ToStdH265Pps(pps, spsIt->second);
				StdPpsCache_.Insert({pps.PpsId, stdPps});
			}
			OutDesc.PpsInAu.PushBack(pps);
		}
	}

	// Pass 2: Process slices now that all parameter sets are cached
	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.Size < 2) {
			continue;
		}
		OaU8 temporalId = 0;
		OaH265NalType nalType = OaNalParser::ParseH265NalHeader(data + nalUnit.Offset, nalUnit.Size, temporalId);
		(void)temporalId;

		if (static_cast<OaU8>(nalType) < 32u) {
			sawPicture = true;

			// Manually read PPS ID from the first few bytes of the slice header
			OaU32 ppsId = 0;
			{
				OaVec<OaU8> rbsp = OaNalParser::MakeRbsp(data + nalUnit.Offset, nalUnit.Size);
				OaBitstreamReader reader(rbsp.Data(), rbsp.Size());
				reader.SkipBits(16); // NAL header
				const bool irap = static_cast<OaU8>(nalType) >= 16 && static_cast<OaU8>(nalType) <= 23;
				(void)reader.ReadBit(); // first_slice_segment_in_pic_flag
				if (irap) {
					reader.SkipBits(1);
				}
				ppsId = reader.ReadUE();
			}

			auto ppsIt = OaPpsCache_.Find(ppsId);
			if (ppsIt == OaPpsCache_.End()) {
				return OaStatus::Error("H.265 PPS not found in cache");
			}
			auto spsIt = OaSpsCache_.Find(ppsIt->second.SpsId);
			if (spsIt == OaSpsCache_.End()) {
				return OaStatus::Error("H.265 SPS not found in cache");
			}

			OaH265SliceHeader sliceHeader;
			if (!OaNalParser::ParseH265SliceHeader(
				data + nalUnit.Offset,
				nalUnit.Size,
				nalType,
				spsIt->second,
				ppsIt->second,
				sliceHeader)) {
				return OaStatus::Error("Failed to parse H.265 slice header");
			}
			if (!haveFirstSliceHeader) {
				firstSliceHeader = sliceHeader;
				firstSliceSps = spsIt->second;
				firstSlicePps = ppsIt->second;
				haveFirstSliceHeader = true;
			} else if (sliceHeader.PpsId != firstSliceHeader.PpsId ||
				sliceHeader.SpsId != firstSliceHeader.SpsId ||
				sliceHeader.PicOrderCntVal != firstSliceHeader.PicOrderCntVal ||
				sliceHeader.SliceType != firstSliceHeader.SliceType) {
				return OaStatus::Error(OaStatusCode::Unavailable, "H.265 mixed-picture slice access units are not implemented");
			}
			OutDesc.SliceOffsets.PushBack(static_cast<OaU32>(nalUnit.StartCodeOffset));
		}
	}

	if (!sawPicture) {
		return OaStatus::Ok(); // Parameter-set-only access unit
	}
	if (!haveFirstSliceHeader || OutDesc.SliceOffsets.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.265 picture has no decodable slices");
	}

	OutDesc.HasPicture = true;
	OutDesc.SliceHeader = firstSliceHeader;
	OutDesc.Sps = firstSliceSps;
	OutDesc.Pps = firstSlicePps;
	OutDesc.IsReference = firstSliceHeader.IsReference;
	return OaStatus::Ok();
}

namespace {
struct OaH265CodecRegistrar {
	OaH265CodecRegistrar() {
		auto parser = OaStdMakeUnique<OaVcpH265>();
		OaVideoCodecRegistry::GetInstance().RegisterParser(
			OaVideoCodec::H265,
			OaStdMove(parser));
	}
};
static OaH265CodecRegistrar g_H265Registrar __attribute__((used));
} // namespace

