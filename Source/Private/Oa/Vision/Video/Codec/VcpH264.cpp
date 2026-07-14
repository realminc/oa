// OA Vision — H.264 Codec Parser Implementation
// Extracts and converts H.264 parameter sets

#include "VcpH264.h"
#include "CodecRegistry.h"
#include "NalParser.h"
#include "BitstreamReader.h"
#include <Oa/Vision/VideoDecoder.h>

// ============================================================================
// Level Conversion
// ============================================================================

StdVideoH264LevelIdc OaVcpH264::ToStdH264Level(OaU32 InLevelIdc) {
	switch (InLevelIdc) {
		case 10: return STD_VIDEO_H264_LEVEL_IDC_1_0;
		case 11: return STD_VIDEO_H264_LEVEL_IDC_1_1;
		case 12: return STD_VIDEO_H264_LEVEL_IDC_1_2;
		case 13: return STD_VIDEO_H264_LEVEL_IDC_1_3;
		case 20: return STD_VIDEO_H264_LEVEL_IDC_2_0;
		case 21: return STD_VIDEO_H264_LEVEL_IDC_2_1;
		case 22: return STD_VIDEO_H264_LEVEL_IDC_2_2;
		case 30: return STD_VIDEO_H264_LEVEL_IDC_3_0;
		case 31: return STD_VIDEO_H264_LEVEL_IDC_3_1;
		case 32: return STD_VIDEO_H264_LEVEL_IDC_3_2;
		case 40: return STD_VIDEO_H264_LEVEL_IDC_4_0;
		case 41: return STD_VIDEO_H264_LEVEL_IDC_4_1;
		case 42: return STD_VIDEO_H264_LEVEL_IDC_4_2;
		case 50: return STD_VIDEO_H264_LEVEL_IDC_5_0;
		case 51: return STD_VIDEO_H264_LEVEL_IDC_5_1;
		case 52: return STD_VIDEO_H264_LEVEL_IDC_5_2;
		case 60: return STD_VIDEO_H264_LEVEL_IDC_6_0;
		case 61: return STD_VIDEO_H264_LEVEL_IDC_6_1;
		case 62: return STD_VIDEO_H264_LEVEL_IDC_6_2;
		default: return STD_VIDEO_H264_LEVEL_IDC_INVALID;
	}
}

// ============================================================================
// SPS Conversion
// ============================================================================

StdVideoH264SequenceParameterSet OaVcpH264::ToStdH264Sps(const OaH264SpsData& InSps) {
	StdVideoH264SequenceParameterSet sps = {};
	sps.flags.constraint_set0_flag = (InSps.ConstraintFlags & 0x80) != 0;
	sps.flags.constraint_set1_flag = (InSps.ConstraintFlags & 0x40) != 0;
	sps.flags.constraint_set2_flag = (InSps.ConstraintFlags & 0x20) != 0;
	sps.flags.constraint_set3_flag = (InSps.ConstraintFlags & 0x10) != 0;
	sps.flags.constraint_set4_flag = (InSps.ConstraintFlags & 0x08) != 0;
	sps.flags.constraint_set5_flag = (InSps.ConstraintFlags & 0x04) != 0;
	sps.flags.direct_8x8_inference_flag = InSps.Direct8x8Inference;
	sps.flags.mb_adaptive_frame_field_flag = InSps.MbAdaptiveFrameField;
	sps.flags.frame_mbs_only_flag = InSps.FrameMbsOnly;
	sps.flags.delta_pic_order_always_zero_flag = InSps.DeltaPicOrderAlwaysZero;
	sps.flags.separate_colour_plane_flag = InSps.SeparateColourPlane;
	sps.flags.gaps_in_frame_num_value_allowed_flag = InSps.GapsInFrameNumValueAllowed;
	sps.flags.qpprime_y_zero_transform_bypass_flag = InSps.QpprimeYZeroTransformBypass;
	sps.flags.frame_cropping_flag = InSps.FrameCropping;
	sps.profile_idc = static_cast<StdVideoH264ProfileIdc>(InSps.ProfileIdc);
	sps.level_idc = ToStdH264Level(InSps.LevelIdc);
	sps.chroma_format_idc = static_cast<StdVideoH264ChromaFormatIdc>(InSps.ChromaFormatIdc);
	sps.seq_parameter_set_id = static_cast<uint8_t>(InSps.SpsId);
	sps.bit_depth_luma_minus8 = static_cast<uint8_t>(InSps.BitDepthLumaMinus8);
	sps.bit_depth_chroma_minus8 = static_cast<uint8_t>(InSps.BitDepthChromaMinus8);
	sps.log2_max_frame_num_minus4 = static_cast<uint8_t>(InSps.Log2MaxFrameNumMinus4);
	sps.pic_order_cnt_type = static_cast<StdVideoH264PocType>(InSps.PicOrderCntType);
	sps.offset_for_non_ref_pic = InSps.OffsetForNonRefPic;
	sps.offset_for_top_to_bottom_field = InSps.OffsetForTopToBottomField;
	sps.log2_max_pic_order_cnt_lsb_minus4 = static_cast<uint8_t>(InSps.Log2MaxPicOrderCntLsbMinus4);
	sps.num_ref_frames_in_pic_order_cnt_cycle = static_cast<uint8_t>(InSps.NumRefFramesInPicOrderCntCycle);
	sps.max_num_ref_frames = static_cast<uint8_t>(InSps.MaxNumRefFrames);
	sps.pic_width_in_mbs_minus1 = InSps.PicWidthInMbs - 1;
	sps.pic_height_in_map_units_minus1 = InSps.PicHeightInMbs - 1;
	sps.frame_crop_left_offset = InSps.FrameCropLeftOffset;
	sps.frame_crop_right_offset = InSps.FrameCropRightOffset;
	sps.frame_crop_top_offset = InSps.FrameCropTopOffset;
	sps.frame_crop_bottom_offset = InSps.FrameCropBottomOffset;
	sps.pOffsetForRefFrame = InSps.NumRefFramesInPicOrderCntCycle > 0 ? InSps.OffsetForRefFrame.Data() : nullptr;
	return sps;
}

// ============================================================================
// PPS Conversion
// ============================================================================

StdVideoH264PictureParameterSet OaVcpH264::ToStdH264Pps(const OaH264PpsData& InPps) {
	StdVideoH264PictureParameterSet pps = {};
	pps.flags.transform_8x8_mode_flag = InPps.Transform8x8Mode;
	pps.flags.redundant_pic_cnt_present_flag = InPps.RedundantPicCntPresent;
	pps.flags.constrained_intra_pred_flag = InPps.ConstrainedIntraPred;
	pps.flags.deblocking_filter_control_present_flag = InPps.DeblockingFilterControlPresent;
	pps.flags.weighted_pred_flag = InPps.WeightedPred;
	pps.flags.bottom_field_pic_order_in_frame_present_flag = InPps.BottomFieldPicOrderInFramePresent;
	pps.flags.entropy_coding_mode_flag = InPps.EntropyCodingMode;
	pps.seq_parameter_set_id = static_cast<uint8_t>(InPps.SpsId);
	pps.pic_parameter_set_id = static_cast<uint8_t>(InPps.PpsId);
	pps.num_ref_idx_l0_default_active_minus1 = static_cast<uint8_t>(InPps.NumRefIdxL0DefaultActiveMinus1);
	pps.num_ref_idx_l1_default_active_minus1 = static_cast<uint8_t>(InPps.NumRefIdxL1DefaultActiveMinus1);
	pps.weighted_bipred_idc = static_cast<StdVideoH264WeightedBipredIdc>(InPps.WeightedBipredIdc);
	pps.pic_init_qp_minus26 = static_cast<int8_t>(InPps.PicInitQpMinus26);
	pps.pic_init_qs_minus26 = static_cast<int8_t>(InPps.PicInitQsMinus26);
	pps.chroma_qp_index_offset = static_cast<int8_t>(InPps.ChromaQpIndexOffset);
	pps.second_chroma_qp_index_offset = static_cast<int8_t>(InPps.SecondChromaQpIndexOffset);
	return pps;
}

// ============================================================================
// Parser Implementation
// ============================================================================

OaStatus OaVcpH264::ParseSps(const OaSpan<const OaU8>& InNal) {
	OaH264SpsData spsData;
	if (!OaNalParser::ParseSPS(InNal.Data(), InNal.Size(), spsData)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Failed to parse H.264 SPS");
	}

	// Cache the OA-native structure
	OaSpsCache_.Insert({spsData.SpsId, spsData});

	// Convert to Vulkan Video structure and cache it
	StdVideoH264SequenceParameterSet stdSps = ToStdH264Sps(spsData);
	
	// Store offset array if needed (must persist as long as stdSps is used)
	if (spsData.NumRefFramesInPicOrderCntCycle > 0) {
		OffsetForRefFrameStorage_.Insert({spsData.SpsId, spsData.OffsetForRefFrame});
		auto it = OffsetForRefFrameStorage_.Find(spsData.SpsId);
		if (it != OffsetForRefFrameStorage_.End()) {
			stdSps.pOffsetForRefFrame = it->second.Data();
		}
	}
	
	StdSpsCache_.Insert({spsData.SpsId, stdSps});
	
	return OaStatus::Ok();
}

OaStatus OaVcpH264::ParsePps(const OaSpan<const OaU8>& InNal) {
	OaH264PpsData ppsData;
	if (!OaNalParser::ParsePPS(InNal.Data(), InNal.Size(), ppsData)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Failed to parse H.264 PPS");
	}

	// Cache the OA-native structure
	OaPpsCache_.Insert({ppsData.PpsId, ppsData});

	// Convert to Vulkan Video structure and cache it
	StdVideoH264PictureParameterSet stdPps = ToStdH264Pps(ppsData);
	StdPpsCache_.Insert({ppsData.PpsId, stdPps});
	
	return OaStatus::Ok();
}

const StdVideoH264SequenceParameterSet* OaVcpH264::GetH264Sps(OaU32 InSpsId) const
{
	auto it = StdSpsCache_.Find(InSpsId);
	return (it != StdSpsCache_.End()) ? &it->second : nullptr;
}

const StdVideoH264PictureParameterSet* OaVcpH264::GetH264Pps(OaU32 InPpsId) const
{
	auto it = StdPpsCache_.Find(InPpsId);
	return (it != StdPpsCache_.End()) ? &it->second : nullptr;
}

const OaH264SpsData* OaVcpH264::GetOaSps(OaU32 InSpsId) const
{
	auto it = OaSpsCache_.Find(InSpsId);
	return (it != OaSpsCache_.End()) ? &it->second : nullptr;
}

const OaH264PpsData* OaVcpH264::GetOaPps(OaU32 InPpsId) const
{
	auto it = OaPpsCache_.Find(InPpsId);
	return (it != OaPpsCache_.End()) ? &it->second : nullptr;
}

void OaVcpH264::ClearParameterSets()
{
	OaSpsCache_.Clear();
	OaPpsCache_.Clear();
	StdSpsCache_.Clear();
	StdPpsCache_.Clear();
	OffsetForRefFrameStorage_.Clear();
}

OaVec<OaU32> OaVcpH264::GetCachedSpsIds() const {
	OaVec<OaU32> ids;
	ids.Reserve(OaSpsCache_.Size());
	for (const auto& kv : OaSpsCache_) ids.PushBack(kv.first);
	return ids;
}

OaVec<OaU32> OaVcpH264::GetCachedPpsIds() const {
	OaVec<OaU32> ids;
	ids.Reserve(OaPpsCache_.Size());
	for (const auto& kv : OaPpsCache_) ids.PushBack(kv.first);
	return ids;
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

	OaStatus FindAnnexBNalUnits(const OaSpan<const OaU8>& InBitstream, OaVec<AnnexBNalUnit>& OutNalUnits)
	{
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
			return OaStatus::Error("Invalid H.264 Annex-B bitstream - no NAL start code found");
		}
		return OaStatus::Ok();
	}
} // namespace

OaStatus OaVcpH264::ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaH264PictureDesc& OutDesc)
{
	OutDesc.HasPicture = false;
	OutDesc.SliceStartCodeOffset = 0;
	OutDesc.SliceStartCodeSize = 0;
	OutDesc.SliceNalSize = 0;

	OaVec<AnnexBNalUnit> nalUnits;
	OaStatus findStatus = FindAnnexBNalUnits(InBitstream, nalUnits);
	if (!findStatus.IsOk()) {
		return findStatus;
	}

	const OaU8* data = InBitstream.Data();
	bool sawPicture = false;

	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.Size == 0) {
			continue;
		}

		OaU8 refIdc = 0;
		OaH264NalType nalType = OaNalParser::ParseNalHeader(data[nalUnit.Offset], refIdc);

		if (nalType == OaH264NalType::SPS) {
			OaH264SpsData sps;
			if (!OaNalParser::ParseSPS(data + nalUnit.Offset, nalUnit.Size, sps)) {
				return OaStatus::Error("Failed to parse H.264 SPS");
			}
			OaSpsCache_.Insert({sps.SpsId, sps});
			auto stdSps = ToStdH264Sps(sps);
			StdSpsCache_.Insert({sps.SpsId, stdSps});
			continue;
		}
		if (nalType == OaH264NalType::PPS) {
			OaH264PpsData pps;
			if (!OaNalParser::ParsePPS(data + nalUnit.Offset, nalUnit.Size, pps)) {
				return OaStatus::Error("Failed to parse H.264 PPS");
			}
			OaPpsCache_.Insert({pps.PpsId, pps});
			auto stdPps = ToStdH264Pps(pps);
			StdPpsCache_.Insert({pps.PpsId, stdPps});
			continue;
		}
		if (nalType != OaH264NalType::IDR && nalType != OaH264NalType::NonIDR) {
			continue;
		}

		if (sawPicture) {
			return OaStatus::Error(OaStatusCode::Unavailable, "Only one H.264 slice per access unit is supported currently");
		}
		sawPicture = true;

		OaU32 ppsId = 0;
		const bool isIdr = nalType == OaH264NalType::IDR;
		if (!OaNalParser::ParseSliceHeaderPrefix(data + nalUnit.Offset, nalUnit.Size, isIdr, ppsId)) {
			return OaStatus::Error("Failed to parse H.264 slice header prefix");
		}

		auto ppsIt = OaPpsCache_.Find(ppsId);
		if (ppsIt == OaPpsCache_.End()) {
			return OaStatus::Error("H.264 PPS not found in cache");
		}
		auto spsIt = OaSpsCache_.Find(ppsIt->second.SpsId);
		if (spsIt == OaSpsCache_.End()) {
			return OaStatus::Error("H.264 SPS not found in cache");
		}

		OaSliceHeader sliceHeader;
		if (!OaNalParser::ParseSliceHeader(data + nalUnit.Offset, nalUnit.Size, isIdr, refIdc, spsIt->second, ppsIt->second, sliceHeader)) {
			return OaStatus::Error("Failed to parse H.264 slice header");
		}

		OutDesc.HasPicture = true;
		OutDesc.SliceHeader = sliceHeader;
		OutDesc.Sps = spsIt->second;
		OutDesc.Pps = ppsIt->second;
		OutDesc.SliceStartCodeOffset = static_cast<OaU32>(nalUnit.StartCodeOffset);
		OutDesc.SliceStartCodeSize = static_cast<OaU32>(nalUnit.StartCodeSize);
		OutDesc.SliceNalSize = static_cast<OaU32>(nalUnit.Size);
	}

	if (!sawPicture) {
		return OaStatus::Ok(); // Parameter-set-only access unit
	}
	if (!OutDesc.HasPicture) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "H.264 picture has no decodable slices");
	}
	return OaStatus::Ok();
}

namespace {
struct OaH264CodecRegistrar {
	OaH264CodecRegistrar() {
		auto parser = OaStdMakeUnique<OaVcpH264>();
		OaVideoCodecRegistry::GetInstance().RegisterParser(
			OaVideoCodec::H264,
			OaStdMove(parser));
	}
};
static OaH264CodecRegistrar g_H264Registrar __attribute__((used));
} // namespace

