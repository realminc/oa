// OA Vision — H.264 codec parser Implementation
// Extracts and converts H.264 parameter sets

#include "vcpH264.h"
#include "codecRegistry.h"
#include "nalParser.h"
#include "bitstreamReader.h"
#include <oa/vision/videoDecoder.h>

// ============================================================================
// level conversion
// ============================================================================

StdVideoH264LevelIdc oa::VcpH264::toStdH264Level(oa::U32 inLevelIdc) {
	switch (inLevelIdc) {
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
// SPS conversion
// ============================================================================

StdVideoH264SequenceParameterSet oa::VcpH264::toStdH264Sps(const oa::H264SpsData& inSps) {
	StdVideoH264SequenceParameterSet sps = {};
	sps.flags.constraint_set0_flag = (inSps.constraintFlags & 0x80) != 0;
	sps.flags.constraint_set1_flag = (inSps.constraintFlags & 0x40) != 0;
	sps.flags.constraint_set2_flag = (inSps.constraintFlags & 0x20) != 0;
	sps.flags.constraint_set3_flag = (inSps.constraintFlags & 0x10) != 0;
	sps.flags.constraint_set4_flag = (inSps.constraintFlags & 0x08) != 0;
	sps.flags.constraint_set5_flag = (inSps.constraintFlags & 0x04) != 0;
	sps.flags.direct_8x8_inference_flag = inSps.direct8x8Inference;
	sps.flags.mb_adaptive_frame_field_flag = inSps.mbAdaptiveFrameField;
	sps.flags.frame_mbs_only_flag = inSps.frameMbsOnly;
	sps.flags.delta_pic_order_always_zero_flag = inSps.deltaPicOrderAlwaysZero;
	sps.flags.separate_colour_plane_flag = inSps.separateColourPlane;
	sps.flags.gaps_in_frame_num_value_allowed_flag = inSps.gapsInFrameNumValueAllowed;
	sps.flags.qpprime_y_zero_transform_bypass_flag = inSps.qpprimeYZeroTransformBypass;
	sps.flags.frame_cropping_flag = inSps.frameCropping;
	sps.profile_idc = static_cast<StdVideoH264ProfileIdc>(inSps.profileIdc);
	sps.level_idc = toStdH264Level(inSps.levelIdc);
	sps.chroma_format_idc = static_cast<StdVideoH264ChromaFormatIdc>(inSps.chromaFormatIdc);
	sps.seq_parameter_set_id = static_cast<uint8_t>(inSps.spsId);
	sps.bit_depth_luma_minus8 = static_cast<uint8_t>(inSps.bitDepthLumaMinus8);
	sps.bit_depth_chroma_minus8 = static_cast<uint8_t>(inSps.bitDepthChromaMinus8);
	sps.log2_max_frame_num_minus4 = static_cast<uint8_t>(inSps.log2MaxFrameNumMinus4);
	sps.pic_order_cnt_type = static_cast<StdVideoH264PocType>(inSps.picOrderCntType);
	sps.offset_for_non_ref_pic = inSps.offsetForNonRefPic;
	sps.offset_for_top_to_bottom_field = inSps.offsetForTopToBottomField;
	sps.log2_max_pic_order_cnt_lsb_minus4 = static_cast<uint8_t>(inSps.log2MaxPicOrderCntLsbMinus4);
	sps.num_ref_frames_in_pic_order_cnt_cycle = static_cast<uint8_t>(inSps.numRefFramesInPicOrderCntCycle);
	sps.max_num_ref_frames = static_cast<uint8_t>(inSps.maxNumRefFrames);
	sps.pic_width_in_mbs_minus1 = inSps.picWidthInMbs - 1;
	sps.pic_height_in_map_units_minus1 = inSps.picHeightInMbs - 1;
	sps.frame_crop_left_offset = inSps.frameCropLeftOffset;
	sps.frame_crop_right_offset = inSps.frameCropRightOffset;
	sps.frame_crop_top_offset = inSps.frameCropTopOffset;
	sps.frame_crop_bottom_offset = inSps.frameCropBottomOffset;
	sps.pOffsetForRefFrame = inSps.numRefFramesInPicOrderCntCycle > 0 ? inSps.offsetForRefFrame.data() : nullptr;
	return sps;
}

// ============================================================================
// PPS conversion
// ============================================================================

StdVideoH264PictureParameterSet oa::VcpH264::toStdH264Pps(const oa::H264PpsData& inPps) {
	StdVideoH264PictureParameterSet pps = {};
	pps.flags.transform_8x8_mode_flag = inPps.transform8x8Mode;
	pps.flags.redundant_pic_cnt_present_flag = inPps.redundantPicCntPresent;
	pps.flags.constrained_intra_pred_flag = inPps.constrainedIntraPred;
	pps.flags.deblocking_filter_control_present_flag = inPps.deblockingFilterControlPresent;
	pps.flags.weighted_pred_flag = inPps.weightedPred;
	pps.flags.bottom_field_pic_order_in_frame_present_flag = inPps.bottomFieldPicOrderInFramePresent;
	pps.flags.entropy_coding_mode_flag = inPps.entropyCodingMode;
	pps.seq_parameter_set_id = static_cast<uint8_t>(inPps.spsId);
	pps.pic_parameter_set_id = static_cast<uint8_t>(inPps.ppsId);
	pps.num_ref_idx_l0_default_active_minus1 = static_cast<uint8_t>(inPps.numRefIdxL0DefaultActiveMinus1);
	pps.num_ref_idx_l1_default_active_minus1 = static_cast<uint8_t>(inPps.numRefIdxL1DefaultActiveMinus1);
	pps.weighted_bipred_idc = static_cast<StdVideoH264WeightedBipredIdc>(inPps.weightedBipredIdc);
	pps.pic_init_qp_minus26 = static_cast<int8_t>(inPps.picInitQpMinus26);
	pps.pic_init_qs_minus26 = static_cast<int8_t>(inPps.picInitQsMinus26);
	pps.chroma_qp_index_offset = static_cast<int8_t>(inPps.chromaQpIndexOffset);
	pps.second_chroma_qp_index_offset = static_cast<int8_t>(inPps.secondChromaQpIndexOffset);
	return pps;
}

// ============================================================================
// parser Implementation
// ============================================================================

oa::Status oa::VcpH264::parseSps(const oa::Span<const oa::U8>& inNal) {
	oa::H264SpsData spsData;
	if (!oa::NalParser::parseSPS(inNal.data(), inNal.size(), spsData)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Failed to parse H.264 SPS");
	}

	// cache the OA-native structure
	oaSpsCache_.insert({spsData.spsId, spsData});

	// convert to vulkan Video structure and cache it
	StdVideoH264SequenceParameterSet stdSps = toStdH264Sps(spsData);
	
	// store offset array if needed (must persist as long as stdSps is used)
	if (spsData.numRefFramesInPicOrderCntCycle > 0) {
		offsetForRefFrameStorage_.insert({spsData.spsId, spsData.offsetForRefFrame});
		auto it = offsetForRefFrameStorage_.find(spsData.spsId);
		if (it != offsetForRefFrameStorage_.end()) {
			stdSps.pOffsetForRefFrame = it->second.data();
		}
	}
	
	stdSpsCache_.insert({spsData.spsId, stdSps});
	
	return oa::Status::ok();
}

oa::Status oa::VcpH264::parsePps(const oa::Span<const oa::U8>& inNal) {
	oa::H264PpsData ppsData;
	if (!oa::NalParser::parsePPS(inNal.data(), inNal.size(), ppsData)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Failed to parse H.264 PPS");
	}

	// cache the OA-native structure
	oaPpsCache_.insert({ppsData.ppsId, ppsData});

	// convert to vulkan Video structure and cache it
	StdVideoH264PictureParameterSet stdPps = toStdH264Pps(ppsData);
	stdPpsCache_.insert({ppsData.ppsId, stdPps});
	
	return oa::Status::ok();
}

const StdVideoH264SequenceParameterSet* oa::VcpH264::getH264Sps(oa::U32 inSpsId) const
{
	auto it = stdSpsCache_.find(inSpsId);
	return (it != stdSpsCache_.end()) ? &it->second : nullptr;
}

const StdVideoH264PictureParameterSet* oa::VcpH264::getH264Pps(oa::U32 inPpsId) const
{
	auto it = stdPpsCache_.find(inPpsId);
	return (it != stdPpsCache_.end()) ? &it->second : nullptr;
}

const oa::H264SpsData* oa::VcpH264::getSpsData(oa::U32 inSpsId) const
{
	auto it = oaSpsCache_.find(inSpsId);
	return (it != oaSpsCache_.end()) ? &it->second : nullptr;
}

const oa::H264PpsData* oa::VcpH264::getPpsData(oa::U32 inPpsId) const
{
	auto it = oaPpsCache_.find(inPpsId);
	return (it != oaPpsCache_.end()) ? &it->second : nullptr;
}

void oa::VcpH264::clearParameterSets()
{
	oaSpsCache_.clear();
	oaPpsCache_.clear();
	stdSpsCache_.clear();
	stdPpsCache_.clear();
	offsetForRefFrameStorage_.clear();
}

oa::Vector<oa::U32> oa::VcpH264::getCachedSpsIds() const {
	oa::Vector<oa::U32> ids;
	ids.reserve(oaSpsCache_.size());
	for (const auto& kv : oaSpsCache_) ids.pushBack(kv.first);
	return ids;
}

oa::Vector<oa::U32> oa::VcpH264::getCachedPpsIds() const {
	oa::Vector<oa::U32> ids;
	ids.reserve(oaPpsCache_.size());
	for (const auto& kv : oaPpsCache_) ids.pushBack(kv.first);
	return ids;
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

	oa::Status findAnnexBNalUnits(const oa::Span<const oa::U8>& inBitstream, oa::Vector<AnnexBNalUnit>& outNalUnits)
	{
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
			return oa::Status::error("Invalid H.264 Annex-B bitstream - no NAL start code found");
		}
		return oa::Status::ok();
	}
} // namespace

oa::Status oa::VcpH264::parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, oa::H264PictureDesc& outDesc)
{
	outDesc.hasPicture = false;
	outDesc.sliceStartCodeOffset = 0;
	outDesc.sliceStartCodeSize = 0;
	outDesc.sliceNalSize = 0;

	oa::Vector<AnnexBNalUnit> nalUnits;
	oa::Status findStatus = findAnnexBNalUnits(inBitstream, nalUnits);
	if (!findStatus.isOk()) {
		return findStatus;
	}

	const oa::U8* data = inBitstream.data();
	bool sawPicture = false;

	for (const AnnexBNalUnit& nalUnit : nalUnits) {
		if (nalUnit.size == 0) {
			continue;
		}

		oa::U8 refIdc = 0;
		oa::H264NalType nalType = oa::NalParser::parseNalHeader(data[nalUnit.offset], refIdc);

		if (nalType == oa::H264NalType::SPS) {
			oa::H264SpsData sps;
			if (!oa::NalParser::parseSPS(data + nalUnit.offset, nalUnit.size, sps)) {
				return oa::Status::error("Failed to parse H.264 SPS");
			}
			oaSpsCache_.insert({sps.spsId, sps});
			auto stdSps = toStdH264Sps(sps);
			stdSpsCache_.insert({sps.spsId, stdSps});
			continue;
		}
		if (nalType == oa::H264NalType::PPS) {
			oa::H264PpsData pps;
			if (!oa::NalParser::parsePPS(data + nalUnit.offset, nalUnit.size, pps)) {
				return oa::Status::error("Failed to parse H.264 PPS");
			}
			oaPpsCache_.insert({pps.ppsId, pps});
			auto stdPps = toStdH264Pps(pps);
			stdPpsCache_.insert({pps.ppsId, stdPps});
			continue;
		}
		if (nalType != oa::H264NalType::IDR && nalType != oa::H264NalType::NonIDR) {
			continue;
		}

		if (sawPicture) {
			return oa::Status::error(oa::StatusCode::Unavailable, "Only one H.264 slice per access unit is supported currently");
		}
		sawPicture = true;

		oa::U32 ppsId = 0;
		const bool isIdr = nalType == oa::H264NalType::IDR;
		if (!oa::NalParser::parseSliceHeaderPrefix(data + nalUnit.offset, nalUnit.size, isIdr, ppsId)) {
			return oa::Status::error("Failed to parse H.264 slice header prefix");
		}

		auto ppsIt = oaPpsCache_.find(ppsId);
		if (ppsIt == oaPpsCache_.end()) {
			return oa::Status::error("H.264 PPS not found in cache");
		}
		auto spsIt = oaSpsCache_.find(ppsIt->second.spsId);
		if (spsIt == oaSpsCache_.end()) {
			return oa::Status::error("H.264 SPS not found in cache");
		}

		oa::H264SliceHeader sliceHeader;
		if (!oa::NalParser::parseSliceHeader(data + nalUnit.offset, nalUnit.size, isIdr, refIdc, spsIt->second, ppsIt->second, sliceHeader)) {
			return oa::Status::error("Failed to parse H.264 slice header");
		}

		outDesc.hasPicture = true;
		outDesc.sliceHeader = sliceHeader;
		outDesc.sps = spsIt->second;
		outDesc.pps = ppsIt->second;
		outDesc.sliceStartCodeOffset = static_cast<oa::U32>(nalUnit.startCodeOffset);
		outDesc.sliceStartCodeSize = static_cast<oa::U32>(nalUnit.startCodeSize);
		outDesc.sliceNalSize = static_cast<oa::U32>(nalUnit.size);
	}

	if (!sawPicture) {
		return oa::Status::ok(); // Parameter-set-only access unit
	}
	if (!outDesc.hasPicture) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "H.264 picture has no decodable slices");
	}
	return oa::Status::ok();
}

namespace {
struct H264CodecRegistrar {
	H264CodecRegistrar() {
		auto parser = oa::makeUnique<oa::VcpH264>();
		oa::VideoCodecRegistry::getInstance().registerParser(
			oa::VideoCodec::H264,
			oa::move(parser));
	}
};
static H264CodecRegistrar g_H264Registrar __attribute__((used));
} // namespace

