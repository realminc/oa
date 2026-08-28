// OA Vision — H.264/H.265 NAL Unit parser
// parse SPS, PPS, and slice headers

#pragma once

#include <oa/core/types.h>
#include <oa/vision/videoCodecParameterSets.h>
#include <vk_video/vulkan_video_codec_h265std_decode.h>
#include "bitstreamReader.h"

namespace oa {

// H.264 NAL unit types
enum class H264NalType : oa::U8 {
	Unspecified = 0,
	NonIDR = 1,      // Non-IDR slice
	IDR = 5,         // IDR slice
	SEI = 6,         // Supplemental Enhancement Information
	SPS = 7,         // sequence Parameter set
	PPS = 8,         // Picture Parameter set
	AUD = 9,         // access Unit Delimiter
};

// H.264 slice types
enum class H264SliceType : oa::U8 {
	P = 0,   // P slice (predicted)
	B = 1,   // B slice (bi-predicted)
	I = 2,   // I slice (intra)
	SP = 3,  // SP slice (switching P)
	SI = 4,  // SI slice (switching I)
};

enum class H265NalType : oa::U8 {
	TrailN = 0,
	TrailR = 1,
	TsaN = 2,
	TsaR = 3,
	StsaN = 4,
	StsaR = 5,
	RadlN = 6,
	RadlR = 7,
	RaslN = 8,
	RaslR = 9,
	BlaWLp = 16,
	BlaWRadl = 17,
	BlaNLp = 18,
	IdrWRadl = 19,
	IdrNLp = 20,
	CraNut = 21,
	Vps = 32,
	Sps = 33,
	Pps = 34,
	Aud = 35,
	Eos = 36,
	Eob = 37,
	Fd = 38,
	PrefixSei = 39,
	SuffixSei = 40,
};

// MMCO (memory_management_control_operation) command, 14496-10 §7.3.3.3.
struct H264MmcoCommand {
	oa::U32 op = 0;                       // memory_management_control_operation
	oa::U32 differenceOfPicNumsMinus1 = 0; // op 1, 3
	oa::U32 longTermPicNum = 0;            // op 2
	oa::U32 longTermFrameIdx = 0;          // op 3, 6
	oa::U32 maxLongTermFrameIdxPlus1 = 0;  // op 4
};

// parsed slice header (minimal fields for phase 2.4.3)
struct H264SliceHeader {
	oa::U32 firstMbInSlice;
	H264SliceType sliceType;
	oa::U32 ppsId;
	oa::U32 frameNum;
	oa::U32 idrPicId = 0;
	oa::I32 picOrderCntLsb;
	bool fieldPicFlag = false;
	bool bottomFieldFlag = false;
	bool isIdrPic;
	bool isReference;

	// dec_ref_pic_marking() — populated when ref_pic_marking parsing succeeded.
	// For IDR slices: noOutputOfPriorPics + longTermReference flags.
	// For non-IDR: adaptiveRefPicMarking + mmcoCommands list.
	bool noOutputOfPriorPics       = false;
	bool longTermReference         = false;
	bool adaptiveRefPicMarking     = false;
	oa::Vector<H264MmcoCommand> mmcoCommands;
	bool refPicMarkingValid        = false;
};

struct H265SliceHeader {
	oa::U32 ppsId = 0;
	oa::U32 spsId = 0;
	oa::U32 vpsId = 0;
	StdVideoH265SliceType sliceType = STD_VIDEO_H265_SLICE_TYPE_I;
	oa::U32 picOrderCntLsb = 0;
	oa::I32 picOrderCntVal = 0;
	oa::U16 numBitsForSTRefPicSetInSlice = 0;
	oa::Vector<oa::I32> stCurrBeforeDeltaPocs;
	oa::Vector<oa::I32> stCurrAfterDeltaPocs;
	oa::Vector<oa::I32> stFollDeltaPocs;
	oa::U8 nalUnitType = 0;
	oa::U8 temporalId = 0;
	bool firstSliceSegmentInPic = true;
	bool isIrap = false;
	bool isIdr = false;
	bool noOutputOfPriorPics = false;
	bool isReference = true;
	bool shortTermRefPicSetSpsFlag = false;
};

// NAL unit parser
class NalParser {
public:
	static oa::Vector<oa::U8> makeRbsp(const oa::U8* inData, oa::Usize inSize)	{
		oa::Vector<oa::U8> rbsp;
		rbsp.reserve(inSize);
		oa::U32 zeroRun = 0;
		for (oa::Usize i = 0; i < inSize; ++i)
		{
			const oa::U8 byte = inData[i];
			if (zeroRun >= 2 && byte == 0x03)
			{
				zeroRun = 0;
				continue;
			}
			rbsp.pushBack(byte);
			zeroRun = byte == 0 ? zeroRun + 1 : 0;
		}
		return rbsp;
	}

	static bool moreRbspData(const BitstreamReader& inReader, const oa::Vector<oa::U8>& inRbsp) {
		if (inRbsp.empty()) {
			return false;
		}
		oa::I64 stopBitOffset = -1;
		for (oa::I64 byteIndex = static_cast<oa::I64>(inRbsp.size()) - 1; byteIndex >= 0 && stopBitOffset < 0; --byteIndex)	{
			const oa::U8 value = inRbsp[static_cast<oa::Usize>(byteIndex)];
			for (oa::I32 bit = 0; bit < 8; ++bit)	{
				if ((value & (1u << bit)) != 0)	{
					stopBitOffset = byteIndex * 8 + (7 - bit);
					break;
				}
			}
		}
		if (stopBitOffset < 0) {
			return false;
		}
		const oa::Usize currentBitOffset = inReader.getBytePos() * 8 + inReader.getBitPos();
		return currentBitOffset < static_cast<oa::Usize>(stopBitOffset);
	}

	static oa::Usize getBitOffset(const BitstreamReader& inReader) {
		return inReader.getBytePos() * 8 + inReader.getBitPos();
	}

	static oa::U32 ceilLog2(oa::U32 inValue) {
		if (inValue <= 1) {
			return 0;
		}
		oa::U32 bits = 0;
		--inValue;
		while (inValue > 0) {
			++bits;
			inValue >>= 1;
		}
		return bits;
	}

	static void skipScalingList(BitstreamReader& inReader, oa::U32 inSize) {
		oa::I32 lastScale = 8;
		oa::I32 nextScale = 8;
		for (oa::U32 j = 0; j < inSize; ++j) {
			if (nextScale != 0)	{
				const oa::I32 deltaScale = inReader.readSE();
				nextScale = (lastScale + deltaScale + 256) % 256;
			}
			lastScale = nextScale == 0 ? lastScale : nextScale;
		}
	}

	static bool skipH264PredWeightTable(
		BitstreamReader& inReader,
		const H264SpsData& inSps,
		oa::U32 inNumRefIdxL0ActiveMinus1,
		oa::U32 inNumRefIdxL1ActiveMinus1,
		bool inHasList1
	)	{
		// H.264 limits each reference list to 32 frame references. Reject
		// impossible values instead of letting a damaged header drive an
		// unbounded parser walk.
		if (inNumRefIdxL0ActiveMinus1 >= 32
			|| (inHasList1 && inNumRefIdxL1ActiveMinus1 >= 32)) {
			return false;
		}

		(void)inReader.readUE(); // luma_log2_weight_denom
		const oa::U32 chromaArrayType = inSps.separateColourPlane
			? 0U
			: inSps.chromaFormatIdc;
		if (chromaArrayType != 0) {
			(void)inReader.readUE(); // chroma_log2_weight_denom
		}

		auto skipList = [&](oa::U32 inNumRefIdxActiveMinus1) {
			for (oa::U32 i = 0; i <= inNumRefIdxActiveMinus1; ++i) {
				if (inReader.readBit() != 0) {
					(void)inReader.readSE(); // luma_weight
					(void)inReader.readSE(); // luma_offset
				}
				if (chromaArrayType != 0 && inReader.readBit() != 0) {
					for (oa::U32 component = 0; component < 2; ++component) {
						(void)inReader.readSE(); // chroma_weight
						(void)inReader.readSE(); // chroma_offset
					}
				}
			}
		};

		skipList(inNumRefIdxL0ActiveMinus1);
		if (inHasList1) {
			skipList(inNumRefIdxL1ActiveMinus1);
		}
		return true;
	}

	static void readH265ProfileTierLevel(
		BitstreamReader& inReader,
		oa::U32 inMaxSubLayersMinus1,
		H265VpsData* outVps,
		H265SpsData* outSps
	)	{
		inReader.skipBits(2);  // general_profile_space
		const bool tierFlag = inReader.readBit() != 0;
		const oa::U32 profileIdc = inReader.readBits(5);
		inReader.skipBits(32); // general_profile_compatibility_flags
		const bool progressive = inReader.readBit() != 0;
		const bool interlaced = inReader.readBit() != 0;
		const bool nonPacked = inReader.readBit() != 0;
		const bool frameOnly = inReader.readBit() != 0;
		inReader.skipBits(44); // reserved and constraint flags
		const oa::U32 levelIdc = inReader.readBits(8);
		if (outVps) {
			outVps->generalTierFlag = tierFlag;
			outVps->generalProfileIdc = profileIdc;
			outVps->generalLevelIdc = levelIdc;
			outVps->generalProgressiveSourceFlag = progressive;
			outVps->generalInterlacedSourceFlag = interlaced;
			outVps->generalNonPackedConstraintFlag = nonPacked;
			outVps->generalFrameOnlyConstraintFlag = frameOnly;
		}
		(void)outSps;

		oa::Array<bool, 8> subLayerProfilePresent = {};
		oa::Array<bool, 8> subLayerLevelPresent = {};
		for (oa::U32 i = 0; i < inMaxSubLayersMinus1 && i < subLayerProfilePresent.size(); ++i) {
			subLayerProfilePresent[i] = inReader.readBit() != 0;
			subLayerLevelPresent[i] = inReader.readBit() != 0;
		}
		if (inMaxSubLayersMinus1 > 0) {
			for (oa::U32 i = inMaxSubLayersMinus1; i < 8; ++i) {
				inReader.skipBits(2); // reserved_zero_2bits
			}
		}
		for (oa::U32 i = 0; i < inMaxSubLayersMinus1 && i < subLayerProfilePresent.size(); ++i) {
			if (subLayerProfilePresent[i]) {
				inReader.skipBits(2);
				inReader.skipBits(1);
				inReader.skipBits(5);
				inReader.skipBits(32);
				inReader.skipBits(4);
				inReader.skipBits(44);
			}
			if (subLayerLevelPresent[i]) {
				inReader.skipBits(8);
			}
		}
	}

	static void skipH265ProfileTierLevel(BitstreamReader& inReader, oa::U32 inMaxSubLayersMinus1)	{
		readH265ProfileTierLevel(inReader, inMaxSubLayersMinus1, nullptr, nullptr);
	}

	static void skipH265ScalingListData(BitstreamReader& inReader) {
		for (oa::U32 sizeId = 0; sizeId < 4; ++sizeId) {
			const oa::U32 matrixCount = sizeId == 3 ? 2u : 6u;
			for (oa::U32 matrixId = 0; matrixId < matrixCount; ++matrixId) {
				if (inReader.readBit() == 0) {
					(void)inReader.readUE(); // scaling_list_pred_matrix_id_delta
					continue;
				}
				const oa::U32 coefCount = sizeId == 0 ? 16u : 64u;
				if (sizeId > 1) {
					(void)inReader.readSE(); // scaling_list_dc_coef_minus8
				}
				for (oa::U32 i = 0; i < coefCount; ++i) {
					(void)inReader.readSE(); // scaling_list_delta_coef
				}
			}
		}
	}

	static void skipH265ShortTermRefPicSet(
		BitstreamReader& inReader,
		oa::U32 inSetIndex,
		oa::U32 inSetCount,
		oa::Array<oa::U32, 64>& inOutNegativeCounts,
		oa::Array<oa::U32, 64>& inOutPositiveCounts
	)	{
		if (inSetIndex != 0 && inReader.readBit() != 0) {
			if (inSetIndex == inSetCount) {
				(void)inReader.readUE(); // delta_idx_minus1
			}
			inReader.skipBits(1); // delta_rps_sign
			(void)inReader.readUE(); // abs_delta_rps_minus1
			const oa::U32 refIndex = inSetIndex > 0 ? inSetIndex - 1 : 0;
			const oa::U32 deltaCount = inOutNegativeCounts[refIndex] + inOutPositiveCounts[refIndex];
			for (oa::U32 j = 0; j <= deltaCount; ++j) {
				const bool usedByCurrPic = inReader.readBit() != 0;
				if (!usedByCurrPic) {
					inReader.skipBits(1); // use_delta_flag
				}
			}
			inOutNegativeCounts[inSetIndex] = 0;
			inOutPositiveCounts[inSetIndex] = 0;
			return;
		}
		const oa::U32 negativeCount = inReader.readUE();
		const oa::U32 positiveCount = inReader.readUE();
		inOutNegativeCounts[inSetIndex] = negativeCount;
		inOutPositiveCounts[inSetIndex] = positiveCount;
		for (oa::U32 i = 0; i < negativeCount; ++i) {
			(void)inReader.readUE(); // delta_poc_s0_minus1
			inReader.skipBits(1); // used_by_curr_pic_s0_flag
		}
		for (oa::U32 i = 0; i < positiveCount; ++i) {
			(void)inReader.readUE(); // delta_poc_s1_minus1
			inReader.skipBits(1); // used_by_curr_pic_s1_flag
		}
	}

public:
	// parse NAL unit header (first byte after start code)
	static H264NalType parseNalHeader(oa::U8 inNalByte, oa::U8& outRefIdc) {
		// NAL unit header: forbidden_zero_bit(1) + nal_ref_idc(2) + nal_unit_type(5)
		outRefIdc = (inNalByte >> 5) & 0x3;
		return static_cast<H264NalType>(inNalByte & 0x1F);
	}

	static H265NalType parseH265NalHeader(const oa::U8* inData, oa::Usize inSize, oa::U8& outTemporalId) {
		outTemporalId = 0;
		if (inSize < 2) {
			return H265NalType::TrailN;
		}
		const oa::U8 temporalIdPlus1 = static_cast<oa::U8>(inData[1] & 0x07u);
		if (temporalIdPlus1 == 0) {
			return H265NalType::TrailN;
		}
		outTemporalId = static_cast<oa::U8>(temporalIdPlus1 - 1u);
		return static_cast<H265NalType>((inData[0] >> 1) & 0x3Fu);
	}

	static bool parseH265Vps(const oa::U8* inData, oa::Usize inSize, H265VpsData& outVps) {
		if (inSize < 2) {
			return false;
		}
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		reader.skipBits(16); // HEVC NAL header
		outVps.vpsId = reader.readBits(4);
		reader.skipBits(2); // base_layer_internal_flag + base_layer_available_flag
		reader.skipBits(6); // vps_max_layers_minus1
		outVps.maxSubLayersMinus1 = reader.readBits(3);
		outVps.temporalIdNesting = reader.readBit() != 0;
		reader.skipBits(16); // vps_reserved_0xffff_16bits
		readH265ProfileTierLevel(reader, outVps.maxSubLayersMinus1, &outVps, nullptr);
		return true;
	}

	static bool parseH265Sps(const oa::U8* inData, oa::Usize inSize, H265SpsData& outSps) {
		if (inSize < 2) {
			return false;
		}
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		reader.skipBits(16); // HEVC NAL header
		outSps.vpsId = reader.readBits(4);
		outSps.maxSubLayersMinus1 = reader.readBits(3);
		outSps.temporalIdNesting = reader.readBit() != 0;
		skipH265ProfileTierLevel(reader, outSps.maxSubLayersMinus1);
		outSps.spsId = reader.readUE();
		outSps.chromaFormatIdc = reader.readUE();
		if (outSps.chromaFormatIdc == 3) {
			outSps.separateColourPlane = reader.readBit() != 0;
		}
		outSps.width = reader.readUE();
		outSps.height = reader.readUE();
		outSps.codedWidth = outSps.width;
		outSps.codedHeight = outSps.height;
		if (reader.readBit() != 0) {
			outSps.conformanceWindowLeft = reader.readUE();
			outSps.conformanceWindowRight = reader.readUE();
			outSps.conformanceWindowTop = reader.readUE();
			outSps.conformanceWindowBottom = reader.readUE();
			const oa::U32 cropUnitX = outSps.chromaFormatIdc == 1 ? 2u : 1u;
			const oa::U32 cropUnitY = outSps.chromaFormatIdc == 1 ? 2u : 1u;
			const oa::U32 cropWidth = (outSps.conformanceWindowLeft + outSps.conformanceWindowRight) * cropUnitX;
			const oa::U32 cropHeight = (outSps.conformanceWindowTop + outSps.conformanceWindowBottom) * cropUnitY;
			if (cropWidth < outSps.width) {
				outSps.width -= cropWidth;
			}
			if (cropHeight < outSps.height) {
				outSps.height -= cropHeight;
			}
		}
		outSps.bitDepthLumaMinus8 = reader.readUE();
		outSps.bitDepthChromaMinus8 = reader.readUE();
		outSps.log2MaxPicOrderCntLsbMinus4 = reader.readUE();
		outSps.spsSubLayerOrderingInfoPresent = reader.readBit() != 0;
		const oa::U32 orderingStart = outSps.spsSubLayerOrderingInfoPresent ? 0u : outSps.maxSubLayersMinus1;
		for (oa::U32 i = orderingStart; i <= outSps.maxSubLayersMinus1 && i < outSps.maxDecPicBufferingMinus1.size(); ++i) {
			outSps.maxDecPicBufferingMinus1[i] = reader.readUE();
			outSps.maxNumReorderPics[i] = reader.readUE();
			outSps.maxLatencyIncreasePlus1[i] = reader.readUE();
		}
		if (!outSps.spsSubLayerOrderingInfoPresent) {
			for (oa::U32 i = 0; i < outSps.maxSubLayersMinus1 && i < outSps.maxDecPicBufferingMinus1.size(); ++i) {
				outSps.maxDecPicBufferingMinus1[i] = outSps.maxDecPicBufferingMinus1[outSps.maxSubLayersMinus1];
				outSps.maxNumReorderPics[i] = outSps.maxNumReorderPics[outSps.maxSubLayersMinus1];
				outSps.maxLatencyIncreasePlus1[i] = outSps.maxLatencyIncreasePlus1[outSps.maxSubLayersMinus1];
			}
		}
		outSps.log2MinLumaCodingBlockSizeMinus3 = reader.readUE();
		outSps.log2DiffMaxMinLumaCodingBlockSize = reader.readUE();
		outSps.log2MinLumaTransformBlockSizeMinus2 = reader.readUE();
		outSps.log2DiffMaxMinLumaTransformBlockSize = reader.readUE();
		outSps.maxTransformHierarchyDepthInter = reader.readUE();
		outSps.maxTransformHierarchyDepthIntra = reader.readUE();
		outSps.scalingListEnabled = reader.readBit() != 0;
		if (outSps.scalingListEnabled) {
			outSps.spsScalingListDataPresent = reader.readBit() != 0;
			if (outSps.spsScalingListDataPresent) {
				skipH265ScalingListData(reader);
			}
		}
		outSps.ampEnabled = reader.readBit() != 0;
		outSps.sampleAdaptiveOffsetEnabled = reader.readBit() != 0;
		outSps.pcmEnabled = reader.readBit() != 0;
		if (outSps.pcmEnabled) {
			reader.skipBits(4); // pcm_sample_bit_depth_luma_minus1
			reader.skipBits(4); // pcm_sample_bit_depth_chroma_minus1
			(void)reader.readUE(); // log2_min_pcm_luma_coding_block_size_minus3
			(void)reader.readUE(); // log2_diff_max_min_pcm_luma_coding_block_size
			reader.skipBits(1); // pcm_loop_filter_disabled_flag
		}
		outSps.numShortTermRefPicSets = reader.readUE();
		oa::Array<oa::U32, 64> negativeCounts = {};
		oa::Array<oa::U32, 64> positiveCounts = {};
		for (oa::U32 i = 0; i < outSps.numShortTermRefPicSets && i < negativeCounts.size(); ++i) {
			skipH265ShortTermRefPicSet(reader, i, outSps.numShortTermRefPicSets, negativeCounts, positiveCounts);
		}
		outSps.longTermRefPicsPresent = reader.readBit() != 0;
		if (outSps.longTermRefPicsPresent) {
			const oa::U32 longTermCount = reader.readUE();
			for (oa::U32 i = 0; i < longTermCount; ++i) {
				reader.skipBits(outSps.log2MaxPicOrderCntLsbMinus4 + 4);
				reader.skipBits(1);
			}
		}
		outSps.temporalMvpEnabled = reader.readBit() != 0;
		outSps.strongIntraSmoothingEnabled = reader.readBit() != 0;
		return outSps.width > 0 && outSps.height > 0;
	}

	static bool parseH265Pps(
		const oa::U8* inData,
		oa::Usize inSize,
		H265PpsData& outPps)
	{
		if (inSize < 2) {
			return false;
		}
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		reader.skipBits(16); // HEVC NAL header
		outPps.ppsId = reader.readUE();
		outPps.spsId = reader.readUE();
		outPps.dependentSliceSegmentsEnabled = reader.readBit() != 0;
		outPps.outputFlagPresent = reader.readBit() != 0;
		outPps.numExtraSliceHeaderBits = reader.readBits(3);
		outPps.signDataHidingEnabled = reader.readBit() != 0;
		outPps.cabacInitPresent = reader.readBit() != 0;
		outPps.numRefIdxL0DefaultActiveMinus1 = reader.readUE();
		outPps.numRefIdxL1DefaultActiveMinus1 = reader.readUE();
		outPps.initQpMinus26 = reader.readSE();
		outPps.constrainedIntraPred = reader.readBit() != 0;
		outPps.transformSkipEnabled = reader.readBit() != 0;
		outPps.cuQpDeltaEnabled = reader.readBit() != 0;
		if (outPps.cuQpDeltaEnabled) {
			outPps.diffCuQpDeltaDepth = reader.readUE();
		}
		outPps.cbQpOffset = reader.readSE();
		outPps.crQpOffset = reader.readSE();
		outPps.ppsSliceChromaQpOffsetsPresent = reader.readBit() != 0;
		outPps.weightedPred = reader.readBit() != 0;
		outPps.weightedBipred = reader.readBit() != 0;
		outPps.transquantBypassEnabled = reader.readBit() != 0;
		outPps.tilesEnabled = reader.readBit() != 0;
		outPps.entropyCodingSyncEnabled = reader.readBit() != 0;
		if (outPps.tilesEnabled) {
			outPps.numTileColumnsMinus1 = reader.readUE();
			outPps.numTileRowsMinus1 = reader.readUE();
			outPps.uniformSpacing = reader.readBit() != 0;
			if (!outPps.uniformSpacing) {
				for (oa::U32 i = 0; i < outPps.numTileColumnsMinus1; ++i) {
					(void)reader.readUE();
				}
				for (oa::U32 i = 0; i < outPps.numTileRowsMinus1; ++i) {
					(void)reader.readUE();
				}
			}
			outPps.loopFilterAcrossTilesEnabled = reader.readBit() != 0;
		}
		outPps.ppsLoopFilterAcrossSlicesEnabled = reader.readBit() != 0;
		outPps.deblockingFilterControlPresent = reader.readBit() != 0;
		if (outPps.deblockingFilterControlPresent) {
			outPps.deblockingFilterOverrideEnabled = reader.readBit() != 0;
			outPps.ppsDeblockingFilterDisabled = reader.readBit() != 0;
			if (!outPps.ppsDeblockingFilterDisabled) {
				outPps.betaOffsetDiv2 = reader.readSE();
				outPps.tcOffsetDiv2 = reader.readSE();
			}
		}
		outPps.ppsScalingListDataPresent = reader.readBit() != 0;
		if (outPps.ppsScalingListDataPresent) {
			skipH265ScalingListData(reader);
		}
		outPps.listsModificationPresent = reader.readBit() != 0;
		outPps.log2ParallelMergeLevelMinus2 = reader.readUE();
		outPps.sliceSegmentHeaderExtensionPresent = reader.readBit() != 0;
		outPps.ppsExtensionPresent = reader.readBit() != 0;
		return true;
	}

	static bool parseH265SliceHeader(
		const oa::U8* inData,
		oa::Usize inSize,
		H265NalType inNalType,
		const H265SpsData& inSps,
		const H265PpsData& inPps,
		H265SliceHeader& outHeader)
	{
		if (inSize < 2) {
			return false;
		}
		oa::U8 temporalId = 0;
		(void)parseH265NalHeader(inData, inSize, temporalId);
		outHeader.nalUnitType = static_cast<oa::U8>(inNalType);
		outHeader.temporalId = temporalId;
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		reader.skipBits(16); // HEVC NAL header
		outHeader.firstSliceSegmentInPic = reader.readBit() != 0;
		outHeader.isIrap = static_cast<oa::U8>(inNalType) >= 16 && static_cast<oa::U8>(inNalType) <= 23;
		outHeader.isIdr = inNalType == H265NalType::IdrWRadl || inNalType == H265NalType::IdrNLp;
		outHeader.isReference =
			outHeader.nalUnitType >= 16u ||
			(outHeader.nalUnitType & 1u) != 0u;
		if (outHeader.isIrap) {
			outHeader.noOutputOfPriorPics = reader.readBit() != 0;
		}

		outHeader.ppsId = reader.readUE();
		if (outHeader.ppsId != inPps.ppsId) {
			return false;
		}
		outHeader.spsId = inPps.spsId;
		outHeader.vpsId = inSps.vpsId;

		bool dependentSliceSegment = false;
		if (!outHeader.firstSliceSegmentInPic) {
			if (inPps.dependentSliceSegmentsEnabled) {
				dependentSliceSegment = reader.readBit() != 0;
			}
			oa::U32 ctbSize = 1u << (inSps.log2MinLumaCodingBlockSizeMinus3 + 3u + inSps.log2DiffMaxMinLumaCodingBlockSize);
			if (ctbSize == 0) {
				ctbSize = 64;
			}
			const oa::U32 ctbWidth = (inSps.codedWidth + ctbSize - 1u) / ctbSize;
			const oa::U32 ctbHeight = (inSps.codedHeight + ctbSize - 1u) / ctbSize;
			oa::U32 addressBits = 0;
			for (oa::U32 value = ctbWidth * ctbHeight; value > 1; value = (value + 1u) >> 1u) {
				++addressBits;
			}
			reader.skipBits(addressBits);
		}
		if (dependentSliceSegment) {
			return false;
		}

		reader.skipBits(inPps.numExtraSliceHeaderBits);
		const oa::U32 sliceType = reader.readUE();
		if (sliceType == 0) {
			outHeader.sliceType = STD_VIDEO_H265_SLICE_TYPE_B;
		} else if (sliceType == 1) {
			outHeader.sliceType = STD_VIDEO_H265_SLICE_TYPE_P;
		} else if (sliceType == 2) {
			outHeader.sliceType = STD_VIDEO_H265_SLICE_TYPE_I;
		} else {
			return false;
		}

		if (inPps.outputFlagPresent) {
			reader.skipBits(1); // pic_output_flag
		}
		if (inSps.separateColourPlane) {
			reader.skipBits(2); // colour_plane_id
		}
		if (!outHeader.isIdr) {
			outHeader.picOrderCntLsb =
				reader.readBits(inSps.log2MaxPicOrderCntLsbMinus4 + 4u);
			outHeader.picOrderCntVal =
				static_cast<oa::I32>(outHeader.picOrderCntLsb);
			outHeader.shortTermRefPicSetSpsFlag = reader.readBit() != 0;
			if (outHeader.shortTermRefPicSetSpsFlag) {
				const oa::U32 stRpsBits = ceilLog2(inSps.numShortTermRefPicSets);
				if (stRpsBits > 0) {
					reader.skipBits(stRpsBits); // short_term_ref_pic_set_idx
				}
			} else {
				const oa::Usize stRpsStart = getBitOffset(reader);
				if (inSps.numShortTermRefPicSets != 0) {
					// Inter-RPS prediction requires the selected SPS RPS contents,
					// which are not retained by the minimal parameter-set parser yet.
					return false;
				}
				const oa::U32 negativeCount = reader.readUE();
				const oa::U32 positiveCount = reader.readUE();
				if (negativeCount > 16 || positiveCount > 16) {
					return false;
				}
				oa::I32 deltaPoc = 0;
				for (oa::U32 i = 0; i < negativeCount; ++i) {
					deltaPoc -= static_cast<oa::I32>(reader.readUE() + 1u);
					if (reader.readBit() != 0) {
						outHeader.stCurrBeforeDeltaPocs.pushBack(deltaPoc);
					} else {
						outHeader.stFollDeltaPocs.pushBack(deltaPoc);
					}
				}
				deltaPoc = 0;
				for (oa::U32 i = 0; i < positiveCount; ++i) {
					deltaPoc += static_cast<oa::I32>(reader.readUE() + 1u);
					if (reader.readBit() != 0) {
						outHeader.stCurrAfterDeltaPocs.pushBack(deltaPoc);
					} else {
						outHeader.stFollDeltaPocs.pushBack(deltaPoc);
					}
				}
				const oa::Usize stRpsBits = getBitOffset(reader) - stRpsStart;
				if (stRpsBits > 0xffffu) {
					return false;
				}
				outHeader.numBitsForSTRefPicSetInSlice = static_cast<oa::U16>(stRpsBits);
			}
		}
		return true;
	}

	// parse SPS (sequence Parameter set)
	static bool parseSPS(
		const oa::U8* inData,
		oa::Usize inSize,
		H264SpsData& outSps)
	{
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		// Skip NAL header (already parsed)
		reader.skipBits(8);

		// profile_idc
		outSps.profileIdc = reader.readBits(8);

		// constraint_set_flags + reserved_zero_2bits
		outSps.constraintFlags = reader.readBits(8);

		// level_idc
		outSps.levelIdc = reader.readBits(8);

		// seq_parameter_set_id
		outSps.spsId = reader.readUE();

		if (outSps.profileIdc == 100 || outSps.profileIdc == 110 || outSps.profileIdc == 122 ||
			outSps.profileIdc == 244 || outSps.profileIdc == 44 || outSps.profileIdc == 83 ||
			outSps.profileIdc == 86 || outSps.profileIdc == 118 || outSps.profileIdc == 128 ||
			outSps.profileIdc == 138 || outSps.profileIdc == 139 || outSps.profileIdc == 134 ||
			outSps.profileIdc == 135)
		{
			outSps.chromaFormatIdc = reader.readUE();
			if (outSps.chromaFormatIdc == 3)
			{
				outSps.separateColourPlane = reader.readBit() != 0;
			}
			outSps.bitDepthLumaMinus8 = reader.readUE();
			outSps.bitDepthChromaMinus8 = reader.readUE();
			outSps.qpprimeYZeroTransformBypass = reader.readBit() != 0;
			if (reader.readBit())
			{
				// seq_scaling_matrix_present_flag. Skip scaling lists for now;
				// default matrices are valid when pScalingLists is null.
				oa::U32 scalingListCount = outSps.chromaFormatIdc != 3 ? 8 : 12;
				for (oa::U32 i = 0; i < scalingListCount; ++i)
				{
					if (reader.readBit())
					{
						skipScalingList(reader, i < 6 ? 16 : 64);
					}
				}
			}
		}

		// log2_max_frame_num_minus4
		outSps.log2MaxFrameNumMinus4 = reader.readUE();

		// pic_order_cnt_type
		outSps.picOrderCntType = reader.readUE();
		if (outSps.picOrderCntType == 0)
		{
			// log2_max_pic_order_cnt_lsb_minus4
			outSps.log2MaxPicOrderCntLsbMinus4 = reader.readUE();
		}
		else if (outSps.picOrderCntType == 1)
		{
			// delta_pic_order_always_zero_flag
			outSps.deltaPicOrderAlwaysZero = reader.readBit() != 0;
			// offset_for_non_ref_pic
			outSps.offsetForNonRefPic = reader.readSE();
			// offset_for_top_to_bottom_field
			outSps.offsetForTopToBottomField = reader.readSE();
			// num_ref_frames_in_pic_order_cnt_cycle
			outSps.numRefFramesInPicOrderCntCycle = reader.readUE();
			if (outSps.numRefFramesInPicOrderCntCycle > outSps.offsetForRefFrame.size())
			{
				return false;
			}
			for (oa::U32 i = 0; i < outSps.numRefFramesInPicOrderCntCycle; ++i)
			{
				outSps.offsetForRefFrame[i] = reader.readSE();
			}
		}

		// max_num_ref_frames
		outSps.maxNumRefFrames = reader.readUE();

		// gaps_in_frame_num_value_allowed_flag
		outSps.gapsInFrameNumValueAllowed = reader.readBit() != 0;

		// pic_width_in_mbs_minus1
		outSps.picWidthInMbs = reader.readUE() + 1;

		// pic_height_in_map_units_minus1
		outSps.picHeightInMbs = reader.readUE() + 1;

		// frame_mbs_only_flag
		outSps.frameMbsOnly = reader.readBit() != 0;
		if (!outSps.frameMbsOnly)
		{
			// mb_adaptive_frame_field_flag
			outSps.mbAdaptiveFrameField = reader.readBit() != 0;
		}

		// direct_8x8_inference_flag
		outSps.direct8x8Inference = reader.readBit() != 0;

		// frame_cropping_flag
		outSps.frameCropping = reader.readBit() != 0;
		if (outSps.frameCropping)
		{
			outSps.frameCropLeftOffset = reader.readUE();
			outSps.frameCropRightOffset = reader.readUE();
			outSps.frameCropTopOffset = reader.readUE();
			outSps.frameCropBottomOffset = reader.readUE();
		}

		return true;
	}

	// parse PPS (Picture Parameter set)
	static bool parsePPS(
		const oa::U8* inData,
		oa::Usize inSize,
		H264PpsData& outPps)
	{
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		// Skip NAL header
		reader.skipBits(8);

		// pic_parameter_set_id
		outPps.ppsId = reader.readUE();

		// seq_parameter_set_id
		outPps.spsId = reader.readUE();

		outPps.entropyCodingMode = reader.readBit() != 0;
		outPps.bottomFieldPicOrderInFramePresent = reader.readBit() != 0;
		oa::U32 numSliceGroupsMinus1 = reader.readUE();
		if (numSliceGroupsMinus1 != 0)
		{
			return false;
		}
		outPps.numRefIdxL0DefaultActiveMinus1 = reader.readUE();
		outPps.numRefIdxL1DefaultActiveMinus1 = reader.readUE();
		outPps.weightedPred = reader.readBit() != 0;
		outPps.weightedBipredIdc = reader.readBits(2);
		outPps.picInitQpMinus26 = reader.readSE();
		outPps.picInitQsMinus26 = reader.readSE();
		outPps.chromaQpIndexOffset = reader.readSE();
		outPps.deblockingFilterControlPresent = reader.readBit() != 0;
		outPps.constrainedIntraPred = reader.readBit() != 0;
		outPps.redundantPicCntPresent = reader.readBit() != 0;
		if (moreRbspData(reader, rbsp))
		{
			outPps.transform8x8Mode = reader.readBit() != 0;
			if (reader.readBit())
			{
				const oa::U32 scalingListCount = 6 + (outPps.transform8x8Mode ? 2 : 0);
				for (oa::U32 i = 0; i < scalingListCount; ++i)
				{
					if (reader.readBit())
					{
						skipScalingList(reader, i < 6 ? 16 : 64);
					}
				}
			}
			outPps.secondChromaQpIndexOffset = reader.readSE();
		}
		else
		{
			outPps.secondChromaQpIndexOffset = outPps.chromaQpIndexOffset;
		}

		return true;
	}

	static bool parseSliceHeaderPrefix(
		const oa::U8* inData,
		oa::Usize inSize,
		bool inIsIdr,
		oa::U32& outPpsId)
	{
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		// Skip NAL header
		reader.skipBits(8);

		reader.readUE(); // first_mb_in_slice
		reader.readUE(); // slice_type
		outPpsId = reader.readUE();
		(void)inIsIdr;
		return true;
	}

	// parse slice header (phase 2.4.3)
	static bool parseSliceHeader(
		const oa::U8* inData,
		oa::Usize inSize,
		bool inIsIdr,
		oa::U8 inNalRefIdc,
		const H264SpsData& inSps,
		const H264PpsData& inPps,
		H264SliceHeader& outHeader)
	{
		oa::Vector<oa::U8> rbsp = makeRbsp(inData, inSize);
		BitstreamReader reader(rbsp.data(), rbsp.size());

		reader.skipBits(8); // NAL header

		outHeader.firstMbInSlice = reader.readUE();

		oa::U32 sliceType = reader.readUE();
		if (sliceType >= 5)
		{
			sliceType -= 5;
		}
		if (sliceType > static_cast<oa::U32>(H264SliceType::SI))
		{
			return false;
		}
		outHeader.sliceType = static_cast<H264SliceType>(sliceType);
		outHeader.ppsId = reader.readUE();
		if (outHeader.ppsId != inPps.ppsId)
		{
			return false;
		}

		const oa::U32 frameNumBits = inSps.log2MaxFrameNumMinus4 + 4;
		if (frameNumBits == 0 || frameNumBits > 16)
		{
			return false;
		}
		outHeader.frameNum = reader.readBits(frameNumBits);
		if (!inSps.frameMbsOnly)
		{
			outHeader.fieldPicFlag = reader.readBit() != 0;
			if (outHeader.fieldPicFlag)
			{
				outHeader.bottomFieldFlag = reader.readBit() != 0;
			}
		}

		// IDR-specific fields
		outHeader.isIdrPic = inIsIdr;
		if (inIsIdr)
		{
			outHeader.idrPicId = reader.readUE();
		}

		outHeader.picOrderCntLsb = 0;
		if (inSps.picOrderCntType == 0)
		{
			const oa::U32 pocBits = inSps.log2MaxPicOrderCntLsbMinus4 + 4;
			if (pocBits == 0 || pocBits > 16)
			{
				return false;
			}
			outHeader.picOrderCntLsb = static_cast<oa::I32>(reader.readBits(pocBits));
			if (inPps.bottomFieldPicOrderInFramePresent && !inSps.frameMbsOnly)
			{
				reader.readSE(); // delta_pic_order_cnt_bottom
			}
		}
		else if (inSps.picOrderCntType == 1)
		{
			// Full POC type 1 derivation requires previous-picture state. Keep the
			// parse valid but use frame_num as the monotonic DPB order fallback.
			outHeader.picOrderCntLsb = static_cast<oa::I32>(outHeader.frameNum);
		}

		outHeader.isReference = inNalRefIdc != 0;

		// Best-effort walk to dec_ref_pic_marking(). The intermediate fields
		// (redundant_pic_cnt / direct_spatial_mv_pred / num_ref_idx_active_override
		// / ref_pic_list_modification / pred_weight_table) cover a lot of stream
		// variants; we parse the common cases conservatively and bail out by
		// marking refPicMarkingValid=false when we can't follow safely.
		auto attemptRefPicMarking = [&]() -> bool {
			if (inPps.redundantPicCntPresent) {
				(void)reader.readUE(); // redundant_pic_cnt
			}
			const bool isB  = outHeader.sliceType == H264SliceType::B;
			const bool isP  = outHeader.sliceType == H264SliceType::P
			               || outHeader.sliceType == H264SliceType::SP;
			const bool isI  = outHeader.sliceType == H264SliceType::I
			               || outHeader.sliceType == H264SliceType::SI;

			if (isB) {
				reader.skipBits(1); // direct_spatial_mv_pred_flag
			}

			oa::U32 numRefIdxL0 = inPps.numRefIdxL0DefaultActiveMinus1;
			oa::U32 numRefIdxL1 = inPps.numRefIdxL1DefaultActiveMinus1;
			if (isP || isB) {
				const bool overrideFlag = reader.readBit() != 0;
				if (overrideFlag) {
					numRefIdxL0 = reader.readUE();
					if (isB) {
						numRefIdxL1 = reader.readUE();
					}
				}
			}
			if (numRefIdxL0 >= 32 || (isB && numRefIdxL1 >= 32)) {
				return false;
			}

			// ref_pic_list_modification(): skip lists for P/SP/B slices.
			auto skipRefPicListModification = [&](bool inActive) {
				if (!inActive) { return; }
				const bool present = reader.readBit() != 0;
				if (!present) { return; }
				for (oa::U32 guard = 0; guard < 256; ++guard) {
					const oa::U32 op = reader.readUE();
					if (op == 3) { break; }
					(void)reader.readUE();  // abs_diff_pic_num_minus1 / long_term_pic_num
				}
			};
			if (isP || isB) { skipRefPicListModification(true); }
			if (isB)        { skipRefPicListModification(true); }

			// pred_weight_table() — only when explicit weighting is in play.
			const bool weightedPred = isP && inPps.weightedPred;
			const bool weightedBi   = isB && (inPps.weightedBipredIdc == 1);
			if (weightedPred || weightedBi) {
				if (!skipH264PredWeightTable(
					reader,
					inSps,
					numRefIdxL0,
					numRefIdxL1,
					isB)) {
					return false;
				}
			}
			(void)isI;

			// dec_ref_pic_marking()
			if (inNalRefIdc == 0) {
				return true;  // No marking section for non-reference slices.
			}
			if (inIsIdr) {
				outHeader.noOutputOfPriorPics = reader.readBit() != 0;
				outHeader.longTermReference   = reader.readBit() != 0;
				return true;
			}
			outHeader.adaptiveRefPicMarking = reader.readBit() != 0;
			if (!outHeader.adaptiveRefPicMarking) {
				return true;
			}
			for (oa::U32 guard = 0; guard < 64; ++guard) {
				H264MmcoCommand cmd{};
				cmd.op = reader.readUE();
				if (cmd.op == 0) { break; }
				switch (cmd.op) {
					case 1: cmd.differenceOfPicNumsMinus1 = reader.readUE(); break;
					case 2: cmd.longTermPicNum             = reader.readUE(); break;
					case 3:
						cmd.differenceOfPicNumsMinus1 = reader.readUE();
						cmd.longTermFrameIdx          = reader.readUE();
						break;
					case 4: cmd.maxLongTermFrameIdxPlus1   = reader.readUE(); break;
					case 5: break;  // no operands
					case 6: cmd.longTermFrameIdx           = reader.readUE(); break;
					default:
						return false;  // unknown op → bail
				}
				outHeader.mmcoCommands.pushBack(cmd);
			}
			return true;
		};
		outHeader.refPicMarkingValid = attemptRefPicMarking();

		return true;
	}
};

} // namespace oa
