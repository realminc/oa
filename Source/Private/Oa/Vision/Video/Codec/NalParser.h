// OA Vision — H.264/H.265 NAL Unit Parser
// Parse SPS, PPS, and slice headers

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoCodecParameterSets.h>
#include "BitstreamReader.h"

// H.264 NAL unit types
enum class OaH264NalType : OaU8
{
	Unspecified = 0,
	NonIDR = 1,      // Non-IDR slice
	IDR = 5,         // IDR slice
	SEI = 6,         // Supplemental Enhancement Information
	SPS = 7,         // Sequence Parameter Set
	PPS = 8,         // Picture Parameter Set
	AUD = 9,         // Access Unit Delimiter
};

// H.264 slice types
enum class OaH264SliceType : OaU8
{
	P = 0,   // P slice (predicted)
	B = 1,   // B slice (bi-predicted)
	I = 2,   // I slice (intra)
	SP = 3,  // SP slice (switching P)
	SI = 4,  // SI slice (switching I)
};

enum class OaH265NalType : OaU8
{
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
struct OaH264MmcoCommand
{
	OaU32 Op = 0;                       // memory_management_control_operation
	OaU32 DifferenceOfPicNumsMinus1 = 0; // op 1, 3
	OaU32 LongTermPicNum = 0;            // op 2
	OaU32 LongTermFrameIdx = 0;          // op 3, 6
	OaU32 MaxLongTermFrameIdxPlus1 = 0;  // op 4
};

// Parsed slice header (minimal fields for Phase 2.4.3)
struct OaSliceHeader
{
	OaU32 FirstMbInSlice;
	OaH264SliceType SliceType;
	OaU32 PpsId;
	OaU32 FrameNum;
	OaU32 IdrPicId = 0;
	OaI32 PicOrderCntLsb;
	bool FieldPicFlag = false;
	bool BottomFieldFlag = false;
	bool IsIdrPic;
	bool IsReference;

	// dec_ref_pic_marking() — populated when ref_pic_marking parsing succeeded.
	// For IDR slices: NoOutputOfPriorPics + LongTermReference flags.
	// For non-IDR: AdaptiveRefPicMarking + MmcoCommands list.
	bool NoOutputOfPriorPics       = false;
	bool LongTermReference         = false;
	bool AdaptiveRefPicMarking     = false;
	OaVec<OaH264MmcoCommand> MmcoCommands;
	bool RefPicMarkingValid        = false;
};

struct OaH265SliceHeader
{
	OaU32 PpsId = 0;
	OaU32 SpsId = 0;
	OaU32 VpsId = 0;
	StdVideoH265SliceType SliceType = STD_VIDEO_H265_SLICE_TYPE_I;
	OaU32 PicOrderCntLsb = 0;
	OaI32 PicOrderCntVal = 0;
	OaU16 NumBitsForSTRefPicSetInSlice = 0;
	OaVec<OaI32> StCurrBeforeDeltaPocs;
	OaVec<OaI32> StCurrAfterDeltaPocs;
	OaVec<OaI32> StFollDeltaPocs;
	OaU8 NalUnitType = 0;
	OaU8 TemporalId = 0;
	bool FirstSliceSegmentInPic = true;
	bool IsIrap = false;
	bool IsIdr = false;
	bool NoOutputOfPriorPics = false;
	bool IsReference = true;
	bool ShortTermRefPicSetSpsFlag = false;
};

// NAL unit parser
class OaNalParser
{
public:
	static OaVec<OaU8> MakeRbsp(const OaU8* InData, OaUsize InSize)
	{
		OaVec<OaU8> rbsp;
		rbsp.Reserve(InSize);
		OaU32 zeroRun = 0;
		for (OaUsize i = 0; i < InSize; ++i)
		{
			const OaU8 byte = InData[i];
			if (zeroRun >= 2 && byte == 0x03)
			{
				zeroRun = 0;
				continue;
			}
			rbsp.PushBack(byte);
			zeroRun = byte == 0 ? zeroRun + 1 : 0;
		}
		return rbsp;
	}

	static bool MoreRbspData(const OaBitstreamReader& InReader, const OaVec<OaU8>& InRbsp)
	{
		if (InRbsp.Empty())
		{
			return false;
		}
		OaI64 stopBitOffset = -1;
		for (OaI64 byteIndex = static_cast<OaI64>(InRbsp.Size()) - 1; byteIndex >= 0 && stopBitOffset < 0; --byteIndex)
		{
			const OaU8 value = InRbsp[static_cast<OaUsize>(byteIndex)];
			for (OaI32 bit = 0; bit < 8; ++bit)
			{
				if ((value & (1u << bit)) != 0)
				{
					stopBitOffset = byteIndex * 8 + (7 - bit);
					break;
				}
			}
		}
		if (stopBitOffset < 0)
		{
			return false;
		}
		const OaUsize currentBitOffset = InReader.GetBytePos() * 8 + InReader.GetBitPos();
		return currentBitOffset < static_cast<OaUsize>(stopBitOffset);
	}

	static OaUsize GetBitOffset(const OaBitstreamReader& InReader)
	{
		return InReader.GetBytePos() * 8 + InReader.GetBitPos();
	}

	static OaU32 CeilLog2(OaU32 InValue)
	{
		if (InValue <= 1) {
			return 0;
		}
		OaU32 bits = 0;
		--InValue;
		while (InValue > 0) {
			++bits;
			InValue >>= 1;
		}
		return bits;
	}

	static void SkipScalingList(OaBitstreamReader& InReader, OaU32 InSize)
	{
		OaI32 lastScale = 8;
		OaI32 nextScale = 8;
		for (OaU32 j = 0; j < InSize; ++j)
		{
			if (nextScale != 0)
			{
				const OaI32 deltaScale = InReader.ReadSE();
				nextScale = (lastScale + deltaScale + 256) % 256;
			}
			lastScale = nextScale == 0 ? lastScale : nextScale;
		}
	}

	static bool SkipH264PredWeightTable(
		OaBitstreamReader& InReader,
		const OaH264SpsData& InSps,
		OaU32 InNumRefIdxL0ActiveMinus1,
		OaU32 InNumRefIdxL1ActiveMinus1,
		bool InHasList1)
	{
		// H.264 limits each reference list to 32 frame references. Reject
		// impossible values instead of letting a damaged header drive an
		// unbounded parser walk.
		if (InNumRefIdxL0ActiveMinus1 >= 32
			|| (InHasList1 && InNumRefIdxL1ActiveMinus1 >= 32)) {
			return false;
		}

		(void)InReader.ReadUE(); // luma_log2_weight_denom
		const OaU32 chromaArrayType = InSps.SeparateColourPlane
			? 0U
			: InSps.ChromaFormatIdc;
		if (chromaArrayType != 0) {
			(void)InReader.ReadUE(); // chroma_log2_weight_denom
		}

		auto skipList = [&](OaU32 InNumRefIdxActiveMinus1) {
			for (OaU32 i = 0; i <= InNumRefIdxActiveMinus1; ++i) {
				if (InReader.ReadBit() != 0) {
					(void)InReader.ReadSE(); // luma_weight
					(void)InReader.ReadSE(); // luma_offset
				}
				if (chromaArrayType != 0 && InReader.ReadBit() != 0) {
					for (OaU32 component = 0; component < 2; ++component) {
						(void)InReader.ReadSE(); // chroma_weight
						(void)InReader.ReadSE(); // chroma_offset
					}
				}
			}
		};

		skipList(InNumRefIdxL0ActiveMinus1);
		if (InHasList1) {
			skipList(InNumRefIdxL1ActiveMinus1);
		}
		return true;
	}

	static void ReadH265ProfileTierLevel(
		OaBitstreamReader& InReader,
		OaU32 InMaxSubLayersMinus1,
		OaH265VpsData* OutVps,
		OaH265SpsData* OutSps)
	{
		InReader.SkipBits(2);  // general_profile_space
		const bool tierFlag = InReader.ReadBit() != 0;
		const OaU32 profileIdc = InReader.ReadBits(5);
		InReader.SkipBits(32); // general_profile_compatibility_flags
		const bool progressive = InReader.ReadBit() != 0;
		const bool interlaced = InReader.ReadBit() != 0;
		const bool nonPacked = InReader.ReadBit() != 0;
		const bool frameOnly = InReader.ReadBit() != 0;
		InReader.SkipBits(44); // reserved and constraint flags
		const OaU32 levelIdc = InReader.ReadBits(8);
		if (OutVps) {
			OutVps->GeneralTierFlag = tierFlag;
			OutVps->GeneralProfileIdc = profileIdc;
			OutVps->GeneralLevelIdc = levelIdc;
			OutVps->GeneralProgressiveSourceFlag = progressive;
			OutVps->GeneralInterlacedSourceFlag = interlaced;
			OutVps->GeneralNonPackedConstraintFlag = nonPacked;
			OutVps->GeneralFrameOnlyConstraintFlag = frameOnly;
		}
		(void)OutSps;

		OaArray<bool, 8> subLayerProfilePresent = {};
		OaArray<bool, 8> subLayerLevelPresent = {};
		for (OaU32 i = 0; i < InMaxSubLayersMinus1 && i < subLayerProfilePresent.Size(); ++i) {
			subLayerProfilePresent[i] = InReader.ReadBit() != 0;
			subLayerLevelPresent[i] = InReader.ReadBit() != 0;
		}
		if (InMaxSubLayersMinus1 > 0) {
			for (OaU32 i = InMaxSubLayersMinus1; i < 8; ++i) {
				InReader.SkipBits(2); // reserved_zero_2bits
			}
		}
		for (OaU32 i = 0; i < InMaxSubLayersMinus1 && i < subLayerProfilePresent.Size(); ++i) {
			if (subLayerProfilePresent[i]) {
				InReader.SkipBits(2);
				InReader.SkipBits(1);
				InReader.SkipBits(5);
				InReader.SkipBits(32);
				InReader.SkipBits(4);
				InReader.SkipBits(44);
			}
			if (subLayerLevelPresent[i]) {
				InReader.SkipBits(8);
			}
		}
	}

	static void SkipH265ProfileTierLevel(OaBitstreamReader& InReader, OaU32 InMaxSubLayersMinus1)
	{
		ReadH265ProfileTierLevel(InReader, InMaxSubLayersMinus1, nullptr, nullptr);
	}

	static void SkipH265ScalingListData(OaBitstreamReader& InReader)
	{
		for (OaU32 sizeId = 0; sizeId < 4; ++sizeId) {
			const OaU32 matrixCount = sizeId == 3 ? 2u : 6u;
			for (OaU32 matrixId = 0; matrixId < matrixCount; ++matrixId) {
				if (InReader.ReadBit() == 0) {
					(void)InReader.ReadUE(); // scaling_list_pred_matrix_id_delta
					continue;
				}
				const OaU32 coefCount = sizeId == 0 ? 16u : 64u;
				if (sizeId > 1) {
					(void)InReader.ReadSE(); // scaling_list_dc_coef_minus8
				}
				for (OaU32 i = 0; i < coefCount; ++i) {
					(void)InReader.ReadSE(); // scaling_list_delta_coef
				}
			}
		}
	}

	static void SkipH265ShortTermRefPicSet(
		OaBitstreamReader& InReader,
		OaU32 InSetIndex,
		OaU32 InSetCount,
		OaArray<OaU32, 64>& InOutNegativeCounts,
		OaArray<OaU32, 64>& InOutPositiveCounts)
	{
		if (InSetIndex != 0 && InReader.ReadBit() != 0) {
			if (InSetIndex == InSetCount) {
				(void)InReader.ReadUE(); // delta_idx_minus1
			}
			InReader.SkipBits(1); // delta_rps_sign
			(void)InReader.ReadUE(); // abs_delta_rps_minus1
			const OaU32 refIndex = InSetIndex > 0 ? InSetIndex - 1 : 0;
			const OaU32 deltaCount = InOutNegativeCounts[refIndex] + InOutPositiveCounts[refIndex];
			for (OaU32 j = 0; j <= deltaCount; ++j) {
				const bool usedByCurrPic = InReader.ReadBit() != 0;
				if (!usedByCurrPic) {
					InReader.SkipBits(1); // use_delta_flag
				}
			}
			InOutNegativeCounts[InSetIndex] = 0;
			InOutPositiveCounts[InSetIndex] = 0;
			return;
		}
		const OaU32 negativeCount = InReader.ReadUE();
		const OaU32 positiveCount = InReader.ReadUE();
		InOutNegativeCounts[InSetIndex] = negativeCount;
		InOutPositiveCounts[InSetIndex] = positiveCount;
		for (OaU32 i = 0; i < negativeCount; ++i) {
			(void)InReader.ReadUE(); // delta_poc_s0_minus1
			InReader.SkipBits(1); // used_by_curr_pic_s0_flag
		}
		for (OaU32 i = 0; i < positiveCount; ++i) {
			(void)InReader.ReadUE(); // delta_poc_s1_minus1
			InReader.SkipBits(1); // used_by_curr_pic_s1_flag
		}
	}

public:
	// Parse NAL unit header (first byte after start code)
	static OaH264NalType ParseNalHeader(OaU8 InNalByte, OaU8& OutRefIdc)
	{
		// NAL unit header: forbidden_zero_bit(1) + nal_ref_idc(2) + nal_unit_type(5)
		OutRefIdc = (InNalByte >> 5) & 0x3;
		return static_cast<OaH264NalType>(InNalByte & 0x1F);
	}

	static OaH265NalType ParseH265NalHeader(const OaU8* InData, OaUsize InSize, OaU8& OutTemporalId)
	{
		OutTemporalId = 0;
		if (InSize < 2) {
			return OaH265NalType::TrailN;
		}
		const OaU8 temporalIdPlus1 = static_cast<OaU8>(InData[1] & 0x07u);
		if (temporalIdPlus1 == 0) {
			return OaH265NalType::TrailN;
		}
		OutTemporalId = static_cast<OaU8>(temporalIdPlus1 - 1u);
		return static_cast<OaH265NalType>((InData[0] >> 1) & 0x3Fu);
	}

	static bool ParseH265Vps(
		const OaU8* InData,
		OaUsize InSize,
		OaH265VpsData& OutVps)
	{
		if (InSize < 2) {
			return false;
		}
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		reader.SkipBits(16); // HEVC NAL header
		OutVps.VpsId = reader.ReadBits(4);
		reader.SkipBits(2); // base_layer_internal_flag + base_layer_available_flag
		reader.SkipBits(6); // vps_max_layers_minus1
		OutVps.MaxSubLayersMinus1 = reader.ReadBits(3);
		OutVps.TemporalIdNesting = reader.ReadBit() != 0;
		reader.SkipBits(16); // vps_reserved_0xffff_16bits
		ReadH265ProfileTierLevel(reader, OutVps.MaxSubLayersMinus1, &OutVps, nullptr);
		return true;
	}

	static bool ParseH265Sps(
		const OaU8* InData,
		OaUsize InSize,
		OaH265SpsData& OutSps)
	{
		if (InSize < 2) {
			return false;
		}
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		reader.SkipBits(16); // HEVC NAL header
		OutSps.VpsId = reader.ReadBits(4);
		OutSps.MaxSubLayersMinus1 = reader.ReadBits(3);
		OutSps.TemporalIdNesting = reader.ReadBit() != 0;
		SkipH265ProfileTierLevel(reader, OutSps.MaxSubLayersMinus1);
		OutSps.SpsId = reader.ReadUE();
		OutSps.ChromaFormatIdc = reader.ReadUE();
		if (OutSps.ChromaFormatIdc == 3) {
			OutSps.SeparateColourPlane = reader.ReadBit() != 0;
		}
		OutSps.Width = reader.ReadUE();
		OutSps.Height = reader.ReadUE();
		OutSps.CodedWidth = OutSps.Width;
		OutSps.CodedHeight = OutSps.Height;
		if (reader.ReadBit() != 0) {
			OutSps.ConformanceWindowLeft = reader.ReadUE();
			OutSps.ConformanceWindowRight = reader.ReadUE();
			OutSps.ConformanceWindowTop = reader.ReadUE();
			OutSps.ConformanceWindowBottom = reader.ReadUE();
			const OaU32 cropUnitX = OutSps.ChromaFormatIdc == 1 ? 2u : 1u;
			const OaU32 cropUnitY = OutSps.ChromaFormatIdc == 1 ? 2u : 1u;
			const OaU32 cropWidth = (OutSps.ConformanceWindowLeft + OutSps.ConformanceWindowRight) * cropUnitX;
			const OaU32 cropHeight = (OutSps.ConformanceWindowTop + OutSps.ConformanceWindowBottom) * cropUnitY;
			if (cropWidth < OutSps.Width) {
				OutSps.Width -= cropWidth;
			}
			if (cropHeight < OutSps.Height) {
				OutSps.Height -= cropHeight;
			}
		}
		OutSps.BitDepthLumaMinus8 = reader.ReadUE();
		OutSps.BitDepthChromaMinus8 = reader.ReadUE();
		OutSps.Log2MaxPicOrderCntLsbMinus4 = reader.ReadUE();
		OutSps.SpsSubLayerOrderingInfoPresent = reader.ReadBit() != 0;
		const OaU32 orderingStart = OutSps.SpsSubLayerOrderingInfoPresent ? 0u : OutSps.MaxSubLayersMinus1;
		for (OaU32 i = orderingStart; i <= OutSps.MaxSubLayersMinus1 && i < OutSps.MaxDecPicBufferingMinus1.Size(); ++i) {
			OutSps.MaxDecPicBufferingMinus1[i] = reader.ReadUE();
			OutSps.MaxNumReorderPics[i] = reader.ReadUE();
			OutSps.MaxLatencyIncreasePlus1[i] = reader.ReadUE();
		}
		if (!OutSps.SpsSubLayerOrderingInfoPresent) {
			for (OaU32 i = 0; i < OutSps.MaxSubLayersMinus1 && i < OutSps.MaxDecPicBufferingMinus1.Size(); ++i) {
				OutSps.MaxDecPicBufferingMinus1[i] = OutSps.MaxDecPicBufferingMinus1[OutSps.MaxSubLayersMinus1];
				OutSps.MaxNumReorderPics[i] = OutSps.MaxNumReorderPics[OutSps.MaxSubLayersMinus1];
				OutSps.MaxLatencyIncreasePlus1[i] = OutSps.MaxLatencyIncreasePlus1[OutSps.MaxSubLayersMinus1];
			}
		}
		OutSps.Log2MinLumaCodingBlockSizeMinus3 = reader.ReadUE();
		OutSps.Log2DiffMaxMinLumaCodingBlockSize = reader.ReadUE();
		OutSps.Log2MinLumaTransformBlockSizeMinus2 = reader.ReadUE();
		OutSps.Log2DiffMaxMinLumaTransformBlockSize = reader.ReadUE();
		OutSps.MaxTransformHierarchyDepthInter = reader.ReadUE();
		OutSps.MaxTransformHierarchyDepthIntra = reader.ReadUE();
		OutSps.ScalingListEnabled = reader.ReadBit() != 0;
		if (OutSps.ScalingListEnabled) {
			OutSps.SpsScalingListDataPresent = reader.ReadBit() != 0;
			if (OutSps.SpsScalingListDataPresent) {
				SkipH265ScalingListData(reader);
			}
		}
		OutSps.AmpEnabled = reader.ReadBit() != 0;
		OutSps.SampleAdaptiveOffsetEnabled = reader.ReadBit() != 0;
		OutSps.PcmEnabled = reader.ReadBit() != 0;
		if (OutSps.PcmEnabled) {
			reader.SkipBits(4); // pcm_sample_bit_depth_luma_minus1
			reader.SkipBits(4); // pcm_sample_bit_depth_chroma_minus1
			(void)reader.ReadUE(); // log2_min_pcm_luma_coding_block_size_minus3
			(void)reader.ReadUE(); // log2_diff_max_min_pcm_luma_coding_block_size
			reader.SkipBits(1); // pcm_loop_filter_disabled_flag
		}
		OutSps.NumShortTermRefPicSets = reader.ReadUE();
		OaArray<OaU32, 64> negativeCounts = {};
		OaArray<OaU32, 64> positiveCounts = {};
		for (OaU32 i = 0; i < OutSps.NumShortTermRefPicSets && i < negativeCounts.Size(); ++i) {
			SkipH265ShortTermRefPicSet(reader, i, OutSps.NumShortTermRefPicSets, negativeCounts, positiveCounts);
		}
		OutSps.LongTermRefPicsPresent = reader.ReadBit() != 0;
		if (OutSps.LongTermRefPicsPresent) {
			const OaU32 longTermCount = reader.ReadUE();
			for (OaU32 i = 0; i < longTermCount; ++i) {
				reader.SkipBits(OutSps.Log2MaxPicOrderCntLsbMinus4 + 4);
				reader.SkipBits(1);
			}
		}
		OutSps.TemporalMvpEnabled = reader.ReadBit() != 0;
		OutSps.StrongIntraSmoothingEnabled = reader.ReadBit() != 0;
		return OutSps.Width > 0 && OutSps.Height > 0;
	}

	static bool ParseH265Pps(
		const OaU8* InData,
		OaUsize InSize,
		OaH265PpsData& OutPps)
	{
		if (InSize < 2) {
			return false;
		}
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		reader.SkipBits(16); // HEVC NAL header
		OutPps.PpsId = reader.ReadUE();
		OutPps.SpsId = reader.ReadUE();
		OutPps.DependentSliceSegmentsEnabled = reader.ReadBit() != 0;
		OutPps.OutputFlagPresent = reader.ReadBit() != 0;
		OutPps.NumExtraSliceHeaderBits = reader.ReadBits(3);
		OutPps.SignDataHidingEnabled = reader.ReadBit() != 0;
		OutPps.CabacInitPresent = reader.ReadBit() != 0;
		OutPps.NumRefIdxL0DefaultActiveMinus1 = reader.ReadUE();
		OutPps.NumRefIdxL1DefaultActiveMinus1 = reader.ReadUE();
		OutPps.InitQpMinus26 = reader.ReadSE();
		OutPps.ConstrainedIntraPred = reader.ReadBit() != 0;
		OutPps.TransformSkipEnabled = reader.ReadBit() != 0;
		OutPps.CuQpDeltaEnabled = reader.ReadBit() != 0;
		if (OutPps.CuQpDeltaEnabled) {
			OutPps.DiffCuQpDeltaDepth = reader.ReadUE();
		}
		OutPps.CbQpOffset = reader.ReadSE();
		OutPps.CrQpOffset = reader.ReadSE();
		OutPps.PpsSliceChromaQpOffsetsPresent = reader.ReadBit() != 0;
		OutPps.WeightedPred = reader.ReadBit() != 0;
		OutPps.WeightedBipred = reader.ReadBit() != 0;
		OutPps.TransquantBypassEnabled = reader.ReadBit() != 0;
		OutPps.TilesEnabled = reader.ReadBit() != 0;
		OutPps.EntropyCodingSyncEnabled = reader.ReadBit() != 0;
		if (OutPps.TilesEnabled) {
			OutPps.NumTileColumnsMinus1 = reader.ReadUE();
			OutPps.NumTileRowsMinus1 = reader.ReadUE();
			OutPps.UniformSpacing = reader.ReadBit() != 0;
			if (!OutPps.UniformSpacing) {
				for (OaU32 i = 0; i < OutPps.NumTileColumnsMinus1; ++i) {
					(void)reader.ReadUE();
				}
				for (OaU32 i = 0; i < OutPps.NumTileRowsMinus1; ++i) {
					(void)reader.ReadUE();
				}
			}
			OutPps.LoopFilterAcrossTilesEnabled = reader.ReadBit() != 0;
		}
		OutPps.PpsLoopFilterAcrossSlicesEnabled = reader.ReadBit() != 0;
		OutPps.DeblockingFilterControlPresent = reader.ReadBit() != 0;
		if (OutPps.DeblockingFilterControlPresent) {
			OutPps.DeblockingFilterOverrideEnabled = reader.ReadBit() != 0;
			OutPps.PpsDeblockingFilterDisabled = reader.ReadBit() != 0;
			if (!OutPps.PpsDeblockingFilterDisabled) {
				OutPps.BetaOffsetDiv2 = reader.ReadSE();
				OutPps.TcOffsetDiv2 = reader.ReadSE();
			}
		}
		OutPps.PpsScalingListDataPresent = reader.ReadBit() != 0;
		if (OutPps.PpsScalingListDataPresent) {
			SkipH265ScalingListData(reader);
		}
		OutPps.ListsModificationPresent = reader.ReadBit() != 0;
		OutPps.Log2ParallelMergeLevelMinus2 = reader.ReadUE();
		OutPps.SliceSegmentHeaderExtensionPresent = reader.ReadBit() != 0;
		OutPps.PpsExtensionPresent = reader.ReadBit() != 0;
		return true;
	}

	static bool ParseH265SliceHeader(
		const OaU8* InData,
		OaUsize InSize,
		OaH265NalType InNalType,
		const OaH265SpsData& InSps,
		const OaH265PpsData& InPps,
		OaH265SliceHeader& OutHeader)
	{
		if (InSize < 2) {
			return false;
		}
		OaU8 temporalId = 0;
		(void)ParseH265NalHeader(InData, InSize, temporalId);
		OutHeader.NalUnitType = static_cast<OaU8>(InNalType);
		OutHeader.TemporalId = temporalId;
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		reader.SkipBits(16); // HEVC NAL header
		OutHeader.FirstSliceSegmentInPic = reader.ReadBit() != 0;
		OutHeader.IsIrap = static_cast<OaU8>(InNalType) >= 16 && static_cast<OaU8>(InNalType) <= 23;
		OutHeader.IsIdr = InNalType == OaH265NalType::IdrWRadl || InNalType == OaH265NalType::IdrNLp;
		OutHeader.IsReference =
			OutHeader.NalUnitType >= 16u ||
			(OutHeader.NalUnitType & 1u) != 0u;
		if (OutHeader.IsIrap) {
			OutHeader.NoOutputOfPriorPics = reader.ReadBit() != 0;
		}

		OutHeader.PpsId = reader.ReadUE();
		if (OutHeader.PpsId != InPps.PpsId) {
			return false;
		}
		OutHeader.SpsId = InPps.SpsId;
		OutHeader.VpsId = InSps.VpsId;

		bool dependentSliceSegment = false;
		if (!OutHeader.FirstSliceSegmentInPic) {
			if (InPps.DependentSliceSegmentsEnabled) {
				dependentSliceSegment = reader.ReadBit() != 0;
			}
			OaU32 ctbSize = 1u << (InSps.Log2MinLumaCodingBlockSizeMinus3 + 3u + InSps.Log2DiffMaxMinLumaCodingBlockSize);
			if (ctbSize == 0) {
				ctbSize = 64;
			}
			const OaU32 ctbWidth = (InSps.CodedWidth + ctbSize - 1u) / ctbSize;
			const OaU32 ctbHeight = (InSps.CodedHeight + ctbSize - 1u) / ctbSize;
			OaU32 addressBits = 0;
			for (OaU32 value = ctbWidth * ctbHeight; value > 1; value = (value + 1u) >> 1u) {
				++addressBits;
			}
			reader.SkipBits(addressBits);
		}
		if (dependentSliceSegment) {
			return false;
		}

		reader.SkipBits(InPps.NumExtraSliceHeaderBits);
		const OaU32 sliceType = reader.ReadUE();
		if (sliceType == 0) {
			OutHeader.SliceType = STD_VIDEO_H265_SLICE_TYPE_B;
		} else if (sliceType == 1) {
			OutHeader.SliceType = STD_VIDEO_H265_SLICE_TYPE_P;
		} else if (sliceType == 2) {
			OutHeader.SliceType = STD_VIDEO_H265_SLICE_TYPE_I;
		} else {
			return false;
		}

		if (InPps.OutputFlagPresent) {
			reader.SkipBits(1); // pic_output_flag
		}
		if (InSps.SeparateColourPlane) {
			reader.SkipBits(2); // colour_plane_id
		}
		if (!OutHeader.IsIdr) {
			OutHeader.PicOrderCntLsb =
				reader.ReadBits(InSps.Log2MaxPicOrderCntLsbMinus4 + 4u);
			OutHeader.PicOrderCntVal =
				static_cast<OaI32>(OutHeader.PicOrderCntLsb);
			OutHeader.ShortTermRefPicSetSpsFlag = reader.ReadBit() != 0;
			if (OutHeader.ShortTermRefPicSetSpsFlag) {
				const OaU32 stRpsBits = CeilLog2(InSps.NumShortTermRefPicSets);
				if (stRpsBits > 0) {
					reader.SkipBits(stRpsBits); // short_term_ref_pic_set_idx
				}
			} else {
				const OaUsize stRpsStart = GetBitOffset(reader);
				if (InSps.NumShortTermRefPicSets != 0) {
					// Inter-RPS prediction requires the selected SPS RPS contents,
					// which are not retained by the minimal parameter-set parser yet.
					return false;
				}
				const OaU32 negativeCount = reader.ReadUE();
				const OaU32 positiveCount = reader.ReadUE();
				if (negativeCount > 16 || positiveCount > 16) {
					return false;
				}
				OaI32 deltaPoc = 0;
				for (OaU32 i = 0; i < negativeCount; ++i) {
					deltaPoc -= static_cast<OaI32>(reader.ReadUE() + 1u);
					if (reader.ReadBit() != 0) {
						OutHeader.StCurrBeforeDeltaPocs.PushBack(deltaPoc);
					} else {
						OutHeader.StFollDeltaPocs.PushBack(deltaPoc);
					}
				}
				deltaPoc = 0;
				for (OaU32 i = 0; i < positiveCount; ++i) {
					deltaPoc += static_cast<OaI32>(reader.ReadUE() + 1u);
					if (reader.ReadBit() != 0) {
						OutHeader.StCurrAfterDeltaPocs.PushBack(deltaPoc);
					} else {
						OutHeader.StFollDeltaPocs.PushBack(deltaPoc);
					}
				}
				const OaUsize stRpsBits = GetBitOffset(reader) - stRpsStart;
				if (stRpsBits > 0xffffu) {
					return false;
				}
				OutHeader.NumBitsForSTRefPicSetInSlice = static_cast<OaU16>(stRpsBits);
			}
		}
		return true;
	}

	// Parse SPS (Sequence Parameter Set)
	static bool ParseSPS(
		const OaU8* InData,
		OaUsize InSize,
		OaH264SpsData& OutSps)
	{
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		// Skip NAL header (already parsed)
		reader.SkipBits(8);

		// profile_idc
		OutSps.ProfileIdc = reader.ReadBits(8);

		// constraint_set_flags + reserved_zero_2bits
		OutSps.ConstraintFlags = reader.ReadBits(8);

		// level_idc
		OutSps.LevelIdc = reader.ReadBits(8);

		// seq_parameter_set_id
		OutSps.SpsId = reader.ReadUE();

		if (OutSps.ProfileIdc == 100 || OutSps.ProfileIdc == 110 || OutSps.ProfileIdc == 122 ||
			OutSps.ProfileIdc == 244 || OutSps.ProfileIdc == 44 || OutSps.ProfileIdc == 83 ||
			OutSps.ProfileIdc == 86 || OutSps.ProfileIdc == 118 || OutSps.ProfileIdc == 128 ||
			OutSps.ProfileIdc == 138 || OutSps.ProfileIdc == 139 || OutSps.ProfileIdc == 134 ||
			OutSps.ProfileIdc == 135)
		{
			OutSps.ChromaFormatIdc = reader.ReadUE();
			if (OutSps.ChromaFormatIdc == 3)
			{
				OutSps.SeparateColourPlane = reader.ReadBit() != 0;
			}
			OutSps.BitDepthLumaMinus8 = reader.ReadUE();
			OutSps.BitDepthChromaMinus8 = reader.ReadUE();
			OutSps.QpprimeYZeroTransformBypass = reader.ReadBit() != 0;
			if (reader.ReadBit())
			{
				// seq_scaling_matrix_present_flag. Skip scaling lists for now;
				// default matrices are valid when pScalingLists is null.
				OaU32 scalingListCount = OutSps.ChromaFormatIdc != 3 ? 8 : 12;
				for (OaU32 i = 0; i < scalingListCount; ++i)
				{
					if (reader.ReadBit())
					{
						SkipScalingList(reader, i < 6 ? 16 : 64);
					}
				}
			}
		}

		// log2_max_frame_num_minus4
		OutSps.Log2MaxFrameNumMinus4 = reader.ReadUE();

		// pic_order_cnt_type
		OutSps.PicOrderCntType = reader.ReadUE();
		if (OutSps.PicOrderCntType == 0)
		{
			// log2_max_pic_order_cnt_lsb_minus4
			OutSps.Log2MaxPicOrderCntLsbMinus4 = reader.ReadUE();
		}
		else if (OutSps.PicOrderCntType == 1)
		{
			// delta_pic_order_always_zero_flag
			OutSps.DeltaPicOrderAlwaysZero = reader.ReadBit() != 0;
			// offset_for_non_ref_pic
			OutSps.OffsetForNonRefPic = reader.ReadSE();
			// offset_for_top_to_bottom_field
			OutSps.OffsetForTopToBottomField = reader.ReadSE();
			// num_ref_frames_in_pic_order_cnt_cycle
			OutSps.NumRefFramesInPicOrderCntCycle = reader.ReadUE();
			if (OutSps.NumRefFramesInPicOrderCntCycle > OutSps.OffsetForRefFrame.Size())
			{
				return false;
			}
			for (OaU32 i = 0; i < OutSps.NumRefFramesInPicOrderCntCycle; ++i)
			{
				OutSps.OffsetForRefFrame[i] = reader.ReadSE();
			}
		}

		// max_num_ref_frames
		OutSps.MaxNumRefFrames = reader.ReadUE();

		// gaps_in_frame_num_value_allowed_flag
		OutSps.GapsInFrameNumValueAllowed = reader.ReadBit() != 0;

		// pic_width_in_mbs_minus1
		OutSps.PicWidthInMbs = reader.ReadUE() + 1;

		// pic_height_in_map_units_minus1
		OutSps.PicHeightInMbs = reader.ReadUE() + 1;

		// frame_mbs_only_flag
		OutSps.FrameMbsOnly = reader.ReadBit() != 0;
		if (!OutSps.FrameMbsOnly)
		{
			// mb_adaptive_frame_field_flag
			OutSps.MbAdaptiveFrameField = reader.ReadBit() != 0;
		}

		// direct_8x8_inference_flag
		OutSps.Direct8x8Inference = reader.ReadBit() != 0;

		// frame_cropping_flag
		OutSps.FrameCropping = reader.ReadBit() != 0;
		if (OutSps.FrameCropping)
		{
			OutSps.FrameCropLeftOffset = reader.ReadUE();
			OutSps.FrameCropRightOffset = reader.ReadUE();
			OutSps.FrameCropTopOffset = reader.ReadUE();
			OutSps.FrameCropBottomOffset = reader.ReadUE();
		}

		return true;
	}

	// Parse PPS (Picture Parameter Set)
	static bool ParsePPS(
		const OaU8* InData,
		OaUsize InSize,
		OaH264PpsData& OutPps)
	{
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		// Skip NAL header
		reader.SkipBits(8);

		// pic_parameter_set_id
		OutPps.PpsId = reader.ReadUE();

		// seq_parameter_set_id
		OutPps.SpsId = reader.ReadUE();

		OutPps.EntropyCodingMode = reader.ReadBit() != 0;
		OutPps.BottomFieldPicOrderInFramePresent = reader.ReadBit() != 0;
		OaU32 numSliceGroupsMinus1 = reader.ReadUE();
		if (numSliceGroupsMinus1 != 0)
		{
			return false;
		}
		OutPps.NumRefIdxL0DefaultActiveMinus1 = reader.ReadUE();
		OutPps.NumRefIdxL1DefaultActiveMinus1 = reader.ReadUE();
		OutPps.WeightedPred = reader.ReadBit() != 0;
		OutPps.WeightedBipredIdc = reader.ReadBits(2);
		OutPps.PicInitQpMinus26 = reader.ReadSE();
		OutPps.PicInitQsMinus26 = reader.ReadSE();
		OutPps.ChromaQpIndexOffset = reader.ReadSE();
		OutPps.DeblockingFilterControlPresent = reader.ReadBit() != 0;
		OutPps.ConstrainedIntraPred = reader.ReadBit() != 0;
		OutPps.RedundantPicCntPresent = reader.ReadBit() != 0;
		if (MoreRbspData(reader, rbsp))
		{
			OutPps.Transform8x8Mode = reader.ReadBit() != 0;
			if (reader.ReadBit())
			{
				const OaU32 scalingListCount = 6 + (OutPps.Transform8x8Mode ? 2 : 0);
				for (OaU32 i = 0; i < scalingListCount; ++i)
				{
					if (reader.ReadBit())
					{
						SkipScalingList(reader, i < 6 ? 16 : 64);
					}
				}
			}
			OutPps.SecondChromaQpIndexOffset = reader.ReadSE();
		}
		else
		{
			OutPps.SecondChromaQpIndexOffset = OutPps.ChromaQpIndexOffset;
		}

		return true;
	}

	static bool ParseSliceHeaderPrefix(
		const OaU8* InData,
		OaUsize InSize,
		bool InIsIdr,
		OaU32& OutPpsId)
	{
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		// Skip NAL header
		reader.SkipBits(8);

		reader.ReadUE(); // first_mb_in_slice
		reader.ReadUE(); // slice_type
		OutPpsId = reader.ReadUE();
		(void)InIsIdr;
		return true;
	}

	// Parse slice header (Phase 2.4.3)
	static bool ParseSliceHeader(
		const OaU8* InData,
		OaUsize InSize,
		bool InIsIdr,
		OaU8 InNalRefIdc,
		const OaH264SpsData& InSps,
		const OaH264PpsData& InPps,
		OaSliceHeader& OutHeader)
	{
		OaVec<OaU8> rbsp = MakeRbsp(InData, InSize);
		OaBitstreamReader reader(rbsp.Data(), rbsp.Size());

		reader.SkipBits(8); // NAL header

		OutHeader.FirstMbInSlice = reader.ReadUE();

		OaU32 sliceType = reader.ReadUE();
		if (sliceType >= 5)
		{
			sliceType -= 5;
		}
		if (sliceType > static_cast<OaU32>(OaH264SliceType::SI))
		{
			return false;
		}
		OutHeader.SliceType = static_cast<OaH264SliceType>(sliceType);
		OutHeader.PpsId = reader.ReadUE();
		if (OutHeader.PpsId != InPps.PpsId)
		{
			return false;
		}

		const OaU32 frameNumBits = InSps.Log2MaxFrameNumMinus4 + 4;
		if (frameNumBits == 0 || frameNumBits > 16)
		{
			return false;
		}
		OutHeader.FrameNum = reader.ReadBits(frameNumBits);
		if (!InSps.FrameMbsOnly)
		{
			OutHeader.FieldPicFlag = reader.ReadBit() != 0;
			if (OutHeader.FieldPicFlag)
			{
				OutHeader.BottomFieldFlag = reader.ReadBit() != 0;
			}
		}

		// IDR-specific fields
		OutHeader.IsIdrPic = InIsIdr;
		if (InIsIdr)
		{
			OutHeader.IdrPicId = reader.ReadUE();
		}

		OutHeader.PicOrderCntLsb = 0;
		if (InSps.PicOrderCntType == 0)
		{
			const OaU32 pocBits = InSps.Log2MaxPicOrderCntLsbMinus4 + 4;
			if (pocBits == 0 || pocBits > 16)
			{
				return false;
			}
			OutHeader.PicOrderCntLsb = static_cast<OaI32>(reader.ReadBits(pocBits));
			if (InPps.BottomFieldPicOrderInFramePresent && !InSps.FrameMbsOnly)
			{
				reader.ReadSE(); // delta_pic_order_cnt_bottom
			}
		}
		else if (InSps.PicOrderCntType == 1)
		{
			// Full POC type 1 derivation requires previous-picture state. Keep the
			// parse valid but use frame_num as the monotonic DPB order fallback.
			OutHeader.PicOrderCntLsb = static_cast<OaI32>(OutHeader.FrameNum);
		}

		OutHeader.IsReference = InNalRefIdc != 0;

		// Best-effort walk to dec_ref_pic_marking(). The intermediate fields
		// (redundant_pic_cnt / direct_spatial_mv_pred / num_ref_idx_active_override
		// / ref_pic_list_modification / pred_weight_table) cover a lot of stream
		// variants; we parse the common cases conservatively and bail out by
		// marking RefPicMarkingValid=false when we can't follow safely.
		auto attemptRefPicMarking = [&]() -> bool {
			if (InPps.RedundantPicCntPresent) {
				(void)reader.ReadUE(); // redundant_pic_cnt
			}
			const bool isB  = OutHeader.SliceType == OaH264SliceType::B;
			const bool isP  = OutHeader.SliceType == OaH264SliceType::P
			               || OutHeader.SliceType == OaH264SliceType::SP;
			const bool isI  = OutHeader.SliceType == OaH264SliceType::I
			               || OutHeader.SliceType == OaH264SliceType::SI;

			if (isB) {
				reader.SkipBits(1); // direct_spatial_mv_pred_flag
			}

			OaU32 numRefIdxL0 = InPps.NumRefIdxL0DefaultActiveMinus1;
			OaU32 numRefIdxL1 = InPps.NumRefIdxL1DefaultActiveMinus1;
			if (isP || isB) {
				const bool overrideFlag = reader.ReadBit() != 0;
				if (overrideFlag) {
					numRefIdxL0 = reader.ReadUE();
					if (isB) {
						numRefIdxL1 = reader.ReadUE();
					}
				}
			}
			if (numRefIdxL0 >= 32 || (isB && numRefIdxL1 >= 32)) {
				return false;
			}

			// ref_pic_list_modification(): skip lists for P/SP/B slices.
			auto skipRefPicListModification = [&](bool InActive) {
				if (!InActive) { return; }
				const bool present = reader.ReadBit() != 0;
				if (!present) { return; }
				for (OaU32 guard = 0; guard < 256; ++guard) {
					const OaU32 op = reader.ReadUE();
					if (op == 3) { break; }
					(void)reader.ReadUE();  // abs_diff_pic_num_minus1 / long_term_pic_num
				}
			};
			if (isP || isB) { skipRefPicListModification(true); }
			if (isB)        { skipRefPicListModification(true); }

			// pred_weight_table() — only when explicit weighting is in play.
			const bool weightedPred = isP && InPps.WeightedPred;
			const bool weightedBi   = isB && (InPps.WeightedBipredIdc == 1);
			if (weightedPred || weightedBi) {
				if (!SkipH264PredWeightTable(
					reader,
					InSps,
					numRefIdxL0,
					numRefIdxL1,
					isB)) {
					return false;
				}
			}
			(void)isI;

			// dec_ref_pic_marking()
			if (InNalRefIdc == 0) {
				return true;  // No marking section for non-reference slices.
			}
			if (InIsIdr) {
				OutHeader.NoOutputOfPriorPics = reader.ReadBit() != 0;
				OutHeader.LongTermReference   = reader.ReadBit() != 0;
				return true;
			}
			OutHeader.AdaptiveRefPicMarking = reader.ReadBit() != 0;
			if (!OutHeader.AdaptiveRefPicMarking) {
				return true;
			}
			for (OaU32 guard = 0; guard < 64; ++guard) {
				OaH264MmcoCommand cmd{};
				cmd.Op = reader.ReadUE();
				if (cmd.Op == 0) { break; }
				switch (cmd.Op) {
					case 1: cmd.DifferenceOfPicNumsMinus1 = reader.ReadUE(); break;
					case 2: cmd.LongTermPicNum             = reader.ReadUE(); break;
					case 3:
						cmd.DifferenceOfPicNumsMinus1 = reader.ReadUE();
						cmd.LongTermFrameIdx          = reader.ReadUE();
						break;
					case 4: cmd.MaxLongTermFrameIdxPlus1   = reader.ReadUE(); break;
					case 5: break;  // no operands
					case 6: cmd.LongTermFrameIdx           = reader.ReadUE(); break;
					default:
						return false;  // unknown op → bail
				}
				OutHeader.MmcoCommands.PushBack(cmd);
			}
			return true;
		};
		OutHeader.RefPicMarkingValid = attemptRefPicMarking();

		return true;
	}
};
