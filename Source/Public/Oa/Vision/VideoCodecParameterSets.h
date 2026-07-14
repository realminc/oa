// OA Vision — H.264/H.265 parameter set structs (CPU parse layer).
// Canonical types for Vcp* parsers and decoder session-parameter upload.

#pragma once

#include <Oa/Core/Types.h>

struct OaH264SpsData {
	OaU32 SpsId;
	OaU32 ProfileIdc;
	OaU32 LevelIdc;
	OaU32 ConstraintFlags;
	OaU32 ChromaFormatIdc = 1;
	OaU32 BitDepthLumaMinus8 = 0;
	OaU32 BitDepthChromaMinus8 = 0;
	OaU32 Log2MaxFrameNumMinus4 = 0;
	OaU32 PicOrderCntType = 0;
	OaI32 OffsetForNonRefPic = 0;
	OaI32 OffsetForTopToBottomField = 0;
	OaU32 Log2MaxPicOrderCntLsbMinus4 = 0;
	OaU32 NumRefFramesInPicOrderCntCycle = 0;
	OaU32 PicWidthInMbs;
	OaU32 PicHeightInMbs;
	OaU32 MaxNumRefFrames;
	OaArray<OaI32, 256> OffsetForRefFrame = {};
	bool DeltaPicOrderAlwaysZero = false;
	bool SeparateColourPlane = false;
	bool QpprimeYZeroTransformBypass = false;
	bool GapsInFrameNumValueAllowed = false;
	bool FrameMbsOnly = true;
	bool MbAdaptiveFrameField = false;
	bool Direct8x8Inference = false;
	bool FrameCropping = false;
	OaU32 FrameCropLeftOffset = 0;
	OaU32 FrameCropRightOffset = 0;
	OaU32 FrameCropTopOffset = 0;
	OaU32 FrameCropBottomOffset = 0;
};

struct OaH264PpsData {
	OaU32 PpsId;
	OaU32 SpsId;
	OaU32 NumRefIdxL0DefaultActiveMinus1 = 0;
	OaU32 NumRefIdxL1DefaultActiveMinus1 = 0;
	OaU32 WeightedBipredIdc = 0;
	OaI32 PicInitQpMinus26 = 0;
	OaI32 PicInitQsMinus26 = 0;
	OaI32 ChromaQpIndexOffset = 0;
	OaI32 SecondChromaQpIndexOffset = 0;
	bool EntropyCodingMode = false;
	bool BottomFieldPicOrderInFramePresent = false;
	bool WeightedPred = false;
	bool DeblockingFilterControlPresent = false;
	bool ConstrainedIntraPred = false;
	bool RedundantPicCntPresent = false;
	bool Transform8x8Mode = false;
};

struct OaH265VpsData {
	OaU32 VpsId = 0;
	OaU32 MaxSubLayersMinus1 = 0;
	bool TemporalIdNesting = false;
	OaU32 GeneralProfileIdc = 1;
	OaU32 GeneralLevelIdc = 0;
	bool GeneralTierFlag = false;
	bool GeneralProgressiveSourceFlag = false;
	bool GeneralInterlacedSourceFlag = false;
	bool GeneralNonPackedConstraintFlag = false;
	bool GeneralFrameOnlyConstraintFlag = false;
};

struct OaH265SpsData {
	OaU32 SpsId = 0;
	OaU32 VpsId = 0;
	OaU32 MaxSubLayersMinus1 = 0;
	OaU32 ChromaFormatIdc = 1;
	OaU32 Width = 0;
	OaU32 Height = 0;
	OaU32 CodedWidth = 0;
	OaU32 CodedHeight = 0;
	OaU32 ConformanceWindowLeft = 0;
	OaU32 ConformanceWindowRight = 0;
	OaU32 ConformanceWindowTop = 0;
	OaU32 ConformanceWindowBottom = 0;
	OaU32 BitDepthLumaMinus8 = 0;
	OaU32 BitDepthChromaMinus8 = 0;
	OaU32 Log2MaxPicOrderCntLsbMinus4 = 0;
	OaU32 Log2MinLumaCodingBlockSizeMinus3 = 0;
	OaU32 Log2DiffMaxMinLumaCodingBlockSize = 0;
	OaU32 Log2MinLumaTransformBlockSizeMinus2 = 0;
	OaU32 Log2DiffMaxMinLumaTransformBlockSize = 0;
	OaU32 MaxTransformHierarchyDepthInter = 0;
	OaU32 MaxTransformHierarchyDepthIntra = 0;
	OaU32 NumShortTermRefPicSets = 0;
	bool TemporalIdNesting = false;
	bool SeparateColourPlane = false;
	bool SpsSubLayerOrderingInfoPresent = false;
	bool ScalingListEnabled = false;
	bool SpsScalingListDataPresent = false;
	bool AmpEnabled = false;
	bool SampleAdaptiveOffsetEnabled = false;
	bool PcmEnabled = false;
	bool LongTermRefPicsPresent = false;
	bool TemporalMvpEnabled = false;
	bool StrongIntraSmoothingEnabled = false;
	OaArray<OaU32, 7> MaxDecPicBufferingMinus1 = {};
	OaArray<OaU32, 7> MaxNumReorderPics = {};
	OaArray<OaU32, 7> MaxLatencyIncreasePlus1 = {};
};

struct OaH265PpsData {
	OaU32 PpsId = 0;
	OaU32 SpsId = 0;
	OaU32 NumExtraSliceHeaderBits = 0;
	OaU32 NumRefIdxL0DefaultActiveMinus1 = 0;
	OaU32 NumRefIdxL1DefaultActiveMinus1 = 0;
	OaI32 InitQpMinus26 = 0;
	OaU32 DiffCuQpDeltaDepth = 0;
	OaI32 CbQpOffset = 0;
	OaI32 CrQpOffset = 0;
	OaI32 BetaOffsetDiv2 = 0;
	OaI32 TcOffsetDiv2 = 0;
	OaU32 Log2ParallelMergeLevelMinus2 = 0;
	OaU32 NumTileColumnsMinus1 = 0;
	OaU32 NumTileRowsMinus1 = 0;
	bool DependentSliceSegmentsEnabled = false;
	bool OutputFlagPresent = false;
	bool SignDataHidingEnabled = false;
	bool CabacInitPresent = false;
	bool ConstrainedIntraPred = false;
	bool TransformSkipEnabled = false;
	bool CuQpDeltaEnabled = false;
	bool PpsSliceChromaQpOffsetsPresent = false;
	bool WeightedPred = false;
	bool WeightedBipred = false;
	bool TransquantBypassEnabled = false;
	bool TilesEnabled = false;
	bool EntropyCodingSyncEnabled = false;
	bool UniformSpacing = true;
	bool LoopFilterAcrossTilesEnabled = false;
	bool PpsLoopFilterAcrossSlicesEnabled = false;
	bool DeblockingFilterControlPresent = false;
	bool DeblockingFilterOverrideEnabled = false;
	bool PpsDeblockingFilterDisabled = false;
	bool PpsScalingListDataPresent = false;
	bool ListsModificationPresent = false;
	bool SliceSegmentHeaderExtensionPresent = false;
	bool PpsExtensionPresent = false;
};
