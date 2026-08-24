// OA Vision — H.264/H.265 parameter set structs (CPU parse layer).
// Canonical types for Vcp* parsers and decoder session-parameter upload.

#pragma once

#include <oa/core/types.h>

namespace oa {

struct H264SpsData {
	oa::U32 spsId;
	oa::U32 profileIdc;
	oa::U32 levelIdc;
	oa::U32 constraintFlags;
	oa::U32 chromaFormatIdc = 1;
	oa::U32 bitDepthLumaMinus8 = 0;
	oa::U32 bitDepthChromaMinus8 = 0;
	oa::U32 log2MaxFrameNumMinus4 = 0;
	oa::U32 picOrderCntType = 0;
	oa::I32 offsetForNonRefPic = 0;
	oa::I32 offsetForTopToBottomField = 0;
	oa::U32 log2MaxPicOrderCntLsbMinus4 = 0;
	oa::U32 numRefFramesInPicOrderCntCycle = 0;
	oa::U32 picWidthInMbs;
	oa::U32 picHeightInMbs;
	oa::U32 maxNumRefFrames;
	oa::Array<oa::I32, 256> offsetForRefFrame = {};
	bool deltaPicOrderAlwaysZero = false;
	bool separateColourPlane = false;
	bool qpprimeYZeroTransformBypass = false;
	bool gapsInFrameNumValueAllowed = false;
	bool frameMbsOnly = true;
	bool mbAdaptiveFrameField = false;
	bool direct8x8Inference = false;
	bool frameCropping = false;
	oa::U32 frameCropLeftOffset = 0;
	oa::U32 frameCropRightOffset = 0;
	oa::U32 frameCropTopOffset = 0;
	oa::U32 frameCropBottomOffset = 0;
};

struct H264PpsData {
	oa::U32 ppsId;
	oa::U32 spsId;
	oa::U32 numRefIdxL0DefaultActiveMinus1 = 0;
	oa::U32 numRefIdxL1DefaultActiveMinus1 = 0;
	oa::U32 weightedBipredIdc = 0;
	oa::I32 picInitQpMinus26 = 0;
	oa::I32 picInitQsMinus26 = 0;
	oa::I32 chromaQpIndexOffset = 0;
	oa::I32 secondChromaQpIndexOffset = 0;
	bool entropyCodingMode = false;
	bool bottomFieldPicOrderInFramePresent = false;
	bool weightedPred = false;
	bool deblockingFilterControlPresent = false;
	bool constrainedIntraPred = false;
	bool redundantPicCntPresent = false;
	bool transform8x8Mode = false;
};

struct H265VpsData {
	oa::U32 vpsId = 0;
	oa::U32 maxSubLayersMinus1 = 0;
	bool temporalIdNesting = false;
	oa::U32 generalProfileIdc = 1;
	oa::U32 generalLevelIdc = 0;
	bool generalTierFlag = false;
	bool generalProgressiveSourceFlag = false;
	bool generalInterlacedSourceFlag = false;
	bool generalNonPackedConstraintFlag = false;
	bool generalFrameOnlyConstraintFlag = false;
};

struct H265SpsData {
	oa::U32 spsId = 0;
	oa::U32 vpsId = 0;
	oa::U32 maxSubLayersMinus1 = 0;
	oa::U32 chromaFormatIdc = 1;
	oa::U32 width = 0;
	oa::U32 height = 0;
	oa::U32 codedWidth = 0;
	oa::U32 codedHeight = 0;
	oa::U32 conformanceWindowLeft = 0;
	oa::U32 conformanceWindowRight = 0;
	oa::U32 conformanceWindowTop = 0;
	oa::U32 conformanceWindowBottom = 0;
	oa::U32 bitDepthLumaMinus8 = 0;
	oa::U32 bitDepthChromaMinus8 = 0;
	oa::U32 log2MaxPicOrderCntLsbMinus4 = 0;
	oa::U32 log2MinLumaCodingBlockSizeMinus3 = 0;
	oa::U32 log2DiffMaxMinLumaCodingBlockSize = 0;
	oa::U32 log2MinLumaTransformBlockSizeMinus2 = 0;
	oa::U32 log2DiffMaxMinLumaTransformBlockSize = 0;
	oa::U32 maxTransformHierarchyDepthInter = 0;
	oa::U32 maxTransformHierarchyDepthIntra = 0;
	oa::U32 numShortTermRefPicSets = 0;
	bool temporalIdNesting = false;
	bool separateColourPlane = false;
	bool spsSubLayerOrderingInfoPresent = false;
	bool scalingListEnabled = false;
	bool spsScalingListDataPresent = false;
	bool ampEnabled = false;
	bool sampleAdaptiveOffsetEnabled = false;
	bool pcmEnabled = false;
	bool longTermRefPicsPresent = false;
	bool temporalMvpEnabled = false;
	bool strongIntraSmoothingEnabled = false;
	oa::Array<oa::U32, 7> maxDecPicBufferingMinus1 = {};
	oa::Array<oa::U32, 7> maxNumReorderPics = {};
	oa::Array<oa::U32, 7> maxLatencyIncreasePlus1 = {};
};

struct H265PpsData {
	oa::U32 ppsId = 0;
	oa::U32 spsId = 0;
	oa::U32 numExtraSliceHeaderBits = 0;
	oa::U32 numRefIdxL0DefaultActiveMinus1 = 0;
	oa::U32 numRefIdxL1DefaultActiveMinus1 = 0;
	oa::I32 initQpMinus26 = 0;
	oa::U32 diffCuQpDeltaDepth = 0;
	oa::I32 cbQpOffset = 0;
	oa::I32 crQpOffset = 0;
	oa::I32 betaOffsetDiv2 = 0;
	oa::I32 tcOffsetDiv2 = 0;
	oa::U32 log2ParallelMergeLevelMinus2 = 0;
	oa::U32 numTileColumnsMinus1 = 0;
	oa::U32 numTileRowsMinus1 = 0;
	bool dependentSliceSegmentsEnabled = false;
	bool outputFlagPresent = false;
	bool signDataHidingEnabled = false;
	bool cabacInitPresent = false;
	bool constrainedIntraPred = false;
	bool transformSkipEnabled = false;
	bool cuQpDeltaEnabled = false;
	bool ppsSliceChromaQpOffsetsPresent = false;
	bool weightedPred = false;
	bool weightedBipred = false;
	bool transquantBypassEnabled = false;
	bool tilesEnabled = false;
	bool entropyCodingSyncEnabled = false;
	bool uniformSpacing = true;
	bool loopFilterAcrossTilesEnabled = false;
	bool ppsLoopFilterAcrossSlicesEnabled = false;
	bool deblockingFilterControlPresent = false;
	bool deblockingFilterOverrideEnabled = false;
	bool ppsDeblockingFilterDisabled = false;
	bool ppsScalingListDataPresent = false;
	bool listsModificationPresent = false;
	bool sliceSegmentHeaderExtensionPresent = false;
	bool ppsExtensionPresent = false;
};

} // namespace oa
