// OA Vision — AV1 codec parser
// Extracts sequence/frame headers from OBUs and converts to vulkan Video
// structures

#pragma once

#include "videoCodecParser.h"
#include <oa/core/std/vec.h>
#include <vk_video/vulkan_video_codec_av1std.h>
#include <vk_video/vulkan_video_codec_av1std_decode.h>

namespace oa {

static constexpr oa::U32 Av1MaxReferencesPerFrame = 7U;

struct Av1IvfFrame {
	oa::Usize offset = 0;
	oa::Usize size = 0;
	oa::U64 timestamp = 0;
};

enum class Av1ObuType : oa::U8 {
	SequenceHeader = 1,
	TemporalDelimiter = 2,
	FrameHeader = 3,
	TileGroup = 4,
	Metadata = 5,
	Frame = 6,
	RedundantFrameHeader = 7,
	TileList = 8,
	Padding = 15,
};

struct Av1Obu {
	Av1ObuType type = Av1ObuType::Padding;
	oa::Usize headerOffset = 0;
	oa::Usize headerSize = 0;
	oa::Usize payloadOffset = 0;
	oa::Usize payloadSize = 0;
};

struct Av1SequenceHeaderInfo {
	StdVideoAV1Profile seqProfile = STD_VIDEO_AV1_PROFILE_MAIN;
	oa::U32 frameWidthBitsMinus1 = 0;
	oa::U32 frameHeightBitsMinus1 = 0;
	oa::U32 maxFrameWidthMinus1 = 0;
	oa::U32 maxFrameHeightMinus1 = 0;
	oa::U32 orderHintBits = 0;
	oa::I32 seqForceScreenContentTools = 2;
	oa::I32 seqForceIntegerMv = 2;
	bool stillPicture = false;
	bool reducedStillPictureHeader = false;
	bool use128x128Superblock = false;
	bool enableFilterIntra = false;
	bool enableIntraEdgeFilter = false;
	bool enableInterIntraCompound = false;
	bool enableMaskedCompound = false;
	bool enableWarpedMotion = false;
	bool enableDualFilter = false;
	bool enableOrderHint = false;
	bool enableJntComp = false;
	bool enableRefFrameMvs = false;
	bool enableSuperres = false;
	bool enableCdef = false;
	bool enableRestoration = false;
	bool filmGrainParamsPresent = false;
	bool frameIdNumbersPresent = false;
	bool timingInfoPresent = false;
	bool initialDisplayDelayPresent = false;
	oa::U8 deltaFrameIdLengthMinus2 = 0;
	oa::U8 additionalFrameIdLengthMinus1 = 0;
	StdVideoAV1TimingInfo timingInfo = {};
	StdVideoAV1ColorConfig colorConfig = {};
};

struct Av1FrameHeaderInfo {
	oa::Usize headerSize = 0;
	bool showExistingFrame = false;
	oa::U8 frameToShowMapIdx = 0;
	oa::U32 tileCols = 1;
	oa::U32 tileRows = 1;
	oa::U32 tileColsLog2 = 0;
	oa::U32 tileRowsLog2 = 0;
	oa::U32 tileSizeBytesMinus1 = 0;
	oa::U32 contextUpdateTileId = 0;
	oa::U32 orderHint = 0;
	oa::U32 primaryRefFrame = STD_VIDEO_AV1_PRIMARY_REF_NONE;
	oa::U32 refreshFrameFlags = 0xff;
	oa::U8 orderHints[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
	oa::U8 refFrameSignBias = 0;
	oa::I32 referenceNameSlotIndices[Av1MaxReferencesPerFrame] = {-1, -1, -1, -1, -1, -1, -1};
	oa::U32 baseQIdx = 128;
	oa::I32 deltaQYDc = 0;
	oa::I32 deltaQUDc = 0;
	oa::I32 deltaQUAc = 0;
	oa::I32 deltaQVDc = 0;
	oa::I32 deltaQVAc = 0;
	oa::U32 qmY = 0;
	oa::U32 qmU = 0;
	oa::U32 qmV = 0;
	oa::U32 deltaQRes = 0;
	oa::U32 deltaLfRes = 0;
	oa::U32 loopFilterSharpness = 0;
	oa::U32 cdefDampingMinus3 = 0;
	oa::U32 cdefBits = 0;
	oa::U8 loopFilterLevels[4] = {};
	oa::U8 loopFilterUpdateRefDelta[STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME] = {};
	oa::I8 loopFilterRefDeltas[STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME] = {1, 0, 0, 0, -1, 0, -1, -1};
	oa::U8 loopFilterUpdateModeDelta[STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS] = {};
	oa::I8 loopFilterModeDeltas[STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS] = {};
	oa::U8 cdefYPriStrength[8] = {};
	oa::U8 cdefYSecStrength[8] = {};
	oa::U8 cdefUvPriStrength[8] = {};
	oa::U8 cdefUvSecStrength[8] = {};
	StdVideoAV1FrameRestorationType restorationTypes[STD_VIDEO_AV1_MAX_NUM_PLANES] = {
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
	};
	oa::U16 restorationSizes[STD_VIDEO_AV1_MAX_NUM_PLANES] = {};
	oa::U8 segmentFeatureEnabled[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = {};
	oa::I16 segmentFeatureData[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = {};
	StdVideoAV1FrameType frameType = STD_VIDEO_AV1_FRAME_TYPE_KEY;
	StdVideoAV1TxMode txMode = STD_VIDEO_AV1_TX_MODE_LARGEST;
	bool isKeyFrame = false;
	bool showFrame = false;
	bool showableFrame = false;
	bool disableCdfUpdate = false;
	bool disableFrameEndUpdateCdf = false;
	bool allowScreenContentTools = false;
	bool usingQMatrix = false;
	bool diffUvDelta = false;
	bool segmentationEnabled = false;
	bool segmentationUpdateMap = false;
	bool segmentationTemporalUpdate = false;
	bool segmentationUpdateData = false;
	bool deltaQPresent = false;
	bool deltaLfPresent = false;
	bool deltaLfMulti = false;
	bool loopFilterDeltaEnabled = false;
	bool loopFilterDeltaUpdate = false;
	bool usesLr = false;
	bool usesChromaLr = false;
	bool reducedTxSet = false;
	bool errorResilientMode = false;
	bool frameSizeOverrideFlag = false;
	bool useSuperres = false;
	oa::U8 codedDenom = 0;
	bool renderAndFrameSizeDifferent = false;
	bool allowIntraBc = false;
	bool forceIntegerMv = true;
	bool allowHighPrecisionMv = false;
	bool isFilterSwitchable = false;
	bool isMotionModeSwitchable = false;
	bool useRefFrameMvs = false;
	bool frameRefsShortSignaling = false;
	bool referenceSelect = false;
	bool skipModePresent = false;
	oa::U8 skipModeFrame[2] = {};
	bool allowWarpedMotion = false;
	bool applyGrain = false;
	StdVideoAV1InterpolationFilter interpolationFilter = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP;
};

struct Av1TileGroupInfo {
	oa::Vec<oa::U32> tileOffsets;
	oa::Vec<oa::U32> tileSizes;
};

// Structural inventory of one container access unit / AV1 temporal unit.  This
// is intentionally independent of vulkan and decoded-picture state: callers can
// use it to detect the common case where one MP4 packet contains several hidden
// reference frames plus a displayed frame.
struct Av1AccessUnitInfo {
	oa::U32 sequenceHeaderCount = 0;
	oa::U32 frameCount = 0;
	oa::U32 frameHeaderCount = 0;
	oa::U32 tileGroupCount = 0;

	[[nodiscard]] oa::U32 pictureCount() const { return frameCount + frameHeaderCount; }
};

// Picture descriptor produced by parsing an AV1 access unit (IVF frame
// payload).
struct Av1PictureDesc {
	bool hasPicture = false;
	bool showExistingFrame = false;
	oa::U8 frameToShowMapIdx = 0;
	Av1IvfFrame frame = {};
	Av1SequenceHeaderInfo sequenceHeader = {};
	Av1FrameHeaderInfo frameHeader = {};
	Av1TileGroupInfo tileGroup = {};
	oa::Usize frameHeaderOffset = 0;
	// Byte range of the decode OBU (OBU_FRAME or OBU_FRAME_HEADER) inside the
	// frame payload.
	oa::Usize decodeObuOffset = 0;
	oa::Usize decodeObuSize = 0;
	oa::Vec<oa::U32> tileOffsets;
	oa::Vec<oa::U32> tileSizes;
};

// AV1 codec parser implementation
class VcpAv1 : public VideoCodecParser {
public:
	VcpAv1() = default;
	~VcpAv1() override = default;

	// parse AV1 oBUs (open bitstream Units)
	// SPS = sequence header OBU
	// PPS = Frame header OBU
	oa::Status parseSps(const oa::Span<const oa::U8>& inObu) override;
	oa::Status parsePps(const oa::Span<const oa::U8>& inObu) override;

	// clear all cached parameter sets
	void clearParameterSets() override;

	// parse a complete access unit (IVF-wrapped AV1 bitstream).
	oa::Status parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, Av1PictureDesc& outDesc);
	oa::Status parseAccessUnitPictures(const oa::Span<const oa::U8>& inBitstream,	oa::Vec<Av1PictureDesc>& outDescs);
	oa::Status inspectAccessUnit(const oa::Span<const oa::U8>& inBitstream,	Av1AccessUnitInfo& outInfo) const;
	[[nodiscard]] bool hasSequenceHeader() const noexcept	{	return hasCachedSequenceHeader_; }
	[[nodiscard]] const Av1SequenceHeaderInfo& getSequenceHeader() const noexcept	{	return cachedSequenceHeader_;	}

private:
	// sequence header cache. in MP4/ISO-BMFF the sequence header is carried
	// out-of-band (av1C) and only prepended to keyframes; inter-frame temporal
	// units omit it entirely. We cache the last parsed sequence header so those
	// frames can still be decoded (mirrors SPS/PPS caching for H.264/H.265).
	Av1SequenceHeaderInfo cachedSequenceHeader_ = {};
	bool hasCachedSequenceHeader_ = false;
	oa::Array<oa::U8, STD_VIDEO_AV1_NUM_REF_FRAMES> cachedRefOrderHints_ = {};
	oa::Array<bool, STD_VIDEO_AV1_NUM_REF_FRAMES> cachedRefValid_ = {};
};

} // namespace oa
