// OA Vision — AV1 Codec Parser
// Extracts sequence/frame headers from OBUs and converts to Vulkan Video
// structures

#pragma once

#include "VideoCodecParser.h"
#include <Oa/Core/Std/Vec.h>
#include <vk_video/vulkan_video_codec_av1std.h>
#include <vk_video/vulkan_video_codec_av1std_decode.h>

static constexpr OaU32 OaAv1MaxReferencesPerFrame = 7U;

struct OaIvfFrame {
	OaUsize Offset = 0;
	OaUsize Size = 0;
	OaU64 Timestamp = 0;
};

enum class OaAv1ObuType : OaU8 {
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

struct OaAv1Obu {
	OaAv1ObuType Type = OaAv1ObuType::Padding;
	OaUsize HeaderOffset = 0;
	OaUsize HeaderSize = 0;
	OaUsize PayloadOffset = 0;
	OaUsize PayloadSize = 0;
};

struct OaAv1SequenceHeaderInfo {
	StdVideoAV1Profile SeqProfile = STD_VIDEO_AV1_PROFILE_MAIN;
	OaU32 FrameWidthBitsMinus1 = 0;
	OaU32 FrameHeightBitsMinus1 = 0;
	OaU32 MaxFrameWidthMinus1 = 0;
	OaU32 MaxFrameHeightMinus1 = 0;
	OaU32 OrderHintBits = 0;
	OaI32 SeqForceScreenContentTools = 2;
	OaI32 SeqForceIntegerMv = 2;
	bool StillPicture = false;
	bool ReducedStillPictureHeader = false;
	bool Use128x128Superblock = false;
	bool EnableFilterIntra = false;
	bool EnableIntraEdgeFilter = false;
	bool EnableInterIntraCompound = false;
	bool EnableMaskedCompound = false;
	bool EnableWarpedMotion = false;
	bool EnableDualFilter = false;
	bool EnableOrderHint = false;
	bool EnableJntComp = false;
	bool EnableRefFrameMvs = false;
	bool EnableSuperres = false;
	bool EnableCdef = false;
	bool EnableRestoration = false;
	bool FilmGrainParamsPresent = false;
	bool FrameIdNumbersPresent = false;
	bool TimingInfoPresent = false;
	bool InitialDisplayDelayPresent = false;
	OaU8 DeltaFrameIdLengthMinus2 = 0;
	OaU8 AdditionalFrameIdLengthMinus1 = 0;
	StdVideoAV1TimingInfo TimingInfo = {};
	StdVideoAV1ColorConfig ColorConfig = {};
};

struct OaAv1FrameHeaderInfo {
	OaUsize HeaderSize = 0;
	bool ShowExistingFrame = false;
	OaU8 FrameToShowMapIdx = 0;
	OaU32 TileCols = 1;
	OaU32 TileRows = 1;
	OaU32 TileColsLog2 = 0;
	OaU32 TileRowsLog2 = 0;
	OaU32 TileSizeBytesMinus1 = 0;
	OaU32 ContextUpdateTileId = 0;
	OaU32 OrderHint = 0;
	OaU32 PrimaryRefFrame = STD_VIDEO_AV1_PRIMARY_REF_NONE;
	OaU32 RefreshFrameFlags = 0xff;
	OaU8 OrderHints[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
	OaU8 RefFrameSignBias = 0;
	OaI32 ReferenceNameSlotIndices[OaAv1MaxReferencesPerFrame] = {-1, -1, -1, -1, -1, -1, -1};
	OaU32 BaseQIdx = 128;
	OaI32 DeltaQYDc = 0;
	OaI32 DeltaQUDc = 0;
	OaI32 DeltaQUAc = 0;
	OaI32 DeltaQVDc = 0;
	OaI32 DeltaQVAc = 0;
	OaU32 QmY = 0;
	OaU32 QmU = 0;
	OaU32 QmV = 0;
	OaU32 DeltaQRes = 0;
	OaU32 DeltaLfRes = 0;
	OaU32 LoopFilterSharpness = 0;
	OaU32 CdefDampingMinus3 = 0;
	OaU32 CdefBits = 0;
	OaU8 LoopFilterLevels[4] = {};
	OaU8 LoopFilterUpdateRefDelta[STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME] = {};
	OaI8 LoopFilterRefDeltas[STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME] = {1, 0, 0, 0, -1, 0, -1, -1};
	OaU8 LoopFilterUpdateModeDelta[STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS] = {};
	OaI8 LoopFilterModeDeltas[STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS] = {};
	OaU8 CdefYPriStrength[8] = {};
	OaU8 CdefYSecStrength[8] = {};
	OaU8 CdefUvPriStrength[8] = {};
	OaU8 CdefUvSecStrength[8] = {};
	StdVideoAV1FrameRestorationType RestorationTypes[STD_VIDEO_AV1_MAX_NUM_PLANES] = {
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
		STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
	};
	OaU16 RestorationSizes[STD_VIDEO_AV1_MAX_NUM_PLANES] = {};
	OaU8 SegmentFeatureEnabled[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = {};
	OaI16 SegmentFeatureData[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = {};
	StdVideoAV1FrameType FrameType = STD_VIDEO_AV1_FRAME_TYPE_KEY;
	StdVideoAV1TxMode TxMode = STD_VIDEO_AV1_TX_MODE_LARGEST;
	bool IsKeyFrame = false;
	bool ShowFrame = false;
	bool ShowableFrame = false;
	bool DisableCdfUpdate = false;
	bool DisableFrameEndUpdateCdf = false;
	bool AllowScreenContentTools = false;
	bool UsingQMatrix = false;
	bool DiffUvDelta = false;
	bool SegmentationEnabled = false;
	bool SegmentationUpdateMap = false;
	bool SegmentationTemporalUpdate = false;
	bool SegmentationUpdateData = false;
	bool DeltaQPresent = false;
	bool DeltaLfPresent = false;
	bool DeltaLfMulti = false;
	bool LoopFilterDeltaEnabled = false;
	bool LoopFilterDeltaUpdate = false;
	bool UsesLr = false;
	bool UsesChromaLr = false;
	bool ReducedTxSet = false;
	bool ErrorResilientMode = false;
	bool FrameSizeOverrideFlag = false;
	bool UseSuperres = false;
	OaU8 CodedDenom = 0;
	bool RenderAndFrameSizeDifferent = false;
	bool AllowIntraBc = false;
	bool ForceIntegerMv = true;
	bool AllowHighPrecisionMv = false;
	bool IsFilterSwitchable = false;
	bool IsMotionModeSwitchable = false;
	bool UseRefFrameMvs = false;
	bool FrameRefsShortSignaling = false;
	bool ReferenceSelect = false;
	bool SkipModePresent = false;
	OaU8 SkipModeFrame[2] = {};
	bool AllowWarpedMotion = false;
	bool ApplyGrain = false;
	StdVideoAV1InterpolationFilter InterpolationFilter = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP;
};

struct OaAv1TileGroupInfo {
	OaVec<OaU32> TileOffsets;
	OaVec<OaU32> TileSizes;
};

// Structural inventory of one container access unit / AV1 temporal unit.  This
// is intentionally independent of Vulkan and decoded-picture state: callers can
// use it to detect the common case where one MP4 packet contains several hidden
// reference frames plus a displayed frame.
struct OaAv1AccessUnitInfo {
	OaU32 SequenceHeaderCount = 0;
	OaU32 FrameCount = 0;
	OaU32 FrameHeaderCount = 0;
	OaU32 TileGroupCount = 0;

	[[nodiscard]] OaU32 PictureCount() const { return FrameCount + FrameHeaderCount; }
};

// Picture descriptor produced by parsing an AV1 access unit (IVF frame
// payload).
struct OaAv1PictureDesc {
	bool HasPicture = false;
	bool ShowExistingFrame = false;
	OaU8 FrameToShowMapIdx = 0;
	OaIvfFrame Frame = {};
	OaAv1SequenceHeaderInfo SequenceHeader = {};
	OaAv1FrameHeaderInfo FrameHeader = {};
	OaAv1TileGroupInfo TileGroup = {};
	OaUsize FrameHeaderOffset = 0;
	// Byte range of the decode OBU (OBU_FRAME or OBU_FRAME_HEADER) inside the
	// frame payload.
	OaUsize DecodeObuOffset = 0;
	OaUsize DecodeObuSize = 0;
	OaVec<OaU32> TileOffsets;
	OaVec<OaU32> TileSizes;
};

// AV1 codec parser implementation
class OaVcpAv1 : public OaVideoCodecParser {
public:
	OaVcpAv1() = default;
	~OaVcpAv1() override = default;

	// Parse AV1 OBUs (Open Bitstream Units)
	// SPS = Sequence Header OBU
	// PPS = Frame Header OBU
	OaStatus ParseSps(const OaSpan<const OaU8>& InObu) override;
	OaStatus ParsePps(const OaSpan<const OaU8>& InObu) override;

	// Clear all cached parameter sets
	void ClearParameterSets() override;

	// Parse a complete access unit (IVF-wrapped AV1 bitstream).
	OaStatus ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaAv1PictureDesc& OutDesc);
	OaStatus ParseAccessUnitPictures(const OaSpan<const OaU8>& InBitstream,
		OaVec<OaAv1PictureDesc>& OutDescs);
	OaStatus InspectAccessUnit(const OaSpan<const OaU8>& InBitstream,
		OaAv1AccessUnitInfo& OutInfo) const;
	[[nodiscard]] bool HasSequenceHeader() const noexcept
	{
		return HasCachedSequenceHeader_;
	}
	[[nodiscard]] const OaAv1SequenceHeaderInfo& GetSequenceHeader() const noexcept
	{
		return CachedSequenceHeader_;
	}

private:
	// Sequence header cache. In MP4/ISO-BMFF the sequence header is carried
	// out-of-band (av1C) and only prepended to keyframes; inter-frame temporal
	// units omit it entirely. We cache the last parsed sequence header so those
	// frames can still be decoded (mirrors SPS/PPS caching for H.264/H.265).
	OaAv1SequenceHeaderInfo CachedSequenceHeader_ = {};
	bool HasCachedSequenceHeader_ = false;
	OaArray<OaU8, STD_VIDEO_AV1_NUM_REF_FRAMES> CachedRefOrderHints_ = {};
	OaArray<bool, STD_VIDEO_AV1_NUM_REF_FRAMES> CachedRefValid_ = {};
};
