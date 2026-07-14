// OA Vision — VP9 Codec Parser
// Extracts frame headers and converts to Vulkan Video structures
// Vulkan Video Extension: VK_KHR_video_decode_vp9

#pragma once

#include "VideoCodecParser.h"
#include <Oa/Core/Std/Vec.h>
#include <vk_video/vulkan_video_codec_vp9std.h>
#include <vk_video/vulkan_video_codec_vp9std_decode.h>

struct OaVp9IvfFrame {
	OaUsize Offset = 0;
	OaUsize Size = 0;
	OaU64 Timestamp = 0;
};

// Picture descriptor produced by parsing a VP9 access unit (IVF frame or raw frame).
struct OaVp9PictureDesc {
	bool HasPicture = false;
	bool ShowExistingFrame = false;
	OaU8 FrameToShowMapIdx = 0;
	OaVp9IvfFrame Frame = {};

	OaU32 FrameWidth = 0;
	OaU32 FrameHeight = 0;
	OaU32 RenderWidth = 0;
	OaU32 RenderHeight = 0;

	StdVideoDecodeVP9PictureInfo StdPictureInfo = {};
	StdVideoVP9ColorConfig ColorConfig = {};
	StdVideoVP9LoopFilter LoopFilter = {};
	StdVideoVP9Segmentation Segmentation = {};

	OaU8 RefFrameIdx[STD_VIDEO_VP9_REFS_PER_FRAME] = {};
	bool FrameIsIntra = false;
	OaU8 ChromaFormat = 1;

	OaU32 UncompressedHeaderOffset = 0;
	OaU32 CompressedHeaderOffset = 0;
	OaU32 TilesOffset = 0;
	OaU32 CompressedHeaderSize = 0;
	OaU32 NumTiles = 0;
};

class OaVcpVp9 : public OaVideoCodecParser {
public:
	OaVcpVp9() = default;
	~OaVcpVp9() override = default;

	OaStatus ParseSps(const OaSpan<const OaU8>& InFrame) override;
	OaStatus ParsePps(const OaSpan<const OaU8>& InFrame) override;
	void ClearParameterSets() override;

	OaStatus ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaVp9PictureDesc& OutDesc);

private:
	OaI8 LoopFilterRefDeltas_[STD_VIDEO_VP9_MAX_REF_FRAMES] = {1, 0, -1, -1};
	OaI8 LoopFilterModeDeltas_[STD_VIDEO_VP9_LOOP_FILTER_ADJUSTMENTS] = {};

	OaI32 LastFrameWidth_ = 0;
	OaI32 LastFrameHeight_ = 0;
	bool LastShowFrame_ = false;

	OaU32 BufferWidth_[STD_VIDEO_VP9_NUM_REF_FRAMES] = {};
	OaU32 BufferHeight_[STD_VIDEO_VP9_NUM_REF_FRAMES] = {};
	StdVideoVP9ColorConfig ColorConfig_ = {};
	bool HasColorConfig_ = false;
};
