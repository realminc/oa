// OA Vision — VP9 codec parser
// Extracts frame headers and converts to vulkan Video structures
// vulkan Video extension: VK_KHR_video_decode_vp9

#pragma once

#include "videoCodecParser.h"
#include <oa/core/std/vec.h>
#include <vk_video/vulkan_video_codec_vp9std.h>
#include <vk_video/vulkan_video_codec_vp9std_decode.h>

namespace oa {

struct Vp9IvfFrame {
	oa::Usize offset = 0;
	oa::Usize size = 0;
	oa::U64 timestamp = 0;
};

// Picture descriptor produced by parsing a VP9 access unit (IVF frame or raw frame).
struct Vp9PictureDesc {
	bool hasPicture = false;
	bool showExistingFrame = false;
	oa::U8 frameToShowMapIdx = 0;
	Vp9IvfFrame frame = {};

	oa::U32 frameWidth = 0;
	oa::U32 frameHeight = 0;
	oa::U32 renderWidth = 0;
	oa::U32 renderHeight = 0;

	StdVideoDecodeVP9PictureInfo stdPictureInfo = {};
	StdVideoVP9ColorConfig colorConfig = {};
	StdVideoVP9LoopFilter loopFilter = {};
	StdVideoVP9Segmentation segmentation = {};

	oa::U8 refFrameIdx[STD_VIDEO_VP9_REFS_PER_FRAME] = {};
	bool frameIsIntra = false;
	oa::U8 chromaFormat = 1;

	oa::U32 uncompressedHeaderOffset = 0;
	oa::U32 compressedHeaderOffset = 0;
	oa::U32 tilesOffset = 0;
	oa::U32 compressedHeaderSize = 0;
	oa::U32 numTiles = 0;
};

class VcpVp9 : public VideoCodecParser {
public:
	VcpVp9() = default;
	~VcpVp9() override = default;

	oa::Status parseSps(const oa::Span<const oa::U8>& inFrame) override;
	oa::Status parsePps(const oa::Span<const oa::U8>& inFrame) override;
	void clearParameterSets() override;

	oa::Status parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, Vp9PictureDesc& outDesc);

private:
	oa::I8 loopFilterRefDeltas_[STD_VIDEO_VP9_MAX_REF_FRAMES] = {1, 0, -1, -1};
	oa::I8 loopFilterModeDeltas_[STD_VIDEO_VP9_LOOP_FILTER_ADJUSTMENTS] = {};

	oa::I32 lastFrameWidth_ = 0;
	oa::I32 lastFrameHeight_ = 0;
	bool lastShowFrame_ = false;

	oa::U32 bufferWidth_[STD_VIDEO_VP9_NUM_REF_FRAMES] = {};
	oa::U32 bufferHeight_[STD_VIDEO_VP9_NUM_REF_FRAMES] = {};
	StdVideoVP9ColorConfig colorConfig_ = {};
	bool hasColorConfig_ = false;
};

} // namespace oa
