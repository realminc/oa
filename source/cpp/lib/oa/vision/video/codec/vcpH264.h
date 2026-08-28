// OA Vision — H.264 codec parser
// Extracts SPS/PPS from NAL units and converts to vulkan Video structures

#pragma once

#include "videoCodecParser.h"
#include <oa/vision/videoDecoder.h>
#include <oa/core/std/hashMap.h>
#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>
#include "nalParser.h"

namespace oa {

// Picture descriptor produced by parsing an H.264 access unit.
struct H264PictureDesc {
	bool hasPicture = false;
	H264SliceHeader sliceHeader;
	H264SpsData sps;
	H264PpsData pps;
	oa::U32 sliceStartCodeOffset = 0;
	oa::U32 sliceStartCodeSize = 0;
	oa::U32 sliceNalSize = 0;
};

// H.264 codec parser implementation
class VcpH264 : public VideoCodecParser {
public:
	VcpH264() = default;
	~VcpH264() override = default;

	// parse H.264 parameter sets
	oa::Status parseSps(const oa::Span<const oa::U8>& inNal) override;
	oa::Status parsePps(const oa::Span<const oa::U8>& inNal) override;

	// get cached parameter sets
	const StdVideoH264SequenceParameterSet* getH264Sps(oa::U32 inSpsId) const override;
	const StdVideoH264PictureParameterSet* getH264Pps(oa::U32 inPpsId) const override;

	// clear all cached parameter sets
	void clearParameterSets() override;

	// parse a complete access unit (Annex-B bitstream).
	oa::Status parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, H264PictureDesc& outDesc);

	// get OA-native SPS/PPS data (for VideoDecoder compatibility)
	const H264SpsData* getSpsData(oa::U32 inSpsId) const;
	const H264PpsData* getPpsData(oa::U32 inPpsId) const;

	// Enumerate cached IDs so the decoder can upload to vulkan session
	oa::Vector<oa::U32> getCachedSpsIds() const;
	oa::Vector<oa::U32> getCachedPpsIds() const;

	// convert OA structures to vulkan Video structures
	static StdVideoH264LevelIdc toStdH264Level(oa::U32 inLevelIdc);
	static StdVideoH264SequenceParameterSet toStdH264Sps(const H264SpsData& inSps);
	static StdVideoH264PictureParameterSet toStdH264Pps(const H264PpsData& inPps);

private:
	// cache for OA-native structures (used by VideoDecoder)
	oa::HashMap<oa::U32, H264SpsData> oaSpsCache_;
	oa::HashMap<oa::U32, H264PpsData> oaPpsCache_;

	// cache for vulkan Video structures
	oa::HashMap<oa::U32, StdVideoH264SequenceParameterSet> stdSpsCache_;
	oa::HashMap<oa::U32, StdVideoH264PictureParameterSet> stdPpsCache_;
	
	// storage for offset arrays (referenced by StdVideoH264SequenceParameterSet)
	oa::HashMap<oa::U32, oa::Array<oa::I32, 256>> offsetForRefFrameStorage_;
};

} // namespace oa
