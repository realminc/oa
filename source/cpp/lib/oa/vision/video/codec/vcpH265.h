// OA Vision — H.265 codec parser
// Extracts VPS/SPS/PPS from NAL units and converts to vulkan Video structures

#pragma once

#include "videoCodecParser.h"
#include <oa/vision/videoDecoder.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/std/vec.h>
#include <vk_video/vulkan_video_codec_h265std.h>
#include <vk_video/vulkan_video_codec_h265std_decode.h>
#include "nalParser.h"

namespace oa {

// Picture descriptor produced by parsing an H.265 access unit.
// contains everything the decoder needs to record vulkan commands.
struct H265PictureDesc {
	bool hasPicture = false;
	H265SliceHeader sliceHeader;
	H265SpsData sps;
	H265PpsData pps;
	oa::Vec<oa::U32> sliceOffsets;  // NAL header byte offsets in the bitstream
	bool isReference = false;
	// Parameter sets found in this access unit (for upload before decode)
	oa::Vec<H265VpsData> vpsInAu;
	oa::Vec<H265SpsData> spsInAu;
	oa::Vec<H265PpsData> ppsInAu;
};

// H.265 codec parser implementation
class VcpH265 : public VideoCodecParser {
public:
	VcpH265() = default;
	~VcpH265() override = default;

	// parse H.265 parameter sets
	oa::Status parseSps(const oa::Span<const oa::U8>& inNal) override;
	oa::Status parsePps(const oa::Span<const oa::U8>& inNal) override;
	oa::Status parseVps(const oa::Span<const oa::U8>& inNal) override;

	// get cached parameter sets
	const StdVideoH265VideoParameterSet* getH265Vps(oa::U32 inVpsId) const override;
	const StdVideoH265SequenceParameterSet* getH265Sps(oa::U32 inSpsId) const override;
	const StdVideoH265PictureParameterSet* getH265Pps(oa::U32 inPpsId) const override;

	// clear all cached parameter sets
	void clearParameterSets() override;

	// parse a complete access unit (Annex-B bitstream).
	// Caches VPS/SPS/PPS internally. On success outDesc.hasPicture tells
	// whether slice NALs were found (parameter-set-only AUs are ok).
	oa::Status parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, H265PictureDesc& outDesc);

	// get OA-native VPS/SPS/PPS data (for VideoDecoder compatibility)
	const H265VpsData* getVpsData(oa::U32 inVpsId) const;
	const H265SpsData* getSpsData(oa::U32 inSpsId) const;
	const H265PpsData* getPpsData(oa::U32 inPpsId) const;

	// Enumerate cached IDs so the decoder can upload to vulkan session
	oa::Vec<oa::U32> getCachedVpsIds() const;
	oa::Vec<oa::U32> getCachedSpsIds() const;
	oa::Vec<oa::U32> getCachedPpsIds() const;

	// convert OA structures to vulkan Video structures
	static StdVideoH265LevelIdc toStdH265Level(oa::U32 inLevelIdc);
	static StdVideoH265ProfileTierLevel toStdH265ProfileTierLevel(const H265VpsData& inVps);
	static StdVideoH265DecPicBufMgr toStdH265DecPicBufMgr(const H265SpsData& inSps);
	static StdVideoH265VideoParameterSet toStdH265Vps(const H265VpsData& inVps, const StdVideoH265ProfileTierLevel& inPtl);
	static StdVideoH265SequenceParameterSet toStdH265Sps(const H265SpsData& inSps, const StdVideoH265ProfileTierLevel& inPtl, const StdVideoH265DecPicBufMgr& inDpb);
	static StdVideoH265PictureParameterSet toStdH265Pps(const H265PpsData& inPps, const H265SpsData& inSps);

private:
	// cache for OA-native structures
	oa::HashMap<oa::U32, H265VpsData> oaVpsCache_;
	oa::HashMap<oa::U32, H265SpsData> oaSpsCache_;
	oa::HashMap<oa::U32, H265PpsData> oaPpsCache_;

	// cache for vulkan Video structures
	oa::HashMap<oa::U32, StdVideoH265VideoParameterSet> stdVpsCache_;
	oa::HashMap<oa::U32, StdVideoH265SequenceParameterSet> stdSpsCache_;
	oa::HashMap<oa::U32, StdVideoH265PictureParameterSet> stdPpsCache_;
	
	// storage for profile tier level (referenced by VPS/SPS)
	oa::HashMap<oa::U32, StdVideoH265ProfileTierLevel> profileTierLevelStorage_;
	oa::HashMap<oa::U32, StdVideoH265DecPicBufMgr> decPicBufMgrStorage_;
};

} // namespace oa
