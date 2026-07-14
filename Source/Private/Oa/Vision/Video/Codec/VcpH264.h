// OA Vision — H.264 Codec Parser
// Extracts SPS/PPS from NAL units and converts to Vulkan Video structures

#pragma once

#include "VideoCodecParser.h"
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Core/Std/HashMap.h>
#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>
#include "NalParser.h"

// Picture descriptor produced by parsing an H.264 access unit.
struct OaH264PictureDesc {
	bool HasPicture = false;
	OaSliceHeader SliceHeader;
	OaH264SpsData Sps;
	OaH264PpsData Pps;
	OaU32 SliceStartCodeOffset = 0;
	OaU32 SliceStartCodeSize = 0;
	OaU32 SliceNalSize = 0;
};

// H.264 codec parser implementation
class OaVcpH264 : public OaVideoCodecParser {
public:
	OaVcpH264() = default;
	~OaVcpH264() override = default;

	// Parse H.264 parameter sets
	OaStatus ParseSps(const OaSpan<const OaU8>& InNal) override;
	OaStatus ParsePps(const OaSpan<const OaU8>& InNal) override;

	// Get cached parameter sets
	const StdVideoH264SequenceParameterSet* GetH264Sps(OaU32 InSpsId) const override;
	const StdVideoH264PictureParameterSet* GetH264Pps(OaU32 InPpsId) const override;

	// Clear all cached parameter sets
	void ClearParameterSets() override;

	// Parse a complete access unit (Annex-B bitstream).
	OaStatus ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaH264PictureDesc& OutDesc);

	// Get OA-native SPS/PPS data (for VideoDecoder compatibility)
	const OaH264SpsData* GetOaSps(OaU32 InSpsId) const;
	const OaH264PpsData* GetOaPps(OaU32 InPpsId) const;

	// Enumerate cached IDs so the decoder can upload to Vulkan session
	OaVec<OaU32> GetCachedSpsIds() const;
	OaVec<OaU32> GetCachedPpsIds() const;

	// Convert OA structures to Vulkan Video structures
	static StdVideoH264LevelIdc ToStdH264Level(OaU32 InLevelIdc);
	static StdVideoH264SequenceParameterSet ToStdH264Sps(const OaH264SpsData& InSps);
	static StdVideoH264PictureParameterSet ToStdH264Pps(const OaH264PpsData& InPps);

private:
	// Cache for OA-native structures (used by VideoDecoder)
	OaStdHashMap<OaU32, OaH264SpsData> OaSpsCache_;
	OaStdHashMap<OaU32, OaH264PpsData> OaPpsCache_;

	// Cache for Vulkan Video structures
	OaStdHashMap<OaU32, StdVideoH264SequenceParameterSet> StdSpsCache_;
	OaStdHashMap<OaU32, StdVideoH264PictureParameterSet> StdPpsCache_;
	
	// Storage for offset arrays (referenced by StdVideoH264SequenceParameterSet)
	OaStdHashMap<OaU32, OaArray<OaI32, 256>> OffsetForRefFrameStorage_;
};

