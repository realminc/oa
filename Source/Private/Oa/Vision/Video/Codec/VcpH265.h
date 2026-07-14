// OA Vision — H.265 Codec Parser
// Extracts VPS/SPS/PPS from NAL units and converts to Vulkan Video structures

#pragma once

#include "VideoCodecParser.h"
#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Core/Std/HashMap.h>
#include <Oa/Core/Std/Vec.h>
#include <vk_video/vulkan_video_codec_h265std.h>
#include <vk_video/vulkan_video_codec_h265std_decode.h>
#include "NalParser.h"

// Picture descriptor produced by parsing an H.265 access unit.
// Contains everything the decoder needs to record Vulkan commands.
struct OaH265PictureDesc {
	bool HasPicture = false;
	OaH265SliceHeader SliceHeader;
	OaH265SpsData Sps;
	OaH265PpsData Pps;
	OaVec<OaU32> SliceOffsets;  // NAL header byte offsets in the bitstream
	bool IsReference = false;
	// Parameter sets found in this access unit (for upload before decode)
	OaVec<OaH265VpsData> VpsInAu;
	OaVec<OaH265SpsData> SpsInAu;
	OaVec<OaH265PpsData> PpsInAu;
};

// H.265 codec parser implementation
class OaVcpH265 : public OaVideoCodecParser {
public:
	OaVcpH265() = default;
	~OaVcpH265() override = default;

	// Parse H.265 parameter sets
	OaStatus ParseSps(const OaSpan<const OaU8>& InNal) override;
	OaStatus ParsePps(const OaSpan<const OaU8>& InNal) override;
	OaStatus ParseVps(const OaSpan<const OaU8>& InNal) override;

	// Get cached parameter sets
	const StdVideoH265VideoParameterSet* GetH265Vps(OaU32 InVpsId) const override;
	const StdVideoH265SequenceParameterSet* GetH265Sps(OaU32 InSpsId) const override;
	const StdVideoH265PictureParameterSet* GetH265Pps(OaU32 InPpsId) const override;

	// Clear all cached parameter sets
	void ClearParameterSets() override;

	// Parse a complete access unit (Annex-B bitstream).
	// Caches VPS/SPS/PPS internally. On success OutDesc.HasPicture tells
	// whether slice NALs were found (parameter-set-only AUs are Ok).
	OaStatus ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaH265PictureDesc& OutDesc);

	// Get OA-native VPS/SPS/PPS data (for VideoDecoder compatibility)
	const OaH265VpsData* GetOaVps(OaU32 InVpsId) const;
	const OaH265SpsData* GetOaSps(OaU32 InSpsId) const;
	const OaH265PpsData* GetOaPps(OaU32 InPpsId) const;

	// Enumerate cached IDs so the decoder can upload to Vulkan session
	OaVec<OaU32> GetCachedVpsIds() const;
	OaVec<OaU32> GetCachedSpsIds() const;
	OaVec<OaU32> GetCachedPpsIds() const;

	// Convert OA structures to Vulkan Video structures
	static StdVideoH265LevelIdc ToStdH265Level(OaU32 InLevelIdc);
	static StdVideoH265ProfileTierLevel ToStdH265ProfileTierLevel(const OaH265VpsData& InVps);
	static StdVideoH265DecPicBufMgr ToStdH265DecPicBufMgr(const OaH265SpsData& InSps);
	static StdVideoH265VideoParameterSet ToStdH265Vps(const OaH265VpsData& InVps, const StdVideoH265ProfileTierLevel& InPtl);
	static StdVideoH265SequenceParameterSet ToStdH265Sps(const OaH265SpsData& InSps, const StdVideoH265ProfileTierLevel& InPtl, const StdVideoH265DecPicBufMgr& InDpb);
	static StdVideoH265PictureParameterSet ToStdH265Pps(const OaH265PpsData& InPps, const OaH265SpsData& InSps);

private:
	// Cache for OA-native structures
	OaStdHashMap<OaU32, OaH265VpsData> OaVpsCache_;
	OaStdHashMap<OaU32, OaH265SpsData> OaSpsCache_;
	OaStdHashMap<OaU32, OaH265PpsData> OaPpsCache_;

	// Cache for Vulkan Video structures
	OaStdHashMap<OaU32, StdVideoH265VideoParameterSet> StdVpsCache_;
	OaStdHashMap<OaU32, StdVideoH265SequenceParameterSet> StdSpsCache_;
	OaStdHashMap<OaU32, StdVideoH265PictureParameterSet> StdPpsCache_;
	
	// Storage for profile tier level (referenced by VPS/SPS)
	OaStdHashMap<OaU32, StdVideoH265ProfileTierLevel> ProfileTierLevelStorage_;
	OaStdHashMap<OaU32, StdVideoH265DecPicBufMgr> DecPicBufMgrStorage_;
};

