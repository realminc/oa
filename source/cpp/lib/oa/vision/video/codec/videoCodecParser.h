// OA Vision — Video codec parser base class
// Pure parser interface — shared between decode and encode. No vulkan.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/std/span.h>

// forward declarations for codec-specific structs
struct StdVideoH264SequenceParameterSet;
struct StdVideoH264PictureParameterSet;
struct StdVideoH265VideoParameterSet;
struct StdVideoH265SequenceParameterSet;
struct StdVideoH265PictureParameterSet;

namespace oa {

// base class for video codec parsers.
// File prefix and derived-class prefix Vcp identify per-codec parsers.
// Each codec (H.264, H.265, AV1, VP9) implements this interface.
class VideoCodecParser {
public:
	virtual ~VideoCodecParser() = default;

	// parse parameter sets (NAL units for H.264/H.265, OBU for AV1)
	virtual oa::Status parseSps(const oa::Span<const oa::U8>& inNal) = 0;
	virtual oa::Status parsePps(const oa::Span<const oa::U8>& inNal) = 0;
	
	// VPS only exists for H.265 - default no-op for other codecs
	virtual oa::Status parseVps(const oa::Span<const oa::U8>& inNal) {
		return oa::Status::ok();
	}

	// codec-specific accessors - return nullptr if not applicable
	// H.264
	virtual const StdVideoH264SequenceParameterSet* getH264Sps(oa::U32 inSpsId) const { return nullptr; }
	virtual const StdVideoH264PictureParameterSet* getH264Pps(oa::U32 inPpsId) const { return nullptr; }
	
	// H.265
	virtual const StdVideoH265VideoParameterSet* getH265Vps(oa::U32 inVpsId) const { return nullptr; }
	virtual const StdVideoH265SequenceParameterSet* getH265Sps(oa::U32 inSpsId) const { return nullptr; }
	virtual const StdVideoH265PictureParameterSet* getH265Pps(oa::U32 inPpsId) const { return nullptr; }

	// clear all cached parameter sets
	virtual void clearParameterSets() = 0;
};

} // namespace oa
