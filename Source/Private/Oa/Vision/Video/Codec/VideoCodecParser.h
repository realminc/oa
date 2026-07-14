// OA Vision — Video Codec Parser Base Class
// Pure parser interface — shared between decode and encode. No Vulkan.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std/Span.h>

// Forward declarations for codec-specific structs
struct StdVideoH264SequenceParameterSet;
struct StdVideoH264PictureParameterSet;
struct StdVideoH265VideoParameterSet;
struct StdVideoH265SequenceParameterSet;
struct StdVideoH265PictureParameterSet;

// Base class for video codec parsers.
// File prefix Vcp* = module layout; derived class prefix OaVcp* = per-codec parsers.
// Each codec (H.264, H.265, AV1, VP9) implements this interface.
class OaVideoCodecParser {
public:
	virtual ~OaVideoCodecParser() = default;

	// Parse parameter sets (NAL units for H.264/H.265, OBU for AV1)
	virtual OaStatus ParseSps(const OaSpan<const OaU8>& InNal) = 0;
	virtual OaStatus ParsePps(const OaSpan<const OaU8>& InNal) = 0;
	
	// VPS only exists for H.265 - default no-op for other codecs
	virtual OaStatus ParseVps(const OaSpan<const OaU8>& InNal) {
		return OaStatus::Ok();
	}

	// Codec-specific accessors - return nullptr if not applicable
	// H.264
	virtual const StdVideoH264SequenceParameterSet* GetH264Sps(OaU32 InSpsId) const { return nullptr; }
	virtual const StdVideoH264PictureParameterSet* GetH264Pps(OaU32 InPpsId) const { return nullptr; }
	
	// H.265
	virtual const StdVideoH265VideoParameterSet* GetH265Vps(OaU32 InVpsId) const { return nullptr; }
	virtual const StdVideoH265SequenceParameterSet* GetH265Sps(OaU32 InSpsId) const { return nullptr; }
	virtual const StdVideoH265PictureParameterSet* GetH265Pps(OaU32 InPpsId) const { return nullptr; }

	// Clear all cached parameter sets
	virtual void ClearParameterSets() = 0;
};

