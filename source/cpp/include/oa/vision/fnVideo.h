// oa::FnVideo — stateless frame views and NAL utilities.
//
// Stateful decoder/encoder commands and lifecycle boundaries live on their
// session classes; do not add mirrored session commands to this namespace.
//
// Schema source of truth:
//   tools/gen/fn/schema/vision/visionFnVideoColor.toml
//   tools/gen/fn/schema/vision/visionFnVideoNal.toml
//
// The schemas own operation metadata. Runtime-specific declarations and
// bodies remain handwritten and use the non-owning frame contract or NAL
// parser without mirroring decoder/encoder session commands.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/vision/videoDecoder.h>

namespace oa {

class Texture;

// Lightweight NAL unit descriptor — payload references back into the
// caller's bitstream buffer (no copy). For Annex-B framing the payload
// begins at the NAL header byte (skip the 00 00 00 01 / 00 00 01 prefix).
struct NalUnit {
	oa::U8 type      = 0;   // H.264 nal_unit_type (low 5 bits)
	oa::U8 refIdc    = 0;   // H.264 nal_ref_idc   (bits 5–6 of header byte)
	oa::Span<const oa::U8> payload;
};

} // namespace oa

namespace oa {

namespace FnVideo {
	// Wrap a buffer- or image-backed render target in the common video-frame
	// contract. The texture remains producer-owned; image readiness can be
	// supplied without a host wait.
	[[nodiscard]] oa::Result<VideoFrame> fromTexture(
		const oa::Texture& inTexture,
		oa::U64 inPts = 0ULL,
		oa::Event inReady = {}
	);

	// ──────────────────────────────────────────────────────────────────────
	// VideoNal — CPU-only, no engine needed.
	// ──────────────────────────────────────────────────────────────────────

	// split an Annex-B byte stream into NAL units (skip 00 00 00 01 /
	// 00 00 01 start codes). Payloads alias back into inBytes; do not
	// outlive the input buffer.
	[[nodiscard]] oa::Vector<NalUnit> parseNalAnnexB(const oa::Span<const oa::U8>& inBytes);

	// Concatenate NAL payloads with 4-byte start codes (00 00 00 01).
	// Caller-owned output buffer.
	[[nodiscard]] oa::Vector<oa::U8> emitNalAnnexB(const oa::Span<const NalUnit>& inUnits);

	// Return the raw bytes (including NAL header) of the first SPS unit
	// found in inNalBytes; empty if none.
	[[nodiscard]] oa::Vector<oa::U8> extractSps(const oa::Span<const oa::U8>& inNalBytes);

	// Same for the first PPS unit.
	[[nodiscard]] oa::Vector<oa::U8> extractPps(const oa::Span<const oa::U8>& inNalBytes);

	// HEVC parameter-set extraction. H.265 uses a two-byte NAL header and
	// different type encoding, so these are deliberately codec-explicit.
	[[nodiscard]] oa::Vector<oa::U8> extractVpsH265(const oa::Span<const oa::U8>& inNalBytes);
	[[nodiscard]] oa::Vector<oa::U8> extractSpsH265(const oa::Span<const oa::U8>& inNalBytes);
	[[nodiscard]] oa::Vector<oa::U8> extractPpsH265(const oa::Span<const oa::U8>& inNalBytes);

} // namespace FnVideo

} // namespace oa
