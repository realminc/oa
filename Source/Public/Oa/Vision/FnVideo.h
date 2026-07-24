// OaFnVideo — Python-simple video API (decode + encode + NAL utilities).
//
// Sister surface to OaFnImage. Same shape: free functions in a namespace,
// engine taken as a parameter where GPU work is needed, stateful session
// objects passed in explicitly.
//
// Schema source of truth:
//   Tools/FnAutogen/Schema/Vision/VisionFnVideoCodec.toml
//   Tools/FnAutogen/Schema/Vision/VisionFnVideoColor.toml
//   Tools/FnAutogen/Schema/Vision/VisionFnVideoNal.toml
//   Tools/FnAutogen/Schema/Vision/VisionFnVideoPreprocess.toml
//
// Today these are handwritten (oafnautogen.py does not yet accept ops
// without a kernel_forward — see OaVideo.md §17.2). When the autogen grows
// `kind = "session_record"` and `kind = "cpu_util"`, the declarations here
// can fall out from the schemas; the bodies will remain hand-rolled because
// they delegate into OaVideoDecoder / OaVideoEncoder / OaNalParser.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/ImageDispatch.h>
#include <Oa/Vision/FnImage.h>         // OaPixelFormat, OaNormalizationParams
#include <Oa/Vision/VideoDecoder.h>    // OaVideoDecoder, OaVideoFrame, OaVideoConversionOptions
#include <Oa/Vision/VideoEncoder.h>    // OaVideoEncoder, OaEncodedFrame

class OaContext;
struct OaTexture;

// Lightweight NAL unit descriptor — payload references back into the
// caller's bitstream buffer (no copy). For Annex-B framing the payload
// begins at the NAL header byte (skip the 00 00 00 01 / 00 00 01 prefix).
struct OaNalUnit {
	OaU8 Type      = 0;   // H.264 nal_unit_type (low 5 bits)
	OaU8 RefIdc    = 0;   // H.264 nal_ref_idc   (bits 5–6 of header byte)
	OaSpan<const OaU8> Payload;
};


namespace OaFnVideo {
	// Wrap a buffer- or image-backed render target in the common video-frame
	// contract. The texture remains producer-owned; image readiness can be
	// supplied without a host wait.
	[[nodiscard]] OaResult<OaVideoFrame> FromTexture(
		const OaTexture& InTexture,
		OaU64 InPts = 0ULL,
		OaEvent InReady = {}
	);

	// ──────────────────────────────────────────────────────────────────────
	// VideoCodec — stateful, delegate into the supplied session.
	// ──────────────────────────────────────────────────────────────────────

	// Decode one compressed access unit. The default overload coordinates with
	// OaContext::GetDefault(); pass a context explicitly when the decoder is
	// attached to a non-default engine.
	[[nodiscard]] OaResult<OaVideoFrame> Decode(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		OaU64 InPts = 0ULL
	);

	[[nodiscard]] OaResult<OaVideoFrame> Decode(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		const OaVideoConversionOptions& InOptions,
		OaU64 InPts = 0ULL
	);

	[[nodiscard]] OaResult<OaVideoFrame> Decode(
		OaContext& InContext,
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		OaU64 InPts = 0ULL
	);

	[[nodiscard]] OaResult<OaVideoFrame> Decode(
		OaContext& InContext,
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		const OaVideoConversionOptions& InOptions,
		OaU64 InPts = 0ULL
	);

	// Status/output overloads for loops that reuse an existing frame variable.
	[[nodiscard]] OaStatus Decode(
		OaContext& InContext,
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	// Convert a buffer-backed RGBA texture and encode one frame. The context is
	// an explicit dependency boundary for pending compute producers; media
	// session behavior remains owned by OaFnVideo/OaVideoEncoder, not Runtime.
	[[nodiscard]] OaStatus Encode(
		OaContext& InContext,
		OaVideoEncoder& InSession,
		const OaTexture& InRgba,
		OaEncodedFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	[[nodiscard]] OaStatus Encode(
		OaVideoEncoder& InSession,
		const OaTexture& InRgba,
		OaEncodedFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	// Compatibility shims. New code should use Decode().
	[[nodiscard]] OaStatus DecodeFrame(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		OaVideoFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	[[nodiscard]] OaStatus DecodeFrame(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	// Encode a single NV12 VkImage into an Annex-B bitstream packet.
	[[nodiscard]] OaStatus EncodeFrame(
		OaVideoEncoder& InSession,
		VkImage InImage,
		OaEncodedFrame& OutFrame,
		OaU64 InPts = 0ULL
	);

	// Flush pending decoder state (output remaining DPB references).
	[[nodiscard]] OaStatus FlushSession(OaVideoDecoder& InSession);

	// Flush pending encoder state (drain reordered frames). When the encoder
	// is IDR/P-only this returns an empty vector successfully.
	[[nodiscard]] OaStatus FlushSession(OaVideoEncoder& InSession, OaVec<OaEncodedFrame>& OutFrames);

	// ──────────────────────────────────────────────────────────────────────
	// VideoNal — CPU-only, no engine needed.
	// ──────────────────────────────────────────────────────────────────────

	// Split an Annex-B byte stream into NAL units (skip 00 00 00 01 /
	// 00 00 01 start codes). Payloads alias back into InBytes; do not
	// outlive the input buffer.
	[[nodiscard]] OaVec<OaNalUnit> ParseNalAnnexB(const OaSpan<const OaU8>& InBytes);

	// Concatenate NAL payloads with 4-byte start codes (00 00 00 01).
	// Caller-owned output buffer.
	[[nodiscard]] OaVec<OaU8> EmitNalAnnexB(const OaSpan<const OaNalUnit>& InUnits);

	// Return the raw bytes (including NAL header) of the first SPS unit
	// found in InNalBytes; empty if none.
	[[nodiscard]] OaVec<OaU8> ExtractSps(const OaSpan<const OaU8>& InNalBytes);

	// Same for the first PPS unit.
	[[nodiscard]] OaVec<OaU8> ExtractPps(const OaSpan<const OaU8>& InNalBytes);

	// HEVC parameter-set extraction. H.265 uses a two-byte NAL header and
	// different type encoding, so these are deliberately codec-explicit.
	[[nodiscard]] OaVec<OaU8> ExtractVpsH265(const OaSpan<const OaU8>& InNalBytes);
	[[nodiscard]] OaVec<OaU8> ExtractSpsH265(const OaSpan<const OaU8>& InNalBytes);
	[[nodiscard]] OaVec<OaU8> ExtractPpsH265(const OaSpan<const OaU8>& InNalBytes);

	// ──────────────────────────────────────────────────────────────────────
	// VideoColor — re-namespaced color converters used by encoder/decoder.
	// These delegate into the same compute kernels as OaFnImage::CvtNv12* /
	// OaVideoEncoder::UploadInputRgba.
	// ──────────────────────────────────────────────────────────────────────

	// Convert packed RGBA8 → NV12 in-place into the encoder's input image.
	// (Encoders own their NV12 picture; we don't manufacture a new one.)
	[[nodiscard]] OaStatus CvtRgbaToNv12(
		OaVideoEncoder& InSession,
		const OaVkBuffer& InRgba,
		OaU32 InVisibleWidth,
		OaU32 InVisibleHeight,
		OaYCbCrModel InColorSpace = OaYCbCrModel::BT709,
		bool InFullRange = false
	);

	// Convert a decoded NV12 frame → RGBA. The output frame is allocated
	// from the decoder's RGB pool (same lifetime as InSession).
	[[nodiscard]] OaStatus CvtNv12ToRgb(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& OutRgbFrame
	);

	[[nodiscard]] OaResult<OaVideoFrame> Convert(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InFrame,
		const OaVideoConversionOptions& InOptions = {}
	);

	// Copy decoded planes to CPU-visible memory (tests/diagnostics).
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackLuma(OaVideoDecoder& InSession,	const OaVideoFrame& InFrame);
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackNv12(OaVideoDecoder& InSession,	const OaVideoFrame& InFrame);
	[[nodiscard]] OaResult<OaVec<OaU8>> ReadbackRgba(OaVideoDecoder& InSession,	const OaVideoFrame& InFrame);

	// Allocate a caller-owned RGBA target from the decoder pool (e.g. reorder buffers).
	[[nodiscard]] OaResult<OaVideoFrame> AllocateRgbaFrame(
		OaVideoDecoder& InSession,
		OaU32 InWidth,
		OaU32 InHeight
	);

	// Write NV12→RGBA into a caller-owned target without reusing the decoder's cached frame.
	[[nodiscard]] OaStatus CvtNv12ToRgbInto(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& InOutRgbTarget
	);
	[[nodiscard]] OaResult<OaVkImageDispatchTicket> CvtNv12ToRgbIntoAsync(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		const OaVideoFrame& InRgbTarget
	);

	// Decode-to-ML bridge: [1, 3, H, W] BF16 RGB from a decoded frame or access unit.
	[[nodiscard]] OaResult<OaMatrix> FrameToBf16(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InFrame,
		bool InNormalizeImageNet = true
	);
	[[nodiscard]] OaResult<OaMatrix> FrameToBf16Hardware(
		OaVideoDecoder& InSession,
		const OaVideoFrame& InFrame,
		bool InNormalizeImageNet = true
	);
	[[nodiscard]] OaResult<OaMatrix> DecodeToBf16(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		bool InNormalizeImageNet = true
	);

	// VideoPreprocess — fused decode → resize → normalize pipeline.

	// Decode one access unit and return a [1, 3, H, W] BF16 RGB tensor
	// normalized with ImageNet stats (or caller-supplied mean/std).
	[[nodiscard]] OaResult<OaMatrix> DecodeResizeNormalize(
		OaVideoDecoder& InSession,
		const OaSpan<const OaU8>& InAccessUnit,
		OaU32 InWidth                       = 224U,
		OaU32 InHeight                      = 224U,
		const OaNormalizationParams& InNorm = OaNormalizationParams{
			.Mean = { 0.485F, 0.456F, 0.406F },
			.Std  = { 0.229F, 0.224F, 0.225F },
		}
	);
}
