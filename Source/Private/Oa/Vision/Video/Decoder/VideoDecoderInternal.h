// Internal accessors for OaVideoDecoder color/readback/test paths.
// Public callers should prefer OaFnVideo:: where a verb exists.

#pragma once

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Runtime/ImageDispatch.h>

struct OaVideoDecoderInternal {
	static OaResult<OaVec<OaU8>> ReadbackLuma(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame);
	static OaResult<OaVec<OaU8>> ReadbackNv12(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame);
	static OaResult<OaVec<OaU8>> ReadbackRgba(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame);

	static OaResult<OaMatrix> ConvertFrameToBf16(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InFrame,
		bool InNormalizeImageNet = true);
	static OaResult<OaMatrix> ConvertFrameToBf16Hardware(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InFrame,
		bool InNormalizeImageNet = true);
	static OaResult<OaMatrix> DecodeFrameToBf16(
		OaVideoDecoder& InDecoder,
		const OaSpan<const OaU8>& InBitstream,
		bool InNormalizeImageNet = true);

	static OaStatus ConvertFrameToRgba(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& OutRgbFrame);
	static OaStatus ConvertNv12ToRgbInto(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		OaVideoFrame& InOutRgbTarget);
	static OaResult<OaVkImageDispatchTicket> ConvertNv12ToRgbIntoAsync(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InNv12Frame,
		const OaVideoConversionOptions& InOptions,
		const OaVideoFrame& InRgbTarget);
	static OaResult<OaVideoFrame> AllocateRgbaFrame(
		OaVideoDecoder& InDecoder,
		OaU32 InWidth,
		OaU32 InHeight);
	static OaStatus RestoreDpbLayerToDecodeLayout(
		OaVideoDecoder& InDecoder,
		const OaVideoFrame& InFrame);

	static OaU32 GetBitstreamRingSize(const OaVideoDecoder& InDecoder) noexcept;
	static OaU64 GetBitstreamBufferCapacity(const OaVideoDecoder& InDecoder) noexcept;
	static OaU32 GetCachedSpsCount(const OaVideoDecoder& InDecoder) noexcept;
	static OaU32 GetCachedPpsCount(const OaVideoDecoder& InDecoder) noexcept;
	static OaU32 GetCachedH265VpsCount(const OaVideoDecoder& InDecoder) noexcept;
	static OaU32 GetCachedH265SpsCount(const OaVideoDecoder& InDecoder) noexcept;
	static OaU32 GetCachedH265PpsCount(const OaVideoDecoder& InDecoder) noexcept;
	static bool HasHardwareYCbCrConversion(OaEngine& InRt);
};
