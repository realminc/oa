// Internal accessors for private oa::VideoDecoder diagnostics and test paths.

#pragma once

#include <oa/vision/videoDecoder.h>

namespace oa {

struct VideoDecoderInternal {
	static oa::Status decodeFrame(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		oa::VideoFrame& outFrame);
	static oa::Status decodeFrameWithConversion(
		oa::VideoDecoder& inDecoder,
		const oa::Span<const oa::U8>& inBitstream,
		const oa::VideoConversionOptions& inOptions,
		oa::VideoFrame& outFrame);
	static oa::Result<oa::Matrix> convertFrameToBf16Hardware(
		oa::VideoDecoder& inDecoder,
		const oa::VideoFrame& inFrame,
		bool inNormalizeImageNet = true);
	static oa::Result<oa::Event> convertIntoAsyncProfiled(
		oa::VideoDecoder& inDecoder,
		const oa::VideoFrame& inFrame,
		const oa::VideoConversionOptions& inOptions,
		const oa::VideoFrame& inRgbTarget,
		oa::Timer& inTimer);
	static oa::Status restoreDpbLayerToDecodeLayout(
		oa::VideoDecoder& inDecoder,
		const oa::VideoFrame& inFrame);

	static oa::U32 getBitstreamRingSize(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U64 getBitstreamBufferCapacity(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U32 getCachedSpsCount(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U32 getCachedPpsCount(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U32 getCachedH265VpsCount(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U32 getCachedH265SpsCount(const oa::VideoDecoder& inDecoder) noexcept;
	static oa::U32 getCachedH265PpsCount(const oa::VideoDecoder& inDecoder) noexcept;
	static bool hasHardwareYCbCrConversion(oa::Engine& inRt);
	static oa::U64 getHardwareYcbcrDispatchCount(
		const oa::VideoDecoder& inDecoder) noexcept;
};

} // namespace oa
