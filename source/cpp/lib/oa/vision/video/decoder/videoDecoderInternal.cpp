#include "videoDecoderInternal.h"
#include "videoDecoderImpl.h"
#include <oa/runtime/timer.h>

oa::Status oa::VideoDecoderInternal::decodeFrame(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	return inDecoder.decodeFrame(inBitstream, outFrame);
}

oa::Status oa::VideoDecoderInternal::decodeFrameWithConversion(
	oa::VideoDecoder& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outFrame)
{
	return inDecoder.decodeFrameWithConversion(inBitstream, inOptions, outFrame);
}

oa::Result<oa::Matrix> oa::VideoDecoderInternal::convertFrameToBf16Hardware(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame,
	bool inNormalizeImageNet)
{
	return inDecoder.convertFrameToBf16Hardware(inFrame, inNormalizeImageNet);
}

oa::Result<oa::Event> oa::VideoDecoderInternal::convertIntoAsyncProfiled(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame,
	const oa::VideoConversionOptions& inOptions,
	const oa::VideoFrame& inRgbTarget,
	oa::Timer& inTimer)
{
	return inDecoder.convertNv12ToRgbIntoAsync(
		inFrame, inOptions, inRgbTarget, &inTimer);
}

oa::Status oa::VideoDecoderInternal::restoreDpbLayerToDecodeLayout(
	oa::VideoDecoder& inDecoder,
	const oa::VideoFrame& inFrame)
{
	return inDecoder.restoreDpbLayerToDecodeLayout(inFrame);
}

oa::U32 oa::VideoDecoderInternal::getBitstreamRingSize(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getBitstreamRingSize();
}

oa::U64 oa::VideoDecoderInternal::getBitstreamBufferCapacity(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getBitstreamBufferCapacity();
}

oa::U32 oa::VideoDecoderInternal::getCachedSpsCount(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getCachedSpsCount();
}

oa::U32 oa::VideoDecoderInternal::getCachedPpsCount(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getCachedPpsCount();
}

oa::U32 oa::VideoDecoderInternal::getCachedH265VpsCount(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getCachedH265VpsCount();
}

oa::U32 oa::VideoDecoderInternal::getCachedH265SpsCount(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getCachedH265SpsCount();
}

oa::U32 oa::VideoDecoderInternal::getCachedH265PpsCount(const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getCachedH265PpsCount();
}

bool oa::VideoDecoderInternal::hasHardwareYCbCrConversion(oa::Engine& inRt)
{
	return oa::VideoDecoder::hasHardwareYCbCrConversion(inRt);
}

oa::U64 oa::VideoDecoderInternal::getHardwareYcbcrDispatchCount(
	const oa::VideoDecoder& inDecoder) noexcept
{
	return inDecoder.getHardwareYcbcrDispatchCount();
}
