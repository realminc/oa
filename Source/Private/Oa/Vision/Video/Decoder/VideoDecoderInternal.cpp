#include "VideoDecoderInternal.h"

OaResult<OaVec<OaU8>> OaVideoDecoderInternal::ReadbackLuma(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame)
{
	return InDecoder.ReadbackLuma(InFrame);
}

OaResult<OaVec<OaU8>> OaVideoDecoderInternal::ReadbackNv12(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame)
{
	return InDecoder.ReadbackNv12(InFrame);
}

OaResult<OaVec<OaU8>> OaVideoDecoderInternal::ReadbackRgba(OaVideoDecoder& InDecoder, const OaVideoFrame& InFrame)
{
	return InDecoder.ReadbackRgba(InFrame);
}

OaResult<OaMatrix> OaVideoDecoderInternal::ConvertFrameToBf16(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InFrame,
	bool InNormalizeImageNet)
{
	return InDecoder.ConvertFrameToBf16(InFrame, InNormalizeImageNet);
}

OaResult<OaMatrix> OaVideoDecoderInternal::ConvertFrameToBf16Hardware(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InFrame,
	bool InNormalizeImageNet)
{
	return InDecoder.ConvertFrameToBf16Hardware(InFrame, InNormalizeImageNet);
}

OaResult<OaMatrix> OaVideoDecoderInternal::DecodeFrameToBf16(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	bool InNormalizeImageNet)
{
	return InDecoder.DecodeFrameToBf16(InBitstream, InNormalizeImageNet);
}

OaStatus OaVideoDecoderInternal::ConvertFrameToRgba(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutRgbFrame)
{
	return InDecoder.ConvertFrameToRgba(InNv12Frame, InOptions, OutRgbFrame);
}

OaStatus OaVideoDecoderInternal::ConvertNv12ToRgbInto(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& InOutRgbTarget)
{
	return InDecoder.ConvertNv12ToRgbInto(InNv12Frame, InOptions, InOutRgbTarget);
}

OaResult<OaVkImageDispatchTicket> OaVideoDecoderInternal::ConvertNv12ToRgbIntoAsync(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	const OaVideoFrame& InRgbTarget)
{
	return InDecoder.ConvertNv12ToRgbIntoAsync(InNv12Frame, InOptions, InRgbTarget);
}

OaResult<OaVideoFrame> OaVideoDecoderInternal::AllocateRgbaFrame(
	OaVideoDecoder& InDecoder,
	OaU32 InWidth,
	OaU32 InHeight)
{
	return InDecoder.AllocateRgbaFrame(InWidth, InHeight, 0);
}

OaStatus OaVideoDecoderInternal::RestoreDpbLayerToDecodeLayout(
	OaVideoDecoder& InDecoder,
	const OaVideoFrame& InFrame)
{
	return InDecoder.RestoreDpbLayerToDecodeLayout(InFrame);
}

OaU32 OaVideoDecoderInternal::GetBitstreamRingSize(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetBitstreamRingSize();
}

OaU64 OaVideoDecoderInternal::GetBitstreamBufferCapacity(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetBitstreamBufferCapacity();
}

OaU32 OaVideoDecoderInternal::GetCachedSpsCount(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetCachedSpsCount();
}

OaU32 OaVideoDecoderInternal::GetCachedPpsCount(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetCachedPpsCount();
}

OaU32 OaVideoDecoderInternal::GetCachedH265VpsCount(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetCachedH265VpsCount();
}

OaU32 OaVideoDecoderInternal::GetCachedH265SpsCount(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetCachedH265SpsCount();
}

OaU32 OaVideoDecoderInternal::GetCachedH265PpsCount(const OaVideoDecoder& InDecoder) noexcept
{
	return InDecoder.GetCachedH265PpsCount();
}

bool OaVideoDecoderInternal::HasHardwareYCbCrConversion(OaEngine& InRt)
{
	return OaVideoDecoder::HasHardwareYCbCrConversion(InRt);
}
