#include "ImageCodecInternal.h"

#if OA_HAS_WEBP
#include <webp/decode.h>
#include <webp/encode.h>
#endif

#include <cstring>

namespace OaImageCodecPrivate {

bool SupportsWebp() noexcept
{
#if OA_HAS_WEBP
	return true;
#else
	return false;
#endif
}

OaResult<Pixels> DecodeWebp(
	OaSpan<const OaU8> InData,
	OaImageFormat InFormat)
{
#if OA_HAS_WEBP
	if (InFormat != OaImageFormat::Rgb
		and InFormat != OaImageFormat::Rgba) {
		return OaStatus::InvalidArgument(
			"OaImageDecoder: WebP output supports Rgb or Rgba");
	}
	if (InData.Empty()) {
		return OaStatus::InvalidArgument(
			"OaImageDecoder: compressed data is empty");
	}

	int width = 0;
	int height = 0;
	if (WebPGetInfo(
		InData.Data(),
		InData.Size(),
		&width,
		&height) == 0) {
		return OaStatus::Error(
			OaStatusCode::FileCorrupt,
			"OaImageDecoder: invalid WebP bitstream");
	}

	OaU8* decoded = InFormat == OaImageFormat::Rgba
		? WebPDecodeRGBA(InData.Data(), InData.Size(), &width, &height)
		: WebPDecodeRGB(InData.Data(), InData.Size(), &width, &height);
	if (decoded == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FileCorrupt,
			"OaImageDecoder: libwebp decode failed");
	}

	const OaI32 channels =
		InFormat == OaImageFormat::Rgba ? 4 : 3;
	const OaU64 byteCount =
		static_cast<OaU64>(width)
		* static_cast<OaU64>(height)
		* static_cast<OaU64>(channels);
	Pixels result;
	result.Data.Resize(static_cast<OaUsize>(byteCount));
	std::memcpy(
		result.Data.Data(),
		decoded,
		static_cast<OaUsize>(byteCount));
	WebPFree(decoded);
	result.Width = width;
	result.Height = height;
	result.Format = InFormat;
	return result;
#else
	(void)InData;
	(void)InFormat;
	return OaStatus::Unimplemented(
		"OaImageDecoder: this build does not include libwebp");
#endif
}

OaResult<OaVec<OaU8>> EncodeWebp(
	const Pixels& InPixels,
	OaU32 InQuality)
{
#if OA_HAS_WEBP
	if (InQuality < 1U or InQuality > 100U) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: WebP quality must be in [1,100]");
	}
	const OaI32 channels = InPixels.Channels();
	if (channels != 3 and channels != 4) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: WebP input must be Rgb or Rgba");
	}
	const OaU64 expectedBytes =
		static_cast<OaU64>(InPixels.Width)
		* static_cast<OaU64>(InPixels.Height)
		* static_cast<OaU64>(channels);
	if (InPixels.Width <= 0 or InPixels.Height <= 0
		or InPixels.Data.Size() != expectedBytes) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: invalid WebP pixel metadata");
	}

	OaU8* encodedData = nullptr;
	const size_t encodedSize = channels == 4
		? WebPEncodeRGBA(
			InPixels.Data.Data(),
			InPixels.Width,
			InPixels.Height,
			InPixels.Width * channels,
			static_cast<float>(InQuality),
			&encodedData)
		: WebPEncodeRGB(
			InPixels.Data.Data(),
			InPixels.Width,
			InPixels.Height,
			InPixels.Width * channels,
			static_cast<float>(InQuality),
			&encodedData);
	if (encodedSize == 0U or encodedData == nullptr) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"OaImageEncoder: libwebp encode failed");
	}

	OaVec<OaU8> encoded;
	encoded.Resize(encodedSize);
	std::memcpy(encoded.Data(), encodedData, encodedSize);
	WebPFree(encodedData);
	return encoded;
#else
	(void)InPixels;
	(void)InQuality;
	return OaStatus::Unimplemented(
		"OaImageEncoder: this build does not include libwebp");
#endif
}

} // namespace OaImageCodecPrivate
