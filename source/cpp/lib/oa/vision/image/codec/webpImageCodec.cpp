#include "imageCodecInternal.h"

#if OA_HAS_WEBP
#include <webp/decode.h>
#include <webp/encode.h>
#endif

#include <cstring>

namespace oa::imageCodec {

bool supportsWebp() noexcept
{
#if OA_HAS_WEBP
	return true;
#else
	return false;
#endif
}

oa::Result<Pixels> decodeWebp(
	oa::Span<const oa::U8> inData,
	oa::ImageFormat inFormat)
{
#if OA_HAS_WEBP
	if (inFormat != oa::ImageFormat::Rgb
		and inFormat != oa::ImageFormat::Rgba) {
		return oa::Status::invalidArgument(
			"oa::FnImage::decodeMemory: WebP output supports Rgb or Rgba");
	}
	if (inData.empty()) {
		return oa::Status::invalidArgument(
			"oa::FnImage::decodeMemory: compressed data is empty");
	}

	int width = 0;
	int height = 0;
	if (WebPGetInfo(
		inData.data(),
		inData.size(),
		&width,
		&height) == 0) {
		return oa::Status::error(
			oa::StatusCode::FileCorrupt,
			"oa::FnImage::decodeMemory: invalid WebP bitstream");
	}

	oa::U8* decoded = inFormat == oa::ImageFormat::Rgba
		? WebPDecodeRGBA(inData.data(), inData.size(), &width, &height)
		: WebPDecodeRGB(inData.data(), inData.size(), &width, &height);
	if (decoded == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FileCorrupt,
			"oa::FnImage::decodeMemory: libwebp decode failed");
	}

	const oa::I32 channels =
		inFormat == oa::ImageFormat::Rgba ? 4 : 3;
	const oa::U64 byteCount =
		static_cast<oa::U64>(width)
		* static_cast<oa::U64>(height)
		* static_cast<oa::U64>(channels);
	Pixels result;
	result.data.resize(static_cast<oa::Usize>(byteCount));
	std::memcpy(
		result.data.data(),
		decoded,
		static_cast<oa::Usize>(byteCount));
	WebPFree(decoded);
	result.width = width;
	result.height = height;
	result.format = inFormat;
	return result;
#else
	(void)inData;
	(void)inFormat;
	return oa::Status::unimplemented(
		"oa::FnImage::decodeMemory: this build does not include libwebp");
#endif
}

oa::Result<oa::Vec<oa::U8>> encodeWebp(
	const Pixels& inPixels,
	oa::U32 inQuality)
{
#if OA_HAS_WEBP
	if (inQuality < 1U or inQuality > 100U) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: WebP quality must be in [1,100]");
	}
	const oa::I32 channels = inPixels.channels();
	if (channels != 3 and channels != 4) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: WebP input must be Rgb or Rgba");
	}
	const oa::U64 expectedBytes =
		static_cast<oa::U64>(inPixels.width)
		* static_cast<oa::U64>(inPixels.height)
		* static_cast<oa::U64>(channels);
	if (inPixels.width <= 0 or inPixels.height <= 0
		or inPixels.data.size() != expectedBytes) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: invalid WebP pixel metadata");
	}

	oa::U8* encodedData = nullptr;
	const size_t encodedSize = channels == 4
		? WebPEncodeRGBA(
			inPixels.data.data(),
			inPixels.width,
			inPixels.height,
			inPixels.width * channels,
			static_cast<float>(inQuality),
			&encodedData)
		: WebPEncodeRGB(
			inPixels.data.data(),
			inPixels.width,
			inPixels.height,
			inPixels.width * channels,
			static_cast<float>(inQuality),
			&encodedData);
	if (encodedSize == 0U or encodedData == nullptr) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"oa::FnImage::encode: libwebp encode failed");
	}

	oa::Vec<oa::U8> encoded;
	encoded.resize(encodedSize);
	std::memcpy(encoded.data(), encodedData, encodedSize);
	WebPFree(encodedData);
	return encoded;
#else
	(void)inPixels;
	(void)inQuality;
	return oa::Status::unimplemented(
		"oa::FnImage::encode: this build does not include libwebp");
#endif
}

} // namespace oa::imageCodec
