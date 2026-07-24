#include "ImageCodecInternal.h"

#include "../../../../../ThirdParty/stb/stb_image.h"
#include "../../../../../ThirdParty/stb/stb_image_write.h"

#include <cstring>
#include <limits>

namespace OaImageCodecPrivate {

namespace {

OaI32 RequestedChannels(OaImageFormat InFormat)
{
	switch (InFormat) {
		case OaImageFormat::Gray: return 1;
		case OaImageFormat::Rgb: return 3;
		case OaImageFormat::Rgba: return 4;
		default: return 0;
	}
}

void AppendEncodedBytes(void* InContext, void* InData, int InSize)
{
	auto& output = *static_cast<OaVec<OaU8>*>(InContext);
	if (InData == nullptr or InSize <= 0) return;
	const OaUsize oldSize = output.Size();
	output.Resize(oldSize + static_cast<OaUsize>(InSize));
	std::memcpy(
		output.Data() + oldSize,
		InData,
		static_cast<OaUsize>(InSize));
}

} // namespace

bool SupportsStbDecode(OaImageCodec InCodec) noexcept
{
	return InCodec == OaImageCodec::Jpeg
		or InCodec == OaImageCodec::Png
		or InCodec == OaImageCodec::Bmp
		or InCodec == OaImageCodec::Tga;
}

bool SupportsStbEncode(OaImageCodec InCodec) noexcept
{
	return SupportsStbDecode(InCodec);
}

OaResult<Pixels> DecodeStb(
	OaSpan<const OaU8> InData,
	OaImageFormat InFormat)
{
	const OaI32 requestedChannels = RequestedChannels(InFormat);
	if (requestedChannels == 0) {
		return OaStatus::InvalidArgument(
			"OaImageDecoder supports Gray, Rgb, or Rgba output");
	}
	if (InData.Empty()) {
		return OaStatus::InvalidArgument(
			"OaImageDecoder: compressed data is empty");
	}
	if (InData.Size() > static_cast<OaUsize>(
		std::numeric_limits<int>::max())) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"OaImageDecoder: compressed data exceeds stb_image limits");
	}

	int width = 0;
	int height = 0;
	int sourceChannels = 0;
	stbi_uc* decoded = stbi_load_from_memory(
		InData.Data(),
		static_cast<int>(InData.Size()),
		&width,
		&height,
		&sourceChannels,
		requestedChannels);
	if (decoded == nullptr) {
		const char* reason = stbi_failure_reason();
		return OaStatus::Error(
			OaStatusCode::FileCorrupt,
			OaString("OaImageDecoder: stb_image decode failed: ")
				+ (reason != nullptr ? reason : "unknown error"));
	}

	const OaU64 byteCount =
		static_cast<OaU64>(width)
		* static_cast<OaU64>(height)
		* static_cast<OaU64>(requestedChannels);
	Pixels result;
	result.Data.Resize(static_cast<OaUsize>(byteCount));
	std::memcpy(
		result.Data.Data(),
		decoded,
		static_cast<OaUsize>(byteCount));
	stbi_image_free(decoded);
	result.Width = width;
	result.Height = height;
	result.Format = InFormat;
	return result;
}

OaResult<OaVec<OaU8>> EncodeStb(
	const Pixels& InPixels,
	OaImageCodec InCodec,
	OaU32 InQuality)
{
	if (not SupportsStbEncode(InCodec)) {
		return OaStatus::Unimplemented(
			"OaImageEncoder: codec is not provided by stb_image_write");
	}
	const int channels = InPixels.Channels();
	if (InPixels.Width <= 0 or InPixels.Height <= 0
		or channels <= 0 or channels > 4) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: invalid packed pixel metadata");
	}
	const OaU64 expectedBytes =
		static_cast<OaU64>(InPixels.Width)
		* static_cast<OaU64>(InPixels.Height)
		* static_cast<OaU64>(channels);
	if (InPixels.Data.Size() != expectedBytes) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: packed pixel size does not match metadata");
	}
	if (InCodec == OaImageCodec::Jpeg
		and (InQuality < 1U or InQuality > 100U)) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: JPEG quality must be in [1,100]");
	}
	if (InCodec == OaImageCodec::Jpeg
		and channels == 2) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: JPEG does not support GrayAlpha input");
	}

	OaVec<OaU8> encoded;
	int status = 0;
	switch (InCodec) {
		case OaImageCodec::Jpeg:
			status = stbi_write_jpg_to_func(
				AppendEncodedBytes,
				&encoded,
				InPixels.Width,
				InPixels.Height,
				channels,
				InPixels.Data.Data(),
				static_cast<int>(InQuality));
			break;
		case OaImageCodec::Png:
			status = stbi_write_png_to_func(
				AppendEncodedBytes,
				&encoded,
				InPixels.Width,
				InPixels.Height,
				channels,
				InPixels.Data.Data(),
				InPixels.Width * channels);
			break;
		case OaImageCodec::Bmp:
			status = stbi_write_bmp_to_func(
				AppendEncodedBytes,
				&encoded,
				InPixels.Width,
				InPixels.Height,
				channels,
				InPixels.Data.Data());
			break;
		case OaImageCodec::Tga:
			status = stbi_write_tga_to_func(
				AppendEncodedBytes,
				&encoded,
				InPixels.Width,
				InPixels.Height,
				channels,
				InPixels.Data.Data());
			break;
		default:
			break;
	}
	if (status == 0 or encoded.Empty()) {
		return OaStatus::Error(OaStatusCode::Internal, "OaImageEncoder: stb_image_write encode failed");
	}
	return encoded;
}

} // namespace OaImageCodecPrivate
