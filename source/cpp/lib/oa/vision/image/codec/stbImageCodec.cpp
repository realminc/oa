#include "imageCodecInternal.h"

#include "../../../../../thirdparty/stb/stb_image.h"
#include "../../../../../thirdparty/stb/stb_image_write.h"

#include <cstring>
#include <limits>

namespace oa::imageCodec {

namespace {

oa::I32 requestedChannels(oa::ImageFormat inFormat)
{
	switch (inFormat) {
		case oa::ImageFormat::Gray: return 1;
		case oa::ImageFormat::Rgb: return 3;
		case oa::ImageFormat::Rgba: return 4;
		default: return 0;
	}
}

void appendEncodedBytes(void* inContext, void* inData, int inSize)
{
	auto& output = *static_cast<oa::Vec<oa::U8>*>(inContext);
	if (inData == nullptr or inSize <= 0) return;
	const oa::Usize oldSize = output.size();
	output.resize(oldSize + static_cast<oa::Usize>(inSize));
	std::memcpy(
		output.data() + oldSize,
		inData,
		static_cast<oa::Usize>(inSize));
}

} // namespace

bool supportsStbDecode(oa::ImageCodec inCodec) noexcept
{
	return inCodec == oa::ImageCodec::Jpeg
		or inCodec == oa::ImageCodec::Png
		or inCodec == oa::ImageCodec::Bmp
		or inCodec == oa::ImageCodec::Tga;
}

bool supportsStbEncode(oa::ImageCodec inCodec) noexcept
{
	return supportsStbDecode(inCodec);
}

oa::Result<Pixels> decodeStb(
	oa::Span<const oa::U8> inData,
	oa::ImageFormat inFormat)
{
	const oa::I32 requestedChannelCount = requestedChannels(inFormat);
	if (requestedChannelCount == 0) {
		return oa::Status::invalidArgument(
			"oa::FnImage::decodeMemory supports Gray, Rgb, or Rgba output");
	}
	if (inData.empty()) {
		return oa::Status::invalidArgument(
			"oa::FnImage::decodeMemory: compressed data is empty");
	}
	if (inData.size() > static_cast<oa::Usize>(
		std::numeric_limits<int>::max())) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::FnImage::decodeMemory: compressed data exceeds stb_image limits");
	}

	int width = 0;
	int height = 0;
	int sourceChannels = 0;
	stbi_uc* decoded = stbi_load_from_memory(
		inData.data(),
		static_cast<int>(inData.size()),
		&width,
		&height,
		&sourceChannels,
		requestedChannelCount);
	if (decoded == nullptr) {
		const char* reason = stbi_failure_reason();
		return oa::Status::error(
			oa::StatusCode::FileCorrupt,
			oa::String("oa::FnImage::decodeMemory: stb_image decode failed: ")
				+ (reason != nullptr ? reason : "unknown error"));
	}

	const oa::U64 byteCount =
		static_cast<oa::U64>(width)
		* static_cast<oa::U64>(height)
		* static_cast<oa::U64>(requestedChannelCount);
	Pixels result;
	result.data.resize(static_cast<oa::Usize>(byteCount));
	std::memcpy(
		result.data.data(),
		decoded,
		static_cast<oa::Usize>(byteCount));
	stbi_image_free(decoded);
	result.width = width;
	result.height = height;
	result.format = inFormat;
	return result;
}

oa::Result<oa::Vec<oa::U8>> encodeStb(
	const Pixels& inPixels,
	oa::ImageCodec inCodec,
	oa::U32 inQuality)
{
	if (not supportsStbEncode(inCodec)) {
		return oa::Status::unimplemented(
			"oa::FnImage::encode: codec is not provided by stb_image_write");
	}
	const int channels = inPixels.channels();
	if (inPixels.width <= 0 or inPixels.height <= 0
		or channels <= 0 or channels > 4) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: invalid packed pixel metadata");
	}
	const oa::U64 expectedBytes =
		static_cast<oa::U64>(inPixels.width)
		* static_cast<oa::U64>(inPixels.height)
		* static_cast<oa::U64>(channels);
	if (inPixels.data.size() != expectedBytes) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: packed pixel size does not match metadata");
	}
	if (inCodec == oa::ImageCodec::Jpeg
		and (inQuality < 1U or inQuality > 100U)) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: JPEG quality must be in [1,100]");
	}
	if (inCodec == oa::ImageCodec::Jpeg
		and channels == 2) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode: JPEG does not support GrayAlpha input");
	}

	oa::Vec<oa::U8> encoded;
	int status = 0;
	switch (inCodec) {
		case oa::ImageCodec::Jpeg:
			status = stbi_write_jpg_to_func(
				appendEncodedBytes,
				&encoded,
				inPixels.width,
				inPixels.height,
				channels,
				inPixels.data.data(),
				static_cast<int>(inQuality));
			break;
		case oa::ImageCodec::Png:
			status = stbi_write_png_to_func(
				appendEncodedBytes,
				&encoded,
				inPixels.width,
				inPixels.height,
				channels,
				inPixels.data.data(),
				inPixels.width * channels);
			break;
		case oa::ImageCodec::Bmp:
			status = stbi_write_bmp_to_func(
				appendEncodedBytes,
				&encoded,
				inPixels.width,
				inPixels.height,
				channels,
				inPixels.data.data());
			break;
		case oa::ImageCodec::Tga:
			status = stbi_write_tga_to_func(
				appendEncodedBytes,
				&encoded,
				inPixels.width,
				inPixels.height,
				channels,
				inPixels.data.data());
			break;
		default:
			break;
	}
	if (status == 0 or encoded.empty()) {
		return oa::Status::error(oa::StatusCode::Internal, "oa::FnImage::encode: stb_image_write encode failed");
	}
	return encoded;
}

} // namespace oa::imageCodec
