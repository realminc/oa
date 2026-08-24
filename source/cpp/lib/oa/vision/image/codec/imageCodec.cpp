#include "imageCodecInternal.h"

#include <cctype>
#include <cstring>

namespace oa::imageCodec {

namespace {

bool hasPrefix(
	oa::Span<const oa::U8> inData,
	const oa::U8* inPrefix,
	oa::Usize inPrefixSize) noexcept
{
	return inData.size() >= inPrefixSize
		and std::memcmp(inData.data(), inPrefix, inPrefixSize) == 0;
}

bool isTga(oa::Span<const oa::U8> inData) noexcept
{
	if (inData.size() < 18U) return false;
	const oa::U8 colorMapType = inData[1];
	const oa::U8 imageType = inData[2];
	const oa::U16 width = static_cast<oa::U16>(inData[12])
		| static_cast<oa::U16>(inData[13] << 8U);
	const oa::U16 height = static_cast<oa::U16>(inData[14])
		| static_cast<oa::U16>(inData[15] << 8U);
	const oa::U8 depth = inData[16];
	const bool validType =
		imageType == 1U or imageType == 2U or imageType == 3U
		or imageType == 9U or imageType == 10U or imageType == 11U;
	const bool validDepth =
		depth == 8U or depth == 15U or depth == 16U
		or depth == 24U or depth == 32U;
	return colorMapType <= 1U and validType and validDepth
		and width > 0U and height > 0U;
}

} // namespace

oa::ImageCodec detectCodec(oa::Span<const oa::U8> inData) noexcept
{
	static constexpr oa::U8 png[] = {
		0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
	static constexpr oa::U8 jpeg[] = {0xFFU, 0xD8U, 0xFFU};
	static constexpr oa::U8 bmp[] = {'B', 'M'};

	if (hasPrefix(inData, png, sizeof(png))) return oa::ImageCodec::Png;
	if (hasPrefix(inData, jpeg, sizeof(jpeg))) return oa::ImageCodec::Jpeg;
	if (hasPrefix(inData, bmp, sizeof(bmp))) return oa::ImageCodec::Bmp;
	if (inData.size() >= 12U
		and std::memcmp(inData.data(), "RIFF", 4U) == 0
		and std::memcmp(inData.data() + 8U, "WEBP", 4U) == 0) {
		return oa::ImageCodec::Webp;
	}
	if (isTga(inData)) return oa::ImageCodec::Tga;
	return oa::ImageCodec::Auto;
}

oa::ImageCodec codecFromPath(const oa::Path& inPath) noexcept
{
	oa::String extension = inPath.extension().string();
	for (char& value : extension) {
		value = static_cast<char>(
			std::tolower(static_cast<unsigned char>(value)));
	}
	if (extension == ".jpg" or extension == ".jpeg") {
		return oa::ImageCodec::Jpeg;
	}
	if (extension == ".png") return oa::ImageCodec::Png;
	if (extension == ".webp") return oa::ImageCodec::Webp;
	if (extension == ".bmp") return oa::ImageCodec::Bmp;
	if (extension == ".tga") return oa::ImageCodec::Tga;
	return oa::ImageCodec::Auto;
}

} // namespace oa::imageCodec
