#include "ImageCodecInternal.h"

#include <cctype>
#include <cstring>

namespace OaImageCodecPrivate {

namespace {

bool HasPrefix(
	OaSpan<const OaU8> InData,
	const OaU8* InPrefix,
	OaUsize InPrefixSize) noexcept
{
	return InData.Size() >= InPrefixSize
		and std::memcmp(InData.Data(), InPrefix, InPrefixSize) == 0;
}

bool IsTga(OaSpan<const OaU8> InData) noexcept
{
	if (InData.Size() < 18U) return false;
	const OaU8 colorMapType = InData[1];
	const OaU8 imageType = InData[2];
	const OaU16 width = static_cast<OaU16>(InData[12])
		| static_cast<OaU16>(InData[13] << 8U);
	const OaU16 height = static_cast<OaU16>(InData[14])
		| static_cast<OaU16>(InData[15] << 8U);
	const OaU8 depth = InData[16];
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

OaImageCodec DetectCodec(OaSpan<const OaU8> InData) noexcept
{
	static constexpr OaU8 png[] = {
		0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
	static constexpr OaU8 jpeg[] = {0xFFU, 0xD8U, 0xFFU};
	static constexpr OaU8 bmp[] = {'B', 'M'};

	if (HasPrefix(InData, png, sizeof(png))) return OaImageCodec::Png;
	if (HasPrefix(InData, jpeg, sizeof(jpeg))) return OaImageCodec::Jpeg;
	if (HasPrefix(InData, bmp, sizeof(bmp))) return OaImageCodec::Bmp;
	if (InData.Size() >= 12U
		and std::memcmp(InData.Data(), "RIFF", 4U) == 0
		and std::memcmp(InData.Data() + 8U, "WEBP", 4U) == 0) {
		return OaImageCodec::Webp;
	}
	if (IsTga(InData)) return OaImageCodec::Tga;
	return OaImageCodec::Auto;
}

OaImageCodec CodecFromPath(const OaPath& InPath) noexcept
{
	OaString extension = InPath.Extension().String();
	for (char& value : extension) {
		value = static_cast<char>(
			std::tolower(static_cast<unsigned char>(value)));
	}
	if (extension == ".jpg" or extension == ".jpeg") {
		return OaImageCodec::Jpeg;
	}
	if (extension == ".png") return OaImageCodec::Png;
	if (extension == ".webp") return OaImageCodec::Webp;
	if (extension == ".bmp") return OaImageCodec::Bmp;
	if (extension == ".tga") return OaImageCodec::Tga;
	return OaImageCodec::Auto;
}

} // namespace OaImageCodecPrivate
