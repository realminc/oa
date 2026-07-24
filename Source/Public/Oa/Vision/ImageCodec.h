// OaImageCodec — still-image codec identity shared by decode and encode.

#pragma once

#include <Oa/Core/Types.h>

enum class OaImageCodec : OaU8 {
	Auto,
	Jpeg,
	Png,
	Webp,
	Bmp,
	Tga,
};

[[nodiscard]] constexpr OaStringView OaImageCodecName(
	OaImageCodec InCodec) noexcept
{
	switch (InCodec) {
		case OaImageCodec::Jpeg: return "JPEG";
		case OaImageCodec::Png: return "PNG";
		case OaImageCodec::Webp: return "WebP";
		case OaImageCodec::Bmp: return "BMP";
		case OaImageCodec::Tga: return "TGA";
		default: return "Auto";
	}
}
