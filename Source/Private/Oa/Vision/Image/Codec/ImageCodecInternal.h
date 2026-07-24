#pragma once

#include <Oa/Core/Image.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/ImageCodec.h>

namespace OaImageCodecPrivate {

struct Pixels {
	OaVec<OaU8> Data;
	OaI32 Width = 0;
	OaI32 Height = 0;
	OaImageFormat Format = OaImageFormat::Rgb;

	[[nodiscard]] OaI32 Channels() const {
		return OaImageFormatChannels(Format);
	}
};

[[nodiscard]] OaImageCodec DetectCodec(OaSpan<const OaU8> InData) noexcept;
[[nodiscard]] OaImageCodec CodecFromPath(const OaPath& InPath) noexcept;

[[nodiscard]] bool SupportsStbDecode(OaImageCodec InCodec) noexcept;
[[nodiscard]] bool SupportsStbEncode(OaImageCodec InCodec) noexcept;
[[nodiscard]] OaResult<Pixels> DecodeStb(
	OaSpan<const OaU8> InData,
	OaImageFormat InFormat);
[[nodiscard]] OaResult<OaVec<OaU8>> EncodeStb(
	const Pixels& InPixels,
	OaImageCodec InCodec,
	OaU32 InQuality);

[[nodiscard]] bool SupportsWebp() noexcept;
[[nodiscard]] OaResult<Pixels> DecodeWebp(
	OaSpan<const OaU8> InData,
	OaImageFormat InFormat);
[[nodiscard]] OaResult<OaVec<OaU8>> EncodeWebp(
	const Pixels& InPixels,
	OaU32 InQuality);

} // namespace OaImageCodecPrivate
