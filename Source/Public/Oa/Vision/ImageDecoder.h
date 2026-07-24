// OaImageDecoder — format-neutral still-image file and memory decode.
//
// Decode is a synchronous CPU codec boundary followed by upload into one
// semantic Float32 NCHW OaImage. Values are normalized to [0,1]. The default
// RGB output is directly consumable by OaFnImage, ML pipelines, and OaViewer.

#pragma once

#include <Oa/Core/Image.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/ImageCodec.h>

struct OaImageDecoder {
	[[nodiscard]] static OaResult<OaImage> LoadFile(
		const OaPath& InPath,
		OaImageFormat InFormat = OaImageFormat::Rgb);

	[[nodiscard]] static OaResult<OaImage> LoadMemory(
		OaSpan<const OaU8> InData,
		OaImageFormat InFormat = OaImageFormat::Rgb);

	// Reports the codec backends compiled into this build. Auto is not a codec
	// and always returns false.
	[[nodiscard]] static bool Supports(OaImageCodec InCodec) noexcept;
};
