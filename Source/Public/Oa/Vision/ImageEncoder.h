// OaImageEncoder — format-neutral still-image memory and file encode.
//
// Encoding is an explicit synchronous host boundary: pending work producing
// the image is completed, the semantic Float32 NCHW/CHW value is read back,
// and the selected codec writes packed pixels. SaveFile infers the codec from
// the path extension.

#pragma once

#include <Oa/Core/Image.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/ImageCodec.h>

struct OaImageEncoder {
	[[nodiscard]] static OaResult<OaVec<OaU8>> Encode(
		const OaImage& InImage,
		OaImageCodec InCodec,
		OaU32 InQuality = 90U);

	[[nodiscard]] static OaStatus SaveFile(
		const OaPath& InPath,
		const OaImage& InImage,
		OaU32 InQuality = 90U);

	// Reports the codec backends compiled into this build. Auto is not a codec
	// and always returns false.
	[[nodiscard]] static bool Supports(OaImageCodec InCodec) noexcept;
};
