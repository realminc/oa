#pragma once

#include <oa/core/image.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/vision/type.h>

namespace oa::imageCodec {

struct Pixels {
	oa::Vec<oa::U8> data;
	oa::I32 width = 0;
	oa::I32 height = 0;
	oa::ImageFormat format = oa::ImageFormat::Rgb;

	[[nodiscard]] oa::I32 channels() const {
		return oa::imageFormatChannels(format);
	}
};

[[nodiscard]] oa::ImageCodec detectCodec(oa::Span<const oa::U8> inData) noexcept;
[[nodiscard]] oa::ImageCodec codecFromPath(const oa::Path& inPath) noexcept;

[[nodiscard]] bool supportsStbDecode(oa::ImageCodec inCodec) noexcept;
[[nodiscard]] bool supportsStbEncode(oa::ImageCodec inCodec) noexcept;
[[nodiscard]] oa::Result<Pixels> decodeStb(
	oa::Span<const oa::U8> inData,
	oa::ImageFormat inFormat);
[[nodiscard]] oa::Result<oa::Vec<oa::U8>> encodeStb(
	const Pixels& inPixels,
	oa::ImageCodec inCodec,
	oa::U32 inQuality);

[[nodiscard]] bool supportsWebp() noexcept;
[[nodiscard]] oa::Result<Pixels> decodeWebp(
	oa::Span<const oa::U8> inData,
	oa::ImageFormat inFormat);
[[nodiscard]] oa::Result<oa::Vec<oa::U8>> encodeWebp(
	const Pixels& inPixels,
	oa::U32 inQuality);

} // namespace oa::imageCodec
