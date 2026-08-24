#include "imageCodecInternal.h"

#include <oa/core/filesystem.h>
#include <oa/core/fnMatrix.h>
#include <oa/vision/fnImage.h>

#include <cmath>

namespace {

oa::U8 quantize(oa::F32 inValue) {
	if (not std::isfinite(inValue)) return 0U;
	const oa::F32 clamped = inValue < 0.0F
		? 0.0F
		: (inValue > 1.0F ? 1.0F : inValue);
	return static_cast<oa::U8>(clamped * 255.0F + 0.5F);
}

oa::Result<oa::imageCodec::Pixels> readPackedPixels(const oa::Image& inImage, oa::ImageCodec inCodec) {
	if (not inImage.validate() or inImage.isEmpty()) {
		return oa::Status::invalidArgument(	"oa::FnImage::encode: image is empty or semantically invalid");
	}
	if (inImage.layout() != oa::ImageLayout::Nchw
		and inImage.layout() != oa::ImageLayout::Chw) {
		return oa::Status::invalidArgument("oa::FnImage::encode: expected Nchw or Chw image layout");
	}
	if (inImage.batchSize() != 1) {
		return oa::Status::invalidArgument("oa::FnImage::encode: batched images require explicit item selection");
	}
	if (inImage.getDtype() != oa::ScalarType::Float32) {
		return oa::Status::invalidArgument("oa::FnImage::encode: expected Float32 image data");
	}

	const oa::I32 sourceChannels = inImage.channels();
	const oa::I32 width = inImage.width();
	const oa::I32 height = inImage.height();
	const oa::I64 pixelCount = static_cast<oa::I64>(width) * static_cast<oa::I64>(height);
	if (width <= 0 or height <= 0	or sourceChannels <= 0 or sourceChannels > 4) {
		return oa::Status::invalidArgument("oa::FnImage::encode: invalid image extent or channel count");
	}

	oa::Vec<oa::F32> planar;
	planar.resize(static_cast<oa::Usize>(pixelCount * sourceChannels));
	const oa::Status copyStatus = oa::FnMatrix::copyToHost(
		inImage.asMatrix(),
		planar.data(),
		planar.size() * sizeof(oa::F32)
	);
	if (copyStatus.isError()) return copyStatus;

	oa::ImageFormat outputFormat = inImage.format();
	if ((inCodec == oa::ImageCodec::Jpeg or inCodec == oa::ImageCodec::Webp) and outputFormat == oa::ImageFormat::GrayAlpha) {
		outputFormat = oa::ImageFormat::Rgb;
	}
	if (inCodec == oa::ImageCodec::Webp	and outputFormat == oa::ImageFormat::Gray) {
		outputFormat = oa::ImageFormat::Rgb;
	}
	if (outputFormat == oa::ImageFormat::Bgr) {
		outputFormat = oa::ImageFormat::Rgb;
	} else if (outputFormat == oa::ImageFormat::Bgra) {
		outputFormat = oa::ImageFormat::Rgba;
	}

	oa::imageCodec::Pixels packed;
	packed.width = width;
	packed.height = height;
	packed.format = outputFormat;
	const oa::I32 outputChannels = packed.channels();
	packed.data.resize(static_cast<oa::Usize>(pixelCount * outputChannels));

	auto sourceValue = [&](oa::I64 inPixel, oa::I32 inChannel) -> oa::F32 {
		return planar[static_cast<oa::Usize>(static_cast<oa::I64>(inChannel) * pixelCount + inPixel)];
	};
	for (oa::I64 pixel = 0; pixel < pixelCount; ++pixel) {
		oa::U8* destination =
			packed.data.data() + pixel * outputChannels;
		switch (inImage.format()) {
			case oa::ImageFormat::Gray:
				if (outputChannels == 1) {
					destination[0] = quantize(sourceValue(pixel, 0));
				} else {
					const oa::U8 gray = quantize(sourceValue(pixel, 0));
					destination[0] = gray;
					destination[1] = gray;
					destination[2] = gray;
				}
				break;
			case oa::ImageFormat::GrayAlpha:
				if (outputChannels == 2) {
					destination[0] = quantize(sourceValue(pixel, 0));
					destination[1] = quantize(sourceValue(pixel, 1));
				} else {
					const oa::U8 gray = quantize(sourceValue(pixel, 0));
					destination[0] = gray;
					destination[1] = gray;
					destination[2] = gray;
				}
				break;
			case oa::ImageFormat::Rgb:
			case oa::ImageFormat::Rgba:
				for (oa::I32 channel = 0;	channel < outputChannels;	++channel) {
					destination[channel] = quantize(sourceValue(pixel, channel));
				}
				break;
			case oa::ImageFormat::Bgr:
			case oa::ImageFormat::Bgra:
				destination[0] = quantize(sourceValue(pixel, 2));
				destination[1] = quantize(sourceValue(pixel, 1));
				destination[2] = quantize(sourceValue(pixel, 0));
				if (outputChannels == 4) {
					destination[3] = quantize(sourceValue(pixel, 3));
				}
				break;
		}
	}
	return packed;
}

} // namespace

oa::Result<oa::Vec<oa::U8>> oa::FnImage::encode(
	const oa::Image& inImage,
	oa::ImageCodec inCodec,
	oa::U32 inQuality)
{
	if (inCodec == oa::ImageCodec::Auto) {
		return oa::Status::invalidArgument(
			"oa::FnImage::encode requires an explicit codec");
	}
	if (not canEncode(inCodec)) {
		return oa::Status::unimplemented(
			oa::String("oa::FnImage::encode: ")
			+ oa::imageCodecToString(inCodec)
			+ " is not available in this build");
	}

	auto packed = readPackedPixels(inImage, inCodec);
	if (packed.isError()) return packed.getStatus();
	return inCodec == oa::ImageCodec::Webp
		? oa::imageCodec::encodeWebp(*packed, inQuality)
		: oa::imageCodec::encodeStb(*packed, inCodec, inQuality);
}

oa::Status oa::FnImage::saveFile(
	const oa::Path& inPath,
	const oa::Image& inImage,
	oa::U32 inQuality)
{
	const oa::ImageCodec codec =
		oa::imageCodec::codecFromPath(inPath);
	if (codec == oa::ImageCodec::Auto) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveFile expected .jpg, .jpeg, .png, .webp, .bmp, or .tga");
	}
	auto encoded = encode(inImage, codec, inQuality);
	if (encoded.isError()) return encoded.getStatus();
	return oa::Filesystem::writeBinary(
		inPath,
		oa::Span<const oa::U8>(encoded->data(), encoded->size()));
}

bool oa::FnImage::canEncode(oa::ImageCodec inCodec) noexcept
{
	using namespace oa::imageCodec;
	return supportsStbEncode(inCodec)
		or (inCodec == oa::ImageCodec::Webp and supportsWebp());
}
