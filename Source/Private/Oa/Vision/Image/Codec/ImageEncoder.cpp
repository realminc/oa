#include "ImageCodecInternal.h"

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Vision/ImageEncoder.h>

#include <cmath>

namespace {

OaU8 Quantize(OaF32 InValue)
{
	if (not std::isfinite(InValue)) return 0U;
	const OaF32 clamped = InValue < 0.0F
		? 0.0F
		: (InValue > 1.0F ? 1.0F : InValue);
	return static_cast<OaU8>(clamped * 255.0F + 0.5F);
}

OaResult<OaImageCodecPrivate::Pixels> ReadPackedPixels(
	const OaImage& InImage,
	OaImageCodec InCodec)
{
	if (not InImage.Validate() or InImage.IsEmpty()) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: image is empty or semantically invalid");
	}
	if (InImage.Layout() != OaImageLayout::Nchw
		and InImage.Layout() != OaImageLayout::Chw) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: expected Nchw or Chw image layout");
	}
	if (InImage.BatchSize() != 1) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: batched images require explicit item selection");
	}
	if (InImage.GetDtype() != OaScalarType::Float32) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: expected Float32 image data");
	}

	const OaI32 sourceChannels = InImage.Channels();
	const OaI32 width = InImage.Width();
	const OaI32 height = InImage.Height();
	const OaI64 pixelCount =
		static_cast<OaI64>(width) * static_cast<OaI64>(height);
	if (width <= 0 or height <= 0
		or sourceChannels <= 0 or sourceChannels > 4) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder: invalid image extent or channel count");
	}

	OaVec<OaF32> planar;
	planar.Resize(static_cast<OaUsize>(pixelCount * sourceChannels));
	const OaStatus copyStatus = OaFnMatrix::CopyToHost(
		InImage.AsMatrix(),
		planar.Data(),
		planar.Size() * sizeof(OaF32));
	if (copyStatus.IsError()) return copyStatus;

	OaImageFormat outputFormat = InImage.Format();
	if ((InCodec == OaImageCodec::Jpeg or InCodec == OaImageCodec::Webp)
		and outputFormat == OaImageFormat::GrayAlpha) {
		outputFormat = OaImageFormat::Rgb;
	}
	if (InCodec == OaImageCodec::Webp
		and outputFormat == OaImageFormat::Gray) {
		outputFormat = OaImageFormat::Rgb;
	}
	if (outputFormat == OaImageFormat::Bgr) {
		outputFormat = OaImageFormat::Rgb;
	} else if (outputFormat == OaImageFormat::Bgra) {
		outputFormat = OaImageFormat::Rgba;
	}

	OaImageCodecPrivate::Pixels packed;
	packed.Width = width;
	packed.Height = height;
	packed.Format = outputFormat;
	const OaI32 outputChannels = packed.Channels();
	packed.Data.Resize(
		static_cast<OaUsize>(pixelCount * outputChannels));

	auto sourceValue = [&](OaI64 InPixel, OaI32 InChannel) -> OaF32 {
		return planar[static_cast<OaUsize>(
			static_cast<OaI64>(InChannel) * pixelCount + InPixel)];
	};
	for (OaI64 pixel = 0; pixel < pixelCount; ++pixel) {
		OaU8* destination =
			packed.Data.Data() + pixel * outputChannels;
		switch (InImage.Format()) {
			case OaImageFormat::Gray:
				if (outputChannels == 1) {
					destination[0] = Quantize(sourceValue(pixel, 0));
				} else {
					const OaU8 gray = Quantize(sourceValue(pixel, 0));
					destination[0] = gray;
					destination[1] = gray;
					destination[2] = gray;
				}
				break;
			case OaImageFormat::GrayAlpha:
				if (outputChannels == 2) {
					destination[0] = Quantize(sourceValue(pixel, 0));
					destination[1] = Quantize(sourceValue(pixel, 1));
				} else {
					const OaU8 gray = Quantize(sourceValue(pixel, 0));
					destination[0] = gray;
					destination[1] = gray;
					destination[2] = gray;
				}
				break;
			case OaImageFormat::Rgb:
			case OaImageFormat::Rgba:
				for (OaI32 channel = 0;
					channel < outputChannels;
					++channel) {
					destination[channel] =
						Quantize(sourceValue(pixel, channel));
				}
				break;
			case OaImageFormat::Bgr:
			case OaImageFormat::Bgra:
				destination[0] = Quantize(sourceValue(pixel, 2));
				destination[1] = Quantize(sourceValue(pixel, 1));
				destination[2] = Quantize(sourceValue(pixel, 0));
				if (outputChannels == 4) {
					destination[3] = Quantize(sourceValue(pixel, 3));
				}
				break;
		}
	}
	return packed;
}

} // namespace

OaResult<OaVec<OaU8>> OaImageEncoder::Encode(
	const OaImage& InImage,
	OaImageCodec InCodec,
	OaU32 InQuality)
{
	if (InCodec == OaImageCodec::Auto) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder::Encode requires an explicit codec");
	}
	if (not Supports(InCodec)) {
		return OaStatus::Unimplemented(
			OaString("OaImageEncoder: ")
			+ OaImageCodecName(InCodec).Data()
			+ " is not available in this build");
	}

	auto packed = ReadPackedPixels(InImage, InCodec);
	if (packed.IsError()) return packed.GetStatus();
	return InCodec == OaImageCodec::Webp
		? OaImageCodecPrivate::EncodeWebp(*packed, InQuality)
		: OaImageCodecPrivate::EncodeStb(*packed, InCodec, InQuality);
}

OaStatus OaImageEncoder::SaveFile(
	const OaPath& InPath,
	const OaImage& InImage,
	OaU32 InQuality)
{
	const OaImageCodec codec =
		OaImageCodecPrivate::CodecFromPath(InPath);
	if (codec == OaImageCodec::Auto) {
		return OaStatus::InvalidArgument(
			"OaImageEncoder::SaveFile expected .jpg, .jpeg, .png, .webp, .bmp, or .tga");
	}
	auto encoded = Encode(InImage, codec, InQuality);
	if (encoded.IsError()) return encoded.GetStatus();
	return OaFilesystem::WriteBinary(
		InPath,
		OaSpan<const OaU8>(encoded->Data(), encoded->Size()));
}

bool OaImageEncoder::Supports(OaImageCodec InCodec) noexcept
{
	using namespace OaImageCodecPrivate;
	return SupportsStbEncode(InCodec)
		or (InCodec == OaImageCodec::Webp and SupportsWebp());
}
