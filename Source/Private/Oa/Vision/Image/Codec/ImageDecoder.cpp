#include "ImageCodecInternal.h"

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Vision/ImageDecoder.h>

namespace {

OaResult<OaImage> UploadDecodedImage(
	const OaImageCodecPrivate::Pixels& InDecoded)
{
	const OaI32 channels = InDecoded.Channels();
	const OaI64 pixelCount =
		static_cast<OaI64>(InDecoded.Width)
		* static_cast<OaI64>(InDecoded.Height);
	if (InDecoded.Width <= 0 or InDecoded.Height <= 0
		or channels <= 0 or InDecoded.Data.Size()
			!= static_cast<OaUsize>(pixelCount * channels)) {
		return OaStatus::Error(
			OaStatusCode::DataLoss,
			"OaImageDecoder: decoded pixel metadata is inconsistent");
	}
	if (OaContext::GetDefaultPtr() == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaImageDecoder: no active OaContext");
	}

	OaVec<OaF32> planar;
	planar.Resize(static_cast<OaUsize>(pixelCount * channels));
	for (OaI32 channel = 0; channel < channels; ++channel) {
		for (OaI64 pixel = 0; pixel < pixelCount; ++pixel) {
			const OaUsize source =
				static_cast<OaUsize>(pixel * channels + channel);
			const OaUsize destination =
				static_cast<OaUsize>(channel * pixelCount + pixel);
			planar[destination] =
				static_cast<OaF32>(InDecoded.Data[source]) / 255.0F;
		}
	}

	OaMatrix matrix = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(planar.Data()),
			planar.Size() * sizeof(OaF32)),
		OaMatrixShape{
			1,
			channels,
			InDecoded.Height,
			InDecoded.Width},
		OaScalarType::Float32);
	if (matrix.IsEmpty()) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaImageDecoder: image upload failed");
	}
	return OaImage(
		OaStdMove(matrix),
		OaImageLayout::Nchw,
		InDecoded.Format);
}

} // namespace

OaResult<OaImage> OaImageDecoder::LoadFile(
	const OaPath& InPath,
	OaImageFormat InFormat)
{
	auto bytes = OaFilesystem::ReadBinary(InPath);
	if (bytes.IsError()) return bytes.GetStatus();
	return LoadMemory(
		OaSpan<const OaU8>(bytes->Data(), bytes->Size()),
		InFormat);
}

OaResult<OaImage> OaImageDecoder::LoadMemory(
	OaSpan<const OaU8> InData,
	OaImageFormat InFormat)
{
	using namespace OaImageCodecPrivate;
	const OaImageCodec codec = DetectCodec(InData);
	if (codec == OaImageCodec::Auto) {
		return OaStatus::Error(
			OaStatusCode::FileCorrupt,
			"OaImageDecoder: unsupported or unrecognized image bitstream");
	}

	OaResult<Pixels> decoded =
		codec == OaImageCodec::Webp
		? DecodeWebp(InData, InFormat)
		: DecodeStb(InData, InFormat);
	if (decoded.IsError()) return decoded.GetStatus();
	return UploadDecodedImage(*decoded);
}

bool OaImageDecoder::Supports(OaImageCodec InCodec) noexcept
{
	using namespace OaImageCodecPrivate;
	return SupportsStbDecode(InCodec)
		or (InCodec == OaImageCodec::Webp and SupportsWebp());
}
