#include "imageCodecInternal.h"

#include <oa/core/filesystem.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/vision/fnImage.h>

namespace {

oa::Result<oa::Image> uploadDecodedImage(
	const oa::imageCodec::Pixels& inDecoded)
{
	const oa::I32 channels = inDecoded.channels();
	const oa::I64 pixelCount =
		static_cast<oa::I64>(inDecoded.width)
		* static_cast<oa::I64>(inDecoded.height);
	if (inDecoded.width <= 0 or inDecoded.height <= 0
		or channels <= 0 or inDecoded.data.size()
			!= static_cast<oa::Usize>(pixelCount * channels)) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"oa::FnImage::decodeMemory: decoded pixel metadata is inconsistent");
	}
	if (oa::ExecutionSession::getActivePtr() == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::FnImage::decodeMemory: no active oa::ExecutionSession");
	}

	oa::Vec<oa::F32> planar;
	planar.resize(static_cast<oa::Usize>(pixelCount * channels));
	for (oa::I32 channel = 0; channel < channels; ++channel) {
		for (oa::I64 pixel = 0; pixel < pixelCount; ++pixel) {
			const oa::Usize source =
				static_cast<oa::Usize>(pixel * channels + channel);
			const oa::Usize destination =
				static_cast<oa::Usize>(channel * pixelCount + pixel);
			planar[destination] =
				static_cast<oa::F32>(inDecoded.data[source]) / 255.0F;
		}
	}

	oa::Matrix matrix = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(planar.data()),
			planar.size() * sizeof(oa::F32)),
		oa::MatrixShape{
			1,
			channels,
			inDecoded.height,
			inDecoded.width},
		oa::ScalarType::Float32);
	if (matrix.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"oa::FnImage::decodeMemory: image upload failed");
	}
	return oa::Image(
		oa::move(matrix),
		oa::ImageLayout::Nchw,
		inDecoded.format);
}

} // namespace

oa::Result<oa::Image> oa::FnImage::decodeFile(
	const oa::Path& inPath,
	oa::ImageFormat inFormat)
{
	auto bytes = oa::Filesystem::readBinary(inPath);
	if (bytes.isError()) return bytes.getStatus();
	return decodeMemory(
		oa::Span<const oa::U8>(bytes->data(), bytes->size()),
		inFormat);
}

oa::Result<oa::Image> oa::FnImage::decodeMemory(
	oa::Span<const oa::U8> inData,
	oa::ImageFormat inFormat)
{
	using namespace oa::imageCodec;
	const oa::ImageCodec codec = detectCodec(inData);
	if (codec == oa::ImageCodec::Auto) {
		return oa::Status::error(
			oa::StatusCode::FileCorrupt,
			"oa::FnImage::decodeMemory: unsupported or unrecognized image bitstream");
	}

	oa::Result<Pixels> decoded =
		codec == oa::ImageCodec::Webp
		? decodeWebp(inData, inFormat)
		: decodeStb(inData, inFormat);
	if (decoded.isError()) return decoded.getStatus();
	return uploadDecodedImage(*decoded);
}

bool oa::FnImage::canDecode(oa::ImageCodec inCodec) noexcept
{
	using namespace oa::imageCodec;
	return supportsStbDecode(inCodec)
		or (inCodec == oa::ImageCodec::Webp and supportsWebp());
}
