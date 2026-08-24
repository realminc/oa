// oa::VideoDecoder preprocessing session commands.
// Decode, resize, and normalize are deliberately expressed as admitted semantic
// operations. There is no separate fused shader contract until profiling and a
// public per-plane decode surface justify one.

#include <oa/vision/videoDecoder.h>
#include <oa/vision/fnImage.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>

oa::Result<oa::Matrix> oa::VideoDecoder::decodeResizeNormalize(
	const oa::Span<const oa::U8>& inAccessUnit,
	oa::U32 inWidth,
	oa::U32 inHeight)
{
	const oa::NormalizationParams norm = {
		.mean = {0.485F, 0.456F, 0.406F},
		.std = {0.229F, 0.224F, 0.225F},
	};
	return decodeResizeNormalize(inAccessUnit, inWidth, inHeight, norm);
}

oa::Result<oa::Matrix> oa::VideoDecoder::decodeResizeNormalize(
	const oa::Span<const oa::U8>& inAccessUnit,
	oa::U32 inWidth,
	oa::U32 inHeight,
	const oa::NormalizationParams& inNorm)
{
	oa::Engine* rt = getEngine();
	if (rt == nullptr) {
		return oa::Status::error("oa::VideoDecoder::decodeResizeNormalize: decoder session has no engine");
	}

	// Decode at native resolution into [1,3,H,W] BF16. Skip ImageNet
	// normalization here so we can apply the caller-supplied mean/std below.
	auto matResult = decodeFrameToBf16(
		inAccessUnit,
		/*inNormalizeImageNet=*/false);
	if (matResult.isError()) {
		return matResult;
	}
	oa::Matrix mat = matResult.getValue();

	const auto shape = mat.getShape();
	const bool nativeMatch =
		shape.rank == 4 &&
		shape[2] == static_cast<oa::I64>(inHeight) &&
		shape[3] == static_cast<oa::I64>(inWidth);

	if (not nativeMatch) {
		mat = oa::FnImage::resize(*rt, mat, inWidth, inHeight, oa::InterpolationMode::Bilinear);
	}

	mat = oa::FnImage::normalize(*rt, mat, inNorm);

	return oa::Result<oa::Matrix>(mat);
}
