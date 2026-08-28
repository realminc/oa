// FnImageFilter.cpp — graph-native NCHW filtering primitives and semantic ops.

#include <oa/vision/fnImage.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>

namespace {

oa::U32 divCeil(oa::U32 inA, oa::U32 inB) {
	return (inA + inB - 1U) / inB;
}

bool isValidImage(const oa::Matrix& inImage, oa::StringView inOperation) {
	const auto shape = inImage.getShape();
	if (inImage.hasStorage() && shape.rank == 4 && shape[0] > 0 && shape[1] > 0 &&
		shape[2] > 0 && shape[3] > 0) {
		return true;
	}
	OaLogWarn(oa::LogComponent::Vision, "%.*s expects a stored, non-empty [B,C,H,W] tensor",
		static_cast<int>(inOperation.size()), inOperation.data());
	return false;
}

bool isValidBorder(oa::BorderMode inBorder) {
	return inBorder == oa::BorderMode::Constant ||
		inBorder == oa::BorderMode::Replicate ||
		inBorder == oa::BorderMode::Reflect ||
		inBorder == oa::BorderMode::Reflect101 ||
		inBorder == oa::BorderMode::Wrap;
}

bool isValidKernel2d(const oa::Matrix& inImage, const oa::Matrix& inKernel) {
	const auto shape = inKernel.getShape();
	return shape.rank == 2 && shape[0] > 0 && shape[1] > 0 &&
		shape[0] <= 31 && shape[1] <= 31 &&
		(shape[0] & 1) != 0 && (shape[1] & 1) != 0 &&
		inKernel.getDtype() == inImage.getDtype() && inKernel.hasStorage();
}

bool isValidKernel1d(const oa::Matrix& inImage, const oa::Matrix& inKernel) {
	const auto shape = inKernel.getShape();
	const bool vectorShape = shape.rank == 1 ||
		(shape.rank == 2 && (shape[0] == 1 || shape[1] == 1));
	const oa::I64 size = shape.numElements();
	return vectorShape && size > 0 && size <= 31 && (size & 1) != 0 &&
		inKernel.getDtype() == inImage.getDtype() && inKernel.hasStorage();
}

oa::Matrix makeKernel2d(
	const oa::Matrix& inImage,
	const oa::Array<oa::F32, 9>& inValues
) {
	const oa::I64 side = static_cast<oa::I64>(oa::sqrt(static_cast<oa::F64>(inValues.size())));
	oa::Vector<oa::F32> values;
	values.reserve(inValues.size());
	for (const oa::F32 value : inValues) values.pushBack(value);
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(values.data()),
			values.size() * sizeof(oa::F32)),
		{side, side}, inImage.getDtype());
}

oa::Matrix derivativeKernel(const oa::Matrix& inImage, oa::U32 inDx, oa::U32 inDy,
	bool inScharr) {
	if (inDx == 1 && inDy == 0) {
		return inScharr
			? makeKernel2d(inImage, {-3, 0, 3, -10, 0, 10, -3, 0, 3})
			: makeKernel2d(inImage, {-1, 0, 1, -2, 0, 2, -1, 0, 1});
	}
	return inScharr
		? makeKernel2d(inImage, {-3, -10, -3, 0, 0, 0, 3, 10, 3})
		: makeKernel2d(inImage, {-1, -2, -1, 0, 0, 0, 1, 2, 1});
}

bool isValidMorphology(const oa::Matrix& inImage, oa::U32 inKernelWidth,
	oa::U32 inKernelHeight, oa::BorderMode inBorder, oa::F32 inBorderValue,
	oa::StringView inOperation) {
	if (isValidImage(inImage, inOperation) &&
		inKernelWidth > 0 && inKernelHeight > 0 &&
		inKernelWidth <= 31 && inKernelHeight <= 31 &&
		(inKernelWidth & 1U) != 0 && (inKernelHeight & 1U) != 0 &&
		isValidBorder(inBorder) && oa::isFinite(inBorderValue)) {
		return true;
	}
	OaLogWarn(oa::LogComponent::Vision,
		"%.*s requires odd kernel dimensions in [1,31], a valid border, and a finite border value",
		static_cast<int>(inOperation.size()), inOperation.data());
	return false;
}

oa::Matrix morphologyPass(const oa::Matrix& inImage, oa::U32 inKernelWidth,
	oa::U32 inKernelHeight, oa::BorderMode inBorder, oa::F32 inBorderValue,
	bool inDilate) {
	const auto shape = inImage.getShape();
	const oa::U32 batch = static_cast<oa::U32>(shape[0]);
	const oa::U32 channels = static_cast<oa::U32>(shape[1]);
	const oa::U32 height = static_cast<oa::U32>(shape[2]);
	const oa::U32 width = static_cast<oa::U32>(shape[3]);
	auto output = oa::FnMatrix::empty(shape, inImage.getDtype());
	struct Push {
		oa::U32 batch, channels, height, width, kernelHeight, kernelWidth;
		oa::U32 operation, border;
		oa::F32 BorderValue;
	} push{batch, channels, height, width, inKernelHeight, inKernelWidth,
		inDilate ? 1U : 0U, static_cast<oa::U32>(inBorder), inBorderValue};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageMorphology", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(width, 16), divCeil(height, 16), batch * channels);
	return output;
}

oa::Matrix neighborhoodPass(const oa::Matrix& inImage, oa::U32 inKernelSize,
	oa::U32 inOperation, oa::BorderMode inBorder, oa::F32 inBorderValue,
	oa::F32 inP0 = 0.0F, oa::F32 inP1 = 0.0F, oa::F32 inP2 = 0.0F) {
	const auto shape = inImage.getShape();
	auto output = oa::FnMatrix::empty(shape, inImage.getDtype());
	struct Push {
		oa::U32 batch, channels, height, width, kernelSize, operation, border;
		oa::F32 BorderValue, P0, P1, P2;
	};
	Push push{static_cast<oa::U32>(shape[0]), static_cast<oa::U32>(shape[1]),
		static_cast<oa::U32>(shape[2]), static_cast<oa::U32>(shape[3]),
		inKernelSize, inOperation, static_cast<oa::U32>(inBorder),
		inBorderValue, inP0, inP1, inP2};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageNeighborhood", {&inImage, &output}, access,
		&push, sizeof(push), divCeil(push.width, 16), divCeil(push.height, 16),
		push.batch * push.channels);
	return output;
}

bool isValidNeighborhood(const oa::Matrix& inImage, oa::U32 inKernelSize,
	oa::BorderMode inBorder, oa::StringView inOperation) {
	if (isValidImage(inImage, inOperation) && inKernelSize > 0 &&
		inKernelSize <= 15 && (inKernelSize & 1U) != 0 && isValidBorder(inBorder)) return true;
	OaLogWarn(oa::LogComponent::Vision,
		"%.*s requires an odd kernel in [1,15] and a valid border",
		static_cast<int>(inOperation.size()), inOperation.data());
	return false;
}

} // namespace

oa::Matrix oa::FnImage::convolve2d(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	const oa::Matrix& inKernel,
	oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	(void)inRt;
	if (!isValidImage(inImage, "oa::FnImage::convolve2d") ||
		!isValidKernel2d(inImage, inKernel) || !isValidBorder(inBorder) ||
		!oa::isFinite(inBorderValue)) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::convolve2d requires an odd <=31 rank-2 kernel with matching dtype and a valid border");
		return inImage;
	}

	const auto imageShape = inImage.getShape();
	const auto kernelShape = inKernel.getShape();
	const oa::U32 batch = static_cast<oa::U32>(imageShape[0]);
	const oa::U32 channels = static_cast<oa::U32>(imageShape[1]);
	const oa::U32 height = static_cast<oa::U32>(imageShape[2]);
	const oa::U32 width = static_cast<oa::U32>(imageShape[3]);
	auto output = oa::FnMatrix::empty(imageShape, inImage.getDtype());

	struct Push {
		oa::U32 batch, channels, height, width, kernelHeight, kernelWidth, border;
		oa::F32 BorderValue;
	} push{batch, channels, height, width,
		static_cast<oa::U32>(kernelShape[0]), static_cast<oa::U32>(kernelShape[1]),
		static_cast<oa::U32>(inBorder), inBorderValue};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::ExecutionSession::getActive().add( "ImageConvolve2d", {&inImage, &inKernel, &output},
		access, &push, sizeof(push), divCeil(width, 16), divCeil(height, 16), batch * channels);
	return output;
}

oa::Matrix oa::FnImage::separableConvolve2d(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	const oa::Matrix& inKernelX,
	const oa::Matrix& inKernelY,
	oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	(void)inRt;
	if (!isValidImage(inImage, "oa::FnImage::separableConvolve2d") ||
		!isValidKernel1d(inImage, inKernelX) || !isValidKernel1d(inImage, inKernelY) ||
		!isValidBorder(inBorder) || !oa::isFinite(inBorderValue)) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::separableConvolve2d requires odd <=31 vector kernels with matching dtype and a valid border");
		return inImage;
	}

	const auto shape = inImage.getShape();
	const oa::U32 batch = static_cast<oa::U32>(shape[0]);
	const oa::U32 channels = static_cast<oa::U32>(shape[1]);
	const oa::U32 height = static_cast<oa::U32>(shape[2]);
	const oa::U32 width = static_cast<oa::U32>(shape[3]);
	auto temporary = oa::FnMatrix::empty(shape, inImage.getDtype());
	auto output = oa::FnMatrix::empty(shape, inImage.getDtype());

	struct Push {
		oa::U32 batch, channels, height, width, kernelSize, horizontal, border;
		oa::F32 BorderValue;
	};
	Push horizontal{batch, channels, height, width,
		static_cast<oa::U32>(inKernelX.numElements()), 1U,
		static_cast<oa::U32>(inBorder), inBorderValue};
	Push vertical{batch, channels, height, width,
		static_cast<oa::U32>(inKernelY.numElements()), 0U,
		static_cast<oa::U32>(inBorder), inBorderValue};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read, oa::BufferAccess::Write};
	auto& context = oa::ExecutionSession::getActive();
	context.add( "ImageSeparableConvolve", {&inImage, &inKernelX, &temporary},
		access, &horizontal, sizeof(horizontal),
		divCeil(width, 16), divCeil(height, 16), batch * channels);
	context.add( "ImageSeparableConvolve", {&temporary, &inKernelY, &output},
		access, &vertical, sizeof(vertical),
		divCeil(width, 16), divCeil(height, 16), batch * channels);
	return output;
}

oa::Matrix oa::FnImage::averageBlur(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder)
{
	if (!isValidImage(inImage, "oa::FnImage::averageBlur") ||
		inKernelWidth == 0 || inKernelHeight == 0 ||
		inKernelWidth > 31 || inKernelHeight > 31 ||
		(inKernelWidth & 1U) == 0 || (inKernelHeight & 1U) == 0) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::averageBlur kernel dimensions must be odd and in [1,31]");
		return inImage;
	}
	const oa::F64 weight = 1.0 / static_cast<oa::F64>(inKernelWidth * inKernelHeight);
	auto kernel = oa::FnMatrix::full(
		{static_cast<oa::I64>(inKernelHeight), static_cast<oa::I64>(inKernelWidth)},
		weight, inImage.getDtype());
	return convolve2d(inRt, inImage, kernel, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::sobel(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inDx, oa::U32 inDy, oa::BorderMode inBorder)
{
	if (!((inDx == 1 && inDy == 0) || (inDx == 0 && inDy == 1))) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::sobel requires derivative (dx,dy) to be (1,0) or (0,1)");
		return inImage;
	}
	auto kernel = derivativeKernel(inImage, inDx, inDy, false);
	return convolve2d(inRt, inImage, kernel, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::scharr(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inDx, oa::U32 inDy, oa::BorderMode inBorder)
{
	if (!((inDx == 1 && inDy == 0) || (inDx == 0 && inDy == 1))) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::scharr requires derivative (dx,dy) to be (1,0) or (0,1)");
		return inImage;
	}
	auto kernel = derivativeKernel(inImage, inDx, inDy, true);
	return convolve2d(inRt, inImage, kernel, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::laplacian(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::BorderMode inBorder)
{
	auto kernel = makeKernel2d(inImage, {0, 1, 0, 1, -4, 1, 0, 1, 0});
	return convolve2d(inRt, inImage, kernel, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::sharpen(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inAmount, oa::BorderMode inBorder) {
	if (!oa::isFinite(inAmount) || inAmount < 0.0F) return inImage;
	auto kernel = makeKernel2d(inImage, {0, -inAmount, 0,
		-inAmount, 1.0F + 4.0F * inAmount, -inAmount,
		0, -inAmount, 0});
	return convolve2d(inRt, inImage, kernel, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::erode(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	(void)inRt;
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::erode")) return inImage;
	return morphologyPass(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, false);
}

oa::Matrix oa::FnImage::dilate(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	(void)inRt;
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::dilate")) return inImage;
	return morphologyPass(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, true);
}

oa::Matrix oa::FnImage::morphologyOpen(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::morphologyOpen")) return inImage;
	auto eroded = morphologyPass(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, false);
	return dilate(inRt, eroded, inKernelWidth, inKernelHeight, inBorder, inBorderValue);
}

oa::Matrix oa::FnImage::morphologyClose(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::morphologyClose")) return inImage;
	auto dilated = morphologyPass(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, true);
	return erode(inRt, dilated, inKernelWidth, inKernelHeight, inBorder, inBorderValue);
}

oa::Matrix oa::FnImage::morphologyGradient(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue)
{
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::morphologyGradient")) return inImage;
	auto dilated = dilate(inRt, inImage, inKernelWidth, inKernelHeight, inBorder, inBorderValue);
	auto eroded = erode(inRt, inImage, inKernelWidth, inKernelHeight, inBorder, inBorderValue);
	return oa::FnMatrix::sub(dilated, eroded);
}

oa::Matrix oa::FnImage::medianBlur(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelSize, oa::BorderMode inBorder) {
	(void)inRt;
	if (!isValidNeighborhood(inImage, inKernelSize, inBorder,
		"oa::FnImage::medianBlur")) return inImage;
	return neighborhoodPass(inImage, inKernelSize, 0, inBorder, 0.0F);
}

oa::Matrix oa::FnImage::bilateralFilter(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelSize, oa::F32 inSigmaColor, oa::F32 inSigmaSpace,
	oa::BorderMode inBorder) {
	(void)inRt;
	if (!isValidNeighborhood(inImage, inKernelSize, inBorder,
		"oa::FnImage::bilateralFilter") || inSigmaColor <= 0.0F ||
		inSigmaSpace <= 0.0F || !oa::isFinite(inSigmaColor) ||
		!oa::isFinite(inSigmaSpace)) return inImage;
	return neighborhoodPass(inImage, inKernelSize, 1, inBorder, 0.0F,
		inSigmaSpace, inSigmaColor);
}

oa::Matrix oa::FnImage::unsharpMask(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::F32 inSigma, oa::F32 inAmount, oa::U32 inKernelSize) {
	if (!isValidImage(inImage, "oa::FnImage::unsharpMask") || inSigma <= 0.0F ||
		!oa::isFinite(inSigma) || !oa::isFinite(inAmount)) return inImage;
	auto blurred = gaussianBlur(inRt, inImage, inSigma, inKernelSize);
	auto detail = oa::FnMatrix::sub(inImage, blurred);
	return oa::FnMatrix::add(inImage, oa::FnMatrix::scale(detail, inAmount));
}

oa::Matrix oa::FnImage::morphologyTopHat(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue) {
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::morphologyTopHat")) return inImage;
	auto opened = morphologyOpen(inRt, inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue);
	return oa::FnMatrix::sub(inImage, opened);
}

oa::Matrix oa::FnImage::morphologyBlackHat(oa::Engine& inRt, const oa::Matrix& inImage,
	oa::U32 inKernelWidth, oa::U32 inKernelHeight, oa::BorderMode inBorder,
	oa::F32 inBorderValue) {
	if (!isValidMorphology(inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue, "oa::FnImage::morphologyBlackHat")) return inImage;
	auto closed = morphologyClose(inRt, inImage, inKernelWidth, inKernelHeight,
		inBorder, inBorderValue);
	return oa::FnMatrix::sub(closed, inImage);
}

oa::Matrix oa::FnImage::adaptiveThresholdMean(oa::Engine& inRt,
	const oa::Matrix& inImage, oa::U32 inKernelSize, oa::F32 inC,
	oa::F32 inMaxValue, oa::BorderMode inBorder) {
	(void)inRt;
	if (!isValidNeighborhood(inImage, inKernelSize, inBorder,
		"oa::FnImage::adaptiveThresholdMean") || !oa::isFinite(inC) ||
		!oa::isFinite(inMaxValue)) return inImage;
	return neighborhoodPass(inImage, inKernelSize, 2, inBorder, 0.0F,
		1.0F, inC, inMaxValue);
}

oa::Matrix oa::FnImage::adaptiveThresholdGaussian(oa::Engine& inRt,
	const oa::Matrix& inImage, oa::U32 inKernelSize, oa::F32 inC,
	oa::F32 inMaxValue, oa::F32 inSigma, oa::BorderMode inBorder) {
	(void)inRt;
	if (!isValidNeighborhood(inImage, inKernelSize, inBorder,
		"oa::FnImage::adaptiveThresholdGaussian") || !oa::isFinite(inC) ||
		!oa::isFinite(inMaxValue) || !oa::isFinite(inSigma) || inSigma < 0.0F) return inImage;
	const oa::F32 sigma = inSigma > 0.0F ? inSigma :
		0.3F * ((static_cast<oa::F32>(inKernelSize) - 1.0F) * 0.5F - 1.0F) + 0.8F;
	return neighborhoodPass(inImage, inKernelSize, 3, inBorder, 0.0F,
		sigma, inC, inMaxValue);
}

oa::Matrix oa::FnImage::gaussianBlur(oa::Engine& inRt, const oa::Matrix& inImage,	oa::F32 inSigma, oa::U32 inKernelSize) {
	if (!isValidImage(inImage, "oa::FnImage::gaussianBlur") ||
		!oa::isFinite(inSigma) || inSigma <= 0.0F ||
		(inKernelSize != 0 && (inKernelSize > 31 || (inKernelSize & 1U) == 0))) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::gaussianBlur requires finite positive sigma and odd kernel <=31 (or zero for automatic)");
		return inImage;
	}

	oa::U32 radius = inKernelSize == 0
		? static_cast<oa::U32>(oa::ceil(3.0F * inSigma))
		: inKernelSize / 2U;
	radius = oa::min(radius, 15U);
	const oa::U32 size = radius * 2U + 1U;
	oa::Vector<oa::F32> values(size);
	oa::F64 sum = 0.0;
	for (oa::U32 i = 0; i < size; ++i) {
		const oa::F64 x = static_cast<oa::F64>(static_cast<oa::I32>(i) - static_cast<oa::I32>(radius));
		const oa::F64 value = oa::exp(-(x * x) / (2.0 * inSigma * inSigma));
		values[i] = static_cast<oa::F32>(value);
		sum += value;
	}
	for (oa::U32 i = 0; i < size; ++i) {
		values[i] /= static_cast<oa::F32>(sum);
	}
	auto kernel = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(values.data()),
			values.size() * sizeof(oa::F32)),
		{static_cast<oa::I64>(size)}, inImage.getDtype());
	return separableConvolve2d(inRt, inImage, kernel, kernel, oa::BorderMode::Replicate, 0.0F);
}
