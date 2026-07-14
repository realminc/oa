// FnImageFilter.cpp — graph-native NCHW filtering primitives and semantic ops.

#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB) {
	return (InA + InB - 1U) / InB;
}

bool IsValidImage(const OaMatrix& InImage, OaStringView InOperation) {
	const auto shape = InImage.GetShape();
	if (InImage.HasStorage() && shape.Rank == 4 && shape[0] > 0 && shape[1] > 0 &&
		shape[2] > 0 && shape[3] > 0) {
		return true;
	}
	OA_LOG_WARN(OaLogComponent::Core, "%.*s expects a stored, non-empty [B,C,H,W] tensor",
		static_cast<int>(InOperation.Size()), InOperation.Data());
	return false;
}

bool IsValidBorder(OaBorderMode InBorder) {
	return InBorder == OaBorderMode::Constant ||
		InBorder == OaBorderMode::Replicate ||
		InBorder == OaBorderMode::Reflect ||
		InBorder == OaBorderMode::Reflect101 ||
		InBorder == OaBorderMode::Wrap;
}

bool IsValidKernel2d(const OaMatrix& InImage, const OaMatrix& InKernel) {
	const auto shape = InKernel.GetShape();
	return shape.Rank == 2 && shape[0] > 0 && shape[1] > 0 &&
		shape[0] <= 31 && shape[1] <= 31 &&
		(shape[0] & 1) != 0 && (shape[1] & 1) != 0 &&
		InKernel.GetDtype() == InImage.GetDtype() && InKernel.HasStorage();
}

bool IsValidKernel1d(const OaMatrix& InImage, const OaMatrix& InKernel) {
	const auto shape = InKernel.GetShape();
	const bool vectorShape = shape.Rank == 1 ||
		(shape.Rank == 2 && (shape[0] == 1 || shape[1] == 1));
	const OaI64 size = shape.NumElements();
	return vectorShape && size > 0 && size <= 31 && (size & 1) != 0 &&
		InKernel.GetDtype() == InImage.GetDtype() && InKernel.HasStorage();
}

OaMatrix MakeKernel2d(const OaMatrix& InImage, std::initializer_list<OaF32> InValues) {
	const OaI64 side = static_cast<OaI64>(std::sqrt(static_cast<OaF64>(InValues.size())));
	auto kernel = OaFnMatrix::Empty({side, side}, InImage.GetDtype());
	OaI64 index = 0;
	for (const OaF32 value : InValues) kernel.Set(index++, value);
	return kernel;
}

OaMatrix DerivativeKernel(const OaMatrix& InImage, OaU32 InDx, OaU32 InDy,
	bool InScharr) {
	if (InDx == 1 && InDy == 0) {
		return InScharr
			? MakeKernel2d(InImage, {-3, 0, 3, -10, 0, 10, -3, 0, 3})
			: MakeKernel2d(InImage, {-1, 0, 1, -2, 0, 2, -1, 0, 1});
	}
	return InScharr
		? MakeKernel2d(InImage, {-3, -10, -3, 0, 0, 0, 3, 10, 3})
		: MakeKernel2d(InImage, {-1, -2, -1, 0, 0, 0, 1, 2, 1});
}

bool IsValidMorphology(const OaMatrix& InImage, OaU32 InKernelWidth,
	OaU32 InKernelHeight, OaBorderMode InBorder, OaF32 InBorderValue,
	OaStringView InOperation) {
	if (IsValidImage(InImage, InOperation) &&
		InKernelWidth > 0 && InKernelHeight > 0 &&
		InKernelWidth <= 31 && InKernelHeight <= 31 &&
		(InKernelWidth & 1U) != 0 && (InKernelHeight & 1U) != 0 &&
		IsValidBorder(InBorder) && std::isfinite(InBorderValue)) {
		return true;
	}
	OA_LOG_WARN(OaLogComponent::Core,
		"%.*s requires odd kernel dimensions in [1,31], a valid border, and a finite border value",
		static_cast<int>(InOperation.Size()), InOperation.Data());
	return false;
}

OaMatrix MorphologyPass(const OaMatrix& InImage, OaU32 InKernelWidth,
	OaU32 InKernelHeight, OaBorderMode InBorder, OaF32 InBorderValue,
	bool InDilate) {
	const auto shape = InImage.GetShape();
	const OaU32 batch = static_cast<OaU32>(shape[0]);
	const OaU32 channels = static_cast<OaU32>(shape[1]);
	const OaU32 height = static_cast<OaU32>(shape[2]);
	const OaU32 width = static_cast<OaU32>(shape[3]);
	auto output = OaFnMatrix::Empty(shape, InImage.GetDtype());
	struct Push {
		OaU32 Batch, Channels, Height, Width, KernelHeight, KernelWidth;
		OaU32 Operation, Border;
		OaF32 BorderValue;
	} push{batch, channels, height, width, InKernelHeight, InKernelWidth,
		InDilate ? 1U : 0U, static_cast<OaU32>(InBorder), InBorderValue};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageMorphology", {&InImage, &output}, access,
		&push, sizeof(push), DivCeil(width, 16), DivCeil(height, 16), batch * channels);
	return output;
}

OaMatrix NeighborhoodPass(const OaMatrix& InImage, OaU32 InKernelSize,
	OaU32 InOperation, OaBorderMode InBorder, OaF32 InBorderValue,
	OaF32 InP0 = 0.0F, OaF32 InP1 = 0.0F, OaF32 InP2 = 0.0F) {
	const auto shape = InImage.GetShape();
	auto output = OaFnMatrix::Empty(shape, InImage.GetDtype());
	struct Push {
		OaU32 Batch, Channels, Height, Width, KernelSize, Operation, Border;
		OaF32 BorderValue, P0, P1, P2;
	};
	Push push{static_cast<OaU32>(shape[0]), static_cast<OaU32>(shape[1]),
		static_cast<OaU32>(shape[2]), static_cast<OaU32>(shape[3]),
		InKernelSize, InOperation, static_cast<OaU32>(InBorder),
		InBorderValue, InP0, InP1, InP2};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageNeighborhood", {&InImage, &output}, access,
		&push, sizeof(push), DivCeil(push.Width, 16), DivCeil(push.Height, 16),
		push.Batch * push.Channels);
	return output;
}

bool IsValidNeighborhood(const OaMatrix& InImage, OaU32 InKernelSize,
	OaBorderMode InBorder, OaStringView InOperation) {
	if (IsValidImage(InImage, InOperation) && InKernelSize > 0 &&
		InKernelSize <= 15 && (InKernelSize & 1U) != 0 && IsValidBorder(InBorder)) return true;
	OA_LOG_WARN(OaLogComponent::Core,
		"%.*s requires an odd kernel in [1,15] and a valid border",
		static_cast<int>(InOperation.Size()), InOperation.Data());
	return false;
}

} // namespace

OaMatrix OaFnImage::Convolve2d(
	OaEngine& InRt,
	const OaMatrix& InImage,
	const OaMatrix& InKernel,
	OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	(void)InRt;
	if (!IsValidImage(InImage, "OaFnImage::Convolve2d") ||
		!IsValidKernel2d(InImage, InKernel) || !IsValidBorder(InBorder) ||
		!std::isfinite(InBorderValue)) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::Convolve2d requires an odd <=31 rank-2 kernel with matching dtype and a valid border");
		return InImage;
	}

	const auto imageShape = InImage.GetShape();
	const auto kernelShape = InKernel.GetShape();
	const OaU32 batch = static_cast<OaU32>(imageShape[0]);
	const OaU32 channels = static_cast<OaU32>(imageShape[1]);
	const OaU32 height = static_cast<OaU32>(imageShape[2]);
	const OaU32 width = static_cast<OaU32>(imageShape[3]);
	auto output = OaFnMatrix::Empty(imageShape, InImage.GetDtype());

	struct Push {
		OaU32 Batch, Channels, Height, Width, KernelHeight, KernelWidth, Border;
		OaF32 BorderValue;
	} push{batch, channels, height, width,
		static_cast<OaU32>(kernelShape[0]), static_cast<OaU32>(kernelShape[1]),
		static_cast<OaU32>(InBorder), InBorderValue};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	OaContext::GetDefault().Add("ImageConvolve2d", {&InImage, &InKernel, &output},
		access, &push, sizeof(push), DivCeil(width, 16), DivCeil(height, 16), batch * channels);
	return output;
}

OaMatrix OaFnImage::SeparableConvolve2d(
	OaEngine& InRt,
	const OaMatrix& InImage,
	const OaMatrix& InKernelX,
	const OaMatrix& InKernelY,
	OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	(void)InRt;
	if (!IsValidImage(InImage, "OaFnImage::SeparableConvolve2d") ||
		!IsValidKernel1d(InImage, InKernelX) || !IsValidKernel1d(InImage, InKernelY) ||
		!IsValidBorder(InBorder) || !std::isfinite(InBorderValue)) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::SeparableConvolve2d requires odd <=31 vector kernels with matching dtype and a valid border");
		return InImage;
	}

	const auto shape = InImage.GetShape();
	const OaU32 batch = static_cast<OaU32>(shape[0]);
	const OaU32 channels = static_cast<OaU32>(shape[1]);
	const OaU32 height = static_cast<OaU32>(shape[2]);
	const OaU32 width = static_cast<OaU32>(shape[3]);
	auto temporary = OaFnMatrix::Empty(shape, InImage.GetDtype());
	auto output = OaFnMatrix::Empty(shape, InImage.GetDtype());

	struct Push {
		OaU32 Batch, Channels, Height, Width, KernelSize, Horizontal, Border;
		OaF32 BorderValue;
	};
	Push horizontal{batch, channels, height, width,
		static_cast<OaU32>(InKernelX.NumElements()), 1U,
		static_cast<OaU32>(InBorder), InBorderValue};
	Push vertical{batch, channels, height, width,
		static_cast<OaU32>(InKernelY.NumElements()), 0U,
		static_cast<OaU32>(InBorder), InBorderValue};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Read, OaBufferAccess::Write};
	auto& context = OaContext::GetDefault();
	context.Add("ImageSeparableConvolve", {&InImage, &InKernelX, &temporary},
		access, &horizontal, sizeof(horizontal),
		DivCeil(width, 16), DivCeil(height, 16), batch * channels);
	context.Add("ImageSeparableConvolve", {&temporary, &InKernelY, &output},
		access, &vertical, sizeof(vertical),
		DivCeil(width, 16), DivCeil(height, 16), batch * channels);
	return output;
}

OaMatrix OaFnImage::AverageBlur(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder)
{
	if (!IsValidImage(InImage, "OaFnImage::AverageBlur") ||
		InKernelWidth == 0 || InKernelHeight == 0 ||
		InKernelWidth > 31 || InKernelHeight > 31 ||
		(InKernelWidth & 1U) == 0 || (InKernelHeight & 1U) == 0) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::AverageBlur kernel dimensions must be odd and in [1,31]");
		return InImage;
	}
	const OaF64 weight = 1.0 / static_cast<OaF64>(InKernelWidth * InKernelHeight);
	auto kernel = OaFnMatrix::Full(
		{static_cast<OaI64>(InKernelHeight), static_cast<OaI64>(InKernelWidth)},
		weight, InImage.GetDtype());
	return Convolve2d(InRt, InImage, kernel, InBorder, 0.0F);
}

OaMatrix OaFnImage::Sobel(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InDx, OaU32 InDy, OaBorderMode InBorder)
{
	if (!((InDx == 1 && InDy == 0) || (InDx == 0 && InDy == 1))) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::Sobel requires derivative (dx,dy) to be (1,0) or (0,1)");
		return InImage;
	}
	auto kernel = DerivativeKernel(InImage, InDx, InDy, false);
	return Convolve2d(InRt, InImage, kernel, InBorder, 0.0F);
}

OaMatrix OaFnImage::Scharr(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InDx, OaU32 InDy, OaBorderMode InBorder)
{
	if (!((InDx == 1 && InDy == 0) || (InDx == 0 && InDy == 1))) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::Scharr requires derivative (dx,dy) to be (1,0) or (0,1)");
		return InImage;
	}
	auto kernel = DerivativeKernel(InImage, InDx, InDy, true);
	return Convolve2d(InRt, InImage, kernel, InBorder, 0.0F);
}

OaMatrix OaFnImage::Laplacian(OaEngine& InRt, const OaMatrix& InImage,
	OaBorderMode InBorder)
{
	auto kernel = MakeKernel2d(InImage, {0, 1, 0, 1, -4, 1, 0, 1, 0});
	return Convolve2d(InRt, InImage, kernel, InBorder, 0.0F);
}

OaMatrix OaFnImage::Sharpen(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InAmount, OaBorderMode InBorder) {
	if (!std::isfinite(InAmount) || InAmount < 0.0F) return InImage;
	auto kernel = MakeKernel2d(InImage, {0, -InAmount, 0,
		-InAmount, 1.0F + 4.0F * InAmount, -InAmount,
		0, -InAmount, 0});
	return Convolve2d(InRt, InImage, kernel, InBorder, 0.0F);
}

OaMatrix OaFnImage::Erode(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	(void)InRt;
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::Erode")) return InImage;
	return MorphologyPass(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, false);
}

OaMatrix OaFnImage::Dilate(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	(void)InRt;
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::Dilate")) return InImage;
	return MorphologyPass(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, true);
}

OaMatrix OaFnImage::MorphologyOpen(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::MorphologyOpen")) return InImage;
	auto eroded = MorphologyPass(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, false);
	return Dilate(InRt, eroded, InKernelWidth, InKernelHeight, InBorder, InBorderValue);
}

OaMatrix OaFnImage::MorphologyClose(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::MorphologyClose")) return InImage;
	auto dilated = MorphologyPass(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, true);
	return Erode(InRt, dilated, InKernelWidth, InKernelHeight, InBorder, InBorderValue);
}

OaMatrix OaFnImage::MorphologyGradient(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue)
{
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::MorphologyGradient")) return InImage;
	auto dilated = Dilate(InRt, InImage, InKernelWidth, InKernelHeight, InBorder, InBorderValue);
	auto eroded = Erode(InRt, InImage, InKernelWidth, InKernelHeight, InBorder, InBorderValue);
	return OaFnMatrix::Sub(dilated, eroded);
}

OaMatrix OaFnImage::MedianBlur(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelSize, OaBorderMode InBorder) {
	(void)InRt;
	if (!IsValidNeighborhood(InImage, InKernelSize, InBorder,
		"OaFnImage::MedianBlur")) return InImage;
	return NeighborhoodPass(InImage, InKernelSize, 0, InBorder, 0.0F);
}

OaMatrix OaFnImage::BilateralFilter(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelSize, OaF32 InSigmaColor, OaF32 InSigmaSpace,
	OaBorderMode InBorder) {
	(void)InRt;
	if (!IsValidNeighborhood(InImage, InKernelSize, InBorder,
		"OaFnImage::BilateralFilter") || InSigmaColor <= 0.0F ||
		InSigmaSpace <= 0.0F || !std::isfinite(InSigmaColor) ||
		!std::isfinite(InSigmaSpace)) return InImage;
	return NeighborhoodPass(InImage, InKernelSize, 1, InBorder, 0.0F,
		InSigmaSpace, InSigmaColor);
}

OaMatrix OaFnImage::UnsharpMask(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InSigma, OaF32 InAmount, OaU32 InKernelSize) {
	if (!IsValidImage(InImage, "OaFnImage::UnsharpMask") || InSigma <= 0.0F ||
		!std::isfinite(InSigma) || !std::isfinite(InAmount)) return InImage;
	auto blurred = GaussianBlur(InRt, InImage, InSigma, InKernelSize);
	auto detail = OaFnMatrix::Sub(InImage, blurred);
	return OaFnMatrix::Add(InImage, OaFnMatrix::Scale(detail, InAmount));
}

OaMatrix OaFnImage::MorphologyTopHat(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue) {
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::MorphologyTopHat")) return InImage;
	auto opened = MorphologyOpen(InRt, InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue);
	return OaFnMatrix::Sub(InImage, opened);
}

OaMatrix OaFnImage::MorphologyBlackHat(OaEngine& InRt, const OaMatrix& InImage,
	OaU32 InKernelWidth, OaU32 InKernelHeight, OaBorderMode InBorder,
	OaF32 InBorderValue) {
	if (!IsValidMorphology(InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue, "OaFnImage::MorphologyBlackHat")) return InImage;
	auto closed = MorphologyClose(InRt, InImage, InKernelWidth, InKernelHeight,
		InBorder, InBorderValue);
	return OaFnMatrix::Sub(closed, InImage);
}

OaMatrix OaFnImage::AdaptiveThresholdMean(OaEngine& InRt,
	const OaMatrix& InImage, OaU32 InKernelSize, OaF32 InC,
	OaF32 InMaxValue, OaBorderMode InBorder) {
	(void)InRt;
	if (!IsValidNeighborhood(InImage, InKernelSize, InBorder,
		"OaFnImage::AdaptiveThresholdMean") || !std::isfinite(InC) ||
		!std::isfinite(InMaxValue)) return InImage;
	return NeighborhoodPass(InImage, InKernelSize, 2, InBorder, 0.0F,
		1.0F, InC, InMaxValue);
}

OaMatrix OaFnImage::AdaptiveThresholdGaussian(OaEngine& InRt,
	const OaMatrix& InImage, OaU32 InKernelSize, OaF32 InC,
	OaF32 InMaxValue, OaF32 InSigma, OaBorderMode InBorder) {
	(void)InRt;
	if (!IsValidNeighborhood(InImage, InKernelSize, InBorder,
		"OaFnImage::AdaptiveThresholdGaussian") || !std::isfinite(InC) ||
		!std::isfinite(InMaxValue) || !std::isfinite(InSigma) || InSigma < 0.0F) return InImage;
	const OaF32 sigma = InSigma > 0.0F ? InSigma :
		0.3F * ((static_cast<OaF32>(InKernelSize) - 1.0F) * 0.5F - 1.0F) + 0.8F;
	return NeighborhoodPass(InImage, InKernelSize, 3, InBorder, 0.0F,
		sigma, InC, InMaxValue);
}

OaMatrix OaFnImage::GaussianBlur(OaEngine& InRt, const OaMatrix& InImage,
	OaF32 InSigma, OaU32 InKernelSize)
{
	if (!IsValidImage(InImage, "OaFnImage::GaussianBlur") ||
		!std::isfinite(InSigma) || InSigma <= 0.0F ||
		(InKernelSize != 0 && (InKernelSize > 31 || (InKernelSize & 1U) == 0))) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::GaussianBlur requires finite positive sigma and odd kernel <=31 (or zero for automatic)");
		return InImage;
	}

	OaU32 radius = InKernelSize == 0
		? static_cast<OaU32>(std::ceil(3.0F * InSigma))
		: InKernelSize / 2U;
	radius = std::min(radius, 15U);
	const OaU32 size = radius * 2U + 1U;
	auto kernel = OaFnMatrix::Empty({static_cast<OaI64>(size)}, InImage.GetDtype());
	OaF64 sum = 0.0;
	for (OaU32 i = 0; i < size; ++i) {
		const OaF64 x = static_cast<OaF64>(static_cast<OaI32>(i) - static_cast<OaI32>(radius));
		const OaF64 value = std::exp(-(x * x) / (2.0 * InSigma * InSigma));
		kernel.Set(i, static_cast<OaF32>(value));
		sum += value;
	}
	for (OaU32 i = 0; i < size; ++i) kernel.Set(i, kernel.At(i) / static_cast<OaF32>(sum));
	return SeparableConvolve2d(InRt, InImage, kernel, kernel, OaBorderMode::Replicate, 0.0F);
}
