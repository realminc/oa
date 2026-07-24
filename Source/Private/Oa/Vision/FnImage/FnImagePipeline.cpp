// FnImagePipeline.cpp — High-level image processing pipelines
//
// Implements:
// - OaFnImage::ResizeBatch          — Batch resize operation
//
// The schema-generated category translation units own the mechanical
// global-engine forwarding overloads for primitive tensor operations.
//
// Future:
// - Video frame batch processing
// - Multi-stage augmentation pipelines

#include <Oa/Vision/FnImage.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>

namespace {

OaU32 DivCeil(OaU32 InA, OaU32 InB)
{
	return (InA + InB - 1U) / InB;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// OaFnImage:: Pipeline Operations
// ═══════════════════════════════════════════════════════════════════════════════

OaVec<OaMatrix> OaFnImage::ResizeBatch(
    OaEngine& InRt,
    const OaVec<OaMatrix>& InImages,
    OaU32 InTargetWidth,
    OaU32 InTargetHeight,
    OaInterpolationMode InMode)
{
    OaVec<OaMatrix> result;
    result.Reserve(InImages.Size());

    for (const auto& img : InImages)
    {
        result.PushBack(Resize(InRt, img, InTargetWidth, InTargetHeight, InMode));
    }

    return result;
}

namespace OaFnImage {
// ─── Phase 2: OaImage overloads ─────────────────────────────────────────

OaImage Resize(const OaImage& InImage, OaU32 InWidth, OaU32 InHeight)
{
	if (InImage.Layout() != OaImageLayout::Nchw && InImage.Layout() != OaImageLayout::Chw) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::Resize(OaImage) currently supports Nchw and Chw layouts only");
		return InImage;
	}
	OaMatrix source = InImage.AsMatrix();
	const bool unbatched = InImage.Layout() == OaImageLayout::Chw;
	if (unbatched) {
		source = OaFnMatrix::Reshape(source,
			OaMatrixShape{1, InImage.Channels(), InImage.Height(), InImage.Width()});
	}
	OaMatrix resized = Resize(source, InWidth, InHeight);
	if (unbatched) {
		resized = OaFnMatrix::Reshape(resized,
			OaMatrixShape{InImage.Channels(), InHeight, InWidth});
	}
	return OaImage(std::move(resized), InImage.Layout(), InImage.Format());
}

OaImage Normalize(const OaImage& InImage, const OaNormalizationParams& InParams)
{
	if (InImage.Layout() != OaImageLayout::Nchw && InImage.Layout() != OaImageLayout::Chw) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::Normalize(OaImage) currently supports Nchw and Chw layouts only");
		return InImage;
	}
	OaMatrix source = InImage.AsMatrix();
	const bool unbatched = InImage.Layout() == OaImageLayout::Chw;
	if (unbatched) {
		source = OaFnMatrix::Reshape(source,
			OaMatrixShape{1, InImage.Channels(), InImage.Height(), InImage.Width()});
	}
	OaMatrix normalized = Normalize(source, InParams);
	if (unbatched) {
		normalized = OaFnMatrix::Reshape(normalized, InImage.AsMatrix().GetShape());
	}
	return OaImage(std::move(normalized), InImage.Layout(), InImage.Format());
}

OaImage BrightnessContrast(
	const OaImage& InImage,
	OaF32 InBrightness,
	OaF32 InContrast)
{
	if (InImage.Layout() != OaImageLayout::Nchw
		and InImage.Layout() != OaImageLayout::Chw) {
		OA_LOG_WARN(
			OaLogComponent::Core,
			"OaFnImage::BrightnessContrast(OaImage) currently supports Nchw and Chw layouts only");
		return InImage;
	}
	return OaImage(
		BrightnessContrast(
			InImage.AsMatrix(),
			InBrightness,
			InContrast),
		InImage.Layout(),
		InImage.Format());
}

OaImage ConvertColor(const OaImage& InImage, OaImageFormat InDstFormat)
{
	if (InImage.Format() == InDstFormat) {
		return InImage;  // No conversion needed
	}

	// For now, we only support RGB↔BGR swaps via channel reordering
	// This is a placeholder - full color conversion will be added in Phase 3
	// with the fused preprocess resolver
	if ((InImage.Format() == OaImageFormat::Rgb && InDstFormat == OaImageFormat::Bgr) ||
	    (InImage.Format() == OaImageFormat::Bgr && InDstFormat == OaImageFormat::Rgb)) {
		// Swap channels R and B using OaFnMatrix operations
		const OaMatrix& src = InImage.AsMatrix();
		const OaI32 C = InImage.Channels();
		if (C != 3) {
			// Invalid for RGB/BGR swap
			return InImage;
		}

		OaI32 channelAxis = -1;
		switch (InImage.Layout()) {
			case OaImageLayout::Nchw: channelAxis = 1; break;
			case OaImageLayout::Nhwc: channelAxis = 3; break;
			case OaImageLayout::Chw:  channelAxis = 0; break;
			case OaImageLayout::Hwc:  channelAxis = 2; break;
			case OaImageLayout::Hw:   break;
		}
		if (channelAxis < 0) {
			return InImage;  // Unsupported layout
		}
		OaMatrix ch0 = OaFnMatrix::Slice(src, channelAxis, 0, 1);
		OaMatrix ch1 = OaFnMatrix::Slice(src, channelAxis, 1, 2);
		OaMatrix ch2 = OaFnMatrix::Slice(src, channelAxis, 2, 3);

		// Swap R and B channels
		OaMatrix channels[] = {ch2, ch1, ch0};
		OaMatrix result = OaFnMatrix::Concat(OaSpan<OaMatrix>(channels, 3), channelAxis);
		return OaImage(std::move(result), InImage.Layout(), InDstFormat);
	}

	// Other conversions not yet implemented
	return InImage;
}

// ─── Phase 3: Fused Preprocess ─────────────────────────────────────────

OaImage ResizeNormalize(const OaImage& InImage, OaU32 InWidth, OaU32 InHeight,
                       const OaNormalizationParams& InParams)
{
	if (InImage.Layout() != OaImageLayout::Nchw && InImage.Layout() != OaImageLayout::Chw) {
		OA_LOG_WARN(OaLogComponent::Core,
			"OaFnImage::ResizeNormalize currently supports Nchw and Chw layouts only");
		return InImage;
	}
	OaMatrix src = InImage.AsMatrix();
	const bool unbatched = InImage.Layout() == OaImageLayout::Chw;
	if (unbatched) {
		src = OaFnMatrix::Reshape(src,
			OaMatrixShape{1, InImage.Channels(), InImage.Height(), InImage.Width()});
	}

	OaU32 B = InImage.BatchSize();
	OaU32 C = InImage.Channels();
	OaU32 H_in = InImage.Height();
	OaU32 W_in = InImage.Width();

	// Allocate output tensor
	OaMatrix result = OaFnMatrix::Empty(OaMatrixShape{B, C, InHeight, InWidth}, src.GetDtype());

	// Record the fused operation in the same graph as its producers/consumers.
	// Mixing deferred OaFnMatrix nodes with an immediate dispatch here allowed
	// the fused kernel to read stale allocator contents from an unfinished input.
	struct ResizeNormalizePush {
		OaU32 BatchSize, Channels, HIn, WIn, HOut, WOut;
		OaF32 Mean0, Mean1, Mean2, Std0, Std1, Std2;
	} push{B, C, H_in, W_in, InHeight, InWidth,
		InParams.Mean[0], InParams.Mean[1], InParams.Mean[2],
		InParams.Std[0], InParams.Std[1], InParams.Std[2]};
	OaBufferAccess access[] = {OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("ResizeNormalizeNchw", {&src, &result}, access, &push, sizeof(push),
		DivCeil(InWidth, 16), DivCeil(InHeight, 16), B * C);
	if (unbatched) {
		result = OaFnMatrix::Reshape(result,
			OaMatrixShape{C, InHeight, InWidth});
	}

	// Preserve layout and format from input
	return OaImage(std::move(result), InImage.Layout(), InImage.Format());
}

} // namespace OaFnImage
