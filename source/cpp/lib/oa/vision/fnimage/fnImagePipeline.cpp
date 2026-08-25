// FnImagePipeline.cpp — high-level image processing pipelines
//
// The schema-generated category translation units own the mechanical
// active-context forwarding overloads for primitive tensor operations.
//
// Future:
// - Video frame batch processing
// - Multi-stage augmentation pipelines

#include <oa/vision/fnImage.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/log.h>
#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/autograd.h>
#include <oa/core/op.h>

namespace {

oa::U32 divCeil(oa::U32 inA, oa::U32 inB)
{
	return (inA + inB - 1U) / inB;
}

} // namespace

namespace oa {

namespace FnImage {
// ─── phase 2: oa::Image overloads ─────────────────────────────────────────

oa::Image resize(const oa::Image& inImage, oa::U32 inWidth, oa::U32 inHeight)
{
	if (inImage.layout() != oa::ImageLayout::Nchw && inImage.layout() != oa::ImageLayout::Chw) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::resize(oa::Image) currently supports Nchw and Chw layouts only");
		return inImage;
	}
	oa::Matrix source = inImage.asMatrix();
	const bool unbatched = inImage.layout() == oa::ImageLayout::Chw;
	if (unbatched) {
		source = oa::FnMatrix::reshape(source,
			oa::MatrixShape{1, inImage.channels(), inImage.height(), inImage.width()});
	}
	oa::Matrix resized = resize(source, inWidth, inHeight);
	if (unbatched) {
		resized = oa::FnMatrix::reshape(resized,
			oa::MatrixShape{inImage.channels(), inHeight, inWidth});
	}
	return oa::Image(std::move(resized), inImage.layout(), inImage.format());
}

oa::Image normalize(
	const oa::Image& inImage,
	const oa::NormalizationParams& inParams)
{
	if (inImage.layout() != oa::ImageLayout::Nchw && inImage.layout() != oa::ImageLayout::Chw) {
		OaLogWarn(oa::LogComponent::Vision,
			"oa::FnImage::normalize(oa::Image) currently supports Nchw and Chw layouts only");
		return inImage;
	}
	oa::Matrix source = inImage.asMatrix();
	const bool unbatched = inImage.layout() == oa::ImageLayout::Chw;
	if (unbatched) {
		source = oa::FnMatrix::reshape(source,
			oa::MatrixShape{1, inImage.channels(), inImage.height(), inImage.width()});
	}
	oa::Matrix normalized = normalize(source, inParams);
	if (unbatched) {
		normalized = oa::FnMatrix::reshape(normalized, inImage.asMatrix().getShape());
	}
	return oa::Image(std::move(normalized), inImage.layout(), inImage.format());
}

oa::Image brightnessContrast(
	const oa::Image& inImage,
	oa::F32 inBrightness,
	oa::F32 inContrast)
{
	if (inImage.layout() != oa::ImageLayout::Nchw
		and inImage.layout() != oa::ImageLayout::Chw) {
		OaLogWarn(
			oa::LogComponent::Vision,
			"oa::FnImage::brightnessContrast(oa::Image) currently supports Nchw and Chw layouts only");
		return inImage;
	}
	return oa::Image(
		brightnessContrast(
			inImage.asMatrix(),
			inBrightness,
			inContrast),
		inImage.layout(),
		inImage.format());
}

oa::Image grayscale(const oa::Image& inImage)
{
	if (inImage.layout() != oa::ImageLayout::Nchw) {
		OaLogError(
			oa::LogComponent::Vision,
			"oa::FnImage::grayscale(oa::Image) requires Nchw layout");
		return {};
	}
	if (inImage.format() != oa::ImageFormat::Rgb
		and inImage.format() != oa::ImageFormat::Rgba) {
		OaLogError(
			oa::LogComponent::Vision,
			"oa::FnImage::grayscale(oa::Image) requires RGB or RGBA input");
		return {};
	}
	return oa::Image(
		grayscale(inImage.asMatrix()),
		inImage.layout(),
		oa::ImageFormat::Gray);
}

oa::Image convertColor(const oa::Image& inImage, oa::ImageFormat inDstFormat)
{
	const bool isIdentity = inImage.format() == inDstFormat;
	const bool isRgbSwap =
		(inImage.format() == oa::ImageFormat::Rgb
			and inDstFormat == oa::ImageFormat::Bgr)
		or (inImage.format() == oa::ImageFormat::Bgr
			and inDstFormat == oa::ImageFormat::Rgb);
	if (not isIdentity and not isRgbSwap) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnImage::convertColor supports identity and RGB/BGR conversion only");
		return {};
	}

	const oa::Matrix& source = inImage.asMatrix();
	oa::I32 channelAxis = -1;
	switch (inImage.layout()) {
		case oa::ImageLayout::Nchw: channelAxis = 1; break;
		case oa::ImageLayout::Nhwc: channelAxis = 3; break;
		case oa::ImageLayout::Chw:  channelAxis = 0; break;
		case oa::ImageLayout::Hwc:  channelAxis = 2; break;
		case oa::ImageLayout::Hw:   break;
	}
	if (isRgbSwap and (inImage.channels() != 3 or channelAxis < 0)) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnImage::convertColor RGB/BGR conversion requires a three-channel layout");
		return {};
	}

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	oa::Matrix result;
	if (isIdentity) {
		// The semantic contract never aliases its input, including the identity
		// case, so callers observe one consistent ownership rule.
		result = oa::FnMatrix::copy(source);
	} else {
		oa::Matrix ch0 = oa::FnMatrix::slice(source, channelAxis, 0, 1);
		oa::Matrix ch1 = oa::FnMatrix::slice(source, channelAxis, 1, 2);
		oa::Matrix ch2 = oa::FnMatrix::slice(source, channelAxis, 2, 3);
		oa::Matrix channels[] = {ch2, ch1, ch0};
		result = oa::FnMatrix::concat(
			oa::Span<oa::Matrix>(channels, 3), channelAxis);
	}
	auto semantic = lowering.commitWithId(
		oa::detail::opRegistry::FnImage::convertColor,
		{&source}, {&result},
		{oa::OpAttribute::fromUnsignedInteger(
			"destinationFormat", static_cast<oa::U64>(inDstFormat))});
	if (not semantic.isOk()) return {};
	if (auto grad = result.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return oa::Image(std::move(result), inImage.layout(), inDstFormat);
}

// ─── phase 3: Fused Preprocess ─────────────────────────────────────────

oa::Image resizeNormalize(const oa::Image& inImage, oa::U32 inWidth, oa::U32 inHeight,
			   const oa::NormalizationParams& inParams)
{
	if (inImage.layout() != oa::ImageLayout::Nchw && inImage.layout() != oa::ImageLayout::Chw) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnImage::resizeNormalize currently supports Nchw and Chw layouts only");
		return {};
	}
	const oa::Matrix& semanticInput = inImage.asMatrix();
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	oa::Matrix src = inImage.asMatrix();
	const bool unbatched = inImage.layout() == oa::ImageLayout::Chw;
	if (unbatched) {
		src = oa::FnMatrix::reshape(src,
			oa::MatrixShape{1, inImage.channels(), inImage.height(), inImage.width()});
	}

	oa::U32 B = inImage.batchSize();
	oa::U32 C = inImage.channels();
	oa::U32 H_in = inImage.height();
	oa::U32 W_in = inImage.width();

	// allocate output tensor
	oa::Matrix result = oa::FnMatrix::empty(oa::MatrixShape{B, C, inHeight, inWidth}, src.getDtype());

	// Record the fused operation in the same graph as its producers/consumers.
	// Mixing deferred oa::FnMatrix nodes with an immediate dispatch here allowed
	// the fused kernel to read stale allocator contents from an unfinished input.
	struct ResizeNormalizePush {
		oa::U32 batchSize, channels, hIn, wIn, hOut, wOut;
		oa::F32 Mean0, Mean1, Mean2, Std0, Std1, Std2;
	} push{B, C, H_in, W_in, inHeight, inWidth,
		inParams.mean[0], inParams.mean[1], inParams.mean[2],
		inParams.std[0], inParams.std[1], inParams.std[2]};
	oa::BufferAccess access[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
	ctx.add( "ResizeNormalizeNchw", {&src, &result}, access, &push, sizeof(push),
		divCeil(inWidth, 16), divCeil(inHeight, 16), B * C);
	if (unbatched) {
		result = oa::FnMatrix::reshape(result,
			oa::MatrixShape{C, inHeight, inWidth});
	}
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnImage::resizeNormalize,
		{&semanticInput}, {&result},
		{
			oa::OpAttribute::fromUnsignedInteger("width", inWidth),
			oa::OpAttribute::fromUnsignedInteger("height", inHeight),
			oa::OpAttribute::fromFloat("mean0", inParams.mean[0]),
			oa::OpAttribute::fromFloat("mean1", inParams.mean[1]),
			oa::OpAttribute::fromFloat("mean2", inParams.mean[2]),
			oa::OpAttribute::fromFloat("std0", inParams.std[0]),
			oa::OpAttribute::fromFloat("std1", inParams.std[1]),
			oa::OpAttribute::fromFloat("std2", inParams.std[2]),
		});
	if (not status.isOk()) return {};

	// Preserve layout and format from input
	return oa::Image(std::move(result), inImage.layout(), inImage.format());
}

} // namespace FnImage

} // namespace oa
