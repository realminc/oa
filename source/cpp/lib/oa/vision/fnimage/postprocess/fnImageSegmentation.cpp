#include <oa/vision/fnImage.h>

#include <oa/core/bufferAccess.h>
#include <oa/core/log.h>
#include <oa/core/op.h>
#include <oa/runtime/executionSession.h>

#include <cmath>
#include <limits>

oa::Matrix oa::FnImage::segmentationOverlay(
	oa::Engine& inRt,
	const oa::Matrix& inImage,
	const oa::Matrix& inMask,
	const oa::Matrix& inPalette,
	oa::F32 inAlpha
) {
	auto& context = oa::ExecutionSession::forEngine(inRt);
	oa::ExecutionSession::RecordingScope recording(context);
	const bool imageValid = inImage.rank() == 4
		&& (inImage.size(1) == 3 || inImage.size(1) == 4)
		&& inImage.getDtype() == oa::ScalarType::Float32;
	const bool maskValid = inMask.getDtype() == oa::ScalarType::Int32
		&& ((inMask.rank() == 3
			&& inMask.size(0) == inImage.size(0)
			&& inMask.size(1) == inImage.size(2)
			&& inMask.size(2) == inImage.size(3))
		|| (inMask.rank() == 4 && inMask.size(1) == 1
			&& inMask.size(0) == inImage.size(0)
			&& inMask.size(2) == inImage.size(2)
			&& inMask.size(3) == inImage.size(3)));
	const bool paletteValid = inPalette.rank() == 2
		&& inPalette.size(0) > 0 && inPalette.size(1) == 3
		&& inPalette.getDtype() == oa::ScalarType::Float32;
	const oa::U64 elements = static_cast<oa::U64>(inImage.numElements());
	if (!imageValid || !maskValid || !paletteValid
		|| !std::isfinite(inAlpha) || inAlpha < 0.0F || inAlpha > 1.0F
		|| elements > std::numeric_limits<oa::U32>::max()) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnImage::segmentationOverlay expects FP32 NCHW RGB/RGBA, matching Int32 labels, FP32 palette [K,3], and alpha in [0,1]");
		return {};
	}
	oa::Matrix out = oa::FnMatrix::empty(inImage.getShape(), oa::ScalarType::Float32);
	struct Push {
		oa::U32 batch, channels, height, width, classes;
		oa::F32 alpha;
	} push{
		static_cast<oa::U32>(inImage.size(0)),
		static_cast<oa::U32>(inImage.size(1)),
		static_cast<oa::U32>(inImage.size(2)),
		static_cast<oa::U32>(inImage.size(3)),
		static_cast<oa::U32>(inPalette.size(0)), inAlpha};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read, oa::BufferAccess::Read,
		oa::BufferAccess::Read, oa::BufferAccess::Write};
	oa::OpLoweringScope lowering(context);
	context.add( "ImageSegmentationOverlay",
		{&inImage, &inMask, &inPalette, &out}, access,
		&push, sizeof(push), (static_cast<oa::U32>(elements) + 255U) / 256U);
	const auto status = lowering.commit(
		oa::detail::opRegistry::FnImage::segmentationOverlay,
		{&inImage, &inMask, &inPalette}, {&out},
		{oa::OpAttribute::fromFloat("alpha", inAlpha)});
	if (not status.isOk()) return {};
	return out;
}

oa::Matrix oa::FnImage::segmentationOverlay(
	const oa::Matrix& inImage,
	const oa::Matrix& inMask,
	const oa::Matrix& inPalette,
	oa::F32 inAlpha) {
	auto* context = oa::ExecutionSession::getActivePtr();
	if (context == nullptr) {
		OaLogError(oa::LogComponent::Vision,
			"oa::FnImage::segmentationOverlay requires an active engine context");
		return {};
	}
	return segmentationOverlay(
		context->engine(), inImage, inMask, inPalette, inAlpha);
}
