// oa::VideoDecoder high-level session coordination. The owning engine selects
// the matching private producer recorder; Runtime does not know decoder types.

#include <oa/vision/videoDecoder.h>
#include <oa/runtime/executionSession.h>

#include "videoDecoderInternal.h"

static oa::Status decodeRecorded(
	oa::ExecutionSession& inContext,
	oa::VideoDecoder& inSession,
	const oa::Span<const oa::U8>& inAccessUnit,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outFrame,
	oa::U64 inPts)
{
#ifdef OA_ANDROID_ML
	(void)inContext;
	(void)inSession;
	(void)inAccessUnit;
	(void)inOptions;
	(void)inPts;
	(void)outFrame;
	return oa::Status::unimplemented(
		"Video decoding is not part of the android ML profile");
#else
	if (not inSession.isInitialized()) {
		return oa::Status::error(
			"oa::VideoDecoder::decode: decoder session not initialized");
	}
	if (inSession.getEngine() != &inContext.engine()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoDecoder::decode: decoder belongs to a different engine");
	}

	// Decode submits outside the compute graph. drain only when recorded work
	// may produce decoder input or share resources with the session.
	if (inContext.nodeCount() > 0) {
		OA_RETURN_IF_ERROR(inContext.submitAndWait());
	}

	oa::Status status = oa::VideoDecoderInternal::decodeFrameWithConversion(
		inSession, inAccessUnit, inOptions, outFrame);
	if (status.isOk()) outFrame.presentationTimestamp = inPts;
	return status;
#endif
}

static oa::Result<oa::VideoFrame> decodeRecorded(
	oa::ExecutionSession& inContext,
	oa::VideoDecoder& inSession,
	const oa::Span<const oa::U8>& inAccessUnit,
	const oa::VideoConversionOptions& inOptions,
	oa::U64 inPts)
{
	oa::VideoFrame frame = {};
	oa::Status status = decodeRecorded(
		inContext, inSession, inAccessUnit, inOptions, frame, inPts);
	if (not status.isOk()) return status;
	return frame;
}

static oa::Result<oa::ExecutionSession*> decoderContext(oa::VideoDecoder& inSession) {
	auto* engine = inSession.getEngine();
	if (engine == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::VideoDecoder::decode: decoder session not initialized");
	}
	return &oa::ExecutionSession::forEngine(*engine);
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::decode(
	const oa::Span<const oa::U8>& inAccessUnit,
	oa::U64 inPts)
{
	oa::VideoConversionOptions options = {};
	options.convertToRgb = false;
	auto context = decoderContext(*this);
	if (not context.isOk()) return context.getStatus();
	return decodeRecorded(
		**context, *this, inAccessUnit, options, inPts);
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::decode(
	const oa::Span<const oa::U8>& inAccessUnit,
	const oa::VideoConversionOptions& inOptions,
	oa::U64 inPts)
{
	auto context = decoderContext(*this);
	if (not context.isOk()) return context.getStatus();
	return decodeRecorded(
		**context, *this, inAccessUnit, inOptions, inPts);
}

oa::Status oa::VideoDecoder::decode(
	const oa::Span<const oa::U8>& inAccessUnit,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& outFrame,
	oa::U64 inPts)
{
	auto context = decoderContext(*this);
	if (not context.isOk()) return context.getStatus();
	return decodeRecorded(
		**context, *this, inAccessUnit, inOptions, outFrame, inPts);
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::convert(
	const oa::VideoFrame& inFrame,
	const oa::VideoConversionOptions& inOptions)
{
	if (inFrame.isRgb) {
		return inFrame;
	}

	oa::VideoFrame frame = {};
	oa::Status status = convertFrameToRgba(inFrame, inOptions, frame);
	if (not status.isOk()) {
		return status;
	}
	return frame;
}

oa::Result<oa::VideoFrame> oa::VideoDecoder::allocateRgbaFrame(
	oa::U32 inWidth,
	oa::U32 inHeight)
{
	return allocateRgbaFrame_(inWidth, inHeight, 0ULL);
}

oa::Status oa::VideoDecoder::convertInto(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	oa::VideoFrame& inOutRgbTarget)
{
	return convertNv12ToRgbInto(inNv12Frame, inOptions, inOutRgbTarget);
}

oa::Result<oa::Event> oa::VideoDecoder::convertIntoAsync(
	const oa::VideoFrame& inNv12Frame,
	const oa::VideoConversionOptions& inOptions,
	const oa::VideoFrame& inRgbTarget)
{
	return convertNv12ToRgbIntoAsync(inNv12Frame, inOptions, inRgbTarget);
}
