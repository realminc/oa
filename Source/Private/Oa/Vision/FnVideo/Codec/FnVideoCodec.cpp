// OaFnVideo — VideoCodec category.
// DecodeFrame / EncodeFrame / FlushSession session-recorder shims.
//
// OaFnVideo owns media-session coordination. OaContext is accepted only as an
// explicit dependency boundary for pending compute producers; Runtime does not
// know about decoder or encoder types.

#include <Oa/Vision/FnVideo.h>
#include <Oa/Runtime/Context.h>

namespace OaFnVideo
{

OaResult<OaVideoFrame> Decode(
	OaContext& InContext,
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	OaU64 InPts)
{
	OaVideoConversionOptions options = {};
	options.ConvertToRgb = false;
	return Decode(
		InContext,
		InSession,
		InAccessUnit,
		options,
		InPts);
}

OaResult<OaVideoFrame> Decode(
	OaContext& InContext,
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	const OaVideoConversionOptions& InOptions,
	OaU64 InPts)
{
	OaVideoFrame frame = {};
	OaStatus status = Decode(
		InContext,
		InSession,
		InAccessUnit,
		InOptions,
		frame,
		InPts);
	if (not status.IsOk()) {
		return status;
	}
	return frame;
}

OaStatus Decode(
	OaContext& InContext,
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutFrame,
	OaU64 InPts)
{
#ifdef OA_ANDROID_ML
	(void)InContext;
	(void)InSession;
	(void)InAccessUnit;
	(void)InOptions;
	(void)InPts;
	(void)OutFrame;
	return OaStatus::Unimplemented(
		"Video decoding is not part of the Android ML profile");
#else
	if (not InSession.IsInitialized()) {
		return OaStatus::Error(
			"OaFnVideo::Decode: decoder session not initialized");
	}
	if (InSession.GetEngine() != &InContext.Engine()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::Decode: decoder belongs to a different engine");
	}

	// Decode submits outside the compute graph. Drain only when recorded work
	// may produce decoder input or share resources with the session.
	if (InContext.NodeCount() > 0) {
		OA_RETURN_IF_ERROR(InContext.Execute());
	}
	if (InContext.IsAsyncBatchActive()) {
		OA_RETURN_IF_ERROR(InContext.Sync());
	}

	OaStatus status = InSession.DecodeFrameWithConversion(
		InAccessUnit, InOptions, OutFrame);
	if (status.IsOk()) OutFrame.PresentationTimestamp = InPts;
	return status;
#endif
}

OaResult<OaVideoFrame> Decode(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	OaU64 InPts)
{
	OaVideoConversionOptions options = {};
	options.ConvertToRgb = false;
	return Decode(
		OaContext::GetDefault(),
		InSession,
		InAccessUnit,
		options,
		InPts);
}

OaResult<OaVideoFrame> Decode(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	const OaVideoConversionOptions& InOptions,
	OaU64 InPts)
{
	return Decode(
		OaContext::GetDefault(),
		InSession,
		InAccessUnit,
		InOptions,
		InPts);
}

OaStatus DecodeFrame(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	OaVideoFrame& OutFrame,
	OaU64 InPts)
{
	OaVideoConversionOptions options = {};
	options.ConvertToRgb = false;
	return Decode(
		OaContext::GetDefault(),
		InSession,
		InAccessUnit,
		options,
		OutFrame,
		InPts);
}

OaStatus DecodeFrame(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutFrame,
	OaU64 InPts)
{
	return Decode(
		OaContext::GetDefault(),
		InSession,
		InAccessUnit,
		InOptions,
		OutFrame,
		InPts);
}

OaStatus EncodeFrame(
	OaVideoEncoder& InSession,
	VkImage InImage,
	OaEncodedFrame& OutFrame,
	OaU64 InPts)
{
	return InSession.EncodeFrame(InImage, InPts, OutFrame);
}

OaStatus FlushSession(OaVideoDecoder& InSession)
{
	return InSession.Flush();
}

OaStatus FlushSession(OaVideoEncoder& InSession, OaVec<OaEncodedFrame>& OutFrames)
{
	return InSession.Flush(OutFrames);
}

} // namespace OaFnVideo
