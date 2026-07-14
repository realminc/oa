// OaFnVideo — VideoCodec category.
// DecodeFrame / EncodeFrame / FlushSession session-recorder shims.
//
// Decode routes through OaContext::RecordDecode so pending compute work is
// drained only when a real cross-domain dependency exists.
// EncodeFrame mirrors that via the existing RecordEncode pattern.

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
	return InContext.RecordDecode(
		InSession,
		InAccessUnit,
		InOptions,
		InPts,
		OutFrame);
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
