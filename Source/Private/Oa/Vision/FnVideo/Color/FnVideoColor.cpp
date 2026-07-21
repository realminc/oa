// OaFnVideo — VideoColor category (public verbs).
// Implementation: FnVideoDecoderColor.cpp (decoder readback/convert),
// FnVideoColor.gen.cpp (generated kernels), encoder upload in VideoEncoder.

#include <Oa/Runtime/Engine.h>

#include <Oa/Vision/FnVideo.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Ui/Image.h>
#include "Oa/Vision/Video/Decoder/VideoDecoderInternal.h"

namespace OaFnVideo
{

OaResult<OaVideoFrame> FromTexture(
	const OaTexture& InTexture,
	OaU64 InPts,
	OaEvent InReady)
{
	if (not InTexture.IsValid() or InTexture.Width <= 0 or InTexture.Height <= 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::FromTexture requires a valid texture with positive extent");
	}
	OaVideoFrame frame = {};
	frame.Width = static_cast<OaU32>(InTexture.Width);
	frame.Height = static_cast<OaU32>(InTexture.Height);
	frame.PresentationTimestamp = InPts;
	frame.IsRgb = true;
	frame.ColorSpace = OaYCbCrModel::BT709;
	frame.FullRange = true;
	frame.Ready = InReady;
	if (InTexture.IsImageBacked()) {
		if (InTexture.View == nullptr
			or InTexture.Format == static_cast<OaI32>(VK_FORMAT_UNDEFINED)
			or InTexture.Layout == static_cast<OaI32>(VK_IMAGE_LAYOUT_UNDEFINED)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"OaFnVideo::FromTexture image targets require view, format and current layout");
		}
		const VkFormat format = static_cast<VkFormat>(InTexture.Format);
		if (format != VK_FORMAT_R8G8B8A8_UNORM
			and format != VK_FORMAT_B8G8R8A8_UNORM) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"OaFnVideo::FromTexture supports RGBA8/BGRA8 UNORM render targets");
		}
		frame.Resource = OaVideoFrameResource::Image;
		frame.Image = static_cast<VkImage>(InTexture.Image);
		frame.ImageView = static_cast<VkImageView>(InTexture.View);
		frame.Format = format;
		frame.Layout = static_cast<VkImageLayout>(InTexture.Layout);
	} else {
		if (InTexture.DeviceBuf.Buffer == nullptr) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"OaFnVideo::FromTexture buffer target has no device buffer");
		}
		frame.Resource = OaVideoFrameResource::Buffer;
		frame.Buffer = &InTexture.DeviceBuf;
		frame.Format = VK_FORMAT_R8G8B8A8_UNORM;
	}
	return frame;
}

OaStatus Encode(
	OaContext& InContext,
	OaVideoEncoder& InSession,
	const OaTexture& InRgba,
	OaEncodedFrame& OutFrame,
	OaU64 InPts)
{
#ifdef OA_ANDROID_ML
	(void)InContext;
	(void)InSession;
	(void)InRgba;
	(void)OutFrame;
	(void)InPts;
	return OaStatus::Unimplemented(
		"Video encoding is not part of the Android ML profile");
#else
	if (not InRgba.IsValid()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::Encode: source texture is invalid");
	}
	if (InRgba.IsImageBacked()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::Encode: source texture must be buffer-backed");
	}
	if (InRgba.Width <= 0 or InRgba.Height <= 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::Encode: source texture extent must be positive");
	}
	const OaVkBuffer& buffer = InRgba.DeviceBuf;
	OaEngine& engine = InContext.Engine();
	const OaU64 bytes = static_cast<OaU64>(InRgba.Width)
		* static_cast<OaU64>(InRgba.Height) * 4U;
	if (buffer.Buffer == nullptr or buffer.Size < bytes
		or buffer.BindlessIndex == OA_BINDLESS_INVALID
		or buffer.Allocation == nullptr
		or buffer.AliasIdentity != nullptr or buffer.IsImported()
		or buffer.NodeIndex != 0U
		or buffer.AllocatorIdentity != engine.Allocator.Allocator) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnVideo::Encode: source must be a non-aliased buffer owned by the context engine");
	}
	OaContext::RecordingScope recording(InContext);
	// Encoding is an explicit consumer boundary. Submit and complete any graph
	// work that produces the texture before the encoder snapshots its buffer.
	OA_RETURN_IF_ERROR(InContext.Execute());
	OA_RETURN_IF_ERROR(InContext.Sync());
	OA_RETURN_IF_ERROR(InSession.UploadInputRgba(
		InRgba.DeviceBuf,
		static_cast<OaU32>(InRgba.Width),
		static_cast<OaU32>(InRgba.Height),
		OaYCbCrModel::BT709,
		false));
	return InSession.EncodeFrame(VK_NULL_HANDLE, InPts, OutFrame);
#endif
}

OaStatus Encode(
	OaVideoEncoder& InSession,
	const OaTexture& InRgba,
	OaEncodedFrame& OutFrame,
	OaU64 InPts)
{
	return Encode(
		OaContext::GetDefault(), InSession, InRgba, OutFrame, InPts);
}

OaStatus CvtRgbaToNv12(
	OaVideoEncoder& InSession,
	const OaVkBuffer& InRgba,
	OaU32 InVisibleWidth,
	OaU32 InVisibleHeight,
	OaYCbCrModel InColorSpace,
	bool InFullRange)
{
	return InSession.UploadInputRgba(InRgba, InVisibleWidth, InVisibleHeight, InColorSpace, InFullRange);
}

OaStatus CvtNv12ToRgb(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& OutRgbFrame)
{
	return OaVideoDecoderInternal::ConvertFrameToRgba(InSession, InNv12Frame, InOptions, OutRgbFrame);
}

OaResult<OaVideoFrame> Convert(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InFrame,
	const OaVideoConversionOptions& InOptions)
{
	if (InFrame.IsRgb) {
		return InFrame;
	}

	OaVideoFrame frame = {};
	OaStatus status = OaVideoDecoderInternal::ConvertFrameToRgba(
		InSession,
		InFrame,
		InOptions,
		frame);
	if (not status.IsOk()) {
		return status;
	}
	return frame;
}

OaResult<OaVec<OaU8>> ReadbackLuma(OaVideoDecoder& InSession, const OaVideoFrame& InFrame)
{
	return OaVideoDecoderInternal::ReadbackLuma(InSession, InFrame);
}

OaResult<OaVec<OaU8>> ReadbackNv12(OaVideoDecoder& InSession, const OaVideoFrame& InFrame)
{
	return OaVideoDecoderInternal::ReadbackNv12(InSession, InFrame);
}

OaResult<OaVec<OaU8>> ReadbackRgba(OaVideoDecoder& InSession, const OaVideoFrame& InFrame)
{
	return OaVideoDecoderInternal::ReadbackRgba(InSession, InFrame);
}

OaResult<OaVideoFrame> AllocateRgbaFrame(
	OaVideoDecoder& InSession,
	OaU32 InWidth,
	OaU32 InHeight)
{
	return OaVideoDecoderInternal::AllocateRgbaFrame(InSession, InWidth, InHeight);
}

OaStatus CvtNv12ToRgbInto(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	OaVideoFrame& InOutRgbTarget)
{
	return OaVideoDecoderInternal::ConvertNv12ToRgbInto(
		InSession,
		InNv12Frame,
		InOptions,
		InOutRgbTarget);
}

OaResult<OaVkImageDispatchTicket> CvtNv12ToRgbIntoAsync(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InNv12Frame,
	const OaVideoConversionOptions& InOptions,
	const OaVideoFrame& InRgbTarget)
{
	return OaVideoDecoderInternal::ConvertNv12ToRgbIntoAsync(
		InSession,
		InNv12Frame,
		InOptions,
		InRgbTarget);
}

OaResult<OaMatrix> FrameToBf16(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InFrame,
	bool InNormalizeImageNet)
{
	return OaVideoDecoderInternal::ConvertFrameToBf16(InSession, InFrame, InNormalizeImageNet);
}

OaResult<OaMatrix> FrameToBf16Hardware(
	OaVideoDecoder& InSession,
	const OaVideoFrame& InFrame,
	bool InNormalizeImageNet)
{
	return OaVideoDecoderInternal::ConvertFrameToBf16Hardware(InSession, InFrame, InNormalizeImageNet);
}

OaResult<OaMatrix> DecodeToBf16(
	OaVideoDecoder& InSession,
	const OaSpan<const OaU8>& InAccessUnit,
	bool InNormalizeImageNet)
{
	return OaVideoDecoderInternal::DecodeFrameToBf16(InSession, InAccessUnit, InNormalizeImageNet);
}

} // namespace OaFnVideo
