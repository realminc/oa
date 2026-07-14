// OA Vision — DecodeFrame dispatch into per-codec FnVideoDecoder* impl.

#include <Oa/Vision/VideoDecoder.h>

#include "../../FnVideo/Decoder/Av1/FnVideoDecoderAv1.h"
#include "../../FnVideo/Decoder/H264/FnVideoDecoderH264.h"
#include "../../FnVideo/Decoder/H265/FnVideoDecoderH265.h"
#include "../../FnVideo/Decoder/Vp9/FnVideoDecoderVp9.h"

OaStatus OaVideoDecoder::DecodeFrame(
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame)
{
	if (Session_.Handle() == VK_NULL_HANDLE) {
		return OaStatus::Error("Video decoder not initialized");
	}
	if (InBitstream.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Video bitstream is empty");
	}

	// Containerized AV1 and VP9 inputs are parsed before their elementary
	// frame payload is uploaded at bitstream offset 0.
	if (Profile_.Codec != OaVideoCodec::AV1
		&& Profile_.Codec != OaVideoCodec::VP9
		&& Profile_.Codec != OaVideoCodec::H265
		&& Profile_.Codec != OaVideoCodec::H264) {
		OA_RETURN_IF_ERROR(UploadBitstream(InBitstream));
	}

	switch (Profile_.Codec) {
	case OaVideoCodec::VP9:
		return FnVideoDecoderVp9::DecodeFrame(*this, InBitstream, OutFrame);
	case OaVideoCodec::AV1:
		return FnVideoDecoderAv1::DecodeFrame(*this, InBitstream, OutFrame);
	case OaVideoCodec::H265:
		return FnVideoDecoderH265::DecodeFrame(*this, InBitstream, OutFrame);
	case OaVideoCodec::H264:
		return FnVideoDecoderH264::DecodeFrame(*this, InBitstream, OutFrame);
	default:
		break;
	}

	return OaStatus::Error(OaStatusCode::Unavailable, "Codec not supported by modular parser path");
}
