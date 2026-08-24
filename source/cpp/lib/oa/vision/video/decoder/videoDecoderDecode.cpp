// OA Vision — decodeFrame dispatch into private stateful codec implementations.

#include <oa/vision/videoDecoder.h>
#include "videoDecoderImpl.h"

#include "codec/videoDecoderCodecAccess.h"

oa::Status oa::VideoDecoder::decodeFrame(
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	if (not impl_ or impl_->session.handle() == VK_NULL_HANDLE) {
		return oa::Status::error("Video decoder not initialized");
	}
	if (inBitstream.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Video bitstream is empty");
	}

	// Containerized AV1 and VP9 inputs are parsed before their elementary
	// frame payload is uploaded at bitstream offset 0.
	if (impl_->profile.codec != oa::VideoCodec::AV1
		&& impl_->profile.codec != oa::VideoCodec::VP9
		&& impl_->profile.codec != oa::VideoCodec::H265
		&& impl_->profile.codec != oa::VideoCodec::H264) {
		OA_RETURN_IF_ERROR(uploadBitstream(inBitstream));
	}

	switch (impl_->profile.codec) {
	case oa::VideoCodec::VP9:
		return oa::VideoDecoderCodecAccess::decodeVp9(*this, inBitstream, outFrame);
	case oa::VideoCodec::AV1:
		return oa::VideoDecoderCodecAccess::decodeAv1(*this, inBitstream, outFrame);
	case oa::VideoCodec::H265:
		return oa::VideoDecoderCodecAccess::decodeH265(*this, inBitstream, outFrame);
	case oa::VideoCodec::H264:
		return oa::VideoDecoderCodecAccess::decodeH264(*this, inBitstream, outFrame);
	default:
		break;
	}

	return oa::Status::error(oa::StatusCode::Unavailable, "codec not supported by modular parser path");
}
