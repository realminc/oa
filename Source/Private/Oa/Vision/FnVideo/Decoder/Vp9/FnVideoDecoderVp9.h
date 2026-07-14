// Per-codec decode impl (VP9). Private — called from OaVideoDecoder::DecodeFrame
// dispatch only, not from OaFnVideo::Decode directly.
// Pair: Video/Codec/VcpVp9 (parse) → FnVideo/Decoder/Vp9 (DPB + vkCmdDecodeVideoKHR).
// Session object: Video/Decoder/ | Public shim: FnVideo/Codec/FnVideoCodec.cpp

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoDecoder.h>

namespace FnVideoDecoderVp9 {

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame);

} // namespace FnVideoDecoderVp9