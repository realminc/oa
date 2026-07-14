#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoDecoder.h>

namespace FnVideoDecoderH265 {

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame);

} // namespace FnVideoDecoderH265