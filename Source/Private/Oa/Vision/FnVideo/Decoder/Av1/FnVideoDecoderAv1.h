#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Vision/VideoDecoder.h>

namespace FnVideoDecoderAv1 {

OaStatus DecodeFrame(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InBitstream,
	OaVideoFrame& OutFrame);

} // namespace FnVideoDecoderAv1