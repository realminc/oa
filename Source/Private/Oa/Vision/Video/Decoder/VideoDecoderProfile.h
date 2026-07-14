// Shared VkVideoProfileInfoKHR builder for OaVideoDecoder (deduped from VideoDecoder*.cpp).

#pragma once

#include <Oa/Vision/VideoDecoder.h>

namespace OaVideoDecoderProfile {

VkVideoProfileInfoKHR BuildDecodeProfile(
	OaVideoCodec InCodec,
	VkVideoDecodeH264ProfileInfoKHR& OutH264,
	VkVideoDecodeH265ProfileInfoKHR& OutH265,
	VkVideoDecodeAV1ProfileInfoKHR& OutAV1,
	VkVideoDecodeVP9ProfileInfoKHR& OutVp9);

} // namespace OaVideoDecoderProfile