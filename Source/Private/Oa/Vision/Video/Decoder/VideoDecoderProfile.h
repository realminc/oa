// Shared VkVideoProfileInfoKHR builder for OaVideoDecoder (deduped from
// VideoDecoder*.cpp).

#pragma once

#include <Oa/Vision/VideoDecoder.h>

namespace OaVideoDecoderProfile {

OaResult<OaVideoProfile> ResolveDecodeProfile(const OaVideoProfile& InProfile);
bool IsDecodePathImplemented(const OaVideoProfile& InProfile);

OaResult<VkVideoProfileInfoKHR> BuildDecodeProfile(
	const OaVideoProfile& InProfile,
	VkVideoDecodeH264ProfileInfoKHR& OutH264,
	VkVideoDecodeH265ProfileInfoKHR& OutH265,
	VkVideoDecodeAV1ProfileInfoKHR& OutAV1,
	VkVideoDecodeVP9ProfileInfoKHR& OutVp9);

} // namespace OaVideoDecoderProfile
