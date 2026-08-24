// Shared VkVideoProfileInfoKHR builder for oa::VideoDecoder (deduped from
// VideoDecoder*.cpp).

#pragma once

#include <oa/vision/videoDecoder.h>

namespace oa::videoDecoderProfile {

oa::Result<oa::VideoProfile> resolveDecodeProfile(const oa::VideoProfile& inProfile);
bool isDecodePathImplemented(const oa::VideoProfile& inProfile);

oa::Result<VkVideoProfileInfoKHR> buildDecodeProfile(
	const oa::VideoProfile& inProfile,
	VkVideoDecodeH264ProfileInfoKHR& outH264,
	VkVideoDecodeH265ProfileInfoKHR& outH265,
	VkVideoDecodeAV1ProfileInfoKHR& outAV1,
	VkVideoDecodeVP9ProfileInfoKHR& outVp9);

} // namespace oa::videoDecoderProfile
