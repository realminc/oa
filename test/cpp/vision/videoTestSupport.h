#pragma once

#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoEncoder.h>

[[nodiscard]] inline bool testVideoDecodeSupported(
	oa::Engine& inEngine,
	oa::VideoCodec inCodec)
{
	auto capabilities = oa::VideoDecoder::queryDecodeCapabilities(inEngine, inCodec);
	return capabilities.isOk() && capabilities->supported;
}

[[nodiscard]] inline bool testVideoEncodeSupported(
	oa::Engine& inEngine,
	oa::VideoCodec inCodec)
{
	auto capabilities = oa::VideoEncoder::queryEncodeCapabilities(inEngine, inCodec);
	return capabilities.isOk() && capabilities->supported;
}
