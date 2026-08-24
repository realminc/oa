// OA Vision — Video codec registry Implementation

#include "codecRegistry.h"
#include "vcpH264.h"
#include "vcpH265.h"
#include "vcpAv1.h"
#include "vcpVp9.h"
#include <oa/vision/videoDecoder.h>

oa::VideoCodecRegistry& oa::VideoCodecRegistry::getInstance()
{
	static oa::VideoCodecRegistry instance;
	return instance;
}

oa::VideoCodecParser* oa::VideoCodecRegistry::getParser(oa::VideoCodec inCodec)
{
	auto it = parsers_.find(inCodec);
	if (it != parsers_.end()) {
		return it->second.get();
	}
	return nullptr;
}

oa::UniquePtr<oa::VideoCodecParser> oa::VideoCodecRegistry::createParser(oa::VideoCodec inCodec) const
{
	switch (inCodec) {
	case oa::VideoCodec::H264: return oa::makeUnique<oa::VcpH264>();
	case oa::VideoCodec::H265: return oa::makeUnique<oa::VcpH265>();
	case oa::VideoCodec::AV1:  return oa::makeUnique<oa::VcpAv1>();
	case oa::VideoCodec::VP9:  return oa::makeUnique<oa::VcpVp9>();
	}
	return {};
}

void oa::VideoCodecRegistry::registerParser(oa::VideoCodec inCodec, oa::UniquePtr<oa::VideoCodecParser> inParser)
{
	parsers_.insert({inCodec, oa::move(inParser)});
}

void oa::VideoCodecRegistry::registerAllParsers()
{
	// register H.264 parser
	if (!getParser(oa::VideoCodec::H264)) {
		registerParser(oa::VideoCodec::H264, oa::makeUnique<oa::VcpH264>());
	}

	// register H.265 parser
	if (!getParser(oa::VideoCodec::H265)) {
		registerParser(oa::VideoCodec::H265, oa::makeUnique<oa::VcpH265>());
	}

	// register AV1 parser
	if (!getParser(oa::VideoCodec::AV1)) {
		registerParser(oa::VideoCodec::AV1, oa::makeUnique<oa::VcpAv1>());
	}

	// register VP9 parser (vulkan record path still stub)
	if (!getParser(oa::VideoCodec::VP9)) {
		registerParser(oa::VideoCodec::VP9, oa::makeUnique<oa::VcpVp9>());
	}
}
