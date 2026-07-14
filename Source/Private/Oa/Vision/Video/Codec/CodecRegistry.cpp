// OA Vision — Video Codec Registry Implementation

#include "CodecRegistry.h"
#include "VcpH264.h"
#include "VcpH265.h"
#include "VcpAv1.h"
#include "VcpVp9.h"
#include <Oa/Vision/VideoDecoder.h>

OaVideoCodecRegistry& OaVideoCodecRegistry::GetInstance()
{
	static OaVideoCodecRegistry instance;
	return instance;
}

OaVideoCodecParser* OaVideoCodecRegistry::GetParser(OaVideoCodec InCodec)
{
	auto it = Parsers_.Find(InCodec);
	if (it != Parsers_.End()) {
		return it->second.Get();
	}
	return nullptr;
}

OaStdUniquePtr<OaVideoCodecParser> OaVideoCodecRegistry::CreateParser(OaVideoCodec InCodec) const
{
	switch (InCodec) {
	case OaVideoCodec::H264: return OaStdMakeUnique<OaVcpH264>();
	case OaVideoCodec::H265: return OaStdMakeUnique<OaVcpH265>();
	case OaVideoCodec::AV1:  return OaStdMakeUnique<OaVcpAv1>();
	case OaVideoCodec::VP9:  return OaStdMakeUnique<OaVcpVp9>();
	}
	return {};
}

void OaVideoCodecRegistry::RegisterParser(OaVideoCodec InCodec, OaStdUniquePtr<OaVideoCodecParser> InParser)
{
	Parsers_.Insert({InCodec, OaStdMove(InParser)});
}

void OaVideoCodecRegistry::RegisterAllParsers()
{
	// Register H.264 parser
	if (!GetParser(OaVideoCodec::H264)) {
		RegisterParser(OaVideoCodec::H264, OaStdMakeUnique<OaVcpH264>());
	}

	// Register H.265 parser
	if (!GetParser(OaVideoCodec::H265)) {
		RegisterParser(OaVideoCodec::H265, OaStdMakeUnique<OaVcpH265>());
	}

	// Register AV1 parser
	if (!GetParser(OaVideoCodec::AV1)) {
		RegisterParser(OaVideoCodec::AV1, OaStdMakeUnique<OaVcpAv1>());
	}

	// Register VP9 parser (Vulkan record path still stub)
	if (!GetParser(OaVideoCodec::VP9)) {
		RegisterParser(OaVideoCodec::VP9, OaStdMakeUnique<OaVcpVp9>());
	}
}
