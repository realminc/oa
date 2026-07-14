#include <Oa/Network/Tcp.h>
#include <Oa/Network/TcpFramed.h>

static OaI64 OaTcpReadExact(OaTcpStream &InStream, OaByte *OutBuf, OaU64 InSize) {
	OaU64 got = 0;
	while (got < InSize) {
		const OaI64 chunk = InStream.Read(OutBuf + got, InSize - got);
		if (chunk <= 0) return -1;
		got += static_cast<OaU64>(chunk);
	}
	return static_cast<OaI64>(got);
}

void OaTcpFramed::WriteU32Le(OaByte *OutBuf, OaU32 InValue) {
	OutBuf[0] = static_cast<OaByte>(InValue & 0xffu);
	OutBuf[1] = static_cast<OaByte>((InValue >> 8) & 0xffu);
	OutBuf[2] = static_cast<OaByte>((InValue >> 16) & 0xffu);
	OutBuf[3] = static_cast<OaByte>((InValue >> 24) & 0xffu);
}

bool OaTcpFramed::ReadU32Le(const OaByte *InBuf, OaU32 &OutValue) {
	OutValue = static_cast<OaU32>(InBuf[0]) | (static_cast<OaU32>(InBuf[1]) << 8)
		| (static_cast<OaU32>(InBuf[2]) << 16) | (static_cast<OaU32>(InBuf[3]) << 24);
	return true;
}

OaStatus OaTcpFramed::WriteMessage(OaTcpStream &InStream, OaSpan<const OaByte> InPayload) {
	if (InPayload.size() > kMaxPayloadBytes) {
		return OaStatus::Error("tcp framed: payload too large");
	}
	OaByte header[4];
	WriteU32Le(header, static_cast<OaU32>(InPayload.size()));
	if (InStream.WriteAll(header, sizeof(header)) != static_cast<OaI64>(sizeof(header))) {
		return OaStatus::Error("tcp framed: write header failed");
	}
	if (not InPayload.empty()) {
		if (InStream.WriteAll(InPayload.data(), InPayload.size())
			!= static_cast<OaI64>(InPayload.size())) {
			return OaStatus::Error("tcp framed: write body failed");
		}
	}
	return OaStatus::Ok();
}

OaStatus OaTcpFramed::ReadMessage(OaTcpStream &InStream, OaVec<OaByte> &OutPayload) {
	OaByte header[4];
	if (OaTcpReadExact(InStream, header, sizeof(header)) != static_cast<OaI64>(sizeof(header))) {
		return OaStatus::Error("tcp framed: read header failed");
	}
	OaU32 len = 0;
	(void)ReadU32Le(header, len);
	if (len > kMaxPayloadBytes) {
		return OaStatus::Error("tcp framed: peer length too large");
	}
	OutPayload.resize(len);
	if (len == 0) {
		return OaStatus::Ok();
	}
	if (OaTcpReadExact(InStream, OutPayload.data(), len) != static_cast<OaI64>(len)) {
		return OaStatus::Error("tcp framed: read body failed");
	}
	return OaStatus::Ok();
}
