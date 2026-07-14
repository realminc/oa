#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaTcpStream;

// Length-prefixed messages: 4-byte little-endian payload length, then payload.
// Used by editor automation TCP and MCP bridge; keeps MCP/JSON out of oa.
class OaTcpFramed {
public:
	static constexpr OaU32 kMaxPayloadBytes = 16u * 1024u * 1024u;

	static OaStatus WriteMessage(OaTcpStream &InStream, OaSpan<const OaByte> InPayload);
	static OaStatus ReadMessage(OaTcpStream &InStream, OaVec<OaByte> &OutPayload);

private:
	static void WriteU32Le(OaByte *OutBuf, OaU32 InValue);
	static bool ReadU32Le(const OaByte *InBuf, OaU32 &OutValue);
};
