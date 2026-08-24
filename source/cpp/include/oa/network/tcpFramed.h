#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

class TcpStream;

// length-prefixed messages: 4-byte little-endian payload length, then payload.
// This is a generic binary framing primitive; protocol semantics live with
// their owning sessions.
class TcpFramed {
public:
	static constexpr oa::U32 kMaxPayloadBytes = 16u * 1024u * 1024u;

	static oa::Status writeMessage(TcpStream& inStream, oa::Span<const oa::Byte> inPayload);
	static oa::Status readMessage(
		TcpStream& inStream,
		oa::Vec<oa::Byte>& outPayload,
		oa::U32 inMaxPayloadBytes = kMaxPayloadBytes);

private:
	static void writeU32Le(oa::Byte* outBuf, oa::U32 inValue);
	static bool readU32Le(const oa::Byte* inBuf, oa::U32& outValue);
};

} // namespace oa
