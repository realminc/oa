#include <oa/network/tcp.h>
#include <oa/network/tcpFramed.h>

static oa::I64 tcpReadExact(oa::TcpStream& inStream, oa::Byte* outBuf, oa::U64 inSize) {
	oa::U64 got = 0;
	while (got < inSize) {
		const oa::I64 chunk = inStream.read(outBuf + got, inSize - got);
		if (chunk <= 0) return -1;
		got += static_cast<oa::U64>(chunk);
	}
	return static_cast<oa::I64>(got);
}

void oa::TcpFramed::writeU32Le(oa::Byte* outBuf, oa::U32 inValue) {
	outBuf[0] = static_cast<oa::Byte>(inValue & 0xffu);
	outBuf[1] = static_cast<oa::Byte>((inValue >> 8) & 0xffu);
	outBuf[2] = static_cast<oa::Byte>((inValue >> 16) & 0xffu);
	outBuf[3] = static_cast<oa::Byte>((inValue >> 24) & 0xffu);
}

bool oa::TcpFramed::readU32Le(const oa::Byte* inBuf, oa::U32& outValue) {
	outValue = static_cast<oa::U32>(inBuf[0]) | (static_cast<oa::U32>(inBuf[1]) << 8)
		| (static_cast<oa::U32>(inBuf[2]) << 16) | (static_cast<oa::U32>(inBuf[3]) << 24);
	return true;
}

oa::Status oa::TcpFramed::writeMessage(oa::TcpStream& inStream, oa::Span<const oa::Byte> inPayload) {
	if (inPayload.size() > kMaxPayloadBytes) {
		return oa::Status::error("tcp framed: payload too large");
	}
	oa::Byte header[4];
	writeU32Le(header, static_cast<oa::U32>(inPayload.size()));
	if (inStream.writeAll(header, sizeof(header)) != static_cast<oa::I64>(sizeof(header))) {
		return oa::Status::error("tcp framed: write header failed");
	}
	if (not inPayload.empty()) {
		if (inStream.writeAll(inPayload.data(), inPayload.size())
			!= static_cast<oa::I64>(inPayload.size())) {
			return oa::Status::error("tcp framed: write body failed");
		}
	}
	return oa::Status::ok();
}

oa::Status oa::TcpFramed::readMessage(
	oa::TcpStream& inStream,
	oa::Vector<oa::Byte>& outPayload,
	oa::U32 inMaxPayloadBytes)
{
	if (inMaxPayloadBytes > kMaxPayloadBytes) {
		return oa::Status::invalidArgument(
			"tcp framed: requested payload limit exceeds the transport ceiling");
	}
	oa::Byte header[4];
	if (tcpReadExact(inStream, header, sizeof(header)) != static_cast<oa::I64>(sizeof(header))) {
		return oa::Status::error("tcp framed: read header failed");
	}
	oa::U32 len = 0;
	(void)readU32Le(header, len);
	if (len > inMaxPayloadBytes) {
		return oa::Status::error("tcp framed: peer length too large");
	}
	outPayload.resize(len);
	if (len == 0) {
		return oa::Status::ok();
	}
	if (tcpReadExact(inStream, outPayload.data(), len) != static_cast<oa::I64>(len)) {
		return oa::Status::error("tcp framed: read body failed");
	}
	return oa::Status::ok();
}
