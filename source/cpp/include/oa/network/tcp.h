#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>

namespace oa {

// TcpStream — a connected TCP socket (read/write/close)
// move-only. Obtained from TcpListener::accept() or TcpStream::connect().
class TcpStream {
public:
	oa::I64 read(oa::Byte* outBuf, oa::U64 inSize);
	oa::I64 write(const oa::Byte* inBuf, oa::U64 inSize);
	oa::I64 writeAll(const oa::Byte* inBuf, oa::U64 inSize);
	// Bound blocking reads and writes for stateful protocol consumers. A zero
	// timeout restores the platform's blocking default.
	[[nodiscard]] oa::Status setIoTimeout(oa::U32 inTimeoutMs);
	void close();

	[[nodiscard]] bool isOpen() const noexcept { return fd_ >= 0; }
	// For interrupting blocking Read/Write from another thread (POSIX shutdown).
	[[nodiscard]] oa::I32 nativeHandle() const noexcept { return fd_; }
	[[nodiscard]] oa::String remoteAddr() const { return remoteAddr_; }
	[[nodiscard]] oa::U16 remotePort() const noexcept { return remotePort_; }

	static oa::Result<TcpStream> connect(const oa::String& inHost, oa::U16 inPort);

	TcpStream() = default;
	~TcpStream();
	TcpStream(TcpStream&& inOther) noexcept;
	TcpStream& operator=(TcpStream&& inOther) noexcept;
	TcpStream(const TcpStream&) = delete;
	TcpStream& operator=(const TcpStream&) = delete;

private:
	oa::I32 fd_ = -1;
	oa::String remoteAddr_;
	oa::U16 remotePort_ = 0;
	friend class TcpListener;
	TcpStream(oa::I32 inFd, oa::String inAddr, oa::U16 inPort);
};

// TcpListener — binds and listens on a TCP port, accepts connections
// move-only. Use bind() to create, accept() to get streams.
class TcpListener {
public:
	static oa::Result<TcpListener> bind(oa::U16 inPort, oa::I32 inBacklog = 128);
	static oa::Result<TcpListener> bind(const oa::String& inHost, oa::U16 inPort, oa::I32 inBacklog = 128);

	oa::Result<TcpStream> accept();
	void close();

	[[nodiscard]] bool isOpen() const noexcept { return fd_ >= 0; }
	[[nodiscard]] oa::U16 port() const noexcept { return port_; }

	TcpListener() = default;
	~TcpListener();
	TcpListener(TcpListener&& inOther) noexcept;
	TcpListener& operator=(TcpListener&& inOther) noexcept;
	TcpListener(const TcpListener&) = delete;
	TcpListener& operator=(const TcpListener&) = delete;

private:
	oa::I32 fd_ = -1;
	oa::U16 port_ = 0;
};

} // namespace oa
