#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

// OaTcpStream — a connected TCP socket (read/write/close)
// Move-only. Obtained from OaTcpListener::Accept() or OaTcpStream::Connect().
class OaTcpStream {
public:
	OaI64 Read(OaByte* OutBuf, OaU64 InSize);
	OaI64 Write(const OaByte* InBuf, OaU64 InSize);
	OaI64 WriteAll(const OaByte* InBuf, OaU64 InSize);
	void Close();

	[[nodiscard]] bool IsOpen() const noexcept { return Fd_ >= 0; }
	// For interrupting blocking Read/Write from another thread (POSIX shutdown).
	[[nodiscard]] OaI32 NativeHandle() const noexcept { return Fd_; }
	[[nodiscard]] OaString RemoteAddr() const { return RemoteAddr_; }
	[[nodiscard]] OaU16 RemotePort() const noexcept { return RemotePort_; }

	static OaResult<OaTcpStream> Connect(const OaString& InHost, OaU16 InPort);

	OaTcpStream() = default;
	~OaTcpStream();
	OaTcpStream(OaTcpStream&& InOther) noexcept;
	OaTcpStream& operator=(OaTcpStream&& InOther) noexcept;
	OaTcpStream(const OaTcpStream&) = delete;
	OaTcpStream& operator=(const OaTcpStream&) = delete;

private:
	OaI32 Fd_ = -1;
	OaString RemoteAddr_;
	OaU16 RemotePort_ = 0;
	friend class OaTcpListener;
	OaTcpStream(OaI32 InFd, OaString InAddr, OaU16 InPort);
};

// OaTcpListener — binds and listens on a TCP port, accepts connections
// Move-only. Use Bind() to create, Accept() to get streams.
class OaTcpListener {
public:
	static OaResult<OaTcpListener> Bind(OaU16 InPort, OaI32 InBacklog = 128);
	static OaResult<OaTcpListener> Bind(const OaString& InHost, OaU16 InPort, OaI32 InBacklog = 128);

	OaResult<OaTcpStream> Accept();
	void Close();

	[[nodiscard]] bool IsOpen() const noexcept { return Fd_ >= 0; }
	[[nodiscard]] OaU16 Port() const noexcept { return Port_; }

	OaTcpListener() = default;
	~OaTcpListener();
	OaTcpListener(OaTcpListener&& InOther) noexcept;
	OaTcpListener& operator=(OaTcpListener&& InOther) noexcept;
	OaTcpListener(const OaTcpListener&) = delete;
	OaTcpListener& operator=(const OaTcpListener&) = delete;

private:
	OaI32 Fd_ = -1;
	OaU16 Port_ = 0;
};
