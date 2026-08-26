#include <oa/network/tcp.h>
#include <oa/core/log.h>
#include <oa/core/std/utility.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#endif

#include <errno.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
static void tcpEnsureWinsock() {
	static bool initialized = [] {
		WSADATA data{};
		return wSAStartup(MAKEWORD(2, 2), &data) == 0;
	}();
	(void)initialized;
}

static oa::I32 tcpInvalid() { return static_cast<oa::I32>(INVALID_SOCKET); }
static oa::I32 tcpSocket(int inAf, int inType, int inProtocol) {
	tcpEnsureWinsock();
	return static_cast<oa::I32>(::socket(inAf, inType, inProtocol));
}
static int tcpClose(oa::I32 inFd) { return ::closesocket(static_cast<SOCKET>(inFd)); }
static oa::I64 tcpRead(oa::I32 inFd, oa::Byte* outBuf, oa::U64 inSize) {
	return static_cast<oa::I64>(::recv(static_cast<SOCKET>(inFd), reinterpret_cast<char*>(outBuf), static_cast<int>(inSize), 0));
}
static oa::I64 tcpWrite(oa::I32 inFd, const oa::Byte* inBuf, oa::U64 inSize) {
	return static_cast<oa::I64>(::send(static_cast<SOCKET>(inFd), reinterpret_cast<const char*>(inBuf), static_cast<int>(inSize), 0));
}
#else
static void tcpEnsureWinsock() {}
static oa::I32 tcpInvalid() { return -1; }
static oa::I32 tcpSocket(int inAf, int inType, int inProtocol) { return ::socket(inAf, inType, inProtocol); }
static int tcpClose(oa::I32 inFd) { return ::close(inFd); }
static oa::I64 tcpRead(oa::I32 inFd, oa::Byte* outBuf, oa::U64 inSize) { return static_cast<oa::I64>(::read(inFd, outBuf, inSize)); }
static oa::I64 tcpWrite(oa::I32 inFd, const oa::Byte* inBuf, oa::U64 inSize) { return static_cast<oa::I64>(::write(inFd, inBuf, inSize)); }
#endif

static oa::String tcpFormatU16(oa::U16 inPort) {
	char buf[8];
	::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(inPort));
	return oa::String(buf);
}

// oa::TcpStream

oa::TcpStream::TcpStream(oa::I32 inFd, oa::String inAddr, oa::U16 inPort)
	: fd_(inFd), remoteAddr_(oa::move(inAddr)), remotePort_(inPort) {}

oa::TcpStream::~TcpStream() { close(); }

oa::TcpStream::TcpStream(oa::TcpStream&& inOther) noexcept
	: fd_(inOther.fd_), remoteAddr_(oa::move(inOther.remoteAddr_)), remotePort_(inOther.remotePort_) {
	inOther.fd_ = -1;
	inOther.remotePort_ = 0;
}

oa::TcpStream& oa::TcpStream::operator=(oa::TcpStream&& inOther) noexcept {
	if (this != &inOther) {
		close();
		fd_ = inOther.fd_;
		remoteAddr_ = oa::move(inOther.remoteAddr_);
		remotePort_ = inOther.remotePort_;
		inOther.fd_ = -1;
		inOther.remotePort_ = 0;
	}
	return *this;
}

oa::I64 oa::TcpStream::read(oa::Byte* outBuf, oa::U64 inSize) {
	if (fd_ < 0) return -1;
	return tcpRead(fd_, outBuf, inSize);
}

oa::I64 oa::TcpStream::write(const oa::Byte* inBuf, oa::U64 inSize) {
	if (fd_ < 0) return -1;
	return tcpWrite(fd_, inBuf, inSize);
}

oa::I64 oa::TcpStream::writeAll(const oa::Byte* inBuf, oa::U64 inSize) {
	oa::U64 sent = 0;
	while (sent < inSize) {
		auto n = write(inBuf + sent, inSize - sent);
		if (n <= 0) return (sent > 0) ? static_cast<oa::I64>(sent) : n;
		sent += static_cast<oa::U64>(n);
	}
	return static_cast<oa::I64>(sent);
}

oa::Status oa::TcpStream::setIoTimeout(oa::U32 inTimeoutMs) {
	if (fd_ < 0) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition, "tcp stream is closed");
	}
#if defined(_WIN32)
	const DWORD timeout = static_cast<DWORD>(inTimeoutMs);
	if (::setsockopt(static_cast<SOCKET>(fd_), SOL_SOCKET, SO_RCVTIMEO,
		reinterpret_cast<const char*>(&timeout), sizeof(timeout)) != 0
		or ::setsockopt(static_cast<SOCKET>(fd_), SOL_SOCKET, SO_SNDTIMEO,
			reinterpret_cast<const char*>(&timeout), sizeof(timeout)) != 0)
	{
		return oa::Status::error(oa::StatusCode::Unavailable,
			"failed to configure TCP I/O timeout");
	}
#else
	const timeval timeout{
		.tv_sec = static_cast<time_t>(inTimeoutMs / 1000U),
		.tv_usec = static_cast<suseconds_t>((inTimeoutMs % 1000U) * 1000U),
	};
	if (::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0
		or ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0)
	{
		return oa::Status::error(oa::StatusCode::Unavailable,
			oa::String("failed to configure TCP I/O timeout: ") + ::strerror(errno));
	}
#endif
	return oa::Status::ok();
}

void oa::TcpStream::close() {
	if (fd_ >= 0) {
		tcpClose(fd_);
		fd_ = -1;
	}
}

oa::Result<oa::TcpStream> oa::TcpStream::connect(const oa::String& inHost, oa::U16 inPort) {
	tcpEnsureWinsock();
	struct addrinfo hints{}, *res = nullptr;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	oa::String portStr = tcpFormatU16(inPort);
	if (::getaddrinfo(inHost.cStr(), portStr.cStr(), &hints, &res) != 0) {
		return oa::Status::invalidArgument(oa::String("failed to resolve host: ") + inHost);
	}

	oa::I32 fd = tcpInvalid();
	for (auto* rp = res; rp; rp = rp->ai_next) {
		fd = tcpSocket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (fd < 0) continue;
		if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
		tcpClose(fd);
		fd = tcpInvalid();
	}
	::freeaddrinfo(res);

	if (fd < 0) {
		return oa::Status::error(
			oa::StatusCode::Unavailable,
			oa::String("failed to connect to ") + inHost + ":" + portStr);
	}

	oa::I32 yes = 1;
	::setsockopt(static_cast<SOCKET>(fd), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));

	return oa::TcpStream(fd, inHost, inPort);
}

// oa::TcpListener

oa::TcpListener::~TcpListener() { close(); }

oa::TcpListener::TcpListener(oa::TcpListener&& inOther) noexcept
	: fd_(inOther.fd_), port_(inOther.port_) {
	inOther.fd_ = -1;
	inOther.port_ = 0;
}

oa::TcpListener& oa::TcpListener::operator=(oa::TcpListener&& inOther) noexcept {
	if (this != &inOther) {
		close();
		fd_ = inOther.fd_;
		port_ = inOther.port_;
		inOther.fd_ = -1;
		inOther.port_ = 0;
	}
	return *this;
}

oa::Result<oa::TcpListener> oa::TcpListener::bind(oa::U16 inPort, oa::I32 inBacklog) {
	return bind("0.0.0.0", inPort, inBacklog);
}

oa::Result<oa::TcpListener> oa::TcpListener::bind(const oa::String& inHost, oa::U16 inPort, oa::I32 inBacklog) {
	tcpEnsureWinsock();
	oa::I32 fd = tcpSocket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0) {
		return oa::Status::error("socket() failed: " + oa::String(::strerror(errno)));
	}

	oa::I32 yes = 1;
	::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#if !defined(_WIN32)
	::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif

	struct sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(inPort);
	if (::inet_pton(AF_INET, inHost.cStr(), &addr.sin_addr) != 1) {
		tcpClose(fd);
		return oa::Status::invalidArgument(oa::String("invalid bind address: ") + inHost);
	}

	if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
		tcpClose(fd);
		oa::String bindMsg;
		bindMsg += "bind() failed on port ";
		bindMsg += tcpFormatU16(inPort);
		bindMsg += ": ";
		bindMsg += ::strerror(errno);
		return oa::Status::error(oa::move(bindMsg));
	}

	if (::listen(fd, inBacklog) < 0) {
		tcpClose(fd);
		return oa::Status::error("listen() failed: " + oa::String(::strerror(errno)));
	}

	// Resolve actual port (if inPort was 0, OS picks one)
	struct sockaddr_in bound{};
	socklen_t len = sizeof(bound);
	::getsockname(fd, reinterpret_cast<struct sockaddr*>(&bound), &len);
	oa::U16 actualPort = ntohs(bound.sin_port);

	oa::TcpListener listener;
	listener.fd_ = fd;
	listener.port_ = actualPort;
	return listener;
}

oa::Result<oa::TcpStream> oa::TcpListener::accept() {
	if (fd_ < 0) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "listener is closed");
	}

	struct sockaddr_in clientAddr{};
	socklen_t addrLen = sizeof(clientAddr);
	oa::I32 clientFd = static_cast<oa::I32>(::accept(static_cast<SOCKET>(fd_), reinterpret_cast<struct sockaddr*>(&clientAddr), &addrLen));
	if (clientFd < 0) {
		return oa::Status::error("accept() failed: " + oa::String(::strerror(errno)));
	}

	oa::I32 yes = 1;
	::setsockopt(static_cast<SOCKET>(clientFd), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));

	char addrBuf[INET_ADDRSTRLEN];
	::inet_ntop(AF_INET, &clientAddr.sin_addr, addrBuf, sizeof(addrBuf));

	return oa::TcpStream(clientFd, oa::String(addrBuf), ntohs(clientAddr.sin_port));
}

void oa::TcpListener::close() {
	if (fd_ >= 0) {
		tcpClose(fd_);
		fd_ = -1;
	}
}
