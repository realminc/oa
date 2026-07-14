#include <Oa/Network/Tcp.h>
#include <Oa/Core/Log.h>

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

#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
static void OaTcpEnsureWinsock() {
	static bool initialized = [] {
		WSADATA data{};
		return WSAStartup(MAKEWORD(2, 2), &data) == 0;
	}();
	(void)initialized;
}

static OaI32 OaTcpInvalid() { return static_cast<OaI32>(INVALID_SOCKET); }
static OaI32 OaTcpSocket(int InAf, int InType, int InProtocol) {
	OaTcpEnsureWinsock();
	return static_cast<OaI32>(::socket(InAf, InType, InProtocol));
}
static int OaTcpClose(OaI32 InFd) { return ::closesocket(static_cast<SOCKET>(InFd)); }
static OaI64 OaTcpRead(OaI32 InFd, OaByte* OutBuf, OaU64 InSize) {
	return static_cast<OaI64>(::recv(static_cast<SOCKET>(InFd), reinterpret_cast<char*>(OutBuf), static_cast<int>(InSize), 0));
}
static OaI64 OaTcpWrite(OaI32 InFd, const OaByte* InBuf, OaU64 InSize) {
	return static_cast<OaI64>(::send(static_cast<SOCKET>(InFd), reinterpret_cast<const char*>(InBuf), static_cast<int>(InSize), 0));
}
#else
static void OaTcpEnsureWinsock() {}
static OaI32 OaTcpInvalid() { return -1; }
static OaI32 OaTcpSocket(int InAf, int InType, int InProtocol) { return ::socket(InAf, InType, InProtocol); }
static int OaTcpClose(OaI32 InFd) { return ::close(InFd); }
static OaI64 OaTcpRead(OaI32 InFd, OaByte* OutBuf, OaU64 InSize) { return static_cast<OaI64>(::read(InFd, OutBuf, InSize)); }
static OaI64 OaTcpWrite(OaI32 InFd, const OaByte* InBuf, OaU64 InSize) { return static_cast<OaI64>(::write(InFd, InBuf, InSize)); }
#endif

static OaString OaTcpFormatU16(OaU16 InPort) {
	char buf[8];
	std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(InPort));
	return OaString(buf);
}

// OaTcpStream

OaTcpStream::OaTcpStream(OaI32 InFd, OaString InAddr, OaU16 InPort)
	: Fd_(InFd), RemoteAddr_(std::move(InAddr)), RemotePort_(InPort) {}

OaTcpStream::~OaTcpStream() { Close(); }

OaTcpStream::OaTcpStream(OaTcpStream&& InOther) noexcept
	: Fd_(InOther.Fd_), RemoteAddr_(std::move(InOther.RemoteAddr_)), RemotePort_(InOther.RemotePort_) {
	InOther.Fd_ = -1;
	InOther.RemotePort_ = 0;
}

OaTcpStream& OaTcpStream::operator=(OaTcpStream&& InOther) noexcept {
	if (this != &InOther) {
		Close();
		Fd_ = InOther.Fd_;
		RemoteAddr_ = std::move(InOther.RemoteAddr_);
		RemotePort_ = InOther.RemotePort_;
		InOther.Fd_ = -1;
		InOther.RemotePort_ = 0;
	}
	return *this;
}

OaI64 OaTcpStream::Read(OaByte* OutBuf, OaU64 InSize) {
	if (Fd_ < 0) return -1;
	return OaTcpRead(Fd_, OutBuf, InSize);
}

OaI64 OaTcpStream::Write(const OaByte* InBuf, OaU64 InSize) {
	if (Fd_ < 0) return -1;
	return OaTcpWrite(Fd_, InBuf, InSize);
}

OaI64 OaTcpStream::WriteAll(const OaByte* InBuf, OaU64 InSize) {
	OaU64 sent = 0;
	while (sent < InSize) {
		auto n = Write(InBuf + sent, InSize - sent);
		if (n <= 0) return (sent > 0) ? static_cast<OaI64>(sent) : n;
		sent += static_cast<OaU64>(n);
	}
	return static_cast<OaI64>(sent);
}

void OaTcpStream::Close() {
	if (Fd_ >= 0) {
		OaTcpClose(Fd_);
		Fd_ = -1;
	}
}

OaResult<OaTcpStream> OaTcpStream::Connect(const OaString& InHost, OaU16 InPort) {
	OaTcpEnsureWinsock();
	struct addrinfo hints{}, *res = nullptr;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	OaString portStr = OaTcpFormatU16(InPort);
	if (::getaddrinfo(InHost.c_str(), portStr.c_str(), &hints, &res) != 0) {
		return OaStatus::InvalidArgument(OaString("failed to resolve host: ") + InHost);
	}

	OaI32 fd = OaTcpInvalid();
	for (auto* rp = res; rp; rp = rp->ai_next) {
		fd = OaTcpSocket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (fd < 0) continue;
		if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
		OaTcpClose(fd);
		fd = OaTcpInvalid();
	}
	::freeaddrinfo(res);

	if (fd < 0) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			OaString("failed to connect to ") + InHost + ":" + portStr);
	}

	OaI32 yes = 1;
	::setsockopt(static_cast<SOCKET>(fd), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));

	return OaTcpStream(fd, InHost, InPort);
}

// OaTcpListener

OaTcpListener::~OaTcpListener() { Close(); }

OaTcpListener::OaTcpListener(OaTcpListener&& InOther) noexcept
	: Fd_(InOther.Fd_), Port_(InOther.Port_) {
	InOther.Fd_ = -1;
	InOther.Port_ = 0;
}

OaTcpListener& OaTcpListener::operator=(OaTcpListener&& InOther) noexcept {
	if (this != &InOther) {
		Close();
		Fd_ = InOther.Fd_;
		Port_ = InOther.Port_;
		InOther.Fd_ = -1;
		InOther.Port_ = 0;
	}
	return *this;
}

OaResult<OaTcpListener> OaTcpListener::Bind(OaU16 InPort, OaI32 InBacklog) {
	return Bind("0.0.0.0", InPort, InBacklog);
}

OaResult<OaTcpListener> OaTcpListener::Bind(const OaString& InHost, OaU16 InPort, OaI32 InBacklog) {
	OaTcpEnsureWinsock();
	OaI32 fd = OaTcpSocket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0) {
		return OaStatus::Error("socket() failed: " + OaString(std::strerror(errno)));
	}

	OaI32 yes = 1;
	::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#if !defined(_WIN32)
	::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif

	struct sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(InPort);
	if (::inet_pton(AF_INET, InHost.c_str(), &addr.sin_addr) != 1) {
		OaTcpClose(fd);
		return OaStatus::InvalidArgument(OaString("invalid bind address: ") + InHost);
	}

	if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
		OaTcpClose(fd);
		OaString bindMsg;
		bindMsg += "bind() failed on port ";
		bindMsg += OaTcpFormatU16(InPort);
		bindMsg += ": ";
		bindMsg += std::strerror(errno);
		return OaStatus::Error(std::move(bindMsg));
	}

	if (::listen(fd, InBacklog) < 0) {
		OaTcpClose(fd);
		return OaStatus::Error("listen() failed: " + OaString(std::strerror(errno)));
	}

	// Resolve actual port (if InPort was 0, OS picks one)
	struct sockaddr_in bound{};
	socklen_t len = sizeof(bound);
	::getsockname(fd, reinterpret_cast<struct sockaddr*>(&bound), &len);
	OaU16 actualPort = ntohs(bound.sin_port);

	OaTcpListener listener;
	listener.Fd_ = fd;
	listener.Port_ = actualPort;
	return listener;
}

OaResult<OaTcpStream> OaTcpListener::Accept() {
	if (Fd_ < 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "listener is closed");
	}

	struct sockaddr_in clientAddr{};
	socklen_t addrLen = sizeof(clientAddr);
	OaI32 clientFd = static_cast<OaI32>(::accept(static_cast<SOCKET>(Fd_), reinterpret_cast<struct sockaddr*>(&clientAddr), &addrLen));
	if (clientFd < 0) {
		return OaStatus::Error("accept() failed: " + OaString(std::strerror(errno)));
	}

	OaI32 yes = 1;
	::setsockopt(static_cast<SOCKET>(clientFd), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));

	char addrBuf[INET_ADDRSTRLEN];
	::inet_ntop(AF_INET, &clientAddr.sin_addr, addrBuf, sizeof(addrBuf));

	return OaTcpStream(clientFd, OaString(addrBuf), ntohs(clientAddr.sin_port));
}

void OaTcpListener::Close() {
	if (Fd_ >= 0) {
		OaTcpClose(Fd_);
		Fd_ = -1;
	}
}
