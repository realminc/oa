#include <Oa/Network/Http.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <sstream>

// OaHttpRequest

OaString OaHttpRequest::Header(const OaString& InName) const {
	for (const auto& [k, v] : Headers_) {
		if (k == InName) return v;
	}
	return {};
}

// OaHttpResponse

OaHttpResponse OaHttpResponse::Ok(const OaString& InBody) {
	return {200, InBody, "text/plain"};
}

OaHttpResponse OaHttpResponse::Json(const OaString& InJson) {
	return {200, InJson, "application/json"};
}

OaHttpResponse OaHttpResponse::NotFound() {
	return {404, "Not Found", "text/plain"};
}

OaHttpResponse OaHttpResponse::Error(OaI32 InStatus, const OaString& InMsg) {
	return {InStatus, InMsg, "text/plain"};
}

// OaHttpServer

OaHttpServer::~OaHttpServer() { Destroy(); }

OaHttpServer::OaHttpServer(OaHttpServer&& InOther) noexcept
	: Listener_(std::move(InOther.Listener_)),
		Routes_(std::move(InOther.Routes_)),
		Running_(InOther.Running_.load()) {
	if (InOther.Thread_.joinable()) {
		InOther.Thread_.detach();
	}
}

OaHttpServer& OaHttpServer::operator=(OaHttpServer&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Listener_ = std::move(InOther.Listener_);
		Routes_ = std::move(InOther.Routes_);
		Running_.store(InOther.Running_.load());
		if (InOther.Thread_.joinable()) {
			InOther.Thread_.detach();
		}
	}
	return *this;
}

OaResult<OaHttpServer> OaHttpServer::Create(OaU16 InPort) {
	return Create("0.0.0.0", InPort);
}

OaResult<OaHttpServer> OaHttpServer::Create(const OaString& InHost, OaU16 InPort) {
	auto listener = OaTcpListener::Bind(InHost, InPort);
	if (!listener.IsOk()) return listener.GetStatus();

	OaHttpServer server;
	server.Listener_ = std::move(listener).GetValue();
	return server;
}

void OaHttpServer::Get(const OaString& InPath, OaHttpHandler InHandler) {
	Route("GET", InPath, std::move(InHandler));
}

void OaHttpServer::Post(const OaString& InPath, OaHttpHandler InHandler) {
	Route("POST", InPath, std::move(InHandler));
}

void OaHttpServer::Route(const OaString& InMethod, const OaString& InPath, OaHttpHandler InHandler) {
	Routes_.PushBack({InMethod, InPath, std::move(InHandler)});
}

OaU16 OaHttpServer::Port() const noexcept {
	return Listener_.Port();
}

OaStatus OaHttpServer::Serve() {
	Running_.store(true, std::memory_order_relaxed);
	ServeLoop();
	return OaStatus::Ok();
}

OaStatus OaHttpServer::ServeAsync() {
	Running_.store(true, std::memory_order_relaxed);
	Thread_ = std::thread([this] { ServeLoop(); });
	return OaStatus::Ok();
}

void OaHttpServer::Stop() {
	Running_.store(false, std::memory_order_relaxed);
	// Close listener to unblock Accept()
	Listener_.Close();
}

void OaHttpServer::Destroy() {
	Stop();
	if (Thread_.joinable()) {
		Thread_.join();
	}
}

void OaHttpServer::ServeLoop() {
	while (Running_.load(std::memory_order_relaxed)) {
		auto connResult = Listener_.Accept();
		if (!connResult.IsOk()) {
			if (!Running_.load(std::memory_order_relaxed)) break;
			continue;
		}
		HandleConnection(std::move(connResult).GetValue());
	}
}

void OaHttpServer::HandleConnection(OaTcpStream InConn) {
	OaVec<OaByte> buf(8192);
	OaI64 n = InConn.Read(buf.Data(), buf.Size());
	if (n <= 0) return;

	OaString raw(reinterpret_cast<const char*>(buf.Data()), static_cast<OaU64>(n));
	OaHttpRequest req;
	if (!ParseRequest(raw, req)) {
		auto resp = FormatResponse(OaHttpResponse::Error(400, "Bad Request"));
		InConn.WriteAll(reinterpret_cast<const OaByte*>(resp.Data()), resp.Size());
		return;
	}

	OaHttpResponse resp = OaHttpResponse::NotFound();
	for (const auto& route : Routes_) {
		if (route.Method == req.Method && route.Path == req.Path) {
			resp = route.Handler(req);
			break;
		}
	}

	auto wire = FormatResponse(resp);
	InConn.WriteAll(reinterpret_cast<const OaByte*>(wire.Data()), wire.Size());
}

bool OaHttpServer::ParseRequest(const OaString& InRaw, OaHttpRequest& OutReq) {
	// Request line: METHOD PATH HTTP/1.x\r\n
	auto lineEnd = InRaw.find("\r\n");
	if (lineEnd == OaString::npos) return false;

	auto reqLine = InRaw.substr(0, lineEnd);
	auto sp1 = reqLine.find(' ');
	if (sp1 == OaString::npos) return false;
	auto sp2 = reqLine.find(' ', sp1 + 1);
	if (sp2 == OaString::npos) return false;

	OutReq.Method = reqLine.substr(0, sp1);
	OutReq.Path = reqLine.substr(sp1 + 1, sp2 - sp1 - 1);

	// Headers
	OaU64 pos = lineEnd + 2;
	while (pos < InRaw.size()) {
		auto next = InRaw.find("\r\n", pos);
		if (next == OaString::npos || next == pos) break;
		auto headerLine = InRaw.substr(pos, next - pos);
		auto colon = headerLine.find(':');
		if (colon != OaString::npos) {
			auto key = headerLine.substr(0, colon);
			auto val = headerLine.substr(colon + 1);
			while (!val.empty() && val[0] == ' ') val = val.substr(1);
			OutReq.Headers_.EmplaceBack(std::move(key), std::move(val));
		}
		pos = next + 2;
	}

	// Body (after \r\n\r\n)
	auto bodyStart = InRaw.find("\r\n\r\n");
	if (bodyStart != OaString::npos && bodyStart + 4 < InRaw.size()) {
		OutReq.Body = InRaw.substr(bodyStart + 4);
	}

	return true;
}

static const char* HttpStatusText(OaI32 InCode) {
	switch (InCode) {
		case 200: return "OK";
		case 400: return "Bad Request";
		case 404: return "Not Found";
		case 500: return "Internal Server Error";
		default: return "Unknown";
	}
}

OaString OaHttpServer::FormatResponse(const OaHttpResponse& InResp) {
	std::ostringstream out;
	out << "HTTP/1.1 " << InResp.Status << " " << HttpStatusText(InResp.Status) << "\r\n";
	out << "Content-Type: " << InResp.ContentType << "\r\n";
	out << "Content-Length: " << InResp.Body.size() << "\r\n";
	out << "Connection: close\r\n";
	out << "\r\n";
	out << InResp.Body;
	return OaString(out.str());
}
