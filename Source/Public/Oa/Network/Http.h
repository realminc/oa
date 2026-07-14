#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Network/Tcp.h>

#include <atomic>
#include <functional>
#include <thread>

class OaHttpRequest {
public:
	OaString Method;
	OaString Path;
	OaString Body;

	[[nodiscard]] OaString Header(const OaString& InName) const;

private:
	OaVec<std::pair<OaString, OaString>> Headers_;
	friend class OaHttpServer;
};

class OaHttpResponse {
public:
	OaI32 Status = 200;
	OaString Body;
	OaString ContentType = "text/plain";

	static OaHttpResponse Ok(const OaString& InBody);
	static OaHttpResponse Json(const OaString& InJson);
	static OaHttpResponse NotFound();
	static OaHttpResponse Error(OaI32 InStatus, const OaString& InMsg);
};

using OaHttpHandler = std::function<OaHttpResponse(const OaHttpRequest&)>;

// OaHttpServer — minimal HTTP/1.1 server with route registration
// Single-threaded accept loop. Good for metrics, health, admin endpoints.
// For high-concurrency APIs, use gRPC directly.
class OaHttpServer {
public:
	static OaResult<OaHttpServer> Create(OaU16 InPort);
	static OaResult<OaHttpServer> Create(const OaString& InHost, OaU16 InPort);

	void Get(const OaString& InPath, OaHttpHandler InHandler);
	void Post(const OaString& InPath, OaHttpHandler InHandler);
	void Route(const OaString& InMethod, const OaString& InPath, OaHttpHandler InHandler);

	OaStatus Serve();
	OaStatus ServeAsync();
	void Stop();
	void Destroy();

	[[nodiscard]] OaU16 Port() const noexcept;
	[[nodiscard]] bool IsRunning() const noexcept { return Running_.load(std::memory_order_relaxed); }

	OaHttpServer() = default;
	~OaHttpServer();
	OaHttpServer(OaHttpServer&& InOther) noexcept;
	OaHttpServer& operator=(OaHttpServer&& InOther) noexcept;
	OaHttpServer(const OaHttpServer&) = delete;
	OaHttpServer& operator=(const OaHttpServer&) = delete;

private:
	class RouteEntry {
	public:
		OaString Method;
		OaString Path;
		OaHttpHandler Handler;
	};

	OaTcpListener Listener_;
	OaVec<RouteEntry> Routes_;
	std::atomic<bool> Running_{false};
	std::thread Thread_;

	void ServeLoop();
	void HandleConnection(OaTcpStream InConn);
	static bool ParseRequest(const OaString& InRaw, OaHttpRequest& OutReq);
	static OaString FormatResponse(const OaHttpResponse& InResp);
};
