#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/network/mcp.h>

namespace oa::sdk::webmcp {

inline constexpr oa::Usize kMaxHeaderBytes = 16U * 1024U;
inline constexpr oa::Usize kMaxBodyBytes = 64U * 1024U;

struct Assets {
  oa::String indexHtml;
  oa::String scriptJavaScript;
  oa::String styleCss;
};

struct HttpRequest {
  oa::String method;
  oa::String path;
  oa::String host;
  oa::String origin;
  oa::String authorization;
  oa::String contentType;
  oa::String body;
};

struct HttpResponse {
  oa::U16 status = 200;
  oa::String reason = "OK";
  oa::String contentType = "text/plain; charset=utf-8";
  oa::String body;
};

struct PublicOrigin {
  oa::String origin;
  oa::String host;
};

// Closed fresh-run surface shared by the native controller, its MCP schema,
// and focused boundary tests. It deliberately contains no path, model graph,
// code, shader, device, or allocation handle.
struct TrainingRunConfig {
  oa::I32 totalSteps = 300;
  oa::I32 contextLength = 16;
  oa::I32 modelWidth = 32;
  oa::I32 hiddenWidth = 64;
  oa::I32 batchSize = 64;
  oa::F32 learningRate = 0.01F;
};

[[nodiscard]] oa::Status validateTrainingRunConfig(
    const TrainingRunConfig& inConfig);

// Parses one canonical HTTPS origin for an external reverse proxy. The native
// listener remains loopback-only; this value only controls exact Host/Origin
// validation and the user-facing launch URL.
[[nodiscard]] oa::Result<PublicOrigin> parsePublicOrigin(oa::StringView inOrigin);

// Returns zero until the complete message size is known. Once the header is
// complete, returns header bytes plus the validated Content-Length.
[[nodiscard]] oa::Result<oa::Usize> expectedMessageBytes(
    oa::StringView inBytes,
    oa::Usize inMaxHeaderBytes = kMaxHeaderBytes,
    oa::Usize inMaxBodyBytes = kMaxBodyBytes);

[[nodiscard]] oa::Result<HttpRequest> parseRequest(
    oa::StringView inBytes,
    oa::Usize inMaxHeaderBytes = kMaxHeaderBytes,
    oa::Usize inMaxBodyBytes = kMaxBodyBytes);

[[nodiscard]] oa::String serializeResponse(const HttpResponse& inResponse);

class Gateway {
public:
  Gateway(
      oa::McpServer& inMcp,
      Assets inAssets,
      oa::String inExpectedHost,
      oa::String inExpectedOrigin,
      oa::String inBearerToken);
  ~Gateway();

  Gateway(const Gateway&) = delete;
  Gateway& operator=(const Gateway&) = delete;
  Gateway(Gateway&&) = delete;
  Gateway& operator=(Gateway&&) = delete;

  [[nodiscard]] HttpResponse handle(const HttpRequest& inRequest);

private:
  [[nodiscard]] bool authorized_(oa::StringView inValue) const noexcept;
  [[nodiscard]] HttpResponse asset_(
      oa::StringView inContentType,
      const oa::String& inBody) const;

  oa::McpServer& mcp_;
  Assets assets_;
  oa::String expectedHost_;
  oa::String expectedOrigin_;
  oa::String bearerToken_;
};

} // namespace oa::sdk::webmcp
