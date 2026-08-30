#include "webmcp.h"

#include <oa/core/std/format.h>

#include <gtest/gtest.h>

namespace {

oa::McpTool statusTool() {
  oa::McpTool tool;
  tool.name = "vulkan_status";
  tool.description = "test status";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{},"additionalProperties":false})";
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call = [](const oa::McpArguments& inArguments) -> oa::Result<oa::McpToolResult> {
    if (not inArguments.empty() or inArguments.size() != 0U) {
      return oa::Status::invalidArgument("vulkan_status accepts no arguments");
    }
    oa::McpToolResult result = oa::McpToolResult::success("ready");
    result.structuredContentJson = R"({"deviceName":"test GPU"})";
    return result;
  };
  return tool;
}

oa::sdk::webmcp::Assets assets() {
  return {.indexHtml = "index", .scriptJavaScript = "script", .styleCss = "style"};
}

oa::String request(
    oa::StringView inMethod,
    oa::StringView inPath,
    oa::StringView inBody = {},
    oa::StringView inExtraHeaders = {},
    oa::StringView inHost = "127.0.0.1:9000") {
  oa::String out = oa::format(
      "{} {} HTTP/1.1\r\nHost: {}\r\n{}",
      inMethod, inPath, inHost, inExtraHeaders);
  if (not inBody.empty()) out += oa::format("Content-Length: {}\r\n", inBody.size());
  out += "\r\n";
  out += inBody;
  return out;
}

oa::sdk::webmcp::HttpRequest parse(const oa::String& inWire) {
  auto parsed = oa::sdk::webmcp::parseRequest(inWire);
  EXPECT_TRUE(parsed.isOk()) << parsed.getStatus().toString().cStr();
  return parsed.isOk() ? oa::move(*parsed) : oa::sdk::webmcp::HttpRequest{};
}

} // namespace

TEST(WebMcpHttp, ParsesOnlyOneBoundedCanonicalRequest) {
  const oa::String valid = request("GET", "/");
  auto expected = oa::sdk::webmcp::expectedMessageBytes(valid);
  ASSERT_TRUE(expected.isOk());
  EXPECT_EQ(*expected, valid.size());
  EXPECT_EQ(parse(valid).host, "127.0.0.1:9000");

  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(valid + "x").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "GET / HTTP/1.0\r\nHost: 127.0.0.1:9000\r\n\r\n").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "GET /?token=x HTTP/1.1\r\nHost: 127.0.0.1:9000\r\n\r\n").isError());
}

TEST(WebMcpTrainingRun, ValidatesEveryResourceBoundary) {
  using oa::sdk::webmcp::TrainingRunConfig;
  using oa::sdk::webmcp::validateTrainingRunConfig;

  EXPECT_TRUE(validateTrainingRunConfig({}).isOk());
  EXPECT_TRUE(validateTrainingRunConfig({
      .totalSteps = 2000,
      .contextLength = 64,
      .modelWidth = 256,
      .hiddenWidth = 1024,
      .batchSize = 128,
      .learningRate = 1.0F,
  }).isOk());

  auto config = TrainingRunConfig{};
  config.totalSteps = 0;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.totalSteps = 2001;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.contextLength = 6;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.modelWidth = 12;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.hiddenWidth = 20;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.batchSize = 129;
  config.contextLength = 64;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.learningRate = 0.0F;
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
  config = {};
  config.learningRate = oa::Limits<oa::F32>::quietNaN();
  EXPECT_TRUE(validateTrainingRunConfig(config).isError());
}

TEST(WebMcpHttp, RejectsFramingAndHeaderSmugglingShapes) {
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "POST /api/mcp HTTP/1.1\r\nHost: 127.0.0.1:9000\r\n"
      "Content-Length: 0\r\nContent-Length: 0\r\n\r\n").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "POST /api/mcp HTTP/1.1\r\nHost: 127.0.0.1:9000\r\n"
      "Transfer-Encoding: chunked\r\n\r\n").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "GET / HTTP/1.1\r\nHost: 127.0.0.1:9000\r\nHost: attacker\r\n\r\n").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      "GET / HTTP/1.1\r\nHost : 127.0.0.1:9000\r\n\r\n").isError());
  const char nulHeader[] =
      "GET / HTTP/1.1\r\nHost: 127.0.0.1:9000\0evil\r\n\r\n";
  EXPECT_TRUE(oa::sdk::webmcp::parseRequest(
      oa::StringView(nulHeader, sizeof(nulHeader) - 1U)).isError());

  const oa::String oversized = oa::format(
      "POST /api/mcp HTTP/1.1\r\nHost: 127.0.0.1:9000\r\nContent-Length: {}\r\n\r\n",
      oa::sdk::webmcp::kMaxBodyBytes + 1U);
  auto size = oa::sdk::webmcp::expectedMessageBytes(oversized);
  ASSERT_TRUE(size.isError());
  EXPECT_EQ(size.getStatus().getCode(), oa::StatusCode::ResourceExhausted);
}

TEST(WebMcpGateway, ServesOnlyFixedAssetsWithSecurityHeaders) {
  oa::McpServer mcp;
  oa::sdk::webmcp::Gateway gateway(
      mcp, assets(), "127.0.0.1:9000", "http://127.0.0.1:9000", "secret");
  const auto index = gateway.handle(parse(request("GET", "/")));
  EXPECT_EQ(index.status, 200U);
  EXPECT_EQ(index.body, "index");

  const oa::String wire = oa::sdk::webmcp::serializeResponse(index);
  EXPECT_NE(wire.find("Content-Security-Policy:"), oa::String::Npos);
  EXPECT_NE(wire.find("frame-ancestors 'none'"), oa::String::Npos);
  EXPECT_NE(wire.find("Cache-Control: no-store"), oa::String::Npos);
  EXPECT_EQ(gateway.handle(parse(request("GET", "/missing"))).status, 404U);
}

TEST(WebMcpGateway, RequiresExactLoopbackHostOriginAndBearerCredential) {
  oa::McpServer mcp;
  ASSERT_TRUE(mcp.addTool(statusTool()).isOk());
  oa::sdk::webmcp::Gateway gateway(
      mcp, assets(), "127.0.0.1:9000", "http://127.0.0.1:9000", "secret");
  const oa::StringView body =
      R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"name":"vulkan_status","arguments":{}}})";
  const oa::StringView headers =
      "Origin: http://127.0.0.1:9000\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\n";

  const auto accepted = gateway.handle(parse(request("POST", "/api/mcp", body, headers)));
  EXPECT_EQ(accepted.status, 200U);
  EXPECT_NE(accepted.body.find(R"("deviceName":"test GPU")"), oa::String::Npos);

  const oa::StringView spacedBody =
      R"({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"name":"vulkan_status","arguments":{ }}})";
  const auto spaced = gateway.handle(parse(request("POST", "/api/mcp", spacedBody, headers)));
  EXPECT_EQ(spaced.status, 200U);
  EXPECT_NE(spaced.body.find(R"("deviceName":"test GPU")"), oa::String::Npos);

  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body,
      "Origin: http://evil.invalid\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\n"))).status, 403U);
  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body,
      "Origin: http://127.0.0.1:9000\r\nAuthorization: Bearer wrong\r\nContent-Type: application/json\r\n"))).status, 401U);
  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body,
      "Origin: http://127.0.0.1:9000\r\nAuthorization: Bearer secret\r\nContent-Type: text/plain\r\n"))).status, 415U);
}

TEST(WebMcpGateway, ParsesAndRequiresOneCanonicalExternalHttpsOrigin) {
  auto publicOrigin = oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com");
  ASSERT_TRUE(publicOrigin.isOk());
  EXPECT_EQ(publicOrigin->origin, "https://webmcp.example.com");
  EXPECT_EQ(publicOrigin->host, "webmcp.example.com");

  auto withPort = oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com:8443");
  ASSERT_TRUE(withPort.isOk());
  EXPECT_EQ(withPort->host, "webmcp.example.com:8443");

  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "http://webmcp.example.com").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin("https://").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com/path").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://user@webmcp.example.com").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://WebMcp.example.com").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com.").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp..example.com").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://-webmcp.example.com").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com:443").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com:0").isError());
  EXPECT_TRUE(oa::sdk::webmcp::parsePublicOrigin(
      "https://webmcp.example.com:65536").isError());

  oa::McpServer mcp;
  ASSERT_TRUE(mcp.addTool(statusTool()).isOk());
  oa::sdk::webmcp::Gateway gateway(
      mcp, assets(), publicOrigin->host, publicOrigin->origin, "secret");
  const oa::StringView body =
      R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"name":"vulkan_status","arguments":{}}})";
  const oa::StringView headers =
      "Origin: https://webmcp.example.com\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\n";

  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body, headers, "webmcp.example.com"))).status, 200U);
  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body, headers, "127.0.0.1:9000"))).status, 421U);
  EXPECT_EQ(gateway.handle(parse(request(
      "POST", "/api/mcp", body,
      "Origin: http://webmcp.example.com\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\n",
      "webmcp.example.com"))).status, 403U);
}
