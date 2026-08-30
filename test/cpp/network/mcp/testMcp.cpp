#include <oa/network/mcp.h>

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace {

constexpr const char *ModernMeta =
    R"("_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}})";

oa::String modernRequest(oa::StringView inMethod, oa::StringView inId,
                       oa::StringView inExtraParams = {}) {
  oa::String request = R"({"jsonrpc":"2.0","id":)";
  request += inId;
  request += R"(,"method":")";
  request += inMethod;
  request += R"(","params":{)";
  request += ModernMeta;
  if (not inExtraParams.empty()) {
    request += ',';
    request += inExtraParams;
  }
  request += "}}";
  return request;
}

std::string handle(oa::McpServer &inServer, oa::StringView inRequest) {
  auto response = inServer.handleMessage(inRequest);
  EXPECT_TRUE(response.isOk()) << response.getStatus().toString().cStr();
  return response.isOk()
      ? std::string(response->data(), response->size())
      : std::string{};
}

oa::McpTool echoTool(oa::String inName = "echo") {
  oa::McpTool tool;
  tool.name = oa::move(inName);
  tool.title = "Typed echo";
  tool.description = "Exercises the transport-independent argument boundary.";
  tool.inputSchemaJson = R"({
		"type": "object",
		"properties": {"text":{"type":"string"},"count":{"type":"integer"},"cursor":{"type":"integer","minimum":0},"scale":{"type":"number"},"enabled":{"type":"boolean"}},
		"required": ["text","count","cursor","scale","enabled"]
	})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"ok":{"type":"boolean"}}})";
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call =
      [](const oa::McpArguments &inArguments) -> oa::Result<oa::McpToolResult> {
    auto text = inArguments.string("text");
    if (text.isError())
      return oa::Result<oa::McpToolResult>(text.getStatus());
    auto count = inArguments.integer("count");
    if (count.isError())
      return oa::Result<oa::McpToolResult>(count.getStatus());
    auto cursor = inArguments.unsignedInteger("cursor");
    if (cursor.isError())
      return oa::Result<oa::McpToolResult>(cursor.getStatus());
    auto scale = inArguments.number("scale");
    if (scale.isError())
      return oa::Result<oa::McpToolResult>(scale.getStatus());
    auto enabled = inArguments.boolean("enabled");
    if (enabled.isError())
      return oa::Result<oa::McpToolResult>(enabled.getStatus());
    if (not inArguments.contains("nested")) {
      return oa::Result<oa::McpToolResult>(
          oa::Status::invalidArgument("nested argument is required"));
    }
    auto nested = inArguments.json("nested");
    if (nested.isError())
      return oa::Result<oa::McpToolResult>(nested.getStatus());
    oa::String output = *text;
    output += ':';
    const std::string countText = std::to_string(*count);
    output += oa::StringView(countText.data(), countText.size());
    output += ':';
    output += *enabled ? "true" : "false";
    output += ':';
    output += *nested;
    EXPECT_DOUBLE_EQ(*scale, 1.25);
    EXPECT_EQ(*cursor, std::numeric_limits<oa::U64>::max());
    EXPECT_EQ(
        inArguments.json(),
        R"({"text":"line\n🚀","count":7,"cursor":18446744073709551615,"scale":1.25,"enabled":true,"nested":{"x":1}})");
    oa::McpToolResult result = oa::McpToolResult::success(oa::move(output));
    result.structuredContentJson = R"({"ok":true})";
    return oa::Result<oa::McpToolResult>(oa::move(result));
  };
  return tool;
}

} // namespace

TEST(Mcp, RegistrationValidatesAndFreezesTheStaticSurface) {
  oa::McpServer server;

  oa::McpTool invalid = echoTool("bad name");
  EXPECT_EQ(server.addTool(oa::move(invalid)).getCode(),
            oa::StatusCode::InvalidArgument);

  oa::McpTool badSchema = echoTool("bad_schema");
  badSchema.inputSchemaJson = R"({"type":"array"})";
  EXPECT_EQ(server.addTool(oa::move(badSchema)).getCode(),
            oa::StatusCode::InvalidArgument);

  EXPECT_TRUE(server.addTool(echoTool()).isOk());
  EXPECT_EQ(server.addTool(echoTool()).getCode(), oa::StatusCode::AlreadyExists);

  oa::McpTextResource invalidResource;
  invalidResource.uri = "not absolute";
  invalidResource.name = "bad";
  invalidResource.read = [] { return oa::Result<oa::String>(oa::String("bad")); };
  EXPECT_EQ(server.addTextResource(oa::move(invalidResource)).getCode(),
            oa::StatusCode::InvalidArgument);

  const std::string response = handle(server, modernRequest("ping", "1"));
  EXPECT_NE(response.find(R"("resultType":"complete")"), std::string::npos);
  EXPECT_TRUE(server.isStarted());
  EXPECT_EQ(server.addTool(echoTool("late")).getCode(),
            oa::StatusCode::FailedPrecondition);
}

TEST(Mcp, CurrentDiscoveryListsCapabilitiesAndStableSortedTools) {
  oa::McpServer server({
      .name = "oa-test",
      .version = "1.2.3",
      .instructions = "Use bounded tools only.",
      .cacheTtlMs = 250,
      .cacheScope = oa::McpCacheScope::Private,
  });
  ASSERT_TRUE(server.addTool(echoTool("zeta")).isOk());
  ASSERT_TRUE(server.addTool(echoTool("alpha")).isOk());

  const std::string discover =
      handle(server, modernRequest("server/discover", "\"d-1\""));
  EXPECT_NE(discover.find(R"("supportedVersions":["2026-07-28","2025-11-25")"),
            std::string::npos);
  EXPECT_NE(discover.find(R"("tools":{"listChanged":false})"),
            std::string::npos);
  EXPECT_NE(discover.find(R"("ttlMs":250,"cacheScope":"private")"),
            std::string::npos);
  EXPECT_NE(discover.find(R"("name":"oa-test","version":"1.2.3")"),
            std::string::npos);

  const std::string list = handle(server, modernRequest("tools/list", "2"));
  const auto alpha = list.find(R"("name":"alpha")");
  const auto zeta = list.find(R"("name":"zeta")");
  ASSERT_NE(alpha, std::string::npos);
  ASSERT_NE(zeta, std::string::npos);
  EXPECT_LT(alpha, zeta);
  EXPECT_NE(list.find(R"("inputSchema":{"type":"object")"), std::string::npos);
  EXPECT_NE(
      list.find(
          R"("readOnlyHint":true,"destructiveHint":false,"idempotentHint":true,"openWorldHint":false)"),
      std::string::npos);
}

TEST(Mcp, ClientImplementationMetadataMustMatchTheProtocolShape) {
  oa::McpServer current;
  const std::string currentResponse = handle(
      current,
      R"({"jsonrpc":"2.0","id":1,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"io.modelcontextprotocol/clientInfo":{"name":"client"}}}})");
  EXPECT_NE(currentResponse.find(R"("code":-32602)"), std::string::npos);
  EXPECT_NE(currentResponse.find("requires non-empty name and version strings"),
            std::string::npos);

  oa::McpServer legacy;
  const std::string legacyResponse = handle(
      legacy,
      R"({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"client","version":""}}})");
  EXPECT_NE(legacyResponse.find(R"("code":-32602)"), std::string::npos);
  EXPECT_NE(legacyResponse.find("requires non-empty name and version strings"),
            std::string::npos);
}

TEST(Mcp, CurrentToolCallUsesTypedArgumentsAndEscapesResults) {
  oa::McpServer server;
  ASSERT_TRUE(server.addTool(echoTool()).isOk());
  const oa::String request = modernRequest(
      "tools/call", "9",
      R"("name":"echo","arguments":{"text":"line\n\ud83d\ude80","count":7,"cursor":18446744073709551615,"scale":1.25,"enabled":true,"nested":{"x":1}})");
  const std::string response = handle(server, request);
  EXPECT_NE(response.find(R"("text":"line\n🚀:7:true:{\"x\":1}")"),
            std::string::npos);
  EXPECT_NE(response.find(R"("structuredContent":{"ok":true})"),
            std::string::npos);
  EXPECT_EQ(response.find(R"("isError":true)"), std::string::npos);
}

TEST(Mcp, HandlerFailuresAreVisibleToolErrors) {
  oa::McpServer server;
  oa::McpTool tool;
  tool.name = "fail";
  tool.call = [](const oa::McpArguments &) -> oa::Result<oa::McpToolResult> {
    return oa::Result<oa::McpToolResult>(oa::Status::error(
        oa::StatusCode::Unavailable, "training owner unavailable"));
  };
  ASSERT_TRUE(server.addTool(oa::move(tool)).isOk());
  const std::string response =
      handle(server, modernRequest("tools/call", "4",
                                   R"("name":"fail","arguments":{})"));
  EXPECT_NE(response.find(R"("isError":true)"), std::string::npos);
  EXPECT_NE(response.find("UNAVAILABLE: training owner unavailable"),
            std::string::npos);
  EXPECT_EQ(response.find(R"("error":{"code")"), std::string::npos);
}

TEST(Mcp, TextResourcesAreSortedAndReadThroughInjectedHandlers) {
  oa::McpServer server;
  oa::McpTextResource zeta;
  zeta.uri = "oa://status/zeta";
  zeta.name = "zeta";
  zeta.read = [] { return oa::Result<oa::String>(oa::String("zeta")); };
  ASSERT_TRUE(server.addTextResource(oa::move(zeta)).isOk());

  oa::McpTextResource alpha;
  alpha.uri = "oa://status/alpha";
  alpha.name = "alpha";
  alpha.title = "alpha status";
  alpha.mimeType = "application/json";
  alpha.read = [] { return oa::Result<oa::String>(oa::String(R"({"step":42})")); };
  ASSERT_TRUE(server.addTextResource(oa::move(alpha)).isOk());

  const std::string list = handle(server, modernRequest("resources/list", "5"));
  const auto alphaPos = list.find("oa://status/alpha");
  const auto zetaPos = list.find("oa://status/zeta");
  ASSERT_NE(alphaPos, std::string::npos);
  ASSERT_NE(zetaPos, std::string::npos);
  EXPECT_LT(alphaPos, zetaPos);

  const std::string read =
      handle(server, modernRequest("resources/read", "6",
                                   R"("uri":"oa://status/alpha")"));
  EXPECT_NE(read.find(R"("mimeType":"application/json")"), std::string::npos);
  EXPECT_NE(read.find(R"("text":"{\"step\":42}")"), std::string::npos);
}

TEST(Mcp, ProtocolErrorsAreBoundedAndDoNotReachHandlers) {
  oa::McpServerConfig config;
  config.maxMessageBytes = 1024;
  config.maxNestingDepth = 8;
  oa::McpServer server(oa::move(config));
  EXPECT_NE(handle(server, "{").find(R"("code":-32700)"), std::string::npos);
  EXPECT_NE(handle(server, R"({"jsonrpc":"2.0","id":null,"method":"ping"})")
                .find(R"("code":-32600)"),
            std::string::npos);
  EXPECT_NE(handle(server, R"({"jsonrpc":"2.0","id":1,"id":2,"method":"ping"})")
                .find(R"("code":-32700)"),
            std::string::npos);

  oa::String oversized;
  oversized.resize(1100, 'x');
  EXPECT_NE(handle(server, oversized).find("configured byte limit"),
            std::string::npos);

  const oa::String unsupported =
      R"({"jsonrpc":"2.0","id":7,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2099-01-01","io.modelcontextprotocol/clientCapabilities":{}}}})";
  const std::string versionError = handle(server, unsupported);
  EXPECT_NE(versionError.find(R"("code":-32022)"), std::string::npos);
  EXPECT_NE(versionError.find(R"("requested":"2099-01-01")"),
            std::string::npos);
}

TEST(Mcp, InvalidConfigurationUtf8DepthAndHandlerOutputFailClosed) {
  oa::McpServerConfig invalidConfig;
  invalidConfig.maxMessageBytes = 1;
  oa::McpServer invalidServer(oa::move(invalidConfig));
  EXPECT_EQ(invalidServer.configurationStatus().getCode(),
            oa::StatusCode::InvalidArgument);
  EXPECT_EQ(invalidServer.addTool(echoTool()).getCode(),
            oa::StatusCode::InvalidArgument);
  EXPECT_TRUE(invalidServer.handleMessage("{}").isError());

  oa::McpServer registrationServer;
  oa::McpTool invalidText = echoTool("invalid_utf8");
  invalidText.description = oa::String("\xC3\x28", 2);
  EXPECT_EQ(registrationServer.addTool(oa::move(invalidText)).getCode(),
            oa::StatusCode::InvalidArgument);

  oa::McpServerConfig depthConfig;
  depthConfig.maxNestingDepth = 8;
  oa::McpServer depthServer(oa::move(depthConfig));
  const std::string tooDeep = handle(
      depthServer,
      R"({"jsonrpc":"2.0","id":1,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}},"nested":[[[[[[[[[0]]]]]]]]]}})");
  EXPECT_NE(tooDeep.find(R"("code":-32700)"), std::string::npos);

  oa::McpServer outputServer;
  oa::McpTool output;
  output.name = "invalid_output";
  output.call = [](const oa::McpArguments &) -> oa::Result<oa::McpToolResult> {
    return oa::Result<oa::McpToolResult>(
        oa::McpToolResult::success(oa::String("\xC3\x28", 2)));
  };
  ASSERT_TRUE(outputServer.addTool(oa::move(output)).isOk());
  const std::string invalidOutput = handle(
      outputServer, modernRequest("tools/call", "8",
                                  R"("name":"invalid_output","arguments":{})"));
  EXPECT_NE(invalidOutput.find(R"("code":-32603)"), std::string::npos);
}

TEST(Mcp, ResponseCeilingAndNotificationsDoNotLeakProtocolState) {
  oa::McpServerConfig config;
  config.maxMessageBytes = 1024;
  oa::McpServer server(oa::move(config));
  oa::McpTool large;
  large.name = "large";
  large.call = [](const oa::McpArguments &) -> oa::Result<oa::McpToolResult> {
    oa::String text;
    text.resize(2048, 'x');
    return oa::Result<oa::McpToolResult>(oa::McpToolResult::success(oa::move(text)));
  };
  ASSERT_TRUE(server.addTool(oa::move(large)).isOk());
  const std::string bounded =
      handle(server, modernRequest("tools/call", "1",
                                   R"("name":"large","arguments":{})"));
  EXPECT_LT(bounded.size(), 1024U);
  EXPECT_NE(bounded.find("response exceeds configured byte limit"),
            std::string::npos);

  const oa::String notification =
      R"({"jsonrpc":"2.0","method":"unknown/notification","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}})";
  EXPECT_TRUE(handle(server, notification).empty());
}

TEST(Mcp, DeterministicAdversarialByteStreamNeverEscapesTheBoundary) {
  oa::McpServerConfig config;
  config.maxMessageBytes = 4096;
  oa::McpServer server(oa::move(config));
  oa::U64 state = 0x4f415f4d43505f31ULL;
  for (oa::Usize iteration = 0; iteration < 4096; ++iteration) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    const oa::Usize length = static_cast<oa::Usize>(state % 129U);
    oa::String bytes;
    bytes.resize(length);
    for (oa::Usize i = 0; i < length; ++i) {
      state ^= state << 13U;
      state ^= state >> 7U;
      state ^= state << 17U;
      bytes[i] = static_cast<char>(state & 0xffU);
    }
    auto handled = server.handleMessage(bytes);
    ASSERT_TRUE(handled.isOk()) << "iteration " << iteration;
    EXPECT_LE(handled->size(), 4096U);
  }
}

TEST(Mcp, LegacyInitializeEraRemainsCompatible) {
  oa::McpServer server;
  ASSERT_TRUE(server.addTool(echoTool()).isOk());
  const std::string initialize = handle(
      server,
      R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}})");
  EXPECT_NE(initialize.find(R"("protocolVersion":"2025-11-25")"),
            std::string::npos);
  EXPECT_EQ(initialize.find("resultType"), std::string::npos);

  const std::string notification = handle(
      server,
      R"({"jsonrpc":"2.0","method":"notifications/initialized","params":{}})");
  EXPECT_TRUE(notification.empty());
  const std::string list = handle(
      server,
      R"({"jsonrpc":"2.0","id":"legacy","method":"tools/list","params":{}})");
  EXPECT_NE(list.find(R"("name":"echo")"), std::string::npos);
  EXPECT_EQ(list.find("resultType"), std::string::npos);
}

TEST(Mcp, CloseIsExplicitAndDoesNotPerformIo) {
  oa::McpServer server;
  EXPECT_TRUE(server.close().isOk());
  EXPECT_TRUE(server.isClosed());
  auto handled =
      server.handleMessage(R"({"jsonrpc":"2.0","id":1,"method":"ping"})");
  EXPECT_TRUE(handled.isError());
  EXPECT_EQ(handled.getStatus().getCode(), oa::StatusCode::FailedPrecondition);
}
