#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

inline constexpr oa::StringView kMcpLatestProtocolVersion("2026-07-28", 10);

enum class McpCacheScope : oa::U8 {
  Private,
  Public,
};

struct McpServerConfig {
  oa::String name = "oa";
  oa::String version;
  oa::String instructions;
  oa::Usize maxMessageBytes = 1024U * 1024U;
  oa::U32 maxNestingDepth = 64;
  oa::U32 maxJsonNodes = 65536;
  oa::U32 cacheTtlMs = 0;
  McpCacheScope cacheScope = McpCacheScope::Private;
};

// Tool arguments are validated JSON objects. scalar accessors provide the
// common application boundary; json() and json(name) preserve nested values
// without exposing OA's private protocol parser as a public JSON framework.
class McpArguments {
public:
  [[nodiscard]] oa::StringView json() const noexcept { return json_; }
  [[nodiscard]] bool contains(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::String> json(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::String> string(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::I64> integer(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::U64>
  unsignedInteger(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::F64> number(oa::StringView inName) const;
  [[nodiscard]] oa::Result<oa::Bool> boolean(oa::StringView inName) const;

private:
  friend class McpServer;

  explicit McpArguments(oa::String inJson, oa::U32 inMaxNestingDepth,
                        oa::U32 inMaxJsonNodes)
      : json_(oa::move(inJson)), maxNestingDepth_(inMaxNestingDepth),
        maxJsonNodes_(inMaxJsonNodes) {}

  oa::String json_ = "{}";
  oa::U32 maxNestingDepth_ = 64;
  oa::U32 maxJsonNodes_ = 65536;
};

struct McpToolResult {
  oa::String text;
  // Empty means absent. A non-empty value must contain exactly one valid JSON
  // value and should conform to the registered output schema when present.
  oa::String structuredContentJson;
  oa::Bool isError = false;

  [[nodiscard]] static McpToolResult success(oa::String inText);
  [[nodiscard]] static McpToolResult error(oa::String inText);
};

struct McpTool {
  oa::String name;
  oa::String title;
  oa::String description;
  oa::String inputSchemaJson = R"({"type":"object"})";
  oa::String outputSchemaJson;
  oa::Bool readOnly = false;
  oa::Bool destructive = true;
  oa::Bool idempotent = false;
  oa::Bool openWorld = true;
  oa::Fn<oa::Result<McpToolResult>(const McpArguments &)> call;
};

struct McpTextResource {
  oa::String uri;
  oa::String name;
  oa::String title;
  oa::String description;
  oa::String mimeType = "text/plain";
  oa::Fn<oa::Result<oa::String>()> read;
};

// Stateful MCP protocol session. Registration is frozen by the first handled
// message. handleMessage() is transport-independent; runStdio() adds the
// standard newline-delimited subprocess binding. calls are externally
// synchronized. close() never waits for or drains external I/O.
class McpServer {
public:
  explicit McpServer(McpServerConfig inConfig = {});
  ~McpServer() = default;

  McpServer(const McpServer &) = delete;
  McpServer &operator=(const McpServer &) = delete;
  McpServer(McpServer &&) = delete;
  McpServer &operator=(McpServer &&) = delete;

  [[nodiscard]] oa::Status addTool(McpTool inTool);
  [[nodiscard]] oa::Status addTextResource(McpTextResource inResource);
  [[nodiscard]] bool hasTool(oa::StringView inName) const noexcept;
  [[nodiscard]] bool hasTextResource(oa::StringView inUri) const noexcept;

  // Returns one complete JSON-RPC response. Notifications return an empty
  // string. Protocol errors are encoded as successful JSON-RPC error messages;
  // oa::Result errors are reserved for local lifecycle/I/O failures.
  [[nodiscard]] oa::Result<oa::String> handleMessage(oa::StringView inMessage);
  [[nodiscard]] oa::Status runStdio();
  [[nodiscard]] oa::Status close();

  [[nodiscard]] bool isStarted() const noexcept { return started_; }
  [[nodiscard]] bool isClosed() const noexcept { return closed_; }
  [[nodiscard]] const oa::Status &configurationStatus() const noexcept {
    return configurationStatus_;
  }

private:
  [[nodiscard]] oa::String boundResponse(oa::String inResponse,
                                         oa::StringView inIdJson) const;

  McpServerConfig config_;
  oa::Status configurationStatus_;
  oa::Vec<McpTool> tools_;
  oa::Vec<McpTextResource> resources_;
  oa::String legacyProtocolVersion_;
  oa::Bool legacyInitializeSeen_ = false;
  oa::Bool legacyReady_ = false;
  oa::Bool started_ = false;
  oa::Bool closed_ = false;
};

} // namespace oa
