#include "webmcp.h"

#include <oa/core/std/format.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/scalarMath.h>

namespace {

[[nodiscard]] bool asciiCaseEqual(oa::StringView inA, oa::StringView inB) noexcept {
  if (inA.size() != inB.size()) return false;
  for (oa::Usize i = 0; i < inA.size(); ++i) {
    char a = inA[i];
    char b = inB[i];
    if (a >= 'A' and a <= 'Z') a = static_cast<char>(a + ('a' - 'A'));
    if (b >= 'A' and b <= 'Z') b = static_cast<char>(b + ('a' - 'A'));
    if (a != b) return false;
  }
  return true;
}

[[nodiscard]] bool isHeaderNameCharacter(char inCharacter) noexcept {
  const unsigned char ch = static_cast<unsigned char>(inCharacter);
  if ((ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or
      (ch >= '0' and ch <= '9')) return true;
  switch (ch) {
    case '!': case '#': case '$': case '%': case '&': case '\'': case '*':
    case '+': case '-': case '.': case '^': case '_': case '`': case '|': case '~':
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool containsInvalidFieldValueByte(oa::StringView inValue) noexcept {
  for (const char character : inValue) {
    const unsigned char ch = static_cast<unsigned char>(character);
    if ((ch < 0x20U and ch != '\t') or ch == 0x7fU) return true;
  }
  return false;
}

[[nodiscard]] oa::StringView trim(oa::StringView inValue) noexcept {
  oa::Usize begin = 0;
  oa::Usize end = inValue.size();
  while (begin < end and (inValue[begin] == ' ' or inValue[begin] == '\t')) ++begin;
  while (end > begin and (inValue[end - 1U] == ' ' or inValue[end - 1U] == '\t')) --end;
  return inValue.subStr(begin, end - begin);
}

[[nodiscard]] oa::Result<oa::Usize> parseSize(oa::StringView inValue) {
  if (inValue.empty()) {
    return oa::Status::invalidArgument("HTTP Content-Length is empty");
  }
  oa::Usize value = 0;
  for (const char ch : inValue) {
    if (ch < '0' or ch > '9') {
      return oa::Status::invalidArgument("HTTP Content-Length is not decimal");
    }
    const oa::Usize digit = static_cast<oa::Usize>(ch - '0');
    if (value > (static_cast<oa::Usize>(-1) - digit) / 10U) {
      return oa::Status::invalidArgument("HTTP Content-Length overflows");
    }
    value = value * 10U + digit;
  }
  return value;
}

struct HeaderScan {
  oa::Usize headerBytes = 0;
  oa::Usize contentLength = 0;
  bool hasContentLength = false;
};

[[nodiscard]] oa::Result<HeaderScan> scanHeaders(
    oa::StringView inBytes,
    oa::Usize inMaxHeaderBytes,
    oa::Usize inMaxBodyBytes) {
  const oa::Usize delimiter = inBytes.find("\r\n\r\n");
  if (delimiter == oa::StringView::Npos) {
    if (inBytes.size() >= inMaxHeaderBytes) {
      return oa::Status::error(
          oa::StatusCode::ResourceExhausted, "HTTP headers exceed the configured limit");
    }
    return HeaderScan{};
  }
  const oa::Usize headerBytes = delimiter + 4U;
  if (headerBytes > inMaxHeaderBytes) {
    return oa::Status::error(
        oa::StatusCode::ResourceExhausted, "HTTP headers exceed the configured limit");
  }

  const oa::Usize requestLineEnd = inBytes.find("\r\n");
  if (requestLineEnd == oa::StringView::Npos or requestLineEnd == 0U) {
    return oa::Status::invalidArgument("HTTP request line is malformed");
  }

  HeaderScan scan;
  scan.headerBytes = headerBytes;
  oa::Usize cursor = requestLineEnd + 2U;
  while (cursor < delimiter) {
    const oa::Usize lineEnd = inBytes.find("\r\n", cursor);
    if (lineEnd == oa::StringView::Npos or lineEnd > delimiter) {
      return oa::Status::invalidArgument("HTTP header line is malformed");
    }
    const oa::StringView line = inBytes.subStr(cursor, lineEnd - cursor);
    if (line.empty() or line[0] == ' ' or line[0] == '\t') {
      return oa::Status::invalidArgument("HTTP folded or empty header is not accepted");
    }
    const oa::Usize colon = line.find(':');
    if (colon == oa::StringView::Npos or colon == 0U) {
      return oa::Status::invalidArgument("HTTP header lacks a field name");
    }
    const oa::StringView name = line.subStr(0U, colon);
    const oa::StringView value = trim(line.subStr(colon + 1U));
    for (const char character : name) {
      if (not isHeaderNameCharacter(character)) {
        return oa::Status::invalidArgument("HTTP header field name is invalid");
      }
    }
    if (containsInvalidFieldValueByte(value)) {
      return oa::Status::invalidArgument("HTTP header field value contains a control byte");
    }
    if (asciiCaseEqual(name, "Transfer-Encoding")) {
      return oa::Status::invalidArgument("HTTP Transfer-Encoding is not accepted");
    }
    if (asciiCaseEqual(name, "Content-Length")) {
      if (scan.hasContentLength) {
        return oa::Status::invalidArgument("duplicate HTTP Content-Length");
      }
      auto parsed = parseSize(value);
      if (parsed.isError()) return parsed.getStatus();
      if (*parsed > inMaxBodyBytes) {
        return oa::Status::error(
            oa::StatusCode::ResourceExhausted, "HTTP body exceeds the configured limit");
      }
      scan.contentLength = *parsed;
      scan.hasContentLength = true;
    }
    cursor = lineEnd + 2U;
  }
  return scan;
}

[[nodiscard]] oa::Status assignUniqueHeader(
    oa::String& out,
    oa::StringView inName,
    oa::StringView inValue) {
  if (not out.empty()) {
    oa::String message("duplicate HTTP header: ");
    message += inName;
    return oa::Status::invalidArgument(oa::move(message));
  }
  out = oa::String(inValue);
  return oa::Status::ok();
}

[[nodiscard]] oa::sdk::webmcp::HttpResponse response(
    oa::U16 inStatus,
    const char* inReason,
    const char* inBody) {
  oa::sdk::webmcp::HttpResponse result;
  result.status = inStatus;
  result.reason = inReason;
  result.body = inBody;
  return result;
}

} // namespace

oa::Status oa::sdk::webmcp::validateTrainingRunConfig(
    const TrainingRunConfig& inConfig) {
  if (inConfig.totalSteps < 1 or inConfig.totalSteps > 2000) {
    return oa::Status::invalidArgument("totalSteps must be in [1, 2000]");
  }
  if (inConfig.contextLength < 4 or inConfig.contextLength > 64 or
      inConfig.contextLength % 4 != 0) {
    return oa::Status::invalidArgument(
        "contextLength must be a multiple of 4 in [4, 64]");
  }
  if (inConfig.modelWidth < 8 or inConfig.modelWidth > 256 or
      inConfig.modelWidth % 8 != 0) {
    return oa::Status::invalidArgument(
        "modelWidth must be a multiple of 8 in [8, 256]");
  }
  if (inConfig.hiddenWidth < 8 or inConfig.hiddenWidth > 1024 or
      inConfig.hiddenWidth % 8 != 0) {
    return oa::Status::invalidArgument(
        "hiddenWidth must be a multiple of 8 in [8, 1024]");
  }
  if (inConfig.batchSize < 1 or inConfig.batchSize > 256) {
    return oa::Status::invalidArgument("batchSize must be in [1, 256]");
  }
  const oa::I64 tokens = static_cast<oa::I64>(inConfig.batchSize) *
      static_cast<oa::I64>(inConfig.contextLength);
  if (tokens > 8192) {
    return oa::Status::invalidArgument(
        "batchSize * contextLength must not exceed 8192 tokens");
  }
  if (not oa::isFinite(inConfig.learningRate) or
      inConfig.learningRate < 0.000001F or inConfig.learningRate > 1.0F) {
    return oa::Status::invalidArgument(
        "learningRate must be finite and in [0.000001, 1]");
  }
  return oa::Status::ok();
}

oa::Result<oa::sdk::webmcp::PublicOrigin> oa::sdk::webmcp::parsePublicOrigin(
    oa::StringView inOrigin) {
  constexpr oa::StringView scheme("https://", 8U);
  if (inOrigin.size() <= scheme.size() or
      inOrigin.subStr(0U, scheme.size()) != scheme) {
    return oa::Status::invalidArgument(
        "external WebMCP origin must use canonical https://");
  }
  const oa::StringView authority = inOrigin.subStr(scheme.size());
  if (authority.size() > 259U) {
    return oa::Status::invalidArgument("external WebMCP authority is too long");
  }

  oa::Usize colon = oa::StringView::Npos;
  for (oa::Usize i = 0U; i < authority.size(); ++i) {
    const char ch = authority[i];
    if (ch == ':') {
      if (colon != oa::StringView::Npos) {
        return oa::Status::invalidArgument(
            "external WebMCP origin accepts at most one port separator");
      }
      colon = i;
      continue;
    }
    if (not ((ch >= 'a' and ch <= 'z') or (ch >= '0' and ch <= '9') or
             ch == '.' or ch == '-')) {
      return oa::Status::invalidArgument(
          "external WebMCP host must be a lowercase ASCII DNS name or IPv4 address");
    }
  }

  const oa::StringView hostName = colon == oa::StringView::Npos
      ? authority
      : authority.subStr(0U, colon);
  if (hostName.empty() or hostName.size() > 253U or hostName.back() == '.') {
    return oa::Status::invalidArgument("external WebMCP host length is invalid");
  }
  oa::Usize labelStart = 0U;
  while (labelStart < hostName.size()) {
    const oa::Usize dot = hostName.find('.', labelStart);
    const oa::Usize labelEnd = dot == oa::StringView::Npos ? hostName.size() : dot;
    const oa::Usize labelSize = labelEnd - labelStart;
    if (labelSize == 0U or labelSize > 63U or hostName[labelStart] == '-' or
        hostName[labelEnd - 1U] == '-') {
      return oa::Status::invalidArgument("external WebMCP host has an invalid label");
    }
    if (dot == oa::StringView::Npos) break;
    labelStart = dot + 1U;
  }

  if (colon != oa::StringView::Npos) {
    const oa::StringView portText = authority.subStr(colon + 1U);
    if (portText.empty()) {
      return oa::Status::invalidArgument("external WebMCP port is empty");
    }
    oa::U32 port = 0U;
    for (const char ch : portText) {
      if (ch < '0' or ch > '9') {
        return oa::Status::invalidArgument("external WebMCP port is not decimal");
      }
      const oa::U32 digit = static_cast<oa::U32>(ch - '0');
      if (port > (65535U - digit) / 10U) {
        return oa::Status::invalidArgument("external WebMCP port exceeds 65535");
      }
      port = port * 10U + digit;
    }
    if (port == 0U) {
      return oa::Status::invalidArgument("external WebMCP port must be nonzero");
    }
    if (port == 443U) {
      return oa::Status::invalidArgument(
          "omit the default HTTPS port from the external WebMCP origin");
    }
  }

  PublicOrigin result;
  result.origin = oa::String(inOrigin);
  result.host = oa::String(authority);
  return result;
}

oa::Result<oa::Usize> oa::sdk::webmcp::expectedMessageBytes(
    oa::StringView inBytes,
    oa::Usize inMaxHeaderBytes,
    oa::Usize inMaxBodyBytes) {
  auto scan = scanHeaders(inBytes, inMaxHeaderBytes, inMaxBodyBytes);
  if (scan.isError()) return scan.getStatus();
  if (scan->headerBytes == 0U) return static_cast<oa::Usize>(0U);
  if (scan->contentLength > static_cast<oa::Usize>(-1) - scan->headerBytes) {
    return oa::Status::invalidArgument("HTTP message size overflows");
  }
  return scan->headerBytes + scan->contentLength;
}

oa::Result<oa::sdk::webmcp::HttpRequest> oa::sdk::webmcp::parseRequest(
    oa::StringView inBytes,
    oa::Usize inMaxHeaderBytes,
    oa::Usize inMaxBodyBytes) {
  auto expected = expectedMessageBytes(inBytes, inMaxHeaderBytes, inMaxBodyBytes);
  if (expected.isError()) return expected.getStatus();
  if (*expected == 0U or *expected != inBytes.size()) {
    return oa::Status::invalidArgument("HTTP request is incomplete or has trailing bytes");
  }

  const oa::Usize requestLineEnd = inBytes.find("\r\n");
  const oa::StringView requestLine = inBytes.subStr(0U, requestLineEnd);
  const oa::Usize firstSpace = requestLine.find(' ');
  const oa::Usize secondSpace = firstSpace == oa::StringView::Npos
      ? oa::StringView::Npos
      : requestLine.find(' ', firstSpace + 1U);
  if (firstSpace == oa::StringView::Npos or secondSpace == oa::StringView::Npos or
      requestLine.find(' ', secondSpace + 1U) != oa::StringView::Npos or
      requestLine.subStr(secondSpace + 1U) != "HTTP/1.1") {
    return oa::Status::invalidArgument("only a canonical HTTP/1.1 request line is accepted");
  }

  HttpRequest request;
  request.method = oa::String(requestLine.subStr(0U, firstSpace));
  request.path = oa::String(requestLine.subStr(firstSpace + 1U, secondSpace - firstSpace - 1U));
  if (request.path.empty() or request.path[0] != '/' or
      request.path.find('?') != oa::String::Npos or request.path.find('#') != oa::String::Npos) {
    return oa::Status::invalidArgument("HTTP request target must be an absolute path without query or fragment");
  }

  const oa::Usize delimiter = inBytes.find("\r\n\r\n");
  oa::Usize cursor = requestLineEnd + 2U;
  while (cursor < delimiter) {
    const oa::Usize lineEnd = inBytes.find("\r\n", cursor);
    const oa::StringView line = inBytes.subStr(cursor, lineEnd - cursor);
    const oa::Usize colon = line.find(':');
    const oa::StringView name = line.subStr(0U, colon);
    const oa::StringView value = trim(line.subStr(colon + 1U));
    oa::Status status;
    if (asciiCaseEqual(name, "Host")) {
      status = assignUniqueHeader(request.host, "Host", value);
    } else if (asciiCaseEqual(name, "Origin")) {
      status = assignUniqueHeader(request.origin, "Origin", value);
    } else if (asciiCaseEqual(name, "Authorization")) {
      status = assignUniqueHeader(request.authorization, "Authorization", value);
    } else if (asciiCaseEqual(name, "Content-Type")) {
      status = assignUniqueHeader(request.contentType, "Content-Type", value);
    }
    if (status.isError()) return status;
    cursor = lineEnd + 2U;
  }
  if (request.host.empty()) {
    return oa::Status::invalidArgument("HTTP/1.1 Host is required");
  }
  request.body = oa::String(inBytes.subStr(delimiter + 4U));
  return request;
}

oa::String oa::sdk::webmcp::serializeResponse(const HttpResponse& inResponse) {
  oa::String out = oa::format(
      "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: {}\r\n",
      inResponse.status, inResponse.reason, inResponse.body.size(), inResponse.contentType);
  out += "Connection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n";
  out += "Referrer-Policy: no-referrer\r\nCross-Origin-Resource-Policy: same-origin\r\n";
  out += "Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'\r\n\r\n";
  out += inResponse.body;
  return out;
}

oa::sdk::webmcp::Gateway::Gateway(
    oa::McpServer& inMcp,
    Assets inAssets,
    oa::String inExpectedHost,
    oa::String inExpectedOrigin,
    oa::String inBearerToken)
  : mcp_(inMcp),
    assets_(oa::move(inAssets)),
    expectedHost_(oa::move(inExpectedHost)),
    expectedOrigin_(oa::move(inExpectedOrigin)),
    bearerToken_(oa::move(inBearerToken)) {}

oa::sdk::webmcp::Gateway::~Gateway() {
  bearerToken_.secureWipeSecrets();
}

bool oa::sdk::webmcp::Gateway::authorized_(oa::StringView inValue) const noexcept {
  constexpr oa::StringView prefix("Bearer ", 7U);
  if (inValue.size() != prefix.size() + bearerToken_.size()) return false;
  unsigned char difference = 0U;
  for (oa::Usize i = 0; i < prefix.size(); ++i) {
    difference |= static_cast<unsigned char>(inValue[i] ^ prefix[i]);
  }
  for (oa::Usize i = 0; i < bearerToken_.size(); ++i) {
    difference |= static_cast<unsigned char>(inValue[prefix.size() + i] ^ bearerToken_[i]);
  }
  return difference == 0U;
}

oa::sdk::webmcp::HttpResponse oa::sdk::webmcp::Gateway::asset_(
    oa::StringView inContentType,
    const oa::String& inBody) const {
  HttpResponse result;
  result.contentType = oa::String(inContentType);
  result.body = inBody;
  return result;
}

oa::sdk::webmcp::HttpResponse oa::sdk::webmcp::Gateway::handle(
    const HttpRequest& inRequest) {
  if (inRequest.host != expectedHost_) {
    return response(421U, "Misdirected Request", "rejected Host\n");
  }
  if (inRequest.method == "GET") {
    if (inRequest.path == "/" or inRequest.path == "/index.html") {
      return asset_("text/html; charset=utf-8", assets_.indexHtml);
    }
    if (inRequest.path == "/webmcp.js") {
      return asset_("text/javascript; charset=utf-8", assets_.scriptJavaScript);
    }
    if (inRequest.path == "/style.css") {
      return asset_("text/css; charset=utf-8", assets_.styleCss);
    }
    return response(404U, "Not Found", "not found\n");
  }
  if (inRequest.method != "POST" or inRequest.path != "/api/mcp") {
    return response(405U, "Method Not Allowed", "method not allowed\n");
  }
  if (inRequest.origin != expectedOrigin_) {
    return response(403U, "Forbidden", "rejected Origin\n");
  }
  if (not authorized_(inRequest.authorization)) {
    return response(401U, "Unauthorized", "authorization required\n");
  }
  if (inRequest.contentType != "application/json") {
    return response(415U, "Unsupported Media Type", "application/json required\n");
  }
  auto handled = mcp_.handleMessage(inRequest.body);
  if (handled.isError()) {
    return response(500U, "Internal Server Error", "MCP transport failure\n");
  }
  HttpResponse result;
  result.contentType = "application/json; charset=utf-8";
  result.body = oa::move(*handled);
  return result;
}
