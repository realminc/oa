#include <oa/network/mcp.h>

#include <oa/core/std/algo.h>
#include <oa/core/std/format.h>
#include <oa/core/version.h>

#include <stdio.h>

namespace {

enum class JsonKind : oa::U8 {
  Null,
  Boolean,
  Number,
  String,
  Array,
  Object,
};

struct JsonValue;

struct JsonMember {
  oa::String name;
  oa::UniquePtr<JsonValue> value;

  JsonMember(oa::String inName, oa::UniquePtr<JsonValue> inValue) noexcept
    : name(oa::move(inName))
    , value(oa::move(inValue))
  {}
  JsonMember(JsonMember &&) noexcept = default;
  JsonMember &operator=(JsonMember &&) noexcept = default;
  JsonMember(const JsonMember &) = delete;
  JsonMember &operator=(const JsonMember &) = delete;
  ~JsonMember();
};

struct JsonValue {
  JsonKind kind = JsonKind::Null;
  oa::Bool boolean = false;
  oa::String text;
  oa::Vector<oa::UniquePtr<JsonValue>> array;
  oa::Vector<JsonMember> object;
};

JsonMember::~JsonMember() = default;

class JsonParser {
public:
  JsonParser(oa::StringView inText, oa::U32 inMaxDepth, oa::U32 inMaxNodes)
    : text_(inText)
    , maxDepth_(inMaxDepth)
    , maxNodes_(inMaxNodes)
  {}

  [[nodiscard]] oa::Bool parse(JsonValue &outValue) {
    skipWhitespace();
    if (not parseValue(outValue, 0))
      return false;
    skipWhitespace();
    if (pos_ != text_.size())
      return fail("unexpected trailing data");
    return true;
  }

  [[nodiscard]] const oa::String &error() const noexcept { return error_; }

private:
  [[nodiscard]] oa::Bool parseValue(JsonValue &outValue, oa::U32 inDepth) {
    if (++nodes_ > maxNodes_)
      return fail("JSON node limit exceeded");
    if (pos_ >= text_.size())
      return fail("expected a JSON value");

    const char c = text_[pos_];
    if (c == 'n')
      return parseLiteral("null", JsonKind::Null, outValue);
    if (c == 't') {
      if (not parseLiteral("true", JsonKind::Boolean, outValue))
        return false;
      outValue.boolean = true;
      return true;
    }
    if (c == 'f') {
      if (not parseLiteral("false", JsonKind::Boolean, outValue))
        return false;
      outValue.boolean = false;
      return true;
    }
    if (c == '"') {
      outValue.kind = JsonKind::String;
      return parseString(outValue.text);
    }
    if (c == '[')
      return parseArray(outValue, inDepth);
    if (c == '{')
      return parseObject(outValue, inDepth);
    if (c == '-' or (c >= '0' and c <= '9'))
      return parseNumber(outValue);
    return fail("invalid JSON value");
  }

  [[nodiscard]] oa::Bool parseLiteral(oa::StringView inLiteral, JsonKind inKind, JsonValue &outValue) {
    if (text_.subStr(pos_, inLiteral.size()) != inLiteral) {
      return fail("invalid JSON literal");
    }
    pos_ += inLiteral.size();
    outValue.kind = inKind;
    return true;
  }

  [[nodiscard]] oa::Bool parseArray(JsonValue &outValue, oa::U32 inDepth) {
    if (inDepth >= maxDepth_)
      return fail("JSON nesting depth exceeded");
    outValue.kind = JsonKind::Array;
    ++pos_;
    skipWhitespace();
    if (consume(']'))
      return true;
    for (;;) {
      JsonValue item;
      if (not parseValue(item, inDepth + 1))
        return false;
      outValue.array.pushBack(oa::makeUnique<JsonValue>(oa::move(item)));
      skipWhitespace();
      if (consume(']'))
        return true;
      if (not consume(','))
        return fail("expected ',' or ']' in array");
      skipWhitespace();
    }
  }

  [[nodiscard]] oa::Bool parseObject(JsonValue &outValue, oa::U32 inDepth) {
    if (inDepth >= maxDepth_)
      return fail("JSON nesting depth exceeded");
    outValue.kind = JsonKind::Object;
    ++pos_;
    skipWhitespace();
    if (consume('}'))
      return true;
    for (;;) {
      if (pos_ >= text_.size() or text_[pos_] != '"') {
        return fail("expected object member name");
      }
      oa::String name;
      if (not parseString(name))
        return false;
      for (const auto &member : outValue.object) {
        if (member.name == name)
          return fail("duplicate object member");
      }
      skipWhitespace();
      if (not consume(':'))
        return fail("expected ':' after object member name");
      skipWhitespace();
      JsonValue value;
      if (not parseValue(value, inDepth + 1))
        return false;
      outValue.object.emplaceBack(
          oa::move(name), oa::makeUnique<JsonValue>(oa::move(value)));
      skipWhitespace();
      if (consume('}'))
        return true;
      if (not consume(','))
        return fail("expected ',' or '}' in object");
      skipWhitespace();
    }
  }

  [[nodiscard]] oa::Bool parseString(oa::String &outText) {
    if (not consume('"'))
      return fail("expected JSON string");
    while (pos_ < text_.size()) {
      const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
      if (c == '"')
        return true;
      if (c == '\\') {
        if (pos_ >= text_.size())
          return fail("incomplete JSON escape");
        const char escape = text_[pos_++];
        switch (escape) {
        case '"':
          outText.pushBack('"');
          break;
        case '\\':
          outText.pushBack('\\');
          break;
        case '/':
          outText.pushBack('/');
          break;
        case 'b':
          outText.pushBack('\b');
          break;
        case 'f':
          outText.pushBack('\f');
          break;
        case 'n':
          outText.pushBack('\n');
          break;
        case 'r':
          outText.pushBack('\r');
          break;
        case 't':
          outText.pushBack('\t');
          break;
        case 'u': {
          oa::U32 codepoint = 0;
          if (not parseHex4(codepoint))
            return false;
          if (codepoint >= 0xD800U and codepoint <= 0xDBFFU) {
            if (pos_ + 2 > text_.size() or text_[pos_] != '\\' or
                text_[pos_ + 1] != 'u') {
              return fail("high surrogate without low surrogate");
            }
            pos_ += 2;
            oa::U32 low = 0;
            if (not parseHex4(low))
              return false;
            if (low < 0xDC00U or low > 0xDFFFU) {
              return fail("invalid low surrogate");
            }
            codepoint =
                0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (codepoint >= 0xDC00U and codepoint <= 0xDFFFU) {
            return fail("unpaired low surrogate");
          }
          appendUtf8(outText, codepoint);
          break;
        }
        default:
          return fail("invalid JSON escape");
        }
        continue;
      }
      if (c < 0x20U)
        return fail("unescaped control character in string");
      if (c < 0x80U) {
        outText.pushBack(static_cast<char>(c));
        continue;
      }
      --pos_;
      if (not parseUtf8Sequence(outText))
        return false;
    }
    return fail("unterminated JSON string");
  }

  [[nodiscard]] oa::Bool parseUtf8Sequence(oa::String &outText) {
    const oa::Usize start = pos_;
    const unsigned char first = static_cast<unsigned char>(text_[pos_]);
    oa::U32 codepoint = 0;
    oa::U32 count = 0;
    if (first >= 0xC2U and first <= 0xDFU) {
      codepoint = first & 0x1FU;
      count = 2;
    } else if (first >= 0xE0U and first <= 0xEFU) {
      codepoint = first & 0x0FU;
      count = 3;
    } else if (first >= 0xF0U and first <= 0xF4U) {
      codepoint = first & 0x07U;
      count = 4;
    } else {
      return fail("invalid UTF-8 leading byte");
    }
    if (pos_ + count > text_.size())
      return fail("incomplete UTF-8 sequence");
    for (oa::U32 i = 1; i < count; ++i) {
      const unsigned char continuation =
          static_cast<unsigned char>(text_[pos_ + i]);
      if ((continuation & 0xC0U) != 0x80U) {
        return fail("invalid UTF-8 continuation byte");
      }
      codepoint = (codepoint << 6U) | (continuation & 0x3FU);
    }
    if ((count == 3 and codepoint < 0x800U) or
        (count == 4 and codepoint < 0x10000U) or
        (codepoint >= 0xD800U and codepoint <= 0xDFFFU) or
        codepoint > 0x10FFFFU) {
      return fail("invalid UTF-8 code point");
    }
    outText.append(text_.subStr(start, count));
    pos_ += count;
    return true;
  }

  [[nodiscard]] oa::Bool parseHex4(oa::U32 &outValue) {
    if (pos_ + 4 > text_.size())
      return fail("incomplete Unicode escape");
    outValue = 0;
    for (oa::U32 i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      outValue <<= 4U;
      if (c >= '0' and c <= '9')
        outValue |= static_cast<oa::U32>(c - '0');
      else if (c >= 'a' and c <= 'f')
        outValue |= static_cast<oa::U32>(c - 'a' + 10);
      else if (c >= 'A' and c <= 'F')
        outValue |= static_cast<oa::U32>(c - 'A' + 10);
      else
        return fail("invalid Unicode escape");
    }
    return true;
  }

  static void appendUtf8(oa::String &outText, oa::U32 inCodepoint) {
    if (inCodepoint <= 0x7FU) {
      outText.pushBack(static_cast<char>(inCodepoint));
    } else if (inCodepoint <= 0x7FFU) {
      outText.pushBack(static_cast<char>(0xC0U | (inCodepoint >> 6U)));
      outText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
    } else if (inCodepoint <= 0xFFFFU) {
      outText.pushBack(static_cast<char>(0xE0U | (inCodepoint >> 12U)));
      outText.pushBack(
          static_cast<char>(0x80U | ((inCodepoint >> 6U) & 0x3FU)));
      outText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
    } else {
      outText.pushBack(static_cast<char>(0xF0U | (inCodepoint >> 18U)));
      outText.pushBack(
          static_cast<char>(0x80U | ((inCodepoint >> 12U) & 0x3FU)));
      outText.pushBack(
          static_cast<char>(0x80U | ((inCodepoint >> 6U) & 0x3FU)));
      outText.pushBack(static_cast<char>(0x80U | (inCodepoint & 0x3FU)));
    }
  }

  [[nodiscard]] oa::Bool parseNumber(JsonValue &outValue) {
    const oa::Usize start = pos_;
    (void)consume('-');
    if (pos_ >= text_.size())
      return fail("incomplete JSON number");
    if (text_[pos_] == '0') {
      ++pos_;
      if (pos_ < text_.size() and text_[pos_] >= '0' and text_[pos_] <= '9') {
        return fail("leading zero in JSON number");
      }
    } else {
      if (text_[pos_] < '1' or text_[pos_] > '9')
        return fail("invalid JSON number");
      while (pos_ < text_.size() and text_[pos_] >= '0' and text_[pos_] <= '9')
        ++pos_;
    }
    if (pos_ < text_.size() and text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() or text_[pos_] < '0' or text_[pos_] > '9') {
        return fail("missing fraction digits in JSON number");
      }
      while (pos_ < text_.size() and text_[pos_] >= '0' and text_[pos_] <= '9')
        ++pos_;
    }
    if (pos_ < text_.size() and (text_[pos_] == 'e' or text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() and (text_[pos_] == '+' or text_[pos_] == '-'))
        ++pos_;
      if (pos_ >= text_.size() or text_[pos_] < '0' or text_[pos_] > '9') {
        return fail("missing exponent digits in JSON number");
      }
      while (pos_ < text_.size() and text_[pos_] >= '0' and text_[pos_] <= '9')
        ++pos_;
    }
    outValue.kind = JsonKind::Number;
    outValue.text = oa::String(text_.subStr(start, pos_ - start));
    return true;
  }

  void skipWhitespace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c != ' ' and c != '\t' and c != '\r' and c != '\n')
        break;
      ++pos_;
    }
  }

  [[nodiscard]] oa::Bool consume(char inChar) {
    if (pos_ >= text_.size() or text_[pos_] != inChar)
      return false;
    ++pos_;
    return true;
  }

  [[nodiscard]] oa::Bool fail(const char *inMessage) {
    if (error_.empty()) {
      error_ = inMessage;
      error_ += " at byte ";
      error_ += oa::toString(static_cast<oa::U64>(pos_));
    }
    return false;
  }

  oa::StringView text_;
  oa::Usize pos_ = 0;
  oa::U32 maxDepth_ = 0;
  oa::U32 maxNodes_ = 0;
  oa::U32 nodes_ = 0;
  oa::String error_;
};

[[nodiscard]] oa::Result<JsonValue>
parseJson(oa::StringView inText, oa::U32 inMaxDepth, oa::U32 inMaxNodes) {
  JsonValue value;
  JsonParser parser(inText, inMaxDepth, inMaxNodes);
  if (not parser.parse(value)) {
	return oa::Result<JsonValue>(
		oa::Status::invalidArgument(parser.error()));
  }
  return oa::Result<JsonValue>(oa::move(value));
}

void writeJsonString(oa::String &out, oa::StringView inText) {
  static constexpr char Hex[] = "0123456789abcdef";
  out.pushBack('"');
  for (const char byte : inText) {
    const unsigned char c = static_cast<unsigned char>(byte);
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20U) {
        out += "\\u00";
        out.pushBack(Hex[c >> 4U]);
        out.pushBack(Hex[c & 0x0FU]);
      } else {
        out.pushBack(static_cast<char>(c));
      }
      break;
    }
  }
  out.pushBack('"');
}

void writeJson(oa::String &out, const JsonValue &inValue) {
  switch (inValue.kind) {
  case JsonKind::Null:
    out += "null";
    break;
  case JsonKind::Boolean:
    out += inValue.boolean ? "true" : "false";
    break;
  case JsonKind::Number:
    out += inValue.text;
    break;
  case JsonKind::String:
    writeJsonString(out, inValue.text);
    break;
  case JsonKind::Array:
    out.pushBack('[');
    for (oa::Usize i = 0; i < inValue.array.size(); ++i) {
      if (i != 0)
        out.pushBack(',');
      writeJson(out, *inValue.array[i]);
    }
    out.pushBack(']');
    break;
  case JsonKind::Object:
    out.pushBack('{');
    for (oa::Usize i = 0; i < inValue.object.size(); ++i) {
      if (i != 0)
        out.pushBack(',');
      writeJsonString(out, inValue.object[i].name);
      out.pushBack(':');
      writeJson(out, *inValue.object[i].value);
    }
    out.pushBack('}');
    break;
  }
}

[[nodiscard]] oa::String serializeJson(const JsonValue &inValue) {
  oa::String json;
  writeJson(json, inValue);
  return json;
}

[[nodiscard]] const JsonValue *findMember(const JsonValue &inObject, oa::StringView inName) {
  if (inObject.kind != JsonKind::Object)
    return nullptr;
  for (const auto &member : inObject.object) {
    if (member.name == inName)
      return member.value.get();
  }
  return nullptr;
}

[[nodiscard]] const JsonValue *findMemberOa(const JsonValue &inObject, oa::StringView inName) {
  return findMember(inObject, inName);
}

[[nodiscard]] oa::String jsonString(oa::StringView inText) {
  oa::String json;
  writeJsonString(json, inText);
  return json;
}

[[nodiscard]] oa::String makeError(oa::StringView inIdJson, oa::I32 inCode,
                                   oa::StringView inMessage,
                                   oa::StringView inDataJson = {}) {
  oa::String json = R"({"jsonrpc":"2.0","id":)";
  json.append(inIdJson);
  json += R"(,"error":{"code":)";
  json += oa::toString(static_cast<oa::I64>(inCode));
  json += R"(,"message":)";
  writeJsonString(json, inMessage);
  if (not inDataJson.empty()) {
    json += R"(,"data":)";
    json.append(inDataJson);
  }
  json += "}}";
  return json;
}

[[nodiscard]] oa::String makeSuccess(oa::StringView inIdJson, oa::StringView inResultJson) {
  oa::String response = R"({"jsonrpc":"2.0","id":)";
  response += inIdJson;
  response += R"(,"result":)";
  response += inResultJson;
  response += '}';
  return response;
}

[[nodiscard]] oa::Bool isValidToolName(oa::StringView inName) {
  if (inName.empty() or inName.size() > 128)
    return false;
  for (const char c : inName) {
    const oa::Bool valid = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
                           (c >= '0' and c <= '9') or c == '_' or c == '-' or
                           c == '.';
    if (not valid)
      return false;
  }
  return true;
}

[[nodiscard]] oa::Bool isValidUri(oa::StringView inUri) {
  if (inUri.empty())
    return false;
  oa::Usize colon = oa::StringView::Npos;
  for (oa::Usize i = 0; i < inUri.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(inUri[i]);
    if (c <= 0x20U or c == 0x7FU)
      return false;
    if (inUri[i] == ':' and colon == oa::StringView::Npos)
      colon = i;
  }
  if (colon == oa::StringView::Npos or colon == 0)
    return false;
  for (oa::Usize i = 0; i < colon; ++i) {
    const char c = inUri[i];
    if (i == 0 and not((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))) {
      return false;
    }
    if (i > 0 and
        not((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or c == '+' or c == '-' or c == '.')) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] oa::Status validateUtf8Text(oa::StringView inText,
                                          oa::StringView inField) {
  oa::String quoted;
  writeJsonString(quoted, inText);
  const auto parsed =
      parseJson(oa::StringView(quoted.data(), quoted.size()), 1, 1);
  if (parsed.isError()) {
    return oa::Status::invalidArgument(oa::String(inField) +
                                       " must contain valid UTF-8");
  }
  return oa::Status::ok();
}

[[nodiscard]] oa::Result<oa::String>
canonicalSchema(oa::StringView inSchema, oa::Bool inRequireObjectType,
                oa::U32 inMaxDepth, oa::U32 inMaxNodes) {
  auto parsed = parseJson(inSchema, inMaxDepth, inMaxNodes);
  if (parsed.isError()) {
    return oa::Result<oa::String>(
        oa::Status::invalidArgument(oa::String("invalid MCP JSON schema: ") +
                                    parsed.getStatus().getMessage()));
  }
  if (parsed->kind != JsonKind::Object) {
    return oa::Result<oa::String>(
        oa::Status::invalidArgument("MCP JSON schema root must be an object"));
  }
  if (inRequireObjectType) {
    const JsonValue *type = findMember(*parsed, "type");
    if (type == nullptr or type->kind != JsonKind::String or
        type->text != "object") {
      return oa::Result<oa::String>(oa::Status::invalidArgument(
          "MCP tool input schema requires root type 'object'"));
    }
  }
  return oa::Result<oa::String>(serializeJson(*parsed));
}

[[nodiscard]] oa::String serverInfoJson(const oa::McpServerConfig &inConfig) {
  oa::String json = R"({"name":)";
  json += jsonString(inConfig.name);
  json += R"(,"version":)";
  json += jsonString(inConfig.version);
  json += '}';
  return json;
}

[[nodiscard]] oa::String capabilitiesJson(oa::Bool inHasTools,
                                          oa::Bool inHasResources) {
  oa::String json = "{";
  oa::Bool comma = false;
  if (inHasResources) {
    json += R"("resources":{"subscribe":false,"listChanged":false})";
    comma = true;
  }
  if (inHasTools) {
    if (comma)
      json += ',';
    json += R"("tools":{"listChanged":false})";
  }
  json += '}';
  return json;
}

void appendModernResultFields(oa::String &out,
                              const oa::McpServerConfig &inConfig) {
  out +=
      R"("resultType":"complete","_meta":{"io.modelcontextprotocol/serverInfo":)";
  const oa::String serverInfo = serverInfoJson(inConfig);
  out.append(serverInfo);
  out += '}';
}

void appendCacheFields(oa::String &out, const oa::McpServerConfig &inConfig) {
  out += R"(,"ttlMs":)";
  out += oa::toString(inConfig.cacheTtlMs);
  out += R"(,"cacheScope":")";
  out +=
      inConfig.cacheScope == oa::McpCacheScope::Public ? "public" : "private";
  out += '"';
}

[[nodiscard]] oa::String supportedVersionData(oa::StringView inRequested) {
  oa::String data =
      R"({"supported":["2026-07-28","2025-11-25","2025-06-18","2025-03-26","2024-11-05"],"requested":)";
  data += jsonString(inRequested);
  data += '}';
  return data;
}

[[nodiscard]] oa::Bool isLegacyVersion(oa::StringView inVersion) {
  return inVersion == "2025-11-25" or inVersion == "2025-06-18" or
         inVersion == "2025-03-26" or inVersion == "2024-11-05";
}

struct RequestEnvelope {
  const JsonValue *params = nullptr;
  oa::String idJson = "null";
  oa::String method;
  oa::Bool isNotification = false;
};

[[nodiscard]] oa::Result<RequestEnvelope>
parseEnvelope(const JsonValue &inRoot) {
  if (inRoot.kind != JsonKind::Object) {
    return oa::Result<RequestEnvelope>(
        oa::Status::invalidArgument("JSON-RPC message must be an object"));
  }
  const JsonValue *jsonrpc = findMember(inRoot, "jsonrpc");
  const JsonValue *method = findMember(inRoot, "method");
  if (jsonrpc == nullptr or jsonrpc->kind != JsonKind::String or
      jsonrpc->text != "2.0" or method == nullptr or
      method->kind != JsonKind::String or method->text.empty()) {
    return oa::Result<RequestEnvelope>(
        oa::Status::invalidArgument("invalid JSON-RPC request envelope"));
  }
  RequestEnvelope envelope;
  envelope.method = method->text;
  envelope.params = findMember(inRoot, "params");
  if (envelope.params != nullptr and
      envelope.params->kind != JsonKind::Object) {
    return oa::Result<RequestEnvelope>(
        oa::Status::invalidArgument("JSON-RPC params must be an object"));
  }
  const JsonValue *id = findMember(inRoot, "id");
  if (id == nullptr) {
    envelope.isNotification = true;
  } else {
    if (id->kind != JsonKind::String and id->kind != JsonKind::Number) {
      return oa::Result<RequestEnvelope>(oa::Status::invalidArgument(
          "JSON-RPC id must be a string or number"));
    }
    envelope.idJson = serializeJson(*id);
  }
  return oa::Result<RequestEnvelope>(oa::move(envelope));
}

[[nodiscard]] oa::String requestError(const RequestEnvelope &inRequest,
                                      oa::I32 inCode, oa::StringView inMessage,
                                      oa::StringView inData = {}) {
  if (inRequest.isNotification)
    return {};
  return makeError(inRequest.idJson, inCode, inMessage, inData);
}

[[nodiscard]] oa::Result<const JsonValue *>
requiredMember(const JsonValue *inObject, oa::StringView inName,
               JsonKind inKind) {
  if (inObject == nullptr or inObject->kind != JsonKind::Object) {
    return oa::Result<const JsonValue *>(
        oa::Status::invalidArgument("request params are required"));
  }
  const JsonValue *value = findMember(*inObject, inName);
  if (value == nullptr or value->kind != inKind) {
    return oa::Result<const JsonValue *>(oa::Status::invalidArgument(
        oa::String("missing or invalid request parameter: ") +
        oa::String(inName)));
  }
  return oa::Result<const JsonValue *>(value);
}

[[nodiscard]] oa::Status validateImplementation(const JsonValue *inValue,
                                                oa::StringView inContext) {
  if (inValue == nullptr or inValue->kind != JsonKind::Object) {
    return oa::Status::invalidArgument(oa::String(inContext) +
                                       " must be an object");
  }
  const JsonValue *name = findMember(*inValue, "name");
  const JsonValue *version = findMember(*inValue, "version");
  if (name == nullptr or name->kind != JsonKind::String or name->text.empty() or
      version == nullptr or version->kind != JsonKind::String or
      version->text.empty()) {
    return oa::Status::invalidArgument(
        oa::String(inContext) + " requires non-empty name and version strings");
  }
  return oa::Status::ok();
}

[[nodiscard]] oa::Status
validateModernMetadata(const JsonValue *inParams,
                       oa::String &outRequestedVersion) {
  if (inParams == nullptr or inParams->kind != JsonKind::Object) {
    return oa::Status::invalidArgument("current MCP requests require params");
  }
  const JsonValue *meta = findMember(*inParams, "_meta");
  if (meta == nullptr or meta->kind != JsonKind::Object) {
    return oa::Status::invalidArgument(
        "current MCP requests require params._meta");
  }
  const JsonValue *version =
      findMember(*meta, "io.modelcontextprotocol/protocolVersion");
  if (version == nullptr or version->kind != JsonKind::String) {
    return oa::Status::invalidArgument(
        "current MCP request metadata requires protocolVersion");
  }
  outRequestedVersion = version->text;
  const JsonValue *capabilities =
      findMember(*meta, "io.modelcontextprotocol/clientCapabilities");
  if (capabilities == nullptr or capabilities->kind != JsonKind::Object) {
    return oa::Status::invalidArgument(
        "current MCP request metadata requires clientCapabilities");
  }
  const JsonValue *clientInfo =
      findMember(*meta, "io.modelcontextprotocol/clientInfo");
  if (clientInfo != nullptr) {
    const oa::Status info =
        validateImplementation(clientInfo, "current MCP request clientInfo");
    if (info.isError())
      return info;
  }
  return oa::Status::ok();
}

[[nodiscard]] oa::Result<JsonValue> parseArgumentObject(oa::StringView inJson,
                                                        oa::U32 inMaxDepth,
                                                        oa::U32 inMaxNodes) {
  auto parsed = parseJson(inJson, inMaxDepth, inMaxNodes);
  if (parsed.isError())
    return parsed;
  if (parsed->kind != JsonKind::Object) {
    return oa::Result<JsonValue>(
        oa::Status::invalidArgument("MCP arguments must be a JSON object"));
  }
  return parsed;
}

[[nodiscard]] oa::Result<const JsonValue *>
argumentValue(const JsonValue &inObject, oa::StringView inName) {
  const JsonValue *value = findMemberOa(inObject, inName);
  if (value == nullptr) {
    return oa::Result<const JsonValue *>(oa::Status::notFound(
        oa::String("MCP argument not found: ") + oa::String(inName)));
  }
  return oa::Result<const JsonValue *>(value);
}

} // namespace

bool oa::McpArguments::empty() const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  return parsed.isOk() and parsed->object.empty();
}

oa::Usize oa::McpArguments::size() const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  return parsed.isOk() ? parsed->object.size() : 0U;
}

bool oa::McpArguments::contains(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  return parsed.isOk() and findMemberOa(*parsed, inName) != nullptr;
}

oa::Result<oa::String> oa::McpArguments::json(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::String>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::String>(value.getStatus());
  return oa::Result<oa::String>(serializeJson(**value));
}

oa::Result<oa::String> oa::McpArguments::string(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::String>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::String>(value.getStatus());
  if ((*value)->kind != JsonKind::String) {
    return oa::Result<oa::String>(oa::Status::invalidArgument(
        oa::String("MCP argument must be a string: ") + oa::String(inName)));
  }
  return oa::Result<oa::String>((*value)->text);
}

oa::Result<oa::I64> oa::McpArguments::integer(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::I64>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::I64>(value.getStatus());
  if ((*value)->kind != JsonKind::Number) {
    return oa::Result<oa::I64>(oa::Status::invalidArgument(
        oa::String("MCP argument must be an integer: ") + oa::String(inName)));
  }
  oa::I64 result = 0;
  if (not oa::parseI64((*value)->text.view(), result)) {
    return oa::Result<oa::I64>(oa::Status::invalidArgument(
        oa::String("MCP argument is not a representable integer: ") +
        oa::String(inName)));
  }
  return oa::Result<oa::I64>(result);
}

oa::Result<oa::U64>
oa::McpArguments::unsignedInteger(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::U64>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::U64>(value.getStatus());
  if ((*value)->kind != JsonKind::Number) {
    return oa::Result<oa::U64>(oa::Status::invalidArgument(
        oa::String("MCP argument must be an unsigned integer: ") +
        oa::String(inName)));
  }
  oa::U64 result = 0;
  if (not oa::parseU64((*value)->text.view(), result)) {
    return oa::Result<oa::U64>(oa::Status::invalidArgument(
        oa::String("MCP argument is not a representable unsigned integer: ") +
        oa::String(inName)));
  }
  return oa::Result<oa::U64>(result);
}

oa::Result<oa::F64> oa::McpArguments::number(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::F64>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::F64>(value.getStatus());
  if ((*value)->kind != JsonKind::Number) {
    return oa::Result<oa::F64>(oa::Status::invalidArgument(
        oa::String("MCP argument must be a number: ") + oa::String(inName)));
  }
  oa::F64 result = 0.0;
  if (not oa::parseF64((*value)->text.view(), result) or not oa::isFinite(result)) {
    return oa::Result<oa::F64>(oa::Status::invalidArgument(
        oa::String("MCP argument is not a finite number: ") +
        oa::String(inName)));
  }
  return oa::Result<oa::F64>(result);
}

oa::Result<oa::Bool> oa::McpArguments::boolean(oa::StringView inName) const {
  auto parsed = parseArgumentObject(json_, maxNestingDepth_, maxJsonNodes_);
  if (parsed.isError())
    return oa::Result<oa::Bool>(parsed.getStatus());
  auto value = argumentValue(*parsed, inName);
  if (value.isError())
    return oa::Result<oa::Bool>(value.getStatus());
  if ((*value)->kind != JsonKind::Boolean) {
    return oa::Result<oa::Bool>(oa::Status::invalidArgument(
        oa::String("MCP argument must be a boolean: ") + oa::String(inName)));
  }
  return oa::Result<oa::Bool>((*value)->boolean);
}

oa::McpToolResult oa::McpToolResult::success(oa::String inText) {
  oa::McpToolResult result;
  result.text = oa::move(inText);
  return result;
}

oa::McpToolResult oa::McpToolResult::error(oa::String inText) {
  oa::McpToolResult result;
  result.text = oa::move(inText);
  result.isError = true;
  return result;
}

oa::McpServer::McpServer(oa::McpServerConfig inConfig)
    : config_(oa::move(inConfig)) {
  if (config_.name.empty())
    config_.name = "oa";
  if (config_.version.empty())
    config_.version = oa::version();
  configurationStatus_ = validateUtf8Text(config_.name, "MCP server name");
  if (configurationStatus_.isOk()) {
    configurationStatus_ =
        validateUtf8Text(config_.version, "MCP server version");
  }
  if (configurationStatus_.isOk()) {
    configurationStatus_ =
        validateUtf8Text(config_.instructions, "MCP server instructions");
  }
  if (configurationStatus_.isOk() and config_.maxMessageBytes < 1024) {
    configurationStatus_ = oa::Status::invalidArgument(
        "MCP maxMessageBytes must be at least 1024");
  } else if (configurationStatus_.isOk() and
             (config_.maxNestingDepth == 0 or config_.maxNestingDepth > 256)) {
    configurationStatus_ =
        oa::Status::invalidArgument("MCP maxNestingDepth must be in [1, 256]");
  } else if (configurationStatus_.isOk() and config_.maxJsonNodes == 0) {
    configurationStatus_ = oa::Status::invalidArgument(
        "MCP maxJsonNodes must be greater than zero");
  }
}

oa::Status oa::McpServer::addTool(oa::McpTool inTool) {
  if (configurationStatus_.isError())
    return configurationStatus_;
  if (started_ or closed_) {
    return oa::Status::error(oa::StatusCode::FailedPrecondition,
                             "MCP registration is frozen after start or close");
  }
  if (not isValidToolName(inTool.name)) {
    return oa::Status::invalidArgument(
        "MCP tool name must be 1-128 letters, digits, '_', '-' or '.'");
  }
  if (not inTool.call) {
    return oa::Status::invalidArgument("MCP tool requires a call handler");
  }
  if (inTool.readOnly and inTool.destructive) {
    return oa::Status::invalidArgument(
        "read-only MCP tools cannot carry a destructive hint");
  }
  OA_RETURN_IF_ERROR(validateUtf8Text(inTool.title, "MCP tool title"));
  OA_RETURN_IF_ERROR(
      validateUtf8Text(inTool.description, "MCP tool description"));
  auto inputSchema =
      canonicalSchema(inTool.inputSchemaJson, true, config_.maxNestingDepth,
                      config_.maxJsonNodes);
  if (inputSchema.isError())
    return inputSchema.getStatus();
  inTool.inputSchemaJson = oa::move(*inputSchema);
  if (not inTool.outputSchemaJson.empty()) {
    auto outputSchema =
        canonicalSchema(inTool.outputSchemaJson, false, config_.maxNestingDepth,
                        config_.maxJsonNodes);
    if (outputSchema.isError())
      return outputSchema.getStatus();
    inTool.outputSchemaJson = oa::move(*outputSchema);
  }
  for (const auto &tool : tools_) {
    if (tool.name == inTool.name) {
      return oa::Status::error(oa::StatusCode::AlreadyExists,
                               oa::String("MCP tool already registered: ") +
                                   inTool.name);
    }
  }
  auto position = oa::lowerBound(
      tools_.begin(), tools_.end(), inTool.name,
      [](const oa::McpTool &inExisting, const oa::String &inName) {
        return inExisting.name.compare(inName) < 0;
      });
  tools_.insert(position, oa::move(inTool));
  return oa::Status::ok();
}

oa::Status oa::McpServer::addTextResource(oa::McpTextResource inResource) {
  if (configurationStatus_.isError())
    return configurationStatus_;
  if (started_ or closed_) {
    return oa::Status::error(oa::StatusCode::FailedPrecondition,
                             "MCP registration is frozen after start or close");
  }
  if (not isValidUri(inResource.uri)) {
    return oa::Status::invalidArgument(
        "MCP text resource requires a valid absolute URI");
  }
  if (inResource.name.empty() or not inResource.read) {
    return oa::Status::invalidArgument(
        "MCP text resource requires a name and read handler");
  }
  OA_RETURN_IF_ERROR(validateUtf8Text(inResource.uri, "MCP resource URI"));
  OA_RETURN_IF_ERROR(validateUtf8Text(inResource.name, "MCP resource name"));
  OA_RETURN_IF_ERROR(validateUtf8Text(inResource.title, "MCP resource title"));
  OA_RETURN_IF_ERROR(
      validateUtf8Text(inResource.description, "MCP resource description"));
  OA_RETURN_IF_ERROR(
      validateUtf8Text(inResource.mimeType, "MCP resource MIME type"));
  for (const auto &resource : resources_) {
    if (resource.uri == inResource.uri) {
      return oa::Status::error(oa::StatusCode::AlreadyExists,
                               oa::String("MCP resource already registered: ") +
                                   inResource.uri);
    }
  }
  auto position = oa::lowerBound(
      resources_.begin(), resources_.end(), inResource.uri,
      [](const oa::McpTextResource &inExisting, const oa::String &inUri) {
        return inExisting.uri.compare(inUri) < 0;
      });
  resources_.insert(position, oa::move(inResource));
  return oa::Status::ok();
}

bool oa::McpServer::hasTool(oa::StringView inName) const noexcept {
  for (const auto &tool : tools_) {
    if (tool.name == inName)
      return true;
  }
  return false;
}

bool oa::McpServer::hasTextResource(oa::StringView inUri) const noexcept {
  for (const auto &resource : resources_) {
    if (resource.uri == inUri)
      return true;
  }
  return false;
}

oa::String oa::McpServer::boundResponse(oa::String inResponse,
                                        oa::StringView inIdJson) const {
  if (inResponse.size() <= config_.maxMessageBytes)
    return inResponse;
  return makeError(inIdJson, -32603,
                   "MCP response exceeds configured byte limit");
}

oa::Result<oa::String> oa::McpServer::handleMessage(oa::StringView inMessage) {
  if (configurationStatus_.isError()) {
    return oa::Result<oa::String>(configurationStatus_);
  }
  if (closed_) {
    return oa::Result<oa::String>(oa::Status::error(
        oa::StatusCode::FailedPrecondition, "MCP server is closed"));
  }
  started_ = true;
  if (inMessage.size() > config_.maxMessageBytes) {
    return oa::Result<oa::String>(
        makeError("null", -32600, "MCP message exceeds configured byte limit"));
  }

  auto root =
      parseJson(inMessage, config_.maxNestingDepth, config_.maxJsonNodes);
  if (root.isError()) {
    return oa::Result<oa::String>(makeError("null", -32700, "parse error"));
  }
  auto envelopeResult = parseEnvelope(*root);
  if (envelopeResult.isError()) {
    return oa::Result<oa::String>(makeError("null", -32600, "Invalid Request"));
  }
  RequestEnvelope request = oa::move(*envelopeResult);

  if (request.method == "initialize") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    if (legacyInitializeSeen_) {
      return oa::Result<oa::String>(
          requestError(request, -32600, "MCP session is already initialized"));
    }
    auto version =
        requiredMember(request.params, "protocolVersion", JsonKind::String);
    if (version.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, version.getStatus().getMessage()));
    }
    auto capabilities =
        requiredMember(request.params, "capabilities", JsonKind::Object);
    if (capabilities.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, capabilities.getStatus().getMessage()));
    }
    auto clientInfo =
        requiredMember(request.params, "clientInfo", JsonKind::Object);
    if (clientInfo.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, clientInfo.getStatus().getMessage()));
    }
    const oa::Status clientInfoStatus =
        validateImplementation(*clientInfo, "legacy MCP clientInfo");
    if (clientInfoStatus.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, clientInfoStatus.getMessage()));
    }
    const oa::String requested = (*version)->text;
    legacyProtocolVersion_ = isLegacyVersion(requested)
                                 ? requested
                                 : oa::String("2025-11-25");
    legacyInitializeSeen_ = true;
    legacyReady_ = false;

    oa::String result = R"({"protocolVersion":)";
    result += jsonString(legacyProtocolVersion_);
    result += R"(,"capabilities":)";
    result += capabilitiesJson(not tools_.empty(), not resources_.empty());
    result += R"(,"serverInfo":)";
    result += serverInfoJson(config_);
    if (not config_.instructions.empty()) {
      result += R"(,"instructions":)";
      result += jsonString(config_.instructions);
    }
    result += '}';
    return oa::Result<oa::String>(
        boundResponse(makeSuccess(request.idJson, result), request.idJson));
  }

  if (request.method == "notifications/initialized") {
    if (not request.isNotification) {
      return oa::Result<oa::String>(requestError(
          request, -32600, "notifications/initialized must be a notification"));
    }
    if (legacyInitializeSeen_)
      legacyReady_ = true;
    return oa::Result<oa::String>(oa::String{});
  }
  if (request.method == "notifications/cancelled") {
    return oa::Result<oa::String>(oa::String{});
  }

  oa::Bool modern = false;
  if (legacyReady_) {
    modern = false;
  } else {
    oa::String requestedVersion;
    const oa::Status metadata =
        validateModernMetadata(request.params, requestedVersion);
    if (metadata.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, metadata.getMessage()));
    }
    if (requestedVersion != oa::kMcpLatestProtocolVersion) {
      const oa::String data = supportedVersionData(requestedVersion);
      return oa::Result<oa::String>(requestError(
          request, -32022, "Unsupported MCP protocol version", data));
    }
    modern = true;
  }

  if (request.method == "ping") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    oa::String result = "{";
    if (modern)
      appendModernResultFields(result, config_);
    result += '}';
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  if (modern and request.method == "server/discover") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    oa::String result = "{";
    appendModernResultFields(result, config_);
    appendCacheFields(result, config_);
    result +=
        R"(,"supportedVersions":["2026-07-28","2025-11-25","2025-06-18","2025-03-26","2024-11-05"],"capabilities":)";
    const oa::String capabilities =
        capabilitiesJson(not tools_.empty(), not resources_.empty());
    result.append(capabilities);
    if (not config_.instructions.empty()) {
      result += R"(,"instructions":)";
      writeJsonString(result, config_.instructions);
    }
    result += '}';
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  if (request.method == "tools/list") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    oa::String result = "{";
    if (modern)
      appendModernResultFields(result, config_);
    if (modern)
      appendCacheFields(result, config_);
    if (result.size() > 1)
      result += ',';
    result += R"("tools":[)";
    for (oa::Usize i = 0; i < tools_.size(); ++i) {
      const oa::McpTool &tool = tools_[i];
      if (i != 0)
        result += ',';
      result += R"({"name":)";
      writeJsonString(result, tool.name);
      if (not tool.title.empty()) {
        result += R"(,"title":)";
        writeJsonString(result, tool.title);
      }
      if (not tool.description.empty()) {
        result += R"(,"description":)";
        writeJsonString(result, tool.description);
      }
      result += R"(,"inputSchema":)";
      result.append(tool.inputSchemaJson);
      if (not tool.outputSchemaJson.empty()) {
        result += R"(,"outputSchema":)";
        result.append(tool.outputSchemaJson);
      }
      result += R"(,"annotations":{"readOnlyHint":)";
      result += tool.readOnly ? "true" : "false";
      result += R"(,"destructiveHint":)";
      result += tool.destructive ? "true" : "false";
      result += R"(,"idempotentHint":)";
      result += tool.idempotent ? "true" : "false";
      result += R"(,"openWorldHint":)";
      result += tool.openWorld ? "true" : "false";
      result += "}}";
    }
    result += "]}";
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  if (request.method == "tools/call") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    auto name = requiredMember(request.params, "name", JsonKind::String);
    if (name.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, name.getStatus().getMessage()));
    }
    const oa::McpTool *selected = nullptr;
    for (const auto &tool : tools_) {
      if (tool.name == oa::StringView((*name)->text.data(), (*name)->text.size())) {
        selected = &tool;
        break;
      }
    }
    if (selected == nullptr) {
      return oa::Result<oa::String>(
          requestError(request, -32602, "Unknown MCP tool"));
    }
    const JsonValue *arguments = findMember(*request.params, "arguments");
    if (arguments != nullptr and arguments->kind != JsonKind::Object) {
      return oa::Result<oa::String>(requestError(
          request, -32602, "MCP tool arguments must be an object"));
    }
    oa::McpArguments callArguments(
        arguments == nullptr ? oa::String("{}") : serializeJson(*arguments),
        config_.maxNestingDepth, config_.maxJsonNodes);
    auto called = selected->call(callArguments);
    oa::McpToolResult toolResult;
    if (called.isError()) {
      toolResult = oa::McpToolResult::error(called.getStatus().toString());
    } else {
      toolResult = oa::move(*called);
    }
    if (const oa::Status status =
            validateUtf8Text(toolResult.text, "MCP tool result text");
        status.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32603, status.getMessage()));
    }

    oa::String structured;
    if (not toolResult.structuredContentJson.empty()) {
      auto parsed = parseJson(toolResult.structuredContentJson,
                              config_.maxNestingDepth, config_.maxJsonNodes);
      if (parsed.isError()) {
        return oa::Result<oa::String>(requestError(
            request, -32603, "MCP tool returned invalid structured JSON"));
      }
      structured = serializeJson(*parsed);
    }

    oa::String result = "{";
    if (modern)
      appendModernResultFields(result, config_);
    if (result.size() > 1)
      result += ',';
    result += R"("content":[{"type":"text","text":)";
    writeJsonString(result, toolResult.text);
    result += "}]";
    if (not structured.empty()) {
      result += R"(,"structuredContent":)";
      result.append(structured);
    }
    if (toolResult.isError)
      result += R"(,"isError":true)";
    result += '}';
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  if (request.method == "resources/list") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    oa::String result = "{";
    if (modern)
      appendModernResultFields(result, config_);
    if (modern)
      appendCacheFields(result, config_);
    if (result.size() > 1)
      result += ',';
    result += R"("resources":[)";
    for (oa::Usize i = 0; i < resources_.size(); ++i) {
      const oa::McpTextResource &resource = resources_[i];
      if (i != 0)
        result += ',';
      result += R"({"uri":)";
      writeJsonString(result, resource.uri);
      result += R"(,"name":)";
      writeJsonString(result, resource.name);
      if (not resource.title.empty()) {
        result += R"(,"title":)";
        writeJsonString(result, resource.title);
      }
      if (not resource.description.empty()) {
        result += R"(,"description":)";
        writeJsonString(result, resource.description);
      }
      if (not resource.mimeType.empty()) {
        result += R"(,"mimeType":)";
        writeJsonString(result, resource.mimeType);
      }
      result += '}';
    }
    result += "]}";
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  if (request.method == "resources/read") {
    if (request.isNotification)
      return oa::Result<oa::String>(oa::String{});
    auto uri = requiredMember(request.params, "uri", JsonKind::String);
    if (uri.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32602, uri.getStatus().getMessage()));
    }
    const oa::McpTextResource *selected = nullptr;
    for (const auto &resource : resources_) {
      if (resource.uri == oa::StringView((*uri)->text.data(), (*uri)->text.size())) {
        selected = &resource;
        break;
      }
    }
    if (selected == nullptr) {
      return oa::Result<oa::String>(
          requestError(request, -32602, "Unknown MCP resource URI"));
    }
    auto content = selected->read();
    if (content.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32603, content.getStatus().toString()));
    }
    if (const oa::Status status =
            validateUtf8Text(*content, "MCP resource text");
        status.isError()) {
      return oa::Result<oa::String>(
          requestError(request, -32603, status.getMessage()));
    }
    oa::String result = "{";
    if (modern)
      appendModernResultFields(result, config_);
    if (modern)
      appendCacheFields(result, config_);
    if (result.size() > 1)
      result += ',';
    result += R"("contents":[{"uri":)";
    writeJsonString(result, selected->uri);
    if (not selected->mimeType.empty()) {
      result += R"(,"mimeType":)";
      writeJsonString(result, selected->mimeType);
    }
    result += R"(,"text":)";
    writeJsonString(result, *content);
    result += "}]}";
    return oa::Result<oa::String>(boundResponse(
        makeSuccess(request.idJson, result), request.idJson));
  }

  return oa::Result<oa::String>(
      requestError(request, -32601, "method not found"));
}

oa::Status oa::McpServer::runStdio() {
  if (configurationStatus_.isError())
    return configurationStatus_;
  if (closed_) {
    return oa::Status::error(oa::StatusCode::FailedPrecondition,
                             "MCP server is closed");
  }
  oa::String line;
  line.reserve(oa::min<oa::Usize>(config_.maxMessageBytes, 64U * 1024U));
  oa::Bool oversized = false;
  for (;;) {
    const int value = ::fgetc(stdin);
    if (value == EOF) {
      if (::ferror(stdin) != 0) {
        return oa::Status::error(oa::StatusCode::Unavailable,
                                 "failed to read MCP stdio input");
      }
      if (line.empty() and not oversized)
        return oa::Status::ok();
    }
    if (value != EOF and value != '\n') {
      if (not oversized) {
        if (line.size() >= config_.maxMessageBytes)
          oversized = true;
        else
          line.pushBack(static_cast<char>(value));
      }
      continue;
    }
    if (not line.empty() and line.back() == '\r')
      line.popBack();

    oa::String response;
    if (oversized) {
      response = makeError("null", -32600,
                           "MCP message exceeds configured byte limit");
    } else {
      auto handled = handleMessage(line);
      if (handled.isError())
        return handled.getStatus();
      response = oa::move(*handled);
    }
    if (not response.empty()) {
      if (::fwrite(response.data(), 1, response.size(), stdout) !=
              response.size() or
          ::fputc('\n', stdout) == EOF or ::fflush(stdout) != 0) {
        return oa::Status::error(oa::StatusCode::Unavailable,
                                 "failed to write MCP stdio output");
      }
    }
    line.clear();
    oversized = false;
    if (closed_ or value == EOF)
      return oa::Status::ok();
  }
}

oa::Status oa::McpServer::close() {
  closed_ = true;
  return oa::Status::ok();
}
