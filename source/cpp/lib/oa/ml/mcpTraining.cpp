#include <oa/ml/mcpTraining.h>
#include <oa/core/std/format.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

namespace {

void writeJsonString(oa::String &out, oa::StringView inText) {
  static constexpr char Hex[] = "0123456789abcdef";
  out += '"';
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
        out += Hex[c >> 4U];
        out += Hex[c & 0x0FU];
      } else {
        out += static_cast<char>(c);
      }
      break;
    }
  }
  out += '"';
}

void writeFinite(oa::String &out, oa::F64 inValue) {
  if (not oa::isFinite(inValue)) {
    out += "null";
    return;
  }
  oa::String converted;
  if (not oa::formatF64(inValue, converted)) {
    out += "null";
    return;
  }
  out.append(converted.view());
}

[[nodiscard]] oa::StringView stateName(TrainingState inState) {
  switch (inState) {
  case TrainingState::Running:
    return "running";
  case TrainingState::Paused:
    return "paused";
  case TrainingState::Stopping:
    return "stopping";
  case TrainingState::Completed:
    return "completed";
  case TrainingState::Failed:
    return "failed";
  default:
    return "unknown";
  }
}

[[nodiscard]] oa::StringView
dispositionName(TrainingCommandDisposition inDisposition) {
  return inDisposition == TrainingCommandDisposition::Applied
    ? oa::StringView("applied")
    : oa::StringView("rejected");
}

[[nodiscard]] oa::String snapshotJson(const oa::TrainingSession &inSession) {
  const TrainingSnapshot snapshot = inSession.currentSnapshot();
  oa::String json = R"({"revision":)";
  json += oa::toString(snapshot.revision);
  json += R"(,"state":)";
  writeJsonString(json, stateName(snapshot.state));
  json += R"(,"step":)";
  json += oa::toString(snapshot.step);
  json += R"(,"epoch":)";
  json += oa::toString(snapshot.epoch);
  json += R"(,"learningRate":)";
  writeFinite(json, snapshot.learningRate);
  json += R"(,"loss":)";
  writeFinite(json, snapshot.loss);
  json += R"(,"gpuMs":)";
  writeFinite(json, snapshot.gpuMs);
  json += R"(,"wallMs":)";
  writeFinite(json, snapshot.wallMs);
  json += '}';
  return json;
}

[[nodiscard]] oa::String metricsJson(const oa::TrainingSession &inSession) {
  const auto latest = inSession.latestSnapshot();
  oa::String json = R"({"metrics":[)";
  if (latest.hasValue()) {
    for (oa::Usize i = 0; i < latest->metrics.size(); ++i) {
      const auto &metric = latest->metrics[i];
      if (i != 0)
        json += ',';
      json += R"({"name":)";
      writeJsonString(json, metric.name);
      json += R"(,"value":)";
      writeFinite(json, metric.value);
      json += R"(,"step":)";
      json += oa::toString(metric.step);
      json += '}';
    }
  }
  json += "]}";
  return json;
}

[[nodiscard]] oa::Result<oa::U64>
expectedRevision(const McpArguments &inArguments) {
  if (not inArguments.contains("expectedRevision"))
    return oa::Result<oa::U64>(0);
  auto revision = inArguments.unsignedInteger("expectedRevision");
  if (revision.isError())
    return oa::Result<oa::U64>(revision.getStatus());
  return revision;
}

[[nodiscard]] oa::Result<TrainingValue>
trainingValue(const McpArguments &inArguments) {
  if (not inArguments.contains("value")) {
    return oa::Result<TrainingValue>(oa::Status::invalidArgument("value is required"));
  }
  if (auto value = inArguments.boolean("value"); value.isOk()) {
    return oa::Result<TrainingValue>(TrainingValue::fromBool(*value));
  }
  if (auto value = inArguments.integer("value"); value.isOk()) {
    return oa::Result<TrainingValue>(TrainingValue::fromInteger(*value));
  }
  if (auto value = inArguments.number("value"); value.isOk()) {
    return oa::Result<TrainingValue>(TrainingValue::fromFloat(*value));
  }
  if (auto value = inArguments.string("value"); value.isOk()) {
    return oa::Result<TrainingValue>(TrainingValue::fromString(oa::move(*value)));
  }
  return oa::Result<TrainingValue>(oa::Status::invalidArgument("value must be a boolean, integer, finite number or string"));
}

[[nodiscard]] oa::Result<McpToolResult> accepted(oa::Result<oa::U64> inSequence) {
  if (inSequence.isError()) {
    return oa::Result<McpToolResult>(inSequence.getStatus());
  }
  McpToolResult result = McpToolResult::success(
    oa::String("training command accepted with sequence ")
    + oa::toString(*inSequence));
  result.structuredContentJson = R"({"sequence":)";
  result.structuredContentJson += oa::toString(*inSequence);
  result.structuredContentJson += '}';
  return oa::Result<McpToolResult>(oa::move(result));
}

[[nodiscard]] McpTool
readTool(oa::String inName, oa::String inDescription, oa::String inOutputSchema, oa::Fn<oa::Result<McpToolResult>(const McpArguments &)> inCall) {
  McpTool tool;
  tool.name = oa::move(inName);
  tool.description = oa::move(inDescription);
  tool.outputSchemaJson = oa::move(inOutputSchema);
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call = oa::move(inCall);
  return tool;
}

[[nodiscard]] McpTool
commandTool(oa::String inName, oa::String inDescription, oa::String inInputSchema,
            oa::Bool inDestructive,
            oa::Fn<oa::Result<McpToolResult>(const McpArguments &)> inCall) {
  McpTool tool;
  tool.name = oa::move(inName);
  tool.description = oa::move(inDescription);
  tool.inputSchemaJson = oa::move(inInputSchema);
  tool.outputSchemaJson = R"({"type":"object","properties":{"sequence":{"type":"integer"}},"required":["sequence"]})";
  tool.readOnly = false;
  tool.destructive = inDestructive;
  tool.idempotent = false;
  tool.openWorld = false;
  tool.call = oa::move(inCall);
  return tool;
}

} // namespace

oa::Status McpTraining::registerTools(McpServer &inServer, oa::TrainingSession &inSession, McpTrainingConfig inConfig) {
  if (inServer.configurationStatus().isError()) {
    return inServer.configurationStatus();
  }
  if (inServer.isStarted() or inServer.isClosed()) {
    return oa::Status::error(
        oa::StatusCode::FailedPrecondition,
        "training MCP tools must be registered before server start");
  }

  oa::Vector<McpTool> tools;
  tools.pushBack(readTool(
      "training_status",
      "Read the latest immutable training state and scalar timing snapshot.",
      R"({"type":"object","properties":{"revision":{"type":"integer"},"state":{"type":"string"},"step":{"type":"integer"},"epoch":{"type":"integer"},"learningRate":{"type":["number","null"]},"loss":{"type":["number","null"]},"gpuMs":{"type":["number","null"]},"wallMs":{"type":["number","null"]}},"required":["revision","state"]})",
      [&inSession](const McpArguments &) -> oa::Result<McpToolResult> {
        const oa::String json = snapshotJson(inSession);
        McpToolResult result = McpToolResult::success(json);
        result.structuredContentJson = json;
        return oa::Result<McpToolResult>(oa::move(result));
      }));
  tools.pushBack(readTool(
      "training_metrics",
      "Read metric samples from the latest immutable training snapshot.",
      R"({"type":"object","properties":{"metrics":{"type":"array","items":{"type":"object"}}},"required":["metrics"]})",
      [&inSession](const McpArguments &) -> oa::Result<McpToolResult> {
        const oa::String json = metricsJson(inSession);
        McpToolResult result = McpToolResult::success(json);
        result.structuredContentJson = json;
        return oa::Result<McpToolResult>(oa::move(result));
      }));
  tools.pushBack(readTool(
      "training_results",
      "Read the bounded command audit stream after a caller-owned sequence "
      "cursor.",
      R"({"type":"object","properties":{"results":{"type":"array","items":{"type":"object"}}},"required":["results"]})",
      [&inSession](
          const McpArguments &inArguments) -> oa::Result<McpToolResult> {
        oa::U64 after = 0;
        if (inArguments.contains("afterSequence")) {
          auto value = inArguments.unsignedInteger("afterSequence");
          if (value.isError())
            return oa::Result<McpToolResult>(value.getStatus());
          after = *value;
        }
        const auto results = inSession.resultsAfter(after);
        oa::String json = R"({"results":[)";
        for (oa::Usize i = 0; i < results.size(); ++i) {
          const auto &entry = results[i];
          if (i != 0)
            json += ',';
          json += R"({"sequence":)";
          json += oa::toString(entry.sequence);
          json += R"(,"revision":)";
          json += oa::toString(entry.revision);
          json += R"(,"disposition":)";
          writeJsonString(json, dispositionName(entry.disposition));
          json += R"(,"state":)";
          writeJsonString(json, stateName(entry.state));
          json += R"(,"status":)";
          writeJsonString(json, entry.status.toString());
          json += '}';
        }
        json += "]}";
        McpToolResult result = McpToolResult::success(json);
        result.structuredContentJson = json;
        return oa::Result<McpToolResult>(oa::move(result));
      }));
  tools.back().inputSchemaJson =
      R"({"type":"object","properties":{"afterSequence":{"type":"integer","minimum":0}}})";

  if (inConfig.enableCommands) {
    const oa::String revisionSchema =
        R"({"type":"object","properties":{"expectedRevision":{"type":"integer","minimum":0}}})";
    tools.pushBack(commandTool(
        "training_pause", "Pause at the next training safe point.",
        revisionSchema, false, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.pause(*revision));
        }));
    tools.pushBack(commandTool(
        "training_resume",
        "Resume a paused training session at its next safe point.",
        revisionSchema, false, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.resume(*revision));
        }));
    tools.pushBack(commandTool(
        "training_checkpoint",
        "Request a checkpoint at the next training safe point.", revisionSchema,
        false, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.checkpoint(*revision));
        }));
    tools.pushBack(commandTool(
        "training_evaluate",
        "Request evaluation at the next training safe point.", revisionSchema,
        false, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.evaluate(*revision));
        }));
    tools.pushBack(commandTool(
        "training_set_parameter",
        "queue one allowlisted typed parameter update for a training safe "
        "point.",
        R"({"type":"object","properties":{"name":{"type":"string"},"value":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"}]},"expectedRevision":{"type":"integer","minimum":0}},"required":["name","value"]})",
        false, [&inSession](const McpArguments &inArguments) {
          auto name = inArguments.string("name");
          if (name.isError())
            return oa::Result<McpToolResult>(name.getStatus());
          auto value = trainingValue(inArguments);
          if (value.isError())
            return oa::Result<McpToolResult>(value.getStatus());
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.setParameter(oa::move(*name),
                                                 oa::move(*value), *revision));
        }));
    tools.pushBack(commandTool(
        "training_request_recapture",
        "Request graph recapture for a paused training session.",
        revisionSchema, false, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.requestRecapture(*revision));
        }));
  }
  if (inConfig.enableStop) {
    tools.pushBack(commandTool(
        "training_stop",
        "Request terminal training stop at the next safe point.",
        R"({"type":"object","properties":{"expectedRevision":{"type":"integer","minimum":0}}})",
        true, [&inSession](const McpArguments &inArguments) {
          auto revision = expectedRevision(inArguments);
          if (revision.isError()) {
            return oa::Result<McpToolResult>(revision.getStatus());
          }
          return accepted(inSession.stop(*revision));
        }));
  }

  for (const auto &tool : tools) {
    if (inServer.hasTool(tool.name)) {
      return oa::Status::error(oa::StatusCode::AlreadyExists,
                             oa::String("training MCP tool already exists: ") +
                                 tool.name);
    }
  }
  for (auto &tool : tools) {
    const oa::Status status = inServer.addTool(oa::move(tool));
    if (status.isError())
      return status;
  }
  return oa::Status::ok();
}

} // namespace oa
