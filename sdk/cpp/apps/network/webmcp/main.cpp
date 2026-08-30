#include "webAssets.gen.h"
#include "webmcp.h"

#include <ml/nlpSuite.h>
#include <oa/core/device.h>
#include <oa/core/filesystem.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/paths.h>
#include <oa/core/std/array.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/format.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/print.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/sync.h>
#include <oa/core/thread.h>
#include <oa/core/version.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/itTraining.h>
#include <oa/ml/optim.h>
#include <oa/ml/trainingSession.h>
#include <oa/network/tcp.h>
#include <oa/runtime/engine.h>

#include <errno.h>

#if defined(OA_PLATFORM_WINDOWS)
  #include <bcrypt.h>
#else
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace {

struct Options {
  oa::U16 port = 0U;
  oa::U64 maxRequests = 0U;
  oa::String externalOrigin;
  oa::String externalHost;
};

[[nodiscard]] oa::Result<oa::U64> parseUnsigned(oa::StringView inText) {
  if (inText.empty()) return oa::Status::invalidArgument("empty integer option");
  oa::U64 value = 0U;
  for (const char ch : inText) {
    if (ch < '0' or ch > '9') {
      return oa::Status::invalidArgument("integer option is not decimal");
    }
    const oa::U64 digit = static_cast<oa::U64>(ch - '0');
    if (value > (static_cast<oa::U64>(-1) - digit) / 10U) {
      return oa::Status::invalidArgument("integer option overflows");
    }
    value = value * 10U + digit;
  }
  return value;
}

[[nodiscard]] oa::Result<Options> parseOptions(int inArgc, char** inArgv) {
  Options options;
  for (int i = 1; i < inArgc; ++i) {
    const oa::StringView argument(inArgv[i]);
    if (argument == "--external-origin" and i + 1 < inArgc) {
      if (not options.externalOrigin.empty()) {
        return oa::Status::invalidArgument("duplicate --external-origin");
      }
      auto parsed = oa::sdk::webmcp::parsePublicOrigin(oa::StringView(inArgv[++i]));
      if (parsed.isError()) return parsed.getStatus();
      options.externalOrigin = oa::move(parsed->origin);
      options.externalHost = oa::move(parsed->host);
      continue;
    }
    if ((argument == "--port" or argument == "--max-requests") and i + 1 < inArgc) {
      auto parsed = parseUnsigned(oa::StringView(inArgv[++i]));
      if (parsed.isError()) return parsed.getStatus();
      if (argument == "--port") {
        if (*parsed > 65535U) return oa::Status::invalidArgument("--port exceeds 65535");
        options.port = static_cast<oa::U16>(*parsed);
      } else {
        options.maxRequests = *parsed;
      }
      continue;
    }
    return oa::Status::invalidArgument(
        "usage: oa-webmcp [--port 0..65535] [--max-requests N] "
        "[--external-origin https://host[:port]]");
  }
  return options;
}

[[nodiscard]] oa::Result<oa::Array<oa::Byte, 32>> secureRandomBytes() {
  oa::Array<oa::Byte, 32> bytes{};
#if defined(OA_PLATFORM_WINDOWS)
  const NTSTATUS status = BCryptGenRandom(
      nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
      BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (status != 0) {
    return oa::Status::error(
        oa::StatusCode::Unavailable, "operating-system random source failed");
  }
#else
  int fd = -1;
  do {
    fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  } while (fd < 0 and errno == EINTR);
  if (fd < 0) {
    return oa::Status::error(
        oa::StatusCode::Unavailable, "operating-system random source is unavailable");
  }
  oa::Usize offset = 0U;
  while (offset < bytes.size()) {
    const auto count = ::read(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 and errno == EINTR) continue;
    if (count <= 0) {
      (void)::close(fd);
      oa::memzeroSecure(bytes.data(), bytes.size());
      return oa::Status::error(
          oa::StatusCode::Unavailable, "operating-system random source failed");
    }
    offset += static_cast<oa::Usize>(count);
  }
  (void)::close(fd);
#endif
  return bytes;
}

[[nodiscard]] oa::Result<oa::String> makeToken() {
  auto random = secureRandomBytes();
  if (random.isError()) return random.getStatus();
  constexpr char hex[] = "0123456789abcdef";
  oa::String token;
  token.reserve(random->size() * 2U);
  for (const oa::Byte byte : *random) {
    token += hex[(byte >> 4U) & 0x0fU];
    token += hex[byte & 0x0fU];
  }
  oa::memzeroSecure(random->data(), random->size());
  return token;
}

void appendJsonString(oa::String& out, oa::StringView inText) {
  constexpr char hex[] = "0123456789abcdef";
  out += '"';
  for (const char ch : inText) {
    const unsigned char byte = static_cast<unsigned char>(ch);
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (byte < 0x20U) {
          out += "\\u00";
          out += hex[(byte >> 4U) & 0x0fU];
          out += hex[byte & 0x0fU];
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

struct TrainingArtifactSnapshot {
  oa::Bool checkpointReady = false;
  oa::Bool evaluationReady = false;
  oa::Bool qualified = false;
  oa::Bool checkpointRoundTrip = false;
  oa::Bool generationQualityPassed = false;
  oa::F32 accuracy = 0.0F;
  oa::U64 optimizerStep = 0U;
  oa::U64 parameterHash = 0U;
  oa::String generated;
  oa::String failure;
};

class TrainingArtifacts {
public:
  void reset() {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    snapshot_ = {};
  }

  void recordCheckpoint(oa::U64 inOptimizerStep) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    snapshot_.checkpointReady = true;
    snapshot_.optimizerStep = inOptimizerStep;
    snapshot_.failure.clear();
  }

  void recordEvaluation(oa::F32 inAccuracy) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    snapshot_.evaluationReady = true;
    snapshot_.accuracy = inAccuracy;
    snapshot_.failure.clear();
  }

  void recordQualification(TrainingArtifactSnapshot inSnapshot) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    snapshot_ = oa::move(inSnapshot);
  }

  void recordFailure(const oa::Status& inStatus) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    snapshot_.qualified = false;
    snapshot_.failure = inStatus.toString();
  }

  [[nodiscard]] TrainingArtifactSnapshot snapshot() const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    return snapshot_;
  }

private:
  mutable oa::Mutex mutex_;
  TrainingArtifactSnapshot snapshot_;
};

using TrainingLabConfig = oa::sdk::webmcp::TrainingRunConfig;

[[nodiscard]] oa::Status validateTrainingLabConfig(
    const TrainingLabConfig& inConfig) {
  return oa::sdk::webmcp::validateTrainingRunConfig(inConfig);
}

[[nodiscard]] oa::StringView trainingStateName(oa::TrainingState inState) {
  switch (inState) {
    case oa::TrainingState::Running: return "running";
    case oa::TrainingState::Paused: return "paused";
    case oa::TrainingState::Stopping: return "stopping";
    case oa::TrainingState::Completed: return "completed";
    case oa::TrainingState::Failed: return "failed";
  }
  return "unknown";
}

void appendFinite(oa::String& out, oa::F64 inValue) {
  if (not oa::isFinite(inValue)) {
    out += "null";
    return;
  }
  oa::String converted;
  if (not oa::formatF64(inValue, converted)) {
    out += "null";
    return;
  }
  out += converted;
}

class TrainingLabController {
public:
  explicit TrainingLabController(TrainingArtifacts& inArtifacts)
      : artifacts_(inArtifacts) {}

  void publishRun(
      oa::TrainingSession& inSession,
      const TrainingLabConfig& inConfig,
      oa::U64 inParameterCount) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    ++runId_;
    session_ = &inSession;
    config_ = inConfig;
    parameterCount_ = inParameterCount;
    cachedSnapshot_ = inSession.currentSnapshot();
    startAccepted_ = false;
    artifacts_.reset();
  }

  void unpublishRun(oa::TrainingSession& inSession) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    if (session_ == &inSession) {
      cachedSnapshot_ = inSession.currentSnapshot();
      session_ = nullptr;
    }
  }

  [[nodiscard]] oa::String statusJson() const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    const oa::TrainingSnapshot snapshot = session_ != nullptr
        ? session_->currentSnapshot()
        : cachedSnapshot_;
    oa::String json = oa::format(
        "{{\"runId\":{},\"revision\":{},\"state\":",
        runId_, snapshot.revision);
    appendJsonString(json, trainingStateName(snapshot.state));
    json += oa::format(
        ",\"step\":{},\"epoch\":{},\"totalSteps\":{},"
        "\"contextLength\":{},\"modelWidth\":{},\"hiddenWidth\":{},"
        "\"batchSize\":{},\"parameterCount\":{},\"restartPending\":{},"
        "\"learningRate\":",
        snapshot.step, snapshot.epoch, config_.totalSteps,
        config_.contextLength, config_.modelWidth, config_.hiddenWidth,
        config_.batchSize, parameterCount_, pendingConfig_.hasValue());
    appendFinite(json, snapshot.learningRate);
    json += ",\"loss\":";
    appendFinite(json, snapshot.loss);
    json += ",\"gpuMs\":";
    appendFinite(json, snapshot.gpuMs);
    json += ",\"wallMs\":";
    appendFinite(json, snapshot.wallMs);
    json += '}';
    return json;
  }

  [[nodiscard]] oa::String metricsJson() const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    oa::String json = oa::format("{{\"runId\":{},\"metrics\":[", runId_);
    const auto latest = session_ != nullptr ? session_->latestSnapshot()
                                            : oa::Optional<oa::TrainingSnapshot>{};
    if (latest.hasValue()) {
      for (oa::Usize index = 0U; index < latest->metrics.size(); ++index) {
        if (index != 0U) json += ',';
        const auto& metric = latest->metrics[index];
        json += "{\"name\":";
        appendJsonString(json, metric.name);
        json += ",\"value\":";
        appendFinite(json, metric.value);
        json += oa::format(",\"step\":{}}}", metric.step);
      }
    }
    json += "]}";
    return json;
  }

  [[nodiscard]] oa::Result<oa::String> resultsJson(oa::U64 inAfter) const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    oa::String json = oa::format("{{\"runId\":{},\"results\":[", runId_);
    if (session_ != nullptr) {
      const auto results = session_->resultsAfter(inAfter);
      for (oa::Usize index = 0U; index < results.size(); ++index) {
        if (index != 0U) json += ',';
        const auto& entry = results[index];
        json += oa::format(
            "{{\"sequence\":{},\"revision\":{},\"disposition\":",
            entry.sequence, entry.revision);
        appendJsonString(
            json, entry.disposition == oa::TrainingCommandDisposition::Applied
                ? oa::StringView("applied") : oa::StringView("rejected"));
        json += ",\"state\":";
        appendJsonString(json, trainingStateName(entry.state));
        json += ",\"status\":";
        appendJsonString(json, entry.status.toString());
        json += '}';
      }
    }
    json += "]}";
    return json;
  }

  [[nodiscard]] oa::Result<oa::U64> start() {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    const auto snapshot = session_->currentSnapshot();
    if (snapshot.state != oa::TrainingState::Paused or snapshot.step != 0) {
      return oa::Status::error(
          oa::StatusCode::FailedPrecondition,
          "training_start requires the initial paused run");
    }
    if (startAccepted_) {
      return oa::Status::error(
          oa::StatusCode::FailedPrecondition,
          "training_start was already accepted for this run");
    }
    auto sequence = session_->resume(session_->revision());
    if (sequence.isOk()) startAccepted_ = true;
    return sequence;
  }

  [[nodiscard]] oa::Result<oa::U64> pause(oa::U64 inRevision) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    return session_->pause(inRevision);
  }

  [[nodiscard]] oa::Result<oa::U64> resume(oa::U64 inRevision) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    return session_->resume(inRevision);
  }

  [[nodiscard]] oa::Result<oa::U64> checkpoint(oa::U64 inRevision) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    return session_->checkpoint(inRevision);
  }

  [[nodiscard]] oa::Result<oa::U64> evaluate(oa::U64 inRevision) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    return session_->evaluate(inRevision);
  }

  [[nodiscard]] oa::Result<oa::U64> setLearningRate(
      oa::F64 inValue,
      oa::U64 inRevision) {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    OA_RETURN_IF_ERROR(requireSessionLocked_());
    return session_->setParameter(
        "learning_rate", oa::TrainingValue::fromFloat(inValue), inRevision);
  }

  [[nodiscard]] oa::Result<oa::U64> restart(TrainingLabConfig inConfig) {
    OA_RETURN_IF_ERROR(validateTrainingLabConfig(inConfig));
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    if (shutdown_) {
      return oa::Status::error(
          oa::StatusCode::FailedPrecondition, "Training Lab is shutting down");
    }
    if (pendingConfig_.hasValue()) {
      return oa::Status::error(
          oa::StatusCode::AlreadyExists, "a training restart is already pending");
    }
    pendingConfig_ = inConfig;
    const oa::U64 request = ++restartRequest_;
    artifacts_.reset();
    if (session_ != nullptr) {
      const oa::TrainingState state = session_->state();
      if (state == oa::TrainingState::Running or
          state == oa::TrainingState::Paused) {
        auto stop = session_->stop(session_->revision());
        if (stop.isError()) {
          pendingConfig_.reset();
          return stop.getStatus();
        }
      }
    }
    wake_.notifyAll();
    return request;
  }

  [[nodiscard]] bool restartPending() const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    return pendingConfig_.hasValue();
  }

  [[nodiscard]] oa::Optional<TrainingLabConfig> waitNextConfig() {
    oa::UniqueLock<oa::Mutex> lock(mutex_);
    wake_.wait(lock, [this] { return shutdown_ or pendingConfig_.hasValue(); });
    if (shutdown_) return {};
    TrainingLabConfig config = *pendingConfig_;
    pendingConfig_.reset();
    return config;
  }

  void shutdown() {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    if (shutdown_) return;
    shutdown_ = true;
    if (session_ != nullptr) {
      const oa::TrainingState state = session_->state();
      if (state == oa::TrainingState::Running or
          state == oa::TrainingState::Paused) {
        (void)session_->stop();
      }
    }
    wake_.notifyAll();
  }

  [[nodiscard]] bool isShuttingDown() const {
    oa::ScopedLock<oa::Mutex> lock(mutex_);
    return shutdown_;
  }

private:
  [[nodiscard]] oa::Status requireSessionLocked_() const {
    return session_ != nullptr
        ? oa::Status::ok()
        : oa::Status::error(
            oa::StatusCode::Unavailable, "no Training Lab run is available");
  }

  TrainingArtifacts& artifacts_;
  mutable oa::Mutex mutex_;
  oa::Condition wake_;
  oa::TrainingSession* session_ = nullptr;
  TrainingLabConfig config_;
  oa::Optional<TrainingLabConfig> pendingConfig_;
  oa::TrainingSnapshot cachedSnapshot_;
  oa::U64 runId_ = 0U;
  oa::U64 restartRequest_ = 0U;
  oa::U64 parameterCount_ = 0U;
  oa::Bool startAccepted_ = false;
  oa::Bool shutdown_ = false;
};

[[nodiscard]] oa::Status submitAndWait(oa::Engine& inEngine) {
  auto event = inEngine.submit();
  if (event.isError()) return event.getStatus();
  OA_RETURN_IF_ERROR(inEngine.wait(*event));
  if (not inEngine.ownsEvent(*event) or not event->isComplete()) {
    return oa::Status::error(
        oa::StatusCode::Internal, "Training Lab exact completion failed");
  }
  return oa::Status::ok();
}

[[nodiscard]] oa::Result<oa::F32> evaluateAccuracy(
    oa::Engine& inEngine,
    oa::NlpSuiteModel& inModel,
    const oa::NlpSuiteRecipe& inRecipe) {
  oa::NlpSuiteSampler sampler(inRecipe, oa::NlpSuiteBatchSize);
  oa::Matrix input;
  oa::Matrix target;
  sampler.next(input, target);
  oa::Module::ScopedEval eval(inModel);
  const oa::Matrix logits = inModel.forward(input);
  OA_RETURN_IF_ERROR(submitAndWait(inEngine));

  const oa::Usize rows = static_cast<oa::Usize>(target.numElements());
  const oa::Usize vocab = static_cast<oa::Usize>(inRecipe.vocabSize());
  if (rows == 0U or logits.numElements() != static_cast<oa::I64>(rows * vocab)) {
    return oa::Status::error(
        oa::StatusCode::ShapeMismatch, "Training Lab evaluation logits shape is invalid");
  }
  oa::Vector<oa::F32> hostLogits(static_cast<oa::Usize>(logits.numElements()));
  oa::Vector<oa::U32> hostTargets(rows);
  OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
      logits, hostLogits.data(), hostLogits.size() * sizeof(oa::F32)));
  OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
      target, hostTargets.data(), hostTargets.size() * sizeof(oa::U32)));

  oa::Usize correct = 0U;
  for (oa::Usize row = 0U; row < rows; ++row) {
    oa::Usize best = 0U;
    for (oa::Usize column = 1U; column < vocab; ++column) {
      if (hostLogits[row * vocab + column] > hostLogits[row * vocab + best]) {
        best = column;
      }
    }
    if (best == static_cast<oa::Usize>(hostTargets[row])) ++correct;
  }
  return static_cast<oa::F32>(correct) / static_cast<oa::F32>(rows);
}

[[nodiscard]] oa::Result<oa::String> generateGreedy(
    oa::Engine& inEngine,
    oa::NlpSuiteModel& inModel,
    const oa::NlpSuiteRecipe& inRecipe) {
  oa::NlpSuiteSampler sampler(inRecipe, 1);
  const oa::Usize contextLength = static_cast<oa::Usize>(inRecipe.contextLength());
  const oa::Usize vocab = static_cast<oa::Usize>(inRecipe.vocabSize());
  oa::Vector<oa::I32> context(contextLength, 0);
  const oa::Vector<oa::I32> prompt = sampler.encode(oa::NlpSuiteGenerationPrompt);
  const oa::Usize copyCount = prompt.size() < contextLength ? prompt.size() : contextLength;
  for (oa::Usize index = 0U; index < copyCount; ++index) context[index] = prompt[index];
  oa::Usize filled = copyCount > 0U ? copyCount : 1U;
  oa::Usize logitRow = filled - 1U;
  oa::Vector<oa::I32> generatedTokens;
  generatedTokens.reserve(static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits));
  oa::Module::ScopedEval eval(inModel);
  for (oa::I32 index = 0; index < oa::NlpSuiteGenerationSourceUnits; ++index) {
    const oa::Matrix logits = inModel.forward(sampler.inputMatrix(context));
    OA_RETURN_IF_ERROR(submitAndWait(inEngine));
    oa::Vector<oa::F32> host(static_cast<oa::Usize>(logits.numElements()));
    OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
        logits, host.data(), host.size() * sizeof(oa::F32)));
    if (host.size() != contextLength * vocab) {
      return oa::Status::error(
          oa::StatusCode::ShapeMismatch, "Training Lab generation logits shape is invalid");
    }
    oa::I32 next = 0;
    for (oa::Usize column = 1U; column < vocab; ++column) {
      if (host[logitRow * vocab + column] >
          host[logitRow * vocab + static_cast<oa::Usize>(next)]) {
        next = static_cast<oa::I32>(column);
      }
    }
    generatedTokens.pushBack(next);
    if (filled < contextLength) {
      context[filled++] = next;
      logitRow = filled - 1U;
    } else {
      for (oa::Usize token = 1U; token < contextLength; ++token) {
        context[token - 1U] = context[token];
      }
      context[contextLength - 1U] = next;
      logitRow = contextLength - 1U;
    }
  }
  return oa::String(oa::NlpSuiteGenerationPrompt) + sampler.decode(generatedTokens);
}

[[nodiscard]] oa::Bool generationQuality(oa::StringView inGenerated) {
  const oa::StringView prompt(oa::NlpSuiteGenerationPrompt);
  if (inGenerated.size() != prompt.size() +
      static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits)) {
    return false;
  }
  for (oa::Usize index = prompt.size(); index < inGenerated.size(); ++index) {
    const auto value = static_cast<unsigned char>(inGenerated[index]);
    if (value != ' ' and (value < 'a' or value > 'z')) return false;
  }
  const oa::StringView corpus(oa::NlpSuiteSampler::corpus());
  const oa::StringView continuation = inGenerated.subStr(prompt.size());
  oa::Usize bestPrefix = 0U;
  for (oa::Usize found = corpus.find(prompt); found != oa::StringView::Npos;
       found = corpus.find(prompt, found + 1U)) {
    const oa::Usize start = found + prompt.size();
    oa::Usize matched = 0U;
    while (matched < continuation.size() and start + matched < corpus.size() and
           continuation[matched] == corpus[start + matched]) {
      ++matched;
    }
    if (matched > bestPrefix) bestPrefix = matched;
  }
  constexpr oa::Usize ngram = 8U;
  oa::Usize supported = 0U;
  oa::Usize total = 0U;
  for (oa::Usize index = 0U; index + ngram <= inGenerated.size(); ++index) {
    ++total;
    if (corpus.find(inGenerated.subStr(index, ngram)) != oa::StringView::Npos) {
      ++supported;
    }
  }
  const oa::F32 coverage = total == 0U ? 0.0F
      : static_cast<oa::F32>(supported) / static_cast<oa::F32>(total);
  return bestPrefix >= 16U and coverage >= 0.90F;
}

[[nodiscard]] oa::Result<oa::U64> parameterHash(
    oa::NlpSuiteModel& inModel) {
  oa::U64 hash = 1469598103934665603ULL;
  for (const oa::Parameter* parameter : inModel.allParameterPtrs()) {
    if (parameter == nullptr) {
      return oa::Status::error(oa::StatusCode::Internal, "Training Lab parameter is null");
    }
    oa::Vector<oa::Byte> bytes(static_cast<oa::Usize>(parameter->data.byteSize()));
    // copyToHost flushes pending work and treats an empty recording as a
    // validated no-op. A raw Engine::submit here would reject the common
    // post-training and post-load case because there is no device work left
    // to submit.
    OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
        parameter->data, bytes.data(), bytes.size()));
    for (const oa::Byte byte : bytes) {
      hash ^= static_cast<oa::U64>(byte);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

[[nodiscard]] oa::Result<TrainingArtifactSnapshot> qualifyCheckpoint(
    oa::Engine& inEngine,
    oa::NlpSuiteModel& inModel,
    oa::AdamW& inOptimizer,
    const oa::NlpSuiteRecipe& inRecipe,
    const oa::Path& inCheckpointPath,
    oa::I32 inExpectedSteps,
    oa::Bool inRequireCanonicalQuality) {
  TrainingArtifactSnapshot snapshot;
  snapshot.optimizerStep = inOptimizer.getStep();
  if (snapshot.optimizerStep != static_cast<oa::U64>(inExpectedSteps)) {
    return oa::Status::error(
        oa::StatusCode::FailedPrecondition,
        "Training Lab qualification requires every configured training step");
  }
  auto liveHash = parameterHash(inModel);
  if (liveHash.isError()) return liveHash.getStatus();
  auto liveAccuracy = evaluateAccuracy(inEngine, inModel, inRecipe);
  if (liveAccuracy.isError()) return liveAccuracy.getStatus();
  auto liveGenerated = generateGreedy(inEngine, inModel, inRecipe);
  if (liveGenerated.isError()) return liveGenerated.getStatus();
  OA_RETURN_IF_ERROR(inModel.save(inEngine, inCheckpointPath.string(), inOptimizer));
  snapshot.checkpointReady = true;

  oa::NlpSuiteModel reloaded(inRecipe);
  auto reloadedParameters = reloaded.allParameterPtrs();
  oa::AdamW reloadedOptimizer(reloadedParameters, inRecipe.learningRate());
  OA_RETURN_IF_ERROR(reloaded.load(
      inEngine, inCheckpointPath.string(), reloadedOptimizer));
  auto reloadedHash = parameterHash(reloaded);
  if (reloadedHash.isError()) return reloadedHash.getStatus();
  if (*reloadedHash != *liveHash or reloadedOptimizer.getStep() != inOptimizer.getStep()) {
    return oa::Status::error(
        oa::StatusCode::DataLoss, "Training Lab checkpoint state changed on reload");
  }
  auto reloadedAccuracy = evaluateAccuracy(inEngine, reloaded, inRecipe);
  if (reloadedAccuracy.isError()) return reloadedAccuracy.getStatus();
  auto reloadedGenerated = generateGreedy(inEngine, reloaded, inRecipe);
  if (reloadedGenerated.isError()) return reloadedGenerated.getStatus();
  if (*reloadedAccuracy != *liveAccuracy or *reloadedGenerated != *liveGenerated) {
    return oa::Status::error(
        oa::StatusCode::DataLoss, "Training Lab checkpoint output changed on reload");
  }

  snapshot.evaluationReady = true;
  snapshot.checkpointRoundTrip = true;
  snapshot.accuracy = *reloadedAccuracy;
  snapshot.parameterHash = *reloadedHash;
  snapshot.generated = oa::move(*reloadedGenerated);
  snapshot.generationQualityPassed = generationQuality(snapshot.generated);
  if (inRequireCanonicalQuality and
      (snapshot.accuracy < 0.85F or not snapshot.generationQualityPassed)) {
    return oa::Status::error(
        oa::StatusCode::FailedPrecondition,
        "Training Lab canonical numerical quality gate failed");
  }
  snapshot.qualified = true;
  return snapshot;
}

[[nodiscard]] oa::String vulkanStatusJson(const oa::Engine& inEngine) {
  const oa::MemoryUsage memory = inEngine.getMemoryUsage();
  oa::String json = "{";
  const auto stringField = [&json](const char* inName, oa::StringView inValue) {
    if (json.size() > 1U) json += ',';
    appendJsonString(json, inName);
    json += ':';
    appendJsonString(json, inValue);
  };
  stringField("deviceName", inEngine.deviceName());
  stringField("deviceVendor", inEngine.deviceVendorName());
  stringField("deviceType", oa::deviceTypeName(inEngine.deviceType()));
  stringField("driverName", inEngine.driverName());
  stringField("driverVersion", inEngine.driverVersion());
  stringField("vulkanApiVersion", inEngine.vulkanApiVersion());
  json += oa::format(
      ",\"hasCompute\":{},\"hasGraphics\":{},\"deviceVramBytes\":{},"
      "\"memoryTotalBytes\":{},\"memoryUsedBytes\":{},\"memoryFreeBytes\":{}",
      inEngine.hasCompute(), inEngine.hasGraphics(), inEngine.deviceVramBytes(),
      memory.totalBytes, memory.usedBytes, memory.freeBytes);
  json += '}';
  return json;
}

[[nodiscard]] oa::McpTool vulkanStatusTool(oa::Engine& inEngine) {
  oa::McpTool tool;
  tool.name = "vulkan_status";
  tool.title = "Vulkan runtime status";
  tool.description =
      "Read the active OA engine's Vulkan device, driver, queue, and memory status.";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{},"additionalProperties":false})";
  tool.outputSchemaJson = R"({"type":"object","properties":{"deviceName":{"type":"string"},"deviceVendor":{"type":"string"},"deviceType":{"type":"string"},"driverName":{"type":"string"},"driverVersion":{"type":"string"},"vulkanApiVersion":{"type":"string"},"hasCompute":{"type":"boolean"},"hasGraphics":{"type":"boolean"},"deviceVramBytes":{"type":"integer"},"memoryTotalBytes":{"type":"integer"},"memoryUsedBytes":{"type":"integer"},"memoryFreeBytes":{"type":"integer"}},"required":["deviceName","deviceVendor","deviceType","driverName","driverVersion","vulkanApiVersion","hasCompute","hasGraphics","deviceVramBytes","memoryTotalBytes","memoryUsedBytes","memoryFreeBytes"],"additionalProperties":false})";
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call = [&inEngine](const oa::McpArguments& inArguments) -> oa::Result<oa::McpToolResult> {
    if (not inArguments.empty()) {
      return oa::Status::invalidArgument("vulkan_status accepts no arguments");
    }
    oa::McpToolResult result = oa::McpToolResult::success(
        oa::format("OA Vulkan engine ready on {}", inEngine.deviceName()));
    result.structuredContentJson = vulkanStatusJson(inEngine);
    return result;
  };
  return tool;
}

[[nodiscard]] oa::Result<oa::U64> expectedRevision(
    const oa::McpArguments& inArguments) {
  if (inArguments.size() > 1U or
      (inArguments.size() == 1U and
       not inArguments.contains("expectedRevision"))) {
    return oa::Status::invalidArgument("only expectedRevision is accepted");
  }
  return inArguments.contains("expectedRevision")
      ? inArguments.unsignedInteger("expectedRevision")
      : oa::Result<oa::U64>(0U);
}

[[nodiscard]] oa::Result<oa::McpToolResult> acceptedCommand(
    oa::Result<oa::U64> inSequence) {
  if (inSequence.isError()) return inSequence.getStatus();
  oa::McpToolResult result = oa::McpToolResult::success(
      oa::format("training command accepted with sequence {}", *inSequence));
  result.structuredContentJson = oa::format("{{\"sequence\":{}}}", *inSequence);
  return result;
}

[[nodiscard]] oa::McpTool trainingReadTool(
    oa::String inName,
    oa::String inTitle,
    oa::String inDescription,
    oa::String inOutputSchema,
    oa::Fn<oa::Result<oa::McpToolResult>(const oa::McpArguments&)> inCall) {
  oa::McpTool tool;
  tool.name = oa::move(inName);
  tool.title = oa::move(inTitle);
  tool.description = oa::move(inDescription);
  tool.inputSchemaJson =
      R"({"type":"object","properties":{},"additionalProperties":false})";
  tool.outputSchemaJson = oa::move(inOutputSchema);
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call = oa::move(inCall);
  return tool;
}

[[nodiscard]] oa::McpTool trainingCommandTool(
    oa::String inName,
    oa::String inTitle,
    oa::String inDescription,
    oa::Fn<oa::Result<oa::U64>(oa::U64)> inCommand) {
  oa::McpTool tool;
  tool.name = oa::move(inName);
  tool.title = oa::move(inTitle);
  tool.description = oa::move(inDescription);
  tool.inputSchemaJson =
      R"({"type":"object","properties":{"expectedRevision":{"type":"integer","minimum":0}},"additionalProperties":false})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"sequence":{"type":"integer","minimum":1}},"required":["sequence"],"additionalProperties":false})";
  tool.readOnly = false;
  tool.destructive = false;
  tool.idempotent = false;
  tool.openWorld = false;
  tool.call = [command = oa::move(inCommand)](
      const oa::McpArguments& inArguments) mutable
      -> oa::Result<oa::McpToolResult> {
    auto revision = expectedRevision(inArguments);
    if (revision.isError()) return revision.getStatus();
    return acceptedCommand(command(*revision));
  };
  return tool;
}

[[nodiscard]] oa::McpTool trainingStatusTool(TrainingLabController& inController) {
  return trainingReadTool(
      "training_status", "Training status",
      "Read the active run, bounded configuration, progress, and scalar timings.",
      R"({"type":"object","properties":{"runId":{"type":"integer","minimum":1},"revision":{"type":"integer","minimum":0},"state":{"type":"string"},"step":{"type":"integer","minimum":0},"epoch":{"type":"integer","minimum":0},"totalSteps":{"type":"integer","minimum":1},"contextLength":{"type":"integer","minimum":4},"modelWidth":{"type":"integer","minimum":8},"hiddenWidth":{"type":"integer","minimum":8},"batchSize":{"type":"integer","minimum":1},"parameterCount":{"type":"integer","minimum":0},"restartPending":{"type":"boolean"},"learningRate":{"type":["number","null"]},"loss":{"type":["number","null"]},"gpuMs":{"type":["number","null"]},"wallMs":{"type":["number","null"]}},"required":["runId","revision","state","step","epoch","totalSteps","contextLength","modelWidth","hiddenWidth","batchSize","parameterCount","restartPending","learningRate","loss","gpuMs","wallMs"],"additionalProperties":false})",
      [&inController](const oa::McpArguments& inArguments)
          -> oa::Result<oa::McpToolResult> {
        if (not inArguments.empty()) {
          return oa::Status::invalidArgument("training_status accepts no arguments");
        }
        const oa::String json = inController.statusJson();
        oa::McpToolResult result = oa::McpToolResult::success(json);
        result.structuredContentJson = json;
        return result;
      });
}

[[nodiscard]] oa::McpTool trainingMetricsTool(TrainingLabController& inController) {
  return trainingReadTool(
      "training_metrics", "Training metrics",
      "Read metric samples from the active run's latest immutable snapshot.",
      R"({"type":"object","properties":{"runId":{"type":"integer","minimum":1},"metrics":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"value":{"type":["number","null"]},"step":{"type":"integer","minimum":0}},"required":["name","value","step"],"additionalProperties":false}}},"required":["runId","metrics"],"additionalProperties":false})",
      [&inController](const oa::McpArguments& inArguments)
          -> oa::Result<oa::McpToolResult> {
        if (not inArguments.empty()) {
          return oa::Status::invalidArgument("training_metrics accepts no arguments");
        }
        const oa::String json = inController.metricsJson();
        oa::McpToolResult result = oa::McpToolResult::success(json);
        result.structuredContentJson = json;
        return result;
      });
}

[[nodiscard]] oa::McpTool trainingResultsTool(TrainingLabController& inController) {
  oa::McpTool tool = trainingReadTool(
      "training_results", "Training command results",
      "Read the bounded command audit stream for the active run after a caller-owned cursor.",
      R"({"type":"object","properties":{"runId":{"type":"integer","minimum":1},"results":{"type":"array","items":{"type":"object","properties":{"sequence":{"type":"integer","minimum":1},"revision":{"type":"integer","minimum":0},"disposition":{"type":"string"},"state":{"type":"string"},"status":{"type":"string"}},"required":["sequence","revision","disposition","state","status"],"additionalProperties":false}}},"required":["runId","results"],"additionalProperties":false})",
      [&inController](const oa::McpArguments& inArguments)
          -> oa::Result<oa::McpToolResult> {
        if (inArguments.size() > 1U or
            (inArguments.size() == 1U and
             not inArguments.contains("afterSequence"))) {
          return oa::Status::invalidArgument(
              "training_results accepts only afterSequence");
        }
        oa::U64 after = 0U;
        if (inArguments.contains("afterSequence")) {
          auto value = inArguments.unsignedInteger("afterSequence");
          if (value.isError()) return value.getStatus();
          after = *value;
        }
        auto json = inController.resultsJson(after);
        if (json.isError()) return json.getStatus();
        oa::McpToolResult result = oa::McpToolResult::success(*json);
        result.structuredContentJson = oa::move(*json);
        return result;
      });
  tool.inputSchemaJson =
      R"({"type":"object","properties":{"afterSequence":{"type":"integer","minimum":0}},"additionalProperties":false})";
  return tool;
}

[[nodiscard]] oa::McpTool trainingStartTool(TrainingLabController& inController) {
  oa::McpTool tool;
  tool.name = "training_start";
  tool.title = "Start the Training Lab workload";
  tool.description =
      "Start the configured OA Transformer and byte-tokenizer training run. "
      "The command is accepted into the bounded TrainingSession queue "
      "and applied by the engine-owning thread at a safe point.";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{},"additionalProperties":false})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"sequence":{"type":"integer","minimum":1}},"required":["sequence"],"additionalProperties":false})";
  tool.readOnly = false;
  tool.destructive = false;
  tool.idempotent = false;
  tool.openWorld = false;
  tool.call = [&inController](const oa::McpArguments& inArguments)
      -> oa::Result<oa::McpToolResult> {
    if (not inArguments.empty()) {
      return oa::Status::invalidArgument("training_start accepts no arguments");
    }
    return acceptedCommand(inController.start());
  };
  return tool;
}

[[nodiscard]] oa::McpTool trainingSetParameterTool(
    TrainingLabController& inController) {
  oa::McpTool tool;
  tool.name = "training_set_parameter";
  tool.title = "Set one hot training parameter";
  tool.description =
      "Set the allowlisted learning_rate parameter at a training safe point.";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{"name":{"type":"string","const":"learning_rate"},"value":{"type":"number","minimum":0.000001,"maximum":1},"expectedRevision":{"type":"integer","minimum":0}},"required":["name","value"],"additionalProperties":false})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"sequence":{"type":"integer","minimum":1}},"required":["sequence"],"additionalProperties":false})";
  tool.readOnly = false;
  tool.destructive = false;
  tool.idempotent = false;
  tool.openWorld = false;
  tool.call = [&inController](const oa::McpArguments& inArguments)
      -> oa::Result<oa::McpToolResult> {
    if ((inArguments.size() != 2U and inArguments.size() != 3U) or
        not inArguments.contains("name") or
        not inArguments.contains("value") or
        (inArguments.size() == 3U and
         not inArguments.contains("expectedRevision"))) {
      return oa::Status::invalidArgument(
          "training_set_parameter accepts only name, value, and expectedRevision");
    }
    auto name = inArguments.string("name");
    if (name.isError()) return name.getStatus();
    if (*name != "learning_rate") {
      return oa::Status::invalidArgument("only learning_rate is allowlisted");
    }
    auto value = inArguments.number("value");
    if (value.isError()) return value.getStatus();
    oa::U64 revision = 0U;
    if (inArguments.contains("expectedRevision")) {
      auto parsed = inArguments.unsignedInteger("expectedRevision");
      if (parsed.isError()) return parsed.getStatus();
      revision = *parsed;
    }
    return acceptedCommand(inController.setLearningRate(*value, revision));
  };
  return tool;
}

[[nodiscard]] oa::McpTool trainingRestartTool(
    TrainingLabController& inController) {
  oa::McpTool tool;
  tool.name = "training_restart";
  tool.title = "Restart with a bounded model configuration";
  tool.description =
      "Cooperatively stop the active run and construct a fresh deterministic "
      "Transformer run with bounded steps, dimensions, batch, context, and "
      "initial learning rate. No path, code, shader, or arbitrary model input is accepted.";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{"totalSteps":{"type":"integer","minimum":1,"maximum":2000},"contextLength":{"type":"integer","minimum":4,"maximum":64,"multipleOf":4},"modelWidth":{"type":"integer","minimum":8,"maximum":256,"multipleOf":8},"hiddenWidth":{"type":"integer","minimum":8,"maximum":1024,"multipleOf":8},"batchSize":{"type":"integer","minimum":1,"maximum":256},"learningRate":{"type":"number","minimum":0.000001,"maximum":1}},"required":["totalSteps","contextLength","modelWidth","hiddenWidth","batchSize","learningRate"],"additionalProperties":false})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"requestId":{"type":"integer","minimum":1}},"required":["requestId"],"additionalProperties":false})";
  tool.readOnly = false;
  tool.destructive = true;
  tool.idempotent = false;
  tool.openWorld = false;
  tool.call = [&inController](const oa::McpArguments& inArguments)
      -> oa::Result<oa::McpToolResult> {
    if (inArguments.size() != 6U) {
      return oa::Status::invalidArgument(
          "training_restart requires exactly six bounded configuration fields");
    }
    auto steps = inArguments.integer("totalSteps");
    auto context = inArguments.integer("contextLength");
    auto model = inArguments.integer("modelWidth");
    auto hidden = inArguments.integer("hiddenWidth");
    auto batch = inArguments.integer("batchSize");
    auto rate = inArguments.number("learningRate");
    if (steps.isError()) return steps.getStatus();
    if (context.isError()) return context.getStatus();
    if (model.isError()) return model.getStatus();
    if (hidden.isError()) return hidden.getStatus();
    if (batch.isError()) return batch.getStatus();
    if (rate.isError()) return rate.getStatus();
    if (*steps < 1 or *steps > 2000 or
        *context < 4 or *context > 64 or
        *model < 8 or *model > 256 or
        *hidden < 8 or *hidden > 1024 or
        *batch < 1 or *batch > 256) {
      return oa::Status::invalidArgument(
          "training_restart contains an out-of-range integer");
    }
    const TrainingLabConfig config{
        .totalSteps = static_cast<oa::I32>(*steps),
        .contextLength = static_cast<oa::I32>(*context),
        .modelWidth = static_cast<oa::I32>(*model),
        .hiddenWidth = static_cast<oa::I32>(*hidden),
        .batchSize = static_cast<oa::I32>(*batch),
        .learningRate = static_cast<oa::F32>(*rate),
    };
    auto request = inController.restart(config);
    if (request.isError()) return request.getStatus();
    oa::McpToolResult result = oa::McpToolResult::success(
        oa::format("training restart accepted with request {}", *request));
    result.structuredContentJson = oa::format("{{\"requestId\":{}}}", *request);
    return result;
  };
  return tool;
}

[[nodiscard]] oa::McpTool trainingGenerateTool(const TrainingArtifacts& inArtifacts) {
  oa::McpTool tool;
  tool.name = "training_generate";
  tool.title = "Read qualified checkpoint generation";
  tool.description =
      "Return the bounded greedy generation and numerical evidence produced from "
      "the reloaded application-owned checkpoint. This read-only tool never runs "
      "model or Vulkan work on the gateway thread.";
  tool.inputSchemaJson =
      R"({"type":"object","properties":{},"additionalProperties":false})";
  tool.outputSchemaJson =
      R"({"type":"object","properties":{"checkpointRoundTrip":{"type":"boolean"},"generationQualityPassed":{"type":"boolean"},"accuracy":{"type":"number","minimum":0,"maximum":1},"optimizerStep":{"type":"integer","minimum":0},"parameterHash":{"type":"string","pattern":"^[0-9a-f]{16}$"},"prompt":{"type":"string"},"generated":{"type":"string"}},"required":["checkpointRoundTrip","generationQualityPassed","accuracy","optimizerStep","parameterHash","prompt","generated"],"additionalProperties":false})";
  tool.readOnly = true;
  tool.destructive = false;
  tool.idempotent = true;
  tool.openWorld = false;
  tool.call = [&inArtifacts](const oa::McpArguments& inArguments)
      -> oa::Result<oa::McpToolResult> {
    if (not inArguments.empty()) {
      return oa::Status::invalidArgument("training_generate accepts no arguments");
    }
    const TrainingArtifactSnapshot snapshot = inArtifacts.snapshot();
    if (not snapshot.qualified) {
      return oa::Status::error(
          oa::StatusCode::FailedPrecondition,
          snapshot.failure.empty()
              ? "qualified checkpoint generation is not ready"
              : snapshot.failure);
    }
    oa::String json = oa::format(
        "{{\"checkpointRoundTrip\":{},\"generationQualityPassed\":{},"
        "\"accuracy\":{},\"optimizerStep\":{},\"parameterHash\":\"{:016x}\",\"prompt\":",
        snapshot.checkpointRoundTrip, snapshot.generationQualityPassed,
        snapshot.accuracy, snapshot.optimizerStep, snapshot.parameterHash);
    appendJsonString(json, oa::NlpSuiteGenerationPrompt);
    json += ",\"generated\":";
    appendJsonString(json, snapshot.generated);
    json += '}';
    oa::McpToolResult result = oa::McpToolResult::success(
        oa::format(
            "qualified checkpoint generation ready at {:.3f} accuracy",
            snapshot.accuracy));
    result.structuredContentJson = oa::move(json);
    return result;
  };
  return tool;
}

[[nodiscard]] oa::Result<oa::String> readRequest(oa::TcpStream& inStream) {
  oa::String bytes;
  bytes.reserve(4096U);
  oa::Array<oa::Byte, 4096> chunk{};
  for (;;) {
    const oa::I64 count = inStream.read(chunk.data(), chunk.size());
    if (count <= 0) {
      return oa::Status::error(
          oa::StatusCode::ConnectionFailed, "HTTP peer closed before a complete request");
    }
    bytes.append(oa::StringView(
        reinterpret_cast<const char*>(chunk.data()), static_cast<oa::Usize>(count)));
    auto expected = oa::sdk::webmcp::expectedMessageBytes(bytes);
    if (expected.isError()) return expected.getStatus();
    if (*expected == 0U) continue;
    if (bytes.size() > *expected) {
      return oa::Status::invalidArgument("HTTP request contains trailing bytes");
    }
    if (bytes.size() == *expected) return bytes;
  }
}

[[nodiscard]] oa::Status serve(
    oa::TcpListener& inListener,
    oa::sdk::webmcp::Gateway& inGateway,
    TrainingLabController& inController,
    oa::U64 inMaxRequests) {
  oa::U64 served = 0U;
  oa::Status status = oa::Status::ok();
  while (inMaxRequests == 0U or served < inMaxRequests) {
    auto stream = inListener.accept();
    if (stream.isError()) {
      status = stream.getStatus();
      break;
    }
    (void)stream->setIoTimeout(3000U);
    auto bytes = readRequest(*stream);
    oa::sdk::webmcp::HttpResponse response;
    if (bytes.isError()) {
      response.status = bytes.getStatus().getCode() == oa::StatusCode::ResourceExhausted
          ? 413U
          : 400U;
      response.reason = response.status == 413U ? "Content Too Large" : "Bad Request";
      response.body = "invalid request\n";
    } else {
      auto request = oa::sdk::webmcp::parseRequest(*bytes);
      response = request.isError()
          ? oa::sdk::webmcp::HttpResponse{400U, "Bad Request", "text/plain; charset=utf-8", "invalid request\n"}
          : inGateway.handle(*request);
    }
    const oa::String wire = oa::sdk::webmcp::serializeResponse(response);
    (void)stream->writeAll(
        reinterpret_cast<const oa::Byte*>(wire.data()), wire.size());
    stream->close();
    ++served;
  }

  inController.shutdown();
  return status;
}

[[nodiscard]] oa::Result<oa::Optional<TrainingLabConfig>> runTrainingOnce(
    oa::Engine& inEngine,
    const TrainingLabConfig& inConfig,
    const oa::Path& inCheckpointPath,
    TrainingArtifacts& inArtifacts,
    TrainingLabController& inController,
    const oa::Fn<oa::Status()>& inOnPublished) {
  OA_RETURN_IF_ERROR(validateTrainingLabConfig(inConfig));
  const oa::NlpSuiteRecipe recipe(
      oa::NlpArchitecture::Transformer,
      oa::NlpTokenizerKind::Byte,
      inConfig.contextLength,
      inConfig.modelWidth,
      inConfig.hiddenWidth);
  oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
  oa::NlpSuiteModel model(recipe);
  auto parameters = model.allParameterPtrs();
  oa::AdamW optimizer(parameters, inConfig.learningRate);
  oa::ItTrainingConfig trainingConfig;
  trainingConfig.totalSteps = inConfig.totalSteps;
  trainingConfig.batchSize = inConfig.batchSize;
  trainingConfig.sequenceLength = inConfig.contextLength;
  trainingConfig.sequenceUnit = "token";
  trainingConfig.sourceUnitsPerSample = inConfig.contextLength;
  trainingConfig.sourceUnit = "byte";
  trainingConfig.timerName = recipe.timerName();
  oa::ItTraining training(inEngine, optimizer, oa::move(trainingConfig));
  oa::TrainingSessionConfig sessionConfig;
  sessionConfig.handlers.checkpoint = [&] {
    const oa::Status status = model.save(
        inEngine, inCheckpointPath.string(), optimizer);
    if (status.isOk()) {
      inArtifacts.recordCheckpoint(optimizer.getStep());
    } else {
      inArtifacts.recordFailure(status);
    }
    return status;
  };
  sessionConfig.handlers.evaluate = [&] {
    auto accuracy = evaluateAccuracy(inEngine, model, recipe);
    if (accuracy.isError()) {
      inArtifacts.recordFailure(accuracy.getStatus());
      return accuracy.getStatus();
    }
    inArtifacts.recordEvaluation(*accuracy);
    return oa::Status::ok();
  };
  oa::TrainingSession trainingSession(training, oa::move(sessionConfig));
  auto initialPause = trainingSession.pause(trainingSession.revision());
  if (initialPause.isError() or trainingSession.poll().isError()) {
    return oa::Status::error(
        oa::StatusCode::Internal, "Training Lab initial pause failed");
  }
  inController.publishRun(
      trainingSession, inConfig, static_cast<oa::U64>(model.numParameters()));
  if (const oa::Status status = inOnPublished(); status.isError()) {
    inController.unpublishRun(trainingSession);
    return status;
  }

  oa::NlpSuiteSampler sampler(recipe, inConfig.batchSize);
  oa::Matrix input;
  oa::Matrix target;
  while (trainingSession.waitBeginStep()) {
    sampler.next(input, target);
    optimizer.zeroGrad();
    oa::GradientTape tape;
    auto logits = model.forward(input);
    auto loss = oa::FnLoss::crossEntropy(
        logits, target.reshape({target.numElements()}));
    tape.backward(loss);
    training.recordSourceUnits(sampler.lastSourceUnits());
    trainingSession.publishMetric("batch_size", inConfig.batchSize);
    trainingSession.publishMetric("context_length", inConfig.contextLength);
    trainingSession.publishMetric(
        "parameter_count", static_cast<oa::F64>(model.numParameters()));
    training.next(loss);
  }
  const oa::Status trainingStatus = training.finish();
  if (trainingStatus.isError()) {
    inArtifacts.recordFailure(trainingStatus);
    inController.unpublishRun(trainingSession);
    return trainingStatus;
  }

  const oa::TrainingSnapshot finalSnapshot = trainingSession.currentSnapshot();
  const oa::Bool completedEveryStep =
      finalSnapshot.step == inConfig.totalSteps and
      not inController.restartPending() and
      not inController.isShuttingDown();
  if (completedEveryStep) {
    const oa::Bool canonicalQuality =
        inConfig.totalSteps == oa::NlpSuiteTrainingSteps and
        inConfig.contextLength == oa::NlpSuiteContextLength and
        inConfig.modelWidth == oa::NlpSuiteModelWidth and
        inConfig.hiddenWidth == oa::NlpSuiteHiddenWidth and
        inConfig.batchSize == oa::NlpSuiteBatchSize;
    auto qualification = qualifyCheckpoint(
        inEngine, model, optimizer, recipe, inCheckpointPath,
        inConfig.totalSteps, canonicalQuality);
    if (qualification.isError()) {
      if (not inController.restartPending()) {
        inArtifacts.recordFailure(qualification.getStatus());
      }
      oa::print(
          oa::PrintStream::Error,
          "Training Lab qualification failed: {}",
          qualification.getStatus().toString());
    } else if (not inController.restartPending()) {
      inArtifacts.recordQualification(oa::move(*qualification));
      const TrainingArtifactSnapshot snapshot = inArtifacts.snapshot();
      oa::print(
          "Run complete: step={} accuracy={:.3f} roundTrip={} quality={}",
          snapshot.optimizerStep, snapshot.accuracy,
          snapshot.checkpointRoundTrip, snapshot.generationQualityPassed);
    }
  }

  auto nextConfig = inController.waitNextConfig();
  inController.unpublishRun(trainingSession);
  return nextConfig;
}

[[nodiscard]] int runTrainingLab(oa::Engine& inEngine, const Options& inOptions) {
  const oa::Path stateDirectory = oa::Paths::var("webmcp");
  if (const oa::Status status = oa::Filesystem::createDirectories(stateDirectory);
      status.isError()) {
    oa::print(
        oa::PrintStream::Error,
        "Training Lab state directory failed: {}", status.toString());
    return 1;
  }
  const oa::Path checkpointPath = stateDirectory / "transformer_byte.oam";
  TrainingArtifacts artifacts;
  TrainingLabController controller(artifacts);

  oa::McpServerConfig mcpConfig;
  mcpConfig.name = "oa-network-webmcp";
  mcpConfig.version = OA_VERSION_STRING;
  mcpConfig.instructions =
      "Inspect and control the native OA Vulkan Training Lab. "
      "Mutations are limited to safe-point start, pause, resume, checkpoint, "
      "evaluation, the allowlisted learning_rate update, and bounded fresh-run "
      "configuration. Restart never accepts paths, code, shaders, or arbitrary models.";
  mcpConfig.maxMessageBytes = oa::sdk::webmcp::kMaxBodyBytes;
  oa::McpServer mcp(oa::move(mcpConfig));
  oa::Vector<oa::McpTool> tools;
  tools.pushBack(vulkanStatusTool(inEngine));
  tools.pushBack(trainingStatusTool(controller));
  tools.pushBack(trainingMetricsTool(controller));
  tools.pushBack(trainingResultsTool(controller));
  tools.pushBack(trainingStartTool(controller));
  tools.pushBack(trainingCommandTool(
      "training_pause", "Pause training",
      "Pause the active run at its next safe point.",
      [&controller](oa::U64 inRevision) { return controller.pause(inRevision); }));
  tools.pushBack(trainingCommandTool(
      "training_resume", "Resume training",
      "Resume the paused active run at its next safe point.",
      [&controller](oa::U64 inRevision) { return controller.resume(inRevision); }));
  tools.pushBack(trainingCommandTool(
      "training_checkpoint", "Checkpoint training",
      "Write the active run to the one application-owned checkpoint slot at a safe point.",
      [&controller](oa::U64 inRevision) { return controller.checkpoint(inRevision); }));
  tools.pushBack(trainingCommandTool(
      "training_evaluate", "Evaluate training",
      "Evaluate the active run at a safe point without exposing GPU work to the gateway thread.",
      [&controller](oa::U64 inRevision) { return controller.evaluate(inRevision); }));
  tools.pushBack(trainingSetParameterTool(controller));
  tools.pushBack(trainingRestartTool(controller));
  tools.pushBack(trainingGenerateTool(artifacts));
  for (auto& tool : tools) {
    if (const oa::Status status = mcp.addTool(oa::move(tool)); status.isError()) {
      oa::print(
          oa::PrintStream::Error,
          "MCP tool registration failed: {}", status.toString());
      return 1;
    }
  }

  auto listener = oa::TcpListener::bind("127.0.0.1", inOptions.port, 16);
  if (listener.isError()) {
    oa::print(oa::PrintStream::Error, "Loopback bind failed: {}", listener.getStatus().toString());
    return 1;
  }
  auto token = makeToken();
  if (token.isError()) {
    oa::print(oa::PrintStream::Error, "Credential generation failed: {}", token.getStatus().toString());
    return 1;
  }

  const oa::String loopbackHost = oa::format("127.0.0.1:{}", listener->port());
  const oa::String expectedHost = inOptions.externalHost.empty()
      ? loopbackHost
      : inOptions.externalHost;
  const oa::String expectedOrigin = inOptions.externalOrigin.empty()
      ? "http://" + loopbackHost
      : inOptions.externalOrigin;
  oa::sdk::webmcp::Assets assets{
      .indexHtml = oa::sdk::webmcp::kIndexHtml,
      .scriptJavaScript = oa::sdk::webmcp::kScriptJavaScript,
      .styleCss = oa::sdk::webmcp::kStyleCss,
  };
  oa::sdk::webmcp::Gateway gateway(
      mcp, oa::move(assets), expectedHost, expectedOrigin, *token);

  oa::print("oa::Network - WebMCP for native Vulkan applications");
  oa::print("Device: {} ({})", inEngine.deviceName(), inEngine.vulkanApiVersion());
  oa::print("Open: {}/#token={}", expectedOrigin, *token);
  if (not inOptions.externalOrigin.empty()) {
    oa::print(
        "Listener: http://{} (loopback only; external HTTPS proxy must preserve Host)",
        loopbackHost);
  }
  oa::print("The credential stays in the URL fragment and is removed by the page after load.");

  oa::Status gatewayStatus = oa::Status::ok();
  oa::Optional<oa::Thread> gatewayThread;
  oa::Bool gatewayStarted = false;
  const oa::Fn<oa::Status()> startGateway = [&]() -> oa::Status {
    if (gatewayStarted) return oa::Status::ok();
    auto created = oa::Thread::create([&] {
      gatewayStatus = serve(*listener, gateway, controller, inOptions.maxRequests);
    });
    if (created.isError()) return created.getStatus();
    gatewayThread = oa::move(*created);
    gatewayStarted = true;
    return oa::Status::ok();
  };

  TrainingLabConfig config;
  oa::Status runStatus = oa::Status::ok();
  for (;;) {
    auto next = runTrainingOnce(
        inEngine, config, checkpointPath, artifacts, controller, startGateway);
    if (next.isError()) {
      runStatus = next.getStatus();
      controller.shutdown();
      break;
    }
    if (not next->hasValue()) break;
    config = **next;
  }

  listener->close();
  const oa::Status joinStatus = gatewayThread.hasValue()
      ? gatewayThread->join()
      : oa::Status::ok();

  token->secureWipeSecrets();
  (void)mcp.close();
  if (runStatus.isError()) {
    oa::print(oa::PrintStream::Error, "Training Lab failed: {}", runStatus.toString());
    return 1;
  }
  if (joinStatus.isError()) {
    oa::print(oa::PrintStream::Error, "Gateway join failed: {}", joinStatus.toString());
    return 1;
  }
  if (gatewayStatus.isError()) {
    oa::print(oa::PrintStream::Error, "Gateway failed: {}", gatewayStatus.toString());
    return 1;
  }
  return 0;
}

[[nodiscard]] int run(const Options& inOptions) {
  oa::EngineConfig engineConfig;
  engineConfig.appName = "oa::Network WebMCP";
  engineConfig.presentationMode = oa::PresentationMode::None;
  auto engine = oa::Engine::create(engineConfig);
  if (engine.isError()) {
    oa::print(oa::PrintStream::Error, "OA engine creation failed: {}", engine.getStatus().toString());
    return 1;
  }

  const int result = runTrainingLab(**engine, inOptions);
  const oa::Status closeStatus = (*engine)->close();
  if (closeStatus.isError()) {
    oa::print(oa::PrintStream::Error, "OA engine shutdown failed: {}", closeStatus.toString());
    return 1;
  }
  return result;
}

} // namespace

int main(int inArgc, char** inArgv) {
  auto options = parseOptions(inArgc, inArgv);
  if (options.isError()) {
    oa::print(oa::PrintStream::Error, "{}", options.getStatus().toString());
    return 2;
  }
  return run(*options);
}
