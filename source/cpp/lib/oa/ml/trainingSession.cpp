#include <oa/ml/trainingSession.h>

#include <oa/ml/itTraining.h>
#include <oa/ml/optim.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

namespace oa {

TrainingValue TrainingValue::fromBool(oa::Bool inValue) {
  TrainingValue value;
  value.kind = TrainingValueKind::Boolean;
  value.booleanValue = inValue;
  return value;
}

TrainingValue TrainingValue::fromInteger(oa::I64 inValue) {
  TrainingValue value;
  value.kind = TrainingValueKind::Integer;
  value.integer = inValue;
  return value;
}

TrainingValue TrainingValue::fromFloat(oa::F64 inValue) {
  TrainingValue value;
  value.kind = TrainingValueKind::Float;
  value.floatValue = inValue;
  return value;
}

TrainingValue TrainingValue::fromString(oa::String inValue) {
  TrainingValue value;
  value.kind = TrainingValueKind::String;
  value.string = oa::move(inValue);
  return value;
}

oa::Optional<oa::F64> TrainingValue::asNumber() const {
  if (kind == TrainingValueKind::Float)
    return floatValue;
  if (kind == TrainingValueKind::Integer) {
    return static_cast<oa::F64>(integer);
  }
  return {};
}

struct TrainingSession::Impl {
  oa::ItTraining *training = nullptr;
  TrainingSessionConfig config;
  mutable std::mutex mutex;
  std::condition_variable wake;
  TrainingState state = TrainingState::Running;
  oa::U64 revision = 0;
  oa::U64 nextSequence = 1;
  oa::U64 takeSequence = 0;
  std::deque<TrainingCommand> commands;
  std::deque<TrainingCommandResult> results;
  std::deque<TrainingSnapshot> snapshots;
  std::vector<TrainingParameterDesc> parameters;
  std::vector<TrainingMetricSample> pendingMetrics;

  void boundResults() {
    while (results.size() > config.resultCapacity)
      results.pop_front();
  }

  void boundSnapshots() {
    while (snapshots.size() > config.snapshotCapacity)
      snapshots.pop_front();
  }
};

namespace {

bool isTerminal(TrainingState inState) {
  return inState == TrainingState::Stopping ||
         inState == TrainingState::Completed ||
         inState == TrainingState::Failed;
}

oa::Status missingHandler(const char *inName) {
  return oa::Status::error(oa::StatusCode::FailedPrecondition,
                         oa::String("oa::TrainingSession: no ") + inName +
                             " handler is registered");
}

} // namespace

TrainingSession::TrainingSession(oa::ItTraining &inTraining,
                                 TrainingSessionConfig inConfig)
    : impl_(oa::makeUnique<Impl>()) {
  impl_->training = &inTraining;
  impl_->config = oa::move(inConfig);
  impl_->config.commandCapacity =
      std::max<oa::U32>(impl_->config.commandCapacity, 1);
  impl_->config.resultCapacity =
      std::max<oa::U32>(impl_->config.resultCapacity, 1);
  impl_->config.snapshotCapacity =
      std::max<oa::U32>(impl_->config.snapshotCapacity, 1);
  inTraining.attachSession(this);

  TrainingParameterDesc learningRate;
  learningRate.name = "learning_rate";
  learningRate.parameterClass = TrainingParameterClass::Hot;
  learningRate.kind = TrainingValueKind::Float;
  learningRate.minimum = std::numeric_limits<oa::F64>::min();
  learningRate.get = [&inTraining]() {
    return TrainingValue::fromFloat(inTraining.optimizer().getLr());
  };
  learningRate.set = [&inTraining](const TrainingValue &inValue) {
    const auto number = inValue.asNumber();
    if (!number.hasValue() || !std::isfinite(*number) || *number <= 0.0 ||
        *number > static_cast<oa::F64>(std::numeric_limits<oa::F32>::max())) {
      return oa::Status::invalidArgument(
          "learning_rate expects one finite positive Float value");
    }
    inTraining.optimizer().setLr(static_cast<oa::F32>(*number));
    return oa::Status::ok();
  };
  (void)registerParameter(oa::move(learningRate));
  onReset(inTraining);
}

TrainingSession::~TrainingSession() {
  if (impl_ && impl_->training && impl_->training->session() == this) {
    impl_->training->attachSession(nullptr);
  }
}

oa::Result<oa::U64> TrainingSession::enqueue(TrainingCommand inCommand) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->commands.size() >= impl_->config.commandCapacity) {
    return oa::Result<oa::U64>(
        oa::Status::error(oa::StatusCode::ResourceExhausted,
                        "oa::TrainingSession command queue is full"));
  }
  // sequence numbers are session-owned so duplicate or out-of-order caller
  // values cannot break independent result cursors.
  inCommand.sequence = impl_->nextSequence++;
  const oa::U64 sequence = inCommand.sequence;
  impl_->commands.push_back(oa::move(inCommand));
  impl_->wake.notify_all();
  return oa::Result<oa::U64>(sequence);
}

oa::Result<oa::U64> TrainingSession::pause(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::Pause});
}

oa::Result<oa::U64> TrainingSession::resume(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::Resume});
}

oa::Result<oa::U64> TrainingSession::stop(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::Stop});
}

oa::Result<oa::U64> TrainingSession::checkpoint(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::Checkpoint});
}

oa::Result<oa::U64> TrainingSession::evaluate(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::Evaluate});
}

oa::Result<oa::U64> TrainingSession::setParameter(
    oa::String inName,
    TrainingValue inValue,
    oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::SetParameter,
                  .parameter = oa::move(inName),
                  .value = oa::move(inValue)});
}

oa::Result<oa::U64>
TrainingSession::requestRecapture(oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::RequestRecapture});
}

oa::Result<oa::U64> TrainingSession::requestRebuild(
    TrainingValue inConfig,
    oa::U64 inExpectedRevision) {
  return enqueue({.expectedRevision = inExpectedRevision,
                  .kind = TrainingCommandKind::RequestRebuild,
                  .value = oa::move(inConfig)});
}

oa::Status TrainingSession::registerParameter(TrainingParameterDesc inDesc) {
  if (inDesc.name.empty() || !inDesc.get) {
    return oa::Status::invalidArgument(
        "oa::TrainingSession parameter requires a name and getter");
  }
  if (inDesc.parameterClass != TrainingParameterClass::Immutable &&
      !inDesc.set) {
    return oa::Status::invalidArgument(
        "mutable oa::TrainingSession parameter requires a setter");
  }
  if (inDesc.minimum.hasValue() && inDesc.maximum.hasValue() &&
      *inDesc.minimum > *inDesc.maximum) {
    return oa::Status::invalidArgument(
        "oa::TrainingSession parameter minimum exceeds maximum");
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (const auto &parameter : impl_->parameters) {
    if (parameter.name == inDesc.name) {
      return oa::Status::error(oa::StatusCode::AlreadyExists,
                             "oa::TrainingSession parameter already exists: " +
                                 inDesc.name);
    }
  }
  impl_->parameters.push_back(oa::move(inDesc));
  return oa::Status::ok();
}

oa::Optional<TrainingValue>
TrainingSession::parameter(oa::StringView inName) const {
  oa::Fn<TrainingValue()> getter;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto &parameter : impl_->parameters) {
      if (parameter.name == inName) {
        getter = parameter.get;
        break;
      }
    }
  }
  if (!getter)
    return {};
  return getter();
}

void TrainingSession::publishMetric(oa::String inName, oa::F64 inValue) {
  if (inName.empty() || !std::isfinite(inValue))
    return;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto &metric : impl_->pendingMetrics) {
    if (metric.name == inName) {
      metric.value = inValue;
      return;
    }
  }
  impl_->pendingMetrics.push_back(
      {.name = oa::move(inName), .value = inValue});
}

oa::Status TrainingSession::poll() {
  std::deque<TrainingCommand> commands;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    commands.swap(impl_->commands);
  }

  for (const auto &command : commands) {
    oa::Status status = oa::Status::ok();
    TrainingParameterDesc parameter;
    bool hasParameter = false;
    TrainingState state;
    oa::U64 revision;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      state = impl_->state;
      revision = impl_->revision;
      if (command.expectedRevision != 0 &&
          command.expectedRevision != revision) {
        status = oa::Status::error(oa::StatusCode::Aborted,
                                 "oa::TrainingSession command revision is stale");
      }
      if (status.isOk() &&
          command.kind == TrainingCommandKind::SetParameter) {
        for (const auto &candidate : impl_->parameters) {
          if (candidate.name == command.parameter) {
            parameter = candidate;
            hasParameter = true;
            break;
          }
        }
      }
    }

    if (status.isOk()) {
      switch (command.kind) {
      case TrainingCommandKind::Pause:
        if (state != TrainingState::Running) {
          status = oa::Status::error(oa::StatusCode::FailedPrecondition,
                                   "pause requires a running training session");
        }
        break;
      case TrainingCommandKind::Resume:
        if (state != TrainingState::Paused) {
          status = oa::Status::error(oa::StatusCode::FailedPrecondition,
                                   "resume requires a paused training session");
        }
        break;
      case TrainingCommandKind::Stop:
        if (isTerminal(state)) {
          status = oa::Status::error(oa::StatusCode::FailedPrecondition,
                                   "stop requires an active training session");
        } else {
          impl_->training->requestStop();
        }
        break;
      case TrainingCommandKind::Checkpoint:
        status = impl_->config.handlers.checkpoint
                     ? impl_->config.handlers.checkpoint()
                     : missingHandler("checkpoint");
        break;
      case TrainingCommandKind::Evaluate:
        status = impl_->config.handlers.evaluate
                     ? impl_->config.handlers.evaluate()
                     : missingHandler("evaluation");
        break;
      case TrainingCommandKind::SetParameter: {
        if (!hasParameter) {
          status = oa::Status::notFound("unknown training parameter: " +
                                      command.parameter);
          break;
        }
        if (parameter.parameterClass == TrainingParameterClass::Immutable) {
          status = oa::Status::error(oa::StatusCode::FailedPrecondition,
                                   "training parameter is immutable: " +
                                       command.parameter);
          break;
        }
        if (parameter.parameterClass != TrainingParameterClass::Hot &&
            state != TrainingState::Paused) {
          status = oa::Status::error(
              oa::StatusCode::FailedPrecondition,
              "recapture/rebuild parameters require a paused session");
          break;
        }
        if (command.value.kind != parameter.kind &&
            !(parameter.kind == TrainingValueKind::Float &&
              command.value.kind == TrainingValueKind::Integer)) {
          status = oa::Status::invalidArgument(
              "training parameter value kind does not match its declaration");
          break;
        }
        const auto number = command.value.asNumber();
        if (number.hasValue() &&
            ((parameter.minimum.hasValue() && *number < *parameter.minimum) ||
             (parameter.maximum.hasValue() && *number > *parameter.maximum))) {
          status = oa::Status::error(
              oa::StatusCode::OutOfRange,
              "training parameter value is outside its declared range");
          break;
        }
        status = parameter.set(command.value);
        break;
      }
      case TrainingCommandKind::RequestRecapture:
        if (state != TrainingState::Paused) {
          status =
              oa::Status::error(oa::StatusCode::FailedPrecondition,
                              "program recapture requires a paused session");
        } else {
          status = impl_->training->requestProgramRecapture();
        }
        break;
      case TrainingCommandKind::RequestRebuild:
        if (state != TrainingState::Paused) {
          status =
              oa::Status::error(oa::StatusCode::FailedPrecondition,
                              "training rebuild requires a paused session");
        } else {
          status = impl_->config.handlers.rebuild
                       ? impl_->config.handlers.rebuild(command)
                       : missingHandler("rebuild");
        }
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      if (status.isOk()) {
        switch (command.kind) {
        case TrainingCommandKind::Pause:
          impl_->state = TrainingState::Paused;
          break;
        case TrainingCommandKind::Resume:
          impl_->state = TrainingState::Running;
          break;
        case TrainingCommandKind::Stop:
          impl_->state = TrainingState::Stopping;
          break;
        default:
          break;
        }
        ++impl_->revision;
      }
      impl_->results.push_back({
          .sequence = command.sequence,
          .revision = impl_->revision,
          .disposition = status.isOk() ? TrainingCommandDisposition::Applied
                                       : TrainingCommandDisposition::Rejected,
          .state = impl_->state,
          .status = status,
      });
      impl_->boundResults();
      impl_->wake.notify_all();
    }
  }
  return oa::Status::ok();
}

bool TrainingSession::tryBeginStep() {
  (void)poll();
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state != TrainingState::Running)
      return false;
  }
  return !impl_->training->isDone();
}

bool TrainingSession::waitBeginStep() {
  for (;;) {
    (void)poll();
    {
      std::unique_lock<std::mutex> lock(impl_->mutex);
      if (isTerminal(impl_->state))
        return false;
      if (impl_->state == TrainingState::Running)
        break;
      impl_->wake.wait(lock, [this] {
        return !impl_->commands.empty() ||
               impl_->state != TrainingState::Paused;
      });
    }
  }
  return !impl_->training->isDone();
}

TrainingState TrainingSession::state() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->state;
}

oa::U64 TrainingSession::revision() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->revision;
}

TrainingSnapshot TrainingSession::currentSnapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  TrainingSnapshot snapshot;
  if (not impl_->snapshots.empty())
    snapshot = impl_->snapshots.back();
  snapshot.state = impl_->state;
  snapshot.revision = impl_->revision;
  return snapshot;
}

oa::Optional<TrainingSnapshot> TrainingSession::latestSnapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->snapshots.empty())
    return {};
  return impl_->snapshots.back();
}

oa::Vec<TrainingCommandResult>
TrainingSession::resultsAfter(oa::U64 inSequence) const {
  oa::Vec<TrainingCommandResult> results;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (const auto &result : impl_->results) {
    if (result.sequence > inSequence)
      results.pushBack(result);
  }
  return results;
}

oa::Vec<TrainingCommandResult> TrainingSession::takeResults() {
  oa::Vec<TrainingCommandResult> results;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  results.reserve(impl_->results.size());
  for (const auto &result : impl_->results) {
    if (result.sequence <= impl_->takeSequence)
      continue;
    results.pushBack(result);
    impl_->takeSequence = std::max(impl_->takeSequence, result.sequence);
  }
  return results;
}

void TrainingSession::onStepCompleted(const oa::ItTraining &inTraining) {
  TrainingSnapshot snapshot;
  snapshot.step = inTraining.stepCount();
  snapshot.epoch = inTraining.epoch();
  snapshot.learningRate = inTraining.optimizer().getLr();
  snapshot.loss = inTraining.lastLoss();
  snapshot.gpuMs = inTraining.lastGpuMs();
  snapshot.wallMs = inTraining.wallMsPerStep();
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    snapshot.revision = impl_->revision;
    snapshot.state = impl_->state;
    snapshot.metrics.reserve(static_cast<oa::I64>(impl_->pendingMetrics.size()));
    for (auto metric : impl_->pendingMetrics) {
      metric.step = snapshot.step;
      snapshot.metrics.pushBack(oa::move(metric));
    }
    impl_->snapshots.push_back(oa::move(snapshot));
    impl_->boundSnapshots();
  }
}

void TrainingSession::onReset(const oa::ItTraining &inTraining) {
  TrainingSnapshot snapshot;
  snapshot.step = inTraining.stepCount();
  snapshot.epoch = inTraining.epoch();
  snapshot.learningRate = inTraining.optimizer().getLr();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->state = TrainingState::Running;
  ++impl_->revision;
  snapshot.revision = impl_->revision;
  snapshot.state = impl_->state;
  impl_->snapshots.push_back(oa::move(snapshot));
  impl_->boundSnapshots();
  impl_->wake.notify_all();
}

void TrainingSession::onFinished(const oa::Status &inStatus,
                                 const oa::ItTraining &inTraining) {
  TrainingSnapshot snapshot;
  snapshot.step = inTraining.stepCount();
  snapshot.epoch = inTraining.epoch();
  snapshot.learningRate = inTraining.optimizer().getLr();
  snapshot.loss = inTraining.lastLoss();
  snapshot.gpuMs = inTraining.lastGpuMs();
  snapshot.wallMs = inTraining.wallMsPerStep();
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->state =
      inStatus.isOk() ? TrainingState::Completed : TrainingState::Failed;
  ++impl_->revision;
  snapshot.revision = impl_->revision;
  snapshot.state = impl_->state;
  impl_->snapshots.push_back(oa::move(snapshot));
  impl_->boundSnapshots();
  impl_->wake.notify_all();
}

} // namespace oa
