// oa::TrainingSession - typed live control and observation for oa::ItTraining.
//
// The session is deliberately independent of UI, Python, networking and MCP.
// Those surfaces enqueue the same commands and consume immutable snapshots.
// commands are applied only when the training owner calls tryBeginStep() or
// waitBeginStep(), before a new forward/backward graph is recorded.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

class ItTraining;

enum class TrainingState : oa::U8 {
  Running,
  Paused,
  Stopping,
  Completed,
  Failed,
};

enum class TrainingCommandKind : oa::U8 {
  Pause,
  Resume,
  Stop,
  Checkpoint,
  Evaluate,
  SetParameter,
  RequestRecapture,
  RequestRebuild,
};

enum class TrainingCommandDisposition : oa::U8 {
  Applied,
  Rejected,
};

enum class TrainingParameterClass : oa::U8 {
  Hot,
  Recapture,
  Rebuild,
  Immutable,
};

enum class TrainingValueKind : oa::U8 {
  Empty,
  Boolean,
  Integer,
  Float,
  String,
};

struct TrainingValue {
  TrainingValueKind kind = TrainingValueKind::Empty;
  oa::Bool booleanValue = false;
  oa::I64 integer = 0;
  oa::F64 floatValue = 0.0;
  oa::String string;

  [[nodiscard]] static TrainingValue fromBool(oa::Bool inValue);
  [[nodiscard]] static TrainingValue fromInteger(oa::I64 inValue);
  [[nodiscard]] static TrainingValue fromFloat(oa::F64 inValue);
  [[nodiscard]] static TrainingValue fromString(oa::String inValue);
  [[nodiscard]] oa::Optional<oa::F64> asNumber() const;
};

struct TrainingCommand {
  oa::U64 sequence = 0;
  // Zero accepts the current revision. Non-zero provides optimistic
  // concurrency control for remote or asynchronous clients.
  oa::U64 expectedRevision = 0;
  TrainingCommandKind kind = TrainingCommandKind::Pause;
  oa::String parameter;
  TrainingValue value;
};

struct TrainingCommandResult {
  oa::U64 sequence = 0;
  oa::U64 revision = 0;
  TrainingCommandDisposition disposition =
      TrainingCommandDisposition::Rejected;
  TrainingState state = TrainingState::Running;
  oa::Status status = oa::Status::ok();
};

struct TrainingMetricSample {
  oa::String name;
  oa::F64 value = 0.0;
  oa::I64 step = 0;
};

struct TrainingSnapshot {
  oa::U64 revision = 0;
  TrainingState state = TrainingState::Running;
  oa::I64 step = 0;
  oa::I64 epoch = 0;
  oa::F32 learningRate = 0.0F;
  oa::F32 loss = 0.0F;
  oa::F64 gpuMs = 0.0;
  oa::F64 wallMs = 0.0;
  oa::Vector<TrainingMetricSample> metrics;
};

struct TrainingParameterDesc {
  oa::String name;
  TrainingParameterClass parameterClass = TrainingParameterClass::Hot;
  TrainingValueKind kind = TrainingValueKind::Float;
  oa::Optional<oa::F64> minimum;
  oa::Optional<oa::F64> maximum;
  oa::Fn<TrainingValue()> get;
  oa::Fn<oa::Status(const TrainingValue &)> set;
};

struct TrainingSessionHandlers {
  oa::Fn<oa::Status()> checkpoint;
  oa::Fn<oa::Status()> evaluate;
  oa::Fn<oa::Status(const TrainingCommand &)> rebuild;
};

struct TrainingSessionConfig {
  oa::U32 commandCapacity = 64;
  oa::U32 resultCapacity = 128;
  oa::U32 snapshotCapacity = 256;
  TrainingSessionHandlers handlers;
};

class TrainingSession {
public:
  TrainingSession(oa::ItTraining &inTraining,
                  TrainingSessionConfig inConfig = {});
  ~TrainingSession();

  TrainingSession(const TrainingSession &) = delete;
  TrainingSession &operator=(const TrainingSession &) = delete;
  TrainingSession(TrainingSession &&) = delete;
  TrainingSession &operator=(TrainingSession &&) = delete;

  // Enqueue is non-blocking. A full queue rejects the command explicitly.
  [[nodiscard]] oa::Result<oa::U64> enqueue(TrainingCommand inCommand);
  [[nodiscard]] oa::Result<oa::U64> pause(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> resume(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> stop(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> checkpoint(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> evaluate(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> setParameter(oa::String inName,
                                             TrainingValue inValue,
                                             oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> requestRecapture(oa::U64 inExpectedRevision = 0);
  [[nodiscard]] oa::Result<oa::U64> requestRebuild(TrainingValue inConfig,
                                               oa::U64 inExpectedRevision = 0);

  // Non-blocking safe point for UI/event-loop owned training. Returns true only
  // when one ordinary training step may begin now.
  [[nodiscard]] bool tryBeginStep();
  // Blocking safe point for a dedicated training thread. Enqueued resume/stop
  // commands wake the wait without polling.
  [[nodiscard]] bool waitBeginStep();
  // apply commands without advancing the training iterator.
  [[nodiscard]] oa::Status poll();

  [[nodiscard]] oa::Status registerParameter(TrainingParameterDesc inDesc);
  [[nodiscard]] oa::Optional<TrainingValue> parameter(
      oa::StringView inName) const;
  void publishMetric(oa::String inName, oa::F64 inValue);

  [[nodiscard]] TrainingState state() const;
  [[nodiscard]] oa::U64 revision() const;
  // Atomically combines the live state/revision with the most recently
  // published step metrics. Unlike latestSnapshot(), this reflects commands
  // applied since the last completed training step.
  [[nodiscard]] TrainingSnapshot currentSnapshot() const;
  [[nodiscard]] oa::Optional<TrainingSnapshot> latestSnapshot() const;
  // Non-destructive bounded result view for independent observers such as a
  // viewer and MCP client. results older than the configured ring capacity may
  // have been dropped; callers advance their own sequence cursor.
  [[nodiscard]] oa::Vector<TrainingCommandResult>
  resultsAfter(oa::U64 inSequence) const;
  [[nodiscard]] oa::Vector<TrainingCommandResult> takeResults();

  // Called by oa::ItTraining after a completed step/reset/finish. Public for
  // custom iterator adapters, but ordinary callers do not invoke these.
  void onStepCompleted(const oa::ItTraining &inTraining);
  void onReset(const oa::ItTraining &inTraining);
  void onFinished(const oa::Status &inStatus, const oa::ItTraining &inTraining);

private:
  struct Impl;
  oa::UniquePtr<Impl> impl_;
};

} // namespace oa
