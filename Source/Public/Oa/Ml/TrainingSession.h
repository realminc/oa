// OaTrainingSession - typed live control and observation for OaItTraining.
//
// The session is deliberately independent of UI, Python, networking and MCP.
// Those surfaces enqueue the same commands and consume immutable snapshots.
// Commands are applied only when the training owner calls TryBeginStep() or
// WaitBeginStep(), before a new forward/backward graph is recorded.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaItTraining;

enum class OaTrainingState : OaU8 {
	Running,
	Paused,
	Stopping,
	Completed,
	Failed,
};

enum class OaTrainingCommandKind : OaU8 {
	Pause,
	Resume,
	Stop,
	Checkpoint,
	Evaluate,
	SetParameter,
	RequestRecapture,
	RequestRebuild,
};

enum class OaTrainingCommandDisposition : OaU8 {
	Applied,
	Rejected,
};

enum class OaTrainingParameterClass : OaU8 {
	Hot,
	Recapture,
	Rebuild,
	Immutable,
};

enum class OaTrainingValueKind : OaU8 {
	Empty,
	Boolean,
	Integer,
	Float,
	String,
};

struct OaTrainingValue {
	OaTrainingValueKind Kind = OaTrainingValueKind::Empty;
	OaBool Bool = false;
	OaI64 Integer = 0;
	OaF64 Float = 0.0;
	OaString String;

	[[nodiscard]] static OaTrainingValue FromBool(OaBool InValue);
	[[nodiscard]] static OaTrainingValue FromInteger(OaI64 InValue);
	[[nodiscard]] static OaTrainingValue FromFloat(OaF64 InValue);
	[[nodiscard]] static OaTrainingValue FromString(OaString InValue);
	[[nodiscard]] OaOpt<OaF64> AsNumber() const;
};

struct OaTrainingCommand {
	OaU64 Sequence = 0;
	// Zero accepts the current revision. Non-zero provides optimistic
	// concurrency control for remote or asynchronous clients.
	OaU64 ExpectedRevision = 0;
	OaTrainingCommandKind Kind = OaTrainingCommandKind::Pause;
	OaString Parameter;
	OaTrainingValue Value;
};

struct OaTrainingCommandResult {
	OaU64 Sequence = 0;
	OaU64 Revision = 0;
	OaTrainingCommandDisposition Disposition =
		OaTrainingCommandDisposition::Rejected;
	OaTrainingState State = OaTrainingState::Running;
	OaStatus Status = OaStatus::Ok();
};

struct OaTrainingMetricSample {
	OaString Name;
	OaF64 Value = 0.0;
	OaI64 Step = 0;
};

struct OaTrainingSnapshot {
	OaU64 Revision = 0;
	OaTrainingState State = OaTrainingState::Running;
	OaI64 Step = 0;
	OaI64 Epoch = 0;
	OaF32 LearningRate = 0.0F;
	OaF32 Loss = 0.0F;
	OaF64 GpuMs = 0.0;
	OaF64 WallMs = 0.0;
	OaVec<OaTrainingMetricSample> Metrics;
};

struct OaTrainingParameterDesc {
	OaString Name;
	OaTrainingParameterClass Class = OaTrainingParameterClass::Hot;
	OaTrainingValueKind Kind = OaTrainingValueKind::Float;
	OaOpt<OaF64> Minimum;
	OaOpt<OaF64> Maximum;
	OaFunc<OaTrainingValue()> Get;
	OaFunc<OaStatus(const OaTrainingValue&)> Set;
};

struct OaTrainingSessionHandlers {
	OaFunc<OaStatus()> Checkpoint;
	OaFunc<OaStatus()> Evaluate;
	OaFunc<OaStatus(const OaTrainingCommand&)> Rebuild;
};

struct OaTrainingSessionConfig {
	OaU32 CommandCapacity = 64;
	OaU32 ResultCapacity = 128;
	OaU32 SnapshotCapacity = 256;
	OaTrainingSessionHandlers Handlers;
};

class OaTrainingSession {
public:
	OaTrainingSession(OaItTraining& InTraining, OaTrainingSessionConfig InConfig = {});
	~OaTrainingSession();

	OaTrainingSession(const OaTrainingSession&) = delete;
	OaTrainingSession& operator=(const OaTrainingSession&) = delete;
	OaTrainingSession(OaTrainingSession&&) = delete;
	OaTrainingSession& operator=(OaTrainingSession&&) = delete;

	// Enqueue is non-blocking. A full queue rejects the command explicitly.
	[[nodiscard]] OaResult<OaU64> Enqueue(OaTrainingCommand InCommand);
	[[nodiscard]] OaResult<OaU64> Pause(OaU64 InExpectedRevision = 0);
	[[nodiscard]] OaResult<OaU64> Resume(OaU64 InExpectedRevision = 0);
	[[nodiscard]] OaResult<OaU64> Stop(OaU64 InExpectedRevision = 0);
	[[nodiscard]] OaResult<OaU64> Checkpoint(OaU64 InExpectedRevision = 0);
	[[nodiscard]] OaResult<OaU64> Evaluate(OaU64 InExpectedRevision = 0);
	[[nodiscard]] OaResult<OaU64> SetParameter(
		OaString InName,
		OaTrainingValue InValue,
		OaU64 InExpectedRevision = 0
	);
	[[nodiscard]] OaResult<OaU64> RequestRecapture(
		OaU64 InExpectedRevision = 0
	);
	[[nodiscard]] OaResult<OaU64> RequestRebuild(
		OaTrainingValue InConfig,
		OaU64 InExpectedRevision = 0
	);

	// Non-blocking safe point for UI/event-loop owned training. Returns true only
	// when one ordinary training step may begin now.
	[[nodiscard]] bool TryBeginStep();
	// Blocking safe point for a dedicated training thread. Enqueued resume/stop
	// commands wake the wait without polling.
	[[nodiscard]] bool WaitBeginStep();
	// Apply commands without advancing the training iterator.
	[[nodiscard]] OaStatus Poll();

	[[nodiscard]] OaStatus RegisterParameter(OaTrainingParameterDesc InDesc);
	[[nodiscard]] OaOpt<OaTrainingValue> Parameter(OaStringView InName) const;
	void PublishMetric(OaString InName, OaF64 InValue);

	[[nodiscard]] OaTrainingState State() const;
	[[nodiscard]] OaU64 Revision() const;
	[[nodiscard]] OaOpt<OaTrainingSnapshot> LatestSnapshot() const;
	// Non-destructive bounded result view for independent observers such as a
	// viewer and MCP client. Results older than the configured ring capacity may
	// have been dropped; callers advance their own sequence cursor.
	[[nodiscard]] OaVec<OaTrainingCommandResult> ResultsAfter(
		OaU64 InSequence) const;
	[[nodiscard]] OaVec<OaTrainingCommandResult> TakeResults();

	// Called by OaItTraining after a completed step/reset/finish. Public for
	// custom iterator adapters, but ordinary callers do not invoke these.
	void OnStepCompleted(const OaItTraining& InTraining);
	void OnReset(const OaItTraining& InTraining);
	void OnFinished(const OaStatus& InStatus, const OaItTraining& InTraining);

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
};
