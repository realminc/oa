#include <Oa/Ml/TrainingSession.h>

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Optim.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

OaTrainingValue OaTrainingValue::FromBool(OaBool InValue) {
	OaTrainingValue value;
	value.Kind = OaTrainingValueKind::Boolean;
	value.Bool = InValue;
	return value;
}

OaTrainingValue OaTrainingValue::FromInteger(OaI64 InValue) {
	OaTrainingValue value;
	value.Kind = OaTrainingValueKind::Integer;
	value.Integer = InValue;
	return value;
}

OaTrainingValue OaTrainingValue::FromFloat(OaF64 InValue) {
	OaTrainingValue value;
	value.Kind = OaTrainingValueKind::Float;
	value.Float = InValue;
	return value;
}

OaTrainingValue OaTrainingValue::FromString(OaString InValue) {
	OaTrainingValue value;
	value.Kind = OaTrainingValueKind::String;
	value.String = OaStdMove(InValue);
	return value;
}

OaOpt<OaF64> OaTrainingValue::AsNumber() const {
	if (Kind == OaTrainingValueKind::Float) return Float;
	if (Kind == OaTrainingValueKind::Integer) {
		return static_cast<OaF64>(Integer);
	}
	return {};
}

struct OaTrainingSession::Impl {
	OaItTraining* Training = nullptr;
	OaTrainingSessionConfig Config;
	mutable std::mutex Mutex;
	std::condition_variable Wake;
	OaTrainingState State = OaTrainingState::Running;
	OaU64 Revision = 0;
	OaU64 NextSequence = 1;
	OaU64 TakeSequence = 0;
	std::deque<OaTrainingCommand> Commands;
	std::deque<OaTrainingCommandResult> Results;
	std::deque<OaTrainingSnapshot> Snapshots;
	std::vector<OaTrainingParameterDesc> Parameters;
	std::vector<OaTrainingMetricSample> PendingMetrics;

	void BoundResults() {
		while (Results.size() > Config.ResultCapacity) Results.pop_front();
	}

	void BoundSnapshots() {
		while (Snapshots.size() > Config.SnapshotCapacity) Snapshots.pop_front();
	}
};

namespace {

bool IsTerminal(OaTrainingState InState) {
	return InState == OaTrainingState::Stopping
		|| InState == OaTrainingState::Completed
		|| InState == OaTrainingState::Failed;
}

OaStatus MissingHandler(const char* InName) {
	return OaStatus::Error(OaStatusCode::FailedPrecondition,
		OaString("OaTrainingSession: no ") + InName + " handler is registered");
}

} // namespace

OaTrainingSession::OaTrainingSession(
	OaItTraining& InTraining,
	OaTrainingSessionConfig InConfig)
	: Impl_(OaMakeUniquePtr<Impl>()) {
	Impl_->Training = &InTraining;
	Impl_->Config = OaStdMove(InConfig);
	Impl_->Config.CommandCapacity = std::max<OaU32>(Impl_->Config.CommandCapacity, 1);
	Impl_->Config.ResultCapacity = std::max<OaU32>(Impl_->Config.ResultCapacity, 1);
	Impl_->Config.SnapshotCapacity = std::max<OaU32>(Impl_->Config.SnapshotCapacity, 1);
	InTraining.AttachSession(this);

	OaTrainingParameterDesc learningRate;
	learningRate.Name = "learning_rate";
	learningRate.Class = OaTrainingParameterClass::Hot;
	learningRate.Kind = OaTrainingValueKind::Float;
	learningRate.Minimum = std::numeric_limits<OaF64>::min();
	learningRate.Get = [&InTraining]() {
		return OaTrainingValue::FromFloat(InTraining.Optimizer().GetLr());
	};
	learningRate.Set = [&InTraining](const OaTrainingValue& InValue) {
		const auto number = InValue.AsNumber();
		if (!number.HasValue() || !std::isfinite(*number) || *number <= 0.0
			|| *number > static_cast<OaF64>(std::numeric_limits<OaF32>::max())) {
			return OaStatus::InvalidArgument(
				"learning_rate expects one finite positive Float value");
		}
		InTraining.Optimizer().SetLr(static_cast<OaF32>(*number));
		return OaStatus::Ok();
	};
	(void)RegisterParameter(OaStdMove(learningRate));
	OnReset(InTraining);
}

OaTrainingSession::~OaTrainingSession() {
	if (Impl_ && Impl_->Training && Impl_->Training->Session() == this) {
		Impl_->Training->AttachSession(nullptr);
	}
}

OaResult<OaU64> OaTrainingSession::Enqueue(OaTrainingCommand InCommand) {
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	if (Impl_->Commands.size() >= Impl_->Config.CommandCapacity) {
		return OaResult<OaU64>(OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaTrainingSession command queue is full"));
	}
	// Sequence numbers are session-owned so duplicate or out-of-order caller
	// values cannot break independent result cursors.
	InCommand.Sequence = Impl_->NextSequence++;
	const OaU64 sequence = InCommand.Sequence;
	Impl_->Commands.push_back(OaStdMove(InCommand));
	Impl_->Wake.notify_all();
	return OaResult<OaU64>(sequence);
}

OaResult<OaU64> OaTrainingSession::Pause(OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::Pause});
}

OaResult<OaU64> OaTrainingSession::Resume(OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::Resume});
}

OaResult<OaU64> OaTrainingSession::Stop(OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::Stop});
}

OaResult<OaU64> OaTrainingSession::Checkpoint(OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::Checkpoint});
}

OaResult<OaU64> OaTrainingSession::Evaluate(OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::Evaluate});
}

OaResult<OaU64> OaTrainingSession::SetParameter(
	OaString InName,
	OaTrainingValue InValue,
	OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::SetParameter,
		.Parameter = OaStdMove(InName),
		.Value = OaStdMove(InValue)});
}

OaResult<OaU64> OaTrainingSession::RequestRecapture(
	OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::RequestRecapture});
}

OaResult<OaU64> OaTrainingSession::RequestRebuild(
	OaTrainingValue InConfig,
	OaU64 InExpectedRevision) {
	return Enqueue({.ExpectedRevision = InExpectedRevision,
		.Kind = OaTrainingCommandKind::RequestRebuild,
		.Value = OaStdMove(InConfig)});
}

OaStatus OaTrainingSession::RegisterParameter(OaTrainingParameterDesc InDesc) {
	if (InDesc.Name.empty() || !InDesc.Get) {
		return OaStatus::InvalidArgument(
			"OaTrainingSession parameter requires a name and getter");
	}
	if (InDesc.Class != OaTrainingParameterClass::Immutable && !InDesc.Set) {
		return OaStatus::InvalidArgument(
			"mutable OaTrainingSession parameter requires a setter");
	}
	if (InDesc.Minimum.HasValue() && InDesc.Maximum.HasValue()
		&& *InDesc.Minimum > *InDesc.Maximum) {
		return OaStatus::InvalidArgument(
			"OaTrainingSession parameter minimum exceeds maximum");
	}
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	for (const auto& parameter : Impl_->Parameters) {
		if (parameter.Name == InDesc.Name) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				"OaTrainingSession parameter already exists: " + InDesc.Name);
		}
	}
	Impl_->Parameters.push_back(OaStdMove(InDesc));
	return OaStatus::Ok();
}

OaOpt<OaTrainingValue> OaTrainingSession::Parameter(OaStringView InName) const {
	OaFunc<OaTrainingValue()> getter;
	{
		std::lock_guard<std::mutex> lock(Impl_->Mutex);
		for (const auto& parameter : Impl_->Parameters) {
			if (parameter.Name == InName) {
				getter = parameter.Get;
				break;
			}
		}
	}
	if (!getter) return {};
	return getter();
}

void OaTrainingSession::PublishMetric(OaString InName, OaF64 InValue) {
	if (InName.empty() || !std::isfinite(InValue)) return;
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	for (auto& metric : Impl_->PendingMetrics) {
		if (metric.Name == InName) {
			metric.Value = InValue;
			return;
		}
	}
	Impl_->PendingMetrics.push_back({.Name = OaStdMove(InName), .Value = InValue});
}

OaStatus OaTrainingSession::Poll() {
	std::deque<OaTrainingCommand> commands;
	{
		std::lock_guard<std::mutex> lock(Impl_->Mutex);
		commands.swap(Impl_->Commands);
	}

	for (const auto& command : commands) {
		OaStatus status = OaStatus::Ok();
		OaTrainingParameterDesc parameter;
		bool hasParameter = false;
		OaTrainingState state;
		OaU64 revision;
		{
			std::lock_guard<std::mutex> lock(Impl_->Mutex);
			state = Impl_->State;
			revision = Impl_->Revision;
			if (command.ExpectedRevision != 0
				&& command.ExpectedRevision != revision) {
				status = OaStatus::Error(OaStatusCode::Aborted,
					"OaTrainingSession command revision is stale");
			}
			if (status.IsOk() && command.Kind == OaTrainingCommandKind::SetParameter) {
				for (const auto& candidate : Impl_->Parameters) {
					if (candidate.Name == command.Parameter) {
						parameter = candidate;
						hasParameter = true;
						break;
					}
				}
			}
		}

		if (status.IsOk()) {
			switch (command.Kind) {
				case OaTrainingCommandKind::Pause:
					if (state != OaTrainingState::Running) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"pause requires a running training session");
					}
					break;
				case OaTrainingCommandKind::Resume:
					if (state != OaTrainingState::Paused) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"resume requires a paused training session");
					}
					break;
				case OaTrainingCommandKind::Stop:
					if (IsTerminal(state)) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"stop requires an active training session");
					} else {
						Impl_->Training->RequestStop();
					}
					break;
				case OaTrainingCommandKind::Checkpoint:
					status = Impl_->Config.Handlers.Checkpoint
						? Impl_->Config.Handlers.Checkpoint()
						: MissingHandler("checkpoint");
					break;
				case OaTrainingCommandKind::Evaluate:
					status = Impl_->Config.Handlers.Evaluate
						? Impl_->Config.Handlers.Evaluate()
						: MissingHandler("evaluation");
					break;
				case OaTrainingCommandKind::SetParameter: {
					if (!hasParameter) {
						status = OaStatus::NotFound(
							"unknown training parameter: " + command.Parameter);
						break;
					}
					if (parameter.Class == OaTrainingParameterClass::Immutable) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"training parameter is immutable: " + command.Parameter);
						break;
					}
					if (parameter.Class != OaTrainingParameterClass::Hot
						&& state != OaTrainingState::Paused) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"recapture/rebuild parameters require a paused session");
						break;
					}
					if (command.Value.Kind != parameter.Kind
						&& !(parameter.Kind == OaTrainingValueKind::Float
							&& command.Value.Kind == OaTrainingValueKind::Integer)) {
						status = OaStatus::InvalidArgument(
							"training parameter value kind does not match its declaration");
						break;
					}
					const auto number = command.Value.AsNumber();
					if (number.HasValue()
						&& ((parameter.Minimum.HasValue() && *number < *parameter.Minimum)
							|| (parameter.Maximum.HasValue() && *number > *parameter.Maximum))) {
						status = OaStatus::Error(OaStatusCode::OutOfRange,
							"training parameter value is outside its declared range");
						break;
					}
					status = parameter.Set(command.Value);
					break;
				}
				case OaTrainingCommandKind::RequestRecapture:
					if (state != OaTrainingState::Paused) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"program recapture requires a paused session");
					} else {
						status = Impl_->Training->RequestProgramRecapture();
					}
					break;
				case OaTrainingCommandKind::RequestRebuild:
					if (state != OaTrainingState::Paused) {
						status = OaStatus::Error(OaStatusCode::FailedPrecondition,
							"training rebuild requires a paused session");
					} else {
						status = Impl_->Config.Handlers.Rebuild
							? Impl_->Config.Handlers.Rebuild(command)
							: MissingHandler("rebuild");
					}
					break;
			}
		}

		{
			std::lock_guard<std::mutex> lock(Impl_->Mutex);
			if (status.IsOk()) {
				switch (command.Kind) {
					case OaTrainingCommandKind::Pause:
						Impl_->State = OaTrainingState::Paused;
						break;
					case OaTrainingCommandKind::Resume:
						Impl_->State = OaTrainingState::Running;
						break;
					case OaTrainingCommandKind::Stop:
						Impl_->State = OaTrainingState::Stopping;
						break;
					default:
						break;
				}
				++Impl_->Revision;
			}
			Impl_->Results.push_back({
				.Sequence = command.Sequence,
				.Revision = Impl_->Revision,
				.Disposition = status.IsOk()
					? OaTrainingCommandDisposition::Applied
					: OaTrainingCommandDisposition::Rejected,
				.State = Impl_->State,
				.Status = status,
			});
			Impl_->BoundResults();
			Impl_->Wake.notify_all();
		}
	}
	return OaStatus::Ok();
}

bool OaTrainingSession::TryBeginStep() {
	(void)Poll();
	{
		std::lock_guard<std::mutex> lock(Impl_->Mutex);
		if (Impl_->State != OaTrainingState::Running) return false;
	}
	return !Impl_->Training->IsDone();
}

bool OaTrainingSession::WaitBeginStep() {
	for (;;) {
		(void)Poll();
		{
			std::unique_lock<std::mutex> lock(Impl_->Mutex);
			if (IsTerminal(Impl_->State)) return false;
			if (Impl_->State == OaTrainingState::Running) break;
			Impl_->Wake.wait(lock, [this] {
				return !Impl_->Commands.empty()
					|| Impl_->State != OaTrainingState::Paused;
			});
		}
	}
	return !Impl_->Training->IsDone();
}

OaTrainingState OaTrainingSession::State() const {
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	return Impl_->State;
}

OaU64 OaTrainingSession::Revision() const {
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	return Impl_->Revision;
}

OaOpt<OaTrainingSnapshot> OaTrainingSession::LatestSnapshot() const {
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	if (Impl_->Snapshots.empty()) return {};
	return Impl_->Snapshots.back();
}

OaVec<OaTrainingCommandResult> OaTrainingSession::ResultsAfter(
	OaU64 InSequence) const {
	OaVec<OaTrainingCommandResult> results;
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	for (const auto& result : Impl_->Results) {
		if (result.Sequence > InSequence) results.PushBack(result);
	}
	return results;
}

OaVec<OaTrainingCommandResult> OaTrainingSession::TakeResults() {
	OaVec<OaTrainingCommandResult> results;
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	results.Reserve(Impl_->Results.size());
	for (const auto& result : Impl_->Results) {
		if (result.Sequence <= Impl_->TakeSequence) continue;
		results.PushBack(result);
		Impl_->TakeSequence = std::max(Impl_->TakeSequence, result.Sequence);
	}
	return results;
}

void OaTrainingSession::OnStepCompleted(const OaItTraining& InTraining) {
	OaTrainingSnapshot snapshot;
	snapshot.Step = InTraining.StepCount();
	snapshot.Epoch = InTraining.Epoch();
	snapshot.LearningRate = InTraining.Optimizer().GetLr();
	snapshot.Loss = InTraining.LastLoss();
	snapshot.GpuMs = InTraining.LastGpuMs();
	snapshot.WallMs = InTraining.WallMsPerStep();
	{
		std::lock_guard<std::mutex> lock(Impl_->Mutex);
		snapshot.Revision = Impl_->Revision;
		snapshot.State = Impl_->State;
		snapshot.Metrics.Reserve(static_cast<OaI64>(Impl_->PendingMetrics.size()));
		for (auto metric : Impl_->PendingMetrics) {
			metric.Step = snapshot.Step;
			snapshot.Metrics.PushBack(OaStdMove(metric));
		}
		Impl_->Snapshots.push_back(OaStdMove(snapshot));
		Impl_->BoundSnapshots();
	}
}

void OaTrainingSession::OnReset(const OaItTraining& InTraining) {
	OaTrainingSnapshot snapshot;
	snapshot.Step = InTraining.StepCount();
	snapshot.Epoch = InTraining.Epoch();
	snapshot.LearningRate = InTraining.Optimizer().GetLr();
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	Impl_->State = OaTrainingState::Running;
	++Impl_->Revision;
	snapshot.Revision = Impl_->Revision;
	snapshot.State = Impl_->State;
	Impl_->Snapshots.push_back(OaStdMove(snapshot));
	Impl_->BoundSnapshots();
	Impl_->Wake.notify_all();
}

void OaTrainingSession::OnFinished(
	const OaStatus& InStatus,
	const OaItTraining& InTraining) {
	OaTrainingSnapshot snapshot;
	snapshot.Step = InTraining.StepCount();
	snapshot.Epoch = InTraining.Epoch();
	snapshot.LearningRate = InTraining.Optimizer().GetLr();
	snapshot.Loss = InTraining.LastLoss();
	snapshot.GpuMs = InTraining.LastGpuMs();
	snapshot.WallMs = InTraining.WallMsPerStep();
	std::lock_guard<std::mutex> lock(Impl_->Mutex);
	Impl_->State = InStatus.IsOk()
		? OaTrainingState::Completed : OaTrainingState::Failed;
	++Impl_->Revision;
	snapshot.Revision = Impl_->Revision;
	snapshot.State = Impl_->State;
	Impl_->Snapshots.push_back(OaStdMove(snapshot));
	Impl_->BoundSnapshots();
	Impl_->Wake.notify_all();
}
