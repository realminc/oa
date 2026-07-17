#include <Oa/Ml/Rl/ItRlTraining.h>

#include <Oa/Ml/Optim.h>
#include <Oa/Ml/TrainingSession.h>

#include <limits>

namespace {

bool IsValidConfig(const OaItRlTrainingConfig& InConfig) {
	const OaU64 batch = static_cast<OaU64>(InConfig.Horizon)
		* InConfig.Environments;
	const OaU64 updates = static_cast<OaU64>(InConfig.Rollouts)
		* InConfig.UpdateEpochs;
	return InConfig.Rollouts > 0 && InConfig.Horizon > 0
		&& InConfig.Environments > 0 && InConfig.UpdateEpochs > 0
		&& batch <= static_cast<OaU64>(std::numeric_limits<OaI32>::max())
		&& updates <= static_cast<OaU64>(std::numeric_limits<OaI64>::max());
}

OaItTrainingConfig MakeUpdateConfig(const OaItRlTrainingConfig& InConfig) {
	if (!IsValidConfig(InConfig)) {
		return OaItTrainingConfig{.TotalSteps = 0, .BatchSize = 1};
	}
	return OaItTrainingConfig{
		.TotalSteps = static_cast<OaI64>(InConfig.Rollouts)
			* InConfig.UpdateEpochs,
		.StepsPerEpoch = static_cast<OaI64>(InConfig.UpdateEpochs),
		.BatchSize = static_cast<OaI32>(
			static_cast<OaU64>(InConfig.Horizon) * InConfig.Environments),
		.TimerName = InConfig.TimerName != nullptr
			? InConfig.TimerName : "rl_update",
	};
}

} // namespace

OaItRlTraining::OaItRlTraining(
	OaOptimizer& InOptimizer,
	const OaItRlTrainingConfig& InConfig)
	: Config_(InConfig)
	, Updates_(InOptimizer, MakeUpdateConfig(InConfig)) {
	if (!IsValidConfig(InConfig)) {
		LastStatus_ = OaStatus::InvalidArgument(
			"OaItRlTraining expects non-zero rollouts, horizon, environments and update epochs within indexing limits");
		Phase_ = OaRlTrainingPhase::Complete;
	}
}

OaStatus OaItRlTraining::BeginRollout(OaRlRolloutBuffer& InRollout) {
	if (LastStatus_.IsError()) return LastStatus_;
	if (Phase_ != OaRlTrainingPhase::Collect
		|| RolloutOpen_ || UpdateBodyPending_) {
		return LastStatus_ = OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaItRlTraining::BeginRollout requires the collect phase");
	}
	const auto& rolloutConfig = InRollout.Config();
	if (!InRollout.IsValid()
		|| rolloutConfig.Time != Config_.Horizon
		|| rolloutConfig.Environments != Config_.Environments) {
		return LastStatus_ = OaStatus::Error(
			OaStatusCode::ShapeMismatch,
			"OaItRlTraining::BeginRollout buffer horizon/environments do not match the training configuration");
	}
	InRollout.Reset();
	RolloutOpen_ = true;
	return LastStatus_ = OaStatus::Ok();
}

OaStatus OaItRlTraining::FinalizeRollout(
	OaRlRolloutBuffer& InRollout,
	const OaGaeConfig& InConfig) {
	if (LastStatus_.IsError()) return LastStatus_;
	if (Phase_ != OaRlTrainingPhase::Collect
		|| !RolloutOpen_ || UpdateBodyPending_) {
		return LastStatus_ = OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaItRlTraining::FinalizeRollout requires the collect phase");
	}
	const OaStatus status = InRollout.Finalize(InConfig);
	if (status.IsError()) return LastStatus_ = status;
	RolloutOpen_ = false;
	Phase_ = OaRlTrainingPhase::Update;
	UpdateEpoch_ = 0;
	return LastStatus_ = OaStatus::Ok();
}

bool OaItRlTraining::BeginUpdate() {
	if (LastStatus_.IsError()
		|| Phase_ != OaRlTrainingPhase::Update
		|| UpdateBodyPending_
		|| UpdateEpoch_ >= Config_.UpdateEpochs) {
		return false;
	}
	const bool mayBegin = Updates_.Session() != nullptr
		? Updates_.Session()->TryBeginStep()
		: !Updates_.IsDone();
	if (!mayBegin) {
		if (Updates_.StopRequested()) Phase_ = OaRlTrainingPhase::Complete;
		if (Updates_.Session() != nullptr
			&& Updates_.Session()->State() == OaTrainingState::Paused) {
			return false;
		}
		if (Updates_.StopRequested()) return false;
		LastStatus_ = OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaItRlTraining update iterator completed before the configured rollout schedule");
		Phase_ = OaRlTrainingPhase::Complete;
		return false;
	}
	UpdateBodyPending_ = true;
	return true;
}

OaStatus OaItRlTraining::NextUpdate(const OaMatrix& InLoss) {
	if (LastStatus_.IsError()) return LastStatus_;
	if (!UpdateBodyPending_ || Phase_ != OaRlTrainingPhase::Update) {
		return LastStatus_ = OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaItRlTraining::NextUpdate requires a preceding BeginUpdate");
	}
	if (InLoss.IsEmpty() || InLoss.NumElements() != 1) {
		return LastStatus_ = OaStatus::InvalidArgument(
			"OaItRlTraining::NextUpdate expects one scalar loss");
	}
	Updates_.Next(InLoss);
	UpdateBodyPending_ = false;
	if (Updates_.LastStatus().IsError()) {
		Phase_ = OaRlTrainingPhase::Complete;
		return LastStatus_ = Updates_.LastStatus();
	}

	++UpdateEpoch_;
	if (UpdateEpoch_ == Config_.UpdateEpochs) {
		++RolloutIndex_;
		Phase_ = RolloutIndex_ == Config_.Rollouts
			? OaRlTrainingPhase::Complete
			: OaRlTrainingPhase::Collect;
	}
	return LastStatus_ = OaStatus::Ok();
}

OaStatus OaItRlTraining::Finish() {
	if (LastStatus_.IsError()) return LastStatus_;
	if (!IsDone() || UpdateBodyPending_) {
		return LastStatus_ = OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"OaItRlTraining::Finish requires the completed phase");
	}
	RolloutOpen_ = false;
	Phase_ = OaRlTrainingPhase::Complete;
	return LastStatus_ = Updates_.Finish();
}
