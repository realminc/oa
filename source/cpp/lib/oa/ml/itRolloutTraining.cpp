#include <oa/ml/itRolloutTraining.h>

#include <oa/ml/optim.h>
#include <oa/ml/trainingSession.h>

#include <oa/core/std/limits.h>

namespace {

bool isValidConfig(const oa::ItRolloutTrainingConfig& inConfig) {
	const oa::U64 batch = static_cast<oa::U64>(inConfig.horizon)
		* inConfig.environments;
	const oa::U64 updates = static_cast<oa::U64>(inConfig.rollouts)
		* inConfig.updateEpochs;
	return inConfig.rollouts > 0 && inConfig.horizon > 0
		&& inConfig.environments > 0 && inConfig.updateEpochs > 0
		&& batch <= static_cast<oa::U64>(oa::Limits<oa::I32>::max())
		&& updates <= static_cast<oa::U64>(oa::Limits<oa::I64>::max());
}

oa::ItTrainingConfig makeUpdateConfig(const oa::ItRolloutTrainingConfig& inConfig) {
	if (!isValidConfig(inConfig)) {
		return oa::ItTrainingConfig{.totalSteps = 0, .batchSize = 1};
	}
	return oa::ItTrainingConfig{
		.totalSteps = static_cast<oa::I64>(inConfig.rollouts)
			* inConfig.updateEpochs,
		.stepsPerEpoch = static_cast<oa::I64>(inConfig.updateEpochs),
		.batchSize = static_cast<oa::I32>(
			static_cast<oa::U64>(inConfig.horizon) * inConfig.environments),
		.timerName = inConfig.timerName != nullptr
			? inConfig.timerName : "rl_update",
	};
}

} // namespace

oa::ItRolloutTraining::ItRolloutTraining(
	oa::Engine& inEngine,
	oa::Optimizer& inOptimizer,
	const oa::ItRolloutTrainingConfig& inConfig)
	: config_(inConfig)
	, updates_(inEngine, inOptimizer, makeUpdateConfig(inConfig)) {
	if (!isValidConfig(inConfig)) {
		lastStatus_ = oa::Status::invalidArgument(
			"oa::ItRolloutTraining expects non-zero rollouts, horizon, environments and update epochs within indexing limits");
		phase_ = oa::RolloutTrainingPhase::Complete;
	}
}

oa::Status oa::ItRolloutTraining::beginRollout(oa::RolloutBuffer& inRollout) {
	if (lastStatus_.isError()) return lastStatus_;
	if (phase_ != oa::RolloutTrainingPhase::Collect
		|| rolloutOpen_ || updateBodyPending_) {
		return lastStatus_ = oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining::beginRollout requires the collect phase");
	}
	const auto& rolloutConfig = inRollout.config();
	if (!inRollout.isValid()
		|| rolloutConfig.time != config_.horizon
		|| rolloutConfig.environments != config_.environments) {
		return lastStatus_ = oa::Status::error(
			oa::StatusCode::ShapeMismatch,
			"oa::ItRolloutTraining::beginRollout buffer horizon/environments do not match the training configuration");
	}
	inRollout.reset();
	rolloutOpen_ = true;
	activeRollout_ = &inRollout;
	return lastStatus_ = oa::Status::ok();
}

oa::Status oa::ItRolloutTraining::finalizeRollout(
	oa::RolloutBuffer& inRollout,
	const oa::GaeConfig& inConfig) {
	if (lastStatus_.isError()) return lastStatus_;
	if (phase_ != oa::RolloutTrainingPhase::Collect
		|| !rolloutOpen_ || updateBodyPending_
		|| activeRollout_ != &inRollout) {
		return lastStatus_ = oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining::finalizeRollout requires the collect phase");
	}
	const oa::Status status = inRollout.finalize(inConfig);
	if (status.isError()) return lastStatus_ = status;
	rolloutOpen_ = false;
	phase_ = oa::RolloutTrainingPhase::Update;
	updateEpoch_ = 0;
	return lastStatus_ = oa::Status::ok();
}

oa::Status oa::ItRolloutTraining::abortRollout(oa::RolloutBuffer& inRollout) {
	const bool openCollection =
		phase_ == oa::RolloutTrainingPhase::Collect && rolloutOpen_;
	const bool finalizedCollection =
		phase_ == oa::RolloutTrainingPhase::Update && !rolloutOpen_
		&& updateEpoch_ == 0U;
	if ((!openCollection && !finalizedCollection)
		|| updateBodyPending_ || activeRollout_ != &inRollout) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining::abortRollout requires an unsubmitted collection before its first update");
	}
	inRollout.abortUnsubmitted();
	rolloutOpen_ = false;
	updateBodyPending_ = false;
	activeRollout_ = nullptr;
	phase_ = oa::RolloutTrainingPhase::Collect;
	updateEpoch_ = 0U;
	lastStatus_ = oa::Status::ok();
	return lastStatus_;
}

bool oa::ItRolloutTraining::beginUpdate() {
	if (lastStatus_.isError()
		|| phase_ != oa::RolloutTrainingPhase::Update
		|| updateBodyPending_
		|| updateEpoch_ >= config_.updateEpochs) {
		return false;
	}
	const bool mayBegin = updates_.session() != nullptr
		? updates_.session()->tryBeginStep()
		: !updates_.isDone();
	if (!mayBegin) {
		if (updates_.stopRequested()) phase_ = oa::RolloutTrainingPhase::Complete;
		if (updates_.session() != nullptr
			&& updates_.session()->state() == TrainingState::Paused) {
			return false;
		}
		if (updates_.stopRequested()) return false;
		lastStatus_ = oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining update iterator completed before the configured rollout schedule");
		phase_ = oa::RolloutTrainingPhase::Complete;
		return false;
	}
	updateBodyPending_ = true;
	activeRollout_ = nullptr;
	return true;
}

oa::Status oa::ItRolloutTraining::nextUpdate(const oa::Matrix& inLoss) {
	if (lastStatus_.isError()) return lastStatus_;
	if (!updateBodyPending_ || phase_ != oa::RolloutTrainingPhase::Update) {
		return lastStatus_ = oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining::nextUpdate requires a preceding beginUpdate");
	}
	if (inLoss.isEmpty() || inLoss.numElements() != 1) {
		return lastStatus_ = oa::Status::invalidArgument(
			"oa::ItRolloutTraining::nextUpdate expects one scalar loss");
	}
	updates_.next(inLoss);
	updateBodyPending_ = false;
	if (updates_.lastStatus().isError()) {
		phase_ = oa::RolloutTrainingPhase::Complete;
		return lastStatus_ = updates_.lastStatus();
	}

	++updateEpoch_;
	if (updateEpoch_ == config_.updateEpochs) {
		++rolloutIndex_;
		phase_ = rolloutIndex_ == config_.rollouts
			? oa::RolloutTrainingPhase::Complete
			: oa::RolloutTrainingPhase::Collect;
	}
	return lastStatus_ = oa::Status::ok();
}

oa::Status oa::ItRolloutTraining::finish() {
	if (lastStatus_.isError()) return lastStatus_;
	if (!isDone() || updateBodyPending_) {
		return lastStatus_ = oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ItRolloutTraining::finish requires the completed phase");
	}
	rolloutOpen_ = false;
	activeRollout_ = nullptr;
	phase_ = oa::RolloutTrainingPhase::Complete;
	return lastStatus_ = updates_.finish();
}
