#pragma once

#include <oa/core/status.h>
#include <oa/ml/itTraining.h>
#include <oa/ml/rollout.h>

namespace oa {

class Optimizer;
class Engine;

struct ItRolloutTrainingConfig {
	oa::U32 rollouts = 0;
	oa::U32 horizon = 0;
	oa::U32 environments = 0;
	oa::U32 updateEpochs = 0;
	const char* timerName = "rl_update";
};

enum class RolloutTrainingPhase : oa::U8 {
	Collect,
	Update,
	Complete,
};

/// Coordinates the two-phase lifecycle of synchronous on-policy training.
/// Environment stepping, policy evaluation, and loss construction remain
/// caller supplied; an owned `ItTraining` controls each exact optimizer update.
class ItRolloutTraining {
public:
	ItRolloutTraining(
		oa::Engine& inEngine,
		oa::Optimizer& inOptimizer,
		const ItRolloutTrainingConfig& inConfig);

	// Starts one collection cycle and resets the supplied fixed-capacity buffer.
	[[nodiscard]] oa::Status beginRollout(RolloutBuffer& inRollout);
	// Requires a full buffer, records GAE, and opens the update phase.
	[[nodiscard]] oa::Status finalizeRollout(RolloutBuffer& inRollout, const GaeConfig& inConfig = {});
	// Restores the pre-beginRollout collect state after the caller cancels the
	// unsubmitted command transaction. This is also valid after
	// finalizeRollout and before the first update begins.
	[[nodiscard]] oa::Status abortRollout(RolloutBuffer& inRollout);

	// Must be called immediately before recording one differentiable PPO update.
	// It advances the underlying ItTraining lifecycle and returns false only on
	// invalid phase/configuration or after completion.
	[[nodiscard]] bool beginUpdate();
	// Completes the update recorded after beginUpdate: optimizer, submit, sync,
	// metrics and phase advancement.
	[[nodiscard]] oa::Status nextUpdate(const oa::Matrix& inLoss);

	[[nodiscard]] oa::Status finish();

	[[nodiscard]] bool isValid() const noexcept { return lastStatus_.isOk(); }
	[[nodiscard]] bool isDone() const noexcept {
		return phase_ == RolloutTrainingPhase::Complete || updates_.stopRequested();
	}
	[[nodiscard]] RolloutTrainingPhase phase() const noexcept { return phase_; }
	[[nodiscard]] oa::U32 rolloutIndex() const noexcept { return rolloutIndex_; }
	[[nodiscard]] oa::U32 updateEpoch() const noexcept { return updateEpoch_; }
	[[nodiscard]] const ItRolloutTrainingConfig& config() const noexcept { return config_; }
	[[nodiscard]] const oa::Status& lastStatus() const noexcept { return lastStatus_; }
	[[nodiscard]] oa::ItTraining& updateLoop() noexcept { return updates_; }
	[[nodiscard]] const oa::ItTraining& updateLoop() const noexcept { return updates_; }

private:
	ItRolloutTrainingConfig config_;
	oa::ItTraining updates_;
	oa::Status lastStatus_ = oa::Status::ok();
	RolloutTrainingPhase phase_ = RolloutTrainingPhase::Collect;
	oa::U32 rolloutIndex_ = 0;
	oa::U32 updateEpoch_ = 0;
	bool rolloutOpen_ = false;
	bool updateBodyPending_ = false;
	RolloutBuffer* activeRollout_ = nullptr;
};

} // namespace oa
