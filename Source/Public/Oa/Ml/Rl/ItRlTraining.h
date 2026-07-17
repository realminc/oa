#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Rl/Rollout.h>

class OaOptimizer;

struct OaItRlTrainingConfig {
	OaU32 Rollouts = 0;
	OaU32 Horizon = 0;
	OaU32 Environments = 0;
	OaU32 UpdateEpochs = 0;
	const char* TimerName = "rl_update";
};

enum class OaRlTrainingPhase : OaU8 {
	Collect,
	Update,
	Complete,
};

// Coordinates the two-phase lifecycle of synchronous on-policy training.
// Environment stepping, policy evaluation and loss construction remain caller
// supplied; ordinary OaItTraining owns each exact optimizer update.
class OaItRlTraining {
public:
	OaItRlTraining(OaOptimizer& InOptimizer, const OaItRlTrainingConfig& InConfig);

	// Starts one collection cycle and resets the supplied fixed-capacity buffer.
	[[nodiscard]] OaStatus BeginRollout(OaRlRolloutBuffer& InRollout);
	// Requires a full buffer, records GAE, and opens the update phase.
	[[nodiscard]] OaStatus FinalizeRollout(OaRlRolloutBuffer& InRollout, const OaGaeConfig& InConfig = {});

	// Must be called immediately before recording one differentiable PPO update.
	// It advances the underlying OaItTraining lifecycle and returns false only on
	// invalid phase/configuration or after completion.
	[[nodiscard]] bool BeginUpdate();
	// Completes the update recorded after BeginUpdate: optimizer, submit, sync,
	// metrics and phase advancement.
	[[nodiscard]] OaStatus NextUpdate(const OaMatrix& InLoss);

	[[nodiscard]] OaStatus Finish();

	[[nodiscard]] bool IsValid() const noexcept { return LastStatus_.IsOk(); }
	[[nodiscard]] bool IsDone() const noexcept {
		return Phase_ == OaRlTrainingPhase::Complete || Updates_.StopRequested();
	}
	[[nodiscard]] OaRlTrainingPhase Phase() const noexcept { return Phase_; }
	[[nodiscard]] OaU32 RolloutIndex() const noexcept { return RolloutIndex_; }
	[[nodiscard]] OaU32 UpdateEpoch() const noexcept { return UpdateEpoch_; }
	[[nodiscard]] const OaItRlTrainingConfig& Config() const noexcept { return Config_; }
	[[nodiscard]] const OaStatus& LastStatus() const noexcept { return LastStatus_; }
	[[nodiscard]] OaItTraining& UpdateLoop() noexcept { return Updates_; }
	[[nodiscard]] const OaItTraining& UpdateLoop() const noexcept { return Updates_; }

private:
	OaItRlTrainingConfig Config_;
	OaItTraining Updates_;
	OaStatus LastStatus_ = OaStatus::Ok();
	OaRlTrainingPhase Phase_ = OaRlTrainingPhase::Collect;
	OaU32 RolloutIndex_ = 0;
	OaU32 UpdateEpoch_ = 0;
	bool RolloutOpen_ = false;
	bool UpdateBodyPending_ = false;
};
