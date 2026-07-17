#pragma once

#include <Oa/Ml/Rl/ActorCritic.h>
#include <Oa/Ml/Rl/FnLoss.h>
#include <Oa/Ml/Rl/Gae.h>
#include <Oa/Ml/Rl/ItRlTraining.h>
#include <Oa/Ml/Rl/Policy.h>

class OaOptimizer;

struct OaPpoTrainerConfig {
	OaU32 Rollouts = 0;
	OaU32 Horizon = 0;
	OaU32 Environments = 0;
	OaU32 UpdateEpochs = 0;
	OaMatrixShape ObservationShape;
	OaU64 Seed = 0;
	OaGaeConfig Gae;
	OaPpoLossConfig Loss;
};

struct OaPpoTrainerMetrics {
	OaU32 Rollout = 0;
	OaU32 UpdateEpoch = 0;
	OaF32 TotalLoss = 0.0F;
	OaF32 PolicyLoss = 0.0F;
	OaF32 ValueLoss = 0.0F;
	OaF32 Entropy = 0.0F;
};

// Complete environment-neutral categorical PPO loop. The environment remains
// caller-owned; this class owns collection storage, policy bookkeeping, GAE and
// optimizer updates. No tutorial code is required to train another environment.
class OaPpoTrainer {
public:
	[[nodiscard]] static OaResult<OaUniquePtr<OaPpoTrainer>> Create(
		OaRlActorCritic& InModel,
		OaOptimizer& InOptimizer,
		const OaPpoTrainerConfig& InConfig
	);

	OaPpoTrainer(const OaPpoTrainer&) = delete;
	OaPpoTrainer& operator=(const OaPpoTrainer&) = delete;
	~OaPpoTrainer();

	[[nodiscard]] OaStatus BeginCollection();
	[[nodiscard]] OaRlPolicyResult Act(const OaMatrix& InObservation);
	[[nodiscard]] OaStatus Observe(
		const OaMatrix& InObservation,
		const OaMatrix& InNextObservation,
		const OaMatrix& InReward,
		const OaMatrix& InTerminated,
		const OaMatrix& InTruncated,
		const OaRlPolicyResult& InPolicy
	);
	[[nodiscard]] OaStatus EndCollection();
	// Performs one PPO update epoch. Call until NeedsCollection() or IsDone().
	[[nodiscard]] OaStatus Update();

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] bool IsDone() const noexcept;
	[[nodiscard]] bool NeedsCollection() const noexcept;
	[[nodiscard]] OaRlTrainingPhase Phase() const noexcept;
	[[nodiscard]] const OaPpoTrainerConfig& Config() const noexcept;
	[[nodiscard]] const OaPpoTrainerMetrics& Metrics() const noexcept;
	[[nodiscard]] const OaRlRolloutBatch& Batch() const noexcept;
	// The ordinary optimizer-update iterator shared by supervised and RL
	// training. Attach OaTrainingSession here for live control/observation.
	[[nodiscard]] OaItTraining& TrainingLoop() noexcept;
	[[nodiscard]] const OaItTraining& TrainingLoop() const noexcept;
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);

private:
	struct Impl;
	explicit OaPpoTrainer(OaUniquePtr<Impl> InImpl);
	OaUniquePtr<Impl> Impl_;
};
