#pragma once

#include <oa/ml/actorCritic.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/advantage.h>
#include <oa/ml/itRolloutTraining.h>
#include <oa/ml/policy.h>

namespace oa {

class Optimizer;
class Engine;

struct PpoTrainerConfig {
	oa::U32 rollouts = 0;
	oa::U32 horizon = 0;
	oa::U32 environments = 0;
	oa::U32 updateEpochs = 0;
	oa::MatrixShape observationShape;
	oa::U64 seed = 0;
	GaeConfig gae;
	PpoLossConfig loss;
};

struct PpoTrainerMetrics {
	oa::U32 rollout = 0;
	oa::U32 updateEpoch = 0;
	oa::F32 totalLoss = 0.0F;
	oa::F32 policyLoss = 0.0F;
	oa::F32 valueLoss = 0.0F;
	oa::F32 entropy = 0.0F;
};

/// Complete environment-neutral categorical PPO loop. The environment remains
/// caller-owned; this class owns collection storage, policy bookkeeping, GAE,
/// and optimizer updates. It composes `ItRolloutTraining` because collection
/// and update phases are not the same lifecycle as DQN or SAC.
class PpoTrainer {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<PpoTrainer>> create(
		oa::Engine& inEngine,
		ActorCritic& inModel,
		oa::Optimizer& inOptimizer,
		const PpoTrainerConfig& inConfig
	);

	PpoTrainer(const PpoTrainer&) = delete;
	PpoTrainer& operator=(const PpoTrainer&) = delete;
	~PpoTrainer();

	[[nodiscard]] oa::Status beginCollection();
	[[nodiscard]] PolicyResult act(const oa::Matrix& inObservation);
	[[nodiscard]] oa::Status observe(
		const oa::Matrix& inObservation,
		const oa::Matrix& inNextObservation,
		const oa::Matrix& inReward,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		const PolicyResult& inPolicy
	);
	[[nodiscard]] oa::Status endCollection();
	// Rolls back collection control state after the caller cancels the
	// unsubmitted command transaction. valid from beginCollection through
	// endCollection, until the first update actually begins.
	[[nodiscard]] oa::Status abortCollection();
	// Performs one PPO update epoch. call until needsCollection() or isDone().
	[[nodiscard]] oa::Status update();

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] bool isDone() const noexcept;
	[[nodiscard]] bool needsCollection() const noexcept;
	[[nodiscard]] RolloutTrainingPhase phase() const noexcept;
	[[nodiscard]] const PpoTrainerConfig& config() const noexcept;
	[[nodiscard]] const PpoTrainerMetrics& metrics() const noexcept;
	[[nodiscard]] const RolloutBatch& batch() const noexcept;
	// The ordinary optimizer-update iterator shared by supervised and RL
	// training. attach TrainingSession here for live control/observation.
	[[nodiscard]] oa::ItTraining& trainingLoop() noexcept;
	[[nodiscard]] const oa::ItTraining& trainingLoop() const noexcept;
	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(const oa::String& inPath);

private:
	struct Impl;
	explicit PpoTrainer(oa::UniquePtr<Impl> inImpl);
	oa::UniquePtr<Impl> impl_;
};
} // namespace oa
