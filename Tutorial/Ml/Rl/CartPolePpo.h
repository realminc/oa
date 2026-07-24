#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

struct OaTutorialCartPolePpoConfig {
	OaU32 Environments = 64;
	OaU32 Horizon = 128;
	OaU32 Rollouts = 40;
	OaU32 UpdateEpochs = 4;
	OaU64 TrainingSeed = 0x0a11ce55ULL;
	OaF32 LearningRate = 2.5e-4F;
};

struct OaTutorialCartPolePpoEvaluation {
	OaF64 MeanCompletedReturn = 0.0;
	OaU32 CompletedEpisodes = 0;
};

struct OaTutorialCartPoleSnapshot {
	OaF32 CartPosition = 0.0F;
	OaF32 CartVelocity = 0.0F;
	OaF32 PoleAngle = 0.0F;
	OaF32 PoleAngularVelocity = 0.0F;
};

struct OaTutorialCartPolePpoMetrics {
	OaU32 Rollout = 0;
	OaU32 UpdateEpoch = 0;
	OaF32 TotalLoss = 0.0F;
	OaF32 PolicyLoss = 0.0F;
	OaF32 ValueLoss = 0.0F;
	OaF32 Entropy = 0.0F;
	OaVec<OaF32> LossHistory;
	OaVec<OaF32> PolicyLossHistory;
	OaVec<OaF32> ValueLossHistory;
	OaVec<OaF32> EntropyHistory;
	OaVec<OaF32> EvaluationReturnHistory;
};

class OaTrainingSession;
class OaEngine;

// Tutorial-local incremental PPO session shared by the headless acceptance test
// and OaViewer. Advance() performs at most one optimizer update, allowing the UI
// event loop to remain responsive without creating a second trainer API.
class OaTutorialCartPolePpo {
public:
	static OaResult<OaUniquePtr<OaTutorialCartPolePpo>> Create(
		OaEngine& InEngine,
		const OaTutorialCartPolePpoConfig& InConfig = {});

	OaTutorialCartPolePpo(const OaTutorialCartPolePpo&) = delete;
	OaTutorialCartPolePpo& operator=(const OaTutorialCartPolePpo&) = delete;
	~OaTutorialCartPolePpo();

	[[nodiscard]] OaStatus Advance();
	[[nodiscard]] bool IsDone() const noexcept;
	[[nodiscard]] const OaTutorialCartPolePpoConfig& Config() const noexcept;
	[[nodiscard]] const OaTutorialCartPolePpoMetrics& Metrics() const noexcept;
	[[nodiscard]] OaResult<OaTutorialCartPoleSnapshot> SnapshotLane(
		OaU32 InLane = 0);
	// Advances the vector environment with the greedy policy without updating
	// parameters. Used only by attached visualization after training.
	[[nodiscard]] OaStatus Demonstrate();
	[[nodiscard]] OaResult<OaTutorialCartPolePpoEvaluation> Evaluate(
		OaU64 InSeed,
		OaU32 InEnvironments = 64,
		OaU32 InHorizon = 500);
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);
	[[nodiscard]] OaU64 OptimizerStep() const noexcept;
	// The same typed controller used by ordinary ML, DQN and SAC. The tutorial
	// viewer queues commands here rather than maintaining a second pause state.
	[[nodiscard]] OaTrainingSession& Control() noexcept;
	[[nodiscard]] const OaTrainingSession& Control() const noexcept;

private:
	struct Impl;
	explicit OaTutorialCartPolePpo(OaUniquePtr<Impl> InImpl);
	OaUniquePtr<Impl> Impl_;
};
