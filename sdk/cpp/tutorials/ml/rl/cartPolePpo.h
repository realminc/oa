#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

struct TutorialCartPolePpoConfig {
	oa::U32 environments = 64;
	oa::U32 horizon = 128;
	oa::U32 rollouts = 40;
	oa::U32 updateEpochs = 4;
	oa::U64 trainingSeed = 0x0a11ce55ULL;
	oa::F32 learningRate = 2.5e-4F;
};

struct TutorialCartPolePpoEvaluation {
	oa::F64 meanCompletedReturn = 0.0;
	oa::U32 completedEpisodes = 0;
};

struct TutorialCartPoleSnapshot {
	oa::F32 cartPosition = 0.0F;
	oa::F32 cartVelocity = 0.0F;
	oa::F32 poleAngle = 0.0F;
	oa::F32 poleAngularVelocity = 0.0F;
};

struct TutorialCartPolePpoMetrics {
	oa::U32 rollout = 0;
	oa::U32 updateEpoch = 0;
	oa::F32 totalLoss = 0.0F;
	oa::F32 policyLoss = 0.0F;
	oa::F32 valueLoss = 0.0F;
	oa::F32 entropy = 0.0F;
	oa::Vector<oa::F32> lossHistory;
	oa::Vector<oa::F32> policyLossHistory;
	oa::Vector<oa::F32> valueLossHistory;
	oa::Vector<oa::F32> entropyHistory;
	oa::Vector<oa::F32> evaluationReturnHistory;
};

namespace oa {
class TrainingSession;
class Engine;
} // namespace oa

// Tutorial-local incremental PPO session shared by the headless acceptance test
// and oa::Viewer. advance() performs at most one optimizer update, allowing the UI
// event loop to remain responsive without creating a second trainer API.
class TutorialCartPolePpo {
public:
	static oa::Result<oa::UniquePtr<TutorialCartPolePpo>> create(
		oa::Engine& inEngine,
		const TutorialCartPolePpoConfig& inConfig = {});

	TutorialCartPolePpo(const TutorialCartPolePpo&) = delete;
	TutorialCartPolePpo& operator=(const TutorialCartPolePpo&) = delete;
	~TutorialCartPolePpo();

	[[nodiscard]] oa::Status advance();
	[[nodiscard]] bool isDone() const noexcept;
	[[nodiscard]] const TutorialCartPolePpoConfig& config() const noexcept;
	[[nodiscard]] const TutorialCartPolePpoMetrics& metrics() const noexcept;
	[[nodiscard]] oa::Result<TutorialCartPoleSnapshot> snapshotLane(
		oa::U32 inLane = 0);
	// Advances the vector environment with the greedy policy without updating
	// parameters. Used only by attached visualization after training.
	[[nodiscard]] oa::Status demonstrate();
	[[nodiscard]] oa::Result<TutorialCartPolePpoEvaluation> evaluate(
		oa::U64 inSeed,
		oa::U32 inEnvironments = 64,
		oa::U32 inHorizon = 500);
	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(const oa::String& inPath);
	[[nodiscard]] oa::U64 optimizerStep() const noexcept;
	// The same typed controller used by ordinary ML, DQN and SAC. The tutorial
	// viewer queues commands here rather than maintaining a second pause state.
	[[nodiscard]] oa::TrainingSession& control() noexcept;
	[[nodiscard]] const oa::TrainingSession& control() const noexcept;

private:
	struct Impl;
	explicit TutorialCartPolePpo(oa::UniquePtr<Impl> inImpl);
	oa::UniquePtr<Impl> impl_;
};
