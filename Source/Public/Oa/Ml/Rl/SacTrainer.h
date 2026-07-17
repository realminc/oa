#pragma once

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Rl/FnLoss.h>
#include <Oa/Ml/Rl/Replay.h>

class OaOptimizer;

struct OaSacTrainerConfig {
	OaU32 Updates = 0;
	OaU32 BatchSize = 0;
	OaU32 ActionDimensions = 0;
	OaU32 TargetUpdateInterval = 1;
	OaMatrixShape ObservationShape;
	OaF32 ActionMinimum = -1.0F;
	OaF32 ActionMaximum = 1.0F;
	OaU64 Seed = 0;
	OaSacLossConfig Loss;
};

struct OaSacTrainerMetrics {
	OaU64 Update = 0;
	OaF32 ActorLoss = 0.0F;
	OaF32 CriticLoss = 0.0F;
};

// Minimal fixed-alpha SAC trainer. Actor Forward returns [B, 2*A] containing
// mean then log-standard-deviation; each critic consumes [observation, action]
// concatenated on the last axis and returns [B] or [B,1].
class OaSacTrainer {
public:
	[[nodiscard]] static OaResult<OaUniquePtr<OaSacTrainer>> Create(
		OaModule& InActor,
		OaModule& InCritic1,
		OaModule& InCritic2,
		OaModule& InTargetCritic1,
		OaModule& InTargetCritic2,
		OaOptimizer& InActorOptimizer,
		OaOptimizer& InCriticOptimizer,
		OaRlReplayBuffer& InReplay,
		const OaSacTrainerConfig& InConfig
	);
	~OaSacTrainer();
	OaSacTrainer(const OaSacTrainer&) = delete;
	OaSacTrainer& operator=(const OaSacTrainer&) = delete;

	[[nodiscard]] OaStatus Update();
	[[nodiscard]] OaStatus SyncTargets();
	[[nodiscard]] bool IsDone() const noexcept;
	[[nodiscard]] const OaSacTrainerMetrics& Metrics() const noexcept;
	// SAC has two exact optimizer units. The critic loop is the primary update
	// controller; the actor loop remains separately observable.
	[[nodiscard]] OaItTraining& TrainingLoop() noexcept;
	[[nodiscard]] const OaItTraining& TrainingLoop() const noexcept;
	[[nodiscard]] OaItTraining& ActorTrainingLoop() noexcept;
	[[nodiscard]] const OaItTraining& ActorTrainingLoop() const noexcept;

private:
	struct Impl;
	explicit OaSacTrainer(OaUniquePtr<Impl> InImpl);
	OaUniquePtr<Impl> Impl_;
};
