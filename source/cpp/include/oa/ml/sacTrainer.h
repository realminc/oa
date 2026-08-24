#pragma once

#include <oa/ml/itTraining.h>
#include <oa/ml/module.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/replay.h>

namespace oa {

class Optimizer;
class Engine;

struct SacTrainerConfig {
	oa::U32 updates = 0;
	oa::U32 batchSize = 0;
	oa::U32 actionDimensions = 0;
	oa::U32 targetUpdateInterval = 1;
	oa::MatrixShape observationShape;
	oa::F32 actionMinimum = -1.0F;
	oa::F32 actionMaximum = 1.0F;
	oa::U64 seed = 0;
	SacLossConfig loss;
};

struct SacTrainerMetrics {
	oa::U64 update = 0;
	oa::F32 actorLoss = 0.0F;
	oa::F32 criticLoss = 0.0F;
};

/// Minimal fixed-alpha SAC trainer. Actor forward returns [B, 2*A] containing
/// mean then log-standard-deviation; each critic consumes [observation, action]
/// concatenated on the last axis and returns [B] or [B,1]. Its independent
/// actor and critic update loops are composed rather than hidden by inheritance.
class SacTrainer {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<SacTrainer>> create(
		oa::Engine& inEngine,
		oa::Module& inActor,
		oa::Module& inCritic1,
		oa::Module& inCritic2,
		oa::Module& inTargetCritic1,
		oa::Module& inTargetCritic2,
		oa::Optimizer& inActorOptimizer,
		oa::Optimizer& inCriticOptimizer,
		ReplayBuffer& inReplay,
		const SacTrainerConfig& inConfig
	);
	~SacTrainer();
	SacTrainer(const SacTrainer&) = delete;
	SacTrainer& operator=(const SacTrainer&) = delete;

	[[nodiscard]] oa::Status update();
	[[nodiscard]] oa::Status syncTargets();
	[[nodiscard]] bool isDone() const noexcept;
	[[nodiscard]] const SacTrainerMetrics& metrics() const noexcept;
	// SAC has two exact optimizer units. The critic loop is the primary update
	// controller; the actor loop remains separately observable.
	[[nodiscard]] oa::ItTraining& trainingLoop() noexcept;
	[[nodiscard]] const oa::ItTraining& trainingLoop() const noexcept;
	[[nodiscard]] oa::ItTraining& actorTrainingLoop() noexcept;
	[[nodiscard]] const oa::ItTraining& actorTrainingLoop() const noexcept;

private:
	struct Impl;
	explicit SacTrainer(oa::UniquePtr<Impl> inImpl);
	oa::UniquePtr<Impl> impl_;
};
} // namespace oa
