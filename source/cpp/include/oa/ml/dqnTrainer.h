#pragma once

#include <oa/ml/itTraining.h>
#include <oa/ml/module.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/replay.h>

namespace oa {

class Optimizer;
class Engine;

struct DqnTrainerConfig {
	oa::U32 updates = 0;
	oa::U32 batchSize = 0;
	oa::U32 targetUpdateInterval = 100;
	oa::MatrixShape observationShape;
	oa::U64 seed = 0;
	DqnLossConfig loss;
};

struct DqnTrainerMetrics {
	oa::U64 update = 0;
	oa::F32 loss = 0.0F;
};

/// Environment-neutral DQN update coordinator over caller-owned online and
/// target modules, optimizer, and replay storage. It composes `ItTraining` for
/// the exact optimizer lifecycle; it does not inherit a nominal trainer base.
class DqnTrainer {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<DqnTrainer>> create(
		oa::Engine& inEngine,
		oa::Module& inOnline,
		oa::Module& inTarget,
		oa::Optimizer& inOptimizer,
		ReplayBuffer& inReplay,
		const DqnTrainerConfig& inConfig
	);
	~DqnTrainer();
	DqnTrainer(const DqnTrainer&) = delete;
	DqnTrainer& operator=(const DqnTrainer&) = delete;

	[[nodiscard]] oa::Status update();
	[[nodiscard]] oa::Status syncTarget();
	[[nodiscard]] bool isDone() const noexcept;
	[[nodiscard]] const DqnTrainerMetrics& metrics() const noexcept;
	[[nodiscard]] oa::ItTraining& trainingLoop() noexcept;
	[[nodiscard]] const oa::ItTraining& trainingLoop() const noexcept;
	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(const oa::String& inPath);

private:
	struct Impl;
	explicit DqnTrainer(oa::UniquePtr<Impl> inImpl);
	oa::UniquePtr<Impl> impl_;
};
} // namespace oa
