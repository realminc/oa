#pragma once

#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Rl/FnLoss.h>
#include <Oa/Ml/Rl/Replay.h>

class OaOptimizer;

struct OaDqnTrainerConfig {
	OaU32 Updates = 0;
	OaU32 BatchSize = 0;
	OaU32 TargetUpdateInterval = 100;
	OaMatrixShape ObservationShape;
	OaU64 Seed = 0;
	OaDqnLossConfig Loss;
};

struct OaDqnTrainerMetrics {
	OaU64 Update = 0;
	OaF32 Loss = 0.0F;
};

class OaDqnTrainer {
public:
	[[nodiscard]] static OaResult<OaUniquePtr<OaDqnTrainer>> Create(
		OaModule& InOnline,
		OaModule& InTarget,
		OaOptimizer& InOptimizer,
		OaRlReplayBuffer& InReplay,
		const OaDqnTrainerConfig& InConfig);
	~OaDqnTrainer();
	OaDqnTrainer(const OaDqnTrainer&) = delete;
	OaDqnTrainer& operator=(const OaDqnTrainer&) = delete;

	[[nodiscard]] OaStatus Update();
	[[nodiscard]] OaStatus SyncTarget();
	[[nodiscard]] bool IsDone() const noexcept;
	[[nodiscard]] const OaDqnTrainerMetrics& Metrics() const noexcept;
	[[nodiscard]] OaItTraining& TrainingLoop() noexcept;
	[[nodiscard]] const OaItTraining& TrainingLoop() const noexcept;
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);

private:
	struct Impl;
	explicit OaDqnTrainer(OaUniquePtr<Impl> InImpl);
	OaUniquePtr<Impl> Impl_;
};
