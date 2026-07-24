#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

#include <array>

class OaEngine;

class OaTutorialLunarLander3dPpoConfig {
public:
	OaU32 Environments_ = 7U;
	OaU32 Horizon_ = 16U;
	OaU32 Rollouts_ = 1U;
	OaU32 UpdateEpochs_ = 1U;
	OaI32 HiddenSize_ = 32;
	OaU64 TrainingSeed_ = 0x1a2b3c4dULL;
	OaF32 LearningRate_ = 2.5e-4F;
};

class OaTutorialLunarLander3dPpoMetrics {
public:
	OaU32 Rollout_ = 0U;
	OaU32 UpdateEpoch_ = 0U;
	OaF32 TotalLoss_ = 0.0F;
	OaF32 PolicyLoss_ = 0.0F;
	OaF32 ValueLoss_ = 0.0F;
	OaF32 Entropy_ = 0.0F;
};

// Deterministic flat-v0 imitation curriculum. The teacher seed is deliberately
// disjoint from the fixed evaluator seed below. This warm-start changes only
// policy parameters through a temporary optimizer; PPO's actor/critic optimizer
// remains at step zero so critic moment estimates start cleanly.
class OaTutorialLunarLander3dTeacherConfig {
public:
	OaU32 Episodes_ = 512U;
	OaU32 Epochs_ = 24U;
	OaU32 BatchSize_ = 2048U;
	OaU32 MaximumSamples_ = 524288U;
	OaU64 EnvironmentSeed_ = 0x544541434845525fULL;
	OaU64 ShuffleSeed_ = 0x494d49544154455fULL;
	OaF32 LearningRate_ = 1.0e-3F;
};

class OaTutorialLunarLander3dTeacherMetrics {
public:
	OaU32 Episodes_ = 0U;
	OaU32 SafeLandings_ = 0U;
	OaU32 BodyImpacts_ = 0U;
	OaU32 HardFootImpacts_ = 0U;
	OaU32 OutOfBounds_ = 0U;
	OaU32 TimeLimits_ = 0U;
	OaU32 OtherFailures_ = 0U;
	OaU32 Samples_ = 0U;
	OaU32 OptimizerSteps_ = 0U;
	std::array<OaU64, 8U> ActionCounts_{};
	OaU64 DatasetDigest_ = 0U;
	// Cross-entropy on the same deterministic dataset-prefix probe before and
	// after imitation training. These are not shuffled-minibatch endpoints.
	OaF32 InitialLoss_ = 0.0F;
	OaF32 FinalLoss_ = 0.0F;
};

// Fixed flat-terrain held-out protocol. The default horizon covers the frozen
// 1,200-step episode limit; completed lanes are never reset and remain terminal
// for the rest of the bounded evaluation.
class OaTutorialLunarLander3dFirstEpisodeEvaluationConfig {
public:
	OaU32 Environments_ = 512U;
	OaU32 Horizon_ = 1200U;
	OaU32 SubmissionChunkSteps_ = 16U;
	OaU64 EnvironmentSeed_ = 0x50494c4f545f4556ULL;
};

class OaTutorialLunarLander3dFirstEpisodeEvaluation {
public:
	OaString Distribution_ = "flat";
	OaU32 ExpectedEpisodes_ = 0U;
	OaU32 CompletedEpisodes_ = 0U;
	OaU64 RecordedEnvironmentSteps_ = 0U;
	OaU32 Submissions_ = 0U;

	OaU32 SafeLandings_ = 0U;
	OaU32 BodyImpacts_ = 0U;
	OaU32 HardFootImpacts_ = 0U;
	OaU32 OutOfBounds_ = 0U;
	OaU32 NumericalFailures_ = 0U;
	OaU32 TimeLimits_ = 0U;
	OaU32 ExternalStops_ = 0U;
	OaU32 InvalidActions_ = 0U;
	OaU32 IncompleteEpisodes_ = 0U;

	OaF64 SafeLandingRate_ = 0.0;
	OaF64 WilsonLower95_ = 0.0;
	OaF64 MeanReturn_ = 0.0;
	OaF64 MinReturn_ = 0.0;
	OaF64 MaxReturn_ = 0.0;
	OaF64 MeanEpisodeSteps_ = 0.0;
	OaF64 MeanFuelRemaining_ = 0.0;
	OaF64 MeanTerminalLinearSpeed_ = 0.0;
	OaF64 MeanTerminalAngularSpeed_ = 0.0;
	OaF64 MeanMaximumFootImpulse_ = 0.0;
	std::array<OaU64, 8U> ActionCounts_{};
	OaU64 ActionTraceDigest_ = 0U;
	OaU64 ValueTraceDigest_ = 0U;
};

// Experimental tutorial-local PPO wiring proof. The environment owns command
// recording and exact-event submission; the generic trainer owns rollout,
// GAE, loss construction, and optimizer updates.
class OaTutorialLunarLander3dPpo {
public:
	[[nodiscard]] static OaResult<OaUniquePtr<OaTutorialLunarLander3dPpo>>
		Create(
			OaEngine& InEngine,
			const OaTutorialLunarLander3dPpoConfig& InConfig = {});

	~OaTutorialLunarLander3dPpo();
	OaTutorialLunarLander3dPpo(
		const OaTutorialLunarLander3dPpo&) = delete;
	OaTutorialLunarLander3dPpo& operator=(
		const OaTutorialLunarLander3dPpo&) = delete;

	[[nodiscard]] bool IsDone() const noexcept;
	[[nodiscard]] const OaTutorialLunarLander3dPpoConfig& Config()
		const noexcept;
	[[nodiscard]] const OaTutorialLunarLander3dPpoMetrics& Metrics()
		const noexcept;
	[[nodiscard]] OaI64 ObservationElements() const noexcept;
	[[nodiscard]] OaI64 ActionCount() const noexcept;
	[[nodiscard]] OaF32 GaeGamma() const noexcept;
	[[nodiscard]] OaU64 OptimizerStep() const noexcept;

	[[nodiscard]] OaStatus Advance();
	[[nodiscard]] OaStatus PretrainScriptedTeacher(
		const OaTutorialLunarLander3dTeacherConfig& InConfig = {});
	[[nodiscard]] const OaTutorialLunarLander3dTeacherMetrics& TeacherMetrics()
		const noexcept;
	// Evaluates exactly episode zero of a fresh flat vector environment with a
	// deterministic greedy TopK policy. It never resets completed lanes.
	[[nodiscard]] OaResult<OaTutorialLunarLander3dFirstEpisodeEvaluation>
		EvaluateFirstEpisodes(
			const OaTutorialLunarLander3dFirstEpisodeEvaluationConfig&
				InConfig = {});
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);
	[[nodiscard]] OaStatus Close();

private:
	class Impl;
	explicit OaTutorialLunarLander3dPpo(OaUniquePtr<Impl> InImpl);

	OaUniquePtr<Impl> Impl_;
};
