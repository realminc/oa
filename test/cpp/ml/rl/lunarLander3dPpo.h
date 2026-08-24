#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

#include <array>

namespace oa {
class Engine;
}

class TestLunarLander3dPpoConfig {
public:
	oa::U32 environments_ = 7U;
	oa::U32 horizon_ = 16U;
	oa::U32 rollouts_ = 1U;
	oa::U32 updateEpochs_ = 1U;
	oa::I32 hiddenSize_ = 32;
	oa::U64 trainingSeed_ = 0x1a2b3c4dULL;
	oa::F32 learningRate_ = 2.5e-4F;
};

class TestLunarLander3dPpoMetrics {
public:
	oa::U32 rollout_ = 0U;
	oa::U32 updateEpoch_ = 0U;
	oa::F32 totalLoss_ = 0.0F;
	oa::F32 policyLoss_ = 0.0F;
	oa::F32 valueLoss_ = 0.0F;
	oa::F32 entropy_ = 0.0F;
};

// Deterministic flat-v0 imitation curriculum. The teacher seed is deliberately
// disjoint from the fixed evaluator seed below. This warm-start changes only
// policy parameters through a temporary optimizer; PPO's actor/critic optimizer
// remains at step zero so critic moment estimates start cleanly.
class TestLunarLander3dTeacherConfig {
public:
	oa::U32 episodes_ = 512U;
	oa::U32 epochs_ = 24U;
	oa::U32 batchSize_ = 2048U;
	oa::U32 maximumSamples_ = 524288U;
	oa::U64 environmentSeed_ = 0x544541434845525fULL;
	oa::U64 shuffleSeed_ = 0x494d49544154455fULL;
	oa::F32 learningRate_ = 1.0e-3F;
};

class TestLunarLander3dTeacherMetrics {
public:
	oa::U32 episodes_ = 0U;
	oa::U32 safeLandings_ = 0U;
	oa::U32 bodyImpacts_ = 0U;
	oa::U32 hardFootImpacts_ = 0U;
	oa::U32 outOfBounds_ = 0U;
	oa::U32 timeLimits_ = 0U;
	oa::U32 otherFailures_ = 0U;
	oa::U32 samples_ = 0U;
	oa::U32 optimizerSteps_ = 0U;
	std::array<oa::U64, 8U> actionCounts_{};
	oa::U64 datasetDigest_ = 0U;
	// Cross-entropy on the same deterministic dataset-prefix probe before and
	// after imitation training. These are not shuffled-minibatch endpoints.
	oa::F32 initialLoss_ = 0.0F;
	oa::F32 finalLoss_ = 0.0F;
};

// Fixed flat-terrain held-out protocol. The default horizon covers the frozen
// 1,200-step episode limit; completed lanes are never reset and remain terminal
// for the rest of the bounded evaluation.
class TestLunarLander3dFirstEpisodeEvaluationConfig {
public:
	oa::U32 environments_ = 512U;
	oa::U32 horizon_ = 1200U;
	oa::U32 submissionChunkSteps_ = 16U;
	oa::U64 environmentSeed_ = 0x50494c4f545f4556ULL;
};

class TestLunarLander3dFirstEpisodeEvaluation {
public:
	oa::String distribution_ = "flat";
	oa::U32 expectedEpisodes_ = 0U;
	oa::U32 completedEpisodes_ = 0U;
	oa::U64 recordedEnvironmentSteps_ = 0U;
	oa::U32 submissions_ = 0U;

	oa::U32 safeLandings_ = 0U;
	oa::U32 bodyImpacts_ = 0U;
	oa::U32 hardFootImpacts_ = 0U;
	oa::U32 outOfBounds_ = 0U;
	oa::U32 numericalFailures_ = 0U;
	oa::U32 timeLimits_ = 0U;
	oa::U32 externalStops_ = 0U;
	oa::U32 invalidActions_ = 0U;
	oa::U32 incompleteEpisodes_ = 0U;

	oa::F64 safeLandingRate_ = 0.0;
	oa::F64 wilsonLower95_ = 0.0;
	oa::F64 meanReturn_ = 0.0;
	oa::F64 minReturn_ = 0.0;
	oa::F64 maxReturn_ = 0.0;
	oa::F64 meanEpisodeSteps_ = 0.0;
	oa::F64 meanFuelRemaining_ = 0.0;
	oa::F64 meanTerminalLinearSpeed_ = 0.0;
	oa::F64 meanTerminalAngularSpeed_ = 0.0;
	oa::F64 meanMaximumFootImpulse_ = 0.0;
	std::array<oa::U64, 8U> actionCounts_{};
	oa::U64 actionTraceDigest_ = 0U;
	oa::U64 valueTraceDigest_ = 0U;
};

// Test-local PPO wiring and evidence harness. The environment owns command
// recording and exact-event submission; the generic trainer owns rollout,
// GAE, loss construction, and optimizer updates.
class TestLunarLander3dPpo {
public:
	[[nodiscard]] static oa::Result<oa::UniquePtr<TestLunarLander3dPpo>>
		create(
			oa::Engine& inEngine,
			const TestLunarLander3dPpoConfig& inConfig = {});

	~TestLunarLander3dPpo();
	TestLunarLander3dPpo(
		const TestLunarLander3dPpo&) = delete;
	TestLunarLander3dPpo& operator=(
		const TestLunarLander3dPpo&) = delete;

	[[nodiscard]] bool isDone() const noexcept;
	[[nodiscard]] const TestLunarLander3dPpoConfig& config()
		const noexcept;
	[[nodiscard]] const TestLunarLander3dPpoMetrics& metrics()
		const noexcept;
	[[nodiscard]] oa::I64 observationElements() const noexcept;
	[[nodiscard]] oa::I64 actionCount() const noexcept;
	[[nodiscard]] oa::F32 gaeGamma() const noexcept;
	[[nodiscard]] oa::U64 optimizerStep() const noexcept;

	[[nodiscard]] oa::Status advance();
	[[nodiscard]] oa::Status pretrainScriptedTeacher(
		const TestLunarLander3dTeacherConfig& inConfig = {});
	[[nodiscard]] const TestLunarLander3dTeacherMetrics& teacherMetrics()
		const noexcept;
	// Evaluates exactly episode zero of a fresh flat vector environment with a
	// deterministic greedy topK policy. It never resets completed lanes.
	[[nodiscard]] oa::Result<TestLunarLander3dFirstEpisodeEvaluation>
		evaluateFirstEpisodes(
			const TestLunarLander3dFirstEpisodeEvaluationConfig&
				inConfig = {});
	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(const oa::String& inPath);
	[[nodiscard]] oa::Status close();

private:
	class Impl;
	explicit TestLunarLander3dPpo(oa::UniquePtr<Impl> inImpl);

	// Immutable configuration and last-observed evidence remain queryable after
	// Close releases every engine-owned object in impl_.
	TestLunarLander3dPpoConfig config_;
	TestLunarLander3dPpoMetrics metrics_;
	TestLunarLander3dTeacherMetrics teacherMetrics_;
	oa::I64 observationElements_ = 0;
	oa::I64 actionCount_ = 0;
	oa::F32 gaeGamma_ = 0.0F;
	oa::U64 optimizerStep_ = 0U;
	oa::UniquePtr<Impl> impl_;
};
