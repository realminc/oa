#include "../../oaTest.h"

#include "lunarLander3dPpo.h"

#include <oa/core/envFlag.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

oa::U32 lunarEvaluationU32(
	const char* inName,
	oa::U32 inDefault,
	oa::U32 inMaximum) {
	const oa::I64 value = oa::EnvFlag::getInt(
		inName, static_cast<oa::I64>(inDefault));
	return static_cast<oa::U32>(std::clamp<oa::I64>(
		value, 1, static_cast<oa::I64>(inMaximum)));
}

void lunarLogEvaluation(
	const char* inLabel,
	const TestLunarLander3dFirstEpisodeEvaluation& inEvaluation) {
	std::printf(
		"  %s: safe %u/%u (%.2f%%, Wilson95 lower %.2f%%) | "
		"return mean %.3f [%.3f, %.3f] | completed %u | "
		"body %u hard-foot %u OOB %u numerical %u timeout %u incomplete %u\n"
		"       steps mean %.2f | fuel %.2f | terminal speed %.3f | "
		"angular %.3f | foot impulse %.3f | submissions %u\n"
		"       actions [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu] | "
		"action %016llx value %016llx\n",
		inLabel,
		inEvaluation.safeLandings_, inEvaluation.expectedEpisodes_,
		inEvaluation.safeLandingRate_ * 100.0,
		inEvaluation.wilsonLower95_ * 100.0,
		inEvaluation.meanReturn_, inEvaluation.minReturn_,
		inEvaluation.maxReturn_, inEvaluation.completedEpisodes_,
		inEvaluation.bodyImpacts_, inEvaluation.hardFootImpacts_,
		inEvaluation.outOfBounds_, inEvaluation.numericalFailures_,
		inEvaluation.timeLimits_, inEvaluation.incompleteEpisodes_,
		inEvaluation.meanEpisodeSteps_, inEvaluation.meanFuelRemaining_,
		inEvaluation.meanTerminalLinearSpeed_,
		inEvaluation.meanTerminalAngularSpeed_,
		inEvaluation.meanMaximumFootImpulse_, inEvaluation.submissions_,
		static_cast<unsigned long long>(inEvaluation.actionCounts_[0]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[1]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[2]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[3]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[4]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[5]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[6]),
		static_cast<unsigned long long>(inEvaluation.actionCounts_[7]),
		static_cast<unsigned long long>(inEvaluation.actionTraceDigest_),
		static_cast<unsigned long long>(inEvaluation.valueTraceDigest_));
}

void lunarLogTeacher(
	const TestLunarLander3dTeacherMetrics& inMetrics) {
	std::printf(
		"  teacher: %u samples from %u completed episodes (%u safe, %u timeout, "
		"body %u hard-foot %u OOB %u other %u)\n"
		"           actions [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu] | "
		"steps %u | loss %.6f -> %.6f | dataset %016llx\n",
		inMetrics.samples_, inMetrics.episodes_, inMetrics.safeLandings_,
		inMetrics.timeLimits_, inMetrics.bodyImpacts_,
		inMetrics.hardFootImpacts_, inMetrics.outOfBounds_,
		inMetrics.otherFailures_,
		static_cast<unsigned long long>(inMetrics.actionCounts_[0]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[1]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[2]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[3]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[4]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[5]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[6]),
		static_cast<unsigned long long>(inMetrics.actionCounts_[7]),
		inMetrics.optimizerSteps_, inMetrics.initialLoss_,
		inMetrics.finalLoss_,
		static_cast<unsigned long long>(inMetrics.datasetDigest_));
}

void lunarExpectEvaluationEqual(
	const TestLunarLander3dFirstEpisodeEvaluation& inExpected,
	const TestLunarLander3dFirstEpisodeEvaluation& inActual) {
	EXPECT_EQ(inActual.distribution_, inExpected.distribution_);
	EXPECT_EQ(inActual.expectedEpisodes_, inExpected.expectedEpisodes_);
	EXPECT_EQ(inActual.completedEpisodes_, inExpected.completedEpisodes_);
	EXPECT_EQ(
		inActual.recordedEnvironmentSteps_,
		inExpected.recordedEnvironmentSteps_);
	EXPECT_EQ(inActual.submissions_, inExpected.submissions_);
	EXPECT_EQ(inActual.safeLandings_, inExpected.safeLandings_);
	EXPECT_EQ(inActual.bodyImpacts_, inExpected.bodyImpacts_);
	EXPECT_EQ(inActual.hardFootImpacts_, inExpected.hardFootImpacts_);
	EXPECT_EQ(inActual.outOfBounds_, inExpected.outOfBounds_);
	EXPECT_EQ(inActual.numericalFailures_, inExpected.numericalFailures_);
	EXPECT_EQ(inActual.timeLimits_, inExpected.timeLimits_);
	EXPECT_EQ(inActual.externalStops_, inExpected.externalStops_);
	EXPECT_EQ(inActual.invalidActions_, inExpected.invalidActions_);
	EXPECT_EQ(inActual.incompleteEpisodes_, inExpected.incompleteEpisodes_);
	EXPECT_DOUBLE_EQ(inActual.safeLandingRate_, inExpected.safeLandingRate_);
	EXPECT_DOUBLE_EQ(inActual.wilsonLower95_, inExpected.wilsonLower95_);
	EXPECT_DOUBLE_EQ(inActual.meanReturn_, inExpected.meanReturn_);
	EXPECT_DOUBLE_EQ(inActual.minReturn_, inExpected.minReturn_);
	EXPECT_DOUBLE_EQ(inActual.maxReturn_, inExpected.maxReturn_);
	EXPECT_DOUBLE_EQ(
		inActual.meanEpisodeSteps_, inExpected.meanEpisodeSteps_);
	EXPECT_DOUBLE_EQ(
		inActual.meanFuelRemaining_, inExpected.meanFuelRemaining_);
	EXPECT_DOUBLE_EQ(
		inActual.meanTerminalLinearSpeed_,
		inExpected.meanTerminalLinearSpeed_);
	EXPECT_DOUBLE_EQ(
		inActual.meanTerminalAngularSpeed_,
		inExpected.meanTerminalAngularSpeed_);
	EXPECT_DOUBLE_EQ(
		inActual.meanMaximumFootImpulse_,
		inExpected.meanMaximumFootImpulse_);
	EXPECT_EQ(inActual.actionCounts_, inExpected.actionCounts_);
	EXPECT_EQ(inActual.actionTraceDigest_, inExpected.actionTraceDigest_);
	EXPECT_EQ(inActual.valueTraceDigest_, inExpected.valueTraceDigest_);
}

} // namespace

TEST(TestLunarLander3dPpo, CompletesOneVectorizedGpuUpdate) {
	oa::Engine* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->isReady());
	const TestLunarLander3dPpoConfig config;
	auto created = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto session = oa::move(*created);

	OaLogInfo(oa::LogComponent::Ml,
		"lunarLander3dPpo(environments=%u, horizon=%u, rollouts=%u, epochs=%u, observationElements=%lld, actions=%lld, gaeGamma=%.3f)",
		config.environments_, config.horizon_, config.rollouts_,
		config.updateEpochs_,
		static_cast<long long>(session->observationElements()),
		static_cast<long long>(session->actionCount()),
		static_cast<double>(session->gaeGamma()));
	EXPECT_GT(session->observationElements(), 0);
	EXPECT_GT(session->actionCount(), 1);
	EXPECT_FLOAT_EQ(session->gaeGamma(), 0.99F);
	EXPECT_FALSE(session->isDone());

	const oa::Status advanced = session->advance();
	ASSERT_TRUE(advanced.isOk()) << advanced.toString();
	ASSERT_TRUE(session->isDone());
	const TestLunarLander3dPpoMetrics& metrics = session->metrics();
	EXPECT_TRUE(std::isfinite(metrics.totalLoss_));
	EXPECT_TRUE(std::isfinite(metrics.policyLoss_));
	EXPECT_TRUE(std::isfinite(metrics.valueLoss_));
	EXPECT_TRUE(std::isfinite(metrics.entropy_));
	EXPECT_EQ(metrics.rollout_, 1U);
	EXPECT_EQ(session->optimizerStep(), 1U);
	OaLogInfo(oa::LogComponent::Ml,
		"lunarLander3dPpoUpdate(totalLoss=%.6f, policyLoss=%.6f, valueLoss=%.6f, entropy=%.6f, optimizerStep=%llu)",
		static_cast<double>(metrics.totalLoss_),
		static_cast<double>(metrics.policyLoss_),
		static_cast<double>(metrics.valueLoss_),
		static_cast<double>(metrics.entropy_),
		static_cast<unsigned long long>(session->optimizerStep()));

	const TestLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig{
		.environments_ = 7U,
		.horizon_ = 64U,
		.submissionChunkSteps_ = 8U,
		.environmentSeed_ = 0x4c554e41525f4556ULL,
	};
	auto evaluationResult = session->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(evaluationResult.isOk())
		<< evaluationResult.getStatus().toString();
	const auto evaluation = *evaluationResult;
	EXPECT_EQ(evaluation.distribution_, "flat");
	EXPECT_EQ(evaluation.expectedEpisodes_, evaluationConfig.environments_);
	EXPECT_EQ(
		evaluation.completedEpisodes_ + evaluation.incompleteEpisodes_,
		evaluation.expectedEpisodes_);
	EXPECT_GT(evaluation.actionTraceDigest_, 0U);
	EXPECT_GT(evaluation.valueTraceDigest_, 0U);
	EXPECT_LE(evaluation.submissions_, 8U);
	lunarLogEvaluation("bounded evaluator", evaluation);

	const oa::Path checkpointPath = oa::Paths::temp()
		/ oa::Path("oa_lunar_lander_3d_ppo_smoke.oam");
	if (oa::Filesystem::exists(checkpointPath)) {
		ASSERT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	}
	ASSERT_TRUE(session->save(checkpointPath.string()).isOk());
	auto restoredResult = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(restoredResult.isOk())
		<< restoredResult.getStatus().toString();
	auto restored = oa::move(*restoredResult);
	const oa::Status loaded = restored->load(checkpointPath.string());
	ASSERT_TRUE(loaded.isOk()) << loaded.toString();
	EXPECT_EQ(restored->optimizerStep(), session->optimizerStep());
	auto restoredEvaluationResult =
		restored->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluationResult.isOk())
		<< restoredEvaluationResult.getStatus().toString();
	lunarExpectEvaluationEqual(evaluation, *restoredEvaluationResult);
	EXPECT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	const oa::I64 closedObservationElements = session->observationElements();
	const oa::I64 closedActionCount = session->actionCount();
	const oa::Status restoredClosed = restored->close();
	const oa::Status sessionClosed = session->close();
	EXPECT_TRUE(restoredClosed.isOk()) << restoredClosed.toString();
	EXPECT_TRUE(sessionClosed.isOk()) << sessionClosed.toString();
	EXPECT_EQ(session->config().environments_, config.environments_);
	EXPECT_EQ(session->metrics().rollout_, 1U);
	EXPECT_EQ(session->teacherMetrics().samples_, 0U);
	EXPECT_EQ(session->observationElements(), closedObservationElements);
	EXPECT_EQ(session->actionCount(), closedActionCount);
	EXPECT_FLOAT_EQ(session->gaeGamma(), 0.99F);
	EXPECT_EQ(session->optimizerStep(), 1U);
	EXPECT_TRUE(session->close().isOk());
}

TEST(TestLunarLander3dPpo, PretrainsPolicyWithoutAdvancingPpoOptimizer) {
	oa::Engine* suiteEngine = testEnginePtr();
	ASSERT_NE(suiteEngine, nullptr);
	ASSERT_TRUE(suiteEngine->isReady());
	oa::ExecutionSession* ambientContext = oa::ExecutionSession::getActivePtr();
	ASSERT_NE(ambientContext, nullptr);

	// Keep a different ready engine/context ambient while the PPO session uses
	// this non-default-context owner. Success therefore proves the session never allocates
	// or records through the ambient compatibility context by accident.
	oa::EngineConfig engineConfig = testEngineConfig(oa::Precision::FP32);
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	auto engineResult = oa::Engine::create(engineConfig);
	ASSERT_TRUE(engineResult.isOk())
		<< engineResult.getStatus().toString();
	auto engine = oa::move(*engineResult);
	ASSERT_TRUE(engine->isReady());
	ASSERT_NE(engine.get(), suiteEngine);
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);

	TestLunarLander3dPpoConfig config;
	config.hiddenSize_ = 64;
	auto created = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto session = oa::move(*created);
	ASSERT_EQ(session->optimizerStep(), 0U);
	const TestLunarLander3dTeacherConfig teacherConfig{
		.episodes_ = 16U,
		.epochs_ = 2U,
		.batchSize_ = 512U,
		.maximumSamples_ = 8192U,
	};
	const oa::Status pretrained = session->pretrainScriptedTeacher(teacherConfig);
	ASSERT_TRUE(pretrained.isOk()) << pretrained.toString();
	const auto& metrics = session->teacherMetrics();
	EXPECT_GT(metrics.episodes_, 0U);
	EXPECT_GT(metrics.samples_, 0U);
	EXPECT_GT(metrics.optimizerSteps_, 0U);
	EXPECT_GT(metrics.datasetDigest_, 0U);
	EXPECT_TRUE(std::isfinite(metrics.initialLoss_));
	EXPECT_TRUE(std::isfinite(metrics.finalLoss_));
	EXPECT_LT(metrics.finalLoss_, metrics.initialLoss_);
	EXPECT_EQ(session->optimizerStep(), 0U);
	lunarLogTeacher(metrics);
	EXPECT_TRUE(session->close().isOk());
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);
	const oa::Status engineClosed = engine->close();
	EXPECT_TRUE(engineClosed.isOk()) << engineClosed.toString();
	EXPECT_EQ(testEnginePtr(), suiteEngine);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), ambientContext);
}

// Full deterministic flat-v0 behavior-cloning report. This is separate from
// the raw-PPO calibration below so a sparse-reward failure cannot be hidden by
// the teacher warm-start. The evaluation seed is disjoint from all teacher data.
TEST(TestLunarLander3dPpo, DISABLED_ReportsFlatTeacherLearningEvidence) {
	if (not oa::EnvFlag::isSet("OA_LUNAR_RUN_TEACHER_GATE")) {
		GTEST_SKIP() << "set OA_LUNAR_RUN_TEACHER_GATE=1 to run imitation training";
	}
	oa::Engine* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->isReady());

	TestLunarLander3dPpoConfig config;
	config.hiddenSize_ = static_cast<oa::I32>(lunarEvaluationU32(
		"OA_LUNAR_TRAIN_HIDDEN", 128U, 4096U));
	TestLunarLander3dTeacherConfig teacherConfig;
	teacherConfig.episodes_ = lunarEvaluationU32(
		"OA_LUNAR_TEACHER_EPISODES", teacherConfig.episodes_, 65536U);
	teacherConfig.epochs_ = lunarEvaluationU32(
		"OA_LUNAR_TEACHER_EPOCHS", teacherConfig.epochs_, 256U);
	teacherConfig.batchSize_ = lunarEvaluationU32(
		"OA_LUNAR_TEACHER_BATCH", teacherConfig.batchSize_, 65536U);
	teacherConfig.maximumSamples_ = lunarEvaluationU32(
		"OA_LUNAR_TEACHER_MAX_SAMPLES",
		teacherConfig.maximumSamples_, 16777216U);
	TestLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig;
	evaluationConfig.environments_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_ENVIRONMENTS", 512U, 65536U);
	evaluationConfig.horizon_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_HORIZON", 1200U, 1200U);
	evaluationConfig.submissionChunkSteps_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_CHUNK", 16U, 256U);

	std::printf(
		"\nOA Lunar Lander 3D flat teacher-learning report\n"
		"  teacher: %u episodes, max %u samples, %u epochs, batch %u, "
		"seed %016llx, shuffle %016llx\n"
		"  held out: %u first episodes, horizon %u, chunk %u, seed %016llx\n",
		teacherConfig.episodes_, teacherConfig.maximumSamples_,
		teacherConfig.epochs_, teacherConfig.batchSize_,
		static_cast<unsigned long long>(teacherConfig.environmentSeed_),
		static_cast<unsigned long long>(teacherConfig.shuffleSeed_),
		evaluationConfig.environments_, evaluationConfig.horizon_,
		evaluationConfig.submissionChunkSteps_,
		static_cast<unsigned long long>(evaluationConfig.environmentSeed_));

	auto created = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto session = oa::move(*created);
	auto beforeResult = session->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(beforeResult.isOk()) << beforeResult.getStatus().toString();
	const auto before = *beforeResult;
	lunarLogEvaluation("untrained", before);

	const oa::Status pretrained = session->pretrainScriptedTeacher(teacherConfig);
	ASSERT_TRUE(pretrained.isOk()) << pretrained.toString();
	lunarLogTeacher(session->teacherMetrics());
	EXPECT_EQ(session->optimizerStep(), 0U);
	auto learnedResult = session->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(learnedResult.isOk()) << learnedResult.getStatus().toString();
	const auto learned = *learnedResult;
	lunarLogEvaluation("learned", learned);
	EXPECT_EQ(learned.completedEpisodes_, learned.expectedEpisodes_);
	EXPECT_EQ(learned.incompleteEpisodes_, 0U);
	EXPECT_EQ(learned.numericalFailures_, 0U);
	EXPECT_EQ(learned.invalidActions_, 0U);
	EXPECT_GT(learned.meanReturn_, before.meanReturn_);
	// This disabled test is the evidence gate, not a report-only smoke. Once a
	// caller opts into its cost, a non-learning run must fail without requiring
	// a second environment toggle.
	EXPECT_GE(learned.safeLandingRate_, 0.80);
	EXPECT_GE(learned.wilsonLower95_, 0.75);

	const oa::Path checkpointPath = oa::Paths::temp()
		/ oa::Path("oa_lunar_lander_3d_flat_teacher.oam");
	if (oa::Filesystem::exists(checkpointPath)) {
		ASSERT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	}
	ASSERT_TRUE(session->save(checkpointPath.string()).isOk());
	auto restoredResult = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(restoredResult.isOk())
		<< restoredResult.getStatus().toString();
	auto restored = oa::move(*restoredResult);
	ASSERT_TRUE(restored->load(checkpointPath.string()).isOk());
	EXPECT_EQ(restored->optimizerStep(), 0U);
	auto restoredEvaluation = restored->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluation.isOk())
		<< restoredEvaluation.getStatus().toString();
	lunarExpectEvaluationEqual(learned, *restoredEvaluation);
	EXPECT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	EXPECT_TRUE(restored->close().isOk());
	EXPECT_TRUE(session->close().isOk());
}

// This opt-in report is intentionally excluded from ordinary CTest. It is the
// bounded command used to tune and then freeze a flat-terrain learning
// protocol; the procedural v0 final-test manifest remains a separate L5 gate.
TEST(TestLunarLander3dPpo, DISABLED_ReportsFlatLearningEvidence) {
	if (not oa::EnvFlag::isSet("OA_LUNAR_RUN_LEARNING_GATE")) {
		GTEST_SKIP() << "set OA_LUNAR_RUN_LEARNING_GATE=1 to run training";
	}
	oa::Engine* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->isReady());

	TestLunarLander3dPpoConfig config;
	config.environments_ = lunarEvaluationU32(
		"OA_LUNAR_TRAIN_ENVIRONMENTS", 64U, 65536U);
	config.horizon_ = lunarEvaluationU32(
		"OA_LUNAR_TRAIN_HORIZON", 128U, 1200U);
	config.rollouts_ = lunarEvaluationU32(
		"OA_LUNAR_TRAIN_ROLLOUTS", 40U, 100000U);
	config.updateEpochs_ = lunarEvaluationU32(
		"OA_LUNAR_TRAIN_EPOCHS", 4U, 64U);
	config.hiddenSize_ = static_cast<oa::I32>(lunarEvaluationU32(
		"OA_LUNAR_TRAIN_HIDDEN", 64U, 4096U));

	TestLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig;
	evaluationConfig.environments_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_ENVIRONMENTS", 512U, 65536U);
	evaluationConfig.horizon_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_HORIZON", 1200U, 1200U);
	evaluationConfig.submissionChunkSteps_ = lunarEvaluationU32(
		"OA_LUNAR_EVAL_CHUNK", 16U, 256U);

	std::printf(
		"\nOA Lunar Lander 3D flat PPO report\n"
		"  train: %u environments x %u steps x %u rollouts, %u epochs, hidden %d\n"
		"  held out: %u first episodes, horizon %u, chunk %u, seed %016llx\n",
		config.environments_, config.horizon_, config.rollouts_,
		config.updateEpochs_, config.hiddenSize_,
		evaluationConfig.environments_, evaluationConfig.horizon_,
		evaluationConfig.submissionChunkSteps_,
		static_cast<unsigned long long>(evaluationConfig.environmentSeed_));

	auto created = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto session = oa::move(*created);
	auto beforeResult = session->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(beforeResult.isOk()) << beforeResult.getStatus().toString();
	const auto before = *beforeResult;
	lunarLogEvaluation("untrained", before);

	oa::U32 lastReportedRollout = 0U;
	while (not session->isDone()) {
		const oa::Status advanced = session->advance();
		ASSERT_TRUE(advanced.isOk()) << advanced.toString();
		const auto& metrics = session->metrics();
		if (metrics.rollout_ != lastReportedRollout
			and (metrics.rollout_ % 10U == 0U or session->isDone())) {
			lastReportedRollout = metrics.rollout_;
			std::printf(
				"  rollout %u/%u | loss %.6f policy %.6f value %.6f entropy %.6f\n",
				metrics.rollout_, config.rollouts_, metrics.totalLoss_,
				metrics.policyLoss_, metrics.valueLoss_, metrics.entropy_);
		}
	}

	auto afterResult = session->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(afterResult.isOk()) << afterResult.getStatus().toString();
	const auto after = *afterResult;
	lunarLogEvaluation("trained", after);
	EXPECT_EQ(after.completedEpisodes_, after.expectedEpisodes_);
	EXPECT_EQ(after.incompleteEpisodes_, 0U);

	const oa::Path checkpointPath = oa::Paths::temp()
		/ oa::Path("oa_lunar_lander_3d_flat_learning.oam");
	if (oa::Filesystem::exists(checkpointPath)) {
		ASSERT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	}
	ASSERT_TRUE(session->save(checkpointPath.string()).isOk());
	auto restoredResult = TestLunarLander3dPpo::create(*engine, config);
	ASSERT_TRUE(restoredResult.isOk())
		<< restoredResult.getStatus().toString();
	auto restored = oa::move(*restoredResult);
	ASSERT_TRUE(restored->load(checkpointPath.string()).isOk());
	auto restoredEvaluation = restored->evaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluation.isOk())
		<< restoredEvaluation.getStatus().toString();
	lunarExpectEvaluationEqual(after, *restoredEvaluation);
	EXPECT_EQ(restored->optimizerStep(), session->optimizerStep());

	if (oa::EnvFlag::isSet("OA_LUNAR_REQUIRE_FLAT_GATE")) {
		EXPECT_GE(after.safeLandingRate_, 0.80);
		EXPECT_GE(after.wilsonLower95_, 0.75);
		EXPECT_GT(after.meanReturn_, before.meanReturn_);
	}
	EXPECT_TRUE(oa::Filesystem::removeFile(checkpointPath).isOk());
	EXPECT_TRUE(restored->close().isOk());
	EXPECT_TRUE(session->close().isOk());
}
