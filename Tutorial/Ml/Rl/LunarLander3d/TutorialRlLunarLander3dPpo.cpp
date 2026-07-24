#include "../../../../Test/OaTest.h"

#include "LunarLander3dPpo.h"

#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

OaU32 OaLunarEvaluationU32(
	const char* InName,
	OaU32 InDefault,
	OaU32 InMaximum) {
	const OaI64 value = OaEnvFlag::GetInt(
		InName, static_cast<OaI64>(InDefault));
	return static_cast<OaU32>(std::clamp<OaI64>(
		value, 1, static_cast<OaI64>(InMaximum)));
}

void OaLunarLogEvaluation(
	const char* InLabel,
	const OaTutorialLunarLander3dFirstEpisodeEvaluation& InEvaluation) {
	std::printf(
		"  %s: safe %u/%u (%.2f%%, Wilson95 lower %.2f%%) | "
		"return mean %.3f [%.3f, %.3f] | completed %u | "
		"body %u hard-foot %u OOB %u numerical %u timeout %u incomplete %u\n"
		"       steps mean %.2f | fuel %.2f | terminal speed %.3f | "
		"angular %.3f | foot impulse %.3f | submissions %u\n"
		"       actions [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu] | "
		"action %016llx value %016llx\n",
		InLabel,
		InEvaluation.SafeLandings_, InEvaluation.ExpectedEpisodes_,
		InEvaluation.SafeLandingRate_ * 100.0,
		InEvaluation.WilsonLower95_ * 100.0,
		InEvaluation.MeanReturn_, InEvaluation.MinReturn_,
		InEvaluation.MaxReturn_, InEvaluation.CompletedEpisodes_,
		InEvaluation.BodyImpacts_, InEvaluation.HardFootImpacts_,
		InEvaluation.OutOfBounds_, InEvaluation.NumericalFailures_,
		InEvaluation.TimeLimits_, InEvaluation.IncompleteEpisodes_,
		InEvaluation.MeanEpisodeSteps_, InEvaluation.MeanFuelRemaining_,
		InEvaluation.MeanTerminalLinearSpeed_,
		InEvaluation.MeanTerminalAngularSpeed_,
		InEvaluation.MeanMaximumFootImpulse_, InEvaluation.Submissions_,
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[0]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[1]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[2]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[3]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[4]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[5]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[6]),
		static_cast<unsigned long long>(InEvaluation.ActionCounts_[7]),
		static_cast<unsigned long long>(InEvaluation.ActionTraceDigest_),
		static_cast<unsigned long long>(InEvaluation.ValueTraceDigest_));
}

void OaLunarLogTeacher(
	const OaTutorialLunarLander3dTeacherMetrics& InMetrics) {
	std::printf(
		"  teacher: %u samples from %u completed episodes (%u safe, %u timeout, "
		"body %u hard-foot %u OOB %u other %u)\n"
		"           actions [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu] | "
		"steps %u | loss %.6f -> %.6f | dataset %016llx\n",
		InMetrics.Samples_, InMetrics.Episodes_, InMetrics.SafeLandings_,
		InMetrics.TimeLimits_, InMetrics.BodyImpacts_,
		InMetrics.HardFootImpacts_, InMetrics.OutOfBounds_,
		InMetrics.OtherFailures_,
		static_cast<unsigned long long>(InMetrics.ActionCounts_[0]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[1]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[2]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[3]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[4]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[5]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[6]),
		static_cast<unsigned long long>(InMetrics.ActionCounts_[7]),
		InMetrics.OptimizerSteps_, InMetrics.InitialLoss_,
		InMetrics.FinalLoss_,
		static_cast<unsigned long long>(InMetrics.DatasetDigest_));
}

void OaLunarExpectEvaluationEqual(
	const OaTutorialLunarLander3dFirstEpisodeEvaluation& InExpected,
	const OaTutorialLunarLander3dFirstEpisodeEvaluation& InActual) {
	EXPECT_EQ(InActual.Distribution_, InExpected.Distribution_);
	EXPECT_EQ(InActual.ExpectedEpisodes_, InExpected.ExpectedEpisodes_);
	EXPECT_EQ(InActual.CompletedEpisodes_, InExpected.CompletedEpisodes_);
	EXPECT_EQ(
		InActual.RecordedEnvironmentSteps_,
		InExpected.RecordedEnvironmentSteps_);
	EXPECT_EQ(InActual.Submissions_, InExpected.Submissions_);
	EXPECT_EQ(InActual.SafeLandings_, InExpected.SafeLandings_);
	EXPECT_EQ(InActual.BodyImpacts_, InExpected.BodyImpacts_);
	EXPECT_EQ(InActual.HardFootImpacts_, InExpected.HardFootImpacts_);
	EXPECT_EQ(InActual.OutOfBounds_, InExpected.OutOfBounds_);
	EXPECT_EQ(InActual.NumericalFailures_, InExpected.NumericalFailures_);
	EXPECT_EQ(InActual.TimeLimits_, InExpected.TimeLimits_);
	EXPECT_EQ(InActual.ExternalStops_, InExpected.ExternalStops_);
	EXPECT_EQ(InActual.InvalidActions_, InExpected.InvalidActions_);
	EXPECT_EQ(InActual.IncompleteEpisodes_, InExpected.IncompleteEpisodes_);
	EXPECT_DOUBLE_EQ(InActual.SafeLandingRate_, InExpected.SafeLandingRate_);
	EXPECT_DOUBLE_EQ(InActual.WilsonLower95_, InExpected.WilsonLower95_);
	EXPECT_DOUBLE_EQ(InActual.MeanReturn_, InExpected.MeanReturn_);
	EXPECT_DOUBLE_EQ(InActual.MinReturn_, InExpected.MinReturn_);
	EXPECT_DOUBLE_EQ(InActual.MaxReturn_, InExpected.MaxReturn_);
	EXPECT_DOUBLE_EQ(
		InActual.MeanEpisodeSteps_, InExpected.MeanEpisodeSteps_);
	EXPECT_DOUBLE_EQ(
		InActual.MeanFuelRemaining_, InExpected.MeanFuelRemaining_);
	EXPECT_DOUBLE_EQ(
		InActual.MeanTerminalLinearSpeed_,
		InExpected.MeanTerminalLinearSpeed_);
	EXPECT_DOUBLE_EQ(
		InActual.MeanTerminalAngularSpeed_,
		InExpected.MeanTerminalAngularSpeed_);
	EXPECT_DOUBLE_EQ(
		InActual.MeanMaximumFootImpulse_,
		InExpected.MeanMaximumFootImpulse_);
	EXPECT_EQ(InActual.ActionCounts_, InExpected.ActionCounts_);
	EXPECT_EQ(InActual.ActionTraceDigest_, InExpected.ActionTraceDigest_);
	EXPECT_EQ(InActual.ValueTraceDigest_, InExpected.ValueTraceDigest_);
}

} // namespace

TEST(TutorialRlLunarLander3dPpo, CompletesOneVectorizedGpuUpdate) {
	OaEngine* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->IsReady());
	const OaTutorialLunarLander3dPpoConfig config;
	auto created = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto session = OaStdMove(*created);

	OA_LOG_INFO(OaLogComponent::ML,
		"LunarLander3dPpo(Environments=%u, Horizon=%u, Rollouts=%u, Epochs=%u, ObservationElements=%lld, Actions=%lld, GaeGamma=%.3f)",
		config.Environments_, config.Horizon_, config.Rollouts_,
		config.UpdateEpochs_,
		static_cast<long long>(session->ObservationElements()),
		static_cast<long long>(session->ActionCount()),
		static_cast<double>(session->GaeGamma()));
	EXPECT_GT(session->ObservationElements(), 0);
	EXPECT_GT(session->ActionCount(), 1);
	EXPECT_FLOAT_EQ(session->GaeGamma(), 0.99F);
	EXPECT_FALSE(session->IsDone());

	const OaStatus advanced = session->Advance();
	ASSERT_TRUE(advanced.IsOk()) << advanced.ToString();
	ASSERT_TRUE(session->IsDone());
	const OaTutorialLunarLander3dPpoMetrics& metrics = session->Metrics();
	EXPECT_TRUE(std::isfinite(metrics.TotalLoss_));
	EXPECT_TRUE(std::isfinite(metrics.PolicyLoss_));
	EXPECT_TRUE(std::isfinite(metrics.ValueLoss_));
	EXPECT_TRUE(std::isfinite(metrics.Entropy_));
	EXPECT_EQ(metrics.Rollout_, 1U);
	EXPECT_EQ(session->OptimizerStep(), 1U);
	OA_LOG_INFO(OaLogComponent::ML,
		"LunarLander3dPpoUpdate(TotalLoss=%.6f, PolicyLoss=%.6f, ValueLoss=%.6f, Entropy=%.6f, OptimizerStep=%llu)",
		static_cast<double>(metrics.TotalLoss_),
		static_cast<double>(metrics.PolicyLoss_),
		static_cast<double>(metrics.ValueLoss_),
		static_cast<double>(metrics.Entropy_),
		static_cast<unsigned long long>(session->OptimizerStep()));

	const OaTutorialLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig{
		.Environments_ = 7U,
		.Horizon_ = 64U,
		.SubmissionChunkSteps_ = 8U,
		.EnvironmentSeed_ = 0x4c554e41525f4556ULL,
	};
	auto evaluationResult = session->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(evaluationResult.IsOk())
		<< evaluationResult.GetStatus().ToString();
	const auto evaluation = *evaluationResult;
	EXPECT_EQ(evaluation.Distribution_, "flat");
	EXPECT_EQ(evaluation.ExpectedEpisodes_, evaluationConfig.Environments_);
	EXPECT_EQ(
		evaluation.CompletedEpisodes_ + evaluation.IncompleteEpisodes_,
		evaluation.ExpectedEpisodes_);
	EXPECT_GT(evaluation.ActionTraceDigest_, 0U);
	EXPECT_GT(evaluation.ValueTraceDigest_, 0U);
	EXPECT_LE(evaluation.Submissions_, 8U);
	OaLunarLogEvaluation("bounded evaluator", evaluation);

	const OaPath checkpointPath = OaPaths::Temp()
		/ OaPath("oa_lunar_lander_3d_ppo_smoke.oam");
	if (OaFilesystem::Exists(checkpointPath)) {
		ASSERT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	}
	ASSERT_TRUE(session->Save(checkpointPath.String()).IsOk());
	auto restoredResult = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(restoredResult.IsOk())
		<< restoredResult.GetStatus().ToString();
	auto restored = OaStdMove(*restoredResult);
	const OaStatus loaded = restored->Load(checkpointPath.String());
	ASSERT_TRUE(loaded.IsOk()) << loaded.ToString();
	EXPECT_EQ(restored->OptimizerStep(), session->OptimizerStep());
	auto restoredEvaluationResult =
		restored->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluationResult.IsOk())
		<< restoredEvaluationResult.GetStatus().ToString();
	OaLunarExpectEvaluationEqual(evaluation, *restoredEvaluationResult);
	EXPECT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	const OaStatus restoredClosed = restored->Close();
	const OaStatus sessionClosed = session->Close();
	EXPECT_TRUE(restoredClosed.IsOk()) << restoredClosed.ToString();
	EXPECT_TRUE(sessionClosed.IsOk()) << sessionClosed.ToString();
}

TEST(TutorialRlLunarLander3dPpo, PretrainsPolicyWithoutAdvancingPpoOptimizer) {
	OaEngine* globalEngine = OaEngine::GetGlobal();
	ASSERT_NE(globalEngine, nullptr);
	ASSERT_TRUE(globalEngine->IsReady());
	OaContext* ambientContext = OaContext::GetDefaultPtr();
	ASSERT_NE(ambientContext, nullptr);

	// Keep a different ready engine/context ambient while the PPO session uses
	// this non-global owner. Success therefore proves the session never allocates
	// or records through the ambient compatibility context by accident.
	OaEngineConfig engineConfig = OaTestEngineConfig(OaPrecision::FP32);
	engineConfig.RegisterAsGlobal = false;
	engineConfig.PreloadEmbeddedPipelines = false;
	auto engineResult = OaEngine::Create(engineConfig);
	ASSERT_TRUE(engineResult.IsOk())
		<< engineResult.GetStatus().ToString();
	auto engine = OaStdMove(*engineResult);
	ASSERT_TRUE(engine->IsReady());
	ASSERT_NE(engine.get(), globalEngine);
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);

	OaTutorialLunarLander3dPpoConfig config;
	config.HiddenSize_ = 64;
	auto created = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto session = OaStdMove(*created);
	ASSERT_EQ(session->OptimizerStep(), 0U);
	const OaTutorialLunarLander3dTeacherConfig teacherConfig{
		.Episodes_ = 16U,
		.Epochs_ = 2U,
		.BatchSize_ = 512U,
		.MaximumSamples_ = 8192U,
	};
	const OaStatus pretrained = session->PretrainScriptedTeacher(teacherConfig);
	ASSERT_TRUE(pretrained.IsOk()) << pretrained.ToString();
	const auto& metrics = session->TeacherMetrics();
	EXPECT_GT(metrics.Episodes_, 0U);
	EXPECT_GT(metrics.Samples_, 0U);
	EXPECT_GT(metrics.OptimizerSteps_, 0U);
	EXPECT_GT(metrics.DatasetDigest_, 0U);
	EXPECT_TRUE(std::isfinite(metrics.InitialLoss_));
	EXPECT_TRUE(std::isfinite(metrics.FinalLoss_));
	EXPECT_LT(metrics.FinalLoss_, metrics.InitialLoss_);
	EXPECT_EQ(session->OptimizerStep(), 0U);
	OaLunarLogTeacher(metrics);
	EXPECT_TRUE(session->Close().IsOk());
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);
	const OaStatus engineClosed = engine->Close();
	EXPECT_TRUE(engineClosed.IsOk()) << engineClosed.ToString();
	EXPECT_EQ(OaEngine::GetGlobal(), globalEngine);
	EXPECT_EQ(OaContext::GetDefaultPtr(), ambientContext);
}

// Full deterministic flat-v0 behavior-cloning report. This is separate from
// the raw-PPO calibration below so a sparse-reward failure cannot be hidden by
// the teacher warm-start. The evaluation seed is disjoint from all teacher data.
TEST(TutorialRlLunarLander3dPpo, DISABLED_ReportsFlatTeacherLearningEvidence) {
	if (not OaEnvFlag::IsSet("OA_LUNAR_RUN_TEACHER_GATE")) {
		GTEST_SKIP() << "set OA_LUNAR_RUN_TEACHER_GATE=1 to run imitation training";
	}
	OaEngine* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->IsReady());

	OaTutorialLunarLander3dPpoConfig config;
	config.HiddenSize_ = static_cast<OaI32>(OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_HIDDEN", 128U, 4096U));
	OaTutorialLunarLander3dTeacherConfig teacherConfig;
	teacherConfig.Episodes_ = OaLunarEvaluationU32(
		"OA_LUNAR_TEACHER_EPISODES", teacherConfig.Episodes_, 65536U);
	teacherConfig.Epochs_ = OaLunarEvaluationU32(
		"OA_LUNAR_TEACHER_EPOCHS", teacherConfig.Epochs_, 256U);
	teacherConfig.BatchSize_ = OaLunarEvaluationU32(
		"OA_LUNAR_TEACHER_BATCH", teacherConfig.BatchSize_, 65536U);
	teacherConfig.MaximumSamples_ = OaLunarEvaluationU32(
		"OA_LUNAR_TEACHER_MAX_SAMPLES",
		teacherConfig.MaximumSamples_, 16777216U);
	OaTutorialLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig;
	evaluationConfig.Environments_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_ENVIRONMENTS", 512U, 65536U);
	evaluationConfig.Horizon_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_HORIZON", 1200U, 1200U);
	evaluationConfig.SubmissionChunkSteps_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_CHUNK", 16U, 256U);

	std::printf(
		"\nOA Lunar Lander 3D flat teacher-learning report\n"
		"  teacher: %u episodes, max %u samples, %u epochs, batch %u, "
		"seed %016llx, shuffle %016llx\n"
		"  held out: %u first episodes, horizon %u, chunk %u, seed %016llx\n",
		teacherConfig.Episodes_, teacherConfig.MaximumSamples_,
		teacherConfig.Epochs_, teacherConfig.BatchSize_,
		static_cast<unsigned long long>(teacherConfig.EnvironmentSeed_),
		static_cast<unsigned long long>(teacherConfig.ShuffleSeed_),
		evaluationConfig.Environments_, evaluationConfig.Horizon_,
		evaluationConfig.SubmissionChunkSteps_,
		static_cast<unsigned long long>(evaluationConfig.EnvironmentSeed_));

	auto created = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto session = OaStdMove(*created);
	auto beforeResult = session->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(beforeResult.IsOk()) << beforeResult.GetStatus().ToString();
	const auto before = *beforeResult;
	OaLunarLogEvaluation("untrained", before);

	const OaStatus pretrained = session->PretrainScriptedTeacher(teacherConfig);
	ASSERT_TRUE(pretrained.IsOk()) << pretrained.ToString();
	OaLunarLogTeacher(session->TeacherMetrics());
	EXPECT_EQ(session->OptimizerStep(), 0U);
	auto learnedResult = session->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(learnedResult.IsOk()) << learnedResult.GetStatus().ToString();
	const auto learned = *learnedResult;
	OaLunarLogEvaluation("learned", learned);
	EXPECT_EQ(learned.CompletedEpisodes_, learned.ExpectedEpisodes_);
	EXPECT_EQ(learned.IncompleteEpisodes_, 0U);
	EXPECT_EQ(learned.NumericalFailures_, 0U);
	EXPECT_EQ(learned.InvalidActions_, 0U);
	EXPECT_GT(learned.MeanReturn_, before.MeanReturn_);
	// This disabled test is the evidence gate, not a report-only smoke. Once a
	// caller opts into its cost, a non-learning run must fail without requiring
	// a second environment toggle.
	EXPECT_GE(learned.SafeLandingRate_, 0.80);
	EXPECT_GE(learned.WilsonLower95_, 0.75);

	const OaPath checkpointPath = OaPaths::Temp()
		/ OaPath("oa_lunar_lander_3d_flat_teacher.oam");
	if (OaFilesystem::Exists(checkpointPath)) {
		ASSERT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	}
	ASSERT_TRUE(session->Save(checkpointPath.String()).IsOk());
	auto restoredResult = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(restoredResult.IsOk())
		<< restoredResult.GetStatus().ToString();
	auto restored = OaStdMove(*restoredResult);
	ASSERT_TRUE(restored->Load(checkpointPath.String()).IsOk());
	EXPECT_EQ(restored->OptimizerStep(), 0U);
	auto restoredEvaluation = restored->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluation.IsOk())
		<< restoredEvaluation.GetStatus().ToString();
	OaLunarExpectEvaluationEqual(learned, *restoredEvaluation);
	EXPECT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	EXPECT_TRUE(restored->Close().IsOk());
	EXPECT_TRUE(session->Close().IsOk());
}

// This opt-in report is intentionally excluded from ordinary CTest. It is the
// bounded command used to tune and then freeze a flat-terrain learning
// protocol; the procedural v0 final-test manifest remains a separate L5 gate.
TEST(TutorialRlLunarLander3dPpo, DISABLED_ReportsFlatLearningEvidence) {
	if (not OaEnvFlag::IsSet("OA_LUNAR_RUN_LEARNING_GATE")) {
		GTEST_SKIP() << "set OA_LUNAR_RUN_LEARNING_GATE=1 to run training";
	}
	OaEngine* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	ASSERT_TRUE(engine->IsReady());

	OaTutorialLunarLander3dPpoConfig config;
	config.Environments_ = OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_ENVIRONMENTS", 64U, 65536U);
	config.Horizon_ = OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_HORIZON", 128U, 1200U);
	config.Rollouts_ = OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_ROLLOUTS", 40U, 100000U);
	config.UpdateEpochs_ = OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_EPOCHS", 4U, 64U);
	config.HiddenSize_ = static_cast<OaI32>(OaLunarEvaluationU32(
		"OA_LUNAR_TRAIN_HIDDEN", 64U, 4096U));

	OaTutorialLunarLander3dFirstEpisodeEvaluationConfig evaluationConfig;
	evaluationConfig.Environments_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_ENVIRONMENTS", 512U, 65536U);
	evaluationConfig.Horizon_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_HORIZON", 1200U, 1200U);
	evaluationConfig.SubmissionChunkSteps_ = OaLunarEvaluationU32(
		"OA_LUNAR_EVAL_CHUNK", 16U, 256U);

	std::printf(
		"\nOA Lunar Lander 3D flat PPO report\n"
		"  train: %u environments x %u steps x %u rollouts, %u epochs, hidden %d\n"
		"  held out: %u first episodes, horizon %u, chunk %u, seed %016llx\n",
		config.Environments_, config.Horizon_, config.Rollouts_,
		config.UpdateEpochs_, config.HiddenSize_,
		evaluationConfig.Environments_, evaluationConfig.Horizon_,
		evaluationConfig.SubmissionChunkSteps_,
		static_cast<unsigned long long>(evaluationConfig.EnvironmentSeed_));

	auto created = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto session = OaStdMove(*created);
	auto beforeResult = session->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(beforeResult.IsOk()) << beforeResult.GetStatus().ToString();
	const auto before = *beforeResult;
	OaLunarLogEvaluation("untrained", before);

	OaU32 lastReportedRollout = 0U;
	while (not session->IsDone()) {
		const OaStatus advanced = session->Advance();
		ASSERT_TRUE(advanced.IsOk()) << advanced.ToString();
		const auto& metrics = session->Metrics();
		if (metrics.Rollout_ != lastReportedRollout
			and (metrics.Rollout_ % 10U == 0U or session->IsDone())) {
			lastReportedRollout = metrics.Rollout_;
			std::printf(
				"  rollout %u/%u | loss %.6f policy %.6f value %.6f entropy %.6f\n",
				metrics.Rollout_, config.Rollouts_, metrics.TotalLoss_,
				metrics.PolicyLoss_, metrics.ValueLoss_, metrics.Entropy_);
		}
	}

	auto afterResult = session->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(afterResult.IsOk()) << afterResult.GetStatus().ToString();
	const auto after = *afterResult;
	OaLunarLogEvaluation("trained", after);
	EXPECT_EQ(after.CompletedEpisodes_, after.ExpectedEpisodes_);
	EXPECT_EQ(after.IncompleteEpisodes_, 0U);

	const OaPath checkpointPath = OaPaths::Temp()
		/ OaPath("oa_lunar_lander_3d_flat_learning.oam");
	if (OaFilesystem::Exists(checkpointPath)) {
		ASSERT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	}
	ASSERT_TRUE(session->Save(checkpointPath.String()).IsOk());
	auto restoredResult = OaTutorialLunarLander3dPpo::Create(*engine, config);
	ASSERT_TRUE(restoredResult.IsOk())
		<< restoredResult.GetStatus().ToString();
	auto restored = OaStdMove(*restoredResult);
	ASSERT_TRUE(restored->Load(checkpointPath.String()).IsOk());
	auto restoredEvaluation = restored->EvaluateFirstEpisodes(evaluationConfig);
	ASSERT_TRUE(restoredEvaluation.IsOk())
		<< restoredEvaluation.GetStatus().ToString();
	OaLunarExpectEvaluationEqual(after, *restoredEvaluation);
	EXPECT_EQ(restored->OptimizerStep(), session->OptimizerStep());

	if (OaEnvFlag::IsSet("OA_LUNAR_REQUIRE_FLAT_GATE")) {
		EXPECT_GE(after.SafeLandingRate_, 0.80);
		EXPECT_GE(after.WilsonLower95_, 0.75);
		EXPECT_GT(after.MeanReturn_, before.MeanReturn_);
	}
	EXPECT_TRUE(OaFilesystem::RemoveFile(checkpointPath).IsOk());
	EXPECT_TRUE(restored->Close().IsOk());
	EXPECT_TRUE(session->Close().IsOk());
}
