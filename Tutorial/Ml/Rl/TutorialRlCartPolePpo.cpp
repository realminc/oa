#include "../../../Test/OaTest.h"

#include "CartPolePpo.h"

#include <cstdio>

TEST(TutorialRlCartPolePpo, LearnsFromVectorizedGpuRollouts) {
	constexpr OaU64 evaluationSeed = 0x0e7a1ULL;
	const OaTutorialCartPolePpoConfig config;

	std::printf("\nOaRl — CartPole PPO\n"
		"  separate actor/critic: each 4 -> 64 -> 64\n"
		"  heads: categorical policy 2 · scalar value 1\n"
		"  rollout: %u env x %u steps · %u PPO epochs · %u rollouts\n",
		config.Environments, config.Horizon,
		config.UpdateEpochs, config.Rollouts);

	auto created = OaTutorialCartPolePpo::Create(
		*OaEngine::GetGlobal(), config);
	ASSERT_TRUE(created.IsOk()) << created.GetStatus().ToString();
	auto session = OaStdMove(*created);
	auto beforeResult = session->Evaluate(evaluationSeed);
	ASSERT_TRUE(beforeResult.IsOk()) << beforeResult.GetStatus().ToString();
	const auto before = *beforeResult;
	std::printf("  before: %.2f mean completed return (%u episodes)\n",
		before.MeanCompletedReturn, before.CompletedEpisodes);

	OaU32 lastPrinted = 0;
	while (!session->IsDone()) {
		ASSERT_TRUE(session->Advance().IsOk());
		const auto& metrics = session->Metrics();
		if (metrics.Rollout != lastPrinted && metrics.Rollout % 10U == 0U) {
			lastPrinted = metrics.Rollout;
			std::printf("  rollout %u/%u · loss %.5f\n",
				metrics.Rollout, config.Rollouts, metrics.TotalLoss);
		}
	}

	auto afterResult = session->Evaluate(evaluationSeed);
	ASSERT_TRUE(afterResult.IsOk()) << afterResult.GetStatus().ToString();
	const auto after = *afterResult;

	const OaString checkpointPath = "/tmp/oa_cartpole_ppo.oam";
	ASSERT_TRUE(session->Save(checkpointPath).IsOk());
	auto restoredResult = OaTutorialCartPolePpo::Create(
		*OaEngine::GetGlobal(), config);
	ASSERT_TRUE(restoredResult.IsOk())
		<< restoredResult.GetStatus().ToString();
	auto restoredSession = OaStdMove(*restoredResult);
	ASSERT_TRUE(restoredSession->Load(checkpointPath).IsOk());
	auto restoredResultScore = restoredSession->Evaluate(evaluationSeed);
	ASSERT_TRUE(restoredResultScore.IsOk())
		<< restoredResultScore.GetStatus().ToString();
	const auto restored = *restoredResultScore;
	std::remove(checkpointPath.c_str());

	std::printf("  after:  %.2f mean completed return (%u episodes)\n",
		after.MeanCompletedReturn, after.CompletedEpisodes);
	std::printf("  improvement: %+.2f\n\n",
		after.MeanCompletedReturn - before.MeanCompletedReturn);
	std::printf("  checkpoint: %.2f restored return · AdamW step %llu\n\n",
		restored.MeanCompletedReturn,
		static_cast<unsigned long long>(restoredSession->OptimizerStep()));
	EXPECT_GE(after.MeanCompletedReturn, before.MeanCompletedReturn + 25.0);
	EXPECT_GE(after.MeanCompletedReturn, 75.0);
	EXPECT_DOUBLE_EQ(restored.MeanCompletedReturn, after.MeanCompletedReturn);
	EXPECT_EQ(restored.CompletedEpisodes, after.CompletedEpisodes);
	EXPECT_EQ(restoredSession->OptimizerStep(), session->OptimizerStep());
}

TEST(TutorialRlCartPolePpo, ObservationDoesNotChangeSeededTraining) {
	constexpr OaU64 evaluationSeed = 0x0e7a1ULL;
	OaTutorialCartPolePpoConfig config;
	config.Rollouts = 8;

	auto headlessResult = OaTutorialCartPolePpo::Create(
		*OaEngine::GetGlobal(), config);
	ASSERT_TRUE(headlessResult.IsOk());
	auto headless = OaStdMove(*headlessResult);
	while (!headless->IsDone()) {
		ASSERT_TRUE(headless->Advance().IsOk());
	}
	auto headlessScore = headless->Evaluate(evaluationSeed);
	ASSERT_TRUE(headlessScore.IsOk());

	auto observedResult = OaTutorialCartPolePpo::Create(
		*OaEngine::GetGlobal(), config);
	ASSERT_TRUE(observedResult.IsOk());
	auto observed = OaStdMove(*observedResult);
	while (!observed->IsDone()) {
		ASSERT_TRUE(observed->Advance().IsOk());
		auto snapshot = observed->SnapshotLane(0);
		ASSERT_TRUE(snapshot.IsOk());
	}
	auto observedScore = observed->Evaluate(evaluationSeed);
	ASSERT_TRUE(observedScore.IsOk());

	EXPECT_DOUBLE_EQ(
		observedScore->MeanCompletedReturn,
		headlessScore->MeanCompletedReturn);
	EXPECT_EQ(observedScore->CompletedEpisodes, headlessScore->CompletedEpisodes);
	EXPECT_EQ(observed->OptimizerStep(), headless->OptimizerStep());
	ASSERT_EQ(
		observed->Metrics().LossHistory.Size(),
		headless->Metrics().LossHistory.Size());
	for (OaUsize index = 0; index < headless->Metrics().LossHistory.Size(); ++index) {
		EXPECT_FLOAT_EQ(
			observed->Metrics().LossHistory[index],
			headless->Metrics().LossHistory[index]);
	}
}
