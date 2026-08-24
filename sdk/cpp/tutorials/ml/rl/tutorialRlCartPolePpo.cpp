#include "oaTest.h"

#include "cartPolePpo.h"

#include <cstdio>

TEST(TutorialRlCartPolePpo, LearnsFromVectorizedGpuRollouts) {
	constexpr oa::U64 evaluationSeed = 0x0e7a1ULL;
	const TutorialCartPolePpoConfig config;

	std::printf("\nOA reinforcement learning — CartPole PPO\n"
		"  separate actor/critic: each 4 -> 64 -> 64\n"
		"  heads: categorical policy 2 · scalar value 1\n"
		"  rollout: %u env x %u steps · %u PPO epochs · %u rollouts\n",
		config.environments, config.horizon,
		config.updateEpochs, config.rollouts);

	auto created = TutorialCartPolePpo::create(
		testEngine(), config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto session = oa::move(*created);
	auto beforeResult = session->evaluate(evaluationSeed);
	ASSERT_TRUE(beforeResult.isOk()) << beforeResult.getStatus().toString();
	const auto before = *beforeResult;
	std::printf("  before: %.2f mean completed return (%u episodes)\n",
		before.meanCompletedReturn, before.completedEpisodes);

	oa::U32 lastPrinted = 0;
	while (!session->isDone()) {
		ASSERT_TRUE(session->advance().isOk());
		const auto& metrics = session->metrics();
		if (metrics.rollout != lastPrinted && metrics.rollout % 10U == 0U) {
			lastPrinted = metrics.rollout;
			std::printf("  rollout %u/%u · loss %.5f\n",
				metrics.rollout, config.rollouts, metrics.totalLoss);
		}
	}

	auto afterResult = session->evaluate(evaluationSeed);
	ASSERT_TRUE(afterResult.isOk()) << afterResult.getStatus().toString();
	const auto after = *afterResult;

	const oa::String checkpointPath = "/tmp/oa_cartpole_ppo.oam";
	ASSERT_TRUE(session->save(checkpointPath).isOk());
	auto restoredResult = TutorialCartPolePpo::create(
		testEngine(), config);
	ASSERT_TRUE(restoredResult.isOk())
		<< restoredResult.getStatus().toString();
	auto restoredSession = oa::move(*restoredResult);
	ASSERT_TRUE(restoredSession->load(checkpointPath).isOk());
	auto restoredResultScore = restoredSession->evaluate(evaluationSeed);
	ASSERT_TRUE(restoredResultScore.isOk())
		<< restoredResultScore.getStatus().toString();
	const auto restored = *restoredResultScore;
	std::remove(checkpointPath.cStr());

	std::printf("  after:  %.2f mean completed return (%u episodes)\n",
		after.meanCompletedReturn, after.completedEpisodes);
	std::printf("  improvement: %+.2f\n\n",
		after.meanCompletedReturn - before.meanCompletedReturn);
	std::printf("  checkpoint: %.2f restored return · AdamW step %llu\n\n",
		restored.meanCompletedReturn,
		static_cast<unsigned long long>(restoredSession->optimizerStep()));
	EXPECT_GE(after.meanCompletedReturn, before.meanCompletedReturn + 25.0);
	EXPECT_GE(after.meanCompletedReturn, 75.0);
	EXPECT_DOUBLE_EQ(restored.meanCompletedReturn, after.meanCompletedReturn);
	EXPECT_EQ(restored.completedEpisodes, after.completedEpisodes);
	EXPECT_EQ(restoredSession->optimizerStep(), session->optimizerStep());
}

TEST(TutorialRlCartPolePpo, ObservationDoesNotChangeSeededTraining) {
	constexpr oa::U64 evaluationSeed = 0x0e7a1ULL;
	TutorialCartPolePpoConfig config;
	config.rollouts = 8;

	auto headlessResult = TutorialCartPolePpo::create(
		testEngine(), config);
	ASSERT_TRUE(headlessResult.isOk());
	auto headless = oa::move(*headlessResult);
	while (!headless->isDone()) {
		ASSERT_TRUE(headless->advance().isOk());
	}
	auto headlessScore = headless->evaluate(evaluationSeed);
	ASSERT_TRUE(headlessScore.isOk());

	auto observedResult = TutorialCartPolePpo::create(
		testEngine(), config);
	ASSERT_TRUE(observedResult.isOk());
	auto observed = oa::move(*observedResult);
	while (!observed->isDone()) {
		ASSERT_TRUE(observed->advance().isOk());
		auto snapshot = observed->snapshotLane(0);
		ASSERT_TRUE(snapshot.isOk());
	}
	auto observedScore = observed->evaluate(evaluationSeed);
	ASSERT_TRUE(observedScore.isOk());

	EXPECT_DOUBLE_EQ(
		observedScore->meanCompletedReturn,
		headlessScore->meanCompletedReturn);
	EXPECT_EQ(observedScore->completedEpisodes, headlessScore->completedEpisodes);
	EXPECT_EQ(observed->optimizerStep(), headless->optimizerStep());
	ASSERT_EQ(
		observed->metrics().lossHistory.size(),
		headless->metrics().lossHistory.size());
	for (oa::Usize index = 0; index < headless->metrics().lossHistory.size(); ++index) {
		EXPECT_FLOAT_EQ(
			observed->metrics().lossHistory[index],
			headless->metrics().lossHistory[index]);
	}
}
