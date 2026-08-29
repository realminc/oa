#include "oaTest.h"

#include <ml/rl/cartPole.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml.h>
#include <oa/runtime/engine.h>

namespace {

oa::Matrix matrixF32(const oa::Vector<oa::F32>& inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(oa::F32)),
		inShape, oa::ScalarType::Float32);
}

template<typename T>
oa::Vector<T> copy(const oa::Matrix& inMatrix) {
	oa::Vector<T> result(static_cast<oa::Usize>(inMatrix.numElements()));
	EXPECT_TRUE(oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(T)).isOk());
	return result;
}

} // namespace

TEST(TutorialRlCartPoleRollout, VectorizedGpuCollection) {
	constexpr oa::U32 environments = 32;
	constexpr oa::U32 horizon = 64;
	constexpr oa::U64 seed = 20260716ULL;

	oa::print("\n" "OA reinforcement learning — vectorized GPU CartPole rollout\n" "  environments: {} · horizon: {} · transitions: {}",
		environments, horizon, environments * horizon);

	auto createdEnvironment = oa::CartPole::create(
		testEngine(),
		oa::CartPoleConfig{
			.environments = environments,
			.maxEpisodeSteps = 500,
			.seed = seed,
		});
	ASSERT_TRUE(createdEnvironment.isOk())
		<< createdEnvironment.getStatus().toString();
	auto environment = oa::move(*createdEnvironment);

	// A tiny fixed stochastic policy: logits are [-score, score], where pole
	// angle and angular velocity dominate the score. The tutorial proves the
	// collector path; the following milestone replaces these weights with a
	// trainable actor/critic and PPO update epochs.
	oa::RolloutBuffer rollout;
	const oa::Status recorded = environment.recordCommands([&]() -> oa::Status {
		auto createdRollout = oa::RolloutBuffer::create(oa::RolloutConfig{
			.time = horizon,
			.environments = environments,
			.observationShape = {4},
		});
		if (createdRollout.isError()) return createdRollout.getStatus();
		rollout = oa::move(*createdRollout);
		rollout.reset();
		const oa::Matrix policyWeight = matrixF32({
			0.0F, -0.1F, -4.0F, -1.0F,
			0.0F,  0.1F,  4.0F,  1.0F,
		}, {2, 4});
		const oa::Matrix value = oa::FnMatrix::zeros(
			{static_cast<oa::I64>(environments)}, oa::ScalarType::Float32);
		if (policyWeight.isEmpty() || value.isEmpty()) {
			return oa::Status::error(oa::StatusCode::OutOfMemory,
				"CartPole rollout could not allocate policy storage");
		}
		OA_RETURN_IF_ERROR(environment.reset(seed));
		for (oa::U32 step = 0; step < horizon; ++step) {
			const oa::Matrix logits = oa::FnMatrix::matMulNt(
				environment.observation(), policyWeight);
			const oa::PolicyResult policy = oa::FnPolicy::sampleCategorical(
				logits, value, seed + step + 1U);
			if (!policy.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"CartPole rollout policy evaluation failed");
			const auto transition = environment.step(policy.action);
			if (transition.isError()) return transition.getStatus();
			OA_RETURN_IF_ERROR(rollout.append(oa::RolloutTransition{
				.observation = transition->observation,
				.action = policy.action,
				.reward = transition->reward,
				.value = policy.value,
				.nextValue = value,
				.logProbability = policy.logProbability,
				.terminated = transition->terminated,
				.truncated = transition->truncated,
			}));
			OA_RETURN_IF_ERROR(environment.resetDone());
		}
		return rollout.finalize();
	});
	ASSERT_TRUE(recorded.isOk()) << recorded.toString();
	auto completion = environment.submit();
	ASSERT_TRUE(completion.isOk()) << completion.getStatus().toString();
	ASSERT_TRUE(environment.wait(*completion).isOk());

	const auto reward = copy<oa::F32>(rollout.batch().reward);
	const auto terminated = copy<oa::U8>(rollout.batch().terminated);
	const auto truncated = copy<oa::U8>(rollout.batch().truncated);
	const auto valid = copy<oa::U8>(rollout.batch().valid);
	const auto advantage = copy<oa::F32>(rollout.batch().advantage);
	oa::F64 rewardSum = 0.0;
	oa::U32 episodes = 0;
	for (oa::Usize index = 0; index < reward.size(); ++index) {
		rewardSum += reward[index];
		episodes += static_cast<oa::U32>(terminated[index] != 0
			|| truncated[index] != 0);
		ASSERT_EQ(valid[index], 1U);
		ASSERT_TRUE(oa::isFinite(advantage[index]));
	}

	oa::print("  result: {:.0f} reward · {} completed episodes · {:.2f} reward/env\n" "  path: policy -> sample -> step -> append -> reset-done -> GAE\n" "  host tensor reads during collection: 0\n",
		rewardSum, episodes, rewardSum / environments);
	EXPECT_EQ(rollout.size(), horizon);
	EXPECT_GT(rewardSum, 0.0);
}
