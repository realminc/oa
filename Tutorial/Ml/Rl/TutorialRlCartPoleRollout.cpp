#include "../../../Test/OaTest.h"

#include "CartPole.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Runtime/Engine.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

OaMatrix MatrixF32(const std::vector<OaF32>& InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(InValues.data()),
			InValues.size() * sizeof(OaF32)),
		InShape, OaScalarType::Float32);
}

template<typename T>
std::vector<T> Copy(const OaMatrix& InMatrix) {
	std::vector<T> result(static_cast<OaUsize>(InMatrix.NumElements()));
	EXPECT_TRUE(OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(T)).IsOk());
	return result;
}

} // namespace

TEST(TutorialRlCartPoleRollout, VectorizedGpuCollection) {
	constexpr OaU32 environments = 32;
	constexpr OaU32 horizon = 64;
	constexpr OaU64 seed = 20260716ULL;

	std::printf("\n"
		"OaRl — vectorized GPU CartPole rollout\n"
		"  environments: %u · horizon: %u · transitions: %u\n",
		environments, horizon, environments * horizon);

	auto createdEnvironment = OaTutorialCartPole::Create(
		*OaEngine::GetGlobal(),
		OaTutorialCartPoleConfig{
			.Environments = environments,
			.MaxEpisodeSteps = 500,
			.Seed = seed,
		});
	ASSERT_TRUE(createdEnvironment.IsOk())
		<< createdEnvironment.GetStatus().ToString();
	auto environment = OaStdMove(*createdEnvironment);

	// A tiny fixed stochastic policy: logits are [-score, score], where pole
	// angle and angular velocity dominate the score. The tutorial proves the
	// collector path; the following milestone replaces these weights with a
	// trainable actor/critic and PPO update epochs.
	OaRlRolloutBuffer rollout;
	const OaStatus recorded = environment.RecordCommands([&]() -> OaStatus {
		auto createdRollout = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
			.Time = horizon,
			.Environments = environments,
			.ObservationShape = {4},
		});
		if (createdRollout.IsError()) return createdRollout.GetStatus();
		rollout = OaStdMove(*createdRollout);
		rollout.Reset();
		const OaMatrix policyWeight = MatrixF32({
			0.0F, -0.1F, -4.0F, -1.0F,
			0.0F,  0.1F,  4.0F,  1.0F,
		}, {2, 4});
		const OaMatrix value = OaFnMatrix::Zeros(
			{static_cast<OaI64>(environments)}, OaScalarType::Float32);
		if (policyWeight.IsEmpty() || value.IsEmpty()) {
			return OaStatus::Error(OaStatusCode::OutOfMemory,
				"CartPole rollout could not allocate policy storage");
		}
		OA_RETURN_IF_ERROR(environment.ResetEnvironment(seed));
		for (OaU32 step = 0; step < horizon; ++step) {
			const OaMatrix logits = OaFnMatrix::MatMulNt(
				environment.Observation(), policyWeight);
			const OaRlPolicyResult policy = OaFnRl::SampleCategoricalPolicy(
				logits, value, seed + step + 1U);
			if (!policy.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"CartPole rollout policy evaluation failed");
			const auto transition = environment.Step(policy.Action);
			if (transition.IsError()) return transition.GetStatus();
			OA_RETURN_IF_ERROR(rollout.Append(OaRlTransition{
				.Observation = transition->Observation,
				.Action = policy.Action,
				.Reward = transition->Reward,
				.Value = policy.Value,
				.NextValue = value,
				.LogProbability = policy.LogProbability,
				.Terminated = transition->Terminated,
				.Truncated = transition->Truncated,
			}));
			OA_RETURN_IF_ERROR(environment.ResetDone());
		}
		return rollout.Finalize();
	});
	ASSERT_TRUE(recorded.IsOk()) << recorded.ToString();
	auto completion = environment.Submit();
	ASSERT_TRUE(completion.IsOk()) << completion.GetStatus().ToString();
	ASSERT_TRUE(environment.Wait(*completion).IsOk());

	const auto reward = Copy<OaF32>(rollout.Batch().Reward);
	const auto terminated = Copy<OaU8>(rollout.Batch().Terminated);
	const auto truncated = Copy<OaU8>(rollout.Batch().Truncated);
	const auto valid = Copy<OaU8>(rollout.Batch().Valid);
	const auto advantage = Copy<OaF32>(rollout.Batch().Advantage);
	OaF64 rewardSum = 0.0;
	OaU32 episodes = 0;
	for (OaUsize index = 0; index < reward.size(); ++index) {
		rewardSum += reward[index];
		episodes += static_cast<OaU32>(terminated[index] != 0
			|| truncated[index] != 0);
		ASSERT_EQ(valid[index], 1U);
		ASSERT_TRUE(std::isfinite(advantage[index]));
	}

	std::printf(
		"  result: %.0f reward · %u completed episodes · %.2f reward/env\n"
		"  path: policy -> sample -> step -> append -> reset-done -> GAE\n"
		"  host tensor reads during collection: 0\n\n",
		rewardSum, episodes, rewardSum / environments);
	EXPECT_EQ(rollout.Size(), horizon);
	EXPECT_GT(rewardSum, 0.0);
}
