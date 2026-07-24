#include <Oa/Ml/Rl/Evaluation.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Rl/Policy.h>
#include <Oa/Ml/Rl/Rollout.h>

#include "EnvironmentExecution.h"

#include <algorithm>
#include <limits>

OaResult<OaRlEvaluationMetrics> OaFnRl::EvaluateCategorical(
	OaRlEnvironment& InEnvironment,
	OaRlActorCritic& InModel,
	const OaRlEvaluationConfig& InConfig) {
	if (InConfig.Horizon == 0 || InEnvironment.Environments() == 0) {
		return OaStatus::InvalidArgument(
			"OaFnRl::EvaluateCategorical expects a non-empty horizon and environment");
	}
	OA_RETURN_IF_ERROR(InEnvironment.Spec().ValidateDefinition());
	if (InEnvironment.Spec().Action.Kind != OaRlSpaceKind::Discrete) {
		return OaStatus::InvalidArgument(
			"OaFnRl::EvaluateCategorical requires a discrete action space");
	}
	const OaU32 environments = InEnvironment.Environments();
	OaRlRolloutBuffer rollout;
	const OaI64 observationElements =
		InEnvironment.Spec().Observation.ElementsPerEnvironment();
	const OaStatus recorded = InEnvironment.RecordCommands([&]() -> OaStatus {
		auto rolloutResult = OaRlRolloutBuffer::Create({
			.Time = InConfig.Horizon,
			.Environments = environments,
			.ObservationShape = InEnvironment.Spec().Observation.Shape,
		});
		if (rolloutResult.IsError()) return rolloutResult.GetStatus();
		rollout = OaStdMove(*rolloutResult);
		rollout.Reset();
		OA_RETURN_IF_ERROR(InEnvironment.ResetEnvironment(InConfig.Seed));
		OaGradNo noGrad;
		for (OaU32 step = 0; step < InConfig.Horizon; ++step) {
			const OaMatrix observation = InEnvironment.Observation();
			const OaMatrix flat = OaFnMatrix::Reshape(observation,
				{static_cast<OaI64>(environments), observationElements});
			const OaRlActorCriticOutput network = InModel.Evaluate(flat);
			if (!network.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"OaFnRl::EvaluateCategorical actor/critic evaluation failed");
			const OaMatrix action = OaFnMatrix::SampleLogits(
				network.Logits, 0.0F, 0, 1.0F, InConfig.Seed);
			const OaRlPolicyResult policy = OaFnRl::EvaluateCategoricalPolicy(
				network.Logits, action, network.Value);
			if (!policy.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"OaFnRl::EvaluateCategorical policy evaluation failed");
			auto transition = InEnvironment.StepEnvironment(action);
			if (transition.IsError()) return transition.GetStatus();
			OA_RETURN_IF_ERROR(rollout.Append({
				.Observation = transition->Observation,
				.Action = action,
				.Reward = transition->Reward,
				.Value = network.Value,
				.NextValue = network.Value,
				.LogProbability = policy.LogProbability,
				.Terminated = transition->Terminated,
				.Truncated = transition->Truncated,
			}));
			OA_RETURN_IF_ERROR(InEnvironment.ResetCompleted());
		}
		return OaStatus::Ok();
	});
	if (recorded.IsError()) return recorded;
	auto completion = InEnvironment.Submit();
	if (completion.IsError()) return completion.GetStatus();
	OA_RETURN_IF_ERROR(InEnvironment.Wait(*completion));
	const OaU64 transitions = static_cast<OaU64>(InConfig.Horizon) * environments;
	OaVec<OaF32> reward(static_cast<OaUsize>(transitions));
	OaVec<OaU8> terminated(static_cast<OaUsize>(transitions));
	OaVec<OaU8> truncated(static_cast<OaUsize>(transitions));
	{
		// The environment owns the execution context that produced the rollout.
		// Keep its engine selected for every host readback instead of falling
		// through to an unrelated ambient compatibility context.
		OaRlEnvironmentRecordingScope scope(InEnvironment);
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			rollout.Batch().Reward, reward.data(),
			transitions * sizeof(OaF32)));
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			rollout.Batch().Terminated, terminated.data(), transitions));
		OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
			rollout.Batch().Truncated, truncated.data(), transitions));
	}

	OaVec<OaF32> episodeReturn(environments, 0.0F);
	OaF64 sum = 0.0;
	OaF32 minimum = std::numeric_limits<OaF32>::infinity();
	OaF32 maximum = -std::numeric_limits<OaF32>::infinity();
	OaU64 completed = 0;
	for (OaU32 step = 0; step < InConfig.Horizon; ++step) {
		for (OaU32 lane = 0; lane < environments; ++lane) {
			const OaUsize index = static_cast<OaUsize>(step) * environments + lane;
			episodeReturn[lane] += reward[index];
			if (terminated[index] != 0 || truncated[index] != 0) {
				sum += episodeReturn[lane];
				minimum = std::min(minimum, episodeReturn[lane]);
				maximum = std::max(maximum, episodeReturn[lane]);
				episodeReturn[lane] = 0.0F;
				++completed;
			}
		}
	}
	return OaRlEvaluationMetrics{
		.EnvironmentSteps = InConfig.Horizon,
		.Transitions = transitions,
		.CompletedEpisodes = completed,
		.MeanCompletedReturn = completed == 0
			? 0.0F : static_cast<OaF32>(sum / static_cast<OaF64>(completed)),
		.MinimumCompletedReturn = completed == 0 ? 0.0F : minimum,
		.MaximumCompletedReturn = completed == 0 ? 0.0F : maximum,
	};
}
