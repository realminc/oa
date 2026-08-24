#include <oa/ml/policyEvaluator.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/policy.h>
#include <oa/ml/rollout.h>

#include "environmentExecution.h"

#include <algorithm>
#include <limits>

oa::Result<oa::PolicyEvaluationMetrics> oa::PolicyEvaluator::evaluateCategorical(
	oa::Environment& inEnvironment,
	oa::ActorCritic& inModel,
	const oa::PolicyEvaluationConfig& inConfig) {
	if (inConfig.horizon == 0 || inEnvironment.environments() == 0) {
		return oa::Status::invalidArgument(
			"oa::PolicyEvaluator::evaluateCategorical expects a non-empty horizon and environment");
	}
	OA_RETURN_IF_ERROR(inEnvironment.spec().validateDefinition());
	if (inEnvironment.spec().action.kind != oa::EnvironmentSpaceKind::Discrete) {
		return oa::Status::invalidArgument(
			"oa::PolicyEvaluator::evaluateCategorical requires a discrete action space");
	}
	const oa::U32 environments = inEnvironment.environments();
	oa::RolloutBuffer rollout;
	const oa::I64 observationElements =
		inEnvironment.spec().observation.elementsPerEnvironment();
	const oa::Status recorded = inEnvironment.recordCommands([&]() -> oa::Status {
		auto rolloutResult = oa::RolloutBuffer::create({
			.time = inConfig.horizon,
			.environments = environments,
			.observationShape = inEnvironment.spec().observation.shape,
		});
		if (rolloutResult.isError()) return rolloutResult.getStatus();
		rollout = oa::move(*rolloutResult);
		rollout.reset();
		OA_RETURN_IF_ERROR(inEnvironment.reset(inConfig.seed));
		oa::GradNo noGrad;
		for (oa::U32 step = 0; step < inConfig.horizon; ++step) {
			const oa::Matrix observation = inEnvironment.observation();
			const oa::Matrix flat = oa::FnMatrix::reshape(observation,
				{static_cast<oa::I64>(environments), observationElements});
			const oa::ActorCriticOutput network = inModel.evaluate(flat);
			if (!network.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::PolicyEvaluator::evaluateCategorical actor/critic evaluation failed");
			const oa::Matrix action = oa::FnMatrix::sampleLogits(
				network.logits, 0.0F, 0, 1.0F, inConfig.seed);
			const oa::PolicyResult policy = oa::FnPolicy::evaluateCategorical(
				network.logits, action, network.value);
			if (!policy.isValid()) return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"oa::PolicyEvaluator::evaluateCategorical policy evaluation failed");
			auto transition = inEnvironment.step(action);
			if (transition.isError()) return transition.getStatus();
			OA_RETURN_IF_ERROR(rollout.append({
				.observation = transition->observation,
				.action = action,
				.reward = transition->reward,
				.value = network.value,
				.nextValue = network.value,
				.logProbability = policy.logProbability,
				.terminated = transition->terminated,
				.truncated = transition->truncated,
			}));
			OA_RETURN_IF_ERROR(inEnvironment.resetCompleted());
		}
		return oa::Status::ok();
	});
	if (recorded.isError()) return recorded;
	auto completion = inEnvironment.submit();
	if (completion.isError()) return completion.getStatus();
	OA_RETURN_IF_ERROR(inEnvironment.wait(*completion));
	const oa::U64 transitions = static_cast<oa::U64>(inConfig.horizon) * environments;
	oa::Vec<oa::F32> reward(static_cast<oa::Usize>(transitions));
	oa::Vec<oa::U8> terminated(static_cast<oa::Usize>(transitions));
	oa::Vec<oa::U8> truncated(static_cast<oa::Usize>(transitions));
	{
		// The environment owns the execution context that produced the rollout.
		// Keep its engine selected for every host readback instead of falling
		// through to an unrelated ambient compatibility context.
		oa::EnvironmentRecordingScope scope(inEnvironment);
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			rollout.batch().reward, reward.data(),
			transitions * sizeof(oa::F32)));
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			rollout.batch().terminated, terminated.data(), transitions));
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			rollout.batch().truncated, truncated.data(), transitions));
	}

	oa::Vec<oa::F32> episodeReturn(environments, 0.0F);
	oa::F64 sum = 0.0;
	oa::F32 minimum = std::numeric_limits<oa::F32>::infinity();
	oa::F32 maximum = -std::numeric_limits<oa::F32>::infinity();
	oa::U64 completed = 0;
	for (oa::U32 step = 0; step < inConfig.horizon; ++step) {
		for (oa::U32 lane = 0; lane < environments; ++lane) {
			const oa::Usize index = static_cast<oa::Usize>(step) * environments + lane;
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
	return oa::PolicyEvaluationMetrics{
		.environmentSteps = inConfig.horizon,
		.transitions = transitions,
		.completedEpisodes = completed,
		.meanCompletedReturn = completed == 0
			? 0.0F : static_cast<oa::F32>(sum / static_cast<oa::F64>(completed)),
		.minimumCompletedReturn = completed == 0 ? 0.0F : minimum,
		.maximumCompletedReturn = completed == 0 ? 0.0F : maximum,
	};
}
