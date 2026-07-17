#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Ml/Rl/ActorCritic.h>
#include <Oa/Ml/Rl/Environment.h>

struct OaRlEvaluationConfig {
	OaU32 Horizon = 1000;
	OaU64 Seed = 1;
};

struct OaRlEvaluationMetrics {
	OaU64 EnvironmentSteps = 0;
	OaU64 Transitions = 0;
	OaU64 CompletedEpisodes = 0;
	OaF32 MeanCompletedReturn = 0.0F;
	OaF32 MinimumCompletedReturn = 0.0F;
	OaF32 MaximumCompletedReturn = 0.0F;
};

// Deterministic categorical evaluation over any native OA vector environment.
// Evaluation is an explicit telemetry boundary: it records the entire horizon
// first, then performs one execution/synchronization and two compact readbacks.
namespace OaFnRl {

[[nodiscard]] OaResult<OaRlEvaluationMetrics> EvaluateCategorical(
	OaRlEnvironment& InEnvironment,
	OaRlActorCritic& InModel,
	const OaRlEvaluationConfig& InConfig = {}
);

} // namespace OaFnRl
