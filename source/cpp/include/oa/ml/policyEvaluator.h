#pragma once

#include <oa/core/status.h>
#include <oa/ml/actorCritic.h>
#include <oa/ml/environment.h>

namespace oa {

struct PolicyEvaluationConfig {
	oa::U32 horizon = 1000;
	oa::U64 seed = 1;
};

struct PolicyEvaluationMetrics {
	oa::U64 environmentSteps = 0;
	oa::U64 transitions = 0;
	oa::U64 completedEpisodes = 0;
	oa::F32 meanCompletedReturn = 0.0F;
	oa::F32 minimumCompletedReturn = 0.0F;
	oa::F32 maximumCompletedReturn = 0.0F;
};

// Deterministic categorical evaluation over any native OA vector environment.
// Evaluation is an explicit telemetry boundary: it records the entire horizon
// first, then performs one execution/synchronization and three compact readbacks.
class PolicyEvaluator final {
public:
	PolicyEvaluator() = delete;

	[[nodiscard]] static oa::Result<PolicyEvaluationMetrics> evaluateCategorical(
		Environment& inEnvironment,
		ActorCritic& inModel,
		const PolicyEvaluationConfig& inConfig = {}
	);
};

} // namespace oa
