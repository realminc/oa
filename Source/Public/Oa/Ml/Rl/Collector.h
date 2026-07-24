#pragma once

#include <Oa/Ml/Rl/ActorCritic.h>
#include <Oa/Ml/Rl/Environment.h>
#include <Oa/Ml/Rl/Gae.h>
#include <Oa/Ml/Rl/Policy.h>
#include <Oa/Ml/Rl/Rollout.h>

struct OaRlCollectorConfig {
	OaU32 Horizon = 0;
	OaU64 Seed = 0;
	OaGaeConfig Gae;
};

struct OaRlCollectorMetrics {
	OaU64 Collections = 0;
	OaU64 EnvironmentSteps = 0;
	OaU64 Transitions = 0;
};

// Same-device categorical collector. It owns no environment, network or
// storage; Collect records one complete rollout transaction and returns its
// exact completion event without waiting.
class OaRlCollector {
public:
	[[nodiscard]] static OaResult<OaRlCollector> Create(
		OaRlEnvironment& InEnvironment,
		OaRlActorCritic& InModel,
		const OaRlCollectorConfig& InConfig
	);

	[[nodiscard]] OaResult<OaEvent> Collect(OaRlRolloutBuffer& InOutRollout);
	[[nodiscard]] const OaRlCollectorMetrics& Metrics() const noexcept {
		return Metrics_;
	}

private:
	OaRlEnvironment* Environment_ = nullptr;
	OaRlActorCritic* Model_ = nullptr;
	OaRlCollectorConfig Config_;
	OaRlCollectorMetrics Metrics_;
	OaU64 ActionIndex_ = 0;
};
