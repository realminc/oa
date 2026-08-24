#pragma once

#include <oa/ml/actorCritic.h>
#include <oa/ml/environment.h>
#include <oa/ml/advantage.h>
#include <oa/ml/policy.h>
#include <oa/ml/rollout.h>

namespace oa {

struct RolloutCollectorConfig {
	oa::U32 horizon = 0;
	oa::U64 seed = 0;
	GaeConfig gae;
};

struct RolloutCollectorMetrics {
	oa::U64 collections = 0;
	oa::U64 environmentSteps = 0;
	oa::U64 transitions = 0;
};

// Same-device categorical collector. It owns no environment, network or
// storage; collect records one complete rollout transaction and returns its
// exact completion event without waiting.
class RolloutCollector {
public:
	[[nodiscard]] static oa::Result<RolloutCollector> create(
		Environment& inEnvironment,
		ActorCritic& inModel,
		const RolloutCollectorConfig& inConfig
	);

	[[nodiscard]] oa::Result<oa::Event> collect(RolloutBuffer& inOutRollout);
	[[nodiscard]] const RolloutCollectorMetrics& metrics() const noexcept { return metrics_;	}

private:
	Environment* environment_ = nullptr;
	ActorCritic* model_ = nullptr;
	RolloutCollectorConfig config_;
	RolloutCollectorMetrics metrics_;
	oa::U64 actionIndex_ = 0;
};

} // namespace oa
