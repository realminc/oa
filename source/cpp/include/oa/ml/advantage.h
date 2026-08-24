#pragma once

#include <oa/core/matrix.h>

namespace oa {

struct GaeConfig {
	oa::F32 gamma = 0.99F;
	oa::F32 lambda = 0.95F;
};

struct GaeResult {
	oa::Matrix advantage; // [time, environments], FP32
	oa::Matrix ret;       // advantage + Value, same shape

	[[nodiscard]] bool isValid() const noexcept {
		return !advantage.isEmpty() && !ret.isEmpty();
	}
};

namespace FnAdvantage {
	// Standardizes rollout advantages over every supplied sample. This is target
	// preparation, not an optimization objective, so it remains on the RL
	// algorithm surface rather than FnLoss.
	[[nodiscard]] oa::Matrix normalize(
		const oa::Matrix& inAdvantage,
		oa::F32 inEpsilon = 1.0e-8F
	);

	// Generalized advantage estimation over a fixed [T, E] rollout.
	//
	// reward, Value and nextValue are FP32. terminated and truncated are UInt8
	// masks. Termination disables value bootstrapping. Both termination and
	// truncation stop the reverse advantage trace so autoreset episodes never leak
	// into each other; truncation still bootstraps from nextValue.
	[[nodiscard]] GaeResult gae(
		const oa::Matrix& inReward,
		const oa::Matrix& inValue,
		const oa::Matrix& inNextValue,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		const GaeConfig& inConfig = {}
	);

	// allocation-free form used by fixed-capacity rollout storage.
	[[nodiscard]] oa::Status gaeInto(
		const oa::Matrix& inReward,
		const oa::Matrix& inValue,
		const oa::Matrix& inNextValue,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		oa::Matrix& outAdvantage,
		oa::Matrix& outReturn,
		const GaeConfig& inConfig = {}
	);

} // namespace FnAdvantage

} // namespace oa
