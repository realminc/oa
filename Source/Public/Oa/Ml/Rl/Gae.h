#pragma once

#include <Oa/Core/Matrix.h>

struct OaGaeConfig {
	OaF32 Gamma = 0.99F;
	OaF32 Lambda = 0.95F;
};

struct OaGaeResult {
	OaMatrix Advantage; // [Time, Environments], FP32
	OaMatrix Return;    // Advantage + Value, same shape

	[[nodiscard]] bool IsValid() const noexcept {
		return !Advantage.IsEmpty() && !Return.IsEmpty();
	}
};

namespace OaFnRl {

// Standardizes rollout advantages over every supplied sample. This is target
// preparation, not an optimization objective, so it remains on the RL
// algorithm surface rather than OaFnLoss.
[[nodiscard]] OaMatrix NormalizeAdvantages(
	const OaMatrix& InAdvantage,
	OaF32 InEpsilon = 1.0e-8F
);

// Generalized advantage estimation over a fixed [T, E] rollout.
//
// Reward, Value and NextValue are FP32. Terminated and Truncated are UInt8
// masks. Termination disables value bootstrapping. Both termination and
// truncation stop the reverse advantage trace so autoreset episodes never leak
// into each other; truncation still bootstraps from NextValue.
[[nodiscard]] OaGaeResult Gae(
	const OaMatrix& InReward,
	const OaMatrix& InValue,
	const OaMatrix& InNextValue,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaGaeConfig& InConfig = {}
);

// Allocation-free form used by fixed-capacity rollout storage.
[[nodiscard]] OaStatus GaeInto(
	const OaMatrix& InReward,
	const OaMatrix& InValue,
	const OaMatrix& InNextValue,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	OaMatrix& OutAdvantage,
	OaMatrix& OutReturn,
	const OaGaeConfig& InConfig = {}
);

} // namespace OaFnRl
