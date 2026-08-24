#pragma once

#include <oa/core/matrix.h>

namespace oa {

namespace FnEnvironment {

// graph-native RL transforms. These are deliberately ordinary matrix
// functions rather than a dynamic wrapper hierarchy, so they compose inside
// the same deferred compute graph as an environment or policy.
[[nodiscard]] oa::Matrix normalizeObservation(
	const oa::Matrix& inObservation,
	const oa::Matrix& inMean,
	const oa::Matrix& inStddev,
	oa::F32 inEpsilon = 1.0e-6F,
	oa::F32 inClip = 10.0F
);

[[nodiscard]] oa::Matrix scaleAction(
	const oa::Matrix& inAction,
	oa::F32 inSourceMinimum,
	oa::F32 inSourceMaximum,
	oa::F32 inTargetMinimum,
	oa::F32 inTargetMaximum,
	bool inClamp = true
);

[[nodiscard]] oa::Matrix clipReward(
	const oa::Matrix& inReward,
	oa::F32 inMinimum = -1.0F,
	oa::F32 inMaximum = 1.0F
);

} // namespace FnEnvironment

} // namespace oa
