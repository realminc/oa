#pragma once

#include <Oa/Core/Matrix.h>

namespace OaFnRl {

// Graph-native RL transforms. These are deliberately ordinary matrix
// functions rather than a dynamic wrapper hierarchy, so they compose inside
// the same deferred compute graph as an environment or policy.
[[nodiscard]] OaMatrix NormalizeObservation(
	const OaMatrix& InObservation,
	const OaMatrix& InMean,
	const OaMatrix& InStddev,
	OaF32 InEpsilon = 1.0e-6F,
	OaF32 InClip = 10.0F
);

[[nodiscard]] OaMatrix ScaleAction(
	const OaMatrix& InAction,
	OaF32 InSourceMinimum,
	OaF32 InSourceMaximum,
	OaF32 InTargetMinimum,
	OaF32 InTargetMaximum,
	bool InClamp = true
);

[[nodiscard]] OaMatrix ClipReward(
	const OaMatrix& InReward,
	OaF32 InMinimum = -1.0F,
	OaF32 InMaximum = 1.0F
);

} // namespace OaFnRl
