#pragma once

#include <oa/core/matrix.h>

namespace oa {

struct PolicyResult {
	oa::Matrix action;          // Int32 [environments]
	oa::Matrix logProbability;  // FP32 [environments]
	oa::Matrix entropy;         // FP32 [environments]
	oa::Matrix value;           // FP32 [environments]

	[[nodiscard]] bool isValid() const noexcept {
		return !action.isEmpty() && !logProbability.isEmpty()
			&& !entropy.isEmpty() && !value.isEmpty();
	}
};

// Diagonal Gaussian transformed through tanh into a bounded action interval.
// rawAction is retained because it is the numerically stable carrier for PPO
// re-evaluation; action is the value passed to the environment.
struct ContinuousPolicyResult {
	oa::Matrix action;          // FP32 [environments, actionDimensions]
	oa::Matrix rawAction;       // FP32 [environments, actionDimensions]
	oa::Matrix logProbability;  // FP32 [environments]
	oa::Matrix entropy;         // FP32 [environments], base diagonal Gaussian
	oa::Matrix value;           // FP32 [environments]

	[[nodiscard]] bool isValid() const noexcept {
		return !action.isEmpty() && !rawAction.isEmpty()
			&& !logProbability.isEmpty() && !entropy.isEmpty()
			&& !value.isEmpty();
	}
};

namespace FnPolicy {

// samples one discrete action per environment, then evaluates the selected
// action under the same logits. seed zero follows FnMatrix RNG semantics and
// is nondeterministic; use a non-zero seed for reproducible collection.
[[nodiscard]] PolicyResult sampleCategorical(
	const oa::Matrix& inLogits,
	const oa::Matrix& inValue,
	oa::U64 inSeed = 0
);

// Differentiably evaluates stored actions for PPO updates. inAction must contain
// valid class indices in [0, action-count); indices themselves are not
// differentiable. gradients flow through logits, log-probability and entropy.
[[nodiscard]] PolicyResult evaluateCategorical(
	const oa::Matrix& inLogits,
	const oa::Matrix& inAction,
	const oa::Matrix& inValue
);

// samples a reparameterized diagonal Gaussian and applies a tanh transform to
// [inMinimum, inMaximum]. The corrected log-probability includes the transform
// Jacobian. Seeds are explicit and follow the categorical policy convention.
[[nodiscard]] ContinuousPolicyResult sampleTanhNormal(
	const oa::Matrix& inMean,
	const oa::Matrix& inLogStddev,
	const oa::Matrix& inValue,
	oa::F32 inMinimum = -1.0F,
	oa::F32 inMaximum = 1.0F,
	oa::U64 inSeed = 0,
	oa::F32 inEpsilon = 1.0e-6F
);

// Differentiably re-evaluates a stored pre-tanh action. The environment action
// is reconstructed from rawAction, so no unstable inverse tanh is required.
[[nodiscard]] ContinuousPolicyResult evaluateTanhNormal(
	const oa::Matrix& inMean,
	const oa::Matrix& inLogStddev,
	const oa::Matrix& inRawAction,
	const oa::Matrix& inValue,
	oa::F32 inMinimum = -1.0F,
	oa::F32 inMaximum = 1.0F,
	oa::F32 inEpsilon = 1.0e-6F
);

} // namespace FnPolicy

} // namespace oa
