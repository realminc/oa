#pragma once

#include <Oa/Core/Matrix.h>

struct OaRlPolicyResult {
	OaMatrix Action;          // Int32 [Environments]
	OaMatrix LogProbability;  // FP32 [Environments]
	OaMatrix Entropy;         // FP32 [Environments]
	OaMatrix Value;           // FP32 [Environments]

	[[nodiscard]] bool IsValid() const noexcept {
		return !Action.IsEmpty() && !LogProbability.IsEmpty()
			&& !Entropy.IsEmpty() && !Value.IsEmpty();
	}
};

// Diagonal Gaussian transformed through tanh into a bounded action interval.
// RawAction is retained because it is the numerically stable carrier for PPO
// re-evaluation; Action is the value passed to the environment.
struct OaRlContinuousPolicyResult {
	OaMatrix Action;          // FP32 [Environments, ActionDimensions]
	OaMatrix RawAction;       // FP32 [Environments, ActionDimensions]
	OaMatrix LogProbability;  // FP32 [Environments]
	OaMatrix Entropy;         // FP32 [Environments], base diagonal Gaussian
	OaMatrix Value;           // FP32 [Environments]

	[[nodiscard]] bool IsValid() const noexcept {
		return !Action.IsEmpty() && !RawAction.IsEmpty()
			&& !LogProbability.IsEmpty() && !Entropy.IsEmpty()
			&& !Value.IsEmpty();
	}
};

namespace OaFnRl {

// Samples one discrete action per environment, then evaluates the selected
// action under the same logits. Seed zero follows OaFnMatrix RNG semantics and
// is nondeterministic; use a non-zero seed for reproducible collection.
[[nodiscard]] OaRlPolicyResult SampleCategoricalPolicy(
	const OaMatrix& InLogits,
	const OaMatrix& InValue,
	OaU64 InSeed = 0
);

// Differentiably evaluates stored actions for PPO updates. InAction must contain
// valid class indices in [0, action-count); indices themselves are not
// differentiable. Gradients flow through logits, log-probability and entropy.
[[nodiscard]] OaRlPolicyResult EvaluateCategoricalPolicy(
	const OaMatrix& InLogits,
	const OaMatrix& InAction,
	const OaMatrix& InValue
);

// Samples a reparameterized diagonal Gaussian and applies a tanh transform to
// [InMinimum, InMaximum]. The corrected log-probability includes the transform
// Jacobian. Seeds are explicit and follow the categorical policy convention.
[[nodiscard]] OaRlContinuousPolicyResult SampleTanhNormalPolicy(
	const OaMatrix& InMean,
	const OaMatrix& InLogStddev,
	const OaMatrix& InValue,
	OaF32 InMinimum = -1.0F,
	OaF32 InMaximum = 1.0F,
	OaU64 InSeed = 0,
	OaF32 InEpsilon = 1.0e-6F
);

// Differentiably re-evaluates a stored pre-tanh action. The environment action
// is reconstructed from RawAction, so no unstable inverse tanh is required.
[[nodiscard]] OaRlContinuousPolicyResult EvaluateTanhNormalPolicy(
	const OaMatrix& InMean,
	const OaMatrix& InLogStddev,
	const OaMatrix& InRawAction,
	const OaMatrix& InValue,
	OaF32 InMinimum = -1.0F,
	OaF32 InMaximum = 1.0F,
	OaF32 InEpsilon = 1.0e-6F
);

} // namespace OaFnRl
