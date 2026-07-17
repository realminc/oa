#pragma once

#include <Oa/Core/Matrix.h>

struct OaPpoLossConfig {
	OaF32 ClipEpsilon = 0.2F;
	OaF32 ValueCoefficient = 0.5F;
	OaF32 EntropyCoefficient = 0.01F;
};

struct OaPpoLossResult {
	OaMatrix PolicyLoss;
	OaMatrix ValueLoss;
	OaMatrix Entropy;
	OaMatrix TotalLoss;

	[[nodiscard]] bool IsValid() const noexcept {
		return !PolicyLoss.IsEmpty() && !ValueLoss.IsEmpty() && !Entropy.IsEmpty() && !TotalLoss.IsEmpty();
	}
};

struct OaDqnLossConfig {
	OaF32 Discount = 0.99F;
};

struct OaDqnLossResult {
	OaMatrix SelectedQ;
	OaMatrix TargetQ;
	OaMatrix Loss;

	[[nodiscard]] bool IsValid() const noexcept {
		return !SelectedQ.IsEmpty() && !TargetQ.IsEmpty() && !Loss.IsEmpty();
	}
};

struct OaSacLossConfig {
	OaF32 Discount = 0.99F;
	OaF32 EntropyCoefficient = 0.2F;
};

struct OaSacCriticLossResult {
	OaMatrix TargetQ;
	OaMatrix Q1Loss;
	OaMatrix Q2Loss;
	OaMatrix TotalLoss;
	[[nodiscard]] bool IsValid() const noexcept {
		return !TargetQ.IsEmpty() && !Q1Loss.IsEmpty()
			&& !Q2Loss.IsEmpty() && !TotalLoss.IsEmpty();
	}
};

namespace OaFnLoss {

// Scalar -mean(min(ratio*A, clamp(ratio)*A)). Old log-probabilities and
// advantages are rollout targets; autograd attaches only to InNewLogProbability.
[[nodiscard]] OaMatrix PpoClippedPolicy(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	OaF32 InClipEpsilon = 0.2F
);

// Explicit gradient of PpoClippedPolicy with respect to the new
// log-probability.
[[nodiscard]] OaMatrix PpoClippedPolicyBwd(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	OaF32 InClipEpsilon = 0.2F
);

// PPO's differentiable loss bundle. Entropy is supplied by the policy
// distribution so this primitive remains independent of categorical/continuous
// action representation.
[[nodiscard]] OaPpoLossResult Ppo(
	const OaMatrix& InNewLogProbability,
	const OaMatrix& InOldLogProbability,
	const OaMatrix& InAdvantage,
	const OaMatrix& InValue,
	const OaMatrix& InTargetReturn,
	const OaMatrix& InEntropy,
	const OaPpoLossConfig& InConfig = {}
);

// One-step DQN temporal-difference objective. Termination suppresses the target
// bootstrap; truncation deliberately does not. InNextQ is target-network data
// and must not require gradients.
[[nodiscard]] OaDqnLossResult Dqn(
	const OaMatrix& InQ,
	const OaMatrix& InAction,
	const OaMatrix& InReward,
	const OaMatrix& InNextQ,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaDqnLossConfig& InConfig = {}
);

[[nodiscard]] OaSacCriticLossResult SacCritic(
	const OaMatrix& InQ1,
	const OaMatrix& InQ2,
	const OaMatrix& InReward,
	const OaMatrix& InNextQ1,
	const OaMatrix& InNextQ2,
	const OaMatrix& InNextLogProbability,
	const OaMatrix& InTerminated,
	const OaMatrix& InTruncated,
	const OaSacLossConfig& InConfig = {}
);

[[nodiscard]] OaMatrix SacActor(
	const OaMatrix& InQ1,
	const OaMatrix& InQ2,
	const OaMatrix& InLogProbability,
	OaF32 InEntropyCoefficient = 0.2F
);

} // namespace OaFnLoss
