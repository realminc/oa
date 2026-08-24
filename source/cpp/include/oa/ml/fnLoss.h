#pragma once

#include <oa/core/matrix.h>
#include <oa/core/matrixShape.h>
#include <oa/core/status.h>

/// oa::FnLoss — loss function operations for neural networks.
/// Provides MSE, BCE, CrossEntropy, SmoothL1, L1 and related backward ops.
namespace oa {

struct PpoLossConfig {
	oa::F32 clipEpsilon = 0.2F;
	oa::F32 valueCoefficient = 0.5F;
	oa::F32 entropyCoefficient = 0.01F;
};

struct PpoLossResult {
	oa::Matrix policyLoss;
	oa::Matrix valueLoss;
	oa::Matrix entropy;
	oa::Matrix totalLoss;

	[[nodiscard]] bool isValid() const noexcept {
		return !policyLoss.isEmpty() && !valueLoss.isEmpty()
			&& !entropy.isEmpty() && !totalLoss.isEmpty();
	}
};

struct DqnLossConfig {
	oa::F32 discount = 0.99F;
};

struct DqnLossResult {
	oa::Matrix selectedQ;
	oa::Matrix targetQ;
	oa::Matrix loss;

	[[nodiscard]] bool isValid() const noexcept {
		return !selectedQ.isEmpty() && !targetQ.isEmpty() && !loss.isEmpty();
	}
};

struct SacLossConfig {
	oa::F32 discount = 0.99F;
	oa::F32 entropyCoefficient = 0.2F;
};

struct SacCriticLossResult {
	oa::Matrix targetQ;
	oa::Matrix q1Loss;
	oa::Matrix q2Loss;
	oa::Matrix totalLoss;

	[[nodiscard]] bool isValid() const noexcept {
		return !targetQ.isEmpty() && !q1Loss.isEmpty()
			&& !q2Loss.isEmpty() && !totalLoss.isEmpty();
	}
};

namespace FnLoss {

	// --- loss name tracking ---
	// Each loss function sets the last-called name on invocation.
	// MetricLoss can query this automatically to label the loss metric.

	/// Returns the name of the most recently called loss function, or nullptr.
	const char* lastName();

	/// Internal: called by each loss function to record its name.
	void setLastName(const char* inName);

	// --- loss functions ---

	// Schema-owned forward loss declarations.
	#include <oa/ml/fnloss/fnLoss.gen.h>

	[[nodiscard]] oa::Matrix ppoClippedPolicy(
		const oa::Matrix& inNewLogProbability,
		const oa::Matrix& inOldLogProbability,
		const oa::Matrix& inAdvantage,
		oa::F32 inClipEpsilon = 0.2F
	);

	[[nodiscard]] oa::Matrix ppoClippedPolicyBwd(
		const oa::Matrix& inNewLogProbability,
		const oa::Matrix& inOldLogProbability,
		const oa::Matrix& inAdvantage,
		oa::F32 inClipEpsilon = 0.2F
	);

	[[nodiscard]] PpoLossResult ppo(
		const oa::Matrix& inNewLogProbability,
		const oa::Matrix& inOldLogProbability,
		const oa::Matrix& inAdvantage,
		const oa::Matrix& inValue,
		const oa::Matrix& inTargetReturn,
		const oa::Matrix& inEntropy,
		const PpoLossConfig& inConfig = {}
	);

	[[nodiscard]] DqnLossResult dqn(
		const oa::Matrix& inQ,
		const oa::Matrix& inAction,
		const oa::Matrix& inReward,
		const oa::Matrix& inNextQ,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		const DqnLossConfig& inConfig = {}
	);

	[[nodiscard]] SacCriticLossResult sacCritic(
		const oa::Matrix& inQ1,
		const oa::Matrix& inQ2,
		const oa::Matrix& inReward,
		const oa::Matrix& inNextQ1,
		const oa::Matrix& inNextQ2,
		const oa::Matrix& inNextLogProbability,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		const SacLossConfig& inConfig = {}
	);

	[[nodiscard]] oa::Matrix sacActor(
		const oa::Matrix& inQ1,
		const oa::Matrix& inQ2,
		const oa::Matrix& inLogProbability,
		oa::F32 inEntropyCoefficient = 0.2F
	);

	/// crossEntropyBwd: gradient w.r.t. logits: (softmax(logits) - onehot(targets)) / batch
	/// @param inLogits:  [batch, classes] unnormalized logits
	/// @param inTargets: [batch] class indices (UInt8, UInt32, or non-negative Int32)
	[[nodiscard]] Matrix crossEntropyBwd(const Matrix& inLogits, const Matrix& inTargets);

	/// maskedCrossEntropy: cross-entropy over only rows where inMask is non-zero.
	[[nodiscard]] Matrix maskedCrossEntropy(
		const Matrix& inLogits, const Matrix& inTargets,
		const Matrix& inMask, oa::I32 inValidCount);

	/// maskedCrossEntropyBwd: backward for maskedCrossEntropy.
	[[nodiscard]] Matrix maskedCrossEntropyBwd(
		const Matrix& inLogits, const Matrix& inTargets,
		const Matrix& inMask, oa::I32 inValidCount);

	/// smoothL1: smooth L1 / Huber loss (beta=1.0). mean over all elements.
	[[nodiscard]] Matrix smoothL1(const Matrix& inA, const Matrix& inB);

	/// smoothL1Bwd: gradient w.r.t. inA.
	[[nodiscard]] Matrix smoothL1Bwd(const Matrix& inA, const Matrix& inB);

	/// mse: mean squared error. mean((a-b)^2).
	[[nodiscard]] Matrix mse(const Matrix& inA, const Matrix& inB);

	/// mseBwd: gradient w.r.t. inA: 2*(a-b)/N.
	[[nodiscard]] Matrix mseBwd(const Matrix& inA, const Matrix& inB);

	/// l1: mean absolute error. mean(|a-b|).
	[[nodiscard]] Matrix l1(const Matrix& inA, const Matrix& inB);

	/// l1Bwd: gradient w.r.t. inA: sign(a-b)/N.
	[[nodiscard]] Matrix l1Bwd(const Matrix& inA, const Matrix& inB);

	/// bce: binary cross-entropy. -(b*log(a) + (1-b)*log(1-a)), clamped for stability.
	[[nodiscard]] Matrix bce(const Matrix& inA, const Matrix& inB);

	/// bceBwd: gradient w.r.t. inA: (a-b)/(a*(1-a))/N.
	[[nodiscard]] Matrix bceBwd(const Matrix& inA, const Matrix& inB);
} // namespace FnLoss

} // namespace oa
