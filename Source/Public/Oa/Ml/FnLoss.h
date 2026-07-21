#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Status.h>

/// OaFnLoss — Loss function operations for neural networks.
/// Provides common loss functions (MSE, BCE, CrossEntropy, etc.) through
/// the clean API lvl1 path. Calls record into OaContext and execute with
/// the rest of the current compute graph.
namespace OaFnLoss {

// ─── Loss Name Tracking ───────────────────────────────────────────
//
// Each loss function sets the last-called name on invocation.
// OaMetricLoss can query this automatically to display "cross_entropy"
// or "mse" instead of the generic "loss" label.

/// Returns the name of the most recently called loss function,
// or nullptr if none has been called in this thread.
const char* LastName();

/// Internal: called by each loss function to record its name.
void SetLastName(const char* InName);

// ─── Loss Functions ───────────────────────────────────────────────

// Schema-owned forward loss declarations.
#include "../../../Private/Oa/Ml/FnLoss/FnLoss.gen.h"

/// CrossEntropyBwd: Backward pass for cross-entropy loss.
/// Computes gradient w.r.t. logits: (softmax(Logits) - onehot(Targets)) / batch
/// @param InLogits: [batch, classes] unnormalized logits (forward input)
/// @param InTargets: [batch] class indices (UInt8, UInt32, or non-negative Int32)
/// @return [batch, classes] gradient w.r.t. logits
[[nodiscard]] OaMatrix CrossEntropyBwd(const OaMatrix& InLogits, const OaMatrix& InTargets);

/// Cross-entropy over only rows whose floating mask is non-zero. Padded rows
/// contribute neither loss nor gradient; normalization uses InValidCount.
[[nodiscard]] OaMatrix MaskedCrossEntropy(
	const OaMatrix& InLogits, const OaMatrix& InTargets,
	const OaMatrix& InMask, OaI32 InValidCount);

/// Backward pass for MaskedCrossEntropy.
[[nodiscard]] OaMatrix MaskedCrossEntropyBwd(
	const OaMatrix& InLogits, const OaMatrix& InTargets,
	const OaMatrix& InMask, OaI32 InValidCount);

// CrossEntropyLossGradBwd fused public wrapper removed (OaModule.md Phase 1).
// The CrossEntropyLossGradBwd kernel remains in the registry for Api3-style
// hand-wired graphs; the public C++ surface is forward + Bwd only.

/// SmoothL1: Smooth L1 / Huber loss (beta=1.0).
/// For |A-B| < 1: 0.5*(A-B)^2, else |A-B|-0.5. Mean over all elements.
/// @param InA: predictions (any shape)
/// @param InB: targets (same shape as InA)
/// @return Scalar loss value
[[nodiscard]] OaMatrix SmoothL1(const OaMatrix& InA, const OaMatrix& InB);

/// SmoothL1Bwd: Backward pass for smooth L1 loss.
/// Gradient w.r.t. InA: (A-B)/N if |A-B|<1, else sign(A-B)/N.
/// @param InA: predictions (forward input)
/// @param InB: targets (forward input)
/// @return Gradient w.r.t. InA (same shape as InA)
[[nodiscard]] OaMatrix SmoothL1Bwd(const OaMatrix& InA, const OaMatrix& InB);

/// Mse: Mean squared error loss. mean((A-B)^2).
/// @param InA: predictions (any shape)
/// @param InB: targets (same shape as InA)
/// @return Scalar loss value
[[nodiscard]] OaMatrix Mse(const OaMatrix& InA, const OaMatrix& InB);

/// MseBwd: Backward pass for MSE loss. Gradient w.r.t. InA: 2*(A-B)/N.
[[nodiscard]] OaMatrix MseBwd(const OaMatrix& InA, const OaMatrix& InB);

/// L1: Mean absolute error loss. mean(|A-B|).
/// @param InA: predictions (any shape)
/// @param InB: targets (same shape as InA)
/// @return Scalar loss value
[[nodiscard]] OaMatrix L1(const OaMatrix& InA, const OaMatrix& InB);

/// L1Bwd: Backward pass for L1 loss. Gradient w.r.t. InA: sign(A-B)/N.
[[nodiscard]] OaMatrix L1Bwd(const OaMatrix& InA, const OaMatrix& InB);

/// Bce: Binary cross-entropy loss. -(B*log(A) + (1-B)*log(1-A)), clamped for stability.
/// @param InA: predicted probabilities (any shape, values in (0,1))
/// @param InB: binary targets (same shape as InA)
/// @return Scalar loss value
[[nodiscard]] OaMatrix Bce(const OaMatrix& InA, const OaMatrix& InB);

/// BceBwd: Backward pass for BCE loss. Gradient w.r.t. InA: (A-B)/(A*(1-A))/N.
[[nodiscard]] OaMatrix BceBwd(const OaMatrix& InA, const OaMatrix& InB);

// ─── Loss Operations ──────────────────────────────────────────────
// Migrated forwards are schema-owned; operation-specific lowering and backward
// paths remain private implementations selected below the public contract.

} // namespace OaFnLoss
