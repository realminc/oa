#pragma once

#include <Oa/Core/Matrix.h>

/// Linear flow-matching state and its constant path velocity. Both matrices
/// remain on the active OA device and participate in the normal autograd graph.
struct OaFlowMatchBatch {
	OaMatrix State;
	OaMatrix Velocity;
};

namespace OaFnFlow {

/// Construct x(t) = clean + t * (noise - clean) and the training target
/// v(t) = noise - clean. InTime may be scalar, [B], or already broadcastable
/// to InClean. A [B] vector is expanded across every non-batch dimension.
[[nodiscard]] OaFlowMatchBatch LinearMatch(
	const OaMatrix& InClean,
	const OaMatrix& InNoise,
	const OaMatrix& InTime);

/// One explicit Euler integration step: x(t + dt) = x(t) + dt * v(t).
/// Sampling noise-to-data passes a negative InDeltaTime.
[[nodiscard]] OaMatrix EulerStep(
	const OaMatrix& InState,
	const OaMatrix& InVelocity,
	OaF32 InDeltaTime);

/// Mean squared error over only valid elements. InMask is a binary 0/1 mask
/// and may be scalar or broadcastable to InPrediction (for example [B,S,1]
/// for padded motion tokens). Padding therefore never changes the loss scale.
[[nodiscard]] OaMatrix MaskedMse(
	const OaMatrix& InPrediction,
	const OaMatrix& InTarget,
	const OaMatrix& InMask);

} // namespace OaFnFlow
