#pragma once

#include <oa/core/matrix.h>

namespace oa {

/// Linear flow-matching state and its constant path velocity. Both matrices
/// remain on the active OA device and participate in the normal autograd graph.
struct FlowMatchBatch {
	oa::Matrix state;
	oa::Matrix velocity;
};

namespace FnFlow {

/// Construct x(t) = clean + t * (noise - clean) and the training target
/// v(t) = noise - clean. inTime may be scalar, [B], or already broadcastable
/// to inClean. A [B] vector is expanded across every non-batch dimension.
[[nodiscard]] FlowMatchBatch linearMatch(
	const oa::Matrix& inClean,
	const oa::Matrix& inNoise,
	const oa::Matrix& inTime
);

/// One explicit Euler integration step: x(t + dt) = x(t) + dt * v(t).
/// Sampling noise-to-data passes a negative inDeltaTime.
[[nodiscard]] oa::Matrix eulerStep(
	const oa::Matrix& inState,
	const oa::Matrix& inVelocity,
	oa::F32 inDeltaTime
);

/// Mean squared error over only valid elements. inMask is a binary 0/1 mask
/// and may be scalar or broadcastable to inPrediction (for example [B,S,1]
/// for padded motion tokens). Padding therefore never changes the loss scale.
[[nodiscard]] oa::Matrix maskedMse(
	const oa::Matrix& inPrediction,
	const oa::Matrix& inTarget,
	const oa::Matrix& inMask
);

} // namespace FnFlow

} // namespace oa
