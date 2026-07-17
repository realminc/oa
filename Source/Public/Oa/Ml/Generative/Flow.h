#pragma once

#include <Oa/Ml/Module.h>

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
[[nodiscard]] OaFlowMatchBatch LinearMatch(const OaMatrix& InClean, const OaMatrix& InNoise, const OaMatrix& InTime);

/// One explicit Euler integration step: x(t + dt) = x(t) + dt * v(t).
/// Sampling noise-to-data passes a negative InDeltaTime.
[[nodiscard]] OaMatrix EulerStep(
	const OaMatrix& InState, const OaMatrix& InVelocity, OaF32 InDeltaTime);

/// Mean squared error over only valid elements. InMask is a binary 0/1 mask
/// and may be scalar or any
/// shape broadcastable to InPrediction (for example [B,S,1] for padded motion
/// tokens). The denominator is the number of broadcast-valid elements, so
/// padding never changes the loss scale. An all-zero mask returns zero without
/// a host readback or division by zero.
[[nodiscard]] OaMatrix MaskedMse(
	const OaMatrix& InPrediction,
	const OaMatrix& InTarget,
	const OaMatrix& InMask
);

} // namespace OaFnFlow

/// GPU sinusoidal embedding for normalized continuous time. Only the small,
/// deterministic frequency vector is uploaded at construction; every batch is
/// embedded by OA matrix kernels without a tensor-sized CPU loop or readback.
class OaFlowTimeEmbedding final : public OaModule {
public:
	OaFlowTimeEmbedding(OaI32 InEmbeddingDim, OaF32 InMaxPeriod = 10000.0F, OaF32 InTimeScale = 1000.0F);

	/// InTime is [B] or [B,1]; output is [B,EmbeddingDim].
	OaMatrix Forward(const OaMatrix& InTime) override;

	[[nodiscard]] OaI32 EmbeddingDim() const noexcept { return EmbeddingDim_; }
	[[nodiscard]] OaF32 MaxPeriod() const noexcept { return MaxPeriod_; }
	[[nodiscard]] OaF32 TimeScale() const noexcept { return TimeScale_; }

private:
	OaI32 EmbeddingDim_ = 0;
	OaF32 MaxPeriod_ = 10000.0F;
	OaF32 TimeScale_ = 1000.0F;
	OaMatrix Frequencies_;
};
