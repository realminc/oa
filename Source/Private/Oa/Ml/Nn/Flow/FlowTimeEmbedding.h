#pragma once

#include <Oa/Ml/Module.h>

/// GPU sinusoidal embedding for normalized continuous time. Only the small,
/// deterministic frequency vector is uploaded at construction; every batch is
/// embedded by OA matrix kernels without a tensor-sized CPU loop or readback.
class OaFlowTimeEmbedding final : public OaModule {
public:
	OaFlowTimeEmbedding(
		OaI32 InEmbeddingDim,
		OaF32 InMaxPeriod = 10000.0F,
		OaF32 InTimeScale = 1000.0F);

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
