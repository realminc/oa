#pragma once

#include <oa/ml/module.h>

namespace oa {

/// GPU sinusoidal embedding for normalized continuous time. Only the small,
/// deterministic frequency vector is uploaded at construction; every batch is
/// embedded by OA matrix kernels without a tensor-sized CPU loop or readback.
class FlowTimeEmbedding final : public oa::Module {
public:
	FlowTimeEmbedding(
		oa::I32 inEmbeddingDim,
		oa::F32 inMaxPeriod = 10000.0F,
		oa::F32 inTimeScale = 1000.0F);

	/// inTime is [B] or [B,1]; output is [B,EmbeddingDim].
	oa::Matrix forward(const oa::Matrix& inTime) override;

	[[nodiscard]] oa::I32 embeddingDim() const noexcept { return embeddingDim_; }
	[[nodiscard]] oa::F32 maxPeriod() const noexcept { return maxPeriod_; }
	[[nodiscard]] oa::F32 timeScale() const noexcept { return timeScale_; }

private:
	oa::I32 embeddingDim_ = 0;
	oa::F32 maxPeriod_ = 10000.0F;
	oa::F32 timeScale_ = 1000.0F;
	oa::Matrix frequencies_;
};

} // namespace oa
