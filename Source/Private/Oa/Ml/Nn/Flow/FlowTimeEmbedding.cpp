#include <Oa/Ml/Nn/Flow/FlowTimeEmbedding.h>

#include <Oa/Core/FnMatrix.h>

#include <cmath>
#include <stdexcept>

OaFlowTimeEmbedding::OaFlowTimeEmbedding(
	OaI32 InEmbeddingDim,
	OaF32 InMaxPeriod,
	OaF32 InTimeScale)
	: EmbeddingDim_(InEmbeddingDim),
	  MaxPeriod_(InMaxPeriod),
	  TimeScale_(InTimeScale) {
	if (EmbeddingDim_ <= 0 || (EmbeddingDim_ % 2) != 0 ||
		!std::isfinite(MaxPeriod_) || MaxPeriod_ <= 1.0F ||
		!std::isfinite(TimeScale_) || TimeScale_ <= 0.0F) {
		throw std::invalid_argument("FlowTimeEmbedding requires positive even dimension, max period > 1 and positive scale");
	}

	const OaI32 half = EmbeddingDim_ / 2;
	OaVec<OaF32> frequencies(half);
	const OaF32 logPeriod = std::log(MaxPeriod_);
	for (OaI32 index = 0; index < half; ++index) {
		frequencies[index] = TimeScale_ * std::exp(
			-logPeriod * static_cast<OaF32>(index) / static_cast<OaF32>(half));
	}
	Frequencies_ = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(frequencies.Data()),
			static_cast<OaI64>(frequencies.Size()) * static_cast<OaI64>(sizeof(OaF32))),
		OaMatrixShape{1, half}, OaScalarType::Float32);
	RegisterBuffer("frequencies", Frequencies_, false);
}

OaMatrix OaFlowTimeEmbedding::Forward(const OaMatrix& InTime) {
	if (InTime.Rank() < 1 || InTime.Rank() > 2 ||
		(InTime.Rank() == 2 && InTime.Size(1) != 1)) {
		throw std::invalid_argument("FlowTimeEmbedding expects [B] or [B,1]");
	}
	const OaI64 batch = InTime.Size(0);
	auto time = InTime.Reshape(OaMatrixShape{batch, 1});
	if (time.GetDtype() != OaScalarType::Float32) {
		time = OaFnMatrix::Cast(time, OaScalarType::Float32);
	}
	auto phase = time * Frequencies_;
	auto sin = OaFnMatrix::Sin(phase);
	auto cos = OaFnMatrix::Cos(phase);
	OaMatrix parts[] = {sin, cos};
	return OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 2), 1);
}
