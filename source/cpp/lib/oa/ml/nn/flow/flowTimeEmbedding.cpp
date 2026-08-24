#include <oa/ml/nn/flow/flowTimeEmbedding.h>

#include <oa/core/fnMatrix.h>

#include <cmath>
#include <stdexcept>

namespace oa {

FlowTimeEmbedding::FlowTimeEmbedding(
	oa::I32 inEmbeddingDim,
	oa::F32 inMaxPeriod,
	oa::F32 inTimeScale)
	: embeddingDim_(inEmbeddingDim),
	  maxPeriod_(inMaxPeriod),
	  timeScale_(inTimeScale) {
	if (embeddingDim_ <= 0 || (embeddingDim_ % 2) != 0 ||
		!std::isfinite(maxPeriod_) || maxPeriod_ <= 1.0F ||
		!std::isfinite(timeScale_) || timeScale_ <= 0.0F) {
		throw std::invalid_argument("FlowTimeEmbedding requires positive even dimension, max period > 1 and positive scale");
	}

	const oa::I32 half = embeddingDim_ / 2;
	oa::Vec<oa::F32> frequencies(half);
	const oa::F32 logPeriod = std::log(maxPeriod_);
	for (oa::I32 index = 0; index < half; ++index) {
		frequencies[index] = timeScale_ * std::exp(
			-logPeriod * static_cast<oa::F32>(index) / static_cast<oa::F32>(half));
	}
	frequencies_ = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(frequencies.data()),
			static_cast<oa::I64>(frequencies.size()) * static_cast<oa::I64>(sizeof(oa::F32))),
		oa::MatrixShape{1, half}, oa::ScalarType::Float32);
	registerBuffer("frequencies", frequencies_, false);
}

oa::Matrix FlowTimeEmbedding::forward(const oa::Matrix& inTime) {
	if (inTime.rank() < 1 || inTime.rank() > 2 ||
		(inTime.rank() == 2 && inTime.size(1) != 1)) {
		throw std::invalid_argument("FlowTimeEmbedding expects [B] or [B,1]");
	}
	const oa::I64 batch = inTime.size(0);
	auto time = inTime.reshape(oa::MatrixShape{batch, 1});
	if (time.getDtype() != oa::ScalarType::Float32) {
		time = oa::FnMatrix::cast(time, oa::ScalarType::Float32);
	}
	auto phase = time * frequencies_;
	auto sin = oa::FnMatrix::sin(phase);
	auto cos = oa::FnMatrix::cos(phase);
	oa::Matrix parts[] = {sin, cos};
	return oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 2), 1);
}

} // namespace oa
