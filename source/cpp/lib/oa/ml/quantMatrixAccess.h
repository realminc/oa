#pragma once

#include <oa/ml/quantMatrix.h>

namespace oa {

// Private zero-allocation bridge used by quantization lowering and exact plane
// tests. Consumers operate on oa::QuantMatrix as one semantic weight value.
class QuantMatrixAccess {
public:
	[[nodiscard]] static oa::QuantMatrix make(oa::Matrix inPayload, oa::Matrix inScale,
											  oa::MatrixShape inShape,
											  oa::Quantization inQuantization) {
		oa::QuantMatrix value;
		value.payload_ = oa::move(inPayload);
		value.scale_ = oa::move(inScale);
		value.shape_ = inShape;
		value.quantization_ = inQuantization;
		return value;
	}

	[[nodiscard]] static const oa::Matrix& payload(const oa::QuantMatrix& inValue) noexcept {
		return inValue.payload_;
	}

	[[nodiscard]] static const oa::Matrix& scale(const oa::QuantMatrix& inValue) noexcept {
		return inValue.scale_;
	}
};

} // namespace oa
