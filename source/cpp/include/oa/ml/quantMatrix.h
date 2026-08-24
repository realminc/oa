#pragma once

#include <oa/core/matrix.h>
#include <oa/ml/type.h>

namespace oa {

class QuantMatrixAccess;

/// A GPU-resident weight stored in OA's native Q4 or Q8 block encoding.
///
/// Quantization is deliberately not an `oa::ScalarType`: one logical value owns
/// a packed integer payload and a separate Float32 scale plane. Treating Q4 as
/// a scalar dtype would make matrix byte sizes, strides, views, and element
/// access incorrect. The physical planes remain private and travel as one
/// weight value.
class QuantMatrix {
public:
	QuantMatrix() = default;

	[[nodiscard]] MatrixShape getShape() const noexcept { return shape_; }
	[[nodiscard]] oa::I32 rank() const noexcept { return shape_.rank; }
	[[nodiscard]] oa::I64 size(oa::I32 inDim) const {
		if (inDim < 0)
			inDim += shape_.rank;
		return shape_[inDim];
	}
	[[nodiscard]] oa::I64 numElements() const noexcept { return shape_.numElements(); }
	[[nodiscard]] oa::Quantization getQuantization() const noexcept { return quantization_; }
	[[nodiscard]] oa::Device getDevice() const noexcept { return payload_.getDevice(); }
	[[nodiscard]] bool isEmpty() const noexcept { return payload_.isEmpty() or scale_.isEmpty(); }

private:
	friend class QuantMatrixAccess;

	Matrix payload_;
	Matrix scale_;
	MatrixShape shape_{};
	oa::Quantization quantization_ = oa::Quantization::Q8;
};

} // namespace oa

// Quantized operations extend the canonical stateless oa::FnMatrix namespace.

namespace oa {

namespace FnMatrix {

// quantize Float32 storage into one native OA Q4 or Q8 weight value. The
// returned value retains the source shape; its physical planes remain private.
[[nodiscard]] oa::QuantMatrix quantize(const oa::Matrix& inInput, oa::Quantization inQuantization);

// Materialize a Float32 matrix with the logical shape retained by inInput.
[[nodiscard]] oa::Matrix dequantize(const oa::QuantMatrix& inInput);

// Multiply [..., K] Float32 activations by a quantized [N, K] weight without
// materializing a Float32 weight matrix. The result shape is [..., N].
[[nodiscard]] oa::Matrix matMulNt(const oa::Matrix& inInput, const oa::QuantMatrix& inWeight);

} // namespace FnMatrix

} // namespace oa
