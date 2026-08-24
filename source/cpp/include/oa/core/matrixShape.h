#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

#include <initializer_list>
#include <stdexcept>

constexpr oa::I32 OA_MAX_TENSOR_DIMS = 8;

namespace oa {

// MatrixShape — dimensions of an oa::Matrix (OA's N-D array), rank up to
// OA_MAX_TENSOR_DIMS. Construct with brace-init for any rank:
//   MatrixShape{m, n}            // rank-2
//   MatrixShape{n, c, h, w}      // rank-4 (e.g. conv NCHW)
// (No per-rank helper functions — the initializer_list constructor covers every rank.)
struct MatrixShape {
	oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> dims = {};
	oa::I32 rank = 0;

	MatrixShape() = default;

	MatrixShape(std::initializer_list<oa::I64> inDims) {
		if (inDims.size() > static_cast<oa::Usize>(OA_MAX_TENSOR_DIMS)) {
			throw std::length_error("MatrixShape rank exceeds OA_MAX_TENSOR_DIMS");
		}
		rank = static_cast<oa::I32>(inDims.size());
		oa::Usize idx = 0;
		for (auto dimVal : inDims) dims[idx++] = dimVal;
	}

	[[nodiscard]] oa::I64 operator[](oa::I32 inDim) const {
		return dims[static_cast<oa::Usize>(inDim)];
	}
	oa::I64& operator[](oa::I32 inDim) {
		return dims[static_cast<oa::Usize>(inDim)];
	}

	[[nodiscard]] oa::I64 numElements() const {
		if (rank == 0) {
			return 0;
		}
		oa::I64 num = 1;
		for (oa::I32 dimIdx = 0; dimIdx < rank; ++dimIdx) {
			num *= dims[static_cast<oa::Usize>(dimIdx)];
		}
		return num;
	}

	// Row-major: stride in elements for dimension inDim (product of later dims).
	[[nodiscard]] oa::I64 stride(oa::I32 inDim) const {
		oa::I64 step = 1;
		for (oa::I32 dimIdx = inDim + 1; dimIdx < rank; ++dimIdx) {
			step *= dims[static_cast<oa::Usize>(dimIdx)];
		}
		return step;
	}

	[[nodiscard]] bool operator==(const MatrixShape& inOther) const {
		if (rank != inOther.rank) {
			return false;
		}
		for (oa::I32 dimIdx = 0; dimIdx < rank; ++dimIdx) {
			const oa::Usize idx = static_cast<oa::Usize>(dimIdx);
			if (dims[idx] != inOther.dims[idx]) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] bool operator!=(const MatrixShape& inOther) const { return !(*this == inOther); }

	// NumPy-style broadcast: align right, max dims, error if incompatible.
	[[nodiscard]] oa::Result<MatrixShape> broadcast(const MatrixShape& inOther) const;

	// Compute broadcast strides for this shape against a broadcasted output shape.
	// Returns row-major strides with 0 where this shape has a broadcasted dim (1 or absent).
	[[nodiscard]] Array<oa::I64, OA_MAX_TENSOR_DIMS> broadcastStrides(const MatrixShape& inOut) const;
};

} // namespace oa
