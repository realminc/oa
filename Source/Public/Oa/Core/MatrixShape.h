#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

#include <initializer_list>

constexpr OaI32 OA_MAX_TENSOR_DIMS = 8;

// OaMatrixShape — dimensions of an OaMatrix (OA's N-D array), rank up to
// OA_MAX_TENSOR_DIMS. Construct with brace-init for any rank:
//   OaMatrixShape{m, n}            // rank-2
//   OaMatrixShape{n, c, h, w}      // rank-4 (e.g. conv NCHW)
// (No per-rank helper functions — the initializer_list constructor covers every rank.)
struct OaMatrixShape {
	OaArray<OaI64, OA_MAX_TENSOR_DIMS> Dims = {};
	OaI32 Rank = 0;

	OaMatrixShape() = default;

	OaMatrixShape(std::initializer_list<OaI64> InDims) {
		Rank = static_cast<OaI32>(InDims.size());
		OaI32 idx = 0;
		for (auto dimVal : InDims) Dims[idx++] = dimVal;
	}

	[[nodiscard]] OaI64 operator[](OaI32 InDim) const { return Dims[InDim]; }
	OaI64& operator[](OaI32 InDim) { return Dims[InDim]; }

	[[nodiscard]] OaI64 NumElements() const {
		if (Rank == 0) return 0;
		OaI64 num = 1;
		for (OaI32 dimIdx = 0; dimIdx < Rank; ++dimIdx) num *= Dims[dimIdx];
		return num;
	}

	// Row-major: stride in elements for dimension InDim (product of later dims).
	[[nodiscard]] OaI64 Stride(OaI32 InDim) const {
		OaI64 step = 1;
		for (OaI32 dimIdx = InDim + 1; dimIdx < Rank; ++dimIdx) step *= Dims[dimIdx];
		return step;
	}

	[[nodiscard]] bool operator==(const OaMatrixShape& InOther) const {
		if (Rank != InOther.Rank) return false;
		for (OaI32 dimIdx = 0; dimIdx < Rank; ++dimIdx) {
			if (Dims[dimIdx] != InOther.Dims[dimIdx]) return false;
		}
		return true;
	}

	[[nodiscard]] bool operator!=(const OaMatrixShape& InOther) const { return !(*this == InOther); }

	// NumPy-style broadcast: align right, max dims, error if incompatible.
	[[nodiscard]] OaResult<OaMatrixShape> Broadcast(const OaMatrixShape& InOther) const;

	// Compute broadcast strides for this shape against a broadcasted output shape.
	// Returns row-major strides with 0 where this shape has a broadcasted dim (1 or absent).
	[[nodiscard]] OaStdArray<OaI64, OA_MAX_TENSOR_DIMS> BroadcastStrides(const OaMatrixShape& InOut) const;
};
