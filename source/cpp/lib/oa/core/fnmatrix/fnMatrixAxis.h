#pragma once

#include <oa/core/matrix.h>
#include <oa/core/types.h>

#include <limits>

struct FnMatrixAxisShape {
	oa::U32 outerSize = 0U;
	oa::U32 dimSize = 0U;
	oa::U32 innerSize = 0U;

	[[nodiscard]] oa::U32 groupCount() const noexcept {
		return outerSize * innerSize;
	}
};

[[nodiscard]] inline bool resolveFnMatrixAxis(
	const oa::Matrix& inMatrix, oa::I32 inRequestedDim,
	FnMatrixAxisShape& outShape) noexcept
{
	const oa::I32 rank = inMatrix.rank();
	const oa::I32 dim = inRequestedDim == -1 ? rank - 1 : inRequestedDim;
	if (rank <= 0 or inRequestedDim < -1 or dim < 0 or dim >= rank) return false;

	constexpr oa::U64 maxU32 = std::numeric_limits<oa::U32>::max();
	oa::U64 outerSize = 1U;
	oa::U64 innerSize = 1U;
	for (oa::I32 axis = 0; axis < rank; ++axis) {
		const oa::I64 extent = inMatrix.size(axis);
		if (extent <= 0 or static_cast<oa::U64>(extent) > maxU32) return false;
		if (axis < dim) {
			if (outerSize > maxU32 / static_cast<oa::U64>(extent)) return false;
			outerSize *= static_cast<oa::U64>(extent);
		} else if (axis > dim) {
			if (innerSize > maxU32 / static_cast<oa::U64>(extent)) return false;
			innerSize *= static_cast<oa::U64>(extent);
		}
	}
	const oa::U64 dimSize = static_cast<oa::U64>(inMatrix.size(dim));
	if (outerSize > maxU32 / dimSize) return false;
	if (outerSize * dimSize > maxU32 / innerSize) return false;

	outShape = {
		static_cast<oa::U32>(outerSize),
		static_cast<oa::U32>(dimSize),
		static_cast<oa::U32>(innerSize),
	};
	return true;
}
