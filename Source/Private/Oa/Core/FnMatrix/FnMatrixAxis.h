#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Types.h>

#include <limits>

struct OaFnMatrixAxisShape {
	OaU32 OuterSize = 0U;
	OaU32 DimSize = 0U;
	OaU32 InnerSize = 0U;

	[[nodiscard]] OaU32 GroupCount() const noexcept {
		return OuterSize * InnerSize;
	}
};

[[nodiscard]] inline bool OaResolveFnMatrixAxis(
	const OaMatrix& InMatrix, OaI32 InRequestedDim,
	OaFnMatrixAxisShape& OutShape) noexcept
{
	const OaI32 rank = InMatrix.Rank();
	const OaI32 dim = InRequestedDim == -1 ? rank - 1 : InRequestedDim;
	if (rank <= 0 or InRequestedDim < -1 or dim < 0 or dim >= rank) return false;

	constexpr OaU64 maxU32 = std::numeric_limits<OaU32>::max();
	OaU64 outerSize = 1U;
	OaU64 innerSize = 1U;
	for (OaI32 axis = 0; axis < rank; ++axis) {
		const OaI64 extent = InMatrix.Size(axis);
		if (extent <= 0 or static_cast<OaU64>(extent) > maxU32) return false;
		if (axis < dim) {
			if (outerSize > maxU32 / static_cast<OaU64>(extent)) return false;
			outerSize *= static_cast<OaU64>(extent);
		} else if (axis > dim) {
			if (innerSize > maxU32 / static_cast<OaU64>(extent)) return false;
			innerSize *= static_cast<OaU64>(extent);
		}
	}
	const OaU64 dimSize = static_cast<OaU64>(InMatrix.Size(dim));
	if (outerSize > maxU32 / dimSize) return false;
	if (outerSize * dimSize > maxU32 / innerSize) return false;

	OutShape = {
		static_cast<OaU32>(outerSize),
		static_cast<OaU32>(dimSize),
		static_cast<OaU32>(innerSize),
	};
	return true;
}
