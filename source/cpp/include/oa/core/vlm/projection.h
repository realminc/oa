#pragma once

// OA Vulkan linear-math projection and top-origin viewport operations.

#include <oa/core/std/assert.h>
#include <oa/core/vlm/affine.h>

namespace oa::vlm {

namespace detail {

template <typename T>
struct Viewport {
	static_assert(oa::IsFloatingPointV<T>);

	T x = T(0);
	T y = T(0);
	T width = T(1);
	T height = T(1);
	T minDepth = T(0);
	T maxDepth = T(1);

	[[nodiscard]] bool isValid() const noexcept {
		return oa::isFinite(x) and oa::isFinite(y)
			and oa::isFinite(width) and oa::isFinite(height)
			and oa::isFinite(minDepth) and oa::isFinite(maxDepth)
			and width > T(0) and height > T(0)
			and minDepth >= T(0) and maxDepth <= T(1)
			and minDepth < maxDepth;
	}
};

} // namespace detail

using Viewport = detail::Viewport<F32>;
using DViewport = detail::Viewport<F64>;

template <typename T>
[[nodiscard]] bool tryPerspectiveOffCenter(
	T inLeft,
	T inRight,
	T inBottom,
	T inTop,
	T inNear,
	T inFar,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inLeft) or not oa::isFinite(inRight)
		or not oa::isFinite(inBottom) or not oa::isFinite(inTop)
		or not oa::isFinite(inNear) or not oa::isFinite(inFar)
		or inRight <= inLeft or inTop <= inBottom
		or inNear <= T(0) or inFar <= inNear) {
		return false;
	}
	const T width = inRight - inLeft;
	const T height = inTop - inBottom;
	const T depth = inNear - inFar;
	if (not oa::isFinite(width) or not oa::isFinite(height)
		or not oa::isFinite(depth)) {
		return false;
	}
	const T inverseWidth = T(1) / width;
	const T inverseHeight = T(1) / height;
	const T inverseDepth = T(1) / depth;
	detail::Mat4<T> result{};
	result.m[0][0] = T(2) * inNear * inverseWidth;
	result.m[1][1] = T(2) * inNear * inverseHeight;
	result.m[2][0] = (inRight + inLeft) * inverseWidth;
	result.m[2][1] = (inTop + inBottom) * inverseHeight;
	result.m[2][2] = inFar * inverseDepth;
	result.m[2][3] = T(-1);
	result.m[3][2] = inFar * inNear * inverseDepth;
	if (not result.isFinite()) return false;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspectiveOffCenter(
	T inLeft,
	T inRight,
	T inBottom,
	T inTop,
	T inNear,
	T inFar) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryPerspectiveOffCenter(
		inLeft, inRight, inBottom, inTop, inNear, inFar, result);
	OA_REQUIRE_MSG(valid, "VLM off-center perspective requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryPerspective(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inFovYDeg) or not oa::isFinite(inAspect)
		or inFovYDeg <= T(0) or inFovYDeg >= T(180)
		or inAspect <= T(0) or inNear <= T(0) or inFar <= inNear
		or not oa::isFinite(inNear) or not oa::isFinite(inFar)) {
		return false;
	}
	const T halfHeight = inNear * oa::tan(radians(inFovYDeg) * T(0.5));
	if (not oa::isFinite(halfHeight) or halfHeight <= T(0)) return false;
	const T halfWidth = halfHeight * inAspect;
	return tryPerspectiveOffCenter(
		-halfWidth, halfWidth, -halfHeight, halfHeight,
		inNear, inFar, outProjection);
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspective(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryPerspective(
		inFovYDeg, inAspect, inNear, inFar, result);
	OA_REQUIRE_MSG(valid, "VLM perspective requires finite valid Vulkan projection parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryPerspectiveShifted(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar,
	const detail::Vec2<T>& inNdcOffset,
	detail::Mat4<T>& outProjection) noexcept {
	if (not inNdcOffset.isFinite()) return false;
	detail::Mat4<T> result{};
	if (not tryPerspective(inFovYDeg, inAspect, inNear, inFar, result)) {
		return false;
	}
	result.m[2][0] = -inNdcOffset.x;
	result.m[2][1] = -inNdcOffset.y;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspectiveShifted(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar,
	const detail::Vec2<T>& inNdcOffset) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryPerspectiveShifted(
		inFovYDeg, inAspect, inNear, inFar, inNdcOffset, result);
	OA_REQUIRE_MSG(valid, "VLM shifted perspective requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryOrthographicOffCenter(
	T inLeft,
	T inRight,
	T inBottom,
	T inTop,
	T inNear,
	T inFar,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inLeft) or not oa::isFinite(inRight)
		or not oa::isFinite(inBottom) or not oa::isFinite(inTop)
		or not oa::isFinite(inNear) or not oa::isFinite(inFar)
		or inRight <= inLeft or inTop <= inBottom or inFar <= inNear) {
		return false;
	}
	const T width = inRight - inLeft;
	const T height = inTop - inBottom;
	const T depth = inFar - inNear;
	if (not oa::isFinite(width) or not oa::isFinite(height)
		or not oa::isFinite(depth)) {
		return false;
	}
	const T inverseWidth = T(1) / width;
	const T inverseHeight = T(1) / height;
	const T inverseDepth = T(1) / depth;
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[0][0] = T(2) * inverseWidth;
	result.m[1][1] = T(2) * inverseHeight;
	result.m[2][2] = -inverseDepth;
	result.m[3][0] = -(inRight + inLeft) * inverseWidth;
	result.m[3][1] = -(inTop + inBottom) * inverseHeight;
	result.m[3][2] = -inNear * inverseDepth;
	if (not result.isFinite()) return false;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> orthographicOffCenter(
	T inLeft,
	T inRight,
	T inBottom,
	T inTop,
	T inNear,
	T inFar) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryOrthographicOffCenter(
		inLeft, inRight, inBottom, inTop, inNear, inFar, result);
	OA_REQUIRE_MSG(valid, "VLM off-center orthographic requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryOrthographic(
	T inWidth,
	T inHeight,
	T inNear,
	T inFar,
	T inZoom,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inWidth) or not oa::isFinite(inHeight)
		or not oa::isFinite(inZoom) or inWidth <= T(0)
		or inHeight <= T(0) or inZoom <= T(0)) {
		return false;
	}
	const T halfWidth = inWidth * T(0.5) / inZoom;
	const T halfHeight = inHeight * T(0.5) / inZoom;
	return tryOrthographicOffCenter(
		-halfWidth, halfWidth, -halfHeight, halfHeight,
		inNear, inFar, outProjection);
}

template <typename T>
[[nodiscard]] detail::Mat4<T> orthographic(
	T inWidth,
	T inHeight,
	T inNear,
	T inFar,
	T inZoom = T(1)) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryOrthographic(
		inWidth, inHeight, inNear, inFar, inZoom, result);
	OA_REQUIRE_MSG(valid, "VLM orthographic requires finite valid Vulkan projection parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryOrthographicShifted(
	T inWidth,
	T inHeight,
	T inNear,
	T inFar,
	T inZoom,
	const detail::Vec2<T>& inNdcOffset,
	detail::Mat4<T>& outProjection) noexcept {
	if (not inNdcOffset.isFinite()) return false;
	detail::Mat4<T> result{};
	if (not tryOrthographic(
		inWidth, inHeight, inNear, inFar, inZoom, result)) {
		return false;
	}
	result.m[3][0] = -inNdcOffset.x;
	result.m[3][1] = -inNdcOffset.y;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> orthographicShifted(
	T inWidth,
	T inHeight,
	T inNear,
	T inFar,
	T inZoom,
	const detail::Vec2<T>& inNdcOffset) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryOrthographicShifted(
		inWidth, inHeight, inNear, inFar, inZoom, inNdcOffset, result);
	OA_REQUIRE_MSG(valid, "VLM shifted orthographic requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryPerspectiveReverseZ(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inFovYDeg) or not oa::isFinite(inAspect)
		or not oa::isFinite(inNear) or not oa::isFinite(inFar)
		or inFovYDeg <= T(0) or inFovYDeg >= T(180)
		or inAspect <= T(0) or inNear <= T(0) or inFar <= inNear) {
		return false;
	}
	const T tangent = oa::tan(radians(inFovYDeg) * T(0.5));
	if (not oa::isFinite(tangent) or tangent <= T(0)) return false;
	detail::Mat4<T> result{};
	result.m[0][0] = T(1) / (inAspect * tangent);
	result.m[1][1] = T(1) / tangent;
	result.m[2][2] = inNear / (inFar - inNear);
	result.m[2][3] = T(-1);
	result.m[3][2] = (inFar * inNear) / (inFar - inNear);
	if (not result.isFinite()) return false;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspectiveReverseZ(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryPerspectiveReverseZ(
		inFovYDeg, inAspect, inNear, inFar, result);
	OA_REQUIRE_MSG(valid, "VLM reverse-Z perspective requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryPerspectiveReverseZInfinite(
	T inFovYDeg,
	T inAspect,
	T inNear,
	detail::Mat4<T>& outProjection) noexcept {
	if (not oa::isFinite(inFovYDeg) or not oa::isFinite(inAspect)
		or not oa::isFinite(inNear) or inFovYDeg <= T(0)
		or inFovYDeg >= T(180) or inAspect <= T(0) or inNear <= T(0)) {
		return false;
	}
	const T tangent = oa::tan(radians(inFovYDeg) * T(0.5));
	if (not oa::isFinite(tangent) or tangent <= T(0)) return false;
	detail::Mat4<T> result{};
	result.m[0][0] = T(1) / (inAspect * tangent);
	result.m[1][1] = T(1) / tangent;
	result.m[2][3] = T(-1);
	result.m[3][2] = inNear;
	if (not result.isFinite()) return false;
	outProjection = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspectiveReverseZInfinite(
	T inFovYDeg,
	T inAspect,
	T inNear) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryPerspectiveReverseZInfinite(
		inFovYDeg, inAspect, inNear, result);
	OA_REQUIRE_MSG(valid, "VLM infinite reverse-Z perspective requires finite valid parameters");
	return result;
}

template <typename T>
[[nodiscard]] bool tryProjectToViewport(
	const detail::Vec3<T>& inPoint,
	const detail::Mat4<T>& inWorldViewProjection,
	const detail::Viewport<T>& inViewport,
	detail::Vec3<T>& outViewportPoint,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not inViewport.isValid()) return false;
	detail::Vec3<T> ndc{};
	if (not tryProjectPoint(
		inPoint, inWorldViewProjection, ndc, inTolerance)) {
		return false;
	}
	const detail::Vec3<T> viewportPoint{
		inViewport.x + (ndc.x + T(1)) * T(0.5) * inViewport.width,
		inViewport.y + (T(1) - ndc.y) * T(0.5) * inViewport.height,
		inViewport.minDepth
			+ ndc.z * (inViewport.maxDepth - inViewport.minDepth),
	};
	if (not viewportPoint.isFinite()) return false;
	outViewportPoint = viewportPoint;
	return true;
}

template <typename T>
[[nodiscard]] bool tryUnprojectFromViewport(
	const detail::Vec3<T>& inViewportPoint,
	const detail::Mat4<T>& inWorldViewProjection,
	const detail::Viewport<T>& inViewport,
	detail::Vec3<T>& outPoint,
	T inTolerance = InverseTolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not inViewport.isValid() or not inViewportPoint.isFinite()) return false;
	const T depthRange = inViewport.maxDepth - inViewport.minDepth;
	const detail::Vec3<T> ndc{
		T(2) * (inViewportPoint.x - inViewport.x) / inViewport.width - T(1),
		T(1) - T(2) * (inViewportPoint.y - inViewport.y) / inViewport.height,
		(inViewportPoint.z - inViewport.minDepth) / depthRange,
	};
	detail::Mat4<T> inverse{};
	if (not tryInverse(inWorldViewProjection, inverse, inTolerance)) return false;
	detail::Vec3<T> point{};
	if (not tryProjectPoint(ndc, inverse, point, inTolerance)) return false;
	outPoint = point;
	return true;
}

} // namespace oa::vlm
