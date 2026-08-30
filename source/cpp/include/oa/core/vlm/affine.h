#pragma once

// OA Vulkan linear-math affine operations.
//
// Spatial matrices use the fixed VLM row-vector convention. Consumers use
// these named operations instead of constructing or patching matrix elements.

#include <oa/core/std/assert.h>
#include <oa/core/vlm/math.h>

namespace oa::vlm {

namespace detail {

template <typename T>
struct TrsDecomposition {
	detail::Vec3<T> translation{};
	detail::Quat<T> rotation{};
	detail::Vec3<T> scale{T(1), T(1), T(1)};
	bool reflected = false;
};

template <typename T>
struct AffineDecomposition {
	detail::Vec3<T> translation{};
	detail::Quat<T> rotation{};
	detail::Vec3<T> scale{T(1), T(1), T(1)};
	// Dimensionless row-basis shear factors: XY, XZ, and YZ.
	detail::Vec3<T> shear{};
	bool reflected = false;
};

} // namespace detail

using TrsDecomposition = detail::TrsDecomposition<F32>;
using DTrsDecomposition = detail::TrsDecomposition<F64>;
using AffineDecomposition = detail::AffineDecomposition<F32>;
using DAffineDecomposition = detail::AffineDecomposition<F64>;

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Mat3<T>& inA,
	const detail::Mat3<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>) noexcept {
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			if (not approximatelyEqual(
				inA.m[row][column], inB.m[row][column],
				inAbsoluteTolerance, inRelativeTolerance)) {
				return false;
			}
		}
	}
	return true;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> transpose(
	const detail::Mat3<T>& inMatrix) noexcept {
	detail::Mat3<T> result = inMatrix;
	result.m[0][1] = inMatrix.m[1][0];
	result.m[0][2] = inMatrix.m[2][0];
	result.m[1][0] = inMatrix.m[0][1];
	result.m[1][2] = inMatrix.m[2][1];
	result.m[2][0] = inMatrix.m[0][2];
	result.m[2][1] = inMatrix.m[1][2];
	return result;
}

template <typename T>
[[nodiscard]] constexpr T determinant(
	const detail::Mat3<T>& inMatrix) noexcept {
	return inMatrix.m[0][0]
		* (inMatrix.m[1][1] * inMatrix.m[2][2]
			- inMatrix.m[1][2] * inMatrix.m[2][1])
		- inMatrix.m[0][1]
		* (inMatrix.m[1][0] * inMatrix.m[2][2]
			- inMatrix.m[1][2] * inMatrix.m[2][0])
		+ inMatrix.m[0][2]
		* (inMatrix.m[1][0] * inMatrix.m[2][1]
			- inMatrix.m[1][1] * inMatrix.m[2][0]);
}

template <typename T>
[[nodiscard]] bool tryInverse(
	const detail::Mat3<T>& inMatrix,
	detail::Mat3<T>& outInverse,
	T inTolerance = InverseTolerance<T>) noexcept {
	return detail::tryInverseMat3(inMatrix, outInverse, inTolerance);
}

template <typename T>
[[nodiscard]] constexpr detail::Mat3<T> linearPart(
	const detail::Mat4<T>& inMatrix) noexcept {
	detail::Mat3<T> result{};
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			result.m[row][column] = inMatrix.m[row][column];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> translationPart(
	const detail::Mat4<T>& inMatrix) noexcept {
	return {inMatrix.m[3][0], inMatrix.m[3][1], inMatrix.m[3][2]};
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> affineMatrix(
	const detail::Mat3<T>& inLinear,
	const detail::Vec3<T>& inTranslation = {}) noexcept {
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	for (I32 row = 0; row < 3; ++row) {
		for (I32 column = 0; column < 3; ++column) {
			result.m[row][column] = inLinear.m[row][column];
		}
	}
	result.m[3][0] = inTranslation.x;
	result.m[3][1] = inTranslation.y;
	result.m[3][2] = inTranslation.z;
	return result;
}

template <typename T>
[[nodiscard]] constexpr bool isAffine(
	const detail::Mat4<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	return inMatrix.isFinite()
		and approximatelyEqual(inMatrix.m[0][3], T(0), inTolerance, T(0))
		and approximatelyEqual(inMatrix.m[1][3], T(0), inTolerance, T(0))
		and approximatelyEqual(inMatrix.m[2][3], T(0), inTolerance, T(0))
		and approximatelyEqual(inMatrix.m[3][3], T(1), inTolerance, T(0));
}

template <typename T>
[[nodiscard]] constexpr bool isIdentity(
	const detail::Mat3<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	return approximatelyEqual(
		inMatrix, detail::Mat3<T>::identity(), inTolerance, inTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool isIdentity(
	const detail::Mat4<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	return approximatelyEqual(
		inMatrix, detail::Mat4<T>::identity(), inTolerance, inTolerance);
}

template <typename T>
[[nodiscard]] bool tryAffineInverse(
	const detail::Mat4<T>& inMatrix,
	detail::Mat4<T>& outInverse,
	T inTolerance = InverseTolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not isAffine(inMatrix, inTolerance)) return false;
	detail::Mat3<T> inverseLinear{};
	if (not tryInverse(linearPart(inMatrix), inverseLinear, inTolerance)) {
		return false;
	}
	const detail::Vec3<T> inverseTranslation = transform(
		-translationPart(inMatrix), inverseLinear);
	const detail::Mat4<T> inverse = affineMatrix(
		inverseLinear, inverseTranslation);
	if (not inverseTranslation.isFinite()) return false;
	outInverse = inverse;
	return true;
}

template <typename T>
[[nodiscard]] bool tryNormalMatrix(
	const detail::Mat4<T>& inMatrix,
	detail::Mat3<T>& outNormalMatrix,
	T inTolerance = InverseTolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not isAffine(inMatrix, inTolerance)) return false;
	detail::Mat3<T> inverse{};
	if (not tryInverse(linearPart(inMatrix), inverse, inTolerance)) return false;
	outNormalMatrix = transpose(inverse);
	return true;
}

template <typename T>
[[nodiscard]] bool isOrthogonal(
	const detail::Mat3<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	const detail::Vec3<T> x{
		inMatrix.m[0][0], inMatrix.m[0][1], inMatrix.m[0][2]};
	const detail::Vec3<T> y{
		inMatrix.m[1][0], inMatrix.m[1][1], inMatrix.m[1][2]};
	const detail::Vec3<T> z{
		inMatrix.m[2][0], inMatrix.m[2][1], inMatrix.m[2][2]};
	return x.isFinite() and y.isFinite() and z.isFinite()
		and approximatelyEqual(lengthSquared(x), T(1), inTolerance, inTolerance)
		and approximatelyEqual(lengthSquared(y), T(1), inTolerance, inTolerance)
		and approximatelyEqual(lengthSquared(z), T(1), inTolerance, inTolerance)
		and approximatelyEqual(dot(x, y), T(0), inTolerance, T(0))
		and approximatelyEqual(dot(x, z), T(0), inTolerance, T(0))
		and approximatelyEqual(dot(y, z), T(0), inTolerance, T(0));
}

template <typename T>
[[nodiscard]] bool isOrthonormal(
	const detail::Mat3<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	return isOrthogonal(inMatrix, inTolerance);
}

template <typename T>
[[nodiscard]] bool isProperRotation(
	const detail::Mat3<T>& inMatrix,
	T inTolerance = Tolerance<T>) noexcept {
	return isOrthonormal(inMatrix, inTolerance)
		and approximatelyEqual(
			determinant(inMatrix), T(1), inTolerance, inTolerance);
}

template <typename T>
[[nodiscard]] bool tryOrthonormalBasis(
	const detail::Vec3<T>& inFirst,
	const detail::Vec3<T>& inSecondCandidate,
	detail::Mat3<T>& outBasis,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not detail::hasLengthAboveTolerance(inFirst, inTolerance)) return false;
	detail::Vec3<T> first{};
	if (not tryNormalize(inFirst, first)) return false;
	detail::Vec3<T> rejected{};
	if (not tryRejectVector(
		inSecondCandidate, first, rejected, inTolerance)) {
		return false;
	}
	if (not detail::hasLengthAboveTolerance(rejected, inTolerance)) return false;
	detail::Vec3<T> second{};
	if (not tryNormalize(rejected, second)) return false;
	const detail::Vec3<T> third = cross(first, second);
	detail::Mat3<T> basis{};
	basis.m[0][0] = first.x;
	basis.m[0][1] = first.y;
	basis.m[0][2] = first.z;
	basis.m[1][0] = second.x;
	basis.m[1][1] = second.y;
	basis.m[1][2] = second.z;
	basis.m[2][0] = third.x;
	basis.m[2][1] = third.y;
	basis.m[2][2] = third.z;
	if (not isProperRotation(basis, inTolerance * T(8))) return false;
	outBasis = basis;
	return true;
}

template <typename T>
[[nodiscard]] bool tryQuaternionFromRotationMatrix(
	const detail::Mat3<T>& inMatrix,
	detail::Quat<T>& outRotation,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not isProperRotation(inMatrix, inTolerance)) return false;
	const detail::Quat<T> rotation = detail::quaternionFromRotationMatrixUnchecked(
		affineMatrix(inMatrix));
	if (not rotation.isFinite()) return false;
	outRotation = rotation;
	return true;
}

template <typename T>
[[nodiscard]] bool tryQuaternionFromRotationMatrix(
	const detail::Mat4<T>& inMatrix,
	detail::Quat<T>& outRotation,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not isAffine(inMatrix, inTolerance)) return false;
	return tryQuaternionFromRotationMatrix(
		linearPart(inMatrix), outRotation, inTolerance);
}

template <typename T>
[[nodiscard]] bool tryQuaternionFromTo(
	const detail::Vec3<T>& inFrom,
	const detail::Vec3<T>& inTo,
	detail::Quat<T>& outRotation,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not detail::hasLengthAboveTolerance(inFrom, inTolerance)
		or not detail::hasLengthAboveTolerance(inTo, inTolerance)) {
		return false;
	}
	detail::Vec3<T> from{};
	detail::Vec3<T> to{};
	if (not tryNormalize(inFrom, from) or not tryNormalize(inTo, to)) {
		return false;
	}
	const T cosine = oa::clamp(dot(from, to), T(-1), T(1));
	if (cosine >= T(1) - inTolerance) {
		outRotation = detail::Quat<T>::identity();
		return true;
	}
	if (cosine <= T(-1) + inTolerance) {
		detail::Vec3<T> axis{};
		if (not tryPerpendicular(from, axis, inTolerance)) return false;
		outRotation = detail::Quat<T>::fromAxisAngle(axis, Pi<T>);
		return true;
	}
	const detail::Vec3<T> axis = cross(from, to);
	const detail::Quat<T> rotation{
		axis.x, axis.y, axis.z, T(1) + cosine};
	if (not rotation.isFinite()) return false;
	outRotation = rotation.normalized();
	return true;
}

template <typename T>
[[nodiscard]] T quaternionAngle(
	const detail::Quat<T>& inRotation) noexcept {
	const detail::Quat<T> rotation = inRotation.normalized();
	return T(2) * oa::acos(oa::clamp(oa::abs(rotation.w), T(0), T(1)));
}

template <typename T>
[[nodiscard]] detail::Vec3<T> quaternionAxis(
	const detail::Quat<T>& inRotation,
	T inTolerance = Tolerance<T>) noexcept {
	OA_REQUIRE_MSG(
		isValidTolerance(inTolerance),
		"VLM tolerance must be finite and non-negative");
	detail::Quat<T> rotation = inRotation.normalized();
	if (rotation.w < T(0)) rotation = -rotation;
	const T sineSquared = T(1) - rotation.w * rotation.w;
	if (not oa::isFinite(sineSquared)
		or sineSquared <= inTolerance * inTolerance) {
		return {T(1), T(0), T(0)};
	}
	const T inverseSine = T(1) / oa::sqrt(sineSquared);
	return {
		rotation.x * inverseSine,
		rotation.y * inverseSine,
		rotation.z * inverseSine,
	};
}

template <typename T>
[[nodiscard]] bool tryTransformNormal(
	const detail::Vec3<T>& inNormal,
	const detail::Mat3<T>& inNormalMatrix,
	detail::Vec3<T>& outNormal,
	T inTolerance = InverseTolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const detail::Vec3<T> transformed = transform(inNormal, inNormalMatrix);
	if (not detail::hasLengthAboveTolerance(transformed, inTolerance)) {
		return false;
	}
	detail::Vec3<T> normalized{};
	if (not tryNormalize(transformed, normalized)) return false;
	outNormal = normalized;
	return true;
}

template <typename T>
[[nodiscard]] bool tryViewFromPose(
	const detail::Vec3<T>& inPosition,
	const detail::Quat<T>& inRotation,
	detail::Mat4<T>& outView,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	const T rotationMagnitude = oa::max(
		oa::max(oa::abs(inRotation.x), oa::abs(inRotation.y)),
		oa::max(oa::abs(inRotation.z), oa::abs(inRotation.w)));
	if (not inPosition.isFinite() or not inRotation.isFinite()
		or rotationMagnitude <= T(0)) {
		return false;
	}
	const detail::Mat4<T> view = matrixMul(
		translation(-inPosition),
		quaternionToMatrix(inRotation.conjugate()));
	if (not view.isFinite()) return false;
	outView = view;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> viewFromPose(
	const detail::Vec3<T>& inPosition,
	const detail::Quat<T>& inRotation) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryViewFromPose(inPosition, inRotation, result);
	OA_REQUIRE_MSG(valid, "VLM viewFromPose requires a finite non-degenerate pose");
	return result;
}

template <typename T>
[[nodiscard]] bool tryLookAt(
	const detail::Vec3<T>& inEye,
	const detail::Vec3<T>& inTarget,
	const detail::Vec3<T>& inWorldUp,
	detail::Mat4<T>& outView,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not inEye.isFinite() or not inTarget.isFinite()
		or not inWorldUp.isFinite()) {
		return false;
	}
	const detail::Vec3<T> forwardValue = inTarget - inEye;
	detail::Vec3<T> forward{};
	if (not detail::tryNormalizeAboveTolerance(
		forwardValue, forward, inTolerance)) return false;
	const detail::Vec3<T> rightValue = cross(forward, inWorldUp);
	detail::Vec3<T> right{};
	if (not detail::tryNormalizeAboveTolerance(
		rightValue, right, inTolerance)) return false;
	const detail::Vec3<T> up = cross(right, forward);
	detail::Mat3<T> basis{};
	basis.m[0][0] = right.x;
	basis.m[0][1] = up.x;
	basis.m[0][2] = -forward.x;
	basis.m[1][0] = right.y;
	basis.m[1][1] = up.y;
	basis.m[1][2] = -forward.y;
	basis.m[2][0] = right.z;
	basis.m[2][1] = up.z;
	basis.m[2][2] = -forward.z;
	const detail::Vec3<T> translationValue{
		-dot(right, inEye),
		-dot(up, inEye),
		dot(forward, inEye),
	};
	const detail::Mat4<T> view = affineMatrix(basis, translationValue);
	if (not view.isFinite()) return false;
	outView = view;
	return true;
}

template <typename T>
[[nodiscard]] bool tryLookRotation(
	const detail::Vec3<T>& inForward,
	const detail::Vec3<T>& inWorldUp,
	detail::Quat<T>& outRotation,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	detail::Mat4<T> view{};
	if (not tryLookAt(
		{}, inForward, inWorldUp, view, inTolerance)) {
		return false;
	}
	return tryQuaternionFromRotationMatrix(
		transpose(linearPart(view)), outRotation, inTolerance * T(8));
}

template <typename T>
[[nodiscard]] detail::Quat<T> lookRotation(
	const detail::Vec3<T>& inForward,
	const detail::Vec3<T>& inWorldUp) noexcept {
	detail::Quat<T> result{};
	const bool valid = tryLookRotation(inForward, inWorldUp, result);
	OA_REQUIRE_MSG(valid, "VLM lookRotation requires finite non-collinear directions");
	return result;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> composeAffine(
	const detail::AffineDecomposition<T>& inDecomposition) noexcept {
	const detail::Mat3<T> rotation = linearPart(
		quaternionToMatrix(inDecomposition.rotation));
	const detail::Vec3<T> x{
		rotation.m[0][0], rotation.m[0][1], rotation.m[0][2]};
	const detail::Vec3<T> y{
		rotation.m[1][0], rotation.m[1][1], rotation.m[1][2]};
	const detail::Vec3<T> z{
		rotation.m[2][0], rotation.m[2][1], rotation.m[2][2]};
	const detail::Vec3<T> rowX = x * inDecomposition.scale.x;
	const detail::Vec3<T> rowY =
		(y + x * inDecomposition.shear.x) * inDecomposition.scale.y;
	const detail::Vec3<T> rowZ =
		(z + x * inDecomposition.shear.y
			+ y * inDecomposition.shear.z) * inDecomposition.scale.z;
	detail::Mat3<T> linear{};
	linear.m[0][0] = rowX.x;
	linear.m[0][1] = rowX.y;
	linear.m[0][2] = rowX.z;
	linear.m[1][0] = rowY.x;
	linear.m[1][1] = rowY.y;
	linear.m[1][2] = rowY.z;
	linear.m[2][0] = rowZ.x;
	linear.m[2][1] = rowZ.y;
	linear.m[2][2] = rowZ.z;
	return affineMatrix(linear, inDecomposition.translation);
}

template <typename T>
[[nodiscard]] bool tryDecomposeAffine(
	const detail::Mat4<T>& inMatrix,
	detail::AffineDecomposition<T>& outDecomposition,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	if (not isAffine(inMatrix, inTolerance)) return false;
	const detail::Mat3<T> linear = linearPart(inMatrix);
	detail::Vec3<T> x{linear.m[0][0], linear.m[0][1], linear.m[0][2]};
	detail::Vec3<T> y{linear.m[1][0], linear.m[1][1], linear.m[1][2]};
	detail::Vec3<T> z{linear.m[2][0], linear.m[2][1], linear.m[2][2]};
	T scaleX = length(x);
	if (not oa::isFinite(scaleX) or scaleX <= inTolerance) {
		return false;
	}
	x /= scaleX;
	T shearXy = dot(x, y);
	y -= x * shearXy;
	const T scaleY = length(y);
	if (not oa::isFinite(scaleY) or scaleY <= inTolerance) {
		return false;
	}
	y /= scaleY;
	shearXy /= scaleY;

	T shearXz = dot(x, z);
	z -= x * shearXz;
	T shearYz = dot(y, z);
	z -= y * shearYz;
	const T scaleZ = length(z);
	if (not oa::isFinite(scaleZ) or scaleZ <= inTolerance) {
		return false;
	}
	z /= scaleZ;
	shearXz /= scaleZ;
	shearYz /= scaleZ;
	if (not oa::isFinite(shearXy) or not oa::isFinite(shearXz)
		or not oa::isFinite(shearYz)) {
		return false;
	}

	const T orientation = dot(cross(x, y), z);
	if (not oa::isFinite(orientation)
		or not approximatelyEqual(
			oa::abs(orientation), T(1), inTolerance * T(8), inTolerance * T(8))) {
		return false;
	}
	const bool reflected = orientation < T(0);
	if (reflected) {
		x = -x;
		scaleX = -scaleX;
		shearXy = -shearXy;
		shearXz = -shearXz;
	}
	detail::Mat3<T> rotationMatrix{};
	rotationMatrix.m[0][0] = x.x;
	rotationMatrix.m[0][1] = x.y;
	rotationMatrix.m[0][2] = x.z;
	rotationMatrix.m[1][0] = y.x;
	rotationMatrix.m[1][1] = y.y;
	rotationMatrix.m[1][2] = y.z;
	rotationMatrix.m[2][0] = z.x;
	rotationMatrix.m[2][1] = z.y;
	rotationMatrix.m[2][2] = z.z;
	detail::Quat<T> rotation{};
	if (not tryQuaternionFromRotationMatrix(
		rotationMatrix, rotation, inTolerance * T(8))) {
		return false;
	}
	detail::AffineDecomposition<T> result{};
	result.translation = translationPart(inMatrix);
	result.rotation = rotation;
	result.scale = {scaleX, scaleY, scaleZ};
	result.shear = {shearXy, shearXz, shearYz};
	result.reflected = reflected;
	outDecomposition = result;
	return true;
}

template <typename T>
[[nodiscard]] bool tryDecomposeTrs(
	const detail::Mat4<T>& inMatrix,
	detail::TrsDecomposition<T>& outDecomposition,
	T inTolerance = Tolerance<T>) noexcept {
	if (not isValidTolerance(inTolerance)) return false;
	detail::AffineDecomposition<T> affine{};
	if (not tryDecomposeAffine(inMatrix, affine, inTolerance)) return false;
	if (oa::abs(affine.shear.x) > inTolerance * T(8)
		or oa::abs(affine.shear.y) > inTolerance * T(8)
		or oa::abs(affine.shear.z) > inTolerance * T(8)) {
		return false;
	}
	detail::TrsDecomposition<T> result{};
	result.translation = affine.translation;
	result.rotation = affine.rotation;
	result.scale = affine.scale;
	result.reflected = affine.reflected;
	outDecomposition = result;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> lookAt(
	const detail::Vec3<T>& inEye,
	const detail::Vec3<T>& inTarget,
	const detail::Vec3<T>& inWorldUp) noexcept {
	detail::Mat4<T> result{};
	const bool valid = tryLookAt(inEye, inTarget, inWorldUp, result);
	OA_REQUIRE_MSG(valid, "VLM lookAt requires a finite non-degenerate camera basis");
	return result;
}

} // namespace oa::vlm
