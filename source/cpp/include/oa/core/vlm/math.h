#pragma once

#include <oa/core/vlm/matrix.h>

#include <oa/core/std/algo.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

namespace oa {

namespace vlm {

template <typename T>
inline constexpr T Pi = T(3.141592653589793238462643383279502884L);

template <typename T>
inline constexpr T Tolerance = oa::IsSameV<T, F32> ? T(1.0e-5F) : T(1.0e-12);

template <typename T>
inline constexpr T InverseTolerance = oa::Limits<T>::epsilon() * T(32);

template <typename T>
[[nodiscard]] constexpr T radians(T inDegrees) noexcept {
	return inDegrees * Pi<T> / T(180);
}

template <typename T>
[[nodiscard]] constexpr T degrees(T inRadians) noexcept {
	return inRadians * T(180) / Pi<T>;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y + inA.z * inB.z;
}

template <typename T>
[[nodiscard]] constexpr T dot(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y + inA.z * inB.z + inA.w * inB.w;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> cross(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.y * inB.z - inA.z * inB.y,
		inA.z * inB.x - inA.x * inB.z,
		inA.x * inB.y - inA.y * inB.x,
	};
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec2<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec3<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] constexpr T lengthSquared(const detail::Vec4<T>& inValue) noexcept {
	return inValue.lengthSquared();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec2<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec3<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] T length(const detail::Vec4<T>& inValue) noexcept {
	return inValue.length();
}

template <typename T>
[[nodiscard]] detail::Vec2<T> normalize(const detail::Vec2<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] detail::Vec3<T> normalize(const detail::Vec3<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] detail::Vec4<T> normalize(const detail::Vec4<T>& inValue) noexcept {
	return inValue.normalized();
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> lerp(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB,
	T inT
) noexcept {
	return add(inA, scale(sub(inB, inA), inT));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> lerp(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	T inT
) noexcept {
	return add(inA, scale(sub(inB, inA), inT));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> lerp(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB,
	T inT
) noexcept {
	return add(inA, scale(sub(inB, inA), inT));
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] constexpr T distanceSquared(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return lengthSquared(sub(inB, inA));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] T distance(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB
) noexcept {
	return oa::sqrt(distanceSquared(inA, inB));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> reflect(
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inNormal
) noexcept {
	return sub(inIncident, scale(inNormal, T(2) * dot(inNormal, inIncident)));
}

template <typename T>
[[nodiscard]] detail::Vec3<T> refract(
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inNormal,
	T inEta
) noexcept {
	const T normalDotIncident = dot(inNormal, inIncident);
	const T discriminant = T(1) - inEta * inEta
		* (T(1) - normalDotIncident * normalDotIncident);
	if (discriminant < T(0)) return {};
	return sub(
		scale(inIncident, inEta),
		scale(inNormal, inEta * normalDotIncident + oa::sqrt(discriminant)));
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> faceForward(
	const detail::Vec3<T>& inNormal,
	const detail::Vec3<T>& inIncident,
	const detail::Vec3<T>& inReferenceNormal
) noexcept {
	return dot(inReferenceNormal, inIncident) < T(0) ? inNormal : -inNormal;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec2<T> abs(
	const detail::Vec2<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> abs(
	const detail::Vec3<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
		inValue.z < T(0) ? -inValue.z : inValue.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec4<T> abs(
	const detail::Vec4<T>& inValue
) noexcept {
	return {
		inValue.x < T(0) ? -inValue.x : inValue.x,
		inValue.y < T(0) ? -inValue.y : inValue.y,
		inValue.z < T(0) ? -inValue.z : inValue.z,
		inValue.w < T(0) ? -inValue.w : inValue.w,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> min(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.x < inB.x ? inA.x : inB.x,
		inA.y < inB.y ? inA.y : inB.y,
		inA.z < inB.z ? inA.z : inB.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> max(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB
) noexcept {
	return {
		inA.x > inB.x ? inA.x : inB.x,
		inA.y > inB.y ? inA.y : inB.y,
		inA.z > inB.z ? inA.z : inB.z,
	};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> clamp(
	const detail::Vec3<T>& inValue,
	const detail::Vec3<T>& inMinimum,
	const detail::Vec3<T>& inMaximum
) noexcept {
	return min(max(inValue, inMinimum), inMaximum);
}

template <typename T>
[[nodiscard]] constexpr T componentMin(const detail::Vec3<T>& inValue) noexcept {
	return oa::min(inValue.x, oa::min(inValue.y, inValue.z));
}

template <typename T>
[[nodiscard]] constexpr T componentMax(const detail::Vec3<T>& inValue) noexcept {
	return oa::max(inValue.x, oa::max(inValue.y, inValue.z));
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	T inA,
	T inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	const T difference = inA >= inB ? inA - inB : inB - inA;
	const T magnitudeA = inA >= T(0) ? inA : -inA;
	const T magnitudeB = inB >= T(0) ? inB : -inB;
	const T scaleValue = magnitudeA > magnitudeB ? magnitudeA : magnitudeB;
	return difference <= inAbsoluteTolerance
		or difference <= inRelativeTolerance * scaleValue;
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec2<T>& inA,
	const detail::Vec2<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec3<T>& inA,
	const detail::Vec3<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Vec4<T>& inA,
	const detail::Vec4<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.w, inB.w, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	return approximatelyEqual(inA.x, inB.x, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.y, inB.y, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.z, inB.z, inAbsoluteTolerance, inRelativeTolerance)
		and approximatelyEqual(inA.w, inB.w, inAbsoluteTolerance, inRelativeTolerance);
}

template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
	const detail::Mat4<T>& inA,
	const detail::Mat4<T>& inB,
	T inAbsoluteTolerance = Tolerance<T>,
	T inRelativeTolerance = Tolerance<T>
) noexcept {
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			if (not approximatelyEqual(
				inA.m[row][column], inB.m[row][column],
				inAbsoluteTolerance, inRelativeTolerance)) {
				return false;
			}
		}
	}
	return true;
}

[[nodiscard]] inline constexpr Quat quaternionIdentity() noexcept {
	return Quat::identity();
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromEuler(
	T inYawDeg,
	T inPitchDeg,
	T inRollDeg
) noexcept {
	const T yaw = radians(inYawDeg);
	const T pitch = radians(inPitchDeg);
	const T roll = radians(inRollDeg);
	const T cy = oa::cos(yaw * T(0.5));
	const T sy = oa::sin(yaw * T(0.5));
	const T cp = oa::cos(pitch * T(0.5));
	const T sp = oa::sin(pitch * T(0.5));
	const T cr = oa::cos(roll * T(0.5));
	const T sr = oa::sin(roll * T(0.5));
	return {
		(cy * cp) * sr - (sy * sp) * cr,
		(sy * cp) * sr + (cy * sp) * cr,
		(sy * cp) * cr - (cy * sp) * sr,
		(cy * cp) * cr + (sy * sp) * sr,
	};
}

template <typename T>
[[nodiscard]] detail::Vec3<T> quaternionToEuler(const detail::Quat<T>& inQuaternion) noexcept {
	const T sinRollCosPitch = T(2) * (inQuaternion.w * inQuaternion.x	+ inQuaternion.y * inQuaternion.z);
	const T cosRollCosPitch = T(1) - T(2) * (inQuaternion.x * inQuaternion.x + inQuaternion.y * inQuaternion.y);
	const T roll = oa::atan2(sinRollCosPitch, cosRollCosPitch);
	const T sinPitch = T(2) * (inQuaternion.w * inQuaternion.y - inQuaternion.z * inQuaternion.x);
	const T pitch = oa::abs(sinPitch) >= T(1)
		? oa::copySign(Pi<T> / T(2), sinPitch)
		: oa::asin(sinPitch);
	const T sinYawCosPitch = T(2) * (inQuaternion.w * inQuaternion.z + inQuaternion.x * inQuaternion.y);
	const T cosYawCosPitch = T(1) - T(2) * (inQuaternion.y * inQuaternion.y	+ inQuaternion.z * inQuaternion.z);
	const T yaw = oa::atan2(sinYawCosPitch, cosYawCosPitch);
	return {degrees(yaw), degrees(pitch), degrees(roll)};
}

template <typename T>
[[nodiscard]] constexpr T quaternionDot(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB
) noexcept {
	return inA.x * inB.x + inA.y * inB.y
		+ inA.z * inB.z + inA.w * inB.w;
}

template <typename T>
[[nodiscard]] bool tryInverse(
	const detail::Quat<T>& inQuaternion,
	detail::Quat<T>& outInverse,
	T inTolerance = InverseTolerance<T>
) noexcept {
	const T normSquared = inQuaternion.normSquared();
	if (not oa::isFinite(normSquared) or normSquared <= inTolerance) {
		return false;
	}
	outInverse = inQuaternion.conjugate() / normSquared;
	return outInverse.isFinite();
}

template <typename T>
[[nodiscard]] detail::Quat<T> nlerp(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inT
) noexcept {
	const detail::Quat<T> a = inA.normalized();
	detail::Quat<T> b = inB.normalized();
	if (quaternionDot(a, b) < T(0)) b = -b;
	return (a + (b - a) * inT).normalized();
}

template <typename T>
[[nodiscard]] detail::Quat<T> slerp(
	const detail::Quat<T>& inA,
	const detail::Quat<T>& inB,
	T inT
) noexcept {
	const detail::Quat<T> a = inA.normalized();
	detail::Quat<T> b = inB.normalized();
	T cosine = quaternionDot(a, b);
	if (cosine < T(0)) {
		b = -b;
		cosine = -cosine;
	}
	cosine = oa::clamp(cosine, T(-1), T(1));
	if (cosine > T(1) - Tolerance<T>) return nlerp(a, b, inT);
	const T angle = oa::acos(cosine);
	const T sine = oa::sin(angle);
	const T weightA = oa::sin((T(1) - inT) * angle) / sine;
	const T weightB = oa::sin(inT * angle) / sine;
	return (a * weightA + b * weightB).normalized();
}

template <typename T>
[[nodiscard]] detail::Mat4<T> quaternionToMatrix(const detail::Quat<T>& inQuaternion) noexcept {
	const detail::Quat<T> q = inQuaternion.normalized();
	const T xx = q.x * q.x;
	const T yy = q.y * q.y;
	const T zz = q.z * q.z;
	const T xy = q.x * q.y;
	const T xz = q.x * q.z;
	const T yz = q.y * q.z;
	const T xw = q.x * q.w;
	const T yw = q.y * q.w;
	const T zw = q.z * q.w;
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	// Transposed from the conventional column-vector form so v * M and
	// quaternion.rotate(v) produce the same direction.
	result.m[0][0] = T(1) - T(2) * (yy + zz);
	result.m[0][1] = T(2) * (xy + zw);
	result.m[0][2] = T(2) * (xz - yw);
	result.m[1][0] = T(2) * (xy - zw);
	result.m[1][1] = T(1) - T(2) * (xx + zz);
	result.m[1][2] = T(2) * (yz + xw);
	result.m[2][0] = T(2) * (xz + yw);
	result.m[2][1] = T(2) * (yz - xw);
	result.m[2][2] = T(1) - T(2) * (xx + yy);
	return result;
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromMatrix(const detail::Mat4<T>& inMatrix) noexcept {
	// The standard extraction is expressed for column-vector matrices. OA's
	// row-vector rotation is its transpose, so read through that transpose here.
	const auto at = [&inMatrix](I32 inRow, I32 inColumn) noexcept -> T {
		return inMatrix.m[inColumn][inRow];
	};
	const T m00 = at(0, 0);
	const T m11 = at(1, 1);
	const T m22 = at(2, 2);
	const T trace = m00 + m11 + m22;
	detail::Quat<T> result;
	if (trace > T(0)) {
		const T scale = oa::sqrt(trace + T(1)) * T(2);
		result.w = T(0.25) * scale;
		result.x = (at(2, 1) - at(1, 2)) / scale;
		result.y = (at(0, 2) - at(2, 0)) / scale;
		result.z = (at(1, 0) - at(0, 1)) / scale;
	} else if (m00 > m11 and m00 > m22) {
		const T scale = oa::sqrt(T(1) + m00 - m11 - m22) * T(2);
		result.w = (at(2, 1) - at(1, 2)) / scale;
		result.x = T(0.25) * scale;
		result.y = (at(0, 1) + at(1, 0)) / scale;
		result.z = (at(0, 2) + at(2, 0)) / scale;
	} else if (m11 > m22) {
		const T scale = oa::sqrt(T(1) + m11 - m00 - m22) * T(2);
		result.w = (at(0, 2) - at(2, 0)) / scale;
		result.x = (at(0, 1) + at(1, 0)) / scale;
		result.y = T(0.25) * scale;
		result.z = (at(1, 2) + at(2, 1)) / scale;
	} else {
		const T scale = oa::sqrt(T(1) + m22 - m00 - m11) * T(2);
		result.w = (at(1, 0) - at(0, 1)) / scale;
		result.x = (at(0, 2) + at(2, 0)) / scale;
		result.y = (at(1, 2) + at(2, 1)) / scale;
		result.z = T(0.25) * scale;
	}
	return result.normalized();
}

template <typename T>
[[nodiscard]] detail::Vec3<T> rotateVector(
	const detail::Quat<T>& inQuaternion,
	const detail::Vec3<T>& inVector
) noexcept {
	return inQuaternion.rotate(inVector);
}

[[nodiscard]] inline constexpr Mat4 matrixIdentity() noexcept {
	return Mat4::identity();
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> transpose(const detail::Mat4<T>& inMatrix) noexcept {
	detail::Mat4<T> result{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 col = 0; col < 4; ++col) {
			result.m[row][col] = inMatrix.m[col][row];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] T determinant(const detail::Mat4<T>& inMatrix) noexcept {
	detail::Mat4<T> reduced = inMatrix;
	T result = T(1);
	I32 sign = 1;
	for (I32 pivotColumn = 0; pivotColumn < 4; ++pivotColumn) {
		I32 pivotRow = pivotColumn;
		T pivotMagnitude = oa::abs(reduced.m[pivotRow][pivotColumn]);
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T magnitude = oa::abs(reduced.m[row][pivotColumn]);
			if (magnitude > pivotMagnitude) {
				pivotMagnitude = magnitude;
				pivotRow = row;
			}
		}
		if (not oa::isFinite(pivotMagnitude)) {
			return oa::Limits<T>::quietNaN();
		}
		if (pivotMagnitude == T(0)) return T(0);
		if (pivotRow != pivotColumn) {
			for (I32 column = 0; column < 4; ++column) {
				oa::swapValues(reduced.m[pivotRow][column], reduced.m[pivotColumn][column]);
			}
			sign = -sign;
		}
		const T pivot = reduced.m[pivotColumn][pivotColumn];
		result *= pivot;
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T factor = reduced.m[row][pivotColumn] / pivot;
			for (I32 column = pivotColumn + 1; column < 4; ++column) {
				reduced.m[row][column] -= factor * reduced.m[pivotColumn][column];
			}
		}
	}
	return sign < 0 ? -result : result;
}

template <typename T>
[[nodiscard]] bool tryInverse(
	const detail::Mat4<T>& inMatrix,
	detail::Mat4<T>& outInverse,
	T inTolerance = InverseTolerance<T>
) noexcept {
	T augmented[4][8] = {};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			const T value = inMatrix.m[row][column];
			if (not oa::isFinite(value)) return false;
			augmented[row][column] = value;
		}
		augmented[row][row + 4] = T(1);
	}
	for (I32 pivotColumn = 0; pivotColumn < 4; ++pivotColumn) {
		I32 pivotRow = pivotColumn;
		T pivotMagnitude = oa::abs(augmented[pivotRow][pivotColumn]);
		for (I32 row = pivotColumn + 1; row < 4; ++row) {
			const T magnitude = oa::abs(augmented[row][pivotColumn]);
			if (magnitude > pivotMagnitude) {
				pivotMagnitude = magnitude;
				pivotRow = row;
			}
		}
		if (not oa::isFinite(pivotMagnitude) or pivotMagnitude <= inTolerance) return false;
		if (pivotRow != pivotColumn) {
			for (I32 column = 0; column < 8; ++column) {
				oa::swapValues(augmented[pivotRow][column], augmented[pivotColumn][column]);
			}
		}
		const T inversePivot = T(1) / augmented[pivotColumn][pivotColumn];
		for (I32 column = 0; column < 8; ++column) {
			augmented[pivotColumn][column] *= inversePivot;
		}
		for (I32 row = 0; row < 4; ++row) {
			if (row == pivotColumn) continue;
			const T factor = augmented[row][pivotColumn];
			for (I32 column = 0; column < 8; ++column) {
				augmented[row][column] -= factor * augmented[pivotColumn][column];
			}
		}
	}
	detail::Mat4<T> inverse{};
	for (I32 row = 0; row < 4; ++row) {
		for (I32 column = 0; column < 4; ++column) {
			inverse.m[row][column] = augmented[row][column + 4];
			if (not oa::isFinite(inverse.m[row][column])) return false;
		}
	}
	outInverse = inverse;
	return true;
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transformPoint(
	const detail::Vec3<T>& inPoint,
	const detail::Mat4<T>& inMatrix
) noexcept {
	const detail::Vec4<T> value = transform(
		{inPoint.x, inPoint.y, inPoint.z, T(1)}, inMatrix);
	return {value.x, value.y, value.z};
}

template <typename T>
[[nodiscard]] constexpr detail::Vec3<T> transformDirection(
	const detail::Vec3<T>& inDirection,
	const detail::Mat4<T>& inMatrix
) noexcept {
	const detail::Vec4<T> value = transform(
		{inDirection.x, inDirection.y, inDirection.z, T(0)}, inMatrix);
	return {value.x, value.y, value.z};
}

template <typename T>
[[nodiscard]] bool tryProjectPoint(
	const detail::Vec3<T>& inPoint,
	const detail::Mat4<T>& inMatrix,
	detail::Vec3<T>& outPoint,
	T inTolerance = Tolerance<T>
) noexcept {
	const detail::Vec4<T> homogeneous = transform(
		{inPoint.x, inPoint.y, inPoint.z, T(1)}, inMatrix);
	if (not homogeneous.isFinite() or oa::abs(homogeneous.w) <= inTolerance) return false;
	const detail::Vec3<T> projected{
		homogeneous.x / homogeneous.w,
		homogeneous.y / homogeneous.w,
		homogeneous.z / homogeneous.w,
	};
	if (not projected.isFinite()) return false;
	outPoint = projected;
	return true;
}

template <typename T>
[[nodiscard]] bool tryTransformNormal(
	const detail::Vec3<T>& inNormal,
	const detail::Mat4<T>& inMatrix,
	detail::Vec3<T>& outNormal,
	T inTolerance = InverseTolerance<T>
) noexcept {
	detail::Mat4<T> inverse{};
	if (not tryInverse(inMatrix, inverse, inTolerance)) return false;
	const detail::Vec3<T> transformed = transformDirection(inNormal, transpose(inverse));
	const T transformedLength = length(transformed);
	if (not oa::isFinite(transformedLength) or transformedLength <= inTolerance) return false;
	outNormal = transformed / transformedLength;
	return true;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> perspective(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar
) noexcept {
	const T tangent = oa::tan(radians(inFovYDeg) / T(2));
	detail::Mat4<T> result{};
	result.m[0][0] = T(1) / (inAspect * tangent);
	result.m[1][1] = T(1) / tangent;
	result.m[2][2] = inFar / (inNear - inFar);
	result.m[2][3] = T(-1);
	result.m[3][2] = (inFar * inNear) / (inNear - inFar);
	return result;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> orthographic(
	T inWidth,
	T inHeight,
	T inNear,
	T inFar,
	T inZoom = T(1)
) noexcept {
	const T halfWidth = inWidth * T(0.5) / inZoom;
	const T halfHeight = inHeight * T(0.5) / inZoom;
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[0][0] = T(1) / halfWidth;
	result.m[1][1] = T(1) / halfHeight;
	result.m[2][2] = T(-1) / (inFar - inNear);
	result.m[3][2] = -inNear / (inFar - inNear);
	return result;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> lookAt(
	const detail::Vec3<T>& inEye,
	const detail::Vec3<T>& inTarget,
	const detail::Vec3<T>& inWorldUp
) noexcept {
	const detail::Vec3<T> forward = normalize(inTarget - inEye);
	const detail::Vec3<T> right = normalize(cross(forward, inWorldUp));
	const detail::Vec3<T> up = cross(right, forward);
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[0][0] = right.x;
	result.m[0][1] = up.x;
	result.m[0][2] = -forward.x;
	result.m[1][0] = right.y;
	result.m[1][1] = up.y;
	result.m[1][2] = -forward.y;
	result.m[2][0] = right.z;
	result.m[2][1] = up.z;
	result.m[2][2] = -forward.z;
	result.m[3][0] = -dot(right, inEye);
	result.m[3][1] = -dot(up, inEye);
	result.m[3][2] = dot(forward, inEye);
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> translation(const detail::Vec3<T>& inTranslation) noexcept {
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[3][0] = inTranslation.x;
	result.m[3][1] = inTranslation.y;
	result.m[3][2] = inTranslation.z;
	return result;
}

template <typename T>
[[nodiscard]] constexpr detail::Mat4<T> scaleMatrix(const detail::Vec3<T>& inScale) noexcept {
	detail::Mat4<T> result = detail::Mat4<T>::identity();
	result.m[0][0] = inScale.x;
	result.m[1][1] = inScale.y;
	result.m[2][2] = inScale.z;
	return result;
}

template <typename T>
[[nodiscard]] detail::Mat4<T> composeTrs(
	const detail::Vec3<T>& inTranslation,
	const detail::Quat<T>& inRotation,
	const detail::Vec3<T>& inScale
) noexcept {
	return matrixMul(
		matrixMul(scaleMatrix(inScale), quaternionToMatrix(inRotation)),
		translation(inTranslation));
}

template <typename T>
[[nodiscard]] detail::Vec3<T> sphericalToCartesian(
	T inYawRad,
	T inPitchRad,
	T inRadius
) noexcept {
	const T cosineYaw = oa::cos(inYawRad);
	const T sineYaw = oa::sin(inYawRad);
	const T cosinePitch = oa::cos(inPitchRad);
	const T sinePitch = oa::sin(inPitchRad);
	return {
		(inRadius * sineYaw) * cosinePitch,
		inRadius * sinePitch,
		(inRadius * cosineYaw) * cosinePitch,
	};
}

template <typename T>
[[nodiscard]] detail::Vec3<T> cartesianToSpherical(const detail::Vec3<T>& inValue) noexcept {
	const T radius = length(inValue);
	if (radius < Tolerance<T>) {
		return {};
	}
	return {
		oa::atan2(inValue.x, inValue.z),
		oa::asin(inValue.y / radius),
		radius,
	};
}

} // namespace vlm

} // namespace oa
