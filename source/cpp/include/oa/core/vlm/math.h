#pragma once

#include <oa/core/vlm/matrix.h>

#include <cmath>

namespace oa {

namespace vlm {

inline constexpr F32 kPi = 3.14159265358979323846F;
inline constexpr F32 kEpsilon = 0.0001F;

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

[[nodiscard]] inline constexpr Quat quaternionIdentity() noexcept {
	return Quat::identity();
}

template <typename T>
[[nodiscard]] detail::Quat<T> quaternionFromEuler(
	T inYawDeg,
	T inPitchDeg,
	T inRollDeg
) noexcept {
	const T pi = static_cast<T>(kPi);
	const T yaw = inYawDeg * pi / T(180);
	const T pitch = inPitchDeg * pi / T(180);
	const T roll = inRollDeg * pi / T(180);
	const T cy = std::cos(yaw * T(0.5));
	const T sy = std::sin(yaw * T(0.5));
	const T cp = std::cos(pitch * T(0.5));
	const T sp = std::sin(pitch * T(0.5));
	const T cr = std::cos(roll * T(0.5));
	const T sr = std::sin(roll * T(0.5));
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
	const T roll = std::atan2(sinRollCosPitch, cosRollCosPitch);
	const T sinPitch = T(2) * (inQuaternion.w * inQuaternion.y - inQuaternion.z * inQuaternion.x);
	const T pitch = std::abs(sinPitch) >= T(1)
		? std::copysign(static_cast<T>(kPi) / T(2), sinPitch)
		: std::asin(sinPitch);
	const T sinYawCosPitch = T(2) * (inQuaternion.w * inQuaternion.z + inQuaternion.x * inQuaternion.y);
	const T cosYawCosPitch = T(1) - T(2) * (inQuaternion.y * inQuaternion.y	+ inQuaternion.z * inQuaternion.z);
	const T yaw = std::atan2(sinYawCosPitch, cosYawCosPitch);
	const T degrees = T(180) / static_cast<T>(kPi);
	return {yaw * degrees, pitch * degrees, roll * degrees};
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
		const T scale = std::sqrt(trace + T(1)) * T(2);
		result.w = T(0.25) * scale;
		result.x = (at(2, 1) - at(1, 2)) / scale;
		result.y = (at(0, 2) - at(2, 0)) / scale;
		result.z = (at(1, 0) - at(0, 1)) / scale;
	} else if (m00 > m11 and m00 > m22) {
		const T scale = std::sqrt(T(1) + m00 - m11 - m22) * T(2);
		result.w = (at(2, 1) - at(1, 2)) / scale;
		result.x = T(0.25) * scale;
		result.y = (at(0, 1) + at(1, 0)) / scale;
		result.z = (at(0, 2) + at(2, 0)) / scale;
	} else if (m11 > m22) {
		const T scale = std::sqrt(T(1) + m11 - m00 - m22) * T(2);
		result.w = (at(0, 2) - at(2, 0)) / scale;
		result.x = (at(0, 1) + at(1, 0)) / scale;
		result.y = T(0.25) * scale;
		result.z = (at(1, 2) + at(2, 1)) / scale;
	} else {
		const T scale = std::sqrt(T(1) + m22 - m00 - m11) * T(2);
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
[[nodiscard]] detail::Mat4<T> perspective(
	T inFovYDeg,
	T inAspect,
	T inNear,
	T inFar
) noexcept {
	const T tangent = std::tan(inFovYDeg * static_cast<T>(kPi) / T(360));
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
[[nodiscard]] detail::Vec3<T> sphericalToCartesian(
	T inYawRad,
	T inPitchRad,
	T inRadius
) noexcept {
	const T cosineYaw = std::cos(inYawRad);
	const T sineYaw = std::sin(inYawRad);
	const T cosinePitch = std::cos(inPitchRad);
	const T sinePitch = std::sin(inPitchRad);
	return {
		(inRadius * sineYaw) * cosinePitch,
		inRadius * sinePitch,
		(inRadius * cosineYaw) * cosinePitch,
	};
}

template <typename T>
[[nodiscard]] detail::Vec3<T> cartesianToSpherical(const detail::Vec3<T>& inValue) noexcept {
	const T radius = length(inValue);
	if (radius < static_cast<T>(kEpsilon)) {
		return {};
	}
	return {
		std::atan2(inValue.x, inValue.z),
		std::asin(inValue.y / radius),
		radius,
	};
}

} // namespace vlm

} // namespace oa
